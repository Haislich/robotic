"""The typed boundary around sympy: constants, expressions and their compilation.

This is the only module that touches sympy. Every other module builds on the public
`Scalar` and `Expression` API, so formulas stay typed from the first symbol to the last
simplification, and the type says whether a value is fixed.
"""

import dataclasses
import operator
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    NoReturn,
    Self,
    final,
    overload,
    override,
)

import jax
import jax.numpy as jnp
import sympy
from jax.core import Tracer
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from _typeshed import DataclassInstance

jax.config.update(name="jax_enable_x64", val=True)

_INFINITIES = frozenset({sympy.zoo, sympy.oo, sympy.S.NegativeInfinity})
type Undefined = Literal["undefined", "infinite"]
type ScalarLike = Scalar | int

_DIGITS = 30


class _Incomparable:
    """No equality, hash or truth value: Python's defaults would answer silently."""

    __slots__ = ()

    _ADVICE: ClassVar[str] = (
        "symbolic values have no equality: compare constants with is_close; "
        "evaluate expressions, or compile them and compare the numbers at a "
        "configuration"
    )

    @override
    def __eq__(self, other: object) -> NoReturn:
        raise TypeError(self._ADVICE)

    @override
    def __ne__(self, other: object) -> NoReturn:
        raise TypeError(self._ADVICE)

    @override
    def __hash__(self) -> NoReturn:
        raise TypeError(self._ADVICE)

    def __bool__(self) -> NoReturn:
        raise TypeError(self._ADVICE)


@dataclass(frozen=True)
class _NoParameters:
    """The parameters of a constant quantity: nothing varies."""


class Symbolic[K: (Scalar, Expression)](_Incomparable, ABC):
    """A symbolic quantity: simplifiable, printable as LaTeX and compilable to JAX.

    `K` says whether it is a constant (`Scalar`) or built from a symbol
    (`Expression`). It has no equality, hash or truth value: `==`, `!=`, `hash` and
    `bool` raise `TypeError`.
    """

    __slots__ = ()

    @override
    def __reduce__(self) -> tuple[object, tuple[type[Self]], object]:
        """Rebuild for `copy` and `pickle` from the stored fields, not the constructor.

        Returns:
            The instance factory, its class and the stored fields.
        """
        return object.__new__, (type(self),), self.__getstate__()

    @abstractmethod
    def simplify(self) -> Self:
        """Simplify every scalar of the quantity.

        Returns:
            The same quantity, in the same frames, with simplified scalars.
        """

    @abstractmethod
    def latex(self) -> str:
        """Render the quantity as LaTeX.

        Returns:
            The LaTeX source, frames written as the course writes them.
        """

    @abstractmethod
    def _layout(self) -> Layout: ...

    @staticmethod
    def _bmatrix(rows: Iterable[Iterable[Scalar | Expression]]) -> str:
        lines = (" & ".join(entry.latex() for entry in row) for row in rows)
        return rf"\begin{{bmatrix}} {r' \\ '.join(lines)} \end{{bmatrix}}"

    @overload
    def compile[T: Symbolic[Scalar]](self: T) -> Numeric[T]: ...

    @overload
    def compile[T: Symbolic[Scalar] | Symbolic[Expression], P: DataclassInstance](
        self: T, parameters: type[P]
    ) -> Compiled[P, T]: ...

    def compile[P: DataclassInstance](
        self: Symbolic[Scalar] | Symbolic[Expression],
        parameters: type[P] | None = None,
    ) -> (
        Numeric[Symbolic[Scalar] | Symbolic[Expression]]
        | Compiled[P, Symbolic[Scalar] | Symbolic[Expression]]
    ):
        """Compile the quantity to JAX.

        Args:
            parameters: A dataclass whose fields, each annotated `float` or
                `jax.typing.ArrayLike`, name the symbols of the quantity; extra fields
                are allowed. Omitted for a constant, which has nothing to vary.

        Returns:
            The compiled function of the parameters; for a constant compiled without
            parameters, its compiled value.

        Raises:
            TypeError: A symbolic quantity compiled without parameters.
        """
        if parameters is not None:
            return Compiled(self, parameters)
        if not self._layout().is_constant:
            message = (
                f"{self._layout().label} is symbolic: compile it against a "
                "parameters dataclass naming its symbols"
            )
            raise TypeError(message)
        return Compiled(self, _NoParameters)(_NoParameters())


@dataclass(frozen=True, eq=False)
class Layout:
    """How a quantity flattens to scalars for compilation.

    Attributes:
        label: The quantity's type as printed in a compiled result, e.g.
            `Position[Base]`.
        entries: The scalars, row-major.
        shape: The natural shape of the compiled components.
    """

    label: str
    entries: tuple[Scalar | Expression, ...]
    shape: tuple[int, ...]

    @property
    def is_constant(self) -> bool:
        """Whether every entry is a constant."""
        return all(isinstance(entry, Scalar) for entry in self.entries)


class _Value[K: (Scalar, Expression)](Symbolic[K]):
    """Exact arithmetic and functions shared by constants and expressions.

    An operation returns a constant only when every operand is a constant or an `int`:
    the kind of the result follows the kinds of the operands, not its value.
    """

    __slots__ = ("_expression",)

    _expression: sympy.Expr

    @abstractmethod
    def _of(self, expression: sympy.Expr) -> K: ...

    @abstractmethod
    def _accepts(self, other: object) -> bool: ...

    @staticmethod
    def _is_exact(value: object) -> bool:
        return isinstance(value, Scalar | int) and not isinstance(value, bool)

    @staticmethod
    def _not_exact(value: object) -> str:
        return (
            f"a symbolic scalar takes an exact number, an int, a Scalar or an "
            f"Expression, not {type(value).__name__} {value!r}; write Scalar(3) / 10 "
            "for 0.3 and Scalar.pi() for pi"
        )

    @staticmethod
    def _expression_of(value: ScalarLike | Expression) -> sympy.Expr:
        match value:
            case Scalar() | Expression():
                return value._expression
            case int():
                return sympy.Integer(value)

    def _combined(
        self,
        other: K | ScalarLike,
        operation: Callable[[sympy.Expr, sympy.Expr], sympy.Expr],
    ) -> K:
        if not self._accepts(other):
            return NotImplemented
        return self._of(operation(self._expression, _Value._expression_of(other)))

    def __add__(self, other: K | ScalarLike) -> K:
        """Add.

        Returns:
            `self + other`.
        """
        return self._combined(other, operator.add)

    def __radd__(self, other: ScalarLike) -> K:
        """Add from the left.

        Returns:
            `other + self`.
        """
        return self._combined(other, lambda mine, theirs: theirs + mine)

    def __sub__(self, other: K | ScalarLike) -> K:
        """Subtract.

        Returns:
            `self - other`.
        """
        return self._combined(other, operator.sub)

    def __rsub__(self, other: ScalarLike) -> K:
        """Subtract from a number.

        Returns:
            `other - self`.
        """
        return self._combined(other, lambda mine, theirs: theirs - mine)

    def __mul__(self, other: K | ScalarLike) -> K:
        """Multiply.

        Returns:
            `self * other`.
        """
        return self._combined(other, operator.mul)

    def __rmul__(self, other: ScalarLike) -> K:
        """Multiply from the left.

        Returns:
            `other * self`.
        """
        return self._combined(other, lambda mine, theirs: theirs * mine)

    def __truediv__(self, other: K | ScalarLike) -> K:
        """Divide; integer quotients stay exact fractions.

        Returns:
            `self / other`.
        """
        return self._combined(other, operator.truediv)

    def __rtruediv__(self, other: ScalarLike) -> K:
        """Divide a number.

        Returns:
            `other / self`.
        """
        return self._combined(other, lambda mine, theirs: theirs / mine)

    def __pow__(self, exponent: K | ScalarLike) -> K:
        """Raise to a power.

        Returns:
            `self ** exponent`.
        """
        return self._combined(exponent, operator.pow)

    def __rpow__(self, base: ScalarLike) -> K:
        """Raise a number to this power.

        Returns:
            `base ** self`.
        """
        return self._combined(base, lambda mine, theirs: theirs**mine)

    def __neg__(self) -> K:
        """Negate.

        Returns:
            `-self`.
        """
        return self._of(-self._expression)

    def cos(self) -> K:
        """Cosine, argument in radians.

        Returns:
            `cos(self)`.
        """
        return self._of(sympy.cos(self._expression))

    def sin(self) -> K:
        """Sine, argument in radians.

        Returns:
            `sin(self)`.
        """
        return self._of(sympy.sin(self._expression))

    def tan(self) -> K:
        """Tangent, argument in radians.

        Returns:
            `tan(self)`.
        """
        return self._of(sympy.tan(self._expression))

    @overload
    def atan2(self, x: K | ScalarLike) -> K: ...

    @overload
    def atan2(self, x: Expression) -> Expression: ...

    def atan2(self, x: K | ScalarLike | Expression) -> K | Expression:
        """Two-argument arctangent with `self` as the y coordinate.

        Args:
            x: The x coordinate.

        Returns:
            The angle of (x, self) in (-pi, pi]; an expression if either is one.

        Raises:
            TypeError: `x` is not exact.
        """
        if not (_Value._is_exact(x) or isinstance(x, Expression)):
            raise TypeError(_Value._not_exact(x))
        angle = sympy.atan2(self._expression, _Value._expression_of(x))
        if isinstance(x, Expression):
            return x._of(angle)
        return self._of(angle)

    def sqrt(self) -> K:
        """Exact principal square root.

        Returns:
            `sqrt(self)`.
        """
        return self._of(sympy.sqrt(self._expression))

    def exp(self) -> K:
        """Exact exponential.

        Returns:
            `exp(self)`.
        """
        return self._of(sympy.exp(self._expression))

    def _simplified(self) -> sympy.Expr:
        match sympy.factor_terms(sympy.trigsimp(self._expression, method="fu")):
            case sympy.Expr() as simplified:
                return simplified
            case other:
                message = f"simplification returned a non-expression {other!r}"
                raise TypeError(message)

    @override
    def latex(self) -> str:
        """Render as LaTeX.

        Returns:
            The LaTeX source.
        """
        return str(sympy.latex(self._expression))

    @override
    def _layout(self) -> Layout:
        label = "Scalar" if isinstance(self, Scalar) else "Expression"
        return Layout(label=label, entries=(self._of(self._expression),), shape=())

    @override
    def __repr__(self) -> str:
        """Print as sympy prints the expression.

        Returns:
            The expression, e.g. `pi/2`.
        """
        return str(self._expression)


@final
class Scalar(_Value["Scalar"]):
    """A constant: built from integers and exact constants only, so its value is fixed.

    Arithmetic accepts a constant or an `int` on either side and gives a constant;
    with an expression it gives an expression. A `float` is rejected, so every value
    stays exact (`Scalar(3) / 10`, `Scalar.pi() / 2`). A constant is never undefined
    or infinite: an operation whose result would be (0/0, 1/0) raises
    `UndefinedValueError`. Complex values are not supported. Constants compare with a
    tolerance through `is_close`.
    """

    __slots__ = ()

    def __init__(self, value: ScalarLike) -> None:
        """Build a constant.

        Args:
            value: An exact integer, or a constant to copy.

        Raises:
            TypeError: The value is not an `int` or a `Scalar` (a `float`, a `bool`,
                an expression).
        """
        if not Scalar._is_exact(value):
            raise TypeError(Scalar._not_exact(value))
        self._expression = Scalar._expression_of(value)

    @classmethod
    def _wrap(cls, expression: sympy.Expr) -> Scalar:
        scalar = object.__new__(Scalar)
        scalar._expression = expression
        return scalar

    @override
    def _of(self, expression: sympy.Expr) -> Scalar:
        reason = Scalar._undefined(expression)
        if reason is not None:
            raise UndefinedValueError(str(expression), reason)
        return Scalar._wrap(expression)

    @staticmethod
    def _undefined(expression: sympy.Expr) -> Undefined | None:
        for node in sympy.preorder_traversal(expression):
            if node is sympy.nan:
                return "undefined"
            if node in _INFINITIES:
                return "infinite"
        return None

    @override
    def _accepts(self, other: object) -> bool:
        return Scalar._is_exact(other)

    @classmethod
    def pi(cls) -> Scalar:
        """Exact π.

        Returns:
            The exact constant π.
        """
        return Scalar._wrap(sympy.pi)

    def is_close(self, other: ScalarLike, tolerance: float = 1e-9) -> bool:
        """Whether two constants differ by at most `tolerance`.

        The difference is evaluated to 30 significant digits, so the tolerance is the
        only margin; no simplification is needed to recognize a zero.

        Args:
            other: The constant or `int` to compare with.
            tolerance: The largest absolute difference accepted, not negative.

        Returns:
            True when |self - other| <= tolerance.

        Raises:
            TypeError: `other` is not a constant or an `int` (an expression: evaluate
                it first), or `tolerance` is not a number.
            ValueError: `tolerance` is negative.
        """
        if not Scalar._is_exact(other):
            message = (
                f"is_close compares a constant with a constant or an int, not "
                f"{type(other).__name__} {other!r}; evaluate an expression first"
            )
            raise TypeError(message)
        if isinstance(tolerance, bool) or not isinstance(tolerance, int | float):
            message = f"a tolerance is a number, not {type(tolerance).__name__}"
            raise TypeError(message)
        if tolerance < 0:
            message = f"a tolerance is not negative, not {tolerance}"
            raise ValueError(message)
        difference = sympy.N(self._expression - Scalar._expression_of(other), _DIGITS)
        return bool(abs(difference) <= tolerance)

    @override
    def simplify(self) -> Self:
        """An equal constant, in a simpler form when one is found.

        Returns:
            The simplified constant; its form is not promised.
        """
        made = object.__new__(type(self))
        made._expression = self._simplified()
        return made


class Expression(_Value["Expression"]):
    """A value built from at least one symbol, whatever it simplifies to.

    Its value is not fixed until its symbols are, so it has no tolerance comparison:
    `evaluate` substitutes exact values and gives the constant.
    """

    __slots__ = ()

    def __init__(self, value: ScalarLike | Expression) -> None:
        """Take a constant or an expression as an expression.

        Args:
            value: An exact integer, a constant or an expression to copy.

        Raises:
            TypeError: The value is not exact (a `float`, a `bool`).
        """
        if not (Expression._is_exact(value) or isinstance(value, Expression)):
            raise TypeError(Expression._not_exact(value))
        self._expression = Expression._expression_of(value)

    @override
    def _of(self, expression: sympy.Expr) -> Expression:
        made = object.__new__(Expression)
        made._expression = expression
        return made

    @override
    def _accepts(self, other: object) -> bool:
        return Expression._is_exact(other) or isinstance(other, Expression)

    def evaluate(self, values: Mapping[str, ScalarLike] | None = None) -> Scalar:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed. Omitted when no symbol is left.

        Returns:
            The constant.

        Raises:
            TypeError: A name is not a string or a value is not exact.
            MissingParameterError: A symbol has no value.
            UndefinedValueError: The values make the expression undefined or
                infinite.
        """
        given = {} if values is None else values
        for name, value in given.items():
            if not isinstance(name, str):
                message = f"values are given by symbol name, not {name!r}"
                raise TypeError(message)
            if not Expression._is_exact(value):
                raise TypeError(Expression._not_exact(value))
        names = frozenset(str(symbol) for symbol in self._expression.free_symbols)
        missing = names - frozenset(given)
        if missing:
            raise MissingParameterError(missing)
        substitution = {
            sympy.Symbol(name, real=True): Expression._expression_of(value)
            for name, value in given.items()
        }
        value = self._expression.xreplace(substitution)
        reason = Scalar._undefined(value)
        if reason is not None:
            raise UndefinedValueError(str(value), reason, given)
        return Scalar._wrap(value)

    @override
    def simplify(self) -> Self:
        """An equal expression, in a simpler form when one is found.

        Returns:
            The simplified expression; its form is not promised.
        """
        made = object.__new__(type(self))
        made._expression = self._simplified()
        return made


class Symbol(Expression):
    """A real variable of the formulas, matched by name when evaluated or compiled."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        """Create the real variable with this name.

        Args:
            name: The variable's name; equal names give the same variable.

        Raises:
            TypeError: The name is not a string.
        """
        if not isinstance(name, str):
            message = f"a symbol's name is a string, not {type(name).__name__}"
            raise TypeError(message)
        self._expression = sympy.Symbol(name, real=True)

    @override
    def simplify(self) -> Self:
        """A symbol is already simplest.

        Returns:
            The symbol itself.
        """
        return self


@dataclass(frozen=True, eq=False)
class _Constants:
    """Values that are all constants: a composite built from them is constant."""

    values: tuple[Scalar, ...]

    @classmethod
    def or_expressions(
        cls, values: Iterable[ScalarLike | Expression]
    ) -> _Constants | _Expressions:
        """Take the values as constants, or all as expressions if one is.

        Returns:
            The constants, or the expressions.
        """
        given = tuple(values)
        constants = tuple(value for value in given if not isinstance(value, Expression))
        if len(constants) == len(given):
            return _Constants(
                tuple(
                    value if isinstance(value, Scalar) else Scalar(value)
                    for value in constants
                )
            )
        return _Expressions.of(given)


@dataclass(frozen=True, eq=False)
class _Expressions:
    """Values that are all expressions: a composite built from them is symbolic."""

    values: tuple[Expression, ...]

    @classmethod
    def of(cls, values: Iterable[ScalarLike | Expression]) -> _Expressions:
        """Take every value as an expression, constants included.

        Returns:
            The expressions.
        """
        return _Expressions(
            tuple(
                value if isinstance(value, Expression) else Expression(value)
                for value in values
            )
        )


class MissingParameterError(ValueError):
    """A symbol has no value: no field when compiling, no entry when evaluating."""

    def __init__(
        self,
        names: frozenset[str],
        parameters: type[DataclassInstance] | None = None,
    ) -> None:
        """Name the symbols without a value, and the parameters class if any.

        Args:
            names: The symbols without a value.
            parameters: The parameters dataclass, when compiling.
        """
        listed = ", ".join(sorted(names))
        if parameters is None:
            super().__init__(f"no value given for {listed}")
        else:
            super().__init__(f"{parameters.__name__} has no field for {listed}")


class UndefinedValueError(ArithmeticError):
    """A constant would be undefined or infinite."""

    def __init__(
        self,
        value: str,
        reason: Undefined,
        values: Mapping[str, ScalarLike] | None = None,
    ) -> None:
        """Name the value obtained, whether undefined or infinite, and the values given.

        Args:
            value: The value obtained, as printed.
            reason: Whether the value is undefined or infinite.
            values: The values substituted, when evaluating.
        """
        message = f"the value is {reason} ({value})"
        if values is not None:
            listed = ", ".join(f"{name} = {values[name]}" for name in sorted(values))
            message = f"{message} at {listed}"
        super().__init__(message)


class ParameterTypeError(TypeError):
    """A field of the parameters dataclass is neither `float` nor `ArrayLike`."""

    def __init__(self, parameters: type[DataclassInstance], name: str) -> None:
        """Name the parameters class and the field.

        Args:
            parameters: The parameters dataclass.
            name: The field with another annotation.
        """
        message = "must be annotated float or ArrayLike"
        super().__init__(f"field {name!r} of {parameters.__name__} {message}")


class ParameterShapeError(ValueError):
    """A field of the parameters holds an array: batching goes through `jax.vmap`."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        """Name the field and its shape.

        Args:
            name: The field holding an array.
            shape: The shape of the array it holds.
        """
        super().__init__(
            f"field {name!r} holds an array of shape {shape}; every field must be a "
            "scalar of shape (), batch configurations with jax.vmap"
        )


class Compiled[P: DataclassInstance, T: Symbolic[Scalar] | Symbolic[Expression]]:
    """A symbolic quantity compiled to a JIT-compiled JAX function of `P`."""

    _evaluate: Callable[..., Numeric[T]]

    def __init__(self, quantity: T, parameters: type[P]) -> None:
        """Compile the quantity; validation happens once, here.

        The expressions are compiled as written, without simplification; repeated
        subexpressions are computed once.

        Args:
            quantity: The symbolic quantity.
            parameters: A dataclass class whose fields name the symbols of the
                quantity, each annotated `float` (one configuration) or
                `jax.typing.ArrayLike` (batched code). Extra fields are allowed.

        Raises:
            TypeError: `parameters` is not a dataclass class (an instance included).
            ParameterTypeError: A field is annotated neither `float` nor `ArrayLike`.
            MissingParameterError: A free symbol of the quantity has no field.
        """
        if not (isinstance(parameters, type) and dataclasses.is_dataclass(parameters)):
            message = f"parameters must be a dataclass class, not {parameters!r}"
            raise TypeError(message)
        layout = quantity._layout()
        hints = typing.get_type_hints(parameters)
        names = tuple(field.name for field in dataclasses.fields(parameters))
        for name in names:
            if hints[name] is not float and hints[name] != ArrayLike:
                raise ParameterTypeError(parameters, name)
        expressions = [entry._expression for entry in layout.entries]
        free = frozenset(
            str(symbol)
            for expression in expressions
            for symbol in expression.free_symbols
        )
        missing = free - frozenset(names)
        if missing:
            raise MissingParameterError(missing, parameters)
        symbols = [sympy.Symbol(name, real=True) for name in names]
        self._parameters = parameters
        self._names = names
        self._layout = layout
        self._function = sympy.lambdify(symbols, expressions, modules="jax", cse=True)
        self._evaluate = jax.jit(self._kernel)

    def _kernel(self, *values: jax.Array) -> Numeric[T]:
        for name, value in zip(self._names, values, strict=True):
            if jnp.shape(value):
                raise ParameterShapeError(name, jnp.shape(value))
        results = [
            jnp.asarray(entry, dtype=jnp.float64) for entry in self._function(*values)
        ]
        components = jnp.stack(results).reshape(self._layout.shape)
        return Numeric(self._layout.label, components)

    def __call__(self, values: P) -> Numeric[T]:
        """Evaluate at one configuration.

        Args:
            values: The parameter values, each of shape `()`; JAX tracers are
                accepted, so the call works under `jax.jit`, `jax.vmap` and
                `jax.grad`. A field holding an array raises `ParameterShapeError`
                when JAX traces; batch with `jax.vmap`.

        Returns:
            The compiled result.

        Raises:
            TypeError: `values` is not an instance of the parameters dataclass.
        """
        if not isinstance(values, self._parameters):
            message = (
                f"values must be a {self._parameters.__name__}, "
                f"not {type(values).__name__}"
            )
            raise TypeError(message)
        return self._evaluate(*(getattr(values, name) for name in self._names))


class Numeric[T: Symbolic[Scalar] | Symbolic[Expression]](_Incomparable):
    """The compiled value of a `T`: bare components, with the meaning kept in the type.

    A registered JAX pytree: the components are its only child, the printed type its
    static data, so it passes through `jax.jit`, `jax.vmap` and `jax.grad`. It has no
    equality, hash or truth value: compare `.components` with a tolerance.
    """

    __slots__ = ("_components", "_label")

    _ADVICE: ClassVar[str] = (
        "compiled values have no equality: compare their .components with a "
        "tolerance, e.g. jnp.allclose(a.components, b.components)"
    )

    def __init__(self, label: str, components: jax.Array) -> None:
        """Pair the components with the printed type.

        Args:
            label: The quantity's type as printed, e.g. `Position[Base]`.
            components: Float64 components in the natural shape; leading axes under
                `vmap` are configurations.
        """
        self._label = label
        self._components = components

    @property
    def components(self) -> jax.Array:
        """The components in the natural shape of `T`, per configuration."""
        return self._components

    @override
    def __repr__(self) -> str:
        """Print the type and the components.

        Returns:
            E.g. `Numeric[Position[Base]]([1.0, 0.5, 0.0])`; under tracing the
            components print as JAX prints a tracer.
        """
        components = self._components
        shown = components if isinstance(components, Tracer) else components.tolist()
        return f"Numeric[{self._label}]({shown})"

    @staticmethod
    def _flatten(
        numeric: Numeric[Symbolic[Expression]],
    ) -> tuple[tuple[jax.Array], str]:
        return (numeric._components,), numeric._label

    @staticmethod
    def _unflatten(
        label: str, children: tuple[jax.Array]
    ) -> Numeric[Symbolic[Expression]]:
        (components,) = children
        return Numeric[Symbolic[Expression]](label, components)


jax.tree_util.register_pytree_node(Numeric, Numeric._flatten, Numeric._unflatten)
