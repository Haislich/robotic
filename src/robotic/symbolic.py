"""The typed boundary around sympy: symbolic scalars, compilation and compiled results.

This is the only module that touches sympy. Every other module builds on the public
`Scalar` API, so formulas stay typed from the first symbol to the last simplification.
"""

import dataclasses
import operator
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NoReturn, Self, override

import jax
import jax.numpy as jnp
import sympy
from jax.core import Tracer
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import DataclassInstance

jax.config.update(name="jax_enable_x64", val=True)

type ScalarLike = Scalar | int


class _Incomparable:
    """No equality, hash or truth value: Python's defaults would answer silently."""

    __slots__ = ()

    _ADVICE: ClassVar[str] = (
        "symbolic values have no equality: whether two of them are equal on the "
        "whole domain cannot be known; compile them and compare the numbers at a "
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


class Symbolic(_Incomparable, ABC):
    """A symbolic quantity: simplifiable, printable as LaTeX and compilable to JAX.

    It has no equality, hash or truth value: `==`, `!=`, `hash` and `bool` raise
    `TypeError`.
    """

    __slots__ = ()

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
    def _bmatrix(rows: Iterable[Iterable[Scalar]]) -> str:
        lines = (" & ".join(entry.latex() for entry in row) for row in rows)
        return rf"\begin{{bmatrix}} {r' \\ '.join(lines)} \end{{bmatrix}}"

    def compile[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]:
        """Compile the quantity to a JAX function of a parameters dataclass.

        Args:
            parameters: A dataclass whose fields, each annotated `float` or
                `jax.typing.ArrayLike`, name the symbols of the quantity. Extra fields
                are allowed.

        Returns:
            The compiled function.
        """
        return Compiled(self, parameters)


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
    entries: tuple[Scalar, ...]
    shape: tuple[int, ...]


class Scalar(Symbolic):
    """A symbolic scalar expression, exact: built from integers, constants and symbols.

    Arithmetic accepts a scalar or an `int` on either side; a `float` is rejected, so
    every value stays exact (`Scalar(3) / 10`, `Scalar.pi() / 2`).
    """

    __slots__ = ("_expression",)

    _expression: sympy.Expr

    def __init__(self, value: ScalarLike) -> None:
        """Build a constant scalar.

        Args:
            value: An exact integer, or a scalar to copy.

        Raises:
            TypeError: The value is not an `int` or a `Scalar` (a `float`, a `bool`).
        """
        if not Scalar._is_exact(value):
            raise TypeError(Scalar._not_exact(value))
        self._expression = Scalar._expression_of(value)

    @classmethod
    def _wrap(cls, expression: sympy.Expr) -> Scalar:
        scalar = object.__new__(Scalar)
        scalar._expression = expression
        return scalar

    @staticmethod
    def _is_exact(value: object) -> bool:
        return isinstance(value, Scalar | int) and not isinstance(value, bool)

    @staticmethod
    def _not_exact(value: object) -> str:
        return (
            f"a symbolic scalar takes an exact number, an int or a Scalar, not "
            f"{type(value).__name__} {value!r}; write Scalar(3) / 10 for 0.3 and "
            "Scalar.pi() for pi"
        )

    @staticmethod
    def _expression_of(value: ScalarLike) -> sympy.Expr:
        if not Scalar._is_exact(value):
            raise TypeError(Scalar._not_exact(value))
        match value:
            case Scalar():
                return value._expression
            case int():
                return sympy.Integer(value)

    def _combined(
        self,
        other: ScalarLike,
        operation: Callable[[sympy.Expr, sympy.Expr], sympy.Expr],
    ) -> Scalar:
        if not Scalar._is_exact(other):
            return NotImplemented
        return Scalar._wrap(operation(self._expression, Scalar._expression_of(other)))

    @classmethod
    def symbol(cls, name: str) -> Symbol:
        """Create the real symbol with this name.

        Args:
            name: The symbol's name; equal names give the same symbol.

        Returns:
            The symbol.
        """
        return Symbol(name)

    @classmethod
    def pi(cls) -> Scalar:
        """Exact π.

        Returns:
            The exact constant π.
        """
        return Scalar._wrap(sympy.pi)

    def __add__(self, other: ScalarLike) -> Scalar:
        """Add.

        Returns:
            `self + other`.
        """
        return self._combined(other, operator.add)

    def __radd__(self, other: int) -> Scalar:
        """Add from the left.

        Returns:
            `other + self`.
        """
        return self._combined(other, lambda mine, theirs: theirs + mine)

    def __sub__(self, other: ScalarLike) -> Scalar:
        """Subtract.

        Returns:
            `self - other`.
        """
        return self._combined(other, operator.sub)

    def __rsub__(self, other: int) -> Scalar:
        """Subtract from a number.

        Returns:
            `other - self`.
        """
        return self._combined(other, lambda mine, theirs: theirs - mine)

    def __mul__(self, other: ScalarLike) -> Scalar:
        """Multiply.

        Returns:
            `self * other`.
        """
        return self._combined(other, operator.mul)

    def __rmul__(self, other: int) -> Scalar:
        """Multiply from the left.

        Returns:
            `other * self`.
        """
        return self._combined(other, lambda mine, theirs: theirs * mine)

    def __truediv__(self, other: ScalarLike) -> Scalar:
        """Divide; integer quotients stay exact fractions.

        Returns:
            `self / other`.
        """
        return self._combined(other, operator.truediv)

    def __rtruediv__(self, other: int) -> Scalar:
        """Divide a number.

        Returns:
            `other / self`.
        """
        return self._combined(other, lambda mine, theirs: theirs / mine)

    def __pow__(self, exponent: ScalarLike) -> Scalar:
        """Raise to a power.

        Returns:
            `self ** exponent`.
        """
        return self._combined(exponent, operator.pow)

    def __rpow__(self, base: int) -> Scalar:
        """Raise a number to this power.

        Returns:
            `base ** self`.
        """
        return self._combined(base, lambda mine, theirs: theirs**mine)

    def __neg__(self) -> Scalar:
        """Negate.

        Returns:
            `-self`.
        """
        return Scalar._wrap(-self._expression)

    def cos(self) -> Scalar:
        """Cosine, argument in radians.

        Returns:
            `cos(self)`.
        """
        return Scalar._wrap(sympy.cos(self._expression))

    def sin(self) -> Scalar:
        """Sine, argument in radians.

        Returns:
            `sin(self)`.
        """
        return Scalar._wrap(sympy.sin(self._expression))

    def tan(self) -> Scalar:
        """Tangent, argument in radians.

        Returns:
            `tan(self)`.
        """
        return Scalar._wrap(sympy.tan(self._expression))

    def atan2(self, x: ScalarLike) -> Scalar:
        """Two-argument arctangent with `self` as the y coordinate.

        Args:
            x: The x coordinate.

        Returns:
            The angle of (x, self) in (-pi, pi].
        """
        return Scalar._wrap(sympy.atan2(self._expression, Scalar._expression_of(x)))

    def sqrt(self) -> Scalar:
        """Exact principal square root.

        Returns:
            `sqrt(self)`.
        """
        return Scalar._wrap(sympy.sqrt(self._expression))

    def exp(self) -> Scalar:
        """Exact exponential.

        Returns:
            `exp(self)`.
        """
        return Scalar._wrap(sympy.exp(self._expression))

    @override
    def simplify(self) -> Scalar:
        """An equal expression, in a simpler form when one is found.

        Returns:
            The simplified scalar; its form is not promised.
        """
        return Scalar._wrap(self._simplified())

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
        return Layout(label="Scalar", entries=(self,), shape=())

    @override
    def __repr__(self) -> str:
        """Print as sympy prints the expression.

        Returns:
            The expression, e.g. `pi/2`.
        """
        return str(self._expression)


def _is_zero(value: Scalar) -> bool:
    return bool(value._simplified() == 0)


def _has_symbols(value: Scalar) -> bool:
    return bool(value._expression.free_symbols)


class Symbol(Scalar):
    """A real symbol: a variable of the formulas, matched by name at compile time."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        """Create the real symbol with this name.

        Args:
            name: The symbol's name; equal names give the same symbol.

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


class MissingParameterError(ValueError):
    """A free symbol of a compiled quantity has no field in the parameters dataclass."""

    def __init__(
        self, parameters: type[DataclassInstance], names: frozenset[str]
    ) -> None:
        """Name the parameters class and the symbols without a field.

        Args:
            parameters: The parameters dataclass.
            names: The symbols without a field.
        """
        super().__init__(
            f"{parameters.__name__} has no field for {', '.join(sorted(names))}"
        )


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


class Compiled[P: DataclassInstance, T: Symbolic]:
    """A symbolic quantity compiled to a JIT-compiled JAX function of `P`."""

    _evaluate: Callable[..., Numeric[T]]

    def __init__(self, quantity: T, parameters: type[P]) -> None:
        """Compile the quantity; validation happens once, here.

        The expressions are compiled as written, without simplification; repeated
        subexpressions are computed once.

        Args:
            quantity: The symbolic quantity.
            parameters: A dataclass whose fields name the symbols of the quantity,
                each annotated `float` (one configuration) or `jax.typing.ArrayLike`
                (batched code). Extra fields are allowed.

        Raises:
            TypeError: `parameters` is not a dataclass.
            ParameterTypeError: A field is annotated neither `float` nor `ArrayLike`.
            MissingParameterError: A free symbol of the quantity has no field.
        """
        if not dataclasses.is_dataclass(parameters):
            message = f"parameters must be a dataclass, not {parameters!r}"
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
            raise MissingParameterError(parameters, missing)
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


class Numeric[T: Symbolic](_Incomparable):
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
    def _flatten(numeric: Numeric[Symbolic]) -> tuple[tuple[jax.Array], str]:
        return (numeric._components,), numeric._label

    @staticmethod
    def _unflatten(label: str, children: tuple[jax.Array]) -> Numeric[Symbolic]:
        (components,) = children
        return Numeric[Symbolic](label, components)


jax.tree_util.register_pytree_node(Numeric, Numeric._flatten, Numeric._unflatten)
