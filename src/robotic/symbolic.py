"""The typed boundary around sympy: symbolic scalars, compilation and compiled results.

This is the only module that touches sympy. Every other module builds on the public
`Scalar` API, so formulas stay typed from the first symbol to the last simplification.
"""

import dataclasses
import operator
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import jax
import jax.numpy as jnp
import sympy
from jax.core import Tracer
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import DataclassInstance

    from robotic.geometry import Frame
    from robotic.rotation import AxisAngleSolutions

jax.config.update(name="jax_enable_x64", val=True)

type ScalarLike = Scalar | int | float


class Symbolic(ABC):
    """A symbolic quantity: simplifiable, printable as LaTeX and compilable to JAX."""

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

    def lambdify[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]:
        """Compile the quantity to a JAX function of a parameters dataclass.

        Args:
            parameters: A dataclass whose fields, each annotated `float` or
                `jax.typing.ArrayLike`, name the symbols of the quantity. Extra fields
                are allowed.

        Returns:
            The compiled function.
        """
        return Compiled(self, parameters)


@dataclass(frozen=True)
class Layout:
    """How a quantity flattens to scalars for compilation.

    Attributes:
        label: The quantity's type as printed in a compiled result, e.g. `Point[Base]`.
        entries: The scalars, row-major.
        shape: The natural shape of the compiled components.
        tolerance: When set, the compiled components are the flag `|entry| > tolerance`.
        condition: When set, a validity condition compiled alongside into the result's
            flag; otherwise the flag is true.
    """

    label: str
    entries: tuple[Scalar, ...]
    shape: tuple[int, ...]
    tolerance: float | None = None
    condition: Symbolic | None = None


class Scalar(Symbolic):
    """A symbolic scalar expression.

    Arithmetic accepts a scalar, an `int` or a `float` on either side. Integers stay
    exact; floats enter the expression as floating-point constants, as written.
    """

    __slots__ = ("_expression",)

    _expression: sympy.Expr

    def __init__(self, value: ScalarLike) -> None:
        """Build a constant scalar.

        Args:
            value: An exact integer, a float kept as written, or a scalar to copy.
        """
        self._expression = Scalar._expression_of(value)

    @classmethod
    def _wrap(cls, expression: sympy.Expr) -> Scalar:
        scalar = object.__new__(Scalar)
        scalar._expression = expression
        return scalar

    @staticmethod
    def _expression_of(value: ScalarLike) -> sympy.Expr:
        match value:
            case Scalar():
                return value._expression
            case int():
                return sympy.Integer(value)
            case float():
                return sympy.Float(value)

    def _combined(
        self,
        other: ScalarLike,
        operation: Callable[[sympy.Expr, sympy.Expr], sympy.Expr],
    ) -> Scalar:
        if not isinstance(other, Scalar | int | float):
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

    def __radd__(self, other: float) -> Scalar:
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

    def __rsub__(self, other: float) -> Scalar:
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

    def __rmul__(self, other: float) -> Scalar:
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

    def __rtruediv__(self, other: float) -> Scalar:
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

    def __rpow__(self, base: float) -> Scalar:
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

    def is_zero(self) -> bool:
        """Whether the scalar simplifies to exactly zero.

        Returns:
            True only when simplification proves the scalar is identically zero.
        """
        return bool(sympy.simplify(self._expression) == 0)

    @override
    def simplify(self) -> Scalar:
        """Simplify the expression.

        Returns:
            The simplified scalar.
        """
        return Scalar._wrap(sympy.simplify(self._expression))

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


class Symbol(Scalar):
    """A real symbol: a variable of the formulas, matched by name at compile time."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        """Create the real symbol with this name.

        Args:
            name: The symbol's name; equal names give the same symbol.
        """
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

        Args:
            quantity: The symbolic quantity.
            parameters: A dataclass whose fields name the symbols of the quantity,
                each annotated `float` (one configuration) or `jax.typing.ArrayLike`
                (batched code). Extra fields are allowed.

        Raises:
            ParameterTypeError: A field is annotated neither `float` nor `ArrayLike`.
            MissingParameterError: A free symbol of the quantity has no field.
        """
        layout = quantity._layout()
        hints = typing.get_type_hints(parameters)
        names = tuple(field.name for field in dataclasses.fields(parameters))
        for name in names:
            if hints[name] is not float and hints[name] != ArrayLike:
                raise ParameterTypeError(parameters, name)
        condition = layout.condition._layout() if layout.condition else None
        flag = condition.entries if condition else ()
        expressions = [entry._expression for entry in (*layout.entries, *flag)]
        free = frozenset(
            str(symbol)
            for expression in expressions
            for symbol in expression.free_symbols
        )
        missing = free - frozenset(names)
        if missing:
            raise MissingParameterError(parameters, missing)
        symbols = [sympy.Symbol(name, real=True) for name in names]
        self._names = names
        self._layout = layout
        self._condition = condition
        self._function = sympy.lambdify(symbols, expressions, modules="jax")
        self._evaluate = jax.jit(self._kernel)

    @staticmethod
    def _shaped(values: list[jax.Array], layout: Layout) -> jax.Array:
        components = jnp.stack(values).reshape(layout.shape)
        if layout.tolerance is None:
            return components
        return jnp.abs(components) > layout.tolerance

    def _kernel(self, *values: jax.Array) -> Numeric[T]:
        for name, value in zip(self._names, values, strict=True):
            if jnp.shape(value):
                raise ParameterShapeError(name, jnp.shape(value))
        results = [
            jnp.asarray(entry, dtype=jnp.float64) for entry in self._function(*values)
        ]
        size = len(self._layout.entries)
        components = self._shaped(results[:size], self._layout)
        valid = (
            self._shaped(results[size:], self._condition)
            if self._condition
            else jnp.ones((), dtype=bool)
        )
        return Numeric(self._layout.label, components, valid)

    def __call__(self, values: P) -> Numeric[T]:
        """Evaluate at one configuration.

        Args:
            values: The parameter values, each of shape `()`; JAX tracers are
                accepted, so the call works under `jax.jit`, `jax.vmap` and
                `jax.grad`. A field holding an array raises `ParameterShapeError`
                when JAX traces; batch with `jax.vmap`.

        Returns:
            The compiled result.
        """
        return self._evaluate(*(getattr(values, name) for name in self._names))


class Numeric[T: Symbolic]:
    """The compiled value of a `T`: bare components, with the meaning kept in the type.

    A registered JAX pytree: the components and the validity flag are its children,
    the printed type its static data, so it passes through `jax.jit`, `jax.vmap` and
    `jax.grad`.
    """

    __slots__ = ("_components", "_label", "_valid")

    def __init__(self, label: str, components: jax.Array, valid: jax.Array) -> None:
        """Pair the components with the printed type and the validity flag.

        Args:
            label: The quantity's type as printed, e.g. `Point[Base]`.
            components: Float64 components in the natural shape (a flag for a
                validity condition); leading axes under `vmap` are configurations.
            valid: Where the quantity's validity condition holds; true when it has
                none.
        """
        self._label = label
        self._components = components
        self._valid = valid

    @property
    def components(self) -> jax.Array:
        """The components in the natural shape of `T`, per configuration."""
        return self._components

    def valid[F: Frame](self: Numeric[AxisAngleSolutions[F]]) -> jax.Array:
        """Whether the axis-angle solutions are valid by more than the tolerance.

        Returns:
            A boolean of shape `()` per configuration.
        """
        return self._valid

    @override
    def __repr__(self) -> str:
        """Print the type and the components.

        Returns:
            E.g. `Numeric[Point[Base]]([1.0, 0.5, 0.0])`; under tracing the components
            print as JAX prints a tracer.
        """
        components = self._components
        shown = components if isinstance(components, Tracer) else components.tolist()
        return f"Numeric[{self._label}]({shown})"

    @staticmethod
    def _flatten(
        numeric: Numeric[Symbolic],
    ) -> tuple[tuple[jax.Array, jax.Array], str]:
        return (numeric._components, numeric._valid), numeric._label

    @staticmethod
    def _unflatten(
        label: str, children: tuple[jax.Array, jax.Array]
    ) -> Numeric[Symbolic]:
        components, valid = children
        return Numeric[Symbolic](label, components, valid)


jax.tree_util.register_pytree_node(Numeric, Numeric._flatten, Numeric._unflatten)
