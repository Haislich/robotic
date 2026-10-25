"""The typed boundary around sympy: symbolic scalars, compilation and compiled results.

This is the only module that touches sympy. Every other module builds on the public
`Scalar` API, so formulas stay typed from the first symbol to the last simplification.
"""

import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

import jax
import jax.numpy as jnp
import sympy

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

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

    def lambdify[P: DataclassInstance](self, parameters: type[P]) -> Compiled[P, Self]:
        """Compile the quantity to a JAX function of a parameters dataclass.

        Args:
            parameters: A dataclass whose fields, all annotated `float`, name the
                symbols of the quantity. Extra fields are allowed.

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
        tolerance: When set, the compiled result is the flag `|entry| > tolerance`.
    """

    label: str
    entries: tuple[Scalar, ...]
    shape: tuple[int, ...]
    tolerance: float | None = None


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
        return Scalar._wrap(self._expression + Scalar._expression_of(other))

    def __radd__(self, other: float) -> Scalar:
        """Add from the left.

        Returns:
            `other + self`.
        """
        return Scalar._wrap(Scalar._expression_of(other) + self._expression)

    def __sub__(self, other: ScalarLike) -> Scalar:
        """Subtract.

        Returns:
            `self - other`.
        """
        return Scalar._wrap(self._expression - Scalar._expression_of(other))

    def __rsub__(self, other: float) -> Scalar:
        """Subtract from a number.

        Returns:
            `other - self`.
        """
        return Scalar._wrap(Scalar._expression_of(other) - self._expression)

    def __mul__(self, other: ScalarLike) -> Scalar:
        """Multiply.

        Returns:
            `self * other`.
        """
        return Scalar._wrap(self._expression * Scalar._expression_of(other))

    def __rmul__(self, other: float) -> Scalar:
        """Multiply from the left.

        Returns:
            `other * self`.
        """
        return Scalar._wrap(Scalar._expression_of(other) * self._expression)

    def __truediv__(self, other: ScalarLike) -> Scalar:
        """Divide; integer quotients stay exact fractions.

        Returns:
            `self / other`.
        """
        return Scalar._wrap(self._expression / Scalar._expression_of(other))

    def __rtruediv__(self, other: float) -> Scalar:
        """Divide a number.

        Returns:
            `other / self`.
        """
        return Scalar._wrap(Scalar._expression_of(other) / self._expression)

    def __pow__(self, exponent: ScalarLike) -> Scalar:
        """Raise to a power.

        Returns:
            `self ** exponent`.
        """
        return Scalar._wrap(self._expression ** Scalar._expression_of(exponent))

    def __rpow__(self, base: float) -> Scalar:
        """Raise a number to this power.

        Returns:
            `base ** self`.
        """
        return Scalar._wrap(Scalar._expression_of(base) ** self._expression)

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
    """A field of the parameters dataclass is not annotated `float`."""

    def __init__(self, parameters: type[DataclassInstance], name: str) -> None:
        """Name the parameters class and the field.

        Args:
            parameters: The parameters dataclass.
            name: The field not annotated `float`.
        """
        super().__init__(
            f"field {name!r} of {parameters.__name__} must be annotated float"
        )


class Compiled[P: DataclassInstance, T: Symbolic]:
    """A symbolic quantity compiled to a JIT-compiled JAX function of `P`."""

    def __init__(self, quantity: T, parameters: type[P]) -> None:
        """Compile the quantity; validation happens once, here.

        Args:
            quantity: The symbolic quantity.
            parameters: A dataclass whose fields, all annotated `float`, name the
                symbols of the quantity. Extra fields are allowed.

        Raises:
            ParameterTypeError: A field is not annotated `float`.
            MissingParameterError: A free symbol of the quantity has no field.
        """
        layout = quantity._layout()
        hints = typing.get_type_hints(parameters)
        names = tuple(field.name for field in dataclasses.fields(parameters))
        for name in names:
            if hints[name] is not float:
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
        self._names = names
        self._layout = layout
        self._function = sympy.lambdify(symbols, expressions, modules="jax")
        self._evaluate = jax.jit(self._kernel)

    def _kernel(self, *values: jax.Array) -> jax.Array:
        entries = [
            jnp.asarray(entry, dtype=jnp.float64) for entry in self._function(*values)
        ]
        components = jnp.stack(entries).reshape(self._layout.shape)
        if self._layout.tolerance is None:
            return components
        return jnp.abs(components) > self._layout.tolerance

    def __call__(self, values: P) -> Numeric[T]:
        """Evaluate at one configuration.

        Args:
            values: The parameter values; JAX tracers are accepted, so the call works
                under `jax.jit`, `jax.vmap` and `jax.grad`.

        Returns:
            The compiled result.
        """
        arguments = [getattr(values, name) for name in self._names]
        return Numeric(self._layout.label, self._evaluate(*arguments))


class Numeric[T: Symbolic]:
    """The compiled value of a `T`: bare components, with the meaning kept in the type.

    A registered JAX pytree: the components are its only child, the printed type its
    static data, so it passes through `jax.jit`, `jax.vmap` and `jax.grad`.
    """

    __slots__ = ("_components", "_label")

    def __init__(self, label: str, components: jax.Array) -> None:
        """Pair the components with the printed type.

        Args:
            label: The quantity's type as printed, e.g. `Point[Base]`.
            components: Float64 components in the natural shape (a flag for a
                validity condition); leading axes under `vmap` are configurations.
        """
        self._label = label
        self._components = components

    @property
    def components(self) -> jax.Array:
        """The components: `()`, `(3,)`, `(3, 3)` or `(4, 4)` per configuration."""
        return self._components

    @override
    def __repr__(self) -> str:
        """Print the type and the components.

        Returns:
            E.g. `Numeric[Point[Base]]([1.0, 0.5, 0.0])`.
        """
        return f"Numeric[{self._label}]({self._components.tolist()})"

    @staticmethod
    def _flatten(numeric: Numeric[Symbolic]) -> tuple[tuple[jax.Array], str]:
        return (numeric._components,), numeric._label

    @staticmethod
    def _unflatten(label: str, children: tuple[jax.Array]) -> Numeric[Symbolic]:
        return Numeric[Symbolic](label, children[0])


jax.tree_util.register_pytree_node(Numeric, Numeric._flatten, Numeric._unflatten)
