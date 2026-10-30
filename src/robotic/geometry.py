"""Frames, and positions and displacements resolved in a frame.

A frame is a type the user declares. A position is where something is, measured from
the frame's origin (a point); a displacement is how much and which way to move, read
on the frame's axes only (a free vector). Within one frame the affine rules hold:
position - position is a displacement, position +- displacement is a position,
displacement +- displacement is a displacement, a number times a displacement is a
displacement. Anything else, and any mix of frames, is a type error, and raises when
executed.

Every position and displacement is constant (`[Scalar, F]`) when built from constants
only, and symbolic (`[Expression, F]`) otherwise, with every component an expression.
"""

from typing import TYPE_CHECKING, ClassVar, Self, final, overload, override

from robotic.symbolic import (
    Expression,
    Layout,
    Scalar,
    ScalarLike,
    Symbolic,
    _Constants,
    _Expressions,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class Frame:
    """A coordinate frame, declared by subclassing: `class Base(Frame, notation="0")`.

    Attributes:
        notation: How the course writes the frame in superscripts and subscripts; the
            class name unless declared.
    """

    notation: ClassVar[str] = "Frame"

    def __init_subclass__(cls, *, notation: str | None = None) -> None:
        """Record how the course writes the frame.

        Args:
            notation: The frame as the course writes it, e.g. `"0"` for ^0 or `"3'"`;
                the class name when omitted.

        Raises:
            TypeError: The notation is not a string.
        """
        super().__init_subclass__()
        if notation is not None and not isinstance(notation, str):
            message = f"a frame's notation is a string, not {type(notation).__name__}"
            raise TypeError(message)
        cls.notation = cls.__name__ if notation is None else notation

    @overload
    @classmethod
    def position[J: (Scalar, Expression)](
        cls, x: J, y: J, z: J
    ) -> Position[J, Self]: ...

    @overload
    @classmethod
    def position(
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike
    ) -> Position[Scalar, Self]: ...

    @overload
    @classmethod
    def position(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Position[Expression, Self]: ...

    @classmethod
    def position(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Position[Scalar, Self] | Position[Expression, Self]:
        """The position with these coordinates, measured from the origin.

        Returns:
            The position, resolved in this frame; constant when every coordinate is.
        """
        return Position(x, y, z, cls)

    @overload
    @classmethod
    def displacement[J: (Scalar, Expression)](
        cls, x: J, y: J, z: J
    ) -> Displacement[J, Self]: ...

    @overload
    @classmethod
    def displacement(
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike
    ) -> Displacement[Scalar, Self]: ...

    @overload
    @classmethod
    def displacement(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Displacement[Expression, Self]: ...

    @classmethod
    def displacement(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Displacement[Scalar, Self] | Displacement[Expression, Self]:
        """The displacement with these components along the axes.

        Returns:
            The displacement, resolved in this frame; constant when every component
            is.
        """
        return Displacement(x, y, z, cls)

    @overload
    @classmethod
    def direction[J: (Scalar, Expression)](
        cls, x: J, y: J, z: J
    ) -> Direction[J, Self]: ...

    @overload
    @classmethod
    def direction(
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike
    ) -> Direction[Scalar, Self]: ...

    @overload
    @classmethod
    def direction(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Direction[Expression, Self]: ...

    @classmethod
    def direction(
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
    ) -> Direction[Scalar, Self] | Direction[Expression, Self]:
        """The direction of these components: the displacement normalized.

        Returns:
            The direction, resolved in this frame.
        """
        return Direction(Displacement(x, y, z, cls))

    @classmethod
    def origin(cls) -> Position[Scalar, Self]:
        """The frame's origin.

        Returns:
            The position (0, 0, 0), resolved in this frame.
        """
        return Position(0, 0, 0, cls)


class FrameMismatchError(TypeError):
    """Two values of an operation are resolved in different frames.

    The runtime image of the type error the type checker reports for the same line.
    """

    def __init__(
        self, operation: str, expected: type[Frame], found: type[Frame]
    ) -> None:
        """Name the operation and both frames.

        Args:
            operation: The operation and its operands, e.g. `add Displacement[Tool]
                and Displacement[Base]`.
            expected: The frame the operation requires.
            found: The frame it was given.
        """
        super().__init__(
            f"cannot {operation}: frame {found.__name__} where "
            f"{expected.__name__} is expected"
        )


class ZeroVectorError(ValueError):
    """A constant displacement whose norm is close to zero has no direction."""

    def __init__[J: (Scalar, Expression), G: Frame](
        self, displacement: Displacement[J, G]
    ) -> None:
        """Name the displacement.

        Args:
            displacement: The displacement that was normalized.
        """
        super().__init__(f"cannot normalize the zero vector {displacement!r}")


class _Resolved[K: (Scalar, Expression), F: Frame](Symbolic[K]):
    """Three scalars read in the frame `F`: storage of positions and displacements."""

    __slots__ = ("_frame", "_x", "_y", "_z")

    _x: K
    _y: K
    _z: K
    _frame: type[F]

    def _with(self, x: K, y: K, z: K) -> Self:
        made = object.__new__(type(self))
        made._x, made._y, made._z = x, y, z
        made._frame = self._frame
        return made

    @property
    def frame(self) -> type[F]:
        """The frame the numbers are read in."""
        return self._frame

    @property
    def x(self) -> K:
        """The number along the x axis."""
        return self._x

    @property
    def y(self) -> K:
        """The number along the y axis."""
        return self._y

    @property
    def z(self) -> K:
        """The number along the z axis."""
        return self._z

    def _kind(self) -> str:
        return f"{type(self).__name__}[{self._frame.__name__}]"

    @override
    def simplify(self) -> Self:
        """Simplify each number.

        Returns:
            The same quantity, in the same frame, with simplified numbers.
        """
        return self._with(self._x.simplify(), self._y.simplify(), self._z.simplify())

    @override
    def latex(self) -> str:
        """Render as a column with the frame as left superscript.

        Returns:
            The column of numbers, prefixed by `{}^{F}` with the frame's notation.
        """
        column = ((self._x,), (self._y,), (self._z,))
        return rf"{{}}^{{{self._frame.notation}}}{self._bmatrix(column)}"

    @override
    def _layout(self) -> Layout:
        return Layout(
            label=self._kind(),
            entries=(self._x, self._y, self._z),
            shape=(3,),
        )

    @override
    def __repr__(self) -> str:
        """Print the kind, the frame and the numbers.

        Returns:
            E.g. `Position[Tool](l2, 0, 0)`.
        """
        return f"{self._kind()}({self._x}, {self._y}, {self._z})"


class Displacement[K: (Scalar, Expression), F: Frame](_Resolved[K, F]):
    """How much and which way to move, resolved on the axes of the frame `F`.

    A free vector: it has no anchor, so a change of frame rotates it and never
    translates it. `frame.displacement(x, y, z)` is the usual spelling of
    `Displacement(x, y, z, frame)`.
    """

    __slots__ = ()

    @overload
    def __new__[G: Frame](
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike, frame: type[G]
    ) -> Displacement[Scalar, G]: ...

    @overload
    def __new__[G: Frame](
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
        frame: type[G],
    ) -> Displacement[Expression, G]: ...

    def __new__[G: Frame](
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
        frame: type[G],
    ) -> Displacement[Scalar, G] | Displacement[Expression, G]:
        """Read three numbers on the axes of a frame.

        Args:
            x: The number along the x axis of `frame`.
            y: The number along the y axis of `frame`.
            z: The number along the z axis of `frame`.
            frame: The frame the numbers are read in.

        Returns:
            The displacement; constant when every number is.

        Raises:
            TypeError: `frame` is not a frame, or a number is not exact.
        """
        if not (isinstance(frame, type) and issubclass(frame, Frame)):
            message = f"expected a frame, a subclass of Frame, not {frame!r}"
            raise TypeError(message)
        match _Constants.or_expressions((x, y, z)):
            case _Constants(values):
                return Displacement._of(values, frame)
            case _Expressions(values):
                return Displacement._of(values, frame)

    @classmethod
    def _of[J: (Scalar, Expression), G: Frame](
        cls, values: tuple[J, ...], frame: type[G]
    ) -> Displacement[J, G]:
        made: Displacement[J, G] = object.__new__(Displacement)
        made._x, made._y, made._z = values
        made._frame = frame
        return made

    @overload
    def __add__(self, other: Displacement[K, F]) -> Displacement[K, F]: ...

    @overload
    def __add__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Expression, F]: ...

    def __add__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Scalar, F] | Displacement[Expression, F]:
        """Add two displacements of the same frame.

        Returns:
            The sum, in the same frame.

        Raises:
            FrameMismatchError: The displacements are in different frames.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        if other.frame is not self._frame:
            operation = f"add {self._kind()} and {other._kind()}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        return Displacement(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )

    @overload
    def __sub__(self, other: Displacement[K, F]) -> Displacement[K, F]: ...

    @overload
    def __sub__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Expression, F]: ...

    def __sub__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Scalar, F] | Displacement[Expression, F]:
        """Subtract two displacements of the same frame.

        Returns:
            The difference, in the same frame.

        Raises:
            FrameMismatchError: The displacements are in different frames.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        if other.frame is not self._frame:
            operation = f"subtract {self._kind()} and {other._kind()}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        return Displacement(
            self._x - other.x, self._y - other.y, self._z - other.z, self._frame
        )

    def __neg__(self) -> Displacement[K, F]:
        """Reverse the displacement.

        Returns:
            The opposite displacement, in the same frame.
        """
        return Displacement._of((-self._x, -self._y, -self._z), self._frame)

    @staticmethod
    def _is_factor(value: object) -> bool:
        return isinstance(value, Scalar | Expression | int) and not isinstance(
            value, bool
        )

    @overload
    def __mul__(self, factor: K | ScalarLike) -> Displacement[K, F]: ...

    @overload
    def __mul__(self, factor: Expression) -> Displacement[Expression, F]: ...

    def __mul__(
        self, factor: K | ScalarLike | Expression
    ) -> Displacement[K, F] | Displacement[Expression, F]:
        """Scale by a number.

        Returns:
            The scaled displacement, in the same frame.
        """
        if not Displacement._is_factor(factor):
            return NotImplemented
        return Displacement(
            self._x * factor, self._y * factor, self._z * factor, self._frame
        )

    @overload
    def __rmul__(self, factor: K | ScalarLike) -> Displacement[K, F]: ...

    @overload
    def __rmul__(self, factor: Expression) -> Displacement[Expression, F]: ...

    def __rmul__(
        self, factor: K | ScalarLike | Expression
    ) -> Displacement[K, F] | Displacement[Expression, F]:
        """Scale by a number written on the left, as in `l1 * x_axis`.

        Returns:
            The scaled displacement, in the same frame.
        """
        return self * factor

    @overload
    def __truediv__(self, divisor: K | ScalarLike) -> Displacement[K, F]: ...

    @overload
    def __truediv__(self, divisor: Expression) -> Displacement[Expression, F]: ...

    def __truediv__(
        self, divisor: K | ScalarLike | Expression
    ) -> Displacement[K, F] | Displacement[Expression, F]:
        """Divide by a number.

        Returns:
            The scaled displacement, in the same frame.
        """
        if not Displacement._is_factor(divisor):
            return NotImplemented
        return Displacement(
            self._x / divisor, self._y / divisor, self._z / divisor, self._frame
        )

    @overload
    def dot(self, other: Displacement[K, F]) -> K: ...

    @overload
    def dot(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Expression: ...

    def dot(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Scalar | Expression:
        """Dot product with a displacement of the same frame.

        Returns:
            `self . other`.

        Raises:
            TypeError: `other` is not a displacement.
            FrameMismatchError: The displacements are in different frames.
        """
        if not isinstance(other, Displacement):
            message = f"dot takes a displacement, not {type(other).__name__}"
            raise TypeError(message)
        if other.frame is not self._frame:
            operation = f"take the dot product of {self._kind()} and {other!r}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        return self._x * other.x + self._y * other.y + self._z * other.z

    @overload
    def cross(self, other: Displacement[K, F]) -> Displacement[K, F]: ...

    @overload
    def cross(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Expression, F]: ...

    def cross(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Displacement[Scalar, F] | Displacement[Expression, F]:
        """Cross product with a displacement of the same frame.

        Returns:
            `self x other`, in the same frame.

        Raises:
            TypeError: `other` is not a displacement.
            FrameMismatchError: The displacements are in different frames.
        """
        if not isinstance(other, Displacement):
            message = f"cross takes a displacement, not {type(other).__name__}"
            raise TypeError(message)
        if other.frame is not self._frame:
            operation = f"take the cross product of {self._kind()} and {other!r}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        return Displacement(
            self._y * other.z - self._z * other.y,
            self._z * other.x - self._x * other.z,
            self._x * other.y - self._y * other.x,
            self._frame,
        )

    def norm(self) -> K:
        """Euclidean length, not simplified.

        Returns:
            `sqrt(x**2 + y**2 + z**2)`.
        """
        squares = self._x * self._x + self._y * self._y + self._z * self._z
        return squares ** (Scalar(1) / 2)

    def normalize(self) -> Direction[K, F]:
        """The direction of this displacement; `Direction(self)`.

        Returns:
            The direction, in the same frame.
        """
        return Direction(self)

    def evaluate(
        self: Displacement[Expression, F],
        values: Mapping[str, ScalarLike] | None = None,
    ) -> Displacement[Scalar, F]:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed.

        Returns:
            The constant displacement, in the same frame.
        """
        return Displacement._of(
            (
                self._x.evaluate(values),
                self._y.evaluate(values),
                self._z.evaluate(values),
            ),
            self._frame,
        )


class Direction[K: (Scalar, Expression), F: Frame](Displacement[K, F]):
    """A displacement of length one, obtained only by normalizing."""

    __slots__ = ()

    def __new__[J: (Scalar, Expression), G: Frame](
        cls, displacement: Displacement[J, G]
    ) -> Direction[J, G]:
        """Normalize: divide the components by their norm, not simplified.

        A constant displacement whose norm is close to zero has no direction and
        raises. A symbolic one never raises: where its norm vanishes, at some
        configurations or at all, the direction is undefined.

        Args:
            displacement: Any displacement, in the frame of the result.

        Returns:
            The direction.
        """
        made: Direction[J, G] = object.__new__(Direction)
        made._x, made._y, made._z = cls._unit(displacement)
        made._frame = displacement.frame
        return made

    @staticmethod
    def _unit[J: (Scalar, Expression), G: Frame](
        displacement: Displacement[J, G],
    ) -> tuple[J, J, J]:
        if not isinstance(displacement, Displacement):
            message = f"a direction normalizes a displacement, not {displacement!r}"
            raise TypeError(message)
        length = displacement.norm()
        if isinstance(length, Scalar) and length.is_close(0):
            raise ZeroVectorError(displacement)
        return (
            displacement.x / length,
            displacement.y / length,
            displacement.z / length,
        )

    @override
    def __neg__(self) -> Self:
        """Reverse the direction; the length stays one, so nothing is normalized.

        Returns:
            The opposite direction, in the same frame.
        """
        return self._with(-self._x, -self._y, -self._z)


class FreeAxis[K: (Scalar, Expression), F: Frame](Direction[K, F]):
    """A direction chosen freely, since any axis is valid (the identity rotation)."""

    __slots__ = ()

    def __new__[J: (Scalar, Expression), G: Frame](
        cls, displacement: Displacement[J, G]
    ) -> FreeAxis[J, G]:
        """Normalize the chosen axis.

        Args:
            displacement: The chosen axis, in the frame of the result.

        Returns:
            The free axis.
        """
        made: FreeAxis[J, G] = object.__new__(FreeAxis)
        made._x, made._y, made._z = cls._unit(displacement)
        made._frame = displacement.frame
        return made


@final
class Position[K: (Scalar, Expression), F: Frame](_Resolved[K, F]):
    """Where something is, measured from the origin of the frame `F` (a point).

    A position vector: a change of frame rotates it and moves it with the origin.
    `frame.position(x, y, z)` is the usual spelling of `Position(x, y, z, frame)`.
    """

    __slots__ = ()

    @overload
    def __new__[G: Frame](
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike, frame: type[G]
    ) -> Position[Scalar, G]: ...

    @overload
    def __new__[G: Frame](
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
        frame: type[G],
    ) -> Position[Expression, G]: ...

    def __new__[G: Frame](
        cls,
        x: ScalarLike | Expression,
        y: ScalarLike | Expression,
        z: ScalarLike | Expression,
        frame: type[G],
    ) -> Position[Scalar, G] | Position[Expression, G]:
        """Read three coordinates from the origin of a frame.

        Args:
            x: The coordinate along the x axis of `frame`.
            y: The coordinate along the y axis of `frame`.
            z: The coordinate along the z axis of `frame`.
            frame: The frame the coordinates are read in.

        Returns:
            The position; constant when every coordinate is.

        Raises:
            TypeError: `frame` is not a frame, or a coordinate is not exact.
        """
        if not (isinstance(frame, type) and issubclass(frame, Frame)):
            message = f"expected a frame, a subclass of Frame, not {frame!r}"
            raise TypeError(message)
        match _Constants.or_expressions((x, y, z)):
            case _Constants(values):
                return Position._of(values, frame)
            case _Expressions(values):
                return Position._of(values, frame)

    @classmethod
    def _of[J: (Scalar, Expression), G: Frame](
        cls, values: tuple[J, ...], frame: type[G]
    ) -> Position[J, G]:
        made: Position[J, G] = object.__new__(Position)
        made._x, made._y, made._z = values
        made._frame = frame
        return made

    @overload
    def __sub__(self, other: Position[K, F]) -> Displacement[K, F]: ...

    @overload
    def __sub__(self, other: Displacement[K, F]) -> Position[K, F]: ...

    @overload
    def __sub__(
        self, other: Position[Scalar, F] | Position[Expression, F]
    ) -> Displacement[Expression, F]: ...

    @overload
    def __sub__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Position[Expression, F]: ...

    def __sub__(
        self,
        other: Position[Scalar, F]
        | Position[Expression, F]
        | Displacement[Scalar, F]
        | Displacement[Expression, F],
    ) -> (
        Displacement[Scalar, F]
        | Displacement[Expression, F]
        | Position[Scalar, F]
        | Position[Expression, F]
    ):
        """Subtract a position (the displacement between them) or a displacement.

        Returns:
            `self - other`: a displacement for a position, a position for a
            displacement.

        Raises:
            FrameMismatchError: The operands are in different frames.
        """
        if not isinstance(other, Position | Displacement):
            return NotImplemented
        if other.frame is not self._frame:
            operation = f"subtract {self._kind()} and {other._kind()}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        x, y, z = self._x - other.x, self._y - other.y, self._z - other.z
        match other:
            case Position():
                return Displacement(x, y, z, self._frame)
            case Displacement():
                return Position(x, y, z, self._frame)

    @overload
    def __add__(self, other: Displacement[K, F]) -> Position[K, F]: ...

    @overload
    def __add__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Position[Expression, F]: ...

    def __add__(
        self, other: Displacement[Scalar, F] | Displacement[Expression, F]
    ) -> Position[Scalar, F] | Position[Expression, F]:
        """Move the position by a displacement of the same frame.

        Returns:
            The moved position, in the same frame.

        Raises:
            FrameMismatchError: The operands are in different frames.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        if other.frame is not self._frame:
            operation = f"add {self._kind()} and {other._kind()}"
            raise FrameMismatchError(operation, self._frame, other.frame)
        return Position(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )

    def evaluate(
        self: Position[Expression, F],
        values: Mapping[str, ScalarLike] | None = None,
    ) -> Position[Scalar, F]:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed.

        Returns:
            The constant position, in the same frame.
        """
        return Position._of(
            (
                self._x.evaluate(values),
                self._y.evaluate(values),
                self._z.evaluate(values),
            ),
            self._frame,
        )


Point = Position
