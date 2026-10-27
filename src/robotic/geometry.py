"""Frames, and positions and displacements resolved in a frame.

A frame is a type the user declares. A position is where something is, measured from
the frame's origin (a point); a displacement is how much and which way to move, read
on the frame's axes only (a free vector). Within one frame the affine rules hold:
position - position is a displacement, position +- displacement is a position,
displacement +- displacement is a displacement, a number times a displacement is a
displacement. Anything else, and any mix of frames, is a type error, and raises when
executed.
"""

from typing import ClassVar, Self, final, overload, override

from robotic.symbolic import Layout, Scalar, ScalarLike, Symbolic, _is_zero


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

    @classmethod
    def position(cls, x: ScalarLike, y: ScalarLike, z: ScalarLike) -> Position[Self]:
        """The position with these coordinates, measured from the origin.

        Returns:
            The position, resolved in this frame.
        """
        return Position(x, y, z, cls)

    @classmethod
    def displacement(
        cls, x: ScalarLike, y: ScalarLike, z: ScalarLike
    ) -> Displacement[Self]:
        """The displacement with these components along the axes.

        Returns:
            The displacement, resolved in this frame.
        """
        return Displacement(x, y, z, cls)

    @classmethod
    def direction(cls, x: ScalarLike, y: ScalarLike, z: ScalarLike) -> Direction[Self]:
        """The direction of these components: the displacement normalized.

        Returns:
            The direction, resolved in this frame.
        """
        return Direction(Displacement(x, y, z, cls))

    @classmethod
    def origin(cls) -> Position[Self]:
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
    """A displacement whose components simplify to zero has no direction."""

    def __init__[F: Frame](self, displacement: Displacement[F]) -> None:
        """Name the displacement.

        Args:
            displacement: The displacement that was normalized.
        """
        super().__init__(f"cannot normalize the zero vector {displacement!r}")


class _Resolved[F: Frame](Symbolic):
    """Three scalars read in the frame `F`: storage of positions and displacements."""

    __slots__ = ("_frame", "_x", "_y", "_z")

    def __init__(
        self, x: ScalarLike, y: ScalarLike, z: ScalarLike, frame: type[F]
    ) -> None:
        """Read three numbers in a frame.

        Args:
            x: The number along the x axis of `frame`.
            y: The number along the y axis of `frame`.
            z: The number along the z axis of `frame`.
            frame: The frame the numbers are read in.

        Raises:
            TypeError: `frame` is not a frame.
        """
        if not (isinstance(frame, type) and issubclass(frame, Frame)):
            message = f"expected a frame, a subclass of Frame, not {frame!r}"
            raise TypeError(message)
        self._frame = frame
        self._x = Scalar(x)
        self._y = Scalar(y)
        self._z = Scalar(z)

    @property
    def frame(self) -> type[F]:
        """The frame the numbers are read in."""
        return self._frame

    @property
    def x(self) -> Scalar:
        """The number along the x axis."""
        return self._x

    @property
    def y(self) -> Scalar:
        """The number along the y axis."""
        return self._y

    @property
    def z(self) -> Scalar:
        """The number along the z axis."""
        return self._z

    def _kind(self) -> str:
        return f"{type(self).__name__}[{self._frame.__name__}]"

    def _same_frame(self, verb: str, other: _Resolved[Frame]) -> None:
        if other.frame is not self._frame:
            operation = f"{verb} {self._kind()} and {other._kind()}"
            raise FrameMismatchError(operation, self._frame, other.frame)

    @override
    def simplify(self) -> Self:
        """Simplify each number.

        Returns:
            The same quantity, in the same frame, with simplified numbers.
        """
        return type(self)(
            self._x.simplify(), self._y.simplify(), self._z.simplify(), self._frame
        )

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


class Displacement[F: Frame](_Resolved[F]):
    """How much and which way to move, resolved on the axes of the frame `F`.

    A free vector: it has no anchor, so a change of frame rotates it and never
    translates it. `frame.displacement(x, y, z)` is the usual spelling of
    `Displacement(x, y, z, frame)`.
    """

    __slots__ = ()

    def __add__(self, other: Displacement[F]) -> Displacement[F]:
        """Add two displacements of the same frame.

        Returns:
            The sum, in the same frame.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        self._same_frame("add", other)
        return Displacement(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )

    def __sub__(self, other: Displacement[F]) -> Displacement[F]:
        """Subtract two displacements of the same frame.

        Returns:
            The difference, in the same frame.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        self._same_frame("subtract", other)
        return Displacement(
            self._x - other.x, self._y - other.y, self._z - other.z, self._frame
        )

    def __neg__(self) -> Displacement[F]:
        """Reverse the displacement.

        Returns:
            The opposite displacement, in the same frame.
        """
        return Displacement(-self._x, -self._y, -self._z, self._frame)

    def __mul__(self, factor: ScalarLike) -> Displacement[F]:
        """Scale by a number.

        Returns:
            The scaled displacement, in the same frame.
        """
        scale = Scalar(factor)
        return Displacement(
            self._x * scale, self._y * scale, self._z * scale, self._frame
        )

    def __rmul__(self, factor: ScalarLike) -> Displacement[F]:
        """Scale by a number written on the left, as in `l1 * x_axis`.

        Returns:
            The scaled displacement, in the same frame.
        """
        return self * factor

    def __truediv__(self, divisor: ScalarLike) -> Displacement[F]:
        """Divide by a number.

        Returns:
            The scaled displacement, in the same frame.
        """
        scale = Scalar(divisor)
        return Displacement(
            self._x / scale, self._y / scale, self._z / scale, self._frame
        )

    def dot(self, other: Displacement[F]) -> Scalar:
        """Dot product with a displacement of the same frame.

        Returns:
            The scalar `self . other`.

        Raises:
            TypeError: `other` is not a displacement.
        """
        if not isinstance(other, Displacement):
            message = f"dot takes a displacement, not {type(other).__name__}"
            raise TypeError(message)
        self._same_frame("take the dot product of", other)
        return self._x * other.x + self._y * other.y + self._z * other.z

    def cross(self, other: Displacement[F]) -> Displacement[F]:
        """Cross product with a displacement of the same frame.

        Returns:
            `self x other`, in the same frame.

        Raises:
            TypeError: `other` is not a displacement.
        """
        if not isinstance(other, Displacement):
            message = f"cross takes a displacement, not {type(other).__name__}"
            raise TypeError(message)
        self._same_frame("take the cross product of", other)
        return Displacement(
            self._y * other.z - self._z * other.y,
            self._z * other.x - self._x * other.z,
            self._x * other.y - self._y * other.x,
            self._frame,
        )

    def norm(self) -> Scalar:
        """Euclidean length, not simplified.

        Returns:
            `sqrt(x**2 + y**2 + z**2)`.
        """
        return self.dot(self).sqrt()

    def normalize(self) -> Direction[F]:
        """The direction of this displacement; `Direction(self)`.

        Returns:
            The direction, in the same frame.
        """
        return Direction(self)


class Direction[F: Frame](Displacement[F]):
    """A displacement of length one, obtained only by normalizing."""

    __slots__ = ()

    def __init__(self, displacement: Displacement[F]) -> None:
        """Normalize: divide the components by their simplified norm.

        Args:
            displacement: Any displacement with a direction, in the frame of the
                result.

        Raises:
            TypeError: The argument is not a displacement.
            ZeroVectorError: The norm simplifies to zero.
        """
        if not isinstance(displacement, Displacement):
            message = f"a direction normalizes a displacement, not {displacement!r}"
            raise TypeError(message)
        length = displacement.norm().simplify()
        if _is_zero(length):
            raise ZeroVectorError(displacement)
        super().__init__(
            displacement.x / length,
            displacement.y / length,
            displacement.z / length,
            displacement.frame,
        )

    def _of_unit_components(self, x: Scalar, y: Scalar, z: Scalar) -> Self:
        unit: Self = object.__new__(type(self))
        unit._x, unit._y, unit._z = x, y, z
        unit._frame = self._frame
        return unit

    @override
    def __neg__(self) -> Self:
        """Reverse the direction; the length stays one, so nothing is normalized.

        Returns:
            The opposite direction, in the same frame.
        """
        return self._of_unit_components(-self._x, -self._y, -self._z)

    @override
    def simplify(self) -> Self:
        """Simplify each component; the length stays one, so nothing is normalized.

        Returns:
            The same direction, in the same frame, with simplified components.
        """
        return self._of_unit_components(
            self._x.simplify(), self._y.simplify(), self._z.simplify()
        )


class FreeAxis[F: Frame](Direction[F]):
    """A direction chosen freely, since any axis is valid (the identity rotation)."""

    __slots__ = ()


@final
class Position[F: Frame](_Resolved[F]):
    """Where something is, measured from the origin of the frame `F` (a point).

    A position vector: a change of frame rotates it and moves it with the origin.
    `frame.position(x, y, z)` is the usual spelling of `Position(x, y, z, frame)`.
    """

    __slots__ = ()

    @overload
    def __sub__(self, other: Position[F]) -> Displacement[F]: ...

    @overload
    def __sub__(self, other: Displacement[F]) -> Position[F]: ...

    def __sub__(
        self, other: Position[F] | Displacement[F]
    ) -> Displacement[F] | Position[F]:
        """Subtract a position (the displacement between them) or a displacement.

        Returns:
            `self - other`: a displacement for a position, a position for a
            displacement.
        """
        if not isinstance(other, Position | Displacement):
            return NotImplemented
        self._same_frame("subtract", other)
        x, y, z = self._x - other.x, self._y - other.y, self._z - other.z
        match other:
            case Position():
                return Displacement(x, y, z, self._frame)
            case Displacement():
                return Position(x, y, z, self._frame)

    def __add__(self, other: Displacement[F]) -> Position[F]:
        """Move the position by a displacement of the same frame.

        Returns:
            The moved position, in the same frame.
        """
        if not isinstance(other, Displacement):
            return NotImplemented
        self._same_frame("add", other)
        return Position(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )


Point = Position
