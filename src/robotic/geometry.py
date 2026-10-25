"""Frames, and points and vectors resolved in a frame.

A frame is a type the user declares. A frame gives a point its origin and axes and a
vector its axes only, so within one frame the affine rules hold: point - point is a
vector, point +- vector is a point, vector +- vector is a vector, a number times a
vector is a vector. Anything else, and any mix of frames, is a type error.
"""

from typing import ClassVar, Self, final, overload, override

from robotic.symbolic import Layout, Scalar, ScalarLike, Symbolic


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
        """
        super().__init_subclass__()
        cls.notation = cls.__name__ if notation is None else notation

    @classmethod
    def point(cls, x: ScalarLike, y: ScalarLike, z: ScalarLike) -> Point[Self]:
        """The point with these coordinates: the origin plus the axes scaled.

        Returns:
            The point, resolved in this frame.
        """
        return Point(x, y, z, cls)

    @classmethod
    def vector(cls, x: ScalarLike, y: ScalarLike, z: ScalarLike) -> Vector[Self]:
        """The vector with these components along the axes.

        Returns:
            The vector, resolved in this frame.
        """
        return Vector(x, y, z, cls)

    @classmethod
    def origin(cls) -> Point[Self]:
        """The frame's origin.

        Returns:
            The point (0, 0, 0), resolved in this frame.
        """
        return Point(0, 0, 0, cls)


class ZeroVectorError(ValueError):
    """A vector whose components simplify to zero has no direction to normalize."""

    def __init__[F: Frame](self, vector: Vector[F]) -> None:
        """Name the vector.

        Args:
            vector: The vector that was normalized.
        """
        super().__init__(f"cannot normalize the zero vector {vector!r}")


class _Resolved[F: Frame](Symbolic):
    """Three scalars read in the frame `F`: the storage shared by points and vectors."""

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
        """
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
            label=f"{type(self).__name__}[{self._frame.__name__}]",
            entries=(self._x, self._y, self._z),
            shape=(3,),
        )

    @override
    def __repr__(self) -> str:
        """Print the kind, the frame and the numbers.

        Returns:
            E.g. `Point[Tool](l2, 0, 0)`.
        """
        numbers = f"{self._x}, {self._y}, {self._z}"
        return f"{type(self).__name__}[{self._frame.__name__}]({numbers})"


class Vector[F: Frame](_Resolved[F]):
    """An arrow resolved in the frame `F`: its components along the axes of `F`.

    `frame.vector(x, y, z)` is the usual spelling of `Vector(x, y, z, frame)`.
    """

    __slots__ = ()

    def __add__(self, other: Vector[F]) -> Vector[F]:
        """Add two vectors of the same frame.

        Returns:
            The sum, in the same frame.
        """
        return Vector(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )

    def __sub__(self, other: Vector[F]) -> Vector[F]:
        """Subtract two vectors of the same frame.

        Returns:
            The difference, in the same frame.
        """
        return Vector(
            self._x - other.x, self._y - other.y, self._z - other.z, self._frame
        )

    def __neg__(self) -> Vector[F]:
        """Reverse the arrow.

        Returns:
            The opposite vector, in the same frame.
        """
        return Vector(-self._x, -self._y, -self._z, self._frame)

    def __mul__(self, factor: ScalarLike) -> Vector[F]:
        """Scale by a number.

        Returns:
            The scaled vector, in the same frame.
        """
        return Vector(self._x * factor, self._y * factor, self._z * factor, self._frame)

    def __rmul__(self, factor: Scalar | float) -> Vector[F]:
        """Scale by a number written on the left, as in `l1 * x_axis`.

        Returns:
            The scaled vector, in the same frame.
        """
        return self * factor

    def __truediv__(self, divisor: ScalarLike) -> Vector[F]:
        """Divide by a number.

        Returns:
            The scaled vector, in the same frame.
        """
        return Vector(
            self._x / divisor, self._y / divisor, self._z / divisor, self._frame
        )

    def dot(self, other: Vector[F]) -> Scalar:
        """Dot product with a vector of the same frame.

        Returns:
            The scalar `self . other`.
        """
        return self._x * other.x + self._y * other.y + self._z * other.z

    def cross(self, other: Vector[F]) -> Vector[F]:
        """Cross product with a vector of the same frame.

        Returns:
            `self x other`, in the same frame.
        """
        return Vector(
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

    def normalize(self) -> UnitVector[F]:
        """The unit vector with the same direction; `UnitVector(self)`.

        Returns:
            The unit vector, in the same frame.
        """
        return UnitVector(self)


class UnitVector[F: Frame](Vector[F]):
    """A vector of length one, obtained only by normalizing a vector."""

    __slots__ = ()

    def __init__(self, direction: Vector[F]) -> None:
        """Normalize a direction: divide its components by their simplified norm.

        Args:
            direction: Any vector with a direction, in the frame of the result.

        Raises:
            ZeroVectorError: The norm simplifies to zero.
        """
        length = direction.norm().simplify()
        if length.is_zero():
            raise ZeroVectorError(direction)
        super().__init__(
            direction.x / length,
            direction.y / length,
            direction.z / length,
            direction.frame,
        )

    def _of_unit_components(self, x: Scalar, y: Scalar, z: Scalar) -> Self:
        unit: Self = object.__new__(type(self))
        unit._x, unit._y, unit._z = x, y, z
        unit._frame = self._frame
        return unit

    @override
    def __neg__(self) -> Self:
        """Reverse the arrow; the length stays one, so nothing is normalized again.

        Returns:
            The opposite unit vector, in the same frame.
        """
        return self._of_unit_components(-self._x, -self._y, -self._z)

    @override
    def simplify(self) -> Self:
        """Simplify each component; the length stays one, so nothing is normalized.

        Returns:
            The same unit vector, in the same frame, with simplified components.
        """
        return self._of_unit_components(
            self._x.simplify(), self._y.simplify(), self._z.simplify()
        )


class FreeAxis[F: Frame](UnitVector[F]):
    """A unit vector chosen freely, since any axis is valid (the identity rotation)."""

    __slots__ = ()


@final
class Point[F: Frame](_Resolved[F]):
    """A point resolved in the frame `F`: its coordinates from the origin of `F`.

    `frame.point(x, y, z)` is the usual spelling of `Point(x, y, z, frame)`.
    """

    __slots__ = ()

    @overload
    def __sub__(self, other: Point[F]) -> Vector[F]: ...

    @overload
    def __sub__(self, other: Vector[F]) -> Point[F]: ...

    def __sub__(self, other: Point[F] | Vector[F]) -> Vector[F] | Point[F]:
        """Subtract a point (giving the vector between them) or a vector (moving back).

        Returns:
            `self - other`: a vector for a point, a point for a vector.
        """
        x, y, z = self._x - other.x, self._y - other.y, self._z - other.z
        match other:
            case Point():
                return Vector(x, y, z, self._frame)
            case Vector():
                return Point(x, y, z, self._frame)

    def __add__(self, other: Vector[F]) -> Point[F]:
        """Displace the point by a vector of the same frame.

        Returns:
            The displaced point, in the same frame.
        """
        return Point(
            self._x + other.x, self._y + other.y, self._z + other.z, self._frame
        )
