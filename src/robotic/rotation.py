"""Rotations between frames, ^A R_B, and their symbolic axis-angle solutions."""

from dataclasses import dataclass
from typing import Literal, Self, final, overload, override

from robotic.geometry import (
    Direction,
    Displacement,
    Frame,
    FrameMismatchError,
    FreeAxis,
    Position,
)
from robotic.symbolic import (
    Layout,
    Scalar,
    ScalarLike,
    Symbolic,
    _has_symbols,
    _Incomparable,
    _is_zero,
)
from robotic.transform import Transform, _Parts

type Index3 = Literal[1, 2, 3]

_INDICES: tuple[Index3, ...] = (1, 2, 3)


class _Matrix3:
    """Nine scalars, row-major: the storage of a rotation, without frames."""

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[ScalarLike, ...]) -> None:
        self._entries = tuple(Scalar(entry) for entry in entries)

    @property
    def entries(self) -> tuple[Scalar, ...]:
        return self._entries

    def entry(self, row: Index3, column: Index3) -> Scalar:
        return self._entries[3 * (row - 1) + column - 1]

    def __matmul__(self, other: _Matrix3) -> _Matrix3:
        return _Matrix3(
            tuple(
                self.entry(row, 1) * other.entry(1, column)
                + self.entry(row, 2) * other.entry(2, column)
                + self.entry(row, 3) * other.entry(3, column)
                for row in _INDICES
                for column in _INDICES
            )
        )

    def column_product(
        self, x: Scalar, y: Scalar, z: Scalar
    ) -> tuple[Scalar, Scalar, Scalar]:
        first, second, third = (
            self.entry(row, 1) * x + self.entry(row, 2) * y + self.entry(row, 3) * z
            for row in _INDICES
        )
        return first, second, third

    def transpose(self) -> _Matrix3:
        return _Matrix3(
            tuple(self.entry(column, row) for row in _INDICES for column in _INDICES)
        )

    def simplify(self) -> _Matrix3:
        return _Matrix3(tuple(entry.simplify() for entry in self._entries))

    def rows(self) -> str:
        return ", ".join(
            "[" + ", ".join(str(self.entry(row, column)) for column in _INDICES) + "]"
            for row in _INDICES
        )


_IDENTITY = _Matrix3((1, 0, 0, 0, 1, 0, 0, 0, 1))


@dataclass(frozen=True, eq=False)
class AxisAngle[F: Frame](_Incomparable):
    """One axis-angle representation: a rotation by `angle` about `axis`.

    Attributes:
        axis: The unit axis; it has the same components in both frames of the rotation.
        angle: The angle in radians, in (-pi, pi].
    """

    axis: Direction[F]
    angle: Scalar


class AxisAngleSolutions[F: Frame](Symbolic):
    """The two axis-angle solutions of a rotation.

    Course convention: regular case (r, theta) and (-r, -theta); half turn (r, pi) and
    (-r, pi); identity (a, 0) and (-a, 0) with a free axis a. For a rotation with a
    symbol, the regular formulas, wherever they hold; nothing predicts where they
    break.
    """

    __slots__ = ("_first", "_second")

    def __init__(self, first: AxisAngle[F], second: AxisAngle[F]) -> None:
        """Pair the two solutions.

        Args:
            first: The solution with the angle in [0, pi].
            second: The solution with the opposite axis.
        """
        self._first = first
        self._second = second

    @property
    def first(self) -> AxisAngle[F]:
        """The solution with the angle in [0, pi]."""
        return self._first

    @property
    def second(self) -> AxisAngle[F]:
        """The solution with the opposite axis."""
        return self._second

    @override
    def simplify(self) -> Self:
        """Simplify both solutions.

        Returns:
            The same solutions, simplified.
        """
        return type(self)(
            AxisAngle(self._first.axis.simplify(), self._first.angle.simplify()),
            AxisAngle(self._second.axis.simplify(), self._second.angle.simplify()),
        )

    @override
    def latex(self) -> str:
        """Render both solutions.

        Returns:
            The LaTeX source.
        """
        first, second = self._first, self._second
        return (
            rf"r = {first.axis.latex()},\ \theta = {first.angle.latex()} \quad "
            rf"r = {second.axis.latex()},\ \theta = {second.angle.latex()}"
        )

    @override
    def _layout(self) -> Layout:
        first, second = self._first, self._second
        return Layout(
            label=f"AxisAngleSolutions[{first.axis.frame.__name__}]",
            entries=(
                first.axis.x,
                first.axis.y,
                first.axis.z,
                first.angle,
                second.axis.x,
                second.axis.y,
                second.axis.z,
                second.angle,
            ),
            shape=(2, 4),
        )

    @override
    def __repr__(self) -> str:
        """Print both solutions.

        Returns:
            The solutions as (axis, angle) pairs.
        """
        first, second = self._first, self._second
        return (
            f"AxisAngleSolutions(({first.axis!r}, {first.angle}), "
            f"({second.axis!r}, {second.angle}))"
        )


@final
class Rotation[Target: Frame, Source: Frame](Symbolic):
    """The rotation ^A R_B: maps coordinates expressed in the source B to the target A.

    Its columns are the axes of B resolved in A, so `R @ v` resolves in A an arrow
    given in B.
    """

    __slots__ = ("_matrix", "_source", "_target")

    _target: type[Target]
    _source: type[Source]
    _matrix: _Matrix3

    def __init__(
        self, axis: Direction[Target], angle: ScalarLike, *, source: type[Source]
    ) -> None:
        """The rotation by `angle` about `axis` (Rodrigues); spelled `about_axis`.

        Args:
            axis: The unit axis, resolved in the target frame, which it fixes.
            angle: The angle in radians, positive counterclockwise about the axis.
            source: The frame B of ^A R_B.

        Raises:
            TypeError: The axis is not a direction.
        """
        if not isinstance(axis, Direction):
            message = f"a rotation's axis is a direction, not {axis!r}"
            raise TypeError(message)
        Rotation._require_frame(source)
        cosine, sine = Scalar(angle).cos(), Scalar(angle).sin()
        versine = 1 - cosine
        x, y, z = axis.x, axis.y, axis.z
        self._target = axis.frame
        self._source = source
        self._matrix = _Matrix3(
            (
                cosine + x * x * versine,
                x * y * versine - z * sine,
                x * z * versine + y * sine,
                x * y * versine + z * sine,
                cosine + y * y * versine,
                y * z * versine - x * sine,
                x * z * versine - y * sine,
                y * z * versine + x * sine,
                cosine + z * z * versine,
            )
        )

    @staticmethod
    def _require_frame(value: object) -> None:
        if not (isinstance(value, type) and issubclass(value, Frame)):
            message = f"expected a frame, a subclass of Frame, not {value!r}"
            raise TypeError(message)

    @staticmethod
    def _same_frame(operation: str, expected: type[Frame], found: type[Frame]) -> None:
        if found is not expected:
            raise FrameMismatchError(operation, expected, found)

    @classmethod
    def _of_matrix[A: Frame, B: Frame](
        cls, target: type[A], source: type[B], matrix: _Matrix3
    ) -> Rotation[A, B]:
        Rotation._require_frame(target)
        Rotation._require_frame(source)
        rotation: Rotation[A, B] = object.__new__(Rotation)
        rotation._target = target
        rotation._source = source
        rotation._matrix = matrix
        return rotation

    @classmethod
    def identity[A: Frame, B: Frame](
        cls, *, target: type[A], source: type[B]
    ) -> Rotation[A, B]:
        """The identity: `source` has the orientation of `target`.

        Returns:
            ^A R_B equal to the identity.
        """
        return cls._of_matrix(target, source, _IDENTITY)

    @classmethod
    def about_x[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[A, B]:
        """The elementary rotation about the x axis.

        Returns:
            ^A R_B = Rx(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(target, source, _Matrix3((1, 0, 0, 0, c, -s, 0, s, c)))

    @classmethod
    def about_y[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[A, B]:
        """The elementary rotation about the y axis.

        Returns:
            ^A R_B = Ry(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(target, source, _Matrix3((c, 0, s, 0, 1, 0, -s, 0, c)))

    @classmethod
    def about_z[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[A, B]:
        """The elementary rotation about the z axis.

        Returns:
            ^A R_B = Rz(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(target, source, _Matrix3((c, -s, 0, s, c, 0, 0, 0, 1)))

    @classmethod
    def about_axis[A: Frame, B: Frame](
        cls, axis: Direction[A], angle: ScalarLike, *, source: type[B]
    ) -> Rotation[A, B]:
        """The rotation by `angle` about `axis`; the axis's frame is the target.

        Returns:
            ^A R_B, by Rodrigues' formula.
        """
        return Rotation(axis, angle, source=source)

    @property
    def target(self) -> type[Target]:
        """The target frame A of ^A R_B."""
        return self._target

    @property
    def source(self) -> type[Source]:
        """The source frame B of ^A R_B."""
        return self._source

    def entry(self, row: Index3, column: Index3) -> Scalar:
        """The entry r_{row column}, counted from 1 as the course writes r13.

        Returns:
            The entry.

        Raises:
            IndexError: A row or column outside 1 to 3.
        """
        if row not in _INDICES or column not in _INDICES:
            message = f"rotation entries are counted from 1 to 3, not ({row}, {column})"
            raise IndexError(message)
        return self._matrix.entry(row, column)

    def _kind(self) -> str:
        return f"Rotation[{self._target.__name__}, {self._source.__name__}]"

    def homogeneous(
        self, translation: Position[Target] | None = None
    ) -> Transform[Target, Source]:
        """The homogeneous matrix [R p; 0 1] of this rotation: the transform ^A T_B.

        Args:
            translation: The origin of B as a position in A, whose position vector is
                the fourth column; the origin of A when omitted.

        Returns:
            The transform between the same frames.

        Raises:
            TypeError: The translation is not a position (a displacement has no
                place).
        """
        if translation is None:
            return Transform(_Parts(self, self._target.origin()))
        if not isinstance(translation, Position):
            message = f"the translation is a position, not {translation!r}"
            raise TypeError(message)
        self._same_frame(
            f"translate {self._kind()} by {translation!r}",
            self._target,
            translation.frame,
        )
        return Transform(_Parts(self, translation))

    @overload
    def __matmul__[Inner: Frame](
        self, other: Rotation[Source, Inner]
    ) -> Rotation[Target, Inner]: ...

    @overload
    def __matmul__(self, other: Displacement[Source]) -> Displacement[Target]: ...

    def __matmul__[Inner: Frame](
        self, other: Rotation[Source, Inner] | Displacement[Source]
    ) -> Rotation[Target, Inner] | Displacement[Target]:
        """Compose with ^B R_C, or resolve in A a displacement given in B.

        Returns:
            ^A R_C for a rotation; ^A v = ^A R_B ^B v for a displacement.
        """
        match other:
            case Rotation():
                self._same_frame(
                    f"compose {self._kind()} with {other._kind()}",
                    self._source,
                    other.target,
                )
                return self._of_matrix(
                    self._target, other.source, self._matrix @ other._matrix
                )
            case Displacement():
                self._same_frame(
                    f"apply {self._kind()} to {other!r}", self._source, other.frame
                )
                x, y, z = self._matrix.column_product(other.x, other.y, other.z)
                return self._target.displacement(x, y, z)
            case _:
                return NotImplemented

    def inverse(self) -> Rotation[Source, Target]:
        """The inverse ^B R_A, the transpose.

        Returns:
            The rotation with frames swapped.
        """
        return self._of_matrix(self._source, self._target, self._matrix.transpose())

    def to_axis_angle(
        self, free_axis: Direction[Target] | None = None
    ) -> AxisAngleSolutions[Target]:
        """The two axis-angle solutions, symbolic, following the course convention.

        A rotation whose simplified entries contain no symbol is resolved exactly:
        identity, half turn or regular case. A rotation with a symbol gets the regular
        formulas, whatever values the symbols will take; if its skew part is
        identically zero (a half turn about a symbolic axis), normalizing it raises
        `ZeroVectorError`.

        Args:
            free_axis: The axis reported for the identity; z of A when omitted.

        Returns:
            Both solutions.

        Raises:
            ValueError: An exact rotation whose skew part is zero but whose trace
                proves neither the identity nor a half turn.
        """
        rotation = self.simplify()
        skew = self._target.displacement(
            rotation.entry(3, 2) - rotation.entry(2, 3),
            rotation.entry(1, 3) - rotation.entry(3, 1),
            rotation.entry(2, 1) - rotation.entry(1, 2),
        )
        cosine = (
            rotation.entry(1, 1) + rotation.entry(2, 2) + rotation.entry(3, 3) - 1
        ) / 2
        length = skew.norm().simplify()
        symbolic = any(
            _has_symbols(rotation.entry(row, column))
            for row in _INDICES
            for column in _INDICES
        )
        if symbolic or not _is_zero(length):
            first = AxisAngle(skew.normalize(), (length / 2).atan2(cosine))
            return AxisAngleSolutions(first, AxisAngle(-first.axis, -first.angle))
        if _is_zero(cosine - 1):
            axis = FreeAxis(
                self._target.displacement(0, 0, 1) if free_axis is None else free_axis
            )
            return AxisAngleSolutions(
                AxisAngle(axis, Scalar(0)), AxisAngle(-axis, Scalar(0))
            )
        if _is_zero(cosine + 1):
            column = next(
                index
                for index in _INDICES
                if not _is_zero(rotation.entry(index, index) + 1)
            )
            x, y, z = (
                rotation.entry(row, column) + _IDENTITY.entry(row, column)
                for row in _INDICES
            )
            first = AxisAngle(self._target.direction(x, y, z), Scalar.pi())
            return AxisAngleSolutions(first, AxisAngle(-first.axis, first.angle))
        message = "cannot decide between the identity and a half turn"
        raise ValueError(message)

    @override
    def simplify(self) -> Rotation[Target, Source]:
        """Simplify every entry.

        Returns:
            The same rotation, between the same frames, with simplified entries.
        """
        return self._of_matrix(self._target, self._source, self._matrix.simplify())

    @override
    def latex(self) -> str:
        """Render as ^A R_B = [matrix].

        Returns:
            The LaTeX source.
        """
        rows = ((self.entry(row, column) for column in _INDICES) for row in _INDICES)
        frames = rf"{{}}^{{{self._target.notation}}}R_{{{self._source.notation}}}"
        return f"{frames} = {self._bmatrix(rows)}"

    @override
    def _layout(self) -> Layout:
        return Layout(
            label=f"Rotation[{self._target.__name__}, {self._source.__name__}]",
            entries=self._matrix.entries,
            shape=(3, 3),
        )

    @override
    def __repr__(self) -> str:
        """Print the frames and the rows.

        Returns:
            E.g. `Rotation[Base, Link1]([cos(q1), -sin(q1), 0], ...)`.
        """
        frames = f"{self._target.__name__}, {self._source.__name__}"
        return f"Rotation[{frames}]({self._matrix.rows()})"
