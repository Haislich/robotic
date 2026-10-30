"""Rotations between frames, ^A R_B, and their symbolic axis-angle solutions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, TypeIs, final, overload, override

from robotic.geometry import (
    Direction,
    Displacement,
    Frame,
    FrameMismatchError,
    FreeAxis,
    Position,
)
from robotic.symbolic import (
    Expression,
    Layout,
    Scalar,
    ScalarLike,
    Symbolic,
    _Constants,
    _Expressions,
    _Incomparable,
)
from robotic.transform import Transform, _Parts

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

type Index3 = Literal[1, 2, 3]

_INDICES: tuple[Index3, ...] = (1, 2, 3)


class _Matrix3[K: (Scalar, Expression)]:
    """Nine scalars of one kind, row-major: a rotation's storage, without frames."""

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[K, ...]) -> None:
        self._entries = entries

    @property
    def entries(self) -> tuple[K, ...]:
        return self._entries

    def entry(self, row: Index3, column: Index3) -> K:
        return self._entries[3 * (row - 1) + column - 1]

    def transpose(self) -> _Matrix3[K]:
        return _Matrix3(
            tuple(self.entry(column, row) for row in _INDICES for column in _INDICES)
        )

    def simplify(self) -> _Matrix3[K]:
        return _Matrix3(tuple(entry.simplify() for entry in self._entries))

    def rows(self) -> str:
        return ", ".join(
            "[" + ", ".join(str(self.entry(row, column)) for column in _INDICES) + "]"
            for row in _INDICES
        )


_IDENTITY = _Matrix3(tuple(Scalar(entry) for entry in (1, 0, 0, 0, 1, 0, 0, 0, 1)))

type _AnyRotation[A: Frame, B: Frame] = (
    Rotation[Scalar, A, B] | Rotation[Expression, A, B]
)
type _AnyPosition[A: Frame] = Position[Scalar, A] | Position[Expression, A]


@dataclass(frozen=True, eq=False)
class AxisAngle[K: (Scalar, Expression), F: Frame](_Incomparable):
    """One axis-angle representation: a rotation by `angle` about `axis`.

    Attributes:
        axis: The unit axis; it has the same components in both frames of the rotation.
        angle: The angle in radians, in (-pi, pi].
    """

    axis: Direction[K, F]
    angle: K


class AxisAngleSolutions[K: (Scalar, Expression), F: Frame](Symbolic[K]):
    """The two axis-angle solutions of a rotation.

    Course convention: regular case (r, theta) and (-r, -theta); half turn (r, pi) and
    (-r, pi); identity (a, 0) and (-a, 0) with a free axis a. For a symbolic rotation,
    the regular formulas, wherever they hold; nothing predicts where they break, and
    `evaluate` where they do raises `UndefinedValueError`.
    """

    __slots__ = ("_first", "_second")

    def __init__(self, first: AxisAngle[K, F], second: AxisAngle[K, F]) -> None:
        """Pair the two solutions.

        Args:
            first: The solution with the angle in [0, pi].
            second: The solution with the opposite axis.
        """
        self._first = first
        self._second = second

    @property
    def first(self) -> AxisAngle[K, F]:
        """The solution with the angle in [0, pi]."""
        return self._first

    @property
    def second(self) -> AxisAngle[K, F]:
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

    def evaluate(
        self: AxisAngleSolutions[Expression, F],
        values: Mapping[str, ScalarLike] | None = None,
    ) -> AxisAngleSolutions[Scalar, F]:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed.

        Returns:
            The constant solutions.
        """
        first, second = self._first, self._second
        return AxisAngleSolutions(
            AxisAngle(first.axis.evaluate(values), first.angle.evaluate(values)),
            AxisAngle(second.axis.evaluate(values), second.angle.evaluate(values)),
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
class Rotation[K: (Scalar, Expression), Target: Frame, Source: Frame](Symbolic[K]):
    """The rotation ^A R_B: maps coordinates expressed in the source B to the target A.

    Its columns are the axes of B resolved in A, so `R @ v` resolves in A an arrow
    given in B. It is constant (`Rotation[Scalar, A, B]`) when built from constants
    only, and symbolic (`Rotation[Expression, A, B]`) otherwise. No constant
    component is undefined or infinite: an operation or an `evaluate` that would give
    one raises `UndefinedValueError`.
    """

    __slots__ = ("_matrix", "_source", "_target")

    _target: type[Target]
    _source: type[Source]
    _matrix: _Matrix3[K]

    @overload
    def __new__[A: Frame, B: Frame](
        cls, axis: Direction[Scalar, A], angle: ScalarLike, *, source: type[B]
    ) -> Rotation[Scalar, A, B]: ...

    @overload
    def __new__[A: Frame, B: Frame](
        cls,
        axis: Direction[Scalar, A] | Direction[Expression, A],
        angle: ScalarLike | Expression,
        *,
        source: type[B],
    ) -> Rotation[Expression, A, B]: ...

    def __new__[A: Frame, B: Frame](
        cls,
        axis: Direction[Scalar, A] | Direction[Expression, A],
        angle: ScalarLike | Expression,
        *,
        source: type[B],
    ) -> _AnyRotation[A, B]:
        """The rotation by `angle` about `axis` (Rodrigues); spelled `about_axis`.

        Args:
            axis: The unit axis, resolved in the target frame, which it fixes.
            angle: The angle in radians, positive counterclockwise about the axis.
            source: The frame B of ^A R_B.

        Returns:
            ^A R_B; constant when the axis and the angle are.

        Raises:
            TypeError: The axis is not a direction.
        """
        if not isinstance(axis, Direction):
            message = f"a rotation's axis is a direction, not {axis!r}"
            raise TypeError(message)
        value = Rotation._value(angle)
        cosine, sine = value.cos(), value.sin()
        versine = 1 - cosine
        x, y, z = axis.x, axis.y, axis.z
        return Rotation._of_entries(
            axis.frame,
            source,
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
            ),
        )

    @staticmethod
    def _value(angle: ScalarLike | Expression) -> Scalar | Expression:
        return _Constants.or_expressions((angle,)).values[0]

    @staticmethod
    def _require_frame(value: object) -> None:
        if not (isinstance(value, type) and issubclass(value, Frame)):
            message = f"expected a frame, a subclass of Frame, not {value!r}"
            raise TypeError(message)

    @classmethod
    def _of_matrix[J: (Scalar, Expression), A: Frame, B: Frame](
        cls, target: type[A], source: type[B], matrix: _Matrix3[J]
    ) -> Rotation[J, A, B]:
        Rotation._require_frame(target)
        Rotation._require_frame(source)
        rotation: Rotation[J, A, B] = object.__new__(Rotation)
        rotation._target = target
        rotation._source = source
        rotation._matrix = matrix
        return rotation

    @classmethod
    def _of_entries[A: Frame, B: Frame](
        cls,
        target: type[A],
        source: type[B],
        entries: Iterable[ScalarLike | Expression],
    ) -> _AnyRotation[A, B]:
        match _Constants.or_expressions(entries):
            case _Constants(values):
                return cls._of_matrix(target, source, _Matrix3(values))
            case _Expressions(values):
                return cls._of_matrix(target, source, _Matrix3(values))

    @staticmethod
    def _is_constant[A: Frame, B: Frame](
        rotation: _AnyRotation[A, B],
    ) -> TypeIs[Rotation[Scalar, A, B]]:
        return isinstance(rotation.entry(1, 1), Scalar)

    @staticmethod
    def _is_constant_position[A: Frame](
        position: _AnyPosition[A],
    ) -> TypeIs[Position[Scalar, A]]:
        return isinstance(position.x, Scalar)

    def _promoted(self) -> Rotation[Expression, Target, Source]:
        entries = _Expressions.of(self._matrix.entries).values
        return self._of_matrix(self._target, self._source, _Matrix3(entries))

    @classmethod
    def identity[A: Frame, B: Frame](
        cls, *, target: type[A], source: type[B]
    ) -> Rotation[Scalar, A, B]:
        """The identity: `source` has the orientation of `target`.

        Returns:
            ^A R_B equal to the identity.
        """
        return cls._of_matrix(target, source, _IDENTITY)

    @overload
    @classmethod
    def about_x[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[Scalar, A, B]: ...

    @overload
    @classmethod
    def about_x[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> Rotation[Expression, A, B]: ...

    @classmethod
    def about_x[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> _AnyRotation[A, B]:
        """The elementary rotation about the x axis.

        Returns:
            ^A R_B = Rx(angle).
        """
        value = Rotation._value(angle)
        c, s = value.cos(), value.sin()
        return cls._of_entries(target, source, (1, 0, 0, 0, c, -s, 0, s, c))

    @overload
    @classmethod
    def about_y[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[Scalar, A, B]: ...

    @overload
    @classmethod
    def about_y[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> Rotation[Expression, A, B]: ...

    @classmethod
    def about_y[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> _AnyRotation[A, B]:
        """The elementary rotation about the y axis.

        Returns:
            ^A R_B = Ry(angle).
        """
        value = Rotation._value(angle)
        c, s = value.cos(), value.sin()
        return cls._of_entries(target, source, (c, 0, s, 0, 1, 0, -s, 0, c))

    @overload
    @classmethod
    def about_z[A: Frame, B: Frame](
        cls, angle: ScalarLike, *, target: type[A], source: type[B]
    ) -> Rotation[Scalar, A, B]: ...

    @overload
    @classmethod
    def about_z[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> Rotation[Expression, A, B]: ...

    @classmethod
    def about_z[A: Frame, B: Frame](
        cls, angle: ScalarLike | Expression, *, target: type[A], source: type[B]
    ) -> _AnyRotation[A, B]:
        """The elementary rotation about the z axis.

        Returns:
            ^A R_B = Rz(angle).
        """
        value = Rotation._value(angle)
        c, s = value.cos(), value.sin()
        return cls._of_entries(target, source, (c, -s, 0, s, c, 0, 0, 0, 1))

    @overload
    @classmethod
    def about_axis[A: Frame, B: Frame](
        cls, axis: Direction[Scalar, A], angle: ScalarLike, *, source: type[B]
    ) -> Rotation[Scalar, A, B]: ...

    @overload
    @classmethod
    def about_axis[A: Frame, B: Frame](
        cls,
        axis: Direction[Scalar, A] | Direction[Expression, A],
        angle: ScalarLike | Expression,
        *,
        source: type[B],
    ) -> Rotation[Expression, A, B]: ...

    @classmethod
    def about_axis[A: Frame, B: Frame](
        cls,
        axis: Direction[Scalar, A] | Direction[Expression, A],
        angle: ScalarLike | Expression,
        *,
        source: type[B],
    ) -> _AnyRotation[A, B]:
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

    def entry(self, row: Index3, column: Index3) -> K:
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

    @overload
    def homogeneous(
        self, translation: Position[K, Target] | None = None
    ) -> Transform[K, Target, Source]: ...

    @overload
    def homogeneous(
        self, translation: _AnyPosition[Target]
    ) -> Transform[Expression, Target, Source]: ...

    def homogeneous(
        self: _AnyRotation[Target, Source],
        translation: _AnyPosition[Target] | None = None,
    ) -> Transform[Scalar, Target, Source] | Transform[Expression, Target, Source]:
        """The homogeneous matrix [R p; 0 1] of this rotation: the transform ^A T_B.

        Args:
            translation: The origin of B as a position in A, whose position vector is
                the fourth column; the origin of A when omitted.

        Returns:
            The transform between the same frames; constant when the rotation and the
            translation are.

        Raises:
            TypeError: The translation is not a position (a displacement has no
                place).
            FrameMismatchError: The translation is not in the target frame.
        """
        if translation is None:
            return self._assembled(self._target.origin())
        if not isinstance(translation, Position):
            message = f"the translation is a position, not {translation!r}"
            raise TypeError(message)
        if translation.frame is not self._target:
            kind = f"{type(translation).__name__}[{translation.frame.__name__}]"
            operation = f"translate {self._kind()} by {kind}"
            raise FrameMismatchError(operation, self._target, translation.frame)
        return self._assembled(translation)

    def _assembled(
        self: _AnyRotation[Target, Source], translation: _AnyPosition[Target]
    ) -> Transform[Scalar, Target, Source] | Transform[Expression, Target, Source]:
        if Rotation._is_constant(self):
            if Rotation._is_constant_position(translation):
                return Transform(_Parts(self, translation))
            return Transform(_Parts(self._promoted(), translation))
        if Rotation._is_constant_position(translation):
            promoted = Position(
                Expression(translation.x),
                Expression(translation.y),
                Expression(translation.z),
                translation.frame,
            )
            return Transform(_Parts(self, promoted))
        return Transform(_Parts(self, translation))

    @overload
    def __matmul__[Inner: Frame](
        self, other: Rotation[K, Source, Inner]
    ) -> Rotation[K, Target, Inner]: ...

    @overload
    def __matmul__[Inner: Frame](
        self, other: _AnyRotation[Source, Inner]
    ) -> Rotation[Expression, Target, Inner]: ...

    @overload
    def __matmul__(self, other: Displacement[K, Source]) -> Displacement[K, Target]: ...

    @overload
    def __matmul__(
        self, other: Displacement[Scalar, Source] | Displacement[Expression, Source]
    ) -> Displacement[Expression, Target]: ...

    def __matmul__[Inner: Frame](
        self,
        other: _AnyRotation[Source, Inner]
        | Displacement[Scalar, Source]
        | Displacement[Expression, Source],
    ) -> (
        _AnyRotation[Target, Inner]
        | Displacement[Scalar, Target]
        | Displacement[Expression, Target]
    ):
        """Compose with ^B R_C, or resolve in A a displacement given in B.

        Returns:
            ^A R_C for a rotation; ^A v = ^A R_B ^B v for a displacement.

        Raises:
            FrameMismatchError: The operand is not in the source frame.
        """
        match other:
            case Rotation():
                if other.target is not self._source:
                    operation = f"compose {self._kind()} with {other._kind()}"
                    raise FrameMismatchError(operation, self._source, other.target)
                return self._of_entries(
                    self._target,
                    other.source,
                    (
                        self.entry(row, 1) * other.entry(1, column)
                        + self.entry(row, 2) * other.entry(2, column)
                        + self.entry(row, 3) * other.entry(3, column)
                        for row in _INDICES
                        for column in _INDICES
                    ),
                )
            case Displacement():
                if other.frame is not self._source:
                    kind = f"{type(other).__name__}[{other.frame.__name__}]"
                    operation = f"apply {self._kind()} to {kind}"
                    raise FrameMismatchError(operation, self._source, other.frame)
                x, y, z = (
                    self.entry(row, 1) * other.x
                    + self.entry(row, 2) * other.y
                    + self.entry(row, 3) * other.z
                    for row in _INDICES
                )
                return self._target.displacement(x, y, z)
            case _:
                return NotImplemented

    def inverse(self) -> Rotation[K, Source, Target]:
        """The inverse ^B R_A, the transpose.

        Returns:
            The rotation with frames swapped.
        """
        return self._of_matrix(self._source, self._target, self._matrix.transpose())

    def evaluate(
        self: Rotation[Expression, Target, Source],
        values: Mapping[str, ScalarLike] | None = None,
    ) -> Rotation[Scalar, Target, Source]:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed.

        Returns:
            The constant rotation, between the same frames.
        """
        entries = tuple(entry.evaluate(values) for entry in self._matrix.entries)
        return self._of_matrix(self._target, self._source, _Matrix3(entries))

    @overload
    def to_axis_angle(
        self: Rotation[Scalar, Target, Source],
        free_axis: Direction[Scalar, Target] | None = None,
    ) -> AxisAngleSolutions[Scalar, Target]: ...

    @overload
    def to_axis_angle(
        self: Rotation[Expression, Target, Source],
        free_axis: Direction[Scalar, Target] | None = None,
    ) -> AxisAngleSolutions[Expression, Target]: ...

    def to_axis_angle(
        self: _AnyRotation[Target, Source],
        free_axis: Direction[Scalar, Target] | None = None,
    ) -> AxisAngleSolutions[Scalar, Target] | AxisAngleSolutions[Expression, Target]:
        """The two axis-angle solutions, symbolic, following the course convention.

        A constant rotation is resolved on its values, at the default tolerance of
        `Scalar.is_close`: identity, half turn or regular case. A symbolic rotation gets
        the regular formulas, whatever values the symbols will take; where its skew
        part vanishes the axis is undefined. Evaluate it first for the exact cases.

        Args:
            free_axis: The axis reported for the identity, a constant direction; z of
                A when omitted.

        Returns:
            Both solutions.

        Raises:
            TypeError: The free axis is not a constant direction.
            FrameMismatchError: The free axis is not in the target frame.
        """
        if free_axis is not None:
            if not (
                isinstance(free_axis, Direction) and isinstance(free_axis.x, Scalar)
            ):
                message = f"the free axis is a constant direction, not {free_axis!r}"
                raise TypeError(message)
            if free_axis.frame is not self._target:
                kind = f"{type(free_axis).__name__}[{free_axis.frame.__name__}]"
                operation = f"take {kind} as the free axis of {self._kind()}"
                raise FrameMismatchError(operation, self._target, free_axis.frame)
        if Rotation._is_constant(self):
            return Rotation._constant_solutions(self, free_axis)
        skew, cosine, length = Rotation._skew_parts(self)
        return Rotation._regular(skew, (length / 2).atan2(cosine))

    @staticmethod
    def _skew_parts[J: (Scalar, Expression), A: Frame, B: Frame](
        rotation: Rotation[J, A, B],
    ) -> tuple[Displacement[J, A], J, J]:
        entry = rotation.entry
        skew = rotation.target.displacement(
            entry(3, 2) - entry(2, 3),
            entry(1, 3) - entry(3, 1),
            entry(2, 1) - entry(1, 2),
        )
        cosine = (entry(1, 1) + entry(2, 2) + entry(3, 3) - 1) / 2
        return skew, cosine, skew.norm()

    @staticmethod
    def _regular[J: (Scalar, Expression), A: Frame](
        skew: Displacement[J, A], angle: J
    ) -> AxisAngleSolutions[J, A]:
        first = AxisAngle(skew.normalize(), angle)
        return AxisAngleSolutions(first, AxisAngle(-first.axis, -first.angle))

    @staticmethod
    def _constant_solutions[A: Frame, B: Frame](
        rotation: Rotation[Scalar, A, B], free_axis: Direction[Scalar, A] | None
    ) -> AxisAngleSolutions[Scalar, A]:
        skew, cosine, length = Rotation._skew_parts(rotation)
        if not length.is_close(0):
            return Rotation._regular(skew, (length / 2).atan2(cosine))
        if cosine.is_close(1):
            target = rotation.target
            axis = FreeAxis(
                target.displacement(0, 0, 1) if free_axis is None else free_axis
            )
            return AxisAngleSolutions(
                AxisAngle(axis, Scalar(0)), AxisAngle(-axis, Scalar(0))
            )
        column = next(
            index
            for index in _INDICES
            if not (rotation.entry(index, index) + 1).is_close(0)
        )
        x, y, z = (
            rotation.entry(row, column) + _IDENTITY.entry(row, column)
            for row in _INDICES
        )
        first = AxisAngle(rotation.target.direction(x, y, z), Scalar.pi())
        return AxisAngleSolutions(first, AxisAngle(-first.axis, first.angle))

    @override
    def simplify(self) -> Rotation[K, Target, Source]:
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
        return Layout(label=self._kind(), entries=self._matrix.entries, shape=(3, 3))

    @override
    def __repr__(self) -> str:
        """Print the frames and the rows.

        Returns:
            E.g. `Rotation[Base, Link1]([cos(q1), -sin(q1), 0], ...)`.
        """
        return f"{self._kind()}({self._matrix.rows()})"
