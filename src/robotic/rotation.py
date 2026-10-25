"""Rotations between frames, ^A R_B, and their symbolic axis-angle solutions."""

from dataclasses import dataclass
from typing import Literal, Self, final, overload, override

from robotic.geometry import Frame, FreeAxis, UnitVector, Vector
from robotic.symbolic import Layout, Scalar, ScalarLike, Symbolic

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


class ValidityCondition(Symbolic):
    """Where a symbolic result is valid: wherever `expression` is not zero.

    Compiled, it is the flag `|expression| > tolerance`, so values within the tolerance
    of a singular configuration are reported invalid.
    """

    __slots__ = ("_expression", "_tolerance")

    def __init__(self, expression: Scalar, tolerance: float) -> None:
        """State the condition `expression != 0`.

        Args:
            expression: The scalar that vanishes at singular configurations.
            tolerance: The margin applied when compiled.
        """
        self._expression = expression
        self._tolerance = tolerance

    @property
    def expression(self) -> Scalar:
        """The scalar that vanishes at singular configurations."""
        return self._expression

    @property
    def tolerance(self) -> float:
        """The margin applied when compiled."""
        return self._tolerance

    @override
    def simplify(self) -> Self:
        """Simplify the expression.

        Returns:
            The same condition with a simplified expression.
        """
        return type(self)(self._expression.simplify(), self._tolerance)

    @override
    def latex(self) -> str:
        """Render as the LaTeX of `expression != 0`.

        Returns:
            The LaTeX source.
        """
        return rf"{self._expression.latex()} \neq 0"

    @override
    def _layout(self) -> Layout:
        return Layout(
            label="ValidityCondition",
            entries=(self._expression,),
            shape=(),
            tolerance=self._tolerance,
        )

    @override
    def __repr__(self) -> str:
        """Print the condition.

        Returns:
            E.g. `ValidityCondition(Abs(sin(q)) != 0)`.
        """
        return f"ValidityCondition({self._expression} != 0)"


@dataclass(frozen=True)
class AxisAngle[F: Frame]:
    """One axis-angle representation: a rotation by `angle` about `axis`.

    Attributes:
        axis: The unit axis; it has the same components in both frames of the rotation.
        angle: The angle in radians, in (-pi, pi].
    """

    axis: UnitVector[F]
    angle: Scalar


class AxisAngleSolutions[F: Frame](Symbolic):
    """The two axis-angle solutions of a rotation and where they are valid.

    Course convention: regular case (r, theta) and (-r, -theta); half turn (r, pi) and
    (-r, pi); identity (a, 0) and (-a, 0) with a free axis a.
    """

    __slots__ = ("_first", "_second", "_valid_when")

    def __init__(
        self,
        first: AxisAngle[F],
        second: AxisAngle[F],
        valid_when: ValidityCondition,
    ) -> None:
        """Pair the two solutions with their validity condition.

        Args:
            first: The solution with the angle in [0, pi].
            second: The solution with the opposite axis.
            valid_when: Where both solutions are valid.
        """
        self._first = first
        self._second = second
        self._valid_when = valid_when

    @property
    def first(self) -> AxisAngle[F]:
        """The solution with the angle in [0, pi]."""
        return self._first

    @property
    def second(self) -> AxisAngle[F]:
        """The solution with the opposite axis."""
        return self._second

    @property
    def valid_when(self) -> ValidityCondition:
        """Where both solutions are valid."""
        return self._valid_when

    @override
    def simplify(self) -> Self:
        """Simplify both solutions and the condition.

        Returns:
            The same solutions, simplified.
        """
        return type(self)(
            AxisAngle(self._first.axis.simplify(), self._first.angle.simplify()),
            AxisAngle(self._second.axis.simplify(), self._second.angle.simplify()),
            self._valid_when.simplify(),
        )

    @override
    def latex(self) -> str:
        """Render both solutions and the condition.

        Returns:
            The LaTeX source.
        """
        first, second = self._first, self._second
        return (
            rf"r = {first.axis.latex()},\ \theta = {first.angle.latex()} \quad "
            rf"r = {second.axis.latex()},\ \theta = {second.angle.latex()} \quad "
            rf"\text{{valid when }} {self._valid_when.latex()}"
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
            condition=self._valid_when,
        )

    @override
    def __repr__(self) -> str:
        """Print both solutions and the condition.

        Returns:
            The solutions as (axis, angle) pairs, then the condition.
        """
        first, second = self._first, self._second
        return (
            f"AxisAngleSolutions(({first.axis!r}, {first.angle}), "
            f"({second.axis!r}, {second.angle}), {self._valid_when!r})"
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
        self, axis: UnitVector[Target], angle: ScalarLike, *, source: type[Source]
    ) -> None:
        """The rotation by `angle` about `axis` (Rodrigues); spelled `about_axis`.

        Args:
            axis: The unit axis, resolved in the target frame, which it fixes.
            angle: The angle in radians, positive counterclockwise about the axis.
            source: The frame B of ^A R_B.
        """
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

    @classmethod
    def _of_matrix[A: Frame, B: Frame](
        cls, target: type[A], source: type[B], matrix: _Matrix3
    ) -> Rotation[A, B]:
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
        cls, axis: UnitVector[A], angle: ScalarLike, *, source: type[B]
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
        """
        return self._matrix.entry(row, column)

    @overload
    def __matmul__[Inner: Frame](
        self, other: Rotation[Source, Inner]
    ) -> Rotation[Target, Inner]: ...

    @overload
    def __matmul__(self, other: Vector[Source]) -> Vector[Target]: ...

    def __matmul__[Inner: Frame](
        self, other: Rotation[Source, Inner] | Vector[Source]
    ) -> Rotation[Target, Inner] | Vector[Target]:
        """Compose with ^B R_C, or resolve in A a vector given in B.

        Returns:
            ^A R_C for a rotation; ^A v = ^A R_B ^B v for a vector.
        """
        match other:
            case Rotation():
                return self._of_matrix(
                    self._target, other.source, self._matrix @ other._matrix
                )
            case Vector():
                x, y, z = self._matrix.column_product(other.x, other.y, other.z)
                return self._target.vector(x, y, z)
            case _:
                return NotImplemented

    def inverse(self) -> Rotation[Source, Target]:
        """The inverse ^B R_A, the transpose.

        Returns:
            The rotation with frames swapped.
        """
        return self._of_matrix(self._source, self._target, self._matrix.transpose())

    def to_axis_angle(
        self, tolerance: float = 1e-9, free_axis: UnitVector[Target] | None = None
    ) -> AxisAngleSolutions[Target]:
        """The two axis-angle solutions, symbolic, following the course convention.

        Args:
            tolerance: The margin of the validity flag when compiled.
            free_axis: The axis reported for the identity; z of A when omitted.

        Returns:
            Both solutions and where they are valid. For a symbolic rotation the
            regular-case formulas, valid where sin(theta) is not zero.

        Raises:
            ValueError: The skew part simplifies to zero but the trace proves neither
                the identity nor a half turn.
        """
        skew = self._target.vector(
            self.entry(3, 2) - self.entry(2, 3),
            self.entry(1, 3) - self.entry(3, 1),
            self.entry(2, 1) - self.entry(1, 2),
        )
        cosine = (self.entry(1, 1) + self.entry(2, 2) + self.entry(3, 3) - 1) / 2
        length = skew.norm().simplify()
        if not length.is_zero():
            sine = length / 2
            first = AxisAngle(skew.normalize(), sine.atan2(cosine))
            return AxisAngleSolutions(
                first,
                AxisAngle(-first.axis, -first.angle),
                ValidityCondition(sine, tolerance),
            )
        if (cosine - 1).is_zero():
            axis = FreeAxis(free_axis or self._target.vector(0, 0, 1))
            return AxisAngleSolutions(
                AxisAngle(axis, Scalar(0)),
                AxisAngle(-axis, Scalar(0)),
                ValidityCondition(Scalar(1), tolerance),
            )
        if (cosine + 1).is_zero():
            column = next(
                index
                for index in _INDICES
                if not (self.entry(index, index) + 1).is_zero()
            )
            x, y, z = (
                self.entry(row, column) + _IDENTITY.entry(row, column)
                for row in _INDICES
            )
            first = AxisAngle(self._target.vector(x, y, z).normalize(), Scalar.pi())
            return AxisAngleSolutions(
                first,
                AxisAngle(-first.axis, first.angle),
                ValidityCondition(self.entry(column, column) + 1, tolerance),
            )
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
