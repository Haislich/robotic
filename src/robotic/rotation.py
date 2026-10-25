"""Rotations between frames, ^A R_B, and their symbolic axis-angle solutions."""

from dataclasses import dataclass
from typing import Literal, Self, overload, override

from robotic.geometry import Frame, FreeAxis, Point, UnitVector, Vector
from robotic.symbolic import Layout, Scalar, ScalarLike, Symbolic
from robotic.transform import Transform

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

    def transpose(self) -> _Matrix3:
        return _Matrix3(
            tuple(self.entry(column, row) for row in _INDICES for column in _INDICES)
        )

    def simplify(self) -> _Matrix3:
        return _Matrix3(tuple(entry.simplify() for entry in self._entries))

    def latex(self) -> str:
        rows = (
            " & ".join(self.entry(row, column).latex() for column in _INDICES)
            for row in _INDICES
        )
        return rf"\begin{{bmatrix}} {r' \\ '.join(rows)} \end{{bmatrix}}"

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
            condition=Layout(
                label="ValidityCondition",
                entries=(self._valid_when.expression,),
                shape=(),
                tolerance=self._valid_when.tolerance,
            ),
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


class Rotation[To: Frame, From: Frame](Symbolic):
    """The rotation ^A R_B: the orientation of frame B (`From`) relative to A (`To`).

    Its columns are the axes of B resolved in A, so `R @ v` resolves in A an arrow
    given in B.
    """

    __slots__ = ("_matrix", "_source", "_target")

    _target: type[To]
    _source: type[From]
    _matrix: _Matrix3

    def __init__(
        self, axis: UnitVector[To], angle: ScalarLike, *, of: type[From]
    ) -> None:
        """The rotation by `angle` about `axis` (Rodrigues); spelled `about_axis`.

        Args:
            axis: The unit axis, resolved in the reference frame.
            angle: The angle in radians, positive counterclockwise about the axis.
            of: The frame whose orientation the rotation describes.
        """
        cosine, sine = Scalar(angle).cos(), Scalar(angle).sin()
        versine = 1 - cosine
        x, y, z = axis.x, axis.y, axis.z
        self._target = axis.frame
        self._source = of
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
    def _of_matrix[T: Frame, S: Frame](
        cls, target: type[T], source: type[S], matrix: _Matrix3
    ) -> Rotation[T, S]:
        rotation: Rotation[T, S] = object.__new__(Rotation)
        rotation._target = target
        rotation._source = source
        rotation._matrix = matrix
        return rotation

    @classmethod
    def identity[T: Frame, S: Frame](
        cls, *, of: type[S], relative_to: type[T]
    ) -> Rotation[T, S]:
        """The identity: frame `of` has the orientation of `relative_to`.

        Returns:
            The rotation ^relative_to R_of equal to the identity.
        """
        return cls._of_matrix(relative_to, of, _IDENTITY)

    @classmethod
    def about_x[T: Frame, S: Frame](
        cls, angle: ScalarLike, *, of: type[S], relative_to: type[T]
    ) -> Rotation[T, S]:
        """The elementary rotation about the x axis.

        Returns:
            ^relative_to R_of = Rx(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(relative_to, of, _Matrix3((1, 0, 0, 0, c, -s, 0, s, c)))

    @classmethod
    def about_y[T: Frame, S: Frame](
        cls, angle: ScalarLike, *, of: type[S], relative_to: type[T]
    ) -> Rotation[T, S]:
        """The elementary rotation about the y axis.

        Returns:
            ^relative_to R_of = Ry(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(relative_to, of, _Matrix3((c, 0, s, 0, 1, 0, -s, 0, c)))

    @classmethod
    def about_z[T: Frame, S: Frame](
        cls, angle: ScalarLike, *, of: type[S], relative_to: type[T]
    ) -> Rotation[T, S]:
        """The elementary rotation about the z axis.

        Returns:
            ^relative_to R_of = Rz(angle).
        """
        c, s = Scalar(angle).cos(), Scalar(angle).sin()
        return cls._of_matrix(relative_to, of, _Matrix3((c, -s, 0, s, c, 0, 0, 0, 1)))

    @classmethod
    def about_axis[T: Frame, S: Frame](
        cls, axis: UnitVector[T], angle: ScalarLike, *, of: type[S]
    ) -> Rotation[T, S]:
        """The rotation by `angle` about `axis`; the axis's frame is the reference.

        Returns:
            ^axis.frame R_of, by Rodrigues' formula.
        """
        return Rotation(axis, angle, of=of)

    @property
    def relative_to(self) -> type[To]:
        """The reference frame A of ^A R_B."""
        return self._target

    @property
    def of(self) -> type[From]:
        """The frame B whose orientation ^A R_B describes."""
        return self._source

    def entry(self, row: Index3, column: Index3) -> Scalar:
        """The entry r_{row column}, counted from 1 as the course writes r13.

        Returns:
            The entry.
        """
        return self._matrix.entry(row, column)

    @overload
    def __matmul__[Inner: Frame](
        self, other: Rotation[From, Inner]
    ) -> Rotation[To, Inner]: ...

    @overload
    def __matmul__(self, other: Vector[From]) -> Vector[To]: ...

    def __matmul__[Inner: Frame](
        self, other: Rotation[From, Inner] | Vector[From]
    ) -> Rotation[To, Inner] | Vector[To]:
        """Compose with ^B R_C, or resolve in A a vector given in B.

        Returns:
            ^A R_C for a rotation; ^A v = ^A R_B ^B v for a vector.
        """
        match other:
            case Rotation():
                return self._of_matrix(
                    self._target, other.of, self._matrix @ other._matrix
                )
            case Vector():
                return Vector(
                    self._target,
                    *(
                        self.entry(row, 1) * other.x
                        + self.entry(row, 2) * other.y
                        + self.entry(row, 3) * other.z
                        for row in _INDICES
                    ),
                )

    def inverse(self) -> Rotation[From, To]:
        """The inverse ^B R_A, the transpose.

        Returns:
            The rotation with frames swapped.
        """
        return self._of_matrix(self._source, self._target, self._matrix.transpose())

    def with_origin(self, origin: Point[To]) -> Transform[To, From]:
        """The pose of B in A: this orientation with the origin of B at `origin`.

        Args:
            origin: The origin of B, resolved in A.

        Returns:
            The transform ^A T_B.
        """
        return Transform(self, origin.x, origin.y, origin.z)

    def to_axis_angle(
        self, tolerance: float = 1e-9, free_axis: UnitVector[To] | None = None
    ) -> AxisAngleSolutions[To]:
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
        skew = Vector(
            self._target,
            self.entry(3, 2) - self.entry(2, 3),
            self.entry(1, 3) - self.entry(3, 1),
            self.entry(2, 1) - self.entry(1, 2),
        )
        cosine = (self.entry(1, 1) + self.entry(2, 2) + self.entry(3, 3) - 1) / 2
        always = ValidityCondition(Scalar(1), tolerance)
        if not skew.dot(skew).is_zero():
            sine = skew.norm().simplify() / 2
            angle = sine.atan2(cosine)
            return AxisAngleSolutions(
                AxisAngle(UnitVector(skew), angle),
                AxisAngle(UnitVector(-skew), -angle),
                ValidityCondition(sine, tolerance),
            )
        if (cosine - 1).is_zero():
            axis = FreeAxis(free_axis or self._target.vector(0, 0, 1))
            return AxisAngleSolutions(
                AxisAngle(axis, Scalar(0)),
                AxisAngle(FreeAxis(-axis), Scalar(0)),
                always,
            )
        if (cosine + 1).is_zero():
            column = next(
                index
                for index in _INDICES
                if not (self.entry(index, index) + 1).is_zero()
            )
            half = Vector(
                self._target,
                *(
                    self.entry(row, column) + _IDENTITY.entry(row, column)
                    for row in _INDICES
                ),
            )
            return AxisAngleSolutions(
                AxisAngle(UnitVector(half), Scalar.pi()),
                AxisAngle(UnitVector(-half), Scalar.pi()),
                always,
            )
        message = "cannot decide between the identity and a half turn"
        raise ValueError(message)

    @override
    def simplify(self) -> Self:
        """Simplify every entry.

        Returns:
            The same rotation, between the same frames, with simplified entries.
        """
        rotation: Self = object.__new__(type(self))
        rotation._target = self._target
        rotation._source = self._source
        rotation._matrix = self._matrix.simplify()
        return rotation

    @override
    def latex(self) -> str:
        """Render as ^A R_B = [matrix].

        Returns:
            The LaTeX source.
        """
        return (
            rf"{{}}^{{{self._target.latex}}}R_{{{self._source.latex}}} = "
            f"{self._matrix.latex()}"
        )

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
