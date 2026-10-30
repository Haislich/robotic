"""Homogeneous transformations between frames, ^A T_B: elements of SE(3)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, final, overload, override

from robotic.geometry import Displacement, Frame, FrameMismatchError, Position
from robotic.symbolic import Expression, Layout, Scalar, ScalarLike, Symbolic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from robotic.rotation import Rotation

type Index4 = Literal[1, 2, 3, 4]

_INDICES: tuple[Index4, ...] = (1, 2, 3, 4)
_HOMOGENEOUS: Final = 4


@dataclass(frozen=True, eq=False)
class _Parts[K: (Scalar, Expression), Target: Frame, Source: Frame]:
    """A rotation and its translation of one kind, checked by `Rotation.homogeneous`."""

    rotation: Rotation[K, Target, Source]
    translation: Position[K, Target]


@final
class Transform[K: (Scalar, Expression), Target: Frame, Source: Frame](Symbolic[K]):
    """The transform ^A T_B = [R p; 0 1]: maps positions in B to the target A.

    It is the rotation ^A R_B and the translation, the origin of B as a position in A,
    vector p is the fourth column. On a position it is R p + t, on a displacement
    R v. Built only as `rotation.homogeneous(translation)`; constant when both are.
    """

    __slots__ = ("_rotation", "_translation")

    _rotation: Rotation[K, Target, Source]
    _translation: Position[K, Target]

    def __init__(self, parts: _Parts[K, Target, Source]) -> None:
        """Assemble the parts given by `Rotation.homogeneous`, the only spelling.

        Args:
            parts: The rotation and the translation, already checked.

        Raises:
            TypeError: Called with anything but the parts `homogeneous` builds.
        """
        if not isinstance(parts, _Parts):
            message = (
                "a transform is the homogeneous matrix of a rotation: write "
                "rotation.homogeneous(translation)"
            )
            raise TypeError(message)
        self._rotation = parts.rotation
        self._translation = parts.translation

    @property
    def rotation(self) -> Rotation[K, Target, Source]:
        """The rotation ^A R_B."""
        return self._rotation

    @property
    def translation(self) -> Position[K, Target]:
        """The translation: the origin of B as a position in A."""
        return self._translation

    def entry(self, row: Index4, column: Index4) -> K:
        """The entry of the 4x4 homogeneous matrix, counted from 1.

        Returns:
            r_{row column} inside the rotation block, the translation's position vector
            in column 4, and the row (0, 0, 0, 1) at the bottom.

        Raises:
            IndexError: A row or column outside 1 to 4.
        """
        if row not in _INDICES or column not in _INDICES:
            message = (
                f"transform entries are counted from 1 to 4, not ({row}, {column})"
            )
            raise IndexError(message)
        translation = self._translation
        if row == _HOMOGENEOUS:
            zero = translation.x * 0
            return zero + 1 if column == _HOMOGENEOUS else zero
        if column == _HOMOGENEOUS:
            return (translation.x, translation.y, translation.z)[row - 1]
        return self._rotation.entry(row, column)

    def _kind(self) -> str:
        rotation = self._rotation
        return f"Transform[{rotation.target.__name__}, {rotation.source.__name__}]"

    @overload
    def __matmul__[Inner: Frame](
        self, other: Transform[K, Source, Inner]
    ) -> Transform[K, Target, Inner]: ...

    @overload
    def __matmul__[Inner: Frame](
        self,
        other: Transform[Scalar, Source, Inner] | Transform[Expression, Source, Inner],
    ) -> Transform[Expression, Target, Inner]: ...

    @overload
    def __matmul__(self, other: Position[K, Source]) -> Position[K, Target]: ...

    @overload
    def __matmul__(
        self, other: Position[Scalar, Source] | Position[Expression, Source]
    ) -> Position[Expression, Target]: ...

    @overload
    def __matmul__(self, other: Displacement[K, Source]) -> Displacement[K, Target]: ...

    @overload
    def __matmul__(
        self, other: Displacement[Scalar, Source] | Displacement[Expression, Source]
    ) -> Displacement[Expression, Target]: ...

    def __matmul__[Inner: Frame](
        self,
        other: Transform[Scalar, Source, Inner]
        | Transform[Expression, Source, Inner]
        | Position[Scalar, Source]
        | Position[Expression, Source]
        | Displacement[Scalar, Source]
        | Displacement[Expression, Source],
    ) -> (
        Transform[Scalar, Target, Inner]
        | Transform[Expression, Target, Inner]
        | Position[Scalar, Target]
        | Position[Expression, Target]
        | Displacement[Scalar, Target]
        | Displacement[Expression, Target]
    ):
        """Compose with ^B T_C, or resolve in A a position or displacement given in B.

        Returns:
            ^A T_C = (R1 R2, origin of C resolved in A) for a transform; R p + t for
            a position; R v for a displacement.

        Raises:
            FrameMismatchError: The operand is not in the source frame.
        """
        source = self._rotation.source
        match other:
            case Transform():
                if other.rotation.target is not source:
                    operation = f"compose {self._kind()} with {other._kind()}"
                    raise FrameMismatchError(operation, source, other.rotation.target)
                rotation = self._rotation @ other.rotation
                return rotation.homogeneous(self @ other.translation)
            case Position():
                if other.frame is not source:
                    kind = f"{type(other).__name__}[{other.frame.__name__}]"
                    operation = f"apply {self._kind()} to {kind}"
                    raise FrameMismatchError(operation, source, other.frame)
                return self._translation + self._rotation @ (
                    other - other.frame.origin()
                )
            case Displacement():
                if other.frame is not source:
                    kind = f"{type(other).__name__}[{other.frame.__name__}]"
                    operation = f"apply {self._kind()} to {kind}"
                    raise FrameMismatchError(operation, source, other.frame)
                return self._rotation @ other
            case _:
                return NotImplemented

    @overload
    def inverse(
        self: Transform[Scalar, Target, Source],
    ) -> Transform[Scalar, Source, Target]: ...

    @overload
    def inverse(
        self: Transform[Expression, Target, Source],
    ) -> Transform[Expression, Source, Target]: ...

    def inverse(
        self: Transform[Scalar, Target, Source] | Transform[Expression, Target, Source],
    ) -> Transform[Scalar, Source, Target] | Transform[Expression, Source, Target]:
        """The inverse ^B T_A: rotation R^T, translation the origin of A resolved in B.

        Returns:
            The transform with frames swapped.
        """
        inverse = self._rotation.inverse()
        back = inverse @ (self._rotation.target.origin() - self._translation)
        return inverse.homogeneous(inverse.target.origin() + back)

    def evaluate(
        self: Transform[Expression, Target, Source],
        values: Mapping[str, ScalarLike] | None = None,
    ) -> Transform[Scalar, Target, Source]:
        """Substitute exact values for the symbols.

        Args:
            values: The value of each symbol, by name; values for symbols that do not
                occur are allowed.

        Returns:
            The constant transform, between the same frames.
        """
        rotation = self._rotation.evaluate(values)
        return rotation.homogeneous(self._translation.evaluate(values))

    @override
    def simplify(self) -> Transform[K, Target, Source]:
        """Simplify the rotation and the translation.

        Returns:
            The same transform, between the same frames, simplified.
        """
        return self._rotation.simplify().homogeneous(self._translation.simplify())

    @override
    def latex(self) -> str:
        """Render as ^A T_B = [4x4 matrix].

        Returns:
            The LaTeX source.
        """
        rows = ((self.entry(row, column) for column in _INDICES) for row in _INDICES)
        frames = (
            rf"{{}}^{{{self._rotation.target.notation}}}"
            rf"T_{{{self._rotation.source.notation}}}"
        )
        return f"{frames} = {self._bmatrix(rows)}"

    @override
    def _layout(self) -> Layout:
        return Layout(
            label=self._kind(),
            entries=tuple(
                self.entry(row, column) for row in _INDICES for column in _INDICES
            ),
            shape=(4, 4),
        )

    @override
    def __repr__(self) -> str:
        """Print the rotation and the translation.

        Returns:
            E.g. `Transform(Rotation[Base, Link1](...), Position[Base](0, 0, d1))`.
        """
        return f"Transform({self._rotation!r}, {self._translation!r})"
