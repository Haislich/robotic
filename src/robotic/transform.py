"""Homogeneous transformations between frames, ^A T_B: elements of SE(3)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, final, overload, override

from robotic.geometry import Displacement, Frame, Position
from robotic.symbolic import Layout, Scalar, Symbolic

if TYPE_CHECKING:
    from robotic.rotation import Rotation

type Index4 = Literal[1, 2, 3, 4]

_INDICES: tuple[Index4, ...] = (1, 2, 3, 4)
_HOMOGENEOUS: Final = 4


@dataclass(frozen=True, eq=False)
class _Parts[Target: Frame, Source: Frame]:
    """A rotation and its translation, checked by `Rotation.homogeneous`."""

    rotation: Rotation[Target, Source]
    translation: Position[Target]


@final
class Transform[Target: Frame, Source: Frame](Symbolic):
    """The transform ^A T_B = [R p; 0 1]: maps positions in B to the target A.

    It is the rotation ^A R_B and the translation, the origin of B as a position in A,
    vector p is the fourth column. On a position it is R p + t, on a displacement
    R v. Built only as `rotation.homogeneous(translation)`.
    """

    __slots__ = ("_rotation", "_translation")

    _rotation: Rotation[Target, Source]
    _translation: Position[Target]

    def __init__(self, parts: _Parts[Target, Source]) -> None:
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
    def rotation(self) -> Rotation[Target, Source]:
        """The rotation ^A R_B."""
        return self._rotation

    @property
    def translation(self) -> Position[Target]:
        """The translation: the origin of B as a position in A."""
        return self._translation

    def entry(self, row: Index4, column: Index4) -> Scalar:
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
        if row == _HOMOGENEOUS:
            return Scalar(1 if column == _HOMOGENEOUS else 0)
        if column == _HOMOGENEOUS:
            translation = self._translation
            return (translation.x, translation.y, translation.z)[row - 1]
        return self._rotation.entry(row, column)

    def _kind(self) -> str:
        rotation = self._rotation
        return f"Transform[{rotation.target.__name__}, {rotation.source.__name__}]"

    @overload
    def __matmul__[Inner: Frame](
        self, other: Transform[Source, Inner]
    ) -> Transform[Target, Inner]: ...

    @overload
    def __matmul__(self, other: Position[Source]) -> Position[Target]: ...

    @overload
    def __matmul__(self, other: Displacement[Source]) -> Displacement[Target]: ...

    def __matmul__[Inner: Frame](
        self, other: Transform[Source, Inner] | Position[Source] | Displacement[Source]
    ) -> Transform[Target, Inner] | Position[Target] | Displacement[Target]:
        """Compose with ^B T_C, or resolve in A a position or displacement given in B.

        Returns:
            ^A T_C = (R1 R2, origin of C resolved in A) for a transform; R p + t for
            a position; R v for a displacement.
        """
        match other:
            case Transform():
                return Transform(
                    _Parts(self._rotation @ other.rotation, self @ other.translation)
                )
            case Position():
                return self._translation + self._rotation @ (
                    other - other.frame.origin()
                )
            case Displacement():
                return self._rotation @ other
            case _:
                return NotImplemented

    def inverse(self) -> Transform[Source, Target]:
        """The inverse ^B T_A: rotation R^T, translation the origin of A resolved in B.

        Returns:
            The transform with frames swapped.
        """
        inverse = self._rotation.inverse()
        back = inverse @ (self._rotation.target.origin() - self._translation)
        return Transform(_Parts(inverse, inverse.target.origin() + back))

    @override
    def simplify(self) -> Transform[Target, Source]:
        """Simplify the rotation and the translation.

        Returns:
            The same transform, between the same frames, simplified.
        """
        return Transform(
            _Parts(self._rotation.simplify(), self._translation.simplify())
        )

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
