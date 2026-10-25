"""Homogeneous transformations between frames, ^A T_B."""

from typing import TYPE_CHECKING, Final, Literal, Self, overload, override

from robotic.geometry import Frame, Point, Vector
from robotic.symbolic import Layout, Scalar, ScalarLike, Symbolic

if TYPE_CHECKING:
    from robotic.rotation import Rotation

type Index4 = Literal[1, 2, 3, 4]

_INDICES: tuple[Index4, ...] = (1, 2, 3, 4)
_HOMOGENEOUS: Final = 4


class Transform[To: Frame, From: Frame](Symbolic):
    """The transform ^A T_B: the pose of frame B (`From`) relative to A (`To`).

    It is the rotation ^A R_B and the origin of B resolved in A; on a point it is
    R p + t, on a vector R v.
    """

    __slots__ = ("_origin", "_rotation")

    def __init__(
        self, rotation: Rotation[To, From], x: ScalarLike, y: ScalarLike, z: ScalarLike
    ) -> None:
        """The transform [R | t] with t written as numbers read in A.

        Args:
            rotation: The orientation ^A R_B; it fixes both frames.
            x: The x coordinate of the origin of B, read in A.
            y: The y coordinate of the origin of B, read in A.
            z: The z coordinate of the origin of B, read in A.
        """
        self._rotation = rotation
        self._origin = Point(rotation.relative_to, x, y, z)

    @property
    def rotation(self) -> Rotation[To, From]:
        """The orientation ^A R_B."""
        return self._rotation

    @property
    def origin(self) -> Point[To]:
        """The origin of B, resolved in A."""
        return self._origin

    def entry(self, row: Index4, column: Index4) -> Scalar:
        """The entry of the 4x4 homogeneous matrix, counted from 1.

        Returns:
            r_{row column} inside the rotation block, the origin in column 4, and the
            row (0, 0, 0, 1) at the bottom.
        """
        if row == _HOMOGENEOUS:
            return Scalar(1 if column == _HOMOGENEOUS else 0)
        if column == _HOMOGENEOUS:
            return (self._origin.x, self._origin.y, self._origin.z)[row - 1]
        return self._rotation.entry(row, column)

    @overload
    def __matmul__[Inner: Frame](
        self, other: Transform[From, Inner]
    ) -> Transform[To, Inner]: ...

    @overload
    def __matmul__(self, other: Point[From]) -> Point[To]: ...

    @overload
    def __matmul__(self, other: Vector[From]) -> Vector[To]: ...

    def __matmul__[Inner: Frame](
        self, other: Transform[From, Inner] | Point[From] | Vector[From]
    ) -> Transform[To, Inner] | Point[To] | Vector[To]:
        """Compose with ^B T_C, or resolve in A a point or vector given in B.

        Returns:
            ^A T_C for a transform; R p + t for a point; R v for a vector.
        """
        match other:
            case Transform():
                return (self._rotation @ other.rotation).with_origin(
                    self @ other.origin
                )
            case Point():
                return self._origin + self._rotation @ (other - other.frame.origin())
            case Vector():
                return self._rotation @ other

    def inverse(self) -> Transform[From, To]:
        """The inverse ^B T_A, with rotation R^T and origin -R^T t.

        Returns:
            The transform with frames swapped.
        """
        inverse = self._rotation.inverse()
        origin = inverse.relative_to.origin() - inverse @ (
            self._origin - self._origin.frame.origin()
        )
        return inverse.with_origin(origin)

    @override
    def simplify(self) -> Self:
        """Simplify the rotation and the origin.

        Returns:
            The same transform, between the same frames, simplified.
        """
        origin = self._origin.simplify()
        return type(self)(self._rotation.simplify(), origin.x, origin.y, origin.z)

    @override
    def latex(self) -> str:
        """Render as ^A T_B = [4x4 matrix].

        Returns:
            The LaTeX source.
        """
        rows = (
            " & ".join(self.entry(row, column).latex() for column in _INDICES)
            for row in _INDICES
        )
        frames = (
            rf"{{}}^{{{self._rotation.relative_to.latex}}}"
            rf"T_{{{self._rotation.of.latex}}}"
        )
        return rf"{frames} = \begin{{bmatrix}} {r' \\ '.join(rows)} \end{{bmatrix}}"

    @override
    def _layout(self) -> Layout:
        frames = f"{self._rotation.relative_to.__name__}, {self._rotation.of.__name__}"
        return Layout(
            label=f"Transform[{frames}]",
            entries=tuple(
                self.entry(row, column) for row in _INDICES for column in _INDICES
            ),
            shape=(4, 4),
        )

    @override
    def __repr__(self) -> str:
        """Print the rotation and the origin.

        Returns:
            E.g. `Transform(Rotation[Base, Link1](...), Point[Base](0, 0, d1))`.
        """
        return f"Transform({self._rotation!r}, {self._origin!r})"
