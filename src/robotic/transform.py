"""Homogeneous transformations between frames, ^A T_B."""

from typing import Final, Literal, final, overload, override

from robotic.geometry import Frame, Point, Vector
from robotic.rotation import Rotation
from robotic.symbolic import Layout, Scalar, Symbolic

type Index4 = Literal[1, 2, 3, 4]

_INDICES: tuple[Index4, ...] = (1, 2, 3, 4)
_HOMOGENEOUS: Final = 4


@final
class Transform[Target: Frame, Source: Frame](Symbolic):
    """The transform ^A T_B: maps points expressed in the source B to the target A.

    It is the rotation ^A R_B and the translation ^A p, the origin of B expressed in
    A; on a point it is R p + t, on a vector R v.
    """

    __slots__ = ("_rotation", "_translation")

    _rotation: Rotation[Target, Source]
    _translation: Vector[Target]

    def __init__(self, *, target: type[Target], source: type[Source]) -> None:
        """The identity transform from `source` to `target`.

        Args:
            target: The frame A of ^A T_B.
            source: The frame B of ^A T_B.
        """
        self._rotation = Rotation.identity(target=target, source=source)
        self._translation = target.vector(0, 0, 0)

    @classmethod
    def _of_parts[A: Frame, B: Frame](
        cls, rotation: Rotation[A, B], translation: Vector[A]
    ) -> Transform[A, B]:
        transform: Transform[A, B] = object.__new__(Transform)
        transform._rotation = rotation
        transform._translation = translation
        return transform

    @property
    def rotation(self) -> Rotation[Target, Source]:
        """The rotation ^A R_B."""
        return self._rotation

    @property
    def translation(self) -> Vector[Target]:
        """The translation ^A p: the origin of B expressed in A."""
        return self._translation

    def with_rotation(
        self, rotation: Rotation[Target, Source]
    ) -> Transform[Target, Source]:
        """The same frames and translation, with this rotation.

        Returns:
            A new transform; this one is unchanged.
        """
        return self._of_parts(rotation, self._translation)

    def with_translation(
        self, translation: Vector[Target]
    ) -> Transform[Target, Source]:
        """The same frames and rotation, with this translation.

        Args:
            translation: The origin of B expressed in A.

        Returns:
            A new transform; this one is unchanged.
        """
        return self._of_parts(self._rotation, translation)

    def entry(self, row: Index4, column: Index4) -> Scalar:
        """The entry of the 4x4 homogeneous matrix, counted from 1.

        Returns:
            r_{row column} inside the rotation block, the translation in column 4,
            and the row (0, 0, 0, 1) at the bottom.
        """
        if row == _HOMOGENEOUS:
            return Scalar(1 if column == _HOMOGENEOUS else 0)
        if column == _HOMOGENEOUS:
            translation = self._translation
            return (translation.x, translation.y, translation.z)[row - 1]
        return self._rotation.entry(row, column)

    @overload
    def __matmul__[Inner: Frame](
        self, other: Transform[Source, Inner]
    ) -> Transform[Target, Inner]: ...

    @overload
    def __matmul__(self, other: Point[Source]) -> Point[Target]: ...

    @overload
    def __matmul__(self, other: Vector[Source]) -> Vector[Target]: ...

    def __matmul__[Inner: Frame](
        self, other: Transform[Source, Inner] | Point[Source] | Vector[Source]
    ) -> Transform[Target, Inner] | Point[Target] | Vector[Target]:
        """Compose with ^B T_C, or resolve in A a point or vector given in B.

        Returns:
            ^A T_C = (R1 R2, t1 + R1 t2) for a transform; R p + t for a point; R v
            for a vector.
        """
        match other:
            case Transform():
                return self._of_parts(
                    self._rotation @ other.rotation,
                    self._translation + self._rotation @ other.translation,
                )
            case Point():
                origin = self._rotation.target.origin()
                return (
                    origin
                    + self._translation
                    + self._rotation @ (other - other.frame.origin())
                )
            case Vector():
                return self._rotation @ other

    def inverse(self) -> Transform[Source, Target]:
        """The inverse ^B T_A, with rotation R^T and translation -R^T t.

        Returns:
            The transform with frames swapped.
        """
        inverse = self._rotation.inverse()
        return self._of_parts(inverse, -(inverse @ self._translation))

    @override
    def simplify(self) -> Transform[Target, Source]:
        """Simplify the rotation and the translation.

        Returns:
            The same transform, between the same frames, simplified.
        """
        return self._of_parts(self._rotation.simplify(), self._translation.simplify())

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
            rf"{{}}^{{{self._rotation.target.notation}}}"
            rf"T_{{{self._rotation.source.notation}}}"
        )
        return rf"{frames} = \begin{{bmatrix}} {r' \\ '.join(rows)} \end{{bmatrix}}"

    @override
    def _layout(self) -> Layout:
        rotation = self._rotation
        frames = f"{rotation.target.__name__}, {rotation.source.__name__}"
        return Layout(
            label=f"Transform[{frames}]",
            entries=tuple(
                self.entry(row, column) for row in _INDICES for column in _INDICES
            ),
            shape=(4, 4),
        )

    @override
    def __repr__(self) -> str:
        """Print the rotation and the translation.

        Returns:
            E.g. `Transform(Rotation[Base, Link1](...), Vector[Base](0, 0, d1))`.
        """
        return f"Transform({self._rotation!r}, {self._translation!r})"
