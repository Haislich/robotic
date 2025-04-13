from typing import TypeGuard, cast, overload

import sympy

from robotic.transformations.rotation import Rotation
from robotic.transformations.translation import Translation


class HomogeneousTransformation(sympy.Matrix):
    def __new__(cls, matrix: sympy.Matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Homogeneous transformation must be a 4x4 matrix")

        # Validate components, the underlying constructor should fail
        _rotation = Rotation(cast(sympy.Matrix, matrix[:3, :3]))
        _translation = Translation(cast(sympy.Matrix, matrix[:3, 3]))

        bottom = matrix.row(3)
        if not all(elem.equals(0) for elem in bottom[:3]):
            raise ValueError(
                "The first three components of the last row should be all zeros"
            )
        if bottom[3].equals(0):
            raise ValueError("The scale value cannot be 0")
        return super().__new__(cls, 4, 4, matrix)

    @property
    def scale(self) -> sympy.Expr:
        return cast(sympy.Expr, self[3, 3])

    @staticmethod
    def identity() -> "HomogeneousTransformation":
        return HomogeneousTransformation(sympy.eye(4))

    @staticmethod
    def from_rotation(rotation: Rotation) -> "HomogeneousTransformation":
        top = rotation.row_join(Translation())

        bottom = sympy.Matrix([[0, 0, 0, 1]])

        # Assemble final homogeneous matrix
        matrix = top.col_join(bottom)

        return HomogeneousTransformation(matrix)

    def as_rotation(self) -> "Rotation":
        mat = cast(sympy.Matrix, self[:3, :3])
        return Rotation(mat / self.scale)

    def with_rotation(self, rotation: Rotation) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            rotation.row_join(self.as_translation()).col_join(
                sympy.Matrix([[0, 0, 0, self.scale]])
            )
        )

    @staticmethod
    def is_rotation(obj) -> TypeGuard[Rotation]:
        return isinstance(obj, Rotation)

    @staticmethod
    def from_translation(translation: Translation) -> "HomogeneousTransformation":
        top = Rotation().row_join(translation)

        # Build bottom row: [0 0 0 1]
        bottom = sympy.Matrix([[0, 0, 0, 1]])

        # Assemble final homogeneous matrix
        matrix = top.col_join(bottom)

        return HomogeneousTransformation(matrix)

    def as_translation(self) -> "Translation":
        return Translation(cast(sympy.Matrix, self[:3, 3]))

    @staticmethod
    def is_translation(obj: sympy.Matrix) -> TypeGuard[Translation]:
        return isinstance(obj, Translation)

    def with_translation(
        self, new_translation: Translation
    ) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            self.as_rotation()
            .row_join(new_translation)
            .col_join(sympy.Matrix([[0, 0, 0, self.scale]]))
        )

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(self, other: "Translation") -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: "Rotation") -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "HomogeneousTransformation | Rotation | Translation | sympy.Matrix"
    ) -> "HomogeneousTransformation | sympy.Matrix":
        if self.is_rotation(other):
            return super().__matmul__(HomogeneousTransformation.from_rotation(other))

        if self.is_translation(other):
            return super().__matmul__(HomogeneousTransformation.from_translation(other))

        if self.is_homogeneous(other):
            return HomogeneousTransformation(super().__matmul__((other)))
        return super().__matmul__((other))
