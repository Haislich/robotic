from typing import TypeGuard, overload

import sympy

from robotic.transformations.homogenous import HomogeneousTransformation


class Translation(sympy.Matrix):
    def __new__(cls, vec: sympy.Matrix = sympy.Matrix([0, 0, 0])):
        if vec.shape != (3, 1):
            raise ValueError("Translation must be 2D or 3D column vector.")
        return sympy.Matrix.__new__(cls, vec.rows, vec.cols, vec)

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "HomogeneousTransformation  | Translation | sympy.Matrix"
    ) -> "HomogeneousTransformation | sympy.Matrix":
        if self.is_homogeneous(other):
            return HomogeneousTransformation(super().__matmul__((other)))
        return super().__matmul__((other))
