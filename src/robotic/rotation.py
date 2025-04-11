from typing import Any, Optional, TypeGuard, cast, overload

import sympy

from robotic.axis import Axis


class Rotation(sympy.Matrix):
    _axis: Optional[Axis] = None
    _theta: Optional[sympy.Expr | sympy.Basic | float] = None

    def __new__(cls, axis: Axis, theta: sympy.Expr | sympy.Basic | float):
        # Compute rotation matrix
        identity = sympy.eye(3)
        skew = sympy.Matrix(axis.skew())
        twist = skew * sympy.sin(theta)
        flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
        rot = identity + twist + flatten

        # Create a new Matrix instance with the rotation data
        obj = sympy.Matrix.__new__(cls, rot.rows, rot.cols, rot)
        return obj

    @property
    def axis(self) -> Axis:
        if self._axis is not None:
            return self._axis
        theta = self.theta
        sin_theta = sympy.sin(theta)

        self = cast(Any, self)

        regular_axis = Axis(
            (self[2, 1] - self[1, 2]) / (sympy.Integer(2) * sin_theta),
            (self[0, 2] - self[2, 0]) / (sympy.Integer(2) * sin_theta),
            (self[1, 0] - self[0, 1]) / (sympy.Integer(2) * sin_theta),
        )

        singular_axis_pi = Axis(
            sympy.sqrt((self[0, 0] + 1) / 2),
            sympy.sqrt((self[1, 1] + 1) / 2),
            sympy.sqrt((self[2, 2] + 1) / 2),
        )

        singular_axis_zero = Axis(sympy.nan, sympy.nan, sympy.nan)

        axis = Axis(
            sympy.Piecewise(
                (singular_axis_zero[0], sympy.Eq(theta, 0)),
                (singular_axis_pi[0], sympy.Eq(sin_theta, 0)),
                (regular_axis[0], True),
            ),
            sympy.Piecewise(
                (singular_axis_zero[1], sympy.Eq(theta, 0)),
                (singular_axis_pi[1], sympy.Eq(sin_theta, 0)),
                (regular_axis[1], True),
            ),
            sympy.Piecewise(
                (singular_axis_zero[2], sympy.Eq(theta, 0)),
                (singular_axis_pi[2], sympy.Eq(sin_theta, 0)),
                (regular_axis[2], True),
            ),
        )
        self._axis = axis
        return axis

    @property
    def theta(self) -> sympy.Expr | sympy.Basic | float:
        if self._theta is not None:
            return self._theta
        self = cast(Any, self)
        theta = sympy.atan2(
            sympy.sqrt(
                (self[0, 1] - self[1, 0]) ** 2
                + (self[0, 2] - self[2, 0]) ** 2
                + (self[1, 2] - self[2, 1]) ** 2
            ),
            (self[0, 0] + self[1, 1] + self[2, 2]),
        )
        self._theta = theta
        return theta

    @classmethod
    def from_matrix(cls, mat: sympy.Matrix) -> "Rotation":
        if mat.shape != (3, 3):
            raise ValueError("A rotation matrix is a 3 x 3")
        return super().__new__(cls, 3, 3, mat)

    @staticmethod
    def is_rotation(obj) -> TypeGuard["Rotation"]:
        return isinstance(obj, Rotation)

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(self, other: "Rotation | sympy.Matrix") -> "Rotation | sympy.Matrix":
        obj = super().__matmul__(other)
        if Rotation.is_rotation(other):
            return Rotation.from_matrix(obj)
        return obj

    def __repr__(self) -> str:
        return f"Rotation(axis={self.axis}, theta={self.theta})"

    def __str__(self) -> str:
        return (
            f"[{self[0, 0]}, {self[0, 1]}, {self[0, 2]}]\n"
            f"[{self[1, 0]}, {self[1, 1]}, {self[1, 2]}]\n"
            f"[{self[2, 0]}, {self[2, 1]}, {self[2, 2]}]"
        )
