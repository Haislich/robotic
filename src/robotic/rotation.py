from typing import cast

import sympy


class Axis(sympy.Matrix):
    def __new__(
        cls,
        x1: float | sympy.Float,
        x2: float | sympy.Float,
        x3: float | sympy.Float,
    ):
        obj = sympy.Matrix.__new__(cls, 3, 1, [x1, x2, x3])
        if obj.norm() != 1:
            raise ValueError(f"Axis must have unitary norm, found {obj.norm()}")
        return obj

    def skew(self):
        x = cast(sympy.Expr, self[0])
        y = cast(sympy.Expr, self[1])
        z = cast(sympy.Expr, self[2])
        return sympy.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def __repr__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"


X = Axis(1, 0, 0)
Y = Axis(0, 1, 0)
Z = Axis(0, 0, 1)


class Rotation:
    def __init__(self, axis: Axis, value: sympy.Symbol):
        self.axis = axis
        self.value = value

        identity = sympy.eye(3)
        twist = axis.skew() * sympy.sin(value)
        flatten = (sympy.Integer(1) - sympy.cos(value)) * axis.skew().pow(2)

        self._rot_matrix: sympy.Matrix = identity + twist + flatten

    def debug(self) -> str:
        return f"Axis: {self.axis}, {self.value}:\n{self.__repr__()}"

    def evaluate(self, value: float) -> sympy.Matrix:
        return self._rot_matrix.subs(self.value, value)

    def __repr__(self) -> str:
        return (
            f"[{self._rot_matrix[0, 0]}, {self._rot_matrix[0, 1]}, {self._rot_matrix[0, 2]}]\n"
            + f"[{self._rot_matrix[1, 0]}, {self._rot_matrix[1, 1]}, {self._rot_matrix[1, 2]}]\n"
            + f"[{self._rot_matrix[2, 0]}, {self._rot_matrix[2, 1]}, {self._rot_matrix[2, 2]}]"
        )

    def __matmul__(self, other: "Rotation") -> "Rotation":
        return self._rot_matrix @ other._rot_matrix


# theta = sympy.Symbol("theta")
# print(Rotation(X, theta).debug())
# print(Rotation(Y, theta).debug())
# print(Rotation(Z, theta).debug())
# print(Rotation(Y, theta).debug())
# print(Rotation(Z, theta).debug())
