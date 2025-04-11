from typing import cast

import sympy


class Axis(sympy.Matrix):
    def __new__(
        cls,
        x1: float | sympy.Float | sympy.Expr,
        x2: float | sympy.Float | sympy.Expr,
        x3: float | sympy.Float | sympy.Expr,
    ):
        obj = sympy.Matrix.__new__(cls, 3, 1, [x1, x2, x3])
        return obj

    def skew(self):
        x = cast(sympy.Expr, self[0])
        y = cast(sympy.Expr, self[1])
        z = cast(sympy.Expr, self[2])
        return sympy.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def subs(self, *args, **kwargs) -> "Axis":
        result = super().subs(*args, **kwargs)
        return Axis(*result)

    def __repr__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"

    def __str__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"


X = Axis(1, 0, 0)
Y = Axis(0, 1, 0)
Z = Axis(0, 0, 1)
