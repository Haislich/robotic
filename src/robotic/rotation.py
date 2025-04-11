from typing import Optional, Tuple, cast

import sympy

from robotic.axis import Axis


class Rotation(sympy.Matrix):
    axis: Axis
    theta: sympy.Symbol

    def __new__(cls, axis: Axis, theta: sympy.Symbol):
        # Compute rotation matrix
        identity = sympy.eye(3)
        skew = axis.skew()
        twist = skew * sympy.sin(theta)
        flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
        rot = identity + twist + flatten

        # Create a new Matrix instance with the rotation data
        obj = sympy.Matrix.__new__(cls, rot.rows, rot.cols, rot)
        obj.axis = axis
        obj.theta = theta
        return obj

    def __matmul__(self, other: "Rotation | sympy.Matrix") -> "Rotation | sympy.Matrix":
        obj = super().__matmul__(other)
        if isinstance(other, Rotation):
            theta = sympy.atan2(
                sympy.sqrt(
                    (obj[0, 1] - obj[1, 0]) ** 2
                    + (obj[0, 2] - obj[2, 0]) ** 2
                    + (obj[1, 2] - obj[2, 1]) ** 2
                ),
                (obj[0, 0] + obj[1, 1] + obj[2, 2]),
            )
            obj.theta = theta
            # We now are working at a symbolic level.
            # This means that while the value of the rotation
            # Surely exists, an axis angle representation is not guaranteed.
            # The regular, case is when the solution do exists.
            sin_theta = sympy.sin(theta)
            regular_axis = Axis(
                (obj[2, 1] - obj[1, 2]) / (sympy.Integer(2) * sin_theta),
                (obj[0, 2] - obj[2, 0]) / (sympy.Integer(2) * sin_theta),
                (obj[1, 0] - obj[0, 1]) / (sympy.Integer(2) * sin_theta),
            )
            # Now, in the case of singular matrix, sin(theta) = 0
            # Here, we need to be clever because we need
            # to handle both the positive and negative solution

            singular_axis_pi = Axis(
                sympy.sqrt((obj[0, 0] + 1) / sympy.Integer(2)),
                sympy.sqrt((obj[1, 1] + 1) / sympy.Integer(2)),
                sympy.sqrt((obj[2, 2] + 1) / sympy.Integer(2)),
            )
            # Finally if theta is 0 the axis is undefined
            singular_axis_zero = Axis(
                sympy.nan,
                sympy.nan,
                sympy.nan,
            )
            obj.axis = sympy.Piecewise(
                (
                    (
                        Axis(
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
                        ),
                        -Axis(
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
                        ),
                    ),
                    sympy.Eq(theta**2, sympy.pi**2),
                ),
                (
                    Axis(
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
                    ),
                    True,
                ),
            )

        return obj

    def __repr__(self) -> str:
        return (
            f"[{self[0, 0]}, {self[0, 1]}, {self[0, 2]}]\n"
            + f"[{self[1, 0]}, {self[1, 1]}, {self[1, 2]}]\n"
            + f"[{self[2, 0]}, {self[2, 1]}, {self[2, 2]}]"
        )


theta = sympy.symbols("theta")
# print(Rotation(Z, theta) @)
# print((Rotation(X, theta) @ Rotation(Z, theta)).axis)
# print((Rotation(X, theta) @ Rotation(Z, theta)).axis.subs(theta, 0))
# print()
print(
    (Rotation(X, theta) @ Rotation(Z, theta)).axis.subs(
        {theta: sympy.pi, sympy.Symbol("sign"): 1}
    )
)
# print(sympy.sqrt(theta).subs(theta, 4))
# print(sympy.Symbol("PlusMinus"))

# print(Rotation(X, theta) @ Rotation(Z, theta))
# print(type(Rotation(X, theta) @ sympy.Matrix([1, 2, 3])))

# theta = sympy.Symbol("theta")
# print(Rotation(X, theta).debug())
# print(Rotation(Y, theta).debug())
# print(Rotation(Z, theta).debug())
# print(Rotation(Y, theta).debug())
# print(Rotation(Z, theta).debug())
