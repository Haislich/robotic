# %%
import math

import sympy

from robotic.transformations import (
    Axis,
    EulerOrder,
    EulerSequence,
    HomogeneousTransformation,
    Rotation,
    Translation,
    X,
    Y,
    Z,
)

DEBUG = False


def exercise1(debug) -> None:
    alpha, beta, gamma = sympy.symbols("alpha beta gamma")
    # Initial orientation specified in ZXY euler angles,
    # with angles alpha, beta and gamma.
    initial_rot: Rotation = Rotation.from_euler(
        (alpha, beta, gamma),
        EulerSequence.ZXY,
        EulerOrder.FIXED,
    )
    # And should reach a final orientation specified by r,theta
    theta = sympy.symbols("theta")
    final_rot = Rotation.from_axis_angle(
        Axis(0, -math.sqrt(2) / 2.0, math.sqrt(2) / 2), theta
    )
    # For debuggin purposes, check if the symbolic
    # vaules match the professor calculations
    if debug:
        print(initial_rot)
    initial_rot: Rotation = initial_rot.subs(
        {"alpha": math.pi / 2, "beta": math.pi / 4, "gamma": -math.pi / 4}
    )
    print(initial_rot)
    if debug:
        print(final_rot)
    final_rot: Rotation = final_rot.subs(theta, math.pi / 6)
    print(final_rot)
    # We need to find the transformation from from R_i to R_f => R_{if} = R_i^T R_f.
    r_trans: Rotation = initial_rot.T @ final_rot
    (phi, chi, psi), _, _ = r_trans.to_euler(EulerSequence.YXY, EulerOrder.FIXED)
    print(round(phi.evalf(), 4), round(chi.evalf(), 4), round(psi.evalf(), 4))  # type:ignore


def exercise2(debug):
    # Looking at the frame in red and the frame in black:
    # - We first rotate by pi/2 on Y
    # - On the new Frame we rotate Z by pi/2
    t_0: HomogeneousTransformation = HomogeneousTransformation.from_rotation(
        Rotation.from_axis_angle(Y, math.pi / 2)
        @ Rotation.from_axis_angle(Z, math.pi / 2)
    )
    # The first displacement is like this:
    # by the problem is known that we roll by an amount d
    # This rotation is along the new x_c axis
    d = sympy.symbols("d")
    t_1 = HomogeneousTransformation.identity().with_translation(Translation(d, 0, 0))
    # But since the cylinder rolls without slipping, this also causes a rotation by an amount alpha
    # Where alpha is a counter clockwise rotation (negative) around the z axis of the cylinder
    alpha = sympy.symbols("alpha")
    t_1 = t_1.with_rotation(Rotation.from_axis_angle(Z, alpha))
    theta = sympy.symbols("theta")
    # We need to
    t2w = HomogeneousTransformation.from_rotation(Rotation.from_axis_angle(Z, theta))

    psi = sympy.symbols("psi")

    t3 = HomogeneousTransformation.from_rotation(Rotation.from_axis_angle(Z, psi))

    # Pre multiply if the rotation is in world frame
    # Post multiply if the rotation is in body frame


# exercise1(DEBUG)
exercise2(DEBUG)

# %%
