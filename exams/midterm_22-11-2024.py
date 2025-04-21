# %%
import math

import sympy

from robotic.manipulator import JointType, Manipulator
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


def exercise_identifier(f):
    def wrap(*args, **kwargs):
        print("#" * 40 + f"    {f.__name__}    " + "#" * 40)
        f(*args, **kwargs)
        print("#" * 40 + f"    {f.__name__}    " + "#" * 40)

    return wrap


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
    # We describe this rotation from the world frame to the
    # cylinder frame as:
    h, r = sympy.symbols("h r")
    w_t_c: HomogeneousTransformation = HomogeneousTransformation.from_rotation(
        Rotation.from_axis_angle(Y, math.pi / 2)
        @ Rotation.from_axis_angle(Z, math.pi / 2)
    ).with_translation(Translation(h / 2, 0, r))
    # The first displacement is like this:
    # by the problem is known that we roll by an amount d
    # This rotation is along the new x_c axis
    d = sympy.symbols("d")
    c_t_1 = HomogeneousTransformation.identity().with_translation(Translation(d, 0, 0))
    # But since the cylinder rolls without slipping, this also causes a rotation by an amount alpha
    # Where alpha is a counter clockwise rotation (negative) around the z axis of the cylinder
    alpha = sympy.symbols("alpha")
    c_t_1 = c_t_1.with_rotation(Rotation.from_axis_angle(Z, alpha))

    # A rotation along the world frame Z axis occurs
    # Where the initial w means that this rotation is world frame
    theta = sympy.symbols("theta")
    w_t_1 = HomogeneousTransformation.from_rotation(Rotation.from_axis_angle(Z, theta))

    # Finally a rotation along the cylinder azis is performed.
    psi = sympy.symbols("psi")
    c_t_2 = HomogeneousTransformation.from_rotation(Rotation.from_axis_angle(Z, psi))

    # Pre multiply if the rotation is in world frame -> When a transformation is defined in the world frame, you multiply it on the left.
    # Post multiply if the rotation is in body frame -> When a transformation is defined in the object’s own coordinate frame, you multiply it on the right.
    final_transformation = w_t_1 @ (w_t_c @ (c_t_1 @ c_t_2))

    final_transformation = HomogeneousTransformation(
        final_transformation.subs(
            {
                "h": 0.5,
                "r": 0.1,
                "d": 1.5,
                "theta": sympy.pi / 3,
                "psi": -sympy.pi / 2,
                # When a cylinder rotates of a displacement d
                # the angle is d/r
                "alpha": -1.5 / 0.1,
            }
        )
    )
    print(final_transformation.evalf().round(4))


# @exercise_identifier
def exercise3(debug):
    joint_types = [
        JointType.PRISMATIC,
        JointType.PRISMATIC,
        JointType.REVOLUTE,
    ]
    # a
    link_lengths = [0, 0, sympy.symbols("L")]
    # alpha
    link_twists = [-sympy.pi / 2, -sympy.pi / 2, 0]
    man = Manipulator(
        link_lengths=link_lengths,
        link_twists=link_twists,
        joint_types=joint_types,
        theta_offsets=[0.0, -sympy.pi / 2, 0],
    )
    dh_table = man.dh_table()
    print(dh_table)

    # This is dependant on the choiche that we made at the begginning, but to
    # be consistent with the solutions, we have to see at what rotation is required
    # to go from the world frame to frame 0
    w_t_zero = HomogeneousTransformation.from_rotation(
        Rotation.from_axis_angle(Z, -sympy.pi / 2)
        @ Rotation.from_axis_angle(Y, -sympy.pi / 2)
    )

    print(w_t_zero)

    three_t_zero = HomogeneousTransformation.from_rotation(
        Rotation.from_axis_angle(Y, sympy.pi / 2)
        @ Rotation.from_axis_angle(Z, sympy.pi)
    )
    print(three_t_zero)

    dh_matrix = man.dh_matrix()
    print((w_t_zero @ dh_matrix @ three_t_zero).as_translation())


# exercise1(DEBUG)
# exercise2(DEBUG)
exercise3(DEBUG)
