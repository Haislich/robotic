import sympy as sp

from robotic.axis import Axis
from robotic.rotation import EulerOrder, EulerSequence, Rotation

x, y, z = sp.symbols("x y z")
theta = sp.Symbol("theta")
phi, beta, gamma = sp.symbols("phi beta gamma")
pi = sp.pi


def test_rotation_identity():
    a = Axis(1, 0, 0)
    R = Rotation.direct_axis_angle(a, 0)
    assert sp.simplify(R - sp.eye(3)) == sp.zeros(3)


def test_rotation_orthogonality():
    a = Axis(1, 0, 0)
    R = Rotation.direct_axis_angle(a, theta)
    assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3)


def test_rotation_determinant():
    a = Axis(0, 1, 0)
    R = Rotation.direct_axis_angle(a, theta)
    assert sp.simplify(R.det() - 1) == 0


def test_rodrigues_formula():
    a = Axis(1, 0, 0)
    skew = a.skew()
    expected = (
        sp.eye(3) + skew * sp.sin(theta) + (sp.Integer(1) - sp.cos(theta)) * skew**2
    )
    R = Rotation.direct_axis_angle(a, theta)
    assert sp.simplify(R - expected) == sp.zeros(3)


def test_composition_axis_angle_consistency():
    a = Axis(0, 0, 1)
    R1 = Rotation.direct_axis_angle(a, theta)
    R2 = Rotation.direct_axis_angle(a, theta)
    R3 = R1 @ R2
    axis, _ = R3.inverse_axis_angle()
    approx_axis = axis.subs(theta, sp.pi / 2)
    assert all(isinstance(c, sp.Expr) for c in approx_axis)
    assert sp.simplify(R3.det() - 1) == 0


def test_axis_at_theta_zero():
    a = Axis(0, 0, 1)
    R = Rotation.direct_axis_angle(a, theta)
    R0 = R.subs(theta, 0)
    axis, _ = R0.inverse_axis_angle()
    assert all(sp.simplify(c) == sp.nan for c in axis)


def test_axis_at_theta_pi():
    a = Axis(1, 0, 0)
    R = Rotation.direct_axis_angle(a, theta)
    Rpi = R.subs(theta, pi)
    axis, _ = Rpi.inverse_axis_angle()
    assert all(isinstance(c, sp.Expr) for c in axis)


# def test_inverse_euler_xyz_moving():
#     R = Rotation.direct_euler(
#         angles=(phi, beta, gamma), sequence=EulerSequence.XYZ, order=EulerOrder.MOVING
#     )
#     angles, seq, order = R.inverse_euler(EulerSequence.XYZ, EulerOrder.MOVING)
#     assert seq == EulerSequence.XYZ
#     assert order == EulerOrder.MOVING
#     assert all(isinstance(a, sp.Expr) for a in angles)


# def test_inverse_euler_zxz_fixed():
#     R = Rotation.direct_euler(
#         angles=(phi, beta, gamma), sequence=EulerSequence.ZXZ, order=EulerOrder.FIXED
#     )
#     angles, seq, order = R.inverse_euler(EulerSequence.ZXZ, EulerOrder.FIXED)
#     assert seq == EulerSequence.ZXZ
#     assert order == EulerOrder.FIXED
#     assert all(isinstance(a, sp.Expr) for a in angles)
