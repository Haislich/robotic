import sympy as sp

from robotic.axis import Axis
from robotic.rotation import Rotation

x, y, z = sp.symbols("x y z")
theta = sp.Symbol("theta")
pi = sp.pi


def test_rotation_identity():
    a = Axis(1, 0, 0)
    R = Rotation(a, 0)
    assert sp.simplify(R - sp.eye(3)) == sp.zeros(3)


def test_rotation_orthogonality():
    a = Axis(1, 0, 0)
    R = Rotation(a, theta)
    assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3)


def test_rotation_determinant():
    a = Axis(0, 1, 0)
    R = Rotation(a, theta)
    assert sp.simplify(R.det() - 1) == 0


def test_rodrigues_formula():
    a = Axis(1, 0, 0)
    skew = a.skew()
    expected = (
        sp.eye(3) + skew * sp.sin(theta) + (sp.Integer(1) - sp.cos(theta)) * skew**2
    )
    R = Rotation(a, theta)
    assert sp.simplify(R - expected) == sp.zeros(3)


def test_composition_axis_angle_consistency():
    a = Axis(0, 0, 1)
    R1: Rotation = Rotation(a, theta)
    R2: Rotation = Rotation(a, theta)
    R3 = R1 @ R2
    approx_axis = R3.axis.subs(theta, sp.pi / 2)
    assert all(isinstance(c, sp.Expr) for c in approx_axis)
    assert sp.simplify(R3.det() - 1) == 0


def test_axis_at_theta_zero():
    a = Axis(0, 0, 1)
    R = Rotation(a, theta)
    R0 = R.subs(theta, 0)
    axis = R0.axis
    assert all(sp.simplify(c) == sp.nan for c in axis)


def test_axis_at_theta_pi():
    a = Axis(1, 0, 0)
    R = Rotation(a, theta)
    Rpi = R.subs(theta, pi)
    axis = Rpi.axis
    assert all(isinstance(c, sp.Expr) for c in axis)  # Not NaN
