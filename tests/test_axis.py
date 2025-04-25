import sympy

from robotic.transformations import Axis  # or wherever Axis is defined


def test_axis_creation():
    axis = Axis(1, 2, 3)
    assert axis.shape == (3, 1)
    assert axis[0] == 1
    assert axis[1] == 2
    assert axis[2] == 3


def test_axis_creation_symbolic():
    x, y, z = sympy.symbols("x y z")
    axis = Axis(x, y, z)
    assert axis[0] == x
    assert axis[1] == y
    assert axis[2] == z


def test_skew_matrix_numeric():
    axis = Axis(1, 2, 3)
    skew = axis.skew()
    expected = sympy.Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    assert skew == expected


def test_skew_matrix_symbolic():
    x, y, z = sympy.symbols("x y z")
    axis = Axis(x, y, z)
    skew = axis.skew()
    expected = sympy.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    assert skew == expected


def test_axis_subs():
    x, y, z = sympy.symbols("x y z")
    axis = Axis(x, y, z)
    substituted = axis.subs({x: 1, y: 2, z: 3})
    assert isinstance(substituted, Axis)
    assert substituted == Axis(1, 2, 3)


def test_axis_repr_str():
    axis = Axis(1, 0, 0)
    assert repr(axis) == "Axis(1, 0, 0)"
    assert str(axis) == "Axis(1, 0, 0)"
