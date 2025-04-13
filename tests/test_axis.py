import sympy as sp

from robotic.transformations.axis import Axis

x, y, z = sp.symbols("x y z")


def test_numeric_initialization():
    a = Axis(1, 0, 0)
    assert isinstance(a, Axis)
    assert a.shape == (3, 1)
    assert a[0] == 1 and a[1] == 0 and a[2] == 0


def test_symbolic_initialization():
    a = Axis(x, y, z)
    assert a[0] == x
    assert a[1] == y
    assert a[2] == z


def test_skew_matrix():
    a = Axis(x, y, z)
    S = a.skew()
    expected = sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    assert S == expected
    assert S + S.T == sp.zeros(3)


def test_subs_returns_axis():
    a = Axis(x, y, z)
    b = a.subs(x, 1).subs(y, 2).subs(z, 3)
    assert isinstance(b, Axis)
    assert b == Axis(1, 2, 3)


def test_repr():
    a = Axis(1, 2, 3)
    assert repr(a) == "Axis(1, 2, 3)"


def test_zero_axis():
    a = Axis(0, 0, 0)
    assert a.norm() == 0


def test_partial_subs():
    a = Axis(x, y, 1)
    b = a.subs(x, 2)
    assert b[0] == 2
    assert b[1] == y
    assert b[2] == 1
