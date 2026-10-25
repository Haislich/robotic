"""Scenarios of the `geometry` capability."""

from typing import assert_type

import pytest

from robotic import Frame, FreeAxis, Point, Scalar, UnitVector, Vector, ZeroVectorError


class Base(Frame, latex="0"):
    """Fixed frame, numbered as the exam numbers it."""


class Tool(Frame):
    """Frame of the tool, named by its class."""


q = Scalar.symbol("q")
l2 = Scalar.symbol("l2")


def test_declaring_a_frame() -> None:
    assert_type(Tool.vector(1, 0, 0), Vector[Tool])


def test_point_vector_and_origin() -> None:
    assert_type(Tool.point(1, 2, 3), Point[Tool])
    assert_type(Tool.vector(1, 2, 3), Vector[Tool])
    assert_type(Tool.origin(), Point[Tool])
    assert repr(Tool.origin()) == "Point[Tool](0, 0, 0)"


def test_reading_components_and_frame() -> None:
    tip = Tool.point(l2, 0, 0)
    assert_type(tip.x, Scalar)
    assert str(tip.x) == "l2"
    assert tip.frame is Tool


def test_coordinates_as_a_vector() -> None:
    tip = Tool.point(l2, 1, 2)
    coordinates = tip - Tool.origin()
    assert_type(coordinates, Vector[Tool])
    assert repr(coordinates) == "Vector[Tool](l2, 1, 2)"


def test_displacing_a_point() -> None:
    displaced = Tool.point(1, 0, 0) + Tool.vector(0, q, 0)
    assert_type(displaced, Point[Tool])
    assert repr(displaced) == "Point[Tool](1, q, 0)"
    assert_type(displaced - Tool.vector(0, q, 0), Point[Tool])


def test_vector_arithmetic() -> None:
    approach = Tool.vector(1, 2, 3)
    assert repr(approach + approach - approach) == "Vector[Tool](1, 2, 3)"
    assert repr(-approach) == "Vector[Tool](-1, -2, -3)"
    assert repr(2 * approach / 4) == "Vector[Tool](1/2, 1, 3/2)"
    assert repr(approach * q) == "Vector[Tool](q, 2*q, 3*q)"


def test_cross_product_of_axes() -> None:
    z = Tool.vector(1, 0, 0).cross(Tool.vector(0, 1, 0))
    assert_type(z, Vector[Tool])
    assert repr(z) == "Vector[Tool](0, 0, 1)"


def test_dot_and_norm() -> None:
    assert str(Tool.vector(1, 2, 3).dot(Tool.vector(4, 5, 6))) == "32"
    assert str(Tool.vector(3, 4, 0).norm()) == "5"


def test_building_a_unit_vector_normalizes() -> None:
    unit = UnitVector(Tool.vector(3, 4, 0))
    assert_type(unit, UnitVector[Tool])
    assert repr(unit) == "UnitVector[Tool](3/5, 4/5, 0)"
    assert (
        repr(Tool.vector(1, 1, 0).normalize())
        == "UnitVector[Tool](sqrt(2)/2, sqrt(2)/2, 0)"
    )


def test_simplified_norm() -> None:
    unit = Tool.vector(q.cos(), q.sin(), 0).normalize()
    assert repr(unit) == "UnitVector[Tool](cos(q), sin(q), 0)"


def test_zero_vector_raises() -> None:
    with pytest.raises(ZeroVectorError):
        Tool.vector(0, 0, 0).normalize()
    with pytest.raises(ZeroVectorError):
        Tool.vector(q.cos() ** 2 + q.sin() ** 2 - 1, 0, 0).normalize()


def test_free_axis_is_a_unit_vector() -> None:
    axis = FreeAxis(Base.vector(0, 0, 2))
    assert isinstance(axis, UnitVector)
    assert repr(axis) == "FreeAxis[Base](0, 0, 1)"


def test_simplify_keeps_the_frame() -> None:
    simplified = Base.vector(q.cos() ** 2 + q.sin() ** 2, 0, 0).simplify()
    assert_type(simplified, Vector[Base])
    assert repr(simplified) == "Vector[Base](1, 0, 0)"
    assert_type(Base.point(q, 0, 0).simplify(), Point[Base])
    assert_type(Base.vector(1, 0, 0).normalize().simplify(), UnitVector[Base])
    assert repr(FreeAxis(Base.vector(0, 0, 1)).simplify()) == "FreeAxis[Base](0, 0, 1)"


def test_default_name() -> None:
    assert Tool.latex == "Tool"
    assert "{}^{Tool}" in Tool.vector(1, 0, 0).latex()


def test_declared_name() -> None:
    assert Base.latex == "0"
    assert Base.vector(q, 0, 0).latex() == (
        r"{}^{0}\begin{bmatrix} q \\ 0 \\ 0 \end{bmatrix}"
    )


def test_misspelled_declaration_raises_at_runtime() -> None:
    with pytest.raises(TypeError, match="latx"):
        type("Misspelled", (Frame,), {}, latx="1")
