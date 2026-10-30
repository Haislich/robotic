"""Scenarios of the `geometry` capability."""

import operator
import re
from typing import assert_type

import pytest

from robotic import (
    Direction,
    Displacement,
    Expression,
    Frame,
    FrameMismatchError,
    FreeAxis,
    Point,
    Position,
    Scalar,
    Symbol,
    ZeroVectorError,
)


class Base(Frame, notation="0"):
    """Fixed frame, numbered as the exam numbers it."""


class Tool(Frame):
    """Frame of the tool, named by its class."""


q = Symbol("q")
l2 = Symbol("l2")


def test_declaring_a_frame() -> None:
    assert_type(Tool.displacement(1, 0, 0), Displacement[Scalar, Tool])


def test_position_displacement_and_origin() -> None:
    assert_type(Tool.position(1, 2, 3), Position[Scalar, Tool])
    assert_type(Tool.displacement(1, 2, 3), Displacement[Scalar, Tool])
    assert_type(Tool.origin(), Position[Scalar, Tool])
    assert repr(Tool.origin()) == "Position[Tool](0, 0, 0)"


def test_point_is_a_position() -> None:
    tip: Point[Scalar, Tool] = Tool.position(1, 0, 0)
    assert Point is Position
    assert isinstance(tip, Point)
    match tip:
        case Point():
            matched = True
    assert matched


def test_reading_components_and_frame() -> None:
    tip = Tool.position(l2, 0, 0)
    assert_type(tip.x, Expression)
    assert str(tip.x) == "l2"
    assert tip.frame is Tool


def test_coordinates_as_a_displacement() -> None:
    tip = Tool.position(l2, 1, 2)
    coordinates = tip - Tool.origin()
    assert_type(coordinates, Displacement[Expression, Tool])
    assert repr(coordinates) == "Displacement[Tool](l2, 1, 2)"


def test_moving_a_position() -> None:
    moved = Tool.position(1, 0, 0) + Tool.displacement(0, q, 0)
    assert_type(moved, Position[Expression, Tool])
    assert repr(moved) == "Position[Tool](1, q, 0)"
    assert_type(moved - Tool.displacement(0, q, 0), Position[Expression, Tool])


def test_displacement_arithmetic() -> None:
    approach = Tool.displacement(1, 2, 3)
    assert repr(approach + approach - approach) == "Displacement[Tool](1, 2, 3)"
    assert repr(-approach) == "Displacement[Tool](-1, -2, -3)"
    assert repr(2 * approach / 4) == "Displacement[Tool](1/2, 1, 3/2)"
    assert repr(approach * q) == "Displacement[Tool](q, 2*q, 3*q)"


def test_cross_product_of_axes() -> None:
    z = Tool.displacement(1, 0, 0).cross(Tool.displacement(0, 1, 0))
    assert_type(z, Displacement[Scalar, Tool])
    assert repr(z) == "Displacement[Tool](0, 0, 1)"


def test_dot_and_norm() -> None:
    assert str(Tool.displacement(1, 2, 3).dot(Tool.displacement(4, 5, 6))) == "32"
    assert str(Tool.displacement(3, 4, 0).norm()) == "5"


def test_normalizing_gives_a_direction() -> None:
    unit = Tool.displacement(3, 4, 0).normalize()
    assert_type(unit, Direction[Scalar, Tool])
    assert repr(unit) == "Direction[Tool](3/5, 4/5, 0)"
    assert (
        repr(Tool.displacement(1, 1, 0).normalize())
        == "Direction[Tool](sqrt(2)/2, sqrt(2)/2, 0)"
    )


def test_a_direction_from_components_normalizes() -> None:
    assert_type(Tool.direction(0, 0, 1), Direction[Scalar, Tool])
    assert repr(Tool.direction(0, 0, 1)) == "Direction[Tool](0, 0, 1)"
    assert repr(Tool.direction(1, 1, 0)) == "Direction[Tool](sqrt(2)/2, sqrt(2)/2, 0)"
    with pytest.raises(ZeroVectorError):
        Tool.direction(0, 0, 0)


def test_symbolic_factor_on_the_left() -> None:
    l1 = Symbol("l1")
    x_axis = Tool.displacement(1, 0, 0)
    assert_type(l1 * x_axis, Displacement[Expression, Tool])
    assert repr(l1 * x_axis) == "Displacement[Tool](l1, 0, 0)"
    assert repr(x_axis * l1) == "Displacement[Tool](l1, 0, 0)"


def test_negating_a_direction_keeps_it_unit() -> None:
    unit = Tool.displacement(3, 4, 0).normalize()
    assert_type(-unit, Direction[Scalar, Tool])
    assert repr(-unit) == "Direction[Tool](-3/5, -4/5, 0)"
    assert repr(-FreeAxis(Tool.displacement(0, 0, 1))) == "FreeAxis[Tool](0, 0, -1)"


def test_the_norm_is_not_simplified() -> None:
    unit = Tool.displacement(q.cos(), q.sin(), 0).normalize()
    norm = "sqrt(sin(q)**2 + cos(q)**2)"
    assert repr(unit) == f"Direction[Tool](cos(q)/{norm}, sin(q)/{norm}, 0)"
    assert repr(unit.simplify()) == "Direction[Tool](cos(q), sin(q), 0)"


def test_zero_vector_raises() -> None:
    with pytest.raises(ZeroVectorError):
        Tool.displacement(0, 0, 0).normalize()


def test_a_constant_zero_no_simplification_proves_raises() -> None:
    seventh = Scalar.pi() / 7
    zero = seventh.cos() + (3 * seventh).cos() + (5 * seventh).cos() - Scalar(1) / 2
    with pytest.raises(ZeroVectorError):
        Tool.displacement(zero, 0, 0).normalize()
    with pytest.raises(ZeroVectorError):
        Tool.displacement(Scalar(1) / 10**12, 0, 0).normalize()


def test_a_norm_that_vanishes_at_one_configuration() -> None:
    direction = Tool.displacement(q, 0, 0).normalize()
    assert_type(direction, Direction[Expression, Tool])
    assert repr(direction) == "Direction[Tool](q/Abs(q), 0, 0)"


def test_a_symbolic_zero_does_not_raise() -> None:
    zero = (q + 1) ** 2 - q * q - 2 * q - 1
    assert_type(Tool.displacement(zero, 0, 0).normalize(), Direction[Expression, Tool])
    assert_type(
        Tool.displacement(q.cos() ** 2 + q.sin() ** 2 - 1, 0, 0).normalize(),
        Direction[Expression, Tool],
    )


def test_free_axis_is_a_direction() -> None:
    axis = FreeAxis(Base.displacement(0, 0, 2))
    assert isinstance(axis, Direction)
    assert repr(axis) == "FreeAxis[Base](0, 0, 1)"


def test_simplify_keeps_the_frame() -> None:
    simplified = Base.displacement(q.cos() ** 2 + q.sin() ** 2, 0, 0).simplify()
    assert_type(simplified, Displacement[Expression, Base])
    assert repr(simplified) == "Displacement[Base](1, 0, 0)"
    assert_type(Base.position(q, 0, 0).simplify(), Position[Expression, Base])
    assert_type(Base.direction(1, 0, 0).simplify(), Direction[Scalar, Base])
    assert (
        repr(FreeAxis(Base.displacement(0, 0, 1)).simplify())
        == "FreeAxis[Base](0, 0, 1)"
    )


def test_default_name() -> None:
    assert Tool.notation == "Tool"
    assert "{}^{Tool}" in Tool.displacement(1, 0, 0).latex()


def test_declared_notation() -> None:
    assert Base.notation == "0"
    assert Base.displacement(q, 0, 0).latex() == (
        r"{}^{0}\begin{bmatrix} q \\ 0 \\ 0 \end{bmatrix}"
    )


def test_misspelled_declaration_raises_at_runtime() -> None:
    with pytest.raises(TypeError, match="notaton"):
        type("Misspelled", (Frame,), {}, notaton="1")


def test_frame_mismatch_messages_name_the_operands_by_kind() -> None:
    base, tool = Base.displacement(1, 0, 0), Tool.displacement(0, 1, 0)
    operands = "Displacement[Base] and Displacement[Tool]"
    frames = "frame Tool where Base is expected"

    def message(verb: str) -> str:
        return f"^{re.escape(f'cannot {verb} {operands}: {frames}')}$"

    with pytest.raises(FrameMismatchError, match=message("add")):
        operator.methodcaller("__add__", tool)(base)
    with pytest.raises(FrameMismatchError, match=message("take the dot product of")):
        operator.methodcaller("dot", tool)(base)
    with pytest.raises(FrameMismatchError, match=message("take the cross product of")):
        operator.methodcaller("cross", tool)(base)


def test_an_evaluated_free_axis_stays_a_free_axis() -> None:
    axis = FreeAxis(Base.displacement(q, 0, 1)).evaluate({"q": 0})
    assert repr(axis) == "FreeAxis[Base](0, 0, 1)"
