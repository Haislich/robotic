"""Scenarios of the `scalar` capability."""

import operator
from typing import assert_type

import pytest

from robotic import (
    Expression,
    Frame,
    MissingParameterError,
    Position,
    Rotation,
    Scalar,
    Symbol,
    Transform,
    UndefinedValueError,
)


class Base(Frame):
    """Fixed frame."""


class Tool(Frame):
    """Frame of the tool."""


q = Symbol("q")


def test_mixed_arithmetic_stays_typed() -> None:
    assert_type(q, Symbol)
    assert_type(q + 1, Expression)
    assert_type(1 + q, Expression)
    assert_type(q * q, Expression)
    assert_type(2 * q - Scalar(1) / 2, Expression)
    assert_type(-q, Expression)
    assert_type(q.simplify(), Symbol)
    assert_type(2**q, Expression)
    assert_type(q / 2, Expression)
    values = (q + 1, 1 + q, q * q, 2 * q - Scalar(1) / 2, -q)
    assert [str(value) for value in values] == [
        "q + 1",
        "q + 1",
        "q**2",
        "2*q - 1/2",
        "-q",
    ]


def test_same_name_same_symbol() -> None:
    assert str(Symbol("q2") + Symbol("q2")) == "2*q2"


def test_exact_pi() -> None:
    half_pi = Scalar.pi() / 2
    assert str(half_pi) == "pi/2"


def test_exact_square_root() -> None:
    assert str(Scalar(2).sqrt()) == "sqrt(2)"


def test_trigonometry() -> None:
    assert_type(q.cos(), Expression)
    assert str(q.cos()) == "cos(q)"
    assert str(q.atan2(Symbol("x"))) == "atan2(q, x)"


def test_bool_is_not_a_number() -> None:
    with pytest.raises(TypeError, match="bool"):
        Scalar(value=True)


def test_exact_fraction() -> None:
    assert str(q + Scalar(1) / 10) == "q + 1/10"


def test_simplify_a_trigonometric_identity() -> None:
    assert str((q.cos() ** 2 + q.sin() ** 2).simplify()) == "1"


def test_latex_output() -> None:
    assert q.cos().latex() == r"\cos{\left(q \right)}"


def test_latex_does_not_simplify() -> None:
    identity = q.cos() ** 2 + q.sin() ** 2
    assert identity.latex() == r"\sin^{2}{\left(q \right)} + \cos^{2}{\left(q \right)}"


def test_equal_looking_rotations_have_no_equality() -> None:
    first = Rotation.about_z(q, target=Base, source=Tool)
    second = Rotation.about_z(q, target=Base, source=Tool)
    with pytest.raises(TypeError, match="compile them and compare the numbers"):
        operator.eq(first, second)
    with pytest.raises(TypeError, match="no equality"):
        operator.ne(first, second)


def test_scalar_as_a_truth_value_or_a_key() -> None:
    with pytest.raises(TypeError, match="no equality"):
        bool(q)
    with pytest.raises(TypeError, match="no equality"):
        hash(q)
    with pytest.raises(TypeError, match="no equality"):
        operator.contains([Symbol("q")], q)


class B1(Frame):
    """The user's first frame."""


a, b = Scalar(2), Symbol("b")


def test_constants_stay_constants() -> None:
    assert_type(Scalar(2) + 3, Scalar)
    assert_type(Scalar.pi() / 2, Scalar)
    assert_type(Scalar(2).sqrt(), Scalar)
    assert_type(Scalar(2).cos(), Scalar)
    assert isinstance(Scalar(2) + 3, Scalar)


def test_a_cancelled_symbol_stays_an_expression() -> None:
    cancelled = q - q
    assert_type(cancelled, Expression)
    assert isinstance(cancelled, Expression)
    assert str(cancelled) == "0"


def test_a_constant_with_an_expression_is_an_expression() -> None:
    assert_type(a + q, Expression)
    assert_type(q + a, Expression)
    assert_type(a.atan2(q), Expression)
    assert isinstance(a + q, Expression)
    assert isinstance(a * q, Expression)


def test_constant_or_symbolic_positions() -> None:
    assert_type(B1.position(a, a, a), Position[Scalar, B1])
    assert_type(B1.position(a, a, b), Position[Expression, B1])
    assert_type(B1.position(a, a, a + b), Position[Expression, B1])
    assert_type(Position(a, a, b, B1), Position[Expression, B1])


def test_a_constant_component_of_a_symbolic_position() -> None:
    x = B1.position(a, a, b).x
    assert_type(x, Expression)
    assert isinstance(x, Expression)
    assert str(x) == "2"


def test_constant_or_symbolic_rotation_and_transform() -> None:
    quarter = Rotation.about_z(Scalar.pi() / 2, target=B1, source=Tool)
    assert_type(quarter, Rotation[Scalar, B1, Tool])
    assert_type(
        Rotation.about_z(q, target=B1, source=Tool), Rotation[Expression, B1, Tool]
    )
    transform = quarter.homogeneous(B1.position(0, 0, b))
    assert_type(transform, Transform[Expression, B1, Tool])
    assert isinstance(transform.entry(1, 1), Expression)


def test_a_zero_no_simplification_proves() -> None:
    seventh = Scalar.pi() / 7
    total = seventh.cos() + (3 * seventh).cos() + (5 * seventh).cos()
    assert str(total.simplify() - Scalar(1) / 2) != "0"
    assert total.is_close(Scalar(1) / 2)


def test_chosen_tolerance() -> None:
    nearby = 1 + Scalar(1) / 1000
    assert Scalar(1).is_close(nearby, tolerance=1e-2)
    assert not Scalar(1).is_close(nearby)


def test_nothing_to_substitute() -> None:
    value = B1.position(a, a, b).x.evaluate()
    assert_type(value, Scalar)
    assert isinstance(value, Scalar)
    assert str(value) == "2"


def test_substituting_a_symbol() -> None:
    assert str(B1.position(a, a, b).z.evaluate({"b": 1})) == "1"


def test_a_whole_position() -> None:
    position = B1.position(a, a, b).evaluate({"b": 1, "unused": 5})
    assert_type(position, Position[Scalar, B1])
    assert repr(position) == "Position[B1](2, 2, 1)"
    assert isinstance(position.z, Scalar)


def test_missing_value() -> None:
    with pytest.raises(MissingParameterError, match="b"):
        B1.position(a, a, b).z.evaluate()


def test_equal_constants_have_no_equality() -> None:
    with pytest.raises(TypeError, match="is_close"):
        operator.eq(Scalar(2), Scalar(2))


def test_constants_are_real_numbers() -> None:
    with pytest.raises(UndefinedValueError):
        Scalar(0) / 0
    with pytest.raises(UndefinedValueError):
        Scalar(1) / 0
    with pytest.raises(UndefinedValueError):
        Scalar(0).atan2(0)
    with pytest.raises(UndefinedValueError):
        (Scalar.pi() / 2).tan()


def test_evaluating_where_an_expression_is_undefined() -> None:
    with pytest.raises(UndefinedValueError, match="q = 0"):
        (q.sin() / q).evaluate({"q": 0})


def test_the_error_says_why_a_value_is_not_real() -> None:
    with pytest.raises(UndefinedValueError, match=r"^the value is undefined \(nan\)"):
        Scalar(0) / 0
    with pytest.raises(UndefinedValueError, match=r"^the value is infinite \(zoo\)"):
        Scalar(1) / 0
