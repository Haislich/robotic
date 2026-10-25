"""Scenarios of the `scalar` capability."""

from typing import assert_type

from robotic import Scalar, Symbol

q = Scalar.symbol("q")


def test_mixed_arithmetic_stays_typed() -> None:
    assert_type(q, Symbol)
    assert_type(q + 1, Scalar)
    assert_type(1 + q, Scalar)
    assert_type(q * q, Scalar)
    assert_type(2 * q - 0.5, Scalar)
    assert_type(-q, Scalar)
    assert_type(q.simplify(), Symbol)
    assert_type(2**q, Scalar)
    assert_type(q / 2, Scalar)
    assert [str(value) for value in (q + 1, 1 + q, q * q, 2 * q - 0.5, -q)] == [
        "q + 1",
        "q + 1",
        "q**2",
        "2*q - 0.5",
        "-q",
    ]


def test_same_name_same_symbol() -> None:
    assert str(Scalar.symbol("q2") + Scalar.symbol("q2")) == "2*q2"


def test_exact_pi() -> None:
    half_pi = Scalar.pi() / 2
    assert str(half_pi) == "pi/2"


def test_exact_square_root() -> None:
    assert str(Scalar(2).sqrt()) == "sqrt(2)"


def test_trigonometry() -> None:
    assert_type(q.cos(), Scalar)
    assert str(q.cos()) == "cos(q)"
    assert str(q.atan2(Scalar.symbol("x"))) == "atan2(q, x)"


def test_float_inside_an_expression() -> None:
    assert str(q + 0.1) == "q + 0.1"
    assert str(q + Scalar(1) / 10) == "q + 1/10"


def test_simplify_a_trigonometric_identity() -> None:
    assert str((q.cos() ** 2 + q.sin() ** 2).simplify()) == "1"


def test_latex_output() -> None:
    assert q.cos().latex() == r"\cos{\left(q \right)}"


def test_is_zero_after_simplification() -> None:
    assert (q.cos() ** 2 + q.sin() ** 2 - 1).is_zero()
    assert not q.is_zero()
