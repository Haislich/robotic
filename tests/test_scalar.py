"""Scenarios of the `scalar` capability."""

import operator
from typing import assert_type

import pytest

from robotic import Frame, Rotation, Scalar, Symbol


class Base(Frame):
    """Fixed frame."""


class Tool(Frame):
    """Frame of the tool."""


q = Scalar.symbol("q")


def test_mixed_arithmetic_stays_typed() -> None:
    assert_type(q, Symbol)
    assert_type(q + 1, Scalar)
    assert_type(1 + q, Scalar)
    assert_type(q * q, Scalar)
    assert_type(2 * q - Scalar(1) / 2, Scalar)
    assert_type(-q, Scalar)
    assert_type(q.simplify(), Symbol)
    assert_type(2**q, Scalar)
    assert_type(q / 2, Scalar)
    values = (q + 1, 1 + q, q * q, 2 * q - Scalar(1) / 2, -q)
    assert [str(value) for value in values] == [
        "q + 1",
        "q + 1",
        "q**2",
        "2*q - 1/2",
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


def test_bool_is_not_a_number() -> None:
    with pytest.raises(TypeError, match="bool"):
        Scalar(value=True)


def test_exact_fraction() -> None:
    assert str(q + Scalar(1) / 10) == "q + 1/10"


def test_simplify_a_trigonometric_identity() -> None:
    assert str((q.cos() ** 2 + q.sin() ** 2).simplify()) == "1"


def test_latex_output() -> None:
    assert q.cos().latex() == r"\cos{\left(q \right)}"


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
        operator.contains([Scalar.symbol("q")], q)
