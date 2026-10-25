"""Scenarios of the `rotation` capability, with spatialmath as the oracle."""

import math
from dataclasses import dataclass
from typing import assert_type

import numpy as np
import pytest
from spatialmath import SO3

from robotic import (
    AxisAngleSolutions,
    Frame,
    FreeAxis,
    Rotation,
    Scalar,
    UnitVector,
    Vector,
)


class Base(Frame, latex="0"):
    """Fixed frame."""


class Link1(Frame):
    """Frame of the first link."""


class Tool(Frame):
    """Frame of the tool."""


class Prime(Frame, latex="'"):
    """Frame after the first elementary rotation."""


class Second(Frame, latex="''"):
    """Frame after the second elementary rotation."""


@dataclass(frozen=True)
class Joint:
    """One joint value."""

    q: float


q = Scalar.symbol("q")
q1 = Scalar.symbol("q1")
pi = Scalar.pi()
INDICES = (1, 2, 3)
SAMPLES = (-math.pi + 1e-6, -1.2, -1e-6, 0.0, 1e-6, 0.7, math.pi / 2, math.pi - 1e-6)


def entries[A: Frame, B: Frame](rotation: Rotation[A, B]) -> list[list[str]]:
    return [[str(rotation.entry(row, column)) for column in INDICES] for row in INDICES]


def numbers(rotation: Rotation[Base, Tool], value: float) -> np.ndarray:
    return np.asarray(rotation.lambdify(Joint)(Joint(q=value)).components)


def test_elementary_rotation() -> None:
    rotation = Rotation.about_z(q1, of=Link1, relative_to=Base)
    assert_type(rotation, Rotation[Base, Link1])
    assert entries(rotation) == [
        ["cos(q1)", "-sin(q1)", "0"],
        ["sin(q1)", "cos(q1)", "0"],
        ["0", "0", "1"],
    ]
    assert rotation.relative_to is Base
    assert rotation.of is Link1


def test_reading_r12() -> None:
    rotation = Rotation.about_z(q1, of=Link1, relative_to=Base)
    assert_type(rotation.entry(1, 2), Scalar)
    assert str(rotation.entry(1, 2)) == "-sin(q1)"


def test_rotation_about_an_axis() -> None:
    axis = Base.vector(0, 0, 1).normalize()
    about_axis = Rotation.about_axis(axis, q, of=Tool)
    assert_type(about_axis, Rotation[Base, Tool])
    assert entries(about_axis) == entries(
        Rotation.about_z(q, of=Tool, relative_to=Base)
    )


def test_identity() -> None:
    identity = Rotation.identity(of=Tool, relative_to=Base)
    assert entries(identity) == [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]


def test_chaining() -> None:
    base_link1 = Rotation.about_z(q1, of=Link1, relative_to=Base)
    link1_tool = Rotation.about_x(q, of=Tool, relative_to=Link1)
    assert_type(base_link1 @ link1_tool, Rotation[Base, Tool])


def test_chain_through_intermediate_frames() -> None:
    first = Rotation.about_z(q1, of=Prime, relative_to=Base)
    second = Rotation.about_y(q, of=Second, relative_to=Prime)
    third = Rotation.about_x(q1, of=Tool, relative_to=Second)
    assert_type(first @ second @ third, Rotation[Base, Tool])


def test_inverse_composes_to_identity() -> None:
    rotation = Rotation.about_z(q1, of=Link1, relative_to=Base) @ Rotation.about_x(
        q, of=Tool, relative_to=Link1
    )
    assert_type(rotation.inverse(), Rotation[Tool, Base])
    assert entries((rotation @ rotation.inverse()).simplify()) == [
        ["1", "0", "0"],
        ["0", "1", "0"],
        ["0", "0", "1"],
    ]


def test_resolving_in_the_target_frame() -> None:
    resolved = Rotation.about_z(pi / 2, of=Tool, relative_to=Base) @ Tool.vector(
        1, 0, 0
    )
    assert_type(resolved, Vector[Base])
    assert repr(resolved) == "Vector[Base](0, 1, 0)"


def test_regular_case() -> None:
    solutions = Rotation.about_z(pi / 3, of=Tool, relative_to=Base).to_axis_angle()
    assert_type(solutions, AxisAngleSolutions[Base])
    assert_type(solutions.first.axis, UnitVector[Base])
    assert repr(solutions.first.axis) == "UnitVector[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "pi/3"
    assert repr(solutions.second.axis) == "UnitVector[Base](0, 0, -1)"
    assert str(solutions.second.angle) == "-pi/3"


def test_symbolic_rotation() -> None:
    solutions = Rotation.about_z(q, of=Tool, relative_to=Base).to_axis_angle()
    assert repr(solutions.first.axis) == "UnitVector[Base](0, 0, sin(q)/Abs(sin(q)))"
    assert str(solutions.first.angle) == "atan2(Abs(sin(q)), cos(q))"
    assert str(solutions.valid_when.expression) == "Abs(sin(q))"
    assert solutions.valid_when.tolerance == pytest.approx(1e-9)
    assert (
        solutions.valid_when.latex() == r"\left|{\sin{\left(q \right)}}\right| \neq 0"
    )


def test_half_turn_about_x() -> None:
    solutions = Rotation.about_x(pi, of=Tool, relative_to=Base).to_axis_angle()
    assert repr(solutions.first.axis) == "UnitVector[Base](1, 0, 0)"
    assert str(solutions.first.angle) == "pi"
    assert repr(solutions.second.axis) == "UnitVector[Base](-1, 0, 0)"
    assert str(solutions.second.angle) == "pi"


def test_default_free_axis() -> None:
    solutions = Rotation.identity(of=Tool, relative_to=Base).to_axis_angle()
    assert isinstance(solutions.first.axis, FreeAxis)
    assert repr(solutions.first.axis) == "FreeAxis[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "0"
    assert repr(solutions.second.axis) == "FreeAxis[Base](0, 0, -1)"
    assert str(solutions.second.angle) == "0"


def test_chosen_free_axis() -> None:
    x_axis = Base.vector(1, 0, 0).normalize()
    solutions = Rotation.identity(of=Tool, relative_to=Base).to_axis_angle(
        free_axis=x_axis
    )
    assert repr(solutions.first.axis) == "FreeAxis[Base](1, 0, 0)"
    assert repr(solutions.second.axis) == "FreeAxis[Base](-1, 0, 0)"


def test_simplify_keeps_frames() -> None:
    rotation = Rotation.about_z(q, of=Tool, relative_to=Base)
    assert_type(rotation.simplify(), Rotation[Base, Tool])


def test_latex_names_both_frames() -> None:
    rotation = Rotation.about_z(q1, of=Link1, relative_to=Base)
    assert rotation.latex().startswith(r"{}^{0}R_{Link1} = \begin{bmatrix}")


@pytest.mark.parametrize("value", SAMPLES)
def test_elementary_matrices_match_the_oracle(value: float) -> None:
    for build, oracle in (
        (Rotation.about_x, SO3.Rx),
        (Rotation.about_y, SO3.Ry),
        (Rotation.about_z, SO3.Rz),
    ):
        rotation = build(q, of=Tool, relative_to=Base)
        np.testing.assert_allclose(
            numbers(rotation, value), oracle(value).A, atol=1e-12
        )


@pytest.mark.parametrize("value", SAMPLES)
def test_axis_rotation_matches_the_oracle(value: float) -> None:
    axis = Base.vector(1, 2, 3).normalize()
    rotation = Rotation.about_axis(axis, q, of=Tool)
    oracle = SO3.AngleAxis(value, [1, 2, 3]).A
    np.testing.assert_allclose(numbers(rotation, value), oracle, atol=1e-12)


@pytest.mark.parametrize("value", [1e-6, 1e-3, 0.5, math.pi - 1e-3, math.pi - 1e-6])
def test_axis_angle_matches_the_oracle(value: float) -> None:
    rotation = Rotation.about_axis(Base.vector(1, 2, 3).normalize(), q, of=Tool)
    first = rotation.to_axis_angle().first
    axis = np.asarray(first.axis.lambdify(Joint)(Joint(q=value)).components)
    angle = float(first.angle.lambdify(Joint)(Joint(q=value)).components)
    expected = numbers(rotation, value)
    np.testing.assert_allclose(SO3.AngleAxis(angle, axis).A, expected, atol=1e-9)
    oracle_angle, _ = SO3(expected).angvec()
    assert angle == pytest.approx(oracle_angle, abs=1e-3)
    assert angle == pytest.approx(value, abs=1e-9)
