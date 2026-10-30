"""Scenarios of the `rotation` capability, with spatialmath as the oracle."""

import math
import operator
from dataclasses import dataclass
from typing import assert_type

import numpy as np
import pytest
from spatialmath import SO3

from robotic import (
    AxisAngleSolutions,
    Direction,
    Displacement,
    Expression,
    Frame,
    FreeAxis,
    Rotation,
    Scalar,
    Symbol,
    UndefinedValueError,
)


class Base(Frame, notation="0"):
    """Fixed frame."""


class Link1(Frame):
    """Frame of the first link."""


class Tool(Frame):
    """Frame of the tool."""


class Prime(Frame, notation="'"):
    """Frame after the first elementary rotation."""


class Second(Frame, notation="''"):
    """Frame after the second elementary rotation."""


@dataclass(frozen=True)
class Joint:
    """One joint value."""

    q: float


q = Symbol("q")
q1 = Symbol("q1")
pi = Scalar.pi()
INDICES = (1, 2, 3)
SAMPLES = (-math.pi + 1e-6, -1.2, -1e-6, 0.0, 1e-6, 0.7, math.pi / 2, math.pi - 1e-6)


def entries[K: (Scalar, Expression), A: Frame, B: Frame](
    rotation: Rotation[K, A, B],
) -> list[list[str]]:
    return [[str(rotation.entry(row, column)) for column in INDICES] for row in INDICES]


def numbers(rotation: Rotation[Expression, Base, Tool], value: float) -> np.ndarray:
    return np.asarray(rotation.compile(Joint)(Joint(q=value)).components)


def test_elementary_rotation() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    assert_type(rotation, Rotation[Expression, Base, Link1])
    assert entries(rotation) == [
        ["cos(q1)", "-sin(q1)", "0"],
        ["sin(q1)", "cos(q1)", "0"],
        ["0", "0", "1"],
    ]
    assert rotation.target is Base
    assert rotation.source is Link1


def test_reading_r12() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    assert_type(rotation.entry(1, 2), Expression)
    assert str(rotation.entry(1, 2)) == "-sin(q1)"


def test_rotation_about_an_axis() -> None:
    axis = Base.direction(0, 0, 1)
    about_axis = Rotation.about_axis(axis, q, source=Tool)
    assert_type(about_axis, Rotation[Expression, Base, Tool])
    assert entries(about_axis) == entries(Rotation.about_z(q, target=Base, source=Tool))


def test_identity() -> None:
    identity = Rotation.identity(target=Base, source=Tool)
    assert entries(identity) == [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]


def test_chaining() -> None:
    base_link1 = Rotation.about_z(q1, target=Base, source=Link1)
    link1_tool = Rotation.about_x(q, target=Link1, source=Tool)
    assert_type(base_link1 @ link1_tool, Rotation[Expression, Base, Tool])


def test_chain_through_intermediate_frames() -> None:
    first = Rotation.about_z(q1, target=Base, source=Prime)
    second = Rotation.about_y(q, target=Prime, source=Second)
    third = Rotation.about_x(q1, target=Second, source=Tool)
    assert_type(first @ second @ third, Rotation[Expression, Base, Tool])


def test_inverse_composes_to_identity() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1) @ Rotation.about_x(
        q, target=Link1, source=Tool
    )
    assert_type(rotation.inverse(), Rotation[Expression, Tool, Base])
    assert entries((rotation @ rotation.inverse()).simplify()) == [
        ["1", "0", "0"],
        ["0", "1", "0"],
        ["0", "0", "1"],
    ]


def test_resolving_in_the_target_frame() -> None:
    rotation = Rotation.about_z(pi / 2, target=Base, source=Tool)
    resolved = rotation @ Tool.displacement(1, 0, 0)
    assert_type(resolved, Displacement[Scalar, Base])
    assert repr(resolved) == "Displacement[Base](0, 1, 0)"


def test_regular_case() -> None:
    solutions = Rotation.about_z(pi / 3, target=Base, source=Tool).to_axis_angle()
    assert_type(solutions, AxisAngleSolutions[Scalar, Base])
    assert_type(solutions.first.axis, Direction[Scalar, Base])
    assert repr(solutions.first.axis) == "Direction[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "pi/3"
    assert repr(solutions.second.axis) == "Direction[Base](0, 0, -1)"
    assert str(solutions.second.angle) == "-pi/3"


def test_symbolic_rotation() -> None:
    solutions = Rotation.about_z(q, target=Base, source=Tool).to_axis_angle()
    assert repr(solutions.first.axis) == "Direction[Base](0, 0, sin(q)/Abs(sin(q)))"
    assert str(solutions.first.angle) == "atan2(Abs(sin(q)), cos(q))"
    assert repr(solutions.second.axis) == "Direction[Base](0, 0, -sin(q)/Abs(sin(q)))"
    assert "valid" not in solutions.latex()


def test_symbolic_product_that_is_the_identity() -> None:
    rotation = Rotation.about_x(q, target=Base, source=Tool)
    product = rotation @ rotation.inverse()
    assert_type(product, Rotation[Expression, Base, Base])
    solutions = product.to_axis_angle()
    assert_type(solutions, AxisAngleSolutions[Expression, Base])
    assert not isinstance(solutions.first.axis, FreeAxis)


def test_evaluating_first_gives_the_exact_case() -> None:
    rotation = Rotation.about_x(q, target=Base, source=Tool)
    solutions = (rotation @ rotation.inverse()).evaluate({"q": 1}).to_axis_angle()
    assert repr(solutions.first.axis) == "FreeAxis[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "0"
    assert repr(solutions.second.axis) == "FreeAxis[Base](0, 0, -1)"


def test_the_users_monster_is_the_identity_once_evaluated() -> None:
    rz = Rotation.about_z(q, target=Base, source=Tool)
    rx = Rotation.about_x(q, target=Tool, source=Base)
    monster = rz @ rx @ rz @ rx @ rx.inverse() @ rz.inverse() @ rx.inverse()
    solutions = (monster @ rz.inverse()).evaluate({"q": 1}).to_axis_angle()
    assert repr(solutions.first.axis) == "FreeAxis[Base](0, 0, 1)"


def test_half_turn_about_x() -> None:
    solutions = Rotation.about_x(pi, target=Base, source=Tool).to_axis_angle()
    assert repr(solutions.first.axis) == "Direction[Base](1, 0, 0)"
    assert str(solutions.first.angle) == "pi"
    assert repr(solutions.second.axis) == "Direction[Base](-1, 0, 0)"
    assert str(solutions.second.angle) == "pi"


def test_half_turn_about_a_symbolic_axis_does_not_raise() -> None:
    a = Symbol("a")
    rotation = Rotation.about_axis(Base.direction(a, 1, 0), pi, source=Tool)
    assert_type(rotation.to_axis_angle(), AxisAngleSolutions[Expression, Base])


def test_constant_half_turn_about_z() -> None:
    solutions = Rotation.about_z(pi, target=Base, source=Tool).to_axis_angle()
    assert repr(solutions.first.axis) == "Direction[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "pi"
    assert repr(solutions.second.axis) == "Direction[Base](0, 0, -1)"
    assert str(solutions.second.angle) == "pi"


def test_default_free_axis() -> None:
    solutions = Rotation.identity(target=Base, source=Tool).to_axis_angle()
    assert isinstance(solutions.first.axis, FreeAxis)
    assert repr(solutions.first.axis) == "FreeAxis[Base](0, 0, 1)"
    assert str(solutions.first.angle) == "0"
    assert repr(solutions.second.axis) == "FreeAxis[Base](0, 0, -1)"
    assert str(solutions.second.angle) == "0"


def test_chosen_free_axis() -> None:
    x_axis = Base.direction(1, 0, 0)
    solutions = Rotation.identity(target=Base, source=Tool).to_axis_angle(
        free_axis=x_axis
    )
    assert repr(solutions.first.axis) == "FreeAxis[Base](1, 0, 0)"
    assert repr(solutions.second.axis) == "FreeAxis[Base](-1, 0, 0)"


def test_a_rotation_rejects_a_point_at_runtime() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    with pytest.raises(TypeError, match="@"):
        operator.matmul(rotation, Link1.position(1, 0, 0))


def test_simplify_keeps_frames() -> None:
    rotation = Rotation.about_z(q, target=Base, source=Tool)
    assert_type(rotation.simplify(), Rotation[Expression, Base, Tool])


def test_latex_names_both_frames() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    assert rotation.latex().startswith(r"{}^{0}R_{Link1} = \begin{bmatrix}")


@pytest.mark.parametrize("value", SAMPLES)
def test_elementary_matrices_match_the_oracle(value: float) -> None:
    for build, oracle in (
        (Rotation.about_x, SO3.Rx),
        (Rotation.about_y, SO3.Ry),
        (Rotation.about_z, SO3.Rz),
    ):
        rotation = build(q, target=Base, source=Tool)
        np.testing.assert_allclose(
            numbers(rotation, value), oracle(value).A, atol=1e-12
        )


@pytest.mark.parametrize("value", SAMPLES)
def test_axis_rotation_matches_the_oracle(value: float) -> None:
    axis = Base.direction(1, 2, 3)
    rotation = Rotation.about_axis(axis, q, source=Tool)
    oracle = SO3.AngleAxis(value, [1, 2, 3]).A
    np.testing.assert_allclose(numbers(rotation, value), oracle, atol=1e-12)


@pytest.mark.parametrize("value", [1e-6, 1e-3, 0.5, math.pi - 1e-3, math.pi - 1e-6])
def test_axis_angle_matches_the_oracle(value: float) -> None:
    rotation = Rotation.about_axis(Base.direction(1, 2, 3), q, source=Tool)
    first = rotation.to_axis_angle().first
    axis = np.asarray(first.axis.compile(Joint)(Joint(q=value)).components)
    angle = float(first.angle.compile(Joint)(Joint(q=value)).components)
    expected = numbers(rotation, value)
    np.testing.assert_allclose(SO3.AngleAxis(angle, axis).A, expected, atol=1e-9)
    oracle_angle, _ = SO3(expected).angvec()
    assert angle == pytest.approx(oracle_angle, abs=1e-3)
    assert angle == pytest.approx(value, abs=1e-9)


def test_evaluating_at_a_singular_configuration_raises() -> None:
    solutions = Rotation.about_z(q, target=Base, source=Link1).to_axis_angle()
    with pytest.raises(UndefinedValueError):
        solutions.evaluate({"q": 0})


def test_an_evaluated_axis_is_not_normalized_again() -> None:
    q2 = Symbol("q2")
    rotation = Rotation.about_z(q1, target=Base, source=Link1) @ Rotation.about_x(
        q2, target=Link1, source=Tool
    )
    solutions = rotation.to_axis_angle()
    values = {"q1": 1, "q2": 1}
    evaluated = solutions.evaluate(values).first.axis
    assert_type(evaluated, Direction[Scalar, Base])
    axis = solutions.first.axis
    components = (axis.x, axis.y, axis.z)
    listed = ", ".join(repr(component.evaluate(values)) for component in components)
    assert repr(evaluated) == f"Direction[Base]({listed})"
