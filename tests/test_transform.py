"""Scenarios of the `transform` capability, with spatialmath as the oracle."""

import copy
import math
import operator
from dataclasses import dataclass
from typing import assert_type

import numpy as np
import pytest
from spatialmath import SE3

from robotic import (
    Displacement,
    Expression,
    Frame,
    FreeAxis,
    Position,
    Rotation,
    Scalar,
    Symbol,
    Transform,
)


class Base(Frame, notation="0"):
    """Fixed frame."""


class Link1(Frame):
    """Frame of the first link."""


class Tool(Frame):
    """Frame of the second link."""


@dataclass(frozen=True)
class Joints:
    """Joint values and link lengths of a planar 2R arm."""

    q1: float
    q2: float
    l1: float
    l2: float


q1, q2 = Symbol("q1"), Symbol("q2")
l1, l2, d1 = Symbol("l1"), Symbol("l2"), Symbol("d1")
INDICES = (1, 2, 3, 4)


def entries[K: (Scalar, Expression), A: Frame, B: Frame](
    transform: Transform[K, A, B],
) -> list[list[str]]:
    return [
        [str(transform.entry(row, column)) for column in INDICES] for row in INDICES
    ]


def planar_arm() -> Transform[Expression, Base, Tool]:
    base_link1 = Rotation.about_z(q1, target=Base, source=Link1).homogeneous()
    link1_tool = Rotation.about_z(q2, target=Link1, source=Tool).homogeneous(
        Link1.position(l1, 0, 0)
    )
    return base_link1 @ link1_tool


def base_link1_lifted() -> Transform[Expression, Base, Link1]:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    return rotation.homogeneous(Base.position(0, 0, d1))


def test_matrix_layout() -> None:
    assert entries(base_link1_lifted()) == [
        ["cos(q1)", "-sin(q1)", "0", "0"],
        ["sin(q1)", "cos(q1)", "0", "0"],
        ["0", "0", "1", "d1"],
        ["0", "0", "0", "1"],
    ]


def test_reading_the_translation_from_the_entries() -> None:
    transform = base_link1_lifted()
    assert_type(transform.entry(3, 4), Expression)
    assert str(transform.entry(3, 4)) == "d1"


def test_rotation_and_translation_are_its_parts() -> None:
    transform = base_link1_lifted()
    assert_type(transform.rotation, Rotation[Expression, Base, Link1])
    assert_type(transform.translation, Position[Expression, Base])
    assert repr(transform.translation) == "Position[Base](0, 0, d1)"


def test_coincident_origins() -> None:
    transform = Rotation.identity(target=Base, source=Link1).homogeneous()
    assert_type(transform, Transform[Scalar, Base, Link1])
    assert entries(transform) == [
        ["1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "1"],
    ]


def test_taking_the_homogeneous_matrix_leaves_the_rotation_unchanged() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Link1)
    transform = rotation.homogeneous(Base.position(0, 0, d1))
    assert transform.rotation is rotation
    assert str(rotation.entry(3, 3)) == "1"


def test_forward_kinematics_chain() -> None:
    assert_type(planar_arm(), Transform[Expression, Base, Tool])


def test_inverse_composes_to_identity() -> None:
    transform = planar_arm()
    assert_type(transform.inverse(), Transform[Expression, Tool, Base])
    assert entries((transform @ transform.inverse()).simplify()) == [
        ["1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "1"],
    ]


def test_tip_of_a_planar_2r_arm() -> None:
    tip = (planar_arm() @ Tool.position(l2, 0, 0)).simplify()
    assert_type(tip, Position[Expression, Base])
    assert repr(tip) == (
        "Position[Base](l1*cos(q1) + l2*cos(q1 + q2), l1*sin(q1) + l2*sin(q1 + q2), 0)"
    )


def test_translation_does_not_move_a_displacement() -> None:
    transform = Rotation.identity(target=Base, source=Tool).homogeneous(
        Base.position(5, 0, 0)
    )
    moved = transform @ Tool.displacement(1, 0, 0)
    assert_type(moved, Displacement[Scalar, Base])
    assert repr(moved) == "Displacement[Base](1, 0, 0)"
    assert repr(transform @ Tool.position(1, 0, 0)) == "Position[Base](6, 0, 0)"


def test_a_transform_rejects_another_operand_at_runtime() -> None:
    with pytest.raises(TypeError, match="@"):
        operator.matmul(
            base_link1_lifted(), Rotation.identity(target=Link1, source=Tool)
        )


def test_a_point_in_the_wrong_frame_names_the_transform_and_the_point() -> None:
    with pytest.raises(
        TypeError, match=r"apply Transform\[Base, Link1\] to Position\[Tool\]"
    ):
        operator.matmul(base_link1_lifted(), Tool.position(1, 0, 0))


def test_a_symbolic_translation_makes_a_symbolic_transform() -> None:
    identity = Rotation.identity(target=Base, source=Link1)
    assert_type(identity, Rotation[Scalar, Base, Link1])
    transform = identity.homogeneous(Base.position(0, 0, d1))
    assert_type(transform, Transform[Expression, Base, Link1])
    assert_type(transform.rotation, Rotation[Expression, Base, Link1])
    assert isinstance(transform.rotation.entry(1, 1), Expression)


def test_simplify_keeps_frames() -> None:
    assert_type(planar_arm().simplify(), Transform[Expression, Base, Tool])


def test_latex_names_both_frames() -> None:
    assert planar_arm().latex().startswith(r"{}^{0}T_{Tool} = \begin{bmatrix}")


@pytest.mark.parametrize(
    ("first", "second"),
    [(0.0, 0.0), (0.3, -1.2), (math.pi / 2, math.pi), (-math.pi + 1e-6, 1e-6)],
)
def test_chain_matches_the_oracle(first: float, second: float) -> None:
    compiled = planar_arm().compile(Joints)
    numbers = compiled(Joints(q1=first, q2=second, l1=1.0, l2=0.5)).components
    oracle = SE3.Rz(first) * SE3.Rz(second, t=[1.0, 0, 0])
    np.testing.assert_allclose(np.asarray(numbers), oracle.A, atol=1e-12)
    inverse = planar_arm().inverse().compile(Joints)
    inverse_numbers = inverse(Joints(q1=first, q2=second, l1=1.0, l2=0.5)).components
    np.testing.assert_allclose(np.asarray(inverse_numbers), oracle.inv().A, atol=1e-12)


def test_values_copy() -> None:
    q1 = Symbol("q1")
    rotation = Rotation.about_z(q1, target=Base, source=Tool)
    values = (
        Base.position(1, 2, 3),
        Base.displacement(q1, 0, 1),
        Base.direction(1, 1, 0),
        FreeAxis(Base.displacement(0, 0, 1)),
        rotation,
        rotation.homogeneous(Base.position(1, 0, 0)),
        rotation.to_axis_angle(),
    )
    for value in values:
        assert repr(copy.copy(value)) == repr(value)
        assert repr(copy.deepcopy(value)) == repr(value)
