"""Scenarios of the `transform` capability, with spatialmath as the oracle."""

import math
from dataclasses import dataclass
from typing import assert_type

import numpy as np
import pytest
from spatialmath import SE3

from robotic import Frame, Point, Rotation, Scalar, Transform, Vector


class Base(Frame, latex="0"):
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


q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
l1, l2, d1 = Scalar.symbol("l1"), Scalar.symbol("l2"), Scalar.symbol("d1")
INDICES = (1, 2, 3, 4)


def entries[A: Frame, B: Frame](transform: Transform[A, B]) -> list[list[str]]:
    return [
        [str(transform.entry(row, column)) for column in INDICES] for row in INDICES
    ]


def planar_arm() -> Transform[Base, Tool]:
    base_link1 = Rotation.about_z(q1, of=Link1, relative_to=Base).with_origin(
        Base.origin()
    )
    link1_tool = Rotation.about_z(q2, of=Tool, relative_to=Link1).with_origin(
        Link1.point(l1, 0, 0)
    )
    return base_link1 @ link1_tool


def test_matrix_layout() -> None:
    transform = Transform(Rotation.about_z(q1, of=Link1, relative_to=Base), 0, 0, d1)
    assert entries(transform) == [
        ["cos(q1)", "-sin(q1)", "0", "0"],
        ["sin(q1)", "cos(q1)", "0", "0"],
        ["0", "0", "1", "d1"],
        ["0", "0", "0", "1"],
    ]


def test_reading_the_translation_from_the_entries() -> None:
    transform = Transform(Rotation.about_z(q1, of=Link1, relative_to=Base), 0, 0, d1)
    assert_type(transform.entry(3, 4), Scalar)
    assert str(transform.entry(3, 4)) == "d1"


def test_building_from_a_rotation_and_three_numbers() -> None:
    transform = Transform(Rotation.about_z(q1, of=Link1, relative_to=Base), 0, 0, d1)
    assert_type(transform, Transform[Base, Link1])
    assert_type(transform.origin, Point[Base])
    assert repr(transform.origin) == "Point[Base](0, 0, d1)"
    assert_type(transform.rotation, Rotation[Base, Link1])


def test_building_from_a_rotation_and_an_origin_point() -> None:
    rotation = Rotation.about_z(q1, of=Link1, relative_to=Base)
    transform = rotation.with_origin(Base.point(0, 0, d1))
    assert_type(transform, Transform[Base, Link1])
    assert transform.rotation is rotation
    assert repr(transform.origin) == "Point[Base](0, 0, d1)"


def test_forward_kinematics_chain() -> None:
    assert_type(planar_arm(), Transform[Base, Tool])


def test_inverse_composes_to_identity() -> None:
    transform = planar_arm()
    assert_type(transform.inverse(), Transform[Tool, Base])
    assert entries((transform @ transform.inverse()).simplify()) == [
        ["1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "1"],
    ]


def test_tip_of_a_planar_2r_arm() -> None:
    tip = (planar_arm() @ Tool.point(l2, 0, 0)).simplify()
    assert_type(tip, Point[Base])
    assert repr(tip) == (
        "Point[Base](l1*cos(q1) + l2*cos(q1 + q2), l1*sin(q1) + l2*sin(q1 + q2), 0)"
    )


def test_translation_does_not_move_a_vector() -> None:
    transform = Transform(Rotation.identity(of=Tool, relative_to=Base), 5, 0, 0)
    moved = transform @ Tool.vector(1, 0, 0)
    assert_type(moved, Vector[Base])
    assert repr(moved) == "Vector[Base](1, 0, 0)"
    assert repr(transform @ Tool.point(1, 0, 0)) == "Point[Base](6, 0, 0)"


def test_simplify_keeps_frames() -> None:
    assert_type(planar_arm().simplify(), Transform[Base, Tool])


def test_latex_names_both_frames() -> None:
    assert planar_arm().latex().startswith(r"{}^{0}T_{Tool} = \begin{bmatrix}")


@pytest.mark.parametrize(
    ("first", "second"),
    [(0.0, 0.0), (0.3, -1.2), (math.pi / 2, math.pi), (-math.pi + 1e-6, 1e-6)],
)
def test_chain_matches_the_oracle(first: float, second: float) -> None:
    compiled = planar_arm().lambdify(Joints)
    numbers = compiled(Joints(q1=first, q2=second, l1=1.0, l2=0.5)).components
    oracle = SE3.Rz(first) * SE3.Rz(second, t=[1.0, 0, 0])
    np.testing.assert_allclose(np.asarray(numbers), oracle.A, atol=1e-12)
    inverse = planar_arm().inverse().lambdify(Joints)
    inverse_numbers = inverse(Joints(q1=first, q2=second, l1=1.0, l2=0.5)).components
    np.testing.assert_allclose(np.asarray(inverse_numbers), oracle.inv().A, atol=1e-12)
