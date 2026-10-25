"""Scenarios of the `compilation` capability."""

import dataclasses
import math
from dataclasses import dataclass
from typing import assert_type

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy
from hypothesis import example, given
from hypothesis import strategies as st

from robotic import (
    AxisAngleSolutions,
    Compiled,
    Frame,
    MissingParameterError,
    Numeric,
    ParameterShapeError,
    ParameterTypeError,
    Point,
    Rotation,
    Scalar,
    Transform,
    ValidityCondition,
    Vector,
)
from tests.stringified import Stringified


class Base(Frame, notation="0"):
    """Fixed frame."""


class Link1(Frame):
    """Frame of the first link."""


class Tool(Frame):
    """Frame of the second link."""


@dataclass(frozen=True)
class Arm:
    """Joint values and link lengths of a planar 2R arm, plus an unused field."""

    q1: float
    q2: float
    l1: float
    l2: float
    unused: float


@dataclass(frozen=True)
class Joint:
    """One joint value."""

    q1: float


@dataclass(frozen=True)
class Batch:
    """One joint value, batchable under `jax.vmap`."""

    q1: jax.typing.ArrayLike


@dataclass(frozen=True)
class Shortened:
    """A planar 2R arm missing the second link length."""

    q1: float
    q2: float
    l1: float


@dataclass(frozen=True)
class Misannotated:
    """A joint value annotated as text."""

    q1: str


q1, q2 = Scalar.symbol("q1"), Scalar.symbol("q2")
l1, l2 = Scalar.symbol("l1"), Scalar.symbol("l2")
ARM = Arm(q1=0.5, q2=0.1, l1=1.0, l2=0.5, unused=7.0)
ANGLES = st.floats(min_value=-math.pi, max_value=math.pi)
GRADIENT_AT_HALF = 0.8775825618903728


def planar_arm() -> Transform[Base, Tool]:
    base_link1 = Transform(target=Base, source=Link1).with_rotation(
        Rotation.about_z(q1, target=Base, source=Link1)
    )
    link1_tool = (
        Transform(target=Link1, source=Tool)
        .with_rotation(Rotation.about_z(q2, target=Link1, source=Tool))
        .with_translation(Link1.vector(l1, 0, 0))
    )
    return base_link1 @ link1_tool


def tip() -> Point[Base]:
    return (planar_arm() @ Tool.point(l2, 0, 0)).simplify()


def substituted(scalar: Scalar, values: dict[str, float]) -> float:
    expression = sympy.sympify(str(scalar))
    bindings: dict[sympy.Basic | complex, sympy.Basic | complex] = {
        sympy.Symbol(name): value for name, value in values.items()
    }
    return float(sympy.Float(sympy.N(expression.subs(bindings))))


def test_one_parameters_class_for_the_whole_robot() -> None:
    compiled = tip().lambdify(Arm)
    assert_type(compiled, Compiled[Arm, Point[Base]])
    expected = [
        1.0 * math.cos(0.5) + 0.5 * math.cos(0.6),
        math.sin(0.5) + 0.5 * math.sin(0.6),
        0,
    ]
    np.testing.assert_allclose(
        np.asarray(compiled(ARM).components), expected, atol=1e-15
    )


def test_missing_field() -> None:
    with pytest.raises(MissingParameterError, match="l2"):
        tip().lambdify(Shortened)


def test_field_not_a_float() -> None:
    with pytest.raises(ParameterTypeError, match="q1"):
        q1.lambdify(Misannotated)


def test_string_annotation() -> None:
    assert isinstance(dataclasses.fields(Stringified)[0].type, str)
    result = q1.cos().lambdify(Stringified)(Stringified(q1=0.0))
    assert float(result.components) == pytest.approx(1.0)


def test_natural_shapes_in_double_precision() -> None:
    rotation = planar_arm().rotation
    shapes = {
        "scalar": (q1 * l1).lambdify(Arm)(ARM).components,
        "point": tip().lambdify(Arm)(ARM).components,
        "vector": (planar_arm() @ Tool.vector(1, 0, 0)).lambdify(Arm)(ARM).components,
        "rotation": rotation.lambdify(Arm)(ARM).components,
        "transform": planar_arm().lambdify(Arm)(ARM).components,
    }
    assert {name: array.shape for name, array in shapes.items()} == {
        "scalar": (),
        "point": (3,),
        "vector": (3,),
        "rotation": (3, 3),
        "transform": (4, 4),
    }
    assert {str(array.dtype) for array in shapes.values()} == {"float64"}


def test_compiled_point_keeps_its_frame() -> None:
    result = tip().lambdify(Arm)(ARM)
    assert_type(result, Numeric[Point[Base]])
    assert_type(result.components, jax.Array)
    assert repr(result).startswith("Numeric[Point[Base]]([")


def test_constants_compile() -> None:
    result = Base.vector(1, 0, Scalar.pi()).lambdify(Joint)(Joint(q1=0.0))
    np.testing.assert_allclose(np.asarray(result.components), [1.0, 0.0, math.pi])


def sine_vector() -> Compiled[Batch, Vector[Base]]:
    return Base.vector(q1.sin(), 0, 0).lambdify(Batch)


def sine_x(value: jax.typing.ArrayLike) -> jax.Array:
    return sine_vector()(Batch(q1=value)).components[0]


def sine_at(value: jax.typing.ArrayLike) -> Numeric[Vector[Base]]:
    return sine_vector()(Batch(q1=value))


def test_jit() -> None:
    assert float(jax.jit(sine_x)(0.5)) == pytest.approx(math.sin(0.5))
    assert repr(jax.jit(sine_at)(0.5)).startswith("Numeric[Vector[Base]]")


def test_batched_configurations() -> None:
    batched = jax.vmap(sine_at)(jnp.array([0.1, 0.2]))
    assert batched.components.shape == (2, 3)
    assert repr(batched).startswith("Numeric[Vector[Base]]")


def test_printing_under_tracing() -> None:
    def printed(value: jax.typing.ArrayLike) -> jax.Array:
        result = sine_vector()(Batch(q1=value))
        assert repr(result).startswith("Numeric[Vector[Base]](")
        return result.components

    assert jax.jit(printed)(0.5).shape == (3,)


def test_array_passed_directly() -> None:
    compiled = Rotation.about_z(q1, target=Base, source=Tool).lambdify(Batch)
    with pytest.raises(ParameterShapeError, match=r"q1.*jax\.vmap"):
        compiled(Batch(q1=jnp.array([0.1, 0.2])))


def test_array_batched_with_vmap() -> None:
    compiled = Rotation.about_z(q1, target=Base, source=Tool).lambdify(Batch)

    def at(value: jax.typing.ArrayLike) -> Numeric[Rotation[Base, Tool]]:
        return compiled(Batch(q1=value))

    assert jax.vmap(at)(jnp.array([0.1, 0.2])).components.shape == (2, 3, 3)


def test_gradient() -> None:
    assert float(jax.grad(sine_x)(0.5)) == pytest.approx(GRADIENT_AT_HALF, rel=1e-15)


def solutions_at(value: float) -> Numeric[AxisAngleSolutions[Base]]:
    rotation = Rotation.about_z(q1, target=Base, source=Tool)
    return rotation.to_axis_angle().lambdify(Joint)(Joint(q1=value))


@pytest.mark.parametrize(
    ("value", "valid"),
    [(0.5, True), (1e-6, True), (0.0, False), (1e-12, False), (math.pi, False)],
)
def test_axis_angle_solutions_compile_with_their_flag(
    value: float, *, valid: bool
) -> None:
    result = solutions_at(value)
    assert result.components.shape == (2, 4)
    assert result.valid().shape == ()
    assert bool(result.valid()) is valid


def test_regular_configuration() -> None:
    result = solutions_at(0.5)
    np.testing.assert_allclose(np.asarray(result.components[0]), [0, 0, 1, 0.5])
    np.testing.assert_allclose(np.asarray(result.components[1]), [0, 0, -1, -0.5])


def test_validity_condition_compiles_to_a_flag() -> None:
    condition = (
        Rotation.about_z(q1, target=Base, source=Tool).to_axis_angle().valid_when
    )
    compiled = condition.lambdify(Joint)
    assert_type(compiled, Compiled[Joint, ValidityCondition])
    flags = [
        bool(compiled(Joint(q1=value)).components) for value in (0.5, 1e-12, math.pi)
    ]
    assert flags == [True, False, False]


def test_flag_survives_vmap() -> None:
    rotation = Rotation.about_z(q1, target=Base, source=Tool)
    compiled = rotation.to_axis_angle().lambdify(Batch)

    def at(value: jax.typing.ArrayLike) -> Numeric[AxisAngleSolutions[Base]]:
        return compiled(Batch(q1=value))

    batched = jax.vmap(at)(jnp.array([0.5, 0.0]))
    assert batched.components.shape == (2, 2, 4)
    assert [bool(flag) for flag in batched.valid()] == [True, False]


@given(ANGLES, ANGLES)
@example(0.0, 0.0)
@example(math.pi, -math.pi)
@example(1e-12, math.pi - 1e-12)
def test_compiled_transform_matches_the_substituted_one(
    first: float, second: float
) -> None:
    transform = planar_arm()
    values = {"q1": first, "q2": second, "l1": 1.0, "l2": 0.5}
    compiled = transform.lambdify(Arm)(Arm(**values, unused=0.0)).components
    expected = [
        [substituted(transform.entry(row, column), values) for column in (1, 2, 3, 4)]
        for row in (1, 2, 3, 4)
    ]
    np.testing.assert_allclose(np.asarray(compiled), expected, atol=1e-12)


@given(ANGLES)
@example(0.0)
@example(1e-12)
@example(math.pi)
def test_compiled_axis_matches_the_substituted_one(angle: float) -> None:
    rotation = Rotation.about_axis(Base.vector(1, 2, 3).normalize(), q1, source=Tool)
    first = rotation.to_axis_angle().first
    compiled = float(first.angle.lambdify(Joint)(Joint(q1=angle)).components)
    expected = substituted(first.angle, {"q1": angle})
    assert compiled == pytest.approx(expected, abs=1e-9)
