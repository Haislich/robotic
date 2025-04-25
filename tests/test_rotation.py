import pytest
import sympy

from robotic.transformations import (
    Axis,
    AxisAngleSpec,
    Rotation,
    SymbolicBranch,
    SymbolicConditional,
)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="A rotation matrix is a square matrix"):
        Rotation(sympy.Matrix([[1, 0], [0, 1]]))


def test_non_orthonormal_raises():
    bad_matrix = sympy.Matrix([[1, 2, 3], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match=r"R\.T \* R ≠ I"):
        Rotation(bad_matrix)


def test_rotation_from_numeric_axis_angle():
    axis = Axis(1, 0, 0)
    theta = sympy.pi
    spec = AxisAngleSpec(axis, theta)
    R = Rotation.from_axis_angle(spec)

    expected = sympy.Matrix(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    assert sympy.simplify(R - expected) == sympy.zeros(3)


def test_rotation_identity():
    R = Rotation.identity()
    assert R.shape == (3, 3)
    assert R == sympy.eye(3)


def test_rotation_to_axis_angle_theta_zero_branch():
    R = Rotation(sympy.eye(3))
    result = R.to_axis_angle()
    assert isinstance(result, AxisAngleSpec)
    assert result.theta == 0
    assert result.axis == Axis(1, 0, 0)


def test_rotation_to_axis_angle_theta_pi_branches():
    R = Rotation(
        sympy.Matrix(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )
    )
    result = R.to_axis_angle()
    assert isinstance(result, tuple)
    a1, a2 = result
    assert isinstance(a1, AxisAngleSpec)
    assert a1.theta == sympy.pi
    assert isinstance(a2, AxisAngleSpec)
    assert a2.theta == sympy.pi
    assert a1.axis == -a2.axis


def test_rotation_from_symbolic_axis_angle():
    theta = sympy.symbols("theta")
    axis = Axis(0, 0, 1)
    spec = AxisAngleSpec(axis, theta)
    R = Rotation.from_axis_angle(spec)

    # Should be a rotation around Z
    expected = sympy.Matrix(
        [
            [sympy.cos(theta), -sympy.sin(theta), 0],
            [sympy.sin(theta), sympy.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    diff = sympy.simplify(R - expected)
    assert diff == sympy.zeros(3)


def test_symbolic_axis_angle_resolution():
    x = sympy.symbols("x")
    cond = sympy.Symbol("cond")
    conditional = SymbolicConditional(
        [
            SymbolicBranch(AxisAngleSpec(Axis(1, 0, 0), x), cond),
            SymbolicBranch(AxisAngleSpec(Axis(0, 1, 0), x), sympy.Not(cond)),
        ]
    )
    out = Rotation.from_axis_angle(conditional)
    assert isinstance(out, SymbolicConditional)
    assert len(out.branches) == 2
    assert all(isinstance(b.value, Rotation) for b in out)


def test_rotation_from_symbolic_conditional():
    theta = sympy.symbols("theta")
    cond = sympy.Symbol("cond")

    axis1 = Axis(1, 0, 0)
    axis2 = Axis(0, 1, 0)
    cond_spec = SymbolicConditional(
        [
            SymbolicBranch(AxisAngleSpec(axis1, theta), cond),
            SymbolicBranch(AxisAngleSpec(axis2, theta), ~cond),
        ]
    )
    result = Rotation.from_axis_angle(cond_spec)

    assert isinstance(result, SymbolicConditional)
    assert all(isinstance(branch.value, Rotation) for branch in result)


def test_rotation_rounding_remains_rotation():
    theta = sympy.pi / 3
    R = Rotation.from_axis_angle(AxisAngleSpec(Axis(1, 0, 0), theta))
    R_rounded = R.round(4)
    assert isinstance(R_rounded, Rotation)
    # Sanity check: matrix is close to itself
    assert all(
        abs(R[i, j] - R_rounded[i, j]) < 1e-3  # type: ignore
        for i in range(3)
        for j in range(3)
    )


def test_rotation_to_axis_angle_numeric():
    R = Rotation(
        sympy.Matrix(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        )
    )
    spec = R.to_axis_angle()
    assert isinstance(spec, tuple)
    assert all(isinstance(s, AxisAngleSpec) for s in spec)
    assert spec[0].theta == sympy.pi


def test_rotation_substitution():
    theta = sympy.symbols("theta")
    R = Rotation.from_axis_angle(AxisAngleSpec(Axis(1, 0, 0), theta))
    R_sub = R.subs(theta, 0)
    assert R_sub == sympy.eye(3)


def test_rotation_matmul_rotation():
    R1 = Rotation.from_axis_angle(AxisAngleSpec(Axis(0, 0, 1), sympy.pi / 2))
    R2 = Rotation.from_axis_angle(AxisAngleSpec(Axis(0, 0, 1), sympy.pi / 2))
    R3 = R1 @ R2
    assert isinstance(R3, Rotation)
    assert sympy.simplify(
        R3 - Rotation.from_axis_angle(AxisAngleSpec(Axis(0, 0, 1), sympy.pi))
    ) == sympy.zeros(3)


def test_rotation_setitem_clears_cache():
    R = Rotation.identity()
    _ = R.to_axis_angle()
    _ = R.to_euler()

    assert R._axis_angle_spec is not None
    assert R._euler_spec is not None

    R[0, 0] = sympy.cos(sympy.pi)  # Triggers __setitem__

    assert R._axis_angle_spec is None
    assert R._euler_spec is None


def test_rotation_subs_theta_in_axis_angle():
    theta = sympy.symbols("theta")
    R = Rotation.from_axis_angle(AxisAngleSpec(Axis(1, 0, 0), theta))
    assert isinstance(R, Rotation)
    R2 = R.subs(theta, 0)
    assert R2 == sympy.eye(3)


def test_rotation_properties():
    theta = sympy.symbols("theta")
    R = Rotation.from_axis_angle(AxisAngleSpec(Axis(0, 1, 0), theta))
    assert sympy.simplify(R.det() - 1) == 0
    assert sympy.simplify((R.T * R) - sympy.eye(3)) == sympy.zeros(3)
