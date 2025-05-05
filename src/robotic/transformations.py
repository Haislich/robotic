import enum
import sys
from dataclasses import dataclass
from itertools import product
from math import isclose
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import sympy
from loguru import logger

from robotic import Scalar, scalar_repr_latex

# from sympy.matrices.expressions.matexpr import MatrixElement


T = TypeVar("T")
DEBUG = False
logger.remove()
# logger.add(sys.stdout, format="{level} | {message}", colorize=True)
logger.add(sys.stderr, format="{level} | {message}", colorize=True)


@dataclass
class SymbolicBranch(Generic[T]):
    value: T
    condition: sympy.Basic


class SymbolicConditional(Generic[T]):
    def __init__(self, branches: Sequence[SymbolicBranch[T]]):
        self.branches: Sequence[SymbolicBranch[T]] = branches

    def __iter__(self) -> Iterator[SymbolicBranch[T]]:
        return iter(self.branches)

    def __getitem__(self, index: int) -> SymbolicBranch[T]:
        return self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def __str__(self) -> str:
        return "\n".join([str(elem) for elem in self])

    def subs(self: "SymbolicConditional[Rotation]", *args, **kwargs) -> T:
        for branch in self.branches:
            if branch.condition.subs(*args, **kwargs):
                return branch.value.subs(*args, **kwargs)
        raise ValueError("")


class Axis(sympy.Matrix):
    def __new__(cls, x1: Scalar, x2: Scalar, x3: Scalar):
        vec = sympy.Matrix([x1, x2, x3])
        if not vec.is_symbolic():
            norm = vec.norm()
            if not norm.equals(1):
                logger.warning(f"Normalizing axis vector: norm = {norm}")
                vec = vec / norm
        else:
            logger.warning("Skipping normalization: vector is symbolic")
        return sympy.Matrix.__new__(cls, 3, 1, vec)

    def round(self, digits=4) -> "Axis":
        # TODO: Implement rounding os an axis
        raise NotImplementedError
        # x1 = self[0] if self[0].is_symbol() else

    def simplify(self) -> "Axis":
        return Axis(*sympy.simplify(self))

    def skew(self):
        x = sympy.sympify(self[0])
        y = sympy.sympify(self[1])
        z = sympy.sympify(self[2])
        return sympy.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def subs(self, *args, **kwargs) -> "Axis":
        result = super().subs(*args, **kwargs)
        return Axis(*result)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"

    def _repr_latex_(self):
        repr_x = scalar_repr_latex("x", self[0])  # type:ignore
        repr_y = scalar_repr_latex("y", self[1])  # type:ignore
        repr_z = scalar_repr_latex("z", self[2])  # type:ignore
        return f"Axis({repr_x}, {repr_y}, {repr_z})"


X = Axis(1, 0, 0)
Y = Axis(0, 1, 0)
Z = Axis(0, 0, 1)


class AxisAngleSpec:
    axis: Axis
    theta: Scalar

    def __init__(self, axis: Axis, theta: Scalar) -> None:
        self.axis = axis
        self.theta = theta

    def __repr__(self) -> str:
        return str(self)

    def _repr_latex_(self) -> str:
        repr_theta = scalar_repr_latex(r"\theta", self.theta)
        return f"AxisAngleSpec({self.axis._repr_latex_()},{repr_theta})"

    def __str__(self) -> str:
        axis = self.axis
        theta = self.theta
        return f"AxisAngleSpec({axis},{theta}"


class EulerAngles:
    theta1: Scalar
    theta2: Scalar
    theta3: Scalar

    def __init__(self, theta1: Scalar, theta2: Scalar, theta3: Scalar) -> None:
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    @property
    def T(self) -> "EulerAngles":
        return EulerAngles(self.theta3, self.theta2, self.theta1)

    def __iter__(self):
        yield self.theta1
        yield self.theta2
        yield self.theta3

    def __str__(self) -> str:
        return f"EulerAngles({self.theta1}, {self.theta2}, {self.theta3})"

    def _repr_latex_(self) -> str:
        repr_theta1 = scalar_repr_latex(r"\theta_1", self.theta1)
        repr_theta2 = scalar_repr_latex(r"\theta_2", self.theta2)
        repr_theta3 = scalar_repr_latex(r"\theta_3", self.theta3)
        return f"EulerAngles({repr_theta1}, {repr_theta2}, {repr_theta3})"


class EulerSequence(enum.Enum):
    # Tait-Bryan angles (all axes different)
    XYZ = "XYZ"
    XZY = "XZY"
    YXZ = "YXZ"
    YZX = "YZX"
    ZXY = "ZXY"
    ZYX = "ZYX"

    # Proper Euler angles (first and third axes the same)
    ZXZ = "ZXZ"
    XYX = "XYX"
    YZY = "YZY"
    XZX = "XZX"
    ZYZ = "ZYZ"
    YXY = "YXY"

    def __str__(self) -> str:
        return self.value

    def _repr_latex(self) -> str:
        return str(self)


class EulerOrder(enum.Enum):
    MOVING = "MOVING"
    FIXED = "FIXED"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value

    def _repr_latex(self) -> str:
        return str(self)


class EulerSpec:
    euler_angles: EulerAngles
    euler_sequence: EulerSequence
    euler_order: EulerOrder

    def __init__(
        self,
        euler_angles: EulerAngles | Tuple[Scalar, Scalar, Scalar],
        euler_sequence: EulerSequence
        | Literal[
            "XYZ",
            "XZY",
            "YXZ",
            "YZX",
            "ZXY",
            "ZYX",
            "ZXZ",
            "XYX",
            "YZY",
            "XZX",
            "ZYZ",
            "YXY",
        ],
        euler_order: EulerOrder | Literal["MOVING", "FIXED"],
    ) -> None:
        if isinstance(euler_angles, tuple):
            euler_angles = EulerAngles(
                euler_angles[0], euler_angles[1], euler_angles[2]
            )
        self.euler_angles = euler_angles
        if isinstance(euler_sequence, str):
            euler_sequence = EulerSequence(euler_sequence)
        self.euler_sequence = euler_sequence
        if isinstance(euler_order, str):
            euler_order = EulerOrder(euler_order)
        self.euler_order = euler_order

    def __repr__(self) -> str:
        return f"AxisAngleSpec({str(self.euler_angles)}, {str(self.euler_sequence)}, {str(self.euler_order)})"

    def _repr_latex_(self) -> str:
        return f"AxisAngleSpec({self.euler_angles._repr_latex_()}, {self.euler_sequence._repr_latex()}, {self.euler_order._repr_latex()})"


class Rotation(sympy.Matrix):
    _axis_angle_spec: Optional[
        AxisAngleSpec
        | Tuple[AxisAngleSpec, AxisAngleSpec]
        | SymbolicConditional[AxisAngleSpec]
    ] = None
    _euler_spec: Optional[EulerSpec | Tuple[EulerSpec, EulerSpec]] = None

    def __new__(cls, mat: sympy.Matrix, *, tollerance=1e-4, verbose=True):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("A rotation matrix is a square matrix")
        if mat.shape[0] != 3 or mat.shape[1] != 3:
            raise ValueError("A rotation matrix is a square matrix 3x3")

        if not mat.is_symbolic():
            ortho_check = sympy.simplify(mat.T * mat - sympy.eye(3))

            if not all(isclose(val, 0, abs_tol=tollerance) for val in ortho_check):
                raise ValueError(f"R.T * R ≠ I: found {mat}, {mat.T}")

            determinant = sympy.simplify(mat.det())
            det_check = determinant - 1
            if not isclose(det_check, 0, abs_tol=tollerance):
                raise ValueError(f"Det ≠ 1: found {determinant}")
        else:
            logger.warning("Skipping numeric validation: matrix is symbolic")
        cls.tollerance = tollerance
        return super().__new__(cls, mat.cols, mat.rows, mat)

    @property
    def T(self) -> "Rotation":
        return super().T

    def evalf(
        self,
        n: int = 15,
        subs: Any | None = None,
        maxn: int = 100,
        chop: bool = False,
        strict: bool = False,
        quad: Any | None = None,
        verbose: bool = False,
    ) -> "Rotation":
        return Rotation(super().evalf(n, subs, maxn, chop, strict, quad, verbose))

    def simplify(self) -> "Rotation":
        return Rotation(sympy.simplify(self))

    def round(self, digits=4) -> "Rotation":
        if self.is_symbolic():
            return self
        for row in range(self.rows):
            for col in range(self.cols):
                self[row, col] = round(self[row, col], digits)  # type: ignore
        return Rotation(self)

    @staticmethod
    def identity() -> "Rotation":
        return Rotation(
            sympy.Matrix(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        )

    def subs(self, *args, **kwargs):
        return super().subs(*args, **kwargs)

    @overload
    @staticmethod
    def from_axis_angle(
        axis_angle_spec: AxisAngleSpec,
    ) -> "Rotation": ...
    @overload
    @staticmethod
    def from_axis_angle(
        axis_angle_spec: Tuple[AxisAngleSpec, AxisAngleSpec],
    ) -> "Rotation": ...

    @overload
    @staticmethod
    def from_axis_angle(
        axis_angle_spec: Tuple[Axis, Scalar],
    ) -> "Rotation": ...

    @overload
    @staticmethod
    def from_axis_angle(
        axis_angle_spec: SymbolicConditional[AxisAngleSpec],
    ) -> "SymbolicConditional[Rotation]": ...

    @staticmethod
    def from_axis_angle(
        axis_angle_spec: AxisAngleSpec
        | Tuple[Axis, Scalar]
        | Tuple[AxisAngleSpec, AxisAngleSpec]
        | SymbolicConditional[AxisAngleSpec],
    ) -> "Rotation | SymbolicConditional[Rotation]":
        if isinstance(axis_angle_spec, AxisAngleSpec):
            r = axis_angle_spec.axis
            theta = axis_angle_spec.theta
            identity = sympy.eye(3)
            skew = sympy.Matrix(r.skew())
            twist = skew * sympy.sin(theta)
            flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
            return Rotation(identity + twist + flatten)

        elif isinstance(axis_angle_spec, tuple):
            if isinstance(axis_angle_spec[0], Axis) and isinstance(
                axis_angle_spec[1], Scalar
            ):
                axis_angle_spec = AxisAngleSpec(axis_angle_spec[0], axis_angle_spec[1])
            elif isinstance(axis_angle_spec[0], AxisAngleSpec) and isinstance(
                axis_angle_spec[1], AxisAngleSpec
            ):
                axis_angle_spec = axis_angle_spec[0]
            else:
                raise ValueError(f"Invalid tuple: {axis_angle_spec}")
            return Rotation.from_axis_angle(axis_angle_spec)

        elif isinstance(axis_angle_spec, SymbolicConditional):
            ret = []
            for elem in axis_angle_spec:
                ret.append(
                    SymbolicBranch(Rotation.from_axis_angle(elem.value), elem.condition)
                )
            return SymbolicConditional(ret)
        raise TypeError("Axis angle specification not valid")

    def to_axis_angle(
        self,
    ) -> (
        AxisAngleSpec
        | Tuple[AxisAngleSpec, AxisAngleSpec]
        | SymbolicConditional[AxisAngleSpec]
    ):
        if self._axis_angle_spec is not None:
            return self._axis_angle_spec
        theta = sympy.atan2(
            sympy.sqrt(
                (sympy.sympify(self[0, 1]) - sympy.sympify(self[1, 0])) ** 2
                + (sympy.sympify(self[0, 2]) - sympy.sympify(self[2, 0])) ** 2
                + (sympy.sympify(self[1, 2]) - sympy.sympify(self[2, 1])) ** 2
            ),
            (
                sympy.sympify(self[0, 0])
                + sympy.sympify(self[1, 1])
                + sympy.sympify(self[2, 2])
            )
            - 1,
        )
        sin_theta = sympy.sin(theta)
        if self.is_symbolic():
            branches = [
                # Any solution is good,so I chose the X
                SymbolicBranch(AxisAngleSpec(Axis(1, 0, 0), theta), sympy.Eq(theta, 0)),
                SymbolicBranch(
                    AxisAngleSpec(
                        Axis(
                            sympy.sympify(self[2, 1])
                            - sympy.sympify(self[1, 2])
                            / (sympy.sympify(2) * sin_theta),
                            sympy.sympify(self[0, 2])
                            - sympy.sympify(self[2, 0])
                            / (sympy.sympify(2) * sin_theta),
                            sympy.sympify(self[1, 0])
                            - sympy.sympify(self[0, 1])
                            / (sympy.sympify(2) * sin_theta),
                        ),
                        theta,
                    ),
                    sympy.true,
                ),
            ]
            for sx, sy, sz in product([1, -1], repeat=3):
                ax = sympy.sympify(sx) * sympy.sqrt((sympy.sympify(self[0, 0]) + 1) / 2)
                ay = sympy.sympify(sy) * sympy.sqrt((sympy.sympify(self[1, 1]) + 1) / 2)
                az = sympy.sympify(sz) * sympy.sqrt((sympy.sympify(self[2, 2]) + 1) / 2)

                # Each component of the axis must satisfy the off-diagonal conditions
                cond: sympy.Basic = sympy.And(
                    sympy.Eq(theta, sympy.pi),
                    sympy.Eq(self[0, 1], 2 * ax * ay),
                    sympy.Eq(self[0, 2], 2 * ax * az),
                    sympy.Eq(self[1, 2], 2 * ay * az),
                )
                branches.insert(
                    0, SymbolicBranch(AxisAngleSpec(Axis(ax, ay, az), theta), cond)
                )
            self._axis_angle_spec = SymbolicConditional(branches)
        else:
            if (theta == sympy.pi) or (theta == -sympy.pi):
                logger.warning(f"Singular case, theta = {theta}")
                rx = sympy.sqrt((sympy.sympify(self[0, 0]) + 1) / 2)
                ry = sympy.sqrt((sympy.sympify(self[1, 1]) + 1) / 2)
                rz = sympy.sqrt((sympy.sympify(self[2, 2]) + 1) / 2)

                # Try both sign combinations to check consistency with off-diagonal entries

                for sx, sy, sz in product([1, -1], repeat=3):
                    ax = sympy.sympify(sx) * rx
                    ay = sympy.sympify(sy) * ry
                    az = sympy.sympify(sz) * rz

                    # Check sign consistency using off-diagonal terms
                    if (
                        sympy.simplify(ax * ay - sympy.sympify(self[0, 1]) / 2).equals(
                            0
                        )
                        and sympy.simplify(
                            ax * az - sympy.sympify(self[0, 2]) / 2
                        ).equals(0)
                        and sympy.simplify(
                            ay * az - sympy.sympify(self[1, 2]) / 2
                        ).equals(0)
                    ):
                        axis = Axis(ax, ay, az)
                        self._axis_angle_spec = (
                            AxisAngleSpec(axis, theta),
                            AxisAngleSpec(-axis, theta),
                        )

            elif theta == 0:
                logger.warning(f"Singular case, theta = {theta}")
                self._axis_angle_spec = AxisAngleSpec(Axis(1, 0, 0), theta)
            else:
                # TODO: This should yield positive and negative ?
                # See solution midterm 2020-11-20
                logger.info(f"Regular case, theta = {theta.evalf()}")
                axis = Axis(
                    (self[2, 1] - sympy.sympify(self[1, 2]))
                    / (sympy.sympify(2) * sin_theta),
                    (self[0, 2] - sympy.sympify(self[2, 0]))
                    / (sympy.sympify(2) * sin_theta),
                    (self[1, 0] - sympy.sympify(self[0, 1]))
                    / (sympy.sympify(2) * sin_theta),
                )
                self._axis_angle_spec = (
                    AxisAngleSpec(
                        axis,
                        theta,
                    ),
                    AxisAngleSpec(
                        -axis,
                        theta,
                    ),
                )
        if self._axis_angle_spec is None:
            raise ValueError("...")
        return self._axis_angle_spec

    @staticmethod
    def from_euler(
        euler_spec: EulerSpec
        | Tuple[
            EulerAngles | Tuple[Scalar, Scalar, Scalar],
            EulerSequence
            | Literal[
                "XYZ",
                "XZY",
                "YXZ",
                "YZX",
                "ZXY",
                "ZYX",
                "ZXZ",
                "XYX",
                "YZY",
                "XZX",
                "ZYZ",
                "YXY",
            ],
            EulerOrder | Literal["MOVING", "FIXED"],
        ],
    ) -> "Rotation":
        if isinstance(euler_spec, tuple):
            euler_spec = EulerSpec(euler_spec[0], euler_spec[1], euler_spec[2])
        euler_angles: EulerAngles = euler_spec.euler_angles
        euler_sequence = euler_spec.euler_sequence
        euler_order = euler_spec.euler_order

        angle_map = [euler_angles.theta1, euler_angles.theta2, euler_angles.theta3]
        axis_map = {"X": X, "Y": Y, "Z": Z}

        axes = [axis_map[c] for c in euler_sequence.value]
        # Create each elemental rotation
        R1 = Rotation.from_axis_angle(AxisAngleSpec(axes[0], angle_map[0]))
        R2 = Rotation.from_axis_angle(AxisAngleSpec(axes[1], angle_map[1]))
        R3 = Rotation.from_axis_angle(AxisAngleSpec(axes[2], angle_map[2]))
        if DEBUG:
            print(R1)
            print(R2)
            print(R3)

        if euler_order == euler_order.FIXED:
            mat = R3 @ R2 @ R1  # world frame
        else:
            mat = R1 @ R2 @ R3  # body frame

        mat._euler_spec = EulerSpec(euler_angles, euler_sequence, euler_order)
        return mat

    def to_euler(
        self,
        euler_sequence: EulerSequence
        | Literal[
            "XYZ",
            "XZY",
            "YXZ",
            "YZX",
            "ZXY",
            "ZYX",
            "ZXZ",
            "XYX",
            "YZY",
            "XZX",
            "ZYZ",
            "YXY",
        ] = EulerSequence.XYZ,
        euler_order: EulerOrder | Literal["MOVING", "FIXED"] = EulerOrder.MOVING,
    ) -> EulerSpec | Tuple[EulerSpec, EulerSpec]:
        if isinstance(euler_sequence, str):
            euler_sequence = EulerSequence(euler_sequence)
        if isinstance(euler_order, str):
            euler_order = EulerOrder(euler_order)
        if self._euler_spec is not None:
            if (
                isinstance(self._euler_spec, tuple)
                and (self._euler_spec[0].euler_order == euler_order)
                and (self._euler_spec[0].euler_sequence == euler_sequence)
                and (self._euler_spec[1].euler_order == euler_order)
                and (self._euler_spec[1].euler_sequence == euler_sequence)
            ) or (
                isinstance(self._euler_spec, EulerSpec)
                and (self._euler_spec.euler_order == euler_order)
                and (self._euler_spec.euler_sequence == euler_sequence)
            ):
                return self._euler_spec

        logger.info(
            f"Starting Euler angle extraction with euler sequence {euler_sequence} and euler order {euler_order}",
        )
        R = self.round()
        if euler_order == EulerOrder.MOVING:
            R = Rotation(self.T)
        R = cast(Any, R)

        match euler_sequence:
            case EulerSequence.XYZ:
                logger.info(f"Computing theta2 = asin({R[0, 2]})")
                theta2 = sympy.asin(R[0, 2])
                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2
                logger.info(f"Computing theta1 = atan2({-R[1, 2]}, {R[2, 2]})")
                theta1 = sympy.atan2(-R[1, 2], R[2, 2])
                logger.info(f"Computing theta3 = atan2({-R[0, 1]}, {R[0, 0]})")
                theta3 = sympy.atan2(-R[0, 1], R[0, 0])
                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[1, 2]}, {-R[2, 2]})"
                )
                theta1_bis = sympy.atan2(R[1, 2], -R[2, 2])
                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[0, 1]}, {R[0, 0]})"
                )
                theta3_bis = sympy.atan2(R[0, 1], R[0, 0])

            case EulerSequence.XZY:
                logger.info(f"Computing theta2 = -asin({R[0, 1]})")
                theta2 = -sympy.asin(R[0, 1])
                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2

                logger.info(f"Computing theta1 = atan2({R[2, 1]}, {R[1, 1]})")
                theta1 = sympy.atan2(R[2, 1], R[1, 1])

                logger.info(f"Computing theta3 = atan2({R[0, 2]}, {R[0, 0]})")
                theta3 = sympy.atan2(R[0, 2], R[0, 0])

                logger.info(
                    f"Computing alternative theta1_bis = atan2({-R[2, 1]}, {-R[1, 1]})"
                )
                theta1_bis = sympy.atan2(-R[2, 1], -R[1, 1])

                logger.info(
                    f"Computing alternative theta3_bis = atan2({-R[0, 2]}, {R[0, 0]})"
                )
                theta3_bis = sympy.atan2(-R[0, 2], R[0, 0])

            case EulerSequence.YXZ:
                logger.info(f"Computing theta2 = -asin({R[1, 0]})")
                theta2 = -sympy.asin(R[1, 0])

                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2

                logger.info(f"Computing theta1 = atan2({R[2, 0]}, {R[0, 0]})")
                theta1 = sympy.atan2(R[2, 0], R[0, 0])

                logger.info(f"Computing theta3 = atan2({R[1, 2]}, {R[1, 1]})")
                theta3 = sympy.atan2(R[1, 2], R[1, 1])

                logger.info(
                    f"Computing alternative theta1_bis = atan2({-R[2, 0]}, {-R[0, 0]})"
                )
                theta1_bis = sympy.atan2(-R[2, 0], -R[0, 0])

                logger.info(
                    f"Computing alternative theta3_bis = atan2({-R[1, 2]}, {R[1, 1]})"
                )
                theta3_bis = sympy.atan2(-R[1, 2], R[1, 1])

            case EulerSequence.YZX:
                logger.info(f"Computing theta2 = asin({R[1, 2]})")
                theta2 = sympy.asin(R[1, 2])

                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2

                logger.info(f"Computing theta1 = atan2({-R[0, 2]}, {R[2, 2]})")
                theta1 = sympy.atan2(-R[0, 2], R[2, 2])

                logger.info(f"Computing theta3 = atan2({-R[1, 0]}, {R[1, 1]})")
                theta3 = sympy.atan2(-R[1, 0], R[1, 1])

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[0, 2]}, {-R[2, 2]})"
                )
                theta1_bis = sympy.atan2(R[0, 2], -R[2, 2])

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[1, 0]}, {R[1, 1]})"
                )
                theta3_bis = sympy.atan2(R[1, 0], R[1, 1])

            case EulerSequence.ZXY:
                logger.info(f"Computing theta2 = asin({R[2, 1]})")
                theta2 = sympy.asin(R[2, 1])

                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2

                logger.info(f"Computing theta1 = atan2({-R[0, 1]}, {R[1, 1]})")
                theta1 = sympy.atan2(-R[0, 1], R[1, 1])

                logger.info(f"Computing theta3 = atan2({-R[2, 0]}, {R[2, 2]})")
                theta3 = sympy.atan2(-R[2, 0], R[2, 2])

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[0, 1]}, {-R[1, 1]})"
                )
                theta1_bis = sympy.atan2(R[0, 1], -R[1, 1])

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[2, 0]}, {R[2, 2]})"
                )
                theta3_bis = sympy.atan2(R[2, 0], R[2, 2])

            case EulerSequence.ZYX:
                logger.info(f"Computing theta2 = -asin({R[2, 0]})")
                theta2 = -sympy.asin(R[2, 0])

                logger.info("Computing theta2_bis as pi - theta2")
                theta2_bis = sympy.pi - theta2

                logger.info(f"Computing theta1 = atan2({R[1, 0]}, {R[0, 0]})")
                theta1 = sympy.atan2(R[1, 0], R[0, 0])

                logger.info(f"Computing theta3 = atan2({R[2, 1]}, {R[2, 2]})")
                theta3 = sympy.atan2(R[2, 1], R[2, 2])

                logger.info(
                    f"Computing alternative theta1_bis = atan2({-R[1, 0]}, {-R[0, 0]})"
                )
                theta1_bis = sympy.atan2(-R[1, 0], -R[0, 0])

                logger.info(
                    f"Computing alternative theta3_bis = atan2({-R[2, 1]}, {R[2, 2]})"
                )
                theta3_bis = sympy.atan2(-R[2, 1], R[2, 2])

            # Proper Euler angles (first and third same)
            case EulerSequence.ZYZ:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[2, 0]}^2 + {R[2, 1]}^2), {R[2, 2]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2), R[2, 2])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[1, 2]} / sin(theta2), {R[0, 2]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[1, 2] / sin1, R[0, 2] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[2, 1]} / sin(theta2), {-R[2, 0]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[2, 1] / sin1, -R[2, 0] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[1, 2]} / sin(theta2_bis), {R[0, 2]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[1, 2] / sin2, R[0, 2] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[2, 1]} / sin(theta2_bis), {-R[2, 0]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[2, 1] / sin2, -R[2, 0] / sin2)

            case EulerSequence.ZXZ:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[0, 2]}^2 + {R[1, 2]}^2), {R[2, 2]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[0, 2] ** 2 + R[1, 2] ** 2), R[2, 2])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[0, 2]} / sin(theta2), {-R[1, 2]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[0, 2] / sin1, -R[1, 2] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[2, 0]} / sin(theta2), {R[2, 1]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[2, 0] / sin1, R[2, 1] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[0, 2]} / sin(theta2_bis), {-R[1, 2]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[0, 2] / sin2, -R[1, 2] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[2, 0]} / sin(theta2_bis), {R[2, 1]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[2, 0] / sin2, R[2, 1] / sin2)

            case EulerSequence.XYX:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[1, 0]}^2 + {R[2, 0]}^2), {R[0, 0]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[1, 0] ** 2 + R[2, 0] ** 2), R[0, 0])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[1, 0]} / sin(theta2), {-R[2, 0]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[1, 0] / sin1, -R[2, 0] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[0, 1]} / sin(theta2), {R[0, 2]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[0, 1] / sin1, R[0, 2] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[1, 0]} / sin(theta2_bis), {-R[2, 0]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[1, 0] / sin2, -R[2, 0] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[0, 1]} / sin(theta2_bis), {R[0, 2]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[0, 1] / sin2, R[0, 2] / sin2)

            case EulerSequence.YZY:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[2, 1]}^2 + {R[0, 1]}^2), {R[1, 1]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[2, 1] ** 2 + R[0, 1] ** 2), R[1, 1])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[2, 1]} / sin(theta2), {-R[0, 1]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[2, 1] / sin1, -R[0, 1] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[1, 2]} / sin(theta2), {R[1, 0]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[1, 2] / sin1, R[1, 0] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[2, 1]} / sin(theta2_bis), {-R[0, 1]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[2, 1] / sin2, -R[0, 1] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[1, 2]} / sin(theta2_bis), {R[1, 0]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[1, 2] / sin2, R[1, 0] / sin2)

            case EulerSequence.XZX:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[2, 0]}^2 + {R[1, 0]}^2), {R[0, 0]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[2, 0] ** 2 + R[1, 0] ** 2), R[0, 0])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[2, 0]} / sin(theta2), {R[1, 0]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[2, 0] / sin1, R[1, 0] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[0, 2]} / sin(theta2), {-R[0, 1]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[0, 2] / sin1, -R[0, 1] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[2, 0]} / sin(theta2_bis), {R[1, 0]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[2, 0] / sin2, R[1, 0] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[0, 2]} / sin(theta2_bis), {-R[0, 1]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[0, 2] / sin2, -R[0, 1] / sin2)
            case EulerSequence.YXY:
                logger.info(
                    f"Computing theta2 = atan2(sqrt({R[0, 1]}^2 + {R[2, 1]}^2), {R[1, 1]})"
                )
                theta2 = sympy.atan2(sympy.sqrt(R[0, 1] ** 2 + R[2, 1] ** 2), R[1, 1])

                logger.info("Computing theta2_bis as -theta2")
                theta2_bis = -theta2

                sin1, sin2 = sympy.sin(theta2), sympy.sin(theta2_bis)

                logger.info(
                    f"Computing theta1 = atan2({R[0, 1]} / sin(theta2), {R[2, 1]} / sin(theta2))"
                )
                theta1 = sympy.atan2(R[0, 1] / sin1, R[2, 1] / sin1)

                logger.info(
                    f"Computing theta3 = atan2({R[1, 0]} / sin(theta2), {-R[1, 2]} / sin(theta2))"
                )
                theta3 = sympy.atan2(R[1, 0] / sin1, -R[1, 2] / sin1)

                logger.info(
                    f"Computing alternative theta1_bis = atan2({R[0, 1]} / sin(theta2_bis), {R[2, 1]} / sin(theta2_bis))"
                )
                theta1_bis = sympy.atan2(R[0, 1] / sin2, R[2, 1] / sin2)

                logger.info(
                    f"Computing alternative theta3_bis = atan2({R[1, 0]} / sin(theta2_bis), {-R[1, 2]} / sin(theta2_bis))"
                )
                theta3_bis = sympy.atan2(R[1, 0] / sin2, -R[1, 2] / sin2)

            case _:
                raise NotImplementedError(
                    f"Euler Sequence {euler_sequence} not implemented."
                )
        if (
            theta1.equals(sympy.nan)
            or theta3.equals(sympy.nan)
            or theta1_bis.equals(sympy.nan)
            or theta3_bis.equals(sympy.nan)
        ):
            if euler_sequence in {
                EulerSequence.XYZ,
                EulerSequence.XZY,
                EulerSequence.YXZ,
                EulerSequence.YZX,
                EulerSequence.ZXY,
                EulerSequence.ZYX,
            }:
                logger.warning(
                    f"Singularity detected: {theta2=}, {theta2_bis=} for Tait-Bryan sequence (Gimbal Lock)."
                )
            elif euler_sequence in {
                EulerSequence.ZYZ,
                EulerSequence.ZXZ,
                EulerSequence.XYX,
                EulerSequence.YZY,
                EulerSequence.XZX,
                EulerSequence.YXY,
            }:
                logger.warning(
                    f"Singularity detected: {theta2=}, {theta2_bis=} for Proper Euler sequence (Loss of axis separation)."
                )
            # TODO: Ask de luca, what happens in this case I should just assign any number ? or keep nan and make it invalid
            theta1, theta1_bis = (0, 0)
            theta3, theta3_bis = (0, 0)

        # Always cache properly
        euler_angles = EulerAngles(theta1, theta2, theta3)
        euler_angles_bis = EulerAngles(theta1_bis, theta2_bis, theta3_bis)

        self._euler_spec = (
            EulerSpec(euler_angles, euler_sequence, euler_order),
            EulerSpec(euler_angles_bis, euler_sequence, euler_order),
        )

        return self._euler_spec

    @staticmethod
    def is_axis(obj: sympy.Matrix) -> TypeGuard["Axis"]:
        return isinstance(obj, Axis)

    @staticmethod
    def is_rotation(obj: sympy.Matrix) -> TypeGuard["Rotation"]:
        return isinstance(obj, Rotation)

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(self, other: Axis) -> Axis: ...

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation": ...

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "Rotation | Axis | HomogeneousTransformation | sympy.Matrix"
    ) -> "Rotation | Axis | HomogeneousTransformation| sympy.Matrix":
        if self.is_axis(other):
            return Axis(*(super().__matmul__(sympy.Matrix(other))))
        if self.is_rotation(other):
            return Rotation(super().__matmul__(other))
        if self.is_homogeneous(other):
            return HomogeneousTransformation(
                HomogeneousTransformation.identity().with_rotation(self) @ other
            )
        return super().__matmul__(other)

    def __str__(self) -> str:
        ret = f"{self.__class__.__name__}:\n"
        max_len = max([len(str(elem)) for elem in self])
        for row in range(self.rows):
            ret += "\t["
            for col in range(self.cols - 1):
                ret += f"{str(self[row, col]):^{max_len}}, "
            ret += f"{self[row, self.cols - 1]}]"
            if row < (self.rows - 1):
                ret += "\n"
        return ret

    def __setitem__(self, key, value) -> None:
        # Clear cached specs
        self._axis_angle_spec = None
        self._euler_spec = None

        super().__setitem__(key, value)

    def __getitem__(self, key) -> Scalar:
        return sympy.sympify(super().__getitem__(key))


class Translation(sympy.Matrix):
    def __new__(
        cls,
        x: Scalar = 0,
        y: Scalar = 0,
        z: Scalar = 0,
    ):
        return sympy.Matrix.__new__(cls, 3, 1, [x, y, z])

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "HomogeneousTransformation  | Translation | sympy.Matrix"
    ) -> "HomogeneousTransformation | sympy.Matrix":
        if self.is_homogeneous(other):
            return HomogeneousTransformation(super().__matmul__((other)))
        return super().__matmul__((other))


class HomogeneousTransformation(sympy.Matrix):
    def __new__(cls, matrix: sympy.Matrix):
        if matrix.shape != (4, 4):
            raise ValueError("Homogeneous transformation must be a 4x4 matrix")

        # Validate components, the underlying constructor should fail
        _rotation = Rotation(sympy.Matrix(matrix[:3, :3]))
        x = sympy.sympify(matrix[0, 3])
        y = sympy.sympify(matrix[1, 3])
        z = sympy.sympify(matrix[2, 3])
        _translation = Translation(x, y, z)

        bottom = matrix.row(3)
        if not all(elem.equals(0) for elem in bottom[:3]):
            raise ValueError(
                "The first three components of the last row should be all zeros"
            )
        if bottom[3].equals(0):
            raise ValueError("The scale value cannot be 0")
        return super().__new__(cls, 4, 4, matrix)

    @property
    def scale(self) -> sympy.Expr:
        return cast(sympy.Expr, self[3, 3])

    @staticmethod
    def identity() -> "HomogeneousTransformation":
        return HomogeneousTransformation(sympy.eye(4))

    def round(self, digits=4) -> "HomogeneousTransformation":
        mat = self
        for row in range(self.rows):
            for col in range(self.cols):
                mat[row, col] = round(mat[row, col], digits)  # type: ignore
        return HomogeneousTransformation(mat)

    @staticmethod
    def from_rotation(rotation: Rotation) -> "HomogeneousTransformation":
        top = rotation.row_join(Translation())

        bottom = sympy.Matrix([[0, 0, 0, 1]])

        # Assemble final homogeneous matrix
        matrix = top.col_join(bottom)

        return HomogeneousTransformation(matrix)

    def as_rotation(self) -> "Rotation":
        mat = sympy.Matrix(self[:3, :3])
        return Rotation(mat / self.scale)

    def with_rotation(self, rotation: Rotation) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            rotation.row_join(self.as_translation()).col_join(
                sympy.Matrix([[0, 0, 0, self.scale]])
            )
        )

    @staticmethod
    def from_translation(translation: Translation) -> "HomogeneousTransformation":
        top = Rotation.identity().row_join(translation)

        # Build bottom row: [0 0 0 1]
        bottom = sympy.Matrix([[0, 0, 0, 1]])

        # Assemble final homogeneous matrix
        matrix = top.col_join(bottom)

        return HomogeneousTransformation(matrix)

    def as_translation(self) -> "Translation":
        x, y, z = cast(Any, self[:3, 3])
        return Translation(x, y, z)

    def with_translation(
        self, new_translation: Translation
    ) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            self.as_rotation()
            .row_join(new_translation)
            .col_join(sympy.Matrix([[0, 0, 0, self.scale]]))
        )

    def simplify(self) -> "HomogeneousTransformation":
        return HomogeneousTransformation(sympy.simplify(self))

    @staticmethod
    def is_rotation(obj) -> TypeGuard[Rotation]:
        return isinstance(obj, Rotation)

    @staticmethod
    def is_translation(obj: sympy.Matrix) -> TypeGuard[Translation]:
        return isinstance(obj, Translation)

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(self, other: "Translation") -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: "Rotation") -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "HomogeneousTransformation | Rotation | Translation | sympy.Matrix"
    ) -> "HomogeneousTransformation | sympy.Matrix":
        if self.is_rotation(other):
            return super().__matmul__(HomogeneousTransformation.from_rotation(other))

        if self.is_translation(other):
            return super().__matmul__(HomogeneousTransformation.from_translation(other))

        if self.is_homogeneous(other):
            return HomogeneousTransformation(super().__matmul__((other)))
        return super().__matmul__((other))

    def __str__(self) -> str:
        ret = f"{self.__class__.__name__}:\n"
        max_len = max([len(str(elem)) for elem in self])
        for row in range(self.rows):
            ret += "\t["
            for col in range(self.cols - 1):
                ret += f"{str(self[row, col]):^{max_len}}, "
            ret += f"{self[row, self.cols - 1]}]"
            if row < (self.rows - 1):
                ret += "\n"
        return ret
