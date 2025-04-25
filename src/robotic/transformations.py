import enum
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Generic,
    Iterator,
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

from robotic import Scalar

T = TypeVar("T")  # The payload type, e.g., Axis or Rotation


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

    def subs(self, *args, **kwargs) -> T:
        for branch in self.branches:
            if branch.condition.subs(*args, **kwargs):
                return branch.value.subs(*args, **kwargs)  # type:ignore
        raise ValueError("")

    def __str__(self) -> str:
        return "\n".join([str(elem) for elem in self])


class Axis(sympy.Matrix):
    def __new__(
        cls,
        x1: Scalar,
        x2: Scalar,
        x3: Scalar,
    ):
        obj = sympy.Matrix.__new__(cls, 3, 1, [x1, x2, x3])
        return obj

    def skew(self):
        x = sympy.sympify(self[0])
        y = sympy.sympify(self[1])
        z = sympy.sympify(self[2])
        return sympy.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def subs(self, *args, **kwargs) -> "Axis":
        result = super().subs(*args, **kwargs)
        return Axis(*result)

    def __repr__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"

    def __str__(self):
        return f"Axis({self[0]}, {self[1]}, {self[2]})"


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

    def __str__(self) -> str:
        return f"AxisAngleSpec({self.axis=},{self.theta=})"


@dataclass
class EulerAngles:
    theta1: Scalar
    theta2: Scalar
    theta3: Scalar

    @property
    def T(self) -> "EulerAngles":
        return EulerAngles(self.theta3, self.theta2, self.theta1)


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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


class EulerOrder(enum.Enum):
    MOVING = "MOVING"
    FIXED = "FIXED"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


@dataclass
class EulerSpec:
    euler_angles: EulerAngles
    euler_sequence: EulerSequence
    euler_order: EulerOrder


class Rotation(sympy.Matrix):
    _axis_angle_spec: Optional[
        AxisAngleSpec
        | Tuple[AxisAngleSpec, AxisAngleSpec]
        | SymbolicConditional[AxisAngleSpec]
    ] = None
    _euler_spec: Optional[EulerSpec] = None

    def __new__(cls, mat: sympy.Matrix, *, tollerance=1e-4):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("A rotation matrix is a square matrix")
        determinant = sympy.simplify(mat.det())
        ortho_check = sympy.simplify(mat.T * mat - sympy.eye(3))
        if not mat.is_symbolic():
            if not abs(float(determinant) - 1) < tollerance:
                raise ValueError(f"Det ≠ 1: found {determinant}")
            if not all(abs(float(val)) < tollerance for val in ortho_check):
                raise ValueError(f"R.T * R ≠ I: found {mat}, {mat.T}")
        else:
            logger.warning("Skipping numeric validation: matrix is symbolic")

        return super().__new__(cls, mat.cols, mat.rows, mat)

    def round(self, digits=4) -> "Rotation":
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
        axis_angle_spec: SymbolicConditional[AxisAngleSpec],
    ) -> "SymbolicConditional[Rotation]": ...

    @staticmethod
    def from_axis_angle(
        axis_angle_spec: AxisAngleSpec
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
            return Rotation.from_axis_angle(axis_angle_spec[0])

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
                self._axis_angle_spec = AxisAngleSpec(Axis(1, 0, 0), theta)
            else:
                self._axis_angle_spec = AxisAngleSpec(
                    Axis(
                        (self[2, 1] - sympy.sympify(self[1, 2]))
                        / (sympy.sympify(2) * sin_theta),
                        (self[0, 2] - sympy.sympify(self[2, 0]))
                        / (sympy.sympify(2) * sin_theta),
                        (self[1, 0] - sympy.sympify(self[0, 1]))
                        / (sympy.sympify(2) * sin_theta),
                    ),
                    theta,
                )
        if self._axis_angle_spec is None:
            raise ValueError("...")
        return self._axis_angle_spec

    @staticmethod
    def from_euler(
        angles: EulerAngles,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> "Rotation":
        angle_map = [angles.theta1, angles.theta2, angles.theta3]
        axis_map = {"X": X, "Y": Y, "Z": Z}

        axes = [axis_map[c] for c in sequence.value]
        # Create each elemental rotation
        R1 = Rotation.from_axis_angle(AxisAngleSpec(axes[0], angle_map[0]))
        R2 = Rotation.from_axis_angle(AxisAngleSpec(axes[1], angle_map[1]))
        R3 = Rotation.from_axis_angle(AxisAngleSpec(axes[2], angle_map[2]))

        if order == order.MOVING:
            mat = R3 @ R2 @ R1  # world frame
        else:
            mat = R1 @ R2 @ R3  # body frame

        mat._euler_spec = EulerSpec(angles, sequence, order)
        return mat

    def to_euler(
        self,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> EulerSpec:
        euler_angles = EulerAngles(0, 0, 0)
        if self._euler_spec is None:
            match sequence:
                # FIXED Tait-Bryan
                case EulerSequence.XYZ:
                    theta2 = sympy.asin(self[0, 2])
                    theta1 = sympy.atan2(-sympy.sympify(self[1, 2]), self[2, 2])
                    theta3 = sympy.atan2(-sympy.sympify(self[0, 1]), self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XZY:
                    theta2 = -sympy.asin(self[0, 1])
                    theta1 = sympy.atan2(self[2, 1], self[1, 1])
                    theta3 = sympy.atan2(self[0, 2], self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YXZ:
                    theta2 = sympy.asin(self[0, 2])
                    theta1 = sympy.atan2(-sympy.sympify(self[1, 2]), self[2, 2])
                    theta3 = sympy.atan2(-sympy.sympify(self[0, 1]), self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YZX:
                    theta2 = sympy.asin(self[1, 0])
                    theta1 = sympy.atan2(-sympy.sympify(self[2, 0]), self[0, 0])
                    theta3 = sympy.atan2(-sympy.sympify(self[1, 2]), self[1, 1])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.ZXY:
                    theta2 = sympy.asin(self[2, 1])
                    theta1 = sympy.atan2(-sympy.sympify(self[0, 1]), self[1, 1])
                    theta3 = sympy.atan2(-sympy.sympify(self[2, 0]), self[2, 2])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.ZYX:
                    theta2 = -sympy.asin(self[2, 0])
                    theta1 = sympy.atan2(self[1, 0], self[0, 0])
                    theta3 = sympy.atan2(self[2, 1], self[2, 2])
                    euler_angles = EulerAngles(theta1, theta2, theta3)

                # FIXED: Proper Euler Angles
                case EulerSequence.ZXZ:
                    theta2 = sympy.acos(self[2, 2])
                    theta1 = sympy.atan2(self[0, 2], self[1, 2])
                    theta3 = sympy.atan2(self[2, 0], -sympy.sympify(self[2, 1]))
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XYX:
                    theta2 = sympy.acos(self[0, 0])
                    theta1 = sympy.atan2(self[1, 0], -sympy.sympify(self[2, 0]))
                    theta3 = sympy.atan2(self[0, 1], self[0, 2])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YZY:
                    theta2 = sympy.acos(self[1, 1])
                    theta1 = sympy.atan2(self[2, 1], -sympy.sympify(self[0, 1]))
                    theta3 = sympy.atan2(self[1, 2], self[1, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XZX:
                    theta2 = sympy.acos(self[0, 0])
                    theta1 = sympy.atan2(self[2, 0], self[1, 0])
                    theta3 = sympy.atan2(self[0, 2], -sympy.sympify(self[0, 1]))
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.ZYZ:
                    theta2 = sympy.acos(self[2, 2])
                    theta1 = sympy.atan2(self[1, 2], -sympy.sympify(self[0, 2]))
                    theta3 = sympy.atan2(self[2, 1], self[2, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YXY:
                    theta2 = sympy.acos(self[1, 1])
                    theta1 = sympy.atan2(self[0, 1], self[2, 1])
                    theta3 = sympy.atan2(self[1, 0], -sympy.sympify(self[1, 2]))
                    euler_angles = EulerAngles(theta1, theta2, theta3)

                case _:
                    raise NotImplementedError
        if EulerOrder.MOVING:
            euler_angles = euler_angles.T
        self._euler_spec = EulerSpec(
            euler_angles,
            sequence,
            order,
        )
        return self._euler_spec

    @staticmethod
    def is_rotation(obj) -> TypeGuard["Rotation"]:
        return isinstance(obj, Rotation)

    @staticmethod
    def is_homogeneous(obj: sympy.Matrix) -> TypeGuard["HomogeneousTransformation"]:
        return isinstance(obj, HomogeneousTransformation)

    @overload
    def __matmul__(
        self, other: "HomogeneousTransformation"
    ) -> "HomogeneousTransformation": ...

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "Rotation |HomogeneousTransformation |sympy.Matrix"
    ) -> "Rotation | HomogeneousTransformation| sympy.Matrix":
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
        mat = cast(sympy.Matrix, self[:3, :3])
        return Rotation(mat / self.scale)

    def with_rotation(self, rotation: Rotation) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            rotation.row_join(self.as_translation()).col_join(
                sympy.Matrix([[0, 0, 0, self.scale]])
            )
        )

    @staticmethod
    def is_rotation(obj) -> TypeGuard[Rotation]:
        return isinstance(obj, Rotation)

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

    @staticmethod
    def is_translation(obj: sympy.Matrix) -> TypeGuard[Translation]:
        return isinstance(obj, Translation)

    def with_translation(
        self, new_translation: Translation
    ) -> "HomogeneousTransformation":
        return HomogeneousTransformation(
            self.as_rotation()
            .row_join(new_translation)
            .col_join(sympy.Matrix([[0, 0, 0, self.scale]]))
        )

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


theta = sympy.symbols(names="theta")
r_d = Rotation.from_axis_angle(AxisAngleSpec(X, sympy.pi))

r_d_axis_spec = r_d.to_axis_angle()
print(type(r_d_axis_spec))

print(Rotation.from_axis_angle(r_d_axis_spec).subs(theta, 0))
