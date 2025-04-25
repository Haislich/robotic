import enum
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Generic,
    Iterator,
    List,
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
from sympy.matrices.expressions.matexpr import MatrixElement

from robotic import Scalar

T = TypeVar("T")  # The payload type, e.g., Axis or Rotation


@dataclass
class SymbolicBranch(Generic[T]):
    value: T
    condition: sympy.Basic | bool


class SymbolicConditional(Generic[T]):
    def __init__(self, branches: Sequence[SymbolicBranch[T]]):
        self.branches: Sequence[SymbolicBranch[T]] = branches

    def __iter__(self) -> Iterator[SymbolicBranch[T]]:
        return iter(self.branches)

    def __getitem__(self, index: int) -> SymbolicBranch[T]:
        return self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def subs(self, *args, **kwargs) -> T | Sequence[T]:
        ret = []
        for branch in self.branches:
            condition = sympy.S(branch.condition)
            cond_eval = condition.subs(*args, **kwargs)
            if cond_eval:
                value = sympy.S(branch.value)
                return value
        #         ret.append(value.subs(*args, **kwargs))
        # if len(ret) == 0:
        #     raise ValueError("No matching branch found for the given substitution.")
        # if len(ret) == 1:
        #     return ret[0]
        # return ret

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
        x = cast(sympy.Expr, self[0])
        y = cast(sympy.Expr, self[1])
        z = cast(sympy.Expr, self[2])
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
    axis: Axis | SymbolicConditional[Axis]
    theta: Scalar

    def __init__(self, axis: Axis | SymbolicConditional[Axis], theta: Scalar) -> None:
        self.axis = axis
        self.theta = theta


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
    _axis_angle_spec: Optional[AxisAngleSpec | Tuple[AxisAngleSpec, AxisAngleSpec]] = (
        None
    )
    _euler_spec: Optional[EulerSpec] = None

    def __new__(cls, mat: sympy.Matrix, *, tollerance=1e-4):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("A rotation matrix is a square matrix")
        logger.debug(f"{mat=}")
        determinant = sympy.simplify(mat.det())
        logger.debug(f"{determinant=}")
        ortho_check = sympy.simplify(mat.T * mat - sympy.eye(3))
        logger.debug(f"{ortho_check=}")

        is_numeric = all(entry.is_number for entry in mat)  # type: ignore

        if is_numeric:
            if not abs(float(determinant) - 1) < tollerance:
                raise ValueError(f"Det ≠ 1: found {determinant}")
            if not all(abs(float(val)) < tollerance for val in ortho_check):
                raise ValueError(f"R.T * R ≠ I: found {mat}, {mat.T}")
        else:
            print("⚠️ Skipping numeric validation: matrix is symbolic")

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

    @staticmethod
    def from_axis_angle(
        axis_angle_spec: AxisAngleSpec,
    ) -> "Rotation | SymbolicConditional[Rotation]":
        r = axis_angle_spec.axis
        theta = axis_angle_spec.theta
        if isinstance(r, Axis):
            identity = sympy.eye(3)
            skew = sympy.Matrix(r.skew())
            twist = skew * sympy.sin(theta)
            flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
            return Rotation(identity + twist + flatten)

        ret = []
        for elem in r:
            if any(val.has(sympy.nan) for val in elem.value):
                continue
            identity = sympy.eye(3)
            skew = sympy.Matrix(elem.value.skew())
            twist = skew * sympy.sin(theta)
            flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
            logger.debug(f"{identity}, {twist=},  {flatten=}")

            logger.debug(f"{identity + twist + flatten =}")
            ret.append(
                SymbolicBranch(Rotation(identity + twist + flatten), elem.condition)
            )

        return SymbolicConditional(ret)

    def to_axis_angle(self) -> AxisAngleSpec:
        if self._axis_angle_spec is None:
            theta = sympy.atan2(
                sympy.sqrt(
                    (self[0, 1] - self[1, 0]) ** 2
                    + (self[0, 2] - self[2, 0]) ** 2
                    + (self[1, 2] - self[2, 1]) ** 2
                ),
                (self[0, 0] + self[1, 1] + self[2, 2]) - 1,
            )
            sin_theta = sympy.sin(theta)
            if self.is_symbolic():
                pieces = [
                    # Any solution is good,so I chose the X
                    SymbolicBranch(Axis(1, 0, 0), sympy.Eq(theta, 0)),
                    SymbolicBranch(
                        Axis(
                            (self[2, 1] - self[1, 2]) / (sympy.S(2) * sin_theta),
                            (self[0, 2] - self[2, 0]) / (sympy.S(2) * sin_theta),
                            (self[1, 0] - self[0, 1]) / (sympy.S(2) * sin_theta),
                        ),
                        sympy.Eq(True, True),
                    ),
                ]
                for sx, sy, sz in product([1, -1], repeat=3):
                    ax = sympy.S(sx) * sympy.sqrt((self[0, 0] + 1) / 2)
                    ay = sympy.S(sy) * sympy.sqrt((self[1, 1] + 1) / 2)
                    az = sympy.S(sz) * sympy.sqrt((self[2, 2] + 1) / 2)

                    # Each component of the axis must satisfy the off-diagonal conditions
                    cond: sympy.Basic = sympy.And(
                        sympy.Eq(theta, sympy.pi),
                        sympy.Eq(self[0, 1], 2 * ax * ay),
                        sympy.Eq(self[0, 2], 2 * ax * az),
                        sympy.Eq(self[1, 2], 2 * ay * az),
                    )
                    pieces.insert(0, SymbolicBranch(Axis(ax, ay, az), cond))
                self._axis_angle_spec = AxisAngleSpec(
                    SymbolicConditional(pieces), theta
                )
            else:
                if (theta == sympy.pi) or (theta == -sympy.pi):
                    candidates = []
                    for sx, sy, sz in product([1, -1], repeat=3):
                        ax = float(sx) * sympy.sqrt((self[0, 0] + 1) / 2)
                        ay = float(sy) * sympy.sqrt((self[1, 1] + 1) / 2)
                        az = float(sz) * sympy.sqrt((self[2, 2] + 1) / 2)

                        # Numerical values may have small errors, use a tolerance
                        def isclose(a, b, tol=1e-4):
                            return abs(a - b) < tol

                        if (
                            isclose(self[0, 1], 2 * ax * ay)
                            and isclose(self[0, 2], 2 * ax * az)
                            and isclose(self[1, 2], 2 * ay * az)
                        ):
                            candidates.append(Axis(ax, ay, az))

                    self._axis_angle_spec = AxisAngleSpec(candidates, theta)
                elif theta == 0:
                    self._axis_angle_spec = AxisAngleSpec(Axis(1, 0, 0), theta)
                else:
                    self._axis_angle_spec = AxisAngleSpec(
                        Axis(
                            (self[2, 1] - self[1, 2]) / (sympy.S(2) * sin_theta),
                            (self[0, 2] - self[2, 0]) / (sympy.S(2) * sin_theta),
                            (self[1, 0] - self[0, 1]) / (sympy.S(2) * sin_theta),
                        ),
                        theta,
                    )

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
        self = cast(Any, self)  # Trust me bro
        euler_angles = EulerAngles(0, 0, 0)
        if self._euler_spec is None:
            match sequence:
                # FIXED Tait-Bryan
                case EulerSequence.XYZ:
                    theta2 = sympy.asin(self[0, 2])
                    theta1 = sympy.atan2(-self[1, 2], self[2, 2])
                    theta3 = sympy.atan2(-self[0, 1], self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XZY:
                    theta2 = -sympy.asin(self[0, 1])
                    theta1 = sympy.atan2(self[2, 1], self[1, 1])
                    theta3 = sympy.atan2(self[0, 2], self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YXZ:
                    theta2 = sympy.asin(self[0, 2])
                    theta1 = sympy.atan2(-self[1, 2], self[2, 2])
                    theta3 = sympy.atan2(-self[0, 1], self[0, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YZX:
                    theta2 = sympy.asin(self[1, 0])
                    theta1 = sympy.atan2(-self[2, 0], self[0, 0])
                    theta3 = sympy.atan2(-self[1, 2], self[1, 1])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.ZXY:
                    theta2 = sympy.asin(self[2, 1])
                    theta1 = sympy.atan2(-self[0, 1], self[1, 1])
                    theta3 = sympy.atan2(-self[2, 0], self[2, 2])
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
                    theta3 = sympy.atan2(self[2, 0], -self[2, 1])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XYX:
                    theta2 = sympy.acos(self[0, 0])
                    theta1 = sympy.atan2(self[1, 0], -self[2, 0])
                    theta3 = sympy.atan2(self[0, 1], self[0, 2])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YZY:
                    theta2 = sympy.acos(self[1, 1])
                    theta1 = sympy.atan2(self[2, 1], -self[0, 1])
                    theta3 = sympy.atan2(self[1, 2], self[1, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.XZX:
                    theta2 = sympy.acos(self[0, 0])
                    theta1 = sympy.atan2(self[2, 0], self[1, 0])
                    theta3 = sympy.atan2(self[0, 2], -self[0, 1])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.ZYZ:
                    theta2 = sympy.acos(self[2, 2])
                    theta1 = sympy.atan2(self[1, 2], -self[0, 2])
                    theta3 = sympy.atan2(self[2, 1], self[2, 0])
                    euler_angles = EulerAngles(theta1, theta2, theta3)
                case EulerSequence.YXY:
                    theta2 = sympy.acos(self[1, 1])
                    theta1 = sympy.atan2(self[0, 1], self[2, 1])
                    theta3 = sympy.atan2(self[1, 0], -self[1, 2])
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

        # Perform the assignment
        super().__setitem__(key, value)

        # #Validate the new matrix is still a proper rotation
        # if not self.det().equals(1):
        #     raise ValueError("Matrix must have determinant +1 after update.")
        # if not self.inv().equals(self.T):
        #     raise ValueError(r"Matrix must satisfy $R^T = R^{-1}$ after update.")


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
        _rotation = Rotation(cast(sympy.Matrix, matrix[:3, :3]))
        x, y, z = cast(Any, matrix[:3, 3])
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
# r_d = Rotation(1 / 3 * sympy.Matrix([[-2, 2, -1], [2, 1, -2], [-1, -2, -2]]))
r_d = Rotation.from_axis_angle(AxisAngleSpec(X, theta))
print(r_d.is_symbolic())
# exit()
# r_d = Rotation.from_axis_angle(AxisAngleSpec(X, theta))
# print(theta)
r_d_axis_spec = r_d.to_axis_angle()
# print(r_d_axis_spec)
# nr_d_axis_spec = r_d.to_axis_angle()
print(r_d_axis_spec)
print(Rotation.from_axis_angle(r_d_axis_spec))
print(Rotation.from_axis_angle(r_d_axis_spec).subs(theta, 0))

# print(Rotation.from_axis_angle(nr_d_axis_spec))
