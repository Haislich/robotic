import enum
from typing import Any, Optional, Tuple, TypeGuard, cast, overload

import sympy

from robotic import axis
from robotic.axis import Axis


class EulerOrder(enum.Enum):
    MOVING = enum.auto()
    FIXED = enum.auto()


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


EulerAngles = Tuple[sympy.Expr | float, sympy.Expr | float, sympy.Expr | float]
EulerSpec = Tuple[
    EulerAngles,
    EulerSequence,
    EulerOrder,
]
AxisAngleSpec = Tuple[Axis, sympy.Expr | float]


class Rotation(sympy.Matrix):
    _axis_angle_spec: Optional[AxisAngleSpec] = None
    _euler_spec: Optional[EulerSpec] = None

    def __new__(cls, mat: sympy.Matrix):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("A rotation matrix is a square matrix")
        if not mat.det().equals(1):
            raise ValueError("A rotation matrix has determinant +1")
        if not mat.inv().equals(mat.T):
            raise ValueError(r"A rotation matrix is such that $R^T = R^{-1}$")
        return super().__new__(cls, mat.cols, mat.rows, mat)

    @staticmethod
    def direct_axis_angle(r: Axis, theta: sympy.Expr | float) -> "Rotation":
        identity = sympy.eye(3)
        skew = sympy.Matrix(r.skew())
        twist = skew * sympy.sin(theta)
        flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
        return Rotation(identity + twist + flatten)

    def inverse_axis_angle(self) -> AxisAngleSpec:
        self = cast(Any, self)  # Trust me bro
        if self._axis_angle_spec is not None:
            return self._axis_angle_spec

        theta = sympy.atan2(
            sympy.sqrt(
                (self[0, 1] - self[1, 0]) ** 2
                + (self[0, 2] - self[2, 0]) ** 2
                + (self[1, 2] - self[2, 1]) ** 2
            ),
            (self[0, 0] + self[1, 1] + self[2, 2]),
        )
        sin_theta = sympy.sin(theta)

        regular_axis = Axis(
            (self[2, 1] - self[1, 2]) / (sympy.Integer(2) * sin_theta),
            (self[0, 2] - self[2, 0]) / (sympy.Integer(2) * sin_theta),
            (self[1, 0] - self[0, 1]) / (sympy.Integer(2) * sin_theta),
        )

        singular_axis_pi = Axis(
            sympy.sqrt((self[0, 0] + 1) / 2),
            sympy.sqrt((self[1, 1] + 1) / 2),
            sympy.sqrt((self[2, 2] + 1) / 2),
        )

        singular_axis_zero = Axis(sympy.nan, sympy.nan, sympy.nan)

        axis = Axis(
            sympy.Piecewise(
                (singular_axis_zero[0], sympy.Eq(theta, 0)),
                (singular_axis_pi[0], sympy.Eq(sin_theta, 0)),
                (regular_axis[0], True),
            ),
            sympy.Piecewise(
                (singular_axis_zero[1], sympy.Eq(theta, 0)),
                (singular_axis_pi[1], sympy.Eq(sin_theta, 0)),
                (regular_axis[1], True),
            ),
            sympy.Piecewise(
                (singular_axis_zero[2], sympy.Eq(theta, 0)),
                (singular_axis_pi[2], sympy.Eq(sin_theta, 0)),
                (regular_axis[2], True),
            ),
        )
        self._axis_angle_spec = (axis, theta)
        return self._axis_angle_spec

    @staticmethod
    def direct_euler(
        angles: EulerAngles,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> "Rotation":
        alpha, beta, gamma = angles
        angle_map = [alpha, beta, gamma]
        axis_map = {"X": axis.X, "Y": axis.Y, "Z": axis.Z}

        axes = [axis_map[c] for c in sequence.value]
        # Create each elemental rotation
        R1 = Rotation.direct_axis_angle(axes[0], angle_map[0])
        R2 = Rotation.direct_axis_angle(axes[1], angle_map[1])
        R3 = Rotation.direct_axis_angle(axes[2], angle_map[2])

        if order == order.FIXED:
            mat = R3 @ R2 @ R1  # world frame
        else:
            mat = R1 @ R2 @ R3  # body frame

        mat._euler_spec = (angles, sequence, order)
        return mat

    def inverse_euler(
        self,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> EulerAngles:
        raise NotImplementedError

    @staticmethod
    def is_rotation(obj) -> TypeGuard["Rotation"]:
        return isinstance(obj, Rotation)

    @staticmethod
    def is_homogeneous_rotation(obj: sympy.Matrix) -> TypeGuard["HomogeneousRotation"]:
        return isinstance(obj, HomogeneousRotation)

    @overload
    def __matmul__(self, other: "HomogeneousRotation") -> "HomogeneousRotation": ...

    @overload
    def __matmul__(self, other: "Rotation") -> "Rotation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "Rotation |HomogeneousRotation |sympy.Matrix"
    ) -> "Rotation | HomogeneousRotation| sympy.Matrix":
        if self.is_homogeneous_rotation(other):
            return HomogeneousRotation(super().__matmul__(other.as_rotation()))
        if self.is_rotation(other):
            return Rotation(super().__matmul__(other))
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

        # Validate the new matrix is still a proper rotation
        if not self.det().equals(1):
            raise ValueError("Matrix must have determinant +1 after update.")
        if not self.inv().equals(self.T):
            raise ValueError(r"Matrix must satisfy $R^T = R^{-1}$ after update.")


class HomogeneousRotation(Rotation):
    def __new__(cls, rot: Rotation):
        # if rot.shape != (3, 3):
        #     raise ValueError("Expected a 3x3 rotation matrix")

        # Build top block: [R | 0]
        top = rot.row_join(sympy.zeros(3, 1))

        # Build bottom row: [0 0 0 1]
        bottom = sympy.Matrix([[0, 0, 0, 1]])

        # Assemble final homogeneous matrix
        full = top.col_join(bottom)

        # Create new Matrix instance with Rotation behavior
        return sympy.Matrix.__new__(cls, 4, 4, full)

    def as_rotation(self) -> Rotation:
        scale = cast(sympy.Expr, self[3, 3])
        mat = cast(sympy.Matrix, self[:3, :3])
        if not scale.equals(1):
            return Rotation(mat / scale)
        return Rotation(mat)

    @overload
    def __matmul__(self, other: "HomogeneousRotation") -> "HomogeneousRotation": ...

    @overload
    def __matmul__(self, other: "Rotation") -> "HomogeneousRotation": ...

    @overload
    def __matmul__(self, other: sympy.Matrix) -> sympy.Matrix: ...

    def __matmul__(
        self, other: "HomogeneousRotation | sympy.Matrix"
    ) -> "HomogeneousRotation| sympy.Matrix":
        if self.is_homogeneous_rotation(other):
            return HomogeneousRotation(self.as_rotation() @ other.as_rotation())
        if self.is_rotation(other):
            return HomogeneousRotation(self.as_rotation() @ other)
        return super().__matmul__(other)


# phi, theta, psi = sympy.symbols("phi theta psi")
# rot1 = Rotation.direct_axis_angle(axis.X, theta)
# rot2 = Rotation.direct_axis_angle(axis.Z, psi)


# # print(HomogeneousRotation(rot1) @ rot1)
# # print(HomogeneousRotation(rot1) @ HomogeneousRotation(rot1))
# # print((HomogeneousRotation(rot1) @ HomogeneousRotation(rot1)).as_rotation())
# print(type(rot1 @ HomogeneousRotation(rot1)))
# print((HomogeneousRotation(rot1) @ rot1).as_rotation())
# print(type(HomogeneousRotation(rot1) @ sympy.Matrix(4, 1, [1, 2, 3, 4])))
