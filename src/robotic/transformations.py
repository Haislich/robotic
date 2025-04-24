import enum
from typing import Any, List, Optional, Tuple, TypeGuard, cast, overload

import sympy


class Axis(sympy.Matrix):
    def __new__(
        cls,
        x1: float | sympy.Float | sympy.Expr,
        x2: float | sympy.Float | sympy.Expr,
        x3: float | sympy.Float | sympy.Expr,
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


EulerAngles = (
    Tuple[sympy.Expr | float, sympy.Expr | float, sympy.Expr | float]
    | List[sympy.Expr | float]
)
EulerSpec = Tuple[
    EulerAngles,
    EulerSequence,
    EulerOrder,
]
AxisAngleSpec = Tuple[Axis, sympy.Expr | float]


class Rotation(sympy.Matrix):
    _axis_angle_spec: Optional[AxisAngleSpec] = None
    _euler_spec: Optional[EulerSpec] = None

    def __new__(
        cls,
        mat: sympy.Matrix,
    ):
        # epsilon = 1e-10
        # if mat.shape[0] != mat.shape[1]:
        #     raise ValueError("A rotation matrix is a square matrix")
        # if not sympy.simplify(mat.det().evalf() - 1) < epsilon:
        #     print(mat.det())
        #     raise ValueError("A rotation matrix has determinant +1")

        # if not all(abs(sympy.simplify(mat.T * mat - sympy.eye(3)))) < epsilon:
        #     print(f"{mat=}")
        #     print(f"{mat.T=}")
        #     print(f"{mat * mat.T =}")
        #     print(sympy.simplify(mat.T * mat - sympy.eye(3)))

        #     raise ValueError(r"A rotation matrix is such that $R^T = R^{-1}$")
        return super().__new__(cls, mat.cols, mat.rows, mat)

    def round(self, digits=4) -> "Rotation":
        mat = self
        for row in range(self.rows):
            for col in range(self.cols):
                mat[row, col] = round(mat[row, col], digits)  # type: ignore
        return Rotation(mat)

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
        r: Axis = Axis(0, 0, 0), theta: sympy.Expr | float = 0.0
    ) -> "Rotation":
        identity = sympy.eye(3)
        skew = sympy.Matrix(r.skew())
        twist = skew * sympy.sin(theta)
        flatten = (sympy.Integer(1) - sympy.cos(theta)) * skew**2
        return Rotation(identity + twist + flatten)

    def to_axis_angle(self) -> AxisAngleSpec:
        self = cast(Any, self)  # Trust me bro
        if self._axis_angle_spec is not None:
            return self._axis_angle_spec

        theta = sympy.atan2(
            sympy.sqrt(
                (self[0, 1] - self[1, 0]) ** 2
                + (self[0, 2] - self[2, 0]) ** 2
                + (self[1, 2] - self[2, 1]) ** 2
            ),
            (self[0, 0] + self[1, 1] + self[2, 2]) - 1,
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
    def from_euler(
        angles: EulerAngles,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> "Rotation":
        alpha, beta, gamma = angles
        angle_map = [alpha, beta, gamma]
        axis_map = {"X": X, "Y": Y, "Z": Z}

        axes = [axis_map[c] for c in sequence.value]
        # Create each elemental rotation
        R1 = Rotation.from_axis_angle(axes[0], angle_map[0])
        R2 = Rotation.from_axis_angle(axes[1], angle_map[1])
        R3 = Rotation.from_axis_angle(axes[2], angle_map[2])

        if order == order.MOVING:
            mat = R3 @ R2 @ R1  # world frame
        else:
            mat = R1 @ R2 @ R3  # body frame

        mat._euler_spec = (angles, sequence, order)
        return mat

    def to_euler(
        self,
        sequence: EulerSequence = EulerSequence.XYZ,
        order: EulerOrder = EulerOrder.MOVING,
    ) -> EulerSpec:
        self = cast(Any, self)  # Trust me bro
        if self._euler_spec is not None:
            return self._euler_spec
        THETAS = {
            EulerSequence.XYZ = 
        }
        match (sequence, order):
            # FIXED Tait-Bryan
            case EulerSequence.XYZ, EulerOrder.FIXED:
                theta2 = sympy.asin(self[0, 2])
                theta1 = sympy.atan2(-self[1, 2], self[2, 2])
                theta3 = sympy.atan2(-self[0, 1], self[0, 0])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.XZY, EulerOrder.FIXED:
                theta2 = -sympy.asin(self[0, 1])
                theta1 = sympy.atan2(self[2, 1], self[1, 1])
                theta3 = sympy.atan2(self[0, 2], self[0, 0])

                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.YXZ, EulerOrder.FIXED:
                theta2 = sympy.asin(self[0, 2])
                theta1 = sympy.atan2(-self[1, 2], self[2, 2])
                theta3 = sympy.atan2(-self[0, 1], self[0, 0])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.YZX, EulerOrder.FIXED:
                theta2 = sympy.asin(self[1, 0])
                theta1 = sympy.atan2(-self[2, 0], self[0, 0])
                theta3 = sympy.atan2(-self[1, 2], self[1, 1])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.ZXY, EulerOrder.FIXED:
                theta2 = sympy.asin(self[2, 1])
                theta1 = sympy.atan2(-self[0, 1], self[1, 1])
                theta3 = sympy.atan2(-self[2, 0], self[2, 2])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.ZYX, EulerOrder.FIXED:
                theta2 = -sympy.asin(self[2, 0])
                theta1 = sympy.atan2(self[1, 0], self[0, 0])
                theta3 = sympy.atan2(self[2, 1], self[2, 2])
                return (theta1, theta2, theta3), sequence, order

            # FIXED: Proper Euler Angles
            case EulerSequence.ZXZ, EulerOrder.FIXED:
                theta2 = sympy.acos(self[2, 2])
                theta1 = sympy.atan2(self[0, 2], self[1, 2])
                theta3 = sympy.atan2(self[2, 0], -self[2, 1])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.XYX, EulerOrder.FIXED:
                theta2 = sympy.acos(self[0, 0])
                theta1 = sympy.atan2(self[1, 0], -self[2, 0])
                theta3 = sympy.atan2(self[0, 1], self[0, 2])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.YZY, EulerOrder.FIXED:
                theta2 = sympy.acos(self[1, 1])
                theta1 = sympy.atan2(self[2, 1], -self[0, 1])
                theta3 = sympy.atan2(self[1, 2], self[1, 0])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.XZX, EulerOrder.FIXED:
                theta2 = sympy.acos(self[0, 0])
                theta1 = sympy.atan2(self[2, 0], self[1, 0])
                theta3 = sympy.atan2(self[0, 2], -self[0, 1])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.ZYZ, EulerOrder.FIXED:
                theta2 = sympy.acos(self[2, 2])
                theta1 = sympy.atan2(self[1, 2], -self[0, 2])
                theta3 = sympy.atan2(self[2, 1], self[2, 0])
                return (theta1, theta2, theta3), sequence, order
            case EulerSequence.YXY, EulerOrder.FIXED:
                theta2 = sympy.acos(self[1, 1])
                theta1 = sympy.atan2(self[0, 1], self[2, 1])
                theta3 = sympy.atan2(self[1, 0], -self[1, 2])
                return (theta1, theta2, theta3), sequence, order
            case _:
                raise NotImplementedError

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
        x: sympy.Expr | float = 0,
        y: sympy.Expr | float = 0,
        z: sympy.Expr | float = 0,
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
