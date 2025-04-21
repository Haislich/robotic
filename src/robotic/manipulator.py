# %%
from enum import Enum, auto
from typing import List, Optional

import pandas as pd
import sympy

from robotic.transformations import (
    HomogeneousTransformation,
    Rotation,
    Translation,
    X,
    Z,
)


class JointType(Enum):
    PRISMATIC = auto()
    REVOLUTE = auto()


class Manipulator:
    _dh_table: Optional[pd.DataFrame] = None
    _dh_matrix: Optional[HomogeneousTransformation] = None

    def __init__(
        self,
        link_lengths: List[float | sympy.Expr],
        link_twists: List[float | sympy.Expr],
        joint_types: List[JointType],
        theta_offsets: Optional[List[float | sympy.Expr]] = None,
        d_offsets: Optional[List[float | sympy.Expr]] = None,
    ):
        self.link_lengths = link_lengths
        self.link_twists = link_twists
        self.joint_types = joint_types
        n = len(joint_types)

        self.theta_offsets = theta_offsets or [0] * n
        self.d_offsets = d_offsets or [0] * n

        self.joints = [sympy.symbols(f"q_{i + 1}") for i in range(n)]

    # @property
    def dh_table(self) -> pd.DataFrame:
        if self._dh_table is None:
            rows = []
            for i, joint_type in enumerate(self.joint_types):
                a = self.link_lengths[i]
                alpha = self.link_twists[i]
                q_i = self.joints[i]
                theta_offset = self.theta_offsets[i]
                d_offset = self.d_offsets[i]

                if joint_type == JointType.REVOLUTE:
                    theta = q_i + theta_offset
                    d = d_offset
                else:
                    theta = theta_offset
                    d = q_i + d_offset

                rows.append((a, alpha, d, theta, joint_type.name))
            self._dh_table = pd.DataFrame(
                rows, columns=["a", "alpha", "d", "theta", "type"]
            )
        return self._dh_table

    def dh_matrix(self, simplify: bool = True) -> HomogeneousTransformation:
        T = HomogeneousTransformation.identity()
        for _, row in self.dh_table().iterrows():
            T = T @ (
                HomogeneousTransformation.from_rotation(
                    Rotation.from_axis_angle(Z, row.theta)
                )
                @ HomogeneousTransformation.from_translation(Translation(0, 0, row.d))
                @ HomogeneousTransformation.from_translation(Translation(row.a, 0, 0))
                @ HomogeneousTransformation.from_rotation(
                    Rotation.from_axis_angle(X, row.alpha)
                )
            )
        if simplify:
            T = HomogeneousTransformation(sympy.simplify(T))
        return T


# link_lenghts = [sympy.symbols("a_1"), sympy.symbols("a_2"), 0, 0]
# link_twists = [0, 0, 0, sympy.pi]
# joint_types = [
#     JointType.REVOLUTE,
#     JointType.REVOLUTE,
#     JointType.PRISMATIC,
#     JointType.REVOLUTE,
# ]

# man = Manipulator(
#     link_lenghts=link_lenghts, link_twists=link_twists, joint_types=joint_types
# ).dh_table


# mat = Manipulator(
#     link_lenghts=link_lenghts, link_twists=link_twists, joint_types=joint_types
# ).dh_matrix
# print(mat)
