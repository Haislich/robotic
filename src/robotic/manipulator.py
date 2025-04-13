from enum import Enum, auto
from typing import List, Optional

import pandas as pd
import sympy

from robotic.transformations.homogeneous import HomogeneousTransformation


class JointType(Enum):
    PRISMATIC = auto()
    REVOLUTE = auto()


class Manipulator:
    _dh_table: Optional[pd.DataFrame] = None

    def __init__(
        self,
        link_lenghts: List[float | sympy.Expr],
        link_twists: List[float | sympy.Expr],
        joint_types: List[JointType],
    ) -> None:
        self.link_lenghts = link_lenghts
        self.link_twists = link_twists
        self.joints = [
            sympy.symbols(
                f"theta_{i}" if joint_type == JointType.REVOLUTE else f"d_{i}"
            )
            for joint_type, i in enumerate(joint_types)
        ]

    @property
    def dh_table(self) -> pd.DataFrame:
        if self._dh_table is None:
            rows = []
            for i, joint_type in enumerate(self.joints):
                a = self.link_lenghts[i]
                alpha = self.link_twists[i]
                if joint_type == JointType.REVOLUTE:
                    theta = sympy.symbols(f"theta_{i}")
                    d = 0
                else:
                    d = sympy.symbols(f"d_{i}")
                    theta = 0
                rows.append((a, alpha, d, theta, joint_type.name))
            self._dh_table = pd.DataFrame(
                rows, columns=["a", "alpha", "d", "theta", "type"]
            )
        return self._dh_table

    # @property
    # def dh_matrix(self) -> HomogeneousTransformation: ...


link_lenghts = [sympy.symbols("a_1"), sympy.symbols("a_2"), 0, 0]
link_twists = [0, 0, 0, sympy.pi]
joint_types = [
    JointType.REVOLUTE,
    JointType.REVOLUTE,
    JointType.REVOLUTE,
    JointType.REVOLUTE,
]
print(
    Manipulator(
        link_lenghts=link_lenghts, link_twists=link_twists, joint_types=joint_types
    ).dh_table
)
