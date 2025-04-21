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


class DHTable(pd.DataFrame):
    # Required for pandas subclassing
    _metadata = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If needed, convert entries to sympy objects
        for col in ["a", "alpha", "d", "theta"]:
            if col in self.columns:
                self[col] = self[col].apply(
                    lambda x: sympy.sympify(x) if not isinstance(x, str) else x
                )

    def __str__(self):
        # Pretty plain-text for print()
        # Columns: align for print (as string)
        colwidths = [
            max(len(str(x)) for x in [col] + self[col].astype(str).tolist())
            for col in self.columns
        ]
        header = " | ".join(
            [str(col).ljust(w) for col, w in zip(self.columns, colwidths)]
        )
        lines = [header, "-" * len(header)]
        for _, row in self.iterrows():
            line = " | ".join([str(x).ljust(w) for x, w in zip(row, colwidths)])
            lines.append(line)
        return "\n".join(lines)

    def _repr_html_(self): ...
    def _repr_latex_(self):
        headers = [r"a", r"\alpha", "d", r"\theta", "type"]
        header_row = " & ".join(headers) + r" \\ \hline"
        rows = []

        for _, row in self.iterrows():
            formatted = []
            for x in row:
                if isinstance(x, (int, float, sympy.Basic)):
                    formatted.append(sympy.latex(sympy.sympify(x)))
                elif isinstance(x, str) and x.upper() in {"PRISMATIC", "REVOLUTE"}:
                    formatted.append(r"\mathrm{" + x.capitalize() + r"}")
                else:
                    formatted.append(str(x))
            rows.append(
                " & ".join(formatted) + r" \\[0.5em]"
            )  # Add vertical sympyacing

        latex_table = (
            r"$" + "\n"
            r"\begin{array}{|c|c|c|c|c|}" + "\n"
            r"\hline" + "\n" + header_row + "\n" + "\n".join(rows) + "\n"
            r"\hline" + "\n"
            r"\end{array}" + "\n"
            r"$"
        )
        return latex_table


class JointType(Enum):
    PRISMATIC = auto()
    REVOLUTE = auto()


class Manipulator:
    _dh_table: Optional[DHTable] = None
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

    def dh_table(self) -> DHTable:
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
            self._dh_table = DHTable(rows, columns=["a", "alpha", "d", "theta", "type"])
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


# joint_types = [
#     JointType.PRISMATIC,
#     JointType.PRISMATIC,
#     JointType.REVOLUTE,
# ]
# # a
# link_lengths = [0, 0, sympy.symbols("L")]
# # alpha
# link_twists = [-sympy.pi / 2, -sympy.pi / 2, 0]
# man = Manipulator(
#     link_lengths=link_lengths,
#     link_twists=link_twists,
#     joint_types=joint_types,
#     theta_offsets=[0.0, -sympy.pi / 2, 0],
# )
# man.dh_table()

# %%
