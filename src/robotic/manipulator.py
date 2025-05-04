# %%
import sys
from enum import Enum, auto
from typing import Dict, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from loguru import logger

from robotic import Scalar
from robotic.transformations import (
    AxisAngleSpec,
    EulerOrder,
    EulerSequence,
    HomogeneousTransformation,
    Rotation,
    Translation,
    X,
    Y,
    Z,
)

logger.remove()
logger.add(sys.stderr, format="{level} | {message}", colorize=True)


class DHTable(pd.DataFrame):
    # Required for pandas subclassing
    _metadata = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If needed, convert entries to sympy objects
        for col in ["a", r"\alpha", "d", r"\theta"]:
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
    frames: Optional[Sequence[HomogeneousTransformation]] = None
    n: int

    def __init__(
        self,
        joint_types: Sequence[JointType],
        x_rotations: Sequence[Scalar],
        x_offsets: Sequence[Scalar],
        z_rotations: Sequence[Scalar],
        z_offsets: Sequence[Scalar],
        *,
        link_dimensions: Optional[Sequence[Scalar]] = None,
    ):
        """
        Initialize a serial Manipulator using Denavit-Hartenberg parameters.

        Parameters:
        ----------
        joint_types : Sequence[JointType]
            A list specifying the type of each joint (Revolute or Prismatic).

        x_rotations : Sequence[Scalar]
            The twist angles αᵢ (in radians) — rotation around the local x-axis
            from zᵢ₋₁ to zᵢ.
            (i.e., how much you tilt between consecutive joint axes.)

        x_offsets : Sequence[Scalar]
            The link lengths aᵢ (in meters) — distance along the local x-axis
            from zᵢ₋₁ to zᵢ.
            (i.e., how far the two axes are separated horizontally.)

        z_rotations : Sequence[Scalar]
            The rotation offsets θᵢ (in radians) — additional constant rotations around the local z-axis.
            (i.e., initial rotations needed to align xᵢ₋₁ to xᵢ.)

        z_offsets : Sequence[Scalar]
            The link offsets dᵢ (in meters) — fixed distances along the z-axis between consecutive frames.
            (i.e., how far two frames are separated vertically.)

        link_dimensions : Optional[Sequence[Scalar]]
            (Optional) Physical lengths used for graphical visualization of the robot.
            Defaults to unit lengths if not provided.

        Notes:
        -----
        - For Revolute joints, θᵢ is variable (depends on qᵢ), and dᵢ is constant.
        - For Prismatic joints, dᵢ is variable (depends on qᵢ), and θᵢ is constant.
        - This constructor expects that the values provided already represent
          the *fixed parts* of the D-H parameters, and symbolic variables qᵢ will be added accordingly.

        - Typical mapping to D-H table:
            - aᵢ → x_offsets[i]
            - αᵢ → x_rotations[i]
            - dᵢ → z_offsets[i] (constant part if Revolute)
            - θᵢ → z_rotations[i] (constant part if Prismatic)
        """
        self.joint_types = joint_types
        self.n = len(joint_types)
        if len(x_rotations) < self.n:
            raise ValueError("...")
        if len(x_rotations) > self.n:
            logger.warning("...")
        self.x_rotations = x_rotations
        if len(x_offsets) < self.n:
            raise ValueError("...")
        if len(x_offsets) > self.n:
            logger.warning("...")
        self.x_offsets = x_offsets
        if len(z_rotations) < self.n:
            raise ValueError("...")
        if len(z_rotations) > self.n:
            logger.warning("...")
        self.z_rotations = z_rotations
        if len(z_offsets) < self.n:
            raise ValueError("...")
        if len(z_offsets) > self.n:
            logger.warning("...")
        self.z_offsets = z_offsets

        self.joints = [sympy.symbols(f"q_{i + 1}") for i in range(self.n)]
        self.link_dimensions = link_dimensions or [1.0] * self.n

    @staticmethod
    def from_rotations(
        joint_types: Sequence[JointType],
        rotations: Sequence[Rotation],
        *,
        solution: Literal[0, 1] = 0,
    ) -> "Manipulator":
        logger.warning(
            "This method is still a WIP and might now work properly, double check."
        )
        sequence = EulerSequence.XYZ
        order = EulerOrder.FIXED
        x_rotations = []
        z_rotations = []
        for rotation in rotations:
            euler_spec = rotation.to_euler(sequence, order)
            if isinstance(euler_spec, tuple):
                theta1, _, theta3 = euler_spec[solution].euler_angles
            else:
                theta1, _, theta3 = euler_spec.euler_angles
            x_rotations.append(theta1)
            z_rotations.append(theta3)

        return Manipulator(
            joint_types=joint_types,
            x_rotations=x_rotations,
            x_offsets=[0] * len(joint_types),
            z_rotations=z_rotations,
            z_offsets=[0] * len(joint_types),
        )

    def with_translational_offset(
        self,
        x_offsets: Optional[Sequence[Scalar]] = None,
        z_offsets: Optional[Sequence[Scalar]] = None,
    ) -> "Manipulator":
        if x_offsets is None:
            x_offsets = self.x_offsets
        if z_offsets is None:
            z_offsets = self.z_offsets
        return Manipulator(
            joint_types=self.joint_types,
            x_rotations=self.x_rotations,
            x_offsets=x_offsets,
            z_rotations=self.z_rotations,
            z_offsets=z_offsets,
        )

    def with_rotations(
        self,
        x_rotations: Optional[Sequence[Scalar]] = None,
        z_rotations: Optional[Sequence[Scalar]] = None,
    ) -> "Manipulator":
        if x_rotations is None:
            x_rotations = self.x_rotations
        if z_rotations is None:
            z_rotations = self.z_rotations
        return Manipulator(
            joint_types=self.joint_types,
            x_rotations=x_rotations,
            x_offsets=self.x_offsets,
            z_rotations=z_rotations,
            z_offsets=self.z_offsets,
        )

    def dh_table(self) -> DHTable:
        if self._dh_table is None:
            rows = []
            for i, joint_type in enumerate(self.joint_types):
                a = self.x_offsets[i]
                alpha = self.x_rotations[i]
                q_i = self.joints[i]
                theta_offset = self.z_rotations[i]
                d_offset = self.z_offsets[i]

                if joint_type == JointType.REVOLUTE:
                    theta = q_i + theta_offset
                    d = d_offset
                else:
                    theta = theta_offset
                    d = q_i + d_offset

                rows.append((a, alpha, d, theta, joint_type.name))
            self._dh_table = DHTable(
                rows, columns=["a", r"\alpha", "d", r"\theta", "type"]
            )
        return self._dh_table

    def dh_matrix(self, simplify: bool = True) -> HomogeneousTransformation:
        T = HomogeneousTransformation.identity()
        self.frames = [T]
        for _, row in self.dh_table().iterrows():
            T = T @ (
                HomogeneousTransformation.from_rotation(
                    Rotation.from_axis_angle(AxisAngleSpec(Z, row[r"\theta"]))
                )
                @ HomogeneousTransformation.from_translation(
                    Translation(0, 0, row["d"])
                )
                @ HomogeneousTransformation.from_translation(
                    Translation(row["a"], 0, 0)
                )
                @ HomogeneousTransformation.from_rotation(
                    Rotation.from_axis_angle(AxisAngleSpec(X, row[r"\alpha"]))
                )
            )
        if simplify:
            T = HomogeneousTransformation(sympy.simplify(T))
        self.frames.append(T)
        return T

    def plot_planar(
        self,
        joint_values: Sequence[Scalar],
        symbolic_values: Optional[Dict] = None,
        link_dimensions: Optional[Sequence[Scalar]] = None,
    ):
        subs = {q: val for q, val in zip(self.joints, joint_values)}
        if symbolic_values:
            subs.update(symbolic_values)
        if not link_dimensions:
            link_dimensions = [1.0] * self.n
        x, y = 0.0, 0.0
        theta = 0.0
        points = [(x, y)]

        for i in range(len(self.joint_types)):
            joint_type = self.joint_types[i]
            q_i = self.joints[i]
            q_val = float(sympy.N(q_i.subs(subs)))

            if joint_type == JointType.REVOLUTE:
                theta += q_val
                dx = self.link_dimensions[i] * np.cos(theta)
                dy = self.link_dimensions[i] * np.sin(theta)
            elif joint_type == JointType.PRISMATIC:
                dx = (self.link_dimensions[i] + q_val) * np.cos(theta)  # type: ignore
                dy = (self.link_dimensions[i] + q_val) * np.sin(theta)  # type: ignore
            else:
                raise ValueError(f"Unknown joint type: {joint_type}")

            x += dx
            y += dy
            points.append((x, y))

        xs, ys = zip(*points)

        # Plot setup
        fig, ax = plt.subplots()
        ax.plot(xs, ys, color="black", linewidth=2, zorder=1)
        # Joint visualization
        r, p = True, True
        for i, (x, y) in enumerate(points[:-1]):
            jt = self.joint_types[i]

            if jt == JointType.REVOLUTE:
                ax.scatter(
                    x, y, color="red", s=30, marker="o", label="Revolute" if r else ""
                )
                r = False
            elif jt == JointType.PRISMATIC:
                ax.scatter(
                    x,
                    y,
                    color="green",
                    s=30,
                    marker="s",
                    label="Prismatic" if p else "",
                )
                p = False

        # End-effector point (optional: make it distinct)
        ax.scatter(
            xs[-1], ys[-1], color="blue", s=100, marker="*", label="End-effector"
        )

        # Axes and styling
        max_range = max(max(map(abs, xs)), max(map(abs, ys))) + 0.5
        ax.axhline(0, color="black", linewidth=0.8, zorder=0)
        ax.axvline(0, color="black", linewidth=0.8, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(False)

        # Legend
        ax.legend(loc="upper left", frameon=False)
        ax.set_title("Planar Manipulator (XY view)")
        plt.show()


# joint_types = [
#     JointType.REVOLUTE,
#     JointType.REVOLUTE,
# ]
# x_offsets = [0, 0]
# x_rotations = [
#     0,
#     0,
# ]

# z_offsets = [0, 0]
# z_rotations = [
#     0,
#     0,
# ]

# man = Manipulator(
#     joint_types=joint_types,
#     x_rotations=x_rotations,
#     x_offsets=x_offsets,
#     z_rotations=z_rotations,
#     z_offsets=z_offsets,
# )

# man.plot_workspace(
#     [-sympy.pi / 2, sympy.pi / 2],
# )

# man_l = Manipulator(
#     [JointType.REVOLUTE, JointType.REVOLUTE, JointType.REVOLUTE],
#     [0, 0, 0],
#     [sympy.symbols("L_0"), sympy.symbols("L_{l1}"), sympy.symbols("L_{l2}")],
#     [0, 0, sympy.pi / 2],
#     [0, 0, 0],
# )
# print(man_l.dh_matrix())
