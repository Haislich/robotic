from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sympy
from mpl_toolkits.mplot3d import Axes3D

from robotic.transformations import Rotation


def draw_frame(
    rotation: Rotation = Rotation(
        sympy.Matrix(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    ),
    label: str = "",
    *,
    length: float = 2.5,
    basis=Rotation(
        sympy.Matrix(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
    ),
    plot=True,
):
    """
    Draw a right-handed coordinate frame with X→right, Y↑, Z⊙out of screen.

    Parameters:
        origin: 3-element list or array indicating the frame origin
        rotation: 3x3 rotation matrix to apply on top of screen-oriented frame
        length: arrow length for each axis
    """
    fig = plt.figure()
    ax: Axes3D = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    ax.view_init(elev=30, azim=-60)

    origin = np.array([0, 0, 0])

    # Apply rotation on top of default screen-oriented frame
    frame = np.array(basis) @ np.array(rotation)

    # Scaled axes
    x_axis = frame[:, 0] * length
    y_axis = frame[:, 1] * length
    z_axis = frame[:, 2] * length

    # Draw axes
    ax.quiver(*origin, *x_axis, color="r", label="X")
    ax.quiver(*origin, *y_axis, color="g", label="Y")
    ax.quiver(*origin, *z_axis, color="b", label="Z")

    ax.set_xlim((-1.5, 1.5))
    ax.set_xticks([])
    ax.set_ylim((-1.5, 1.5))
    ax.set_yticks([])
    ax.set_zlim((-1.5, 1.5))
    ax.zaxis.set_ticks([])

    ax.set_axis_off()

    ax.set_title(label)
    ax.legend()
    if plot:
        plt.show()
    return Rotation(sympy.Matrix(frame))
