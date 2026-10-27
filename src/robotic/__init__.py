"""Hand-implemented robot kinematics that reads like a handwritten exam solution."""

from robotic.geometry import (
    Direction,
    Displacement,
    Frame,
    FrameMismatchError,
    FreeAxis,
    Point,
    Position,
    ZeroVectorError,
)
from robotic.rotation import AxisAngle, AxisAngleSolutions, Index3, Rotation
from robotic.symbolic import (
    Compiled,
    MissingParameterError,
    Numeric,
    ParameterShapeError,
    ParameterTypeError,
    Scalar,
    ScalarLike,
    Symbol,
    Symbolic,
)
from robotic.transform import Index4, Transform

__all__ = [
    "AxisAngle",
    "AxisAngleSolutions",
    "Compiled",
    "Direction",
    "Displacement",
    "Frame",
    "FrameMismatchError",
    "FreeAxis",
    "Index3",
    "Index4",
    "MissingParameterError",
    "Numeric",
    "ParameterShapeError",
    "ParameterTypeError",
    "Point",
    "Position",
    "Rotation",
    "Scalar",
    "ScalarLike",
    "Symbol",
    "Symbolic",
    "Transform",
    "ZeroVectorError",
]
