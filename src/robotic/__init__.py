"""Hand-implemented robot kinematics that reads like a handwritten exam solution."""

from robotic.geometry import Frame, FreeAxis, Point, UnitVector, Vector, ZeroVectorError
from robotic.rotation import (
    AxisAngle,
    AxisAngleSolutions,
    Index3,
    Rotation,
    ValidityCondition,
)
from robotic.symbolic import (
    Compiled,
    MissingParameterError,
    Numeric,
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
    "Frame",
    "FreeAxis",
    "Index3",
    "Index4",
    "MissingParameterError",
    "Numeric",
    "ParameterTypeError",
    "Point",
    "Rotation",
    "Scalar",
    "ScalarLike",
    "Symbol",
    "Symbolic",
    "Transform",
    "UnitVector",
    "ValidityCondition",
    "Vector",
    "ZeroVectorError",
]
