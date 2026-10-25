"""Hand-implemented robot kinematics that reads like a handwritten exam solution."""

from robotic.geometry import Frame, FreeAxis, Point, UnitVector, Vector, ZeroVectorError
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

__all__ = [
    "Compiled",
    "Frame",
    "FreeAxis",
    "MissingParameterError",
    "Numeric",
    "ParameterTypeError",
    "Point",
    "Scalar",
    "ScalarLike",
    "Symbol",
    "Symbolic",
    "UnitVector",
    "Vector",
    "ZeroVectorError",
]
