from typing import Tuple, Union
from nptyping import NDArray, Shape, Float

import numpy as np

VectorType = Union[Tuple[float, float], NDArray[Shape["2", Float]]]
Vector2DType = Union[Tuple[VectorType, VectorType], NDArray[Shape["2, 2"], Float]]

def line_slope(
    point_1: VectorType,
    point_2: VectorType,
    eps: float = 1e-6
) -> float:
    """Calculates the slope of the line defined by the two points.

    Args:
        point_1 (VectorType): Point of the line.
        point_2 (VectorType): Other point of the line
        eps (float): Constant to prevent division by 0. Defaults to 1e-6.

    Returns:
        float: Value of the line slope.
    """
    return (point_1[0] - point_2[0]) / (point_1[1] - point_2[1] + eps)

def line_intersection(
    line_1: Vector2DType,
    line_2: Vector2DType
):
    """Calculates the intersection point of two lines.

    Having two lines:
    - L1(x, y) = ax + by + c
    - L2(x, y) = dx + ey + f

    Let two points:
    - P1 = (x1, y1, 1)
    - P2 = (x2, y2, 1)

    With C being the cross product of two points, P1 and P2, that represents a line.
    It is known that P1 and P2 are both part of the line C, since C.P1 = (P1xP2).P1 = 0, and same for P2.

    Let R be the cross product of two lines, L1 and L2, that represents a point.
    R lies on L1, since R.L1 = (L1xL2).L1 = 0, and likewise for L2. Therefore, R must be the intersection point of the two lines.

    Args:
        line_1 (Vector2DType): Representation of the first line via two points of that line.
        line_2 (Vector2DType): Representation of the second line via two points of that line.

    Returns:
        Point2DType: The intersection point of the two lines, if it exists. Returns (inf, inf) if lines don't intersect.
    """
    matrix = np.vstack((line_1, line_2))  # Create 4x2 matrix containing the points
    matrix = np.hstack((matrix, np.ones((4, 1))))  # Pad with the third dimension to obtain 4x3 containing the points.

    l1 = np.cross(matrix[0], matrix[1])
    l2 = np.cross(matrix[2], matrix[3])
    x, y, z = np.cross(l1, l2)

    if z == 0:  # Parallel lines
        return np.array([np.inf, np.inf])

    return np.array([x / z, y / z])