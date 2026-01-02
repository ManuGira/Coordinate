"""Rotation transformation utilities."""

import numpy as np


def rotate2D(angle_rad: float) -> np.ndarray:
    """Creates a 2D rotation matrix."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])
