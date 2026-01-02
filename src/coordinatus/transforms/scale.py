"""Scaling and shearing transformation utilities."""

import numpy as np


def scale2D(sx: float, sy: float) -> np.ndarray:
    """Creates a 2D scaling matrix."""
    return np.array([[sx, 0,  0],
                     [0, sy,  0],
                     [0,  0, 1]])


def shear2D(kx: float, ky: float) -> np.ndarray:
    """Creates a 2D shear matrix."""
    return np.array([[1, kx, 0],
                     [ky, 1, 0],
                     [0,  0, 1]])
