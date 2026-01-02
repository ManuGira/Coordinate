"""Translation transformation utilities."""

import numpy as np


def translate2D(tx: float, ty: float) -> np.ndarray:
    """Creates a 2D translation matrix."""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])
