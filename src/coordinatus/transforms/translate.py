"""Translation transformation utilities."""

import numpy as np
from numpy.typing import ArrayLike

def translate(translation_vector: ArrayLike) -> np.ndarray:
    """Creates a translation matrix for an n-dimensional space."""
    translation_vector = np.asarray(translation_vector)
    dim = translation_vector.shape[0]
    T = np.eye(dim + 1)
    T[:-1, -1] = translation_vector
    return T

def translate2D(tx: float, ty: float) -> np.ndarray:
    """
    Creates a 2D translation matrix.
        [[1, 0, tx]
         [0, 1, ty]
         [0, 0,  1]]
    """
    return translate([tx, ty])

def translate3D(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Creates a 3D translation matrix.
        [[1, 0, 0, tx]
         [0, 1, 0, ty]
         [0, 0, 1, tz]
         [0, 0, 0,  1]]
    """
    return translate([tx, ty, tz])