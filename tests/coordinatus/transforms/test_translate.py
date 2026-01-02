"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    translate2D
)

class TestTranslate2D:
    """Tests for the translate2D function."""

    def test_translate_zero(self):
        """Test translation matrix with zero translation."""
        T = translate2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_positive(self):
        """Test translation with positive values."""
        T = translate2D(3, 5)
        expected = np.array([[1, 0, 3],
                            [0, 1, 5],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_negative(self):
        """Test translation with negative values."""
        T = translate2D(-2, -4)
        expected = np.array([[1, 0, -2],
                            [0, 1, -4],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_point(self):
        """Test that translation matrix correctly transforms a point."""
        T = translate2D(3, 2)
        point = np.array([1, 1, 1])  # Homogeneous coordinates
        result = T @ point
        expected = np.array([4, 3, 1])  # Point moved by (3, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_vector(self):
        """Test that translation matrix doesn't affect vectors (w=0)."""
        T = translate2D(3, 2)
        vector = np.array([1, 1, 0])  # Homogeneous coordinates for vector
        result = T @ vector
        expected = np.array([1, 1, 0])  # Vector unchanged
        np.testing.assert_array_almost_equal(result, expected)

