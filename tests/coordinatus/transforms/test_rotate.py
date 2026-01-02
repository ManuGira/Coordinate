"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    rotate2D,
)


class TestRotate2D:
    """Tests for the rotate2D function."""

    def test_rotate_zero(self):
        """Test rotation matrix with zero angle."""
        R = rotate2D(0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_90_degrees(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        R = rotate2D(np.pi / 2)
        expected = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_180_degrees(self):
        """Test rotation by 180 degrees (π radians)."""
        R = rotate2D(np.pi)
        expected = np.array([[-1,  0, 0],
                            [ 0, -1, 0],
                            [ 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_270_degrees(self):
        """Test rotation by 270 degrees (3π/2 radians)."""
        R = rotate2D(3 * np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_360_degrees(self):
        """Test rotation by 360 degrees (2π radians) returns to identity."""
        R = rotate2D(2 * np.pi)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_45_degrees(self):
        """Test rotation by 45 degrees (π/4 radians)."""
        R = rotate2D(np.pi / 4)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([[sqrt2_over_2, -sqrt2_over_2, 0],
                            [sqrt2_over_2,  sqrt2_over_2, 0],
                            [0,             0,            1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_point(self):
        """Test that rotation matrix correctly rotates a point."""
        R = rotate2D(np.pi / 2)  # 90 degrees
        point = np.array([1, 0, 1])  # Point at (1, 0)
        result = R @ point
        expected = np.array([0, 1, 1])  # Point rotated to (0, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_negative_angle(self):
        """Test rotation with negative angle (clockwise)."""
        R = rotate2D(-np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

