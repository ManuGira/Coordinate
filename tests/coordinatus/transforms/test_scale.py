"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    scale2D, shear2D,
)

class TestScale2D:
    """Tests for the scale2D function."""

    def test_scale_identity(self):
        """Test scaling matrix with unit scale."""
        S = scale2D(1, 1)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_uniform(self):
        """Test uniform scaling."""
        S = scale2D(2, 2)
        expected = np.array([[2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        S = scale2D(3, 2)
        expected = np.array([[3, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_zero(self):
        """Test scaling to zero."""
        S = scale2D(0, 0)
        expected = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_negative(self):
        """Test negative scaling (reflection)."""
        S = scale2D(-1, 1)
        expected = np.array([[-1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_point(self):
        """Test that scaling matrix correctly scales a point."""
        S = scale2D(2, 3)
        point = np.array([4, 5, 1])
        result = S @ point
        expected = np.array([8, 15, 1])  # Point scaled by (2, 3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_fractional(self):
        """Test scaling with fractional values."""
        S = scale2D(0.5, 0.25)
        expected = np.array([[0.5,  0,    0],
                            [0,    0.25, 0],
                            [0,    0,    1]])
        np.testing.assert_array_almost_equal(S, expected)


class TestShear2D:
    """Tests for the shear2D function."""

    def test_shear_identity(self):
        """Test shear matrix with zero shear."""
        K = shear2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_x_only(self):
        """Test shear in x direction only."""
        K = shear2D(0.5, 0)
        expected = np.array([[1, 0.5, 0],
                            [0, 1,   0],
                            [0, 0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_y_only(self):
        """Test shear in y direction only."""
        K = shear2D(0, 0.5)
        expected = np.array([[1,   0, 0],
                            [0.5, 1, 0],
                            [0,   0, 1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_both_axes(self):
        """Test shear in both x and y directions."""
        K = shear2D(0.3, 0.7)
        expected = np.array([[1,   0.3, 0],
                            [0.7, 1,   0],
                            [0,   0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_negative(self):
        """Test shear with negative values."""
        K = shear2D(-0.5, -0.25)
        expected = np.array([[1,     -0.5, 0],
                            [-0.25, 1,    0],
                            [0,     0,    1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_point_x(self):
        """Test that shear matrix correctly shears a point in x direction."""
        K = shear2D(1, 0)  # Shear x by y amount
        point = np.array([0, 2, 1])  # Point at (0, 2)
        result = K @ point
        expected = np.array([2, 2, 1])  # x shifted by y*kx = 2*1 = 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_point_y(self):
        """Test that shear matrix correctly shears a point in y direction."""
        K = shear2D(0, 1)  # Shear y by x amount
        point = np.array([3, 0, 1])  # Point at (3, 0)
        result = K @ point
        expected = np.array([3, 3, 1])  # y shifted by x*ky = 3*1 = 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_vector(self):
        """Test that shear matrix correctly shears a vector."""
        K = shear2D(0.5, 0.5)
        vector = np.array([2, 2, 0])  # Vector (w=0)
        result = K @ vector
        expected = np.array([3, 3, 0])  # x += y*0.5, y += x*0.5
        np.testing.assert_array_almost_equal(result, expected)

