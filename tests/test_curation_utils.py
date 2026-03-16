"""Tests for curation coordinate math and bounding box computation."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add scripts/ to path so we can import curation_utils
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from curation_utils import (
    compute_padded_bounds,
    centroid_bounds_from_tracks,
    crop_to_global_coords,
    global_to_crop_coords,
)


# ============================================================================
# Coordinate conversion tests
# ============================================================================

class TestCoordinateConversion:
    """Test crop <-> global coordinate round-trips."""

    def test_crop_to_global_adds_offsets(self):
        points = np.array([[0, 0, 0, 0], [5, 10, 20, 30]])
        offsets = (100, 50, 200, 300)
        result = crop_to_global_coords(points, offsets)
        expected = np.array([[100, 50, 200, 300], [105, 60, 220, 330]])
        np.testing.assert_array_equal(result, expected)

    def test_global_to_crop_subtracts_offsets(self):
        points = np.array([[100, 50, 200, 300]])
        offsets = (100, 50, 200, 300)
        result = global_to_crop_coords(points, offsets)
        expected = np.array([[0, 0, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_round_trip_crop_global_crop(self):
        """Save then load should give back the original points."""
        original = np.array([
            [3, 12, 45, 67],
            [7, 8, 30, 55],
            [0, 0, 0, 0],
        ])
        offsets = (10, 25, 100, 200)
        global_pts = crop_to_global_coords(original, offsets)
        recovered = global_to_crop_coords(global_pts, offsets)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_round_trip_global_crop_global(self):
        """Load then save should preserve global coordinates."""
        original = np.array([[110, 75, 300, 500], [115, 80, 320, 510]])
        offsets = (10, 25, 100, 200)
        crop_pts = global_to_crop_coords(original, offsets)
        recovered = crop_to_global_coords(crop_pts, offsets)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_single_point(self):
        points = np.array([[5, 10, 15, 20]])
        offsets = (1, 2, 3, 4)
        result = crop_to_global_coords(points, offsets)
        np.testing.assert_array_equal(result, [[6, 12, 18, 24]])

    def test_fractional_coordinates(self):
        """Coordinates from napari can be fractional."""
        points = np.array([[2.5, 7.3, 14.8, 22.1]])
        offsets = (10, 20, 30, 40)
        global_pts = crop_to_global_coords(points, offsets)
        recovered = global_to_crop_coords(global_pts, offsets)
        np.testing.assert_array_almost_equal(recovered, points)


# ============================================================================
# Centroid bounds tests
# ============================================================================

class TestCentroidBounds:
    """Test bounding box computation from track coordinates."""

    def test_basic_bounds(self):
        t = np.array([5, 6, 7, 8])
        z = np.array([50.0, 51.0, 52.0, 53.0])
        y = np.array([100.0, 101.0, 102.0, 103.0])
        x = np.array([200.0, 201.0, 202.0, 203.0])
        result = centroid_bounds_from_tracks(t, z, y, x)
        assert result['t_min'] == 5
        assert result['t_max'] == 8
        assert result['z_min'] == 50
        assert result['z_max'] == 53
        assert result['y_min'] == 100
        assert result['y_max'] == 103
        assert result['x_min'] == 200
        assert result['x_max'] == 203

    def test_fractional_coords_floor_ceil(self):
        """Fractional centroids should be floored (min) and ceiled (max)."""
        t = np.array([0, 1])
        z = np.array([10.3, 12.7])
        y = np.array([20.1, 25.9])
        x = np.array([30.8, 35.2])
        result = centroid_bounds_from_tracks(t, z, y, x)
        assert result['z_min'] == 10  # floor(10.3)
        assert result['z_max'] == 13  # ceil(12.7)
        assert result['y_min'] == 20  # floor(20.1)
        assert result['y_max'] == 26  # ceil(25.9)
        assert result['x_min'] == 30  # floor(30.8)
        assert result['x_max'] == 36  # ceil(35.2)

    def test_single_timepoint(self):
        t = np.array([42])
        z = np.array([100.0])
        y = np.array([200.0])
        x = np.array([300.0])
        result = centroid_bounds_from_tracks(t, z, y, x)
        assert result['t_min'] == result['t_max'] == 42


# ============================================================================
# Padded bounds tests
# ============================================================================

class TestComputePaddedBounds:
    """Test padded bounding box computation with clamping."""

    def _make_centroid_bounds(self, t=(10, 20), z=(50, 60), y=(100, 110), x=(200, 210)):
        return {
            't_min': t[0], 't_max': t[1],
            'z_min': z[0], 'z_max': z[1],
            'y_min': y[0], 'y_max': y[1],
            'x_min': x[0], 'x_max': x[1],
        }

    def test_basic_padding(self):
        bounds = self._make_centroid_bounds()
        # Large data so no clamping needed
        data_shape = (100, 2, 500, 500, 500)
        result = compute_padded_bounds(bounds, padding=(10, 20, 30), data_shape=data_shape)

        assert result['offsets']['t'] == 10
        assert result['offsets']['z'] == 40   # 50 - 10
        assert result['offsets']['y'] == 80   # 100 - 20
        assert result['offsets']['x'] == 170  # 200 - 30

        assert result['slices']['t'] == slice(10, 21)  # t_max + 1
        assert result['slices']['z'] == slice(40, 71)   # 60 + 10 + 1
        assert result['slices']['y'] == slice(80, 131)   # 110 + 20 + 1
        assert result['slices']['x'] == slice(170, 241)  # 210 + 30 + 1

    def test_clamping_at_zero(self):
        """Padding shouldn't go below 0."""
        bounds = self._make_centroid_bounds(z=(5, 15))
        data_shape = (100, 2, 500, 500, 500)
        result = compute_padded_bounds(bounds, padding=(10, 20, 30), data_shape=data_shape)

        assert result['offsets']['z'] == 0  # max(0, 5 - 10) = 0
        assert result['slices']['z'].start == 0

    def test_clamping_at_data_edge(self):
        """Padding shouldn't exceed data dimensions."""
        bounds = self._make_centroid_bounds(z=(90, 95))
        data_shape = (100, 2, 100, 500, 500)  # z_size = 100
        result = compute_padded_bounds(bounds, padding=(10, 20, 30), data_shape=data_shape)

        assert result['slices']['z'].stop == 100  # min(100, 95 + 10 + 1)

    def test_clamping_both_sides(self):
        """When data is smaller than padding on both sides."""
        bounds = self._make_centroid_bounds(z=(2, 8))
        data_shape = (100, 2, 10, 500, 500)  # z_size = 10, so 2-50=clamped, 8+50=clamped
        result = compute_padded_bounds(bounds, padding=(50, 20, 30), data_shape=data_shape)

        assert result['offsets']['z'] == 0
        assert result['slices']['z'] == slice(0, 10)

    def test_shape_matches_slices(self):
        """Shape should be consistent with the slice ranges."""
        bounds = self._make_centroid_bounds()
        data_shape = (100, 2, 500, 500, 500)
        result = compute_padded_bounds(bounds, padding=(10, 20, 30), data_shape=data_shape)

        for dim, key in enumerate(['t', 'z', 'y', 'x']):
            s = result['slices'][key]
            assert result['shape'][dim] == s.stop - s.start

    def test_zero_padding(self):
        bounds = self._make_centroid_bounds(t=(5, 10), z=(50, 60), y=(100, 110), x=(200, 210))
        data_shape = (100, 2, 500, 500, 500)
        result = compute_padded_bounds(bounds, padding=(0, 0, 0), data_shape=data_shape)

        # With zero padding, spatial bounds are just centroid bounds + 1 (exclusive stop)
        assert result['slices']['z'] == slice(50, 61)
        assert result['slices']['y'] == slice(100, 111)
        assert result['slices']['x'] == slice(200, 211)
