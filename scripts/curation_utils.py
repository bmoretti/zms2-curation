"""
Pure computation functions for curation coordinate math.

Extracted from curate_nucleus_spots.py for testability.
"""

import numpy as np


def compute_padded_bounds(centroid_bounds, padding, data_shape):
    """
    Compute padded and clamped bounding box from centroid bounds.

    Args:
        centroid_bounds: dict with keys 't_min', 't_max', 'z_min', 'z_max',
                         'y_min', 'y_max', 'x_min', 'x_max'
        padding: (pad_z, pad_y, pad_x) in pixels
        data_shape: (t_size, c_size, z_size, y_size, x_size)

    Returns:
        Dictionary with 'slices', 'offsets', and 'shape'
    """
    pad_z, pad_y, pad_x = padding
    t_size, _, z_size, y_size, x_size = data_shape

    t_min_pad = max(0, centroid_bounds['t_min'])
    t_max_pad = min(t_size, centroid_bounds['t_max'] + 1)
    z_min_pad = max(0, centroid_bounds['z_min'] - pad_z)
    z_max_pad = min(z_size, centroid_bounds['z_max'] + pad_z + 1)
    y_min_pad = max(0, centroid_bounds['y_min'] - pad_y)
    y_max_pad = min(y_size, centroid_bounds['y_max'] + pad_y + 1)
    x_min_pad = max(0, centroid_bounds['x_min'] - pad_x)
    x_max_pad = min(x_size, centroid_bounds['x_max'] + pad_x + 1)

    crop_shape = (
        t_max_pad - t_min_pad,
        z_max_pad - z_min_pad,
        y_max_pad - y_min_pad,
        x_max_pad - x_min_pad,
    )

    return {
        'slices': {
            't': slice(t_min_pad, t_max_pad),
            'z': slice(z_min_pad, z_max_pad),
            'y': slice(y_min_pad, y_max_pad),
            'x': slice(x_min_pad, x_max_pad),
        },
        'offsets': {
            't': t_min_pad,
            'z': z_min_pad,
            'y': y_min_pad,
            'x': x_min_pad,
        },
        'shape': crop_shape,
    }


def centroid_bounds_from_tracks(t_indices, z_coords, y_coords, x_coords):
    """
    Compute centroid bounding box from track coordinate arrays.

    Args:
        t_indices, z_coords, y_coords, x_coords: arrays of coordinates

    Returns:
        Dictionary with t_min, t_max, z_min, z_max, y_min, y_max, x_min, x_max
    """
    return {
        't_min': int(np.min(t_indices)),
        't_max': int(np.max(t_indices)),
        'z_min': int(np.floor(np.min(z_coords))),
        'z_max': int(np.ceil(np.max(z_coords))),
        'y_min': int(np.floor(np.min(y_coords))),
        'y_max': int(np.ceil(np.max(y_coords))),
        'x_min': int(np.floor(np.min(x_coords))),
        'x_max': int(np.ceil(np.max(x_coords))),
    }


def crop_to_global_coords(points, offsets):
    """
    Convert points from crop coordinates to global coordinates.

    Args:
        points: (N, 4) array with columns (t, z, y, x) in crop space
        offsets: (t_offset, z_offset, y_offset, x_offset)

    Returns:
        (N, 4) array in global coordinates
    """
    points = np.asarray(points, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    return points + offsets


def global_to_crop_coords(points, offsets):
    """
    Convert points from global coordinates to crop coordinates.

    Args:
        points: (N, 4) array with columns (t, z, y, x) in global space
        offsets: (t_offset, z_offset, y_offset, x_offset)

    Returns:
        (N, 4) array in crop coordinates
    """
    points = np.asarray(points, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    return points - offsets
