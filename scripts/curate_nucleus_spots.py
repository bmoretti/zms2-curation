"""
Curate MS2 spots for a specific nucleus on cropped views.

This script loads a cropped region around a specific nucleus across all timepoints,
making curation much faster than working with the full dataset.

Workflow:
1. Set nucleus_id in the CONFIG section below
2. Run the script - it will automatically crop to the nucleus region
3. Add points manually on the MS2 spots you see
4. The script auto-saves periodically, or you can call save_curation() manually
5. Points are automatically converted to global coordinates when saved

The crop includes padding around the nucleus to provide spatial context.
"""

import napari
import pandas as pd
import numpy as np
import zarr
import dask.array as da
import yaml
from pathlib import Path
from napari_keybindings import setup_navigation_keybindings, setup_save_keybinding
from curation_utils import (
    compute_padded_bounds, centroid_bounds_from_tracks,
    crop_to_global_coords, global_to_crop_coords,
)


# ============================================================================
# CONFIG - Edit these for your dataset and nucleus
# ============================================================================
# Path to YAML config file for the dataset
config_file = '../configs/curation/2025-04-17_wt.yaml'

# Nucleus ID to curate
nucleus_id = 187620 # SET THIS! e.g., 1234

# ============================================================================
# Load configuration from YAML
# ============================================================================
def load_config(config_path):
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    # If relative path, resolve relative to script location
    if not config_file.is_absolute():
        script_dir = Path(__file__).parent.parent.parent  # Go up to repo root
        config_file = script_dir / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config

# Load config
config = load_config(config_file)

# Extract paths from config
data_path = config['data']['raw_data']
segments_path = config['data']['segments']
tracks_path = config['data']['tracks']
spots_path = config['data'].get('spots')  # None if not specified
out_dir = config['output']['directory']

# Extract curation settings
crop_padding = (
    config['curation']['crop_padding']['z'],
    config['curation']['crop_padding']['y'],
    config['curation']['crop_padding']['x']
)
preload_to_memory = config['curation']['preload_to_memory']
show_auto_spots = config['curation']['show_auto_spots']


# ============================================================================
# Global state
# ============================================================================
viewer = None           # Napari viewer instance
crop_offsets = None     # (t_offset, z_offset, y_offset, x_offset) for coordinate conversion
nucleus_bounds = None   # Store bounds info for debugging


# ============================================================================
# Helper functions
# ============================================================================

def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)


def _get_save_path(nuc_id):
    """Get the save path for a nucleus."""
    return Path(out_dir) / f'manual_points_df_{nuc_id}.csv'


def load_automatic_spots(nucleus_id):
    """
    Load automatic spots for a given nucleus from the spots pickle file.

    Args:
        nucleus_id: Single nucleus ID to load spots for

    Returns:
        DataFrame with spots (t, z, y, x) or None if no spots file
    """
    if spots_path is None or not Path(spots_path).exists():
        return None

    print(f"\nLoading automatic spots for nucleus {nucleus_id}...")
    spots_df = pd.read_pickle(spots_path)

    # Filter to this nucleus
    nucleus_spots = spots_df[spots_df['nucleus_id'] == nucleus_id].copy()

    if len(nucleus_spots) == 0:
        print(f"  No automatic spots found for nucleus {nucleus_id}")
        return None

    print(f"  Found {len(nucleus_spots)} automatic spots (t={nucleus_spots['t'].min()}-{nucleus_spots['t'].max()})")

    return nucleus_spots[['t', 'z', 'y', 'x', 'spot_id', 'nucleus_id']]


def compute_nucleus_bounds(nucleus_id, padding=(50, 75, 75)):
    """
    Compute the bounding box for a nucleus across all timepoints.

    Args:
        nucleus_id: ID of the nucleus
        padding: (z, y, x) padding in pixels

    Returns:
        Dictionary with crop slices and offsets
    """
    print(f"\nComputing bounding box for nucleus {nucleus_id}...")

    # Load tracks CSV
    print(f"Loading tracks from {tracks_path}...")
    tracks_df = pd.read_csv(tracks_path)

    # Filter to this nucleus using track_id (matches segmentation labels)
    nucleus_tracks = tracks_df[tracks_df['track_id'] == nucleus_id]

    if len(nucleus_tracks) == 0:
        raise ValueError(f"Nucleus {nucleus_id} not found in tracks!")

    print(f"Found nucleus in {len(nucleus_tracks)} timepoints")

    # Extract coordinates and compute centroid bounds
    centroid_bounds = centroid_bounds_from_tracks(
        nucleus_tracks['t'].values,
        nucleus_tracks['z'].values,
        nucleus_tracks['y'].values,
        nucleus_tracks['x'].values,
    )

    print(f"Centroid bounds: t=[{centroid_bounds['t_min']}, {centroid_bounds['t_max']}], "
          f"z=[{centroid_bounds['z_min']}, {centroid_bounds['z_max']}], "
          f"y=[{centroid_bounds['y_min']}, {centroid_bounds['y_max']}], "
          f"x=[{centroid_bounds['x_min']}, {centroid_bounds['x_max']}]")

    # Load data to get full shape
    data_zarr = zarr.open(data_path, 'r')
    if isinstance(data_zarr, zarr.hierarchy.Group):
        data_zarr = data_zarr['0']

    # Compute padded and clamped bounds
    result = compute_padded_bounds(centroid_bounds, padding, data_zarr.shape)

    s = result['slices']
    print(f"Padded bounds: t=[{s['t'].start}, {s['t'].stop}), z=[{s['z'].start}, {s['z'].stop}), "
          f"y=[{s['y'].start}, {s['y'].stop}), x=[{s['x'].start}, {s['x'].stop})")
    print(f"Crop shape (t, z, y, x): {result['shape']}")

    return result


def load_cropped_data(nucleus_id, padding=(50, 75, 75)):
    """
    Load cropped image data around a nucleus.

    Args:
        nucleus_id: ID of the nucleus to crop around
        padding: (z, y, x) padding in pixels

    Returns:
        Dictionary with cropped data arrays and metadata
    """
    global nucleus_bounds, crop_offsets

    # Compute bounds
    bounds = compute_nucleus_bounds(nucleus_id, padding)
    nucleus_bounds = bounds
    crop_offsets = (bounds['offsets']['t'],
                   bounds['offsets']['z'],
                   bounds['offsets']['y'],
                   bounds['offsets']['x'])

    # Load image data lazily
    print("\nLoading cropped image data...")
    data_zarr = zarr.open(data_path, 'r')
    if isinstance(data_zarr, zarr.hierarchy.Group):
        data_zarr = data_zarr['0']

    # Convert to dask array FIRST to ensure lazy slicing
    data_dask = da.from_zarr(data_zarr)

    # Crop: [t, c, z, y, x]
    slices = bounds['slices']
    cropped_data = data_dask[slices['t'], :, slices['z'], slices['y'], slices['x']]

    print(f"Cropped data shape: {cropped_data.shape}")

    # Optionally preload into memory for faster scrolling
    if preload_to_memory:
        print("Preloading data into memory for fast scrolling...")
        print(f"  Estimated memory: ~{cropped_data.nbytes / 1e9:.2f} GB")
        cropped_data = cropped_data.compute()
        print("  ✓ Data loaded into memory")

    # Load segmentation crop lazily
    seg_cropped = None
    if segments_path and Path(segments_path).exists():
        print("Loading cropped segmentation...")
        seg_zarr = zarr.open(segments_path, 'r')
        if isinstance(seg_zarr, zarr.hierarchy.Group):
            seg_zarr = seg_zarr['0']

        # Convert to dask array FIRST to ensure lazy slicing
        seg_dask = da.from_zarr(seg_zarr)
        seg_cropped = seg_dask[slices['t'], slices['z'], slices['y'], slices['x']]

        print(f"Cropped segmentation shape: {seg_cropped.shape}")

        # Optionally preload segmentation into memory too
        if preload_to_memory:
            print("Preloading segmentation into memory...")
            print(f"  Estimated memory: ~{seg_cropped.nbytes / 1e9:.2f} GB")
            seg_cropped = seg_cropped.compute()
            print("  ✓ Segmentation loaded into memory")

    return {
        'data': cropped_data,
        'segmentation': seg_cropped,
        'bounds': bounds,
    }


def save_curation():
    """
    Save the current points layer to disk in global coordinates.

    Returns:
        DataFrame with the saved points
    """
    if nucleus_id is None:
        print("Error: nucleus_id not set in config!")
        return None

    if crop_offsets is None:
        print("Error: No crop loaded. This should not happen.")
        return None

    # Get points from the last points layer (excluding Auto_spots)
    points_layer = None
    for layer in reversed(viewer.layers):
        if isinstance(layer, napari.layers.Points) and layer.name != 'Auto_spots':
            points_layer = layer
            break

    if points_layer is None:
        print("Error: No manual points layer found. Add points before saving.")
        print("  (Auto_spots layer is excluded from saving)")
        return None

    points = points_layer.data

    if len(points) == 0:
        print("Warning: No points to save.")
        return None

    # Convert from crop coordinates to global coordinates
    global_points = crop_to_global_coords(points, crop_offsets)

    df = pd.DataFrame()
    df['t'] = global_points[:, 0]
    df['z'] = global_points[:, 1]
    df['y'] = global_points[:, 2]
    df['x'] = global_points[:, 3]
    df['nucleus_id'] = nucleus_id

    # Save
    _ensure_output_dir()
    save_path = _get_save_path(nucleus_id)
    df.to_csv(save_path, index=False)

    print(f"\n✓ Saved {len(points)} points for nucleus {nucleus_id} to {save_path}")
    print(f"  Points converted from crop coordinates to global coordinates")
    print(f"  Offsets applied: t+{t_offset}, z+{z_offset}, y+{y_offset}, x+{x_offset}")

    return df


def resume_curation():
    """
    Resume curation by loading existing points for the current nucleus.
    Points are automatically converted from global to crop coordinates.

    Returns:
        DataFrame with the loaded points (in global coordinates)
    """
    if nucleus_id is None:
        print("Error: nucleus_id not set in config!")
        return None

    if crop_offsets is None:
        print("Error: No crop loaded. This should not happen.")
        return None

    save_path = _get_save_path(nucleus_id)
    if not save_path.exists():
        print(f"No saved data found for nucleus {nucleus_id}")
        return None

    # Load existing points (in global coordinates)
    df = pd.read_csv(save_path)

    # Convert from global coordinates to crop coordinates
    t_offset, z_offset, y_offset, x_offset = crop_offsets

    points_data = np.column_stack([
        df['t'].values - t_offset,
        df['z'].values - z_offset,
        df['y'].values - y_offset,
        df['x'].values - x_offset,
    ])

    # Add to viewer
    viewer.add_points(points_data, name=f'nucleus_{nucleus_id}_points', size=4)

    print(f"\n✓ Loaded {len(points_data)} existing points for nucleus {nucleus_id}")
    print(f"  Points converted from global to crop coordinates")

    return df


# ============================================================================
# Initialize viewer with cropped data
# ============================================================================

def init_viewer():
    """Initialize napari viewer with cropped data for the specified nucleus."""
    global viewer

    if nucleus_id is None:
        raise ValueError("nucleus_id must be set in the CONFIG section!")

    print("\n" + "="*70)
    print(f"CURATION MODE - Nucleus {nucleus_id}")
    print("="*70)

    crop_data = load_cropped_data(nucleus_id, padding=crop_padding)

    viewer = napari.Viewer()

    # Setup custom key bindings for easy navigation
    setup_navigation_keybindings(viewer)
    setup_save_keybinding(viewer, save_curation)

    # Add image channels
    data = crop_data['data']

    # Get scale from original data metadata
    data_zarr = zarr.open(data_path, 'r')
    if isinstance(data_zarr, zarr.hierarchy.Group):
        # Get scale from OME-Zarr metadata in the multiscales structure
        if 'multiscales' in data_zarr.attrs:
            multiscales = data_zarr.attrs['multiscales']
            # Get the first multiscale (usually there's only one)
            if len(multiscales) > 0:
                datasets = multiscales[0].get('datasets', [])
                # Find the dataset with path "0" (highest resolution)
                for dataset in datasets:
                    if dataset.get('path') == '0':
                        transforms = dataset.get('coordinateTransformations', [])
                        # Look for scale transform
                        for transform in transforms:
                            if transform.get('type') == 'scale':
                                scale_values = transform['scale']
                                # OME-Zarr scale is (t, c, z, y, x), we need (t, z, y, x)
                                # Skip the channel dimension
                                scale = (scale_values[0], scale_values[2], scale_values[3], scale_values[4])
                                print(f"Read scale from OME-Zarr metadata: {scale}")
                                break
                        else:
                            # Fallback if no scale transform found
                            scale = (1.0, 1.0, 0.347, 0.347)
                            print(f"Warning: No scale transform found in dataset, using default: {scale}")
                        break
                else:
                    # Fallback if no dataset with path "0" found
                    scale = (1.0, 1.0, 0.347, 0.347)
                    print(f"Warning: No dataset with path '0' found, using default: {scale}")
            else:
                scale = (1.0, 1.0, 0.347, 0.347)
                print(f"Warning: Empty multiscales metadata, using default: {scale}")
        else:
            scale = (1.0, 1.0, 0.347, 0.347)
            print(f"Warning: No multiscales in metadata, using default: {scale}")
    else:
        scale = (1.0, 1.0, 0.347, 0.347)
        print(f"Warning: Data is not a Zarr group, using default scale: {scale}")

    # Add H2B channel (channel 0)
    viewer.add_image(
        data[:, 0, :, :, :],
        name='H2B',
        colormap='magenta',
        blending='additive',
        opacity=0.47,
        scale=scale
    )

    # Add MCP channel (channel 1)
    viewer.add_image(
        data[:, 1, :, :, :],
        name='MCP',
        colormap='green',
        blending='additive',
        contrast_limits=(0, 3500),
        scale=scale
    )

    # Add segmentation if available
    if crop_data['segmentation'] is not None:
        viewer.add_labels(
            crop_data['segmentation'],
            name='Segments',
            opacity=0.3,
            scale=scale
        )

    # Add automatic spots if available and requested
    if show_auto_spots and spots_path:
        # Load spots for the nucleus ID
        auto_spots_df = load_automatic_spots(nucleus_id)

        if auto_spots_df is not None and len(auto_spots_df) > 0:
            # Convert to crop coordinates
            t_offset, z_offset, y_offset, x_offset = crop_offsets

            auto_points = np.column_stack([
                auto_spots_df['t'].values - t_offset,
                auto_spots_df['z'].values - z_offset,
                auto_spots_df['y'].values - y_offset,
                auto_spots_df['x'].values - x_offset,
            ])

            # Add as a points layer (different color/size from manual)
            viewer.add_points(
                auto_points,
                name='Auto_spots',
                size=3,
                face_color='cyan',
                edge_color='white',
                edge_width=0.5,
                opacity=0.7,
                scale=scale
            )
            print(f"\n  Added {len(auto_points)} automatic spots (cyan) for comparison")

    print("\n" + "="*70)
    print("Napari viewer initialized with cropped data")
    print("="*70)
    print("\nWorkflow:")
    print("  1. Add a Points layer (Ctrl+P or via the layer controls)")
    print("  2. Manually mark MS2 spots by clicking")
    print("  3. Press Ctrl+S to save (or call save_curation() manually)")
    print("  4. To resume existing work, call resume_curation()")
    print("\nNavigation:")
    print("  r/f: Move up/down in Z-axis")
    print("  d/g: Move backward/forward in Time")
    print("\nNote: Points are automatically converted to global coordinates when saved.")
    print("="*70 + "\n")

    return viewer


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    init_viewer()

    # Check if there's existing data to resume
    if _get_save_path(nucleus_id).exists():
        print("\n! Found existing curation data for this nucleus.")
        print("  Call resume_curation() to load existing points,")
        print("  or start fresh by adding a new points layer.\n")

    # Start the napari event loop
    napari.run()
