"""
Process manual spot curations: concatenate and quantify.
"""

import pandas as pd
import zarr
from pathlib import Path
import sys

sys.path.insert(0, '/hpc/projects/group.royer/people/bruno.moretti/code/zms2')

from zms2.spots.detection import extract_spot_voxels_from_zarr, check_voxel_boundary
from zms2.spots.quantification import quantify_spots

# ============================================================================
# CONFIG - Edit these paths
# ============================================================================

manual_curation_dir = '/hpc/projects/group.royer/people/bruno.moretti/data/epfl/2025-04-17/20250417_142938_Experiment/spots/manual_curation'

output_csv = '/hpc/projects/group.royer/people/bruno.moretti/data/epfl/2025-04-17/20250417_142938_Experiment/spots/manual_spots_quantified.csv'

# Path to zarr file for extracting voxel data
zarr_path = '/hpc/projects/group.royer/people/bruno.moretti/data/epfl/2025-04-17/20250417_142938_Experiment/stabilized.zarr/0'

# Which channel contains the spots (MCP)
spot_channel = 1

# ============================================================================
# Processing
# ============================================================================

# Concatenate all manual curation CSV files
curation_dir = Path(manual_curation_dir)
csv_files = list(curation_dir.glob('manual_points_df_*.csv'))

print(f"Found {len(csv_files)} manual curation files")

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

print(f"Total spots: {len(df)}")

# Filter out boundary spots before extraction, so df and voxel_df stay aligned
print("Checking for boundary spots...")
locations = df[['t', 'z', 'y', 'x']].values
voxel_size = (9, 11, 11)  # default used by extract_spot_voxels_from_zarr

data = zarr.open(zarr_path, 'r')
if len(data.shape) == 5:
    im_shape = data.shape[2:]
else:
    im_shape = data.shape[1:]

keep_mask = [not check_voxel_boundary(im_shape, loc[1:], voxel_size) for loc in locations]
n_excluded = len(df) - sum(keep_mask)
if n_excluded > 0:
    print(f"Warning: {n_excluded} spots were too close to boundaries and excluded")
df = df[keep_mask].reset_index(drop=True)

# Extract voxel data from zarr
print("Extracting voxel data from zarr...")
locations = df[['t', 'z', 'y', 'x']].values
voxel_df = extract_spot_voxels_from_zarr(zarr_path, locations, spot_channel=spot_channel)

df['data'] = voxel_df['data'].values
print(f"Extracted voxels for {len(df)} spots")

# Run quantify_spots
print("Quantifying spots with method='gauss3d_dog'...")
df = quantify_spots(df, method='gauss3d_dog')

# Drop the 'data' column before saving (contains arrays that can't be saved to CSV)
df = df.drop('data', axis=1)

# Save as CSV
df.to_csv(output_csv, index=False)

print(f"Saved to {output_csv}")
