# MS2 Spot Curation Tools

Tools for manually curating MS2 spots in napari

## Workflow

### Curate spots for a specific nucleus

1. Edit `curate_nucleus_spots.py` and set the config file path and nucleus ID:
   ```python
   config_file = '../configs/2025-04-17_wt.yaml'
   nucleus_id = 1234
   ```

2. Run the script:
   ```bash
   python curate_nucleus_spots.py
   ```

3. This will:
   - Automatically compute a bounding box around the nucleus across all timepoints
   - Crop the image data to just that region (with padding for context)
   - Open napari with the cropped view

4. Navigate through the data:
   - **r/f** - Move up/down in Z-axis
   - **d/g** - Move backward/forward in Time

5. Add points manually:
   - Click the Points layer button
   - Click on MS2 spots you see in the MCP channel
   - Points are added in the cropped view

6. Save your work:
   - In the napari console or Python session, call: `save_curation()`
   - Points are automatically converted to global coordinates
   - Saved as CSV in the output directory

7. Resume later:
   - Run the script again with the same `nucleus_id`
   - Call `resume_curation()` to load your existing points
   - Continue adding more points
   - Save again when done

### Handle tracking errors (optional)

If tracking errors caused one biological nucleus to have multiple IDs:

```bash
python merge_curations.py 123 124 125
```

This merges the curation files and preserves metadata about the original IDs.

### Final processing

After all curations are complete, quantify MS2 intensity:

1. Edit `process_manual_curations.py` to set paths:
   - `manual_curation_dir` - Where your curation CSVs are
   - `zarr_path` - Path to your image data
   - `output_csv` - Where to save the final result

2. Run:
   ```bash
   python process_manual_curations.py
   ```

This concatenates all curation files and adds MS2 intensity measurements.

## Configuration

### YAML Config Files

The curation workflow uses YAML config files located in `configs/`. These files specify all dataset paths and curation settings. Example structure:

```yaml
# Experiment metadata
experiment:
  name: "wt"
  date: "2025-04-17"

# Data paths
data:
  raw_data: "/path/to/stabilized.zarr"
  segments: "/path/to/segments.zarr"
  tracks: "/path/to/tracks.csv"
  spots: "/path/to/spots.pkl"  # Optional

# Output directory
output:
  directory: "/path/to/manual_curation"

# Curation settings
curation:
  crop_padding:
    z: 50    # ~17 µm at 0.347 µm/pixel
    y: 75    # ~26 µm
    x: 75
  preload_to_memory: true  # Fast scrolling, uses more memory
  show_auto_spots: true    # Show automatic spots for comparison
  live_quantification: true  # Show intensity plot (requires zms2)
  spot_channel: 1            # Channel for spot signal (0=H2B, 1=MCP)

# Optional: path to zms2 package if not on PYTHONPATH
zms2_path: "/path/to/zms2"
```

### Live intensity plot

When `live_quantification` is enabled in the config, an intensity vs. time plot appears at the bottom of the napari window. Each time you add a manual spot, the plot updates with the quantified intensity at that point. This requires the [zms2](https://github.com/royerlab/zms2) package — set `zms2_path` in the config if it's not on your `PYTHONPATH`.

## Output

Curated points are saved as CSV files:
- Filename: `manual_points_df_{nucleus_id}.csv`
- Columns: `t`, `z`, `y`, `x`, `nucleus_id`
- Coordinates are in **global array indices** (not cropped coordinates)

## Requirements

- napari
- napari-ome-zarr
- pandas
- numpy
- zarr
- dask
