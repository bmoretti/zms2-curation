# MS2 Spot Curation Tools

Tools for manually curating MS2 spots, in neuroglancer (`ng_*.py`, below) or napari
(the original per-nucleus workflow, further down).

## Neuroglancer curation

All nuclei of a dataset live in one viewer that streams the full timelapse in a
browser, locally or over an SSH tunnel. Each nucleus is a segment of the ultrack
segmentation (segment id = `track_id`). Spots are 4D point annotations linked to
their nucleus, and only the selected nucleus's spots are shown.

### Setup (once)

```bash
conda create -p /hpc/mydata/$USER/envs/zms2-ng python=3.12 pip
/hpc/mydata/$USER/envs/zms2-ng/bin/pip install neuroglancer tensorstore "zarr>=3" pandas pyarrow pyyaml pytest
```

### Prepare a dataset (once per dataset)

`ng_prepare.py` writes everything to `<experiment>/ng/` and never modifies the original data.
Run it from `scripts/` in that env:

```bash
python ng_prepare.py tracks   --config ../configs/2025-04-17_wt.yaml  # tracks_index.parquet
python ng_prepare.py spots    --config ../configs/2025-04-17_wt.yaml  # zms2 auto spots -> annotations
python ng_prepare.py segprops --config ../configs/2025-04-17_wt.yaml  # per-nucleus properties
python ng_prepare.py segments --config ../configs/2025-04-17_wt.yaml --init
python ng_prepare.py segments --config ../configs/2025-04-17_wt.yaml  # ~5 s/timepoint (wt: ~35 min)
```

- **Why segments are converted:** the ultrack `segments.zarr` has 2 GB chunks and no pyramid, which a browser cannot stream. The conversion writes a sharded OME-Zarr 0.5 label pyramid instead. To split the work, use `ng_prepare.sbatch` (see its header).
- **Image (recommended for every dataset):** re-chunk it into small blocks so stepping in time
  is fast. The original 64x512x512 chunks make each time step download ~107 MB; the 16x64x64
  copy needs ~4 MB. It is a second full copy (~850 GB for wt), about 20-30 min as a SLURM array:
  `python ng_prepare.py images --config ../configs/X.yaml --init`, then
  `sbatch --array=0-19 ng_prepare.sbatch ../configs/X.yaml images 20`.
  Then set `neuroglancer.image` in the config to `<experiment>/ng/image.ome.zarr`.
- **Existing napari curations:** import them once with
  `python ng_utils.py import --config ../configs/X.yaml --from <dir with manual_points_df_*.csv>`.
  Nuclei already in the table are skipped.

Prepared so far: **wt (2025-04-17)**, including the re-chunked image.

### Curate

```bash
python ng_curate.py --config ../configs/2025-04-17_wt.yaml
```

It prints a viewer URL and an `ssh -L ...` command.
- **Laptop:** run the ssh command on the laptop, then open the URL.
- **HPC desktop on the same node:** open the URL directly.

Ports default to 9000 (viewer) and 9001 (data); change them with `--port` / `--data-port`.

The data server reads the next 3 timepoints (and the previous one) into memory as you step
through time, so `]` / `[` are mostly served from RAM (`--read-ahead N` changes this, 0 turns
it off). To see what the browser requests and how long each read takes, restart with
`--log-requests requests.log` (one line per request: start time, ms, status, bytes, method,
range, path, `hit` if served from memory).

**Neuroglancer's own controls** (hover over a panel):

| Key | Action |
|---|---|
| `[` / `]` | previous / next frame |
| `,` / `.` | previous / next z slice |
| double-click a nucleus | show/hide it; the most recently shown nucleus is the *active* one |
| right-click a layer tab | open its side panel (Rendering tab: contrast, colour, shader). A plain click hides/shows the layer |
| Seg tab | find nuclei by id, or sort/filter by `length`, `n_auto_spots`, `#has_auto_spots`, `#divides` |
| ctrl+click | place a spot (the point tool of the `curated` layer is always active) |
| ctrl+right-click | select a spot (details and delete button in the selection panel) |
| ctrl+alt+right-click | delete the spot under the mouse |
| alt+drag | move a spot |
| `alt+=` / `alt+-` | narrow / widen the z range in which spots are drawn in 2D |

**Added for curation:**

| Key | Action |
|---|---|
| `c` | follow nucleus on/off: recentres on the active nucleus whenever t changes |
| `y` / `u` | accept / reject the zms2 auto spot nearest the mouse at the current frame |
| `space` | jump to the next frame with an unreviewed auto spot (then: with no curated spot) |
| `m` | merge the nucleus under the mouse into the active one (tracking split) |
| `ctrl+s` | save now (edits also autosave after 2 s) |

**Layers:**
- `H2B` (magenta) and `MCP` (green): contrast sliders in each layer's tab.
- `nuclei`: curated nuclei are orange in the Seg list.
- `auto_spots`: zms2 spots drawn as rings. Cyan = passed filters, blue = fill-in, grey = failed. Its shader tab has a `min_prob` slider and fill-in / failed toggles.
- `curated`: your spots. Orange = manual, yellow = accepted auto spot, grey = rejected auto spot.

New spots are always linked to the active nucleus, even if you click just outside its mask.

**Where things are saved:**
- `ng/curated_spots.parquet`: every curated spot (`annotation_id, t, z, y, x, nucleus_id, status, source, auto_id, updated`) in voxel coordinates.
- `ng/merges.json`: nucleus merges.
- `ng/session.json`: view, contrast and active nucleus, restored on the next launch.
- `ng/backups/`: a copy of the table at most every 10 min.
- An autosave that would drop more than 20% of the spots at once is refused; press `ctrl+s` to confirm it.

### Export for quantification

```bash
python ng_utils.py export --config ../configs/2025-04-17_wt.yaml [--to DIR]
```

This writes one `manual_points_df_{nucleus_id}.csv` per nucleus (`t,z,y,x,nucleus_id`, spots only, rejected auto spots left out). The default location is `output.directory`. `process_manual_curations.py` and the plotting scripts read these unchanged.

### Tests

- `pytest tests/` runs the unit tests.
- `python tests/e2e_ng_curate.py --config configs/2025-04-17_wt.yaml --nucleus 187620` drives the real viewer in headless Chrome. It uses a temporary copy of the curation state.

## Napari workflow

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
