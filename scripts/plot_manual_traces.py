"""
Plot traces from manual spot curations.

Each trace shows intensity over time for a single nucleus.
For timepoints where the nucleus exists but no spot was manually curated,
intensity is set to 0.

Optionally overlays automatic detection traces for comparison when
'compare_auto' is enabled and 'auto_spots' path is provided in config.
"""

import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, '/hpc/projects/group.royer/people/bruno.moretti/code/zms2')

# Define path to config file
config_path = Path(__file__).parent.parent / "configs" / "2025-04-09_her7_crispr.yaml"

# Load config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set paths from config
plot_cfg = config["plotting"]
spots_csv = plot_cfg["manual_spots_quantified"]
tracks_csv = config["data"]["tracks"]
output_pdf = plot_cfg["output_manual_traces"]
auto_spots_path = config["data"].get("spots", None)

# Get parameters from config
intensity_method = plot_cfg.get("intensity_method", "gauss3d_dog")
overlay_traces = plot_cfg.get("overlay_traces", False)
compare_auto = plot_cfg.get("compare_auto", False)
xlim = plot_cfg.get("xlim", None)
normalize = plot_cfg.get("normalize", False)

spots_df = pd.read_csv(spots_csv)
tracks_df = pd.read_csv(tracks_csv)

# Load automatic spots if comparison is enabled
auto_spots_df = None
if compare_auto:
    if auto_spots_path is None:
        print("Warning: compare_auto is enabled but no 'auto_spots' path in config, skipping auto traces")
        compare_auto = False
    else:
        print(f"Loading automatic spots from {auto_spots_path}...")
        auto_spots_df = pd.read_pickle(auto_spots_path)
        # Keep only spots assigned to a nucleus
        auto_spots_df = auto_spots_df.dropna(subset=['nucleus_id'])
        print(f"Loaded {len(auto_spots_df)} automatic spots with nucleus assignments")

nucleus_ids = spots_df['nucleus_id'].unique()

print(f"Found {len(nucleus_ids)} nuclei with manual curations")


def build_traces(spots, tracks, nucleus_ids, intensity_col):
    """Build intensity traces for given nucleus IDs."""
    result = []
    for nucleus_id in nucleus_ids:
        nucleus_spots = spots[spots['nucleus_id'] == nucleus_id]
        nucleus_track = tracks[tracks['track_id'] == nucleus_id]

        if len(nucleus_track) == 0:
            continue

        t_min = nucleus_track['t'].min()
        t_max = nucleus_track['t'].max()
        t_arr = np.arange(t_min, t_max + 1)
        inten_arr = np.zeros(len(t_arr))

        for i, t in enumerate(t_arr):
            spots_at_t = nucleus_spots[nucleus_spots['t'] == t]
            if len(spots_at_t) > 0:
                inten_arr[i] = spots_at_t[intensity_col].dropna().max()

        # Replace any NaN with 0 (e.g. timepoints where all spot values were NaN)
        np.nan_to_num(inten_arr, copy=False, nan=0.0)
        result.append((t_arr, inten_arr, nucleus_id))
    return result


manual_traces = build_traces(spots_df, tracks_df, nucleus_ids, intensity_method)
print(f"Generated {len(manual_traces)} manual traces")

auto_traces_dict = {}
if compare_auto:
    auto_traces = build_traces(auto_spots_df, tracks_df, nucleus_ids, intensity_method)
    auto_traces_dict = {nid: (t, inten) for t, inten, nid in auto_traces}
    print(f"Generated {len(auto_traces)} automatic traces")

traces = sorted(manual_traces, key=lambda x: x[2])

print(f"Plotting to {output_pdf}...")

def normalize_trace(inten_arr):
    """Normalize trace to 0-1 range."""
    max_val = np.nanmax(inten_arr)
    return inten_arr / max_val if max_val > 0 else inten_arr


with PdfPages(output_pdf) as pdf:
    if overlay_traces:
        # Plot all traces in a single plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for t_arr, inten_arr, nucleus_id in traces:
            inten_plot = normalize_trace(inten_arr) if normalize else inten_arr
            ax.plot(t_arr, inten_plot, 'o-', linewidth=2, markersize=6, label=f'Nucleus {int(nucleus_id)}', alpha=0.7)
        ax.set_xlabel('time (min)', fontsize=16)
        ylabel = 'Normalized MS2 intensity (a.u)' if normalize else 'MS2 intensity (a.u)'
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title('All Manual Traces', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    else:
        # Plot each trace separately
        for t_arr, inten_arr, nucleus_id in traces:
            fig, ax = plt.subplots(figsize=(10, 6))

            inten_plot = normalize_trace(inten_arr) if normalize else inten_arr

            if compare_auto and nucleus_id in auto_traces_dict:
                # Overlay automatic and manual traces
                auto_t, auto_inten = auto_traces_dict[nucleus_id]
                auto_plot = normalize_trace(auto_inten) if normalize else auto_inten
                ax.plot(auto_t, auto_plot, 'o-', color='tab:blue', linewidth=2, markersize=6, alpha=0.5, label='Automatic')
                ax.plot(t_arr, inten_plot, 'o-', color='tab:orange', linewidth=2, markersize=6, label='Manual')
                ax.legend(fontsize=12)
            else:
                ax.plot(t_arr, inten_plot, 'o-', linewidth=2, markersize=6)

            ax.set_xlabel('time (min)', fontsize=16)
            ylabel = 'Normalized MS2 intensity (a.u)' if normalize else 'MS2 intensity (a.u)'
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(f'Nucleus {int(nucleus_id)}', fontsize=20)
            ax.tick_params(labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if xlim:
                ax.set_xlim(xlim)

            plt.tight_layout()
            pdf.savefig()
            plt.close()
