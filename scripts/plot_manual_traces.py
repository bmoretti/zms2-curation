"""
Plot traces from manual spot curations.

Each trace shows intensity over time for a single nucleus.
For timepoints where the nucleus exists but no spot was manually curated,
intensity is set to 0.
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
config_path = Path(__file__).parent.parent.parent.parent / "configs" / "manual_traces" / "2025-04-09_her7_crispr.yaml"

# Load config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set paths from config
spots_csv = config["paths"]["manual_spots_quantified"]
tracks_csv = config["paths"]["tracks"]
output_pdf = config["paths"]["output_manual_traces"]

# Get parameters from config
intensity_method = config.get("intensity_method", "gauss3d_dog")
overlay_traces = config.get("overlay_traces", False)
xlim = config.get("plot", {}).get("xlim", None)
normalize = config.get("plot", {}).get("normalize", False)

spots_df = pd.read_csv(spots_csv)
tracks_df = pd.read_csv(tracks_csv)

nucleus_ids = spots_df['nucleus_id'].unique()

print(f"Found {len(nucleus_ids)} nuclei with manual curations")

traces = []

for nucleus_id in nucleus_ids:
    nucleus_spots = spots_df[spots_df['nucleus_id'] == nucleus_id]
    nucleus_track = tracks_df[tracks_df['track_id'] == nucleus_id]

    if len(nucleus_track) == 0:
        print(f"Warning: Nucleus {nucleus_id} not found in tracks, skipping")
        continue

    # Create time array covering full nucleus lifetime
    t_min = nucleus_track['t'].min()
    t_max = nucleus_track['t'].max()
    t_arr = np.arange(t_min, t_max + 1)
    inten_arr = np.zeros(len(t_arr))

    # Fill in measured intensities (leave zeros where no spot was curated)
    for i, t in enumerate(t_arr):
        spots_at_t = nucleus_spots[nucleus_spots['t'] == t]
        if len(spots_at_t) > 0:
            inten_arr[i] = spots_at_t[intensity_method].max()

    traces.append((t_arr, inten_arr, nucleus_id))

print(f"Generated {len(traces)} traces")

traces = sorted(traces, key=lambda x: x[2])

print(f"Plotting to {output_pdf}...")

with PdfPages(output_pdf) as pdf:
    if overlay_traces:
        # Plot all traces in a single plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for t_arr, inten_arr, nucleus_id in traces:
            if normalize:
                inten_plot = inten_arr / np.max(inten_arr) if np.max(inten_arr) > 0 else inten_arr
            else:
                inten_plot = inten_arr
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
        # Plot each trace separately (original behavior)
        for t_arr, inten_arr, nucleus_id in traces:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Normalize if requested
            if normalize:
                inten_plot = inten_arr / np.max(inten_arr) if np.max(inten_arr) > 0 else inten_arr
            else:
                inten_plot = inten_arr

            ax.plot(t_arr, inten_plot, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('time (min)', fontsize=16)
            ylabel = 'Normalized MS2 intensity (a.u)' if normalize else 'MS2 intensity (a.u)'
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(f'Nucleus {int(nucleus_id)}', fontsize=20)
            ax.tick_params(labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set xlim if specified
            if xlim:
                ax.set_xlim(xlim)

            max_inten = inten_arr.max()
            n_nonzero = np.sum(inten_arr > 0)

            plt.tight_layout()
            pdf.savefig()
            plt.close()
