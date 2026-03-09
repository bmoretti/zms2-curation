"""
Plot traces from manual spot curations with predicted protein.

Each trace shows MS2 intensity and predicted protein over time for a single nucleus.
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

sys.path.insert(0, '/hpc/projects/group.royer/people/bruno.moretti/code/zebrafish-ms2')

from zebrafish_ms2.trace_analysis.trace_analysis import predict_protein

# Define path to config file
config_path = Path(__file__).parent.parent.parent.parent / "configs" / "manual_traces" / "2025-04-09_her7_crispr.yaml"

# Load config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set paths from config
spots_csv = config["paths"]["manual_spots_quantified"]
tracks_csv = config["paths"]["tracks"]
output_pdf = config["paths"]["output_manual_traces_with_protein"]

# Get parameters from config
intensity_method = config.get("intensity_method", "gauss3d_dog")
xlim = config.get("plot", {}).get("xlim", None)

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

    # Skip traces that are too short or have no signal
    if len(t_arr) < 10 or np.max(inten_arr) == 0:
        print(f"Skipping nucleus {nucleus_id}: trace too short or no signal")
        continue

    # Calculate protein prediction
    mrna, protein, fp = predict_protein(inten_arr, t_arr)

    traces.append((t_arr, inten_arr, protein, nucleus_id))

print(f"Generated {len(traces)} traces with protein predictions")

traces = sorted(traces, key=lambda x: x[3])

print(f"Plotting to {output_pdf}...")

with PdfPages(output_pdf) as pdf:
    for t_arr, inten_arr, protein, nucleus_id in traces:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize both traces to [0, 1]
        inten_norm = inten_arr / np.max(inten_arr) if np.max(inten_arr) > 0 else inten_arr
        protein_norm = protein / np.max(protein) if np.max(protein) > 0 else protein

        # Plot both on same axis
        ax.plot(t_arr, inten_norm, 'o-', linewidth=2, markersize=6,
               color='tab:blue', label='MS2')
        ax.plot(t_arr, protein_norm, 's-', linewidth=2, markersize=6,
               color='tab:orange', label='Protein')

        ax.set_xlabel('time (min)', fontsize=16)
        ax.set_ylabel('Normalized intensity (a.u)', fontsize=16)
        ax.set_title(f'Nucleus {int(nucleus_id)}', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set xlim if specified
        if xlim:
            ax.set_xlim(xlim)

        ax.legend(loc='upper left', fontsize=12)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"Done! PDF saved to {output_pdf}")
