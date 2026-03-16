"""
Live intensity quantification widget for napari-based spot curation.

Shows a real-time plot of spot intensity vs time as spots are curated.
Each time a point is added or removed, the plot updates automatically.

Requires the zms2 package for voxel extraction and intensity quantification.
"""

import numpy as np
import pandas as pd
import napari
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from curation_utils import crop_to_global_coords


class LiveIntensityWidget:
    """Dock widget showing live intensity quantification of curated spots."""

    def __init__(self, viewer, zarr_path, spot_channel, crop_offsets,
                 extract_fn, quantify_fn, method='gauss3d_dog'):
        """
        Args:
            viewer: napari.Viewer instance
            zarr_path: Path to zarr array (e.g., .../stabilized.zarr/0)
            spot_channel: Channel index for spot signal (e.g., 1 for MCP)
            crop_offsets: (t, z, y, x) offsets from crop to global coordinates
            extract_fn: Function to extract voxel patches (extract_spot_voxels_from_zarr)
            quantify_fn: Function to quantify spots (quantify_spots)
            method: Quantification method name (default: 'gauss3d_dog')
        """
        self.viewer = viewer
        self.zarr_path = zarr_path
        self.spot_channel = spot_channel
        self.crop_offsets = crop_offsets
        self.extract_fn = extract_fn
        self.quantify_fn = quantify_fn
        self.method = method

        # Cache: (t, z, y, x) rounded global coords -> intensity value
        self._cache = {}
        self._connected_layers = set()

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(111)
        self._setup_axes()
        self.canvas = FigureCanvasQTAgg(self.fig)

        # Add as dock widget
        viewer.window.add_dock_widget(
            self.canvas, name='Intensity Trace', area='bottom'
        )

        # Watch for new layers being added
        viewer.layers.events.inserted.connect(self._on_layer_inserted)

        # Connect to any existing manual points layers
        for layer in viewer.layers:
            self._try_connect_layer(layer)

    def _setup_axes(self):
        """Configure plot axes styling."""
        self.ax.set_xlabel('Time', fontsize=12)
        self.ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        self.ax.set_title('Live Intensity Trace', fontsize=14)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()

    def _on_layer_inserted(self, event):
        """Handle new layer being added to viewer."""
        self._try_connect_layer(event.value)

    def _try_connect_layer(self, layer):
        """Connect to a Points layer if it's a manual curation layer."""
        if (isinstance(layer, napari.layers.Points)
                and layer.name != 'Auto_spots'
                and id(layer) not in self._connected_layers):
            layer.events.data.connect(self._on_data_change)
            self._connected_layers.add(id(layer))
            # If layer already has data (e.g., from resume), process it
            if len(layer.data) > 0:
                self._on_data_change(None)

    def _collect_manual_points(self):
        """Gather all points from manual curation layers."""
        all_points = []
        for layer in self.viewer.layers:
            if (isinstance(layer, napari.layers.Points)
                    and layer.name != 'Auto_spots'
                    and len(layer.data) > 0):
                all_points.append(layer.data)
        if all_points:
            return np.vstack(all_points)
        return np.empty((0, 4))

    def _on_data_change(self, event):
        """Called when points data changes (point added or removed)."""
        crop_points = self._collect_manual_points()

        if len(crop_points) == 0:
            self._cache.clear()
            self._update_plot()
            return

        # Convert to global coordinates
        global_points = crop_to_global_coords(crop_points, self.crop_offsets)

        # Build set of current point keys (rounded global coords)
        current_keys = set()
        new_points = []
        new_keys = []

        for pt in global_points:
            key = tuple(np.round(pt).astype(int))
            current_keys.add(key)
            if key not in self._cache:
                new_points.append(pt)
                new_keys.append(key)

        # Remove cached entries for deleted points
        for key in set(self._cache.keys()) - current_keys:
            del self._cache[key]

        # Quantify new points
        if new_points:
            self._quantify_points(np.array(new_points), new_keys)

        self._update_plot()

    def _quantify_points(self, global_coords, keys):
        """Extract voxels and quantify intensity for new points one at a time."""
        for pt, key in zip(global_coords, keys):
            try:
                locations = pt[np.newaxis, :4]  # (1, 4) array
                voxel_df = self.extract_fn(
                    self.zarr_path, locations, spot_channel=self.spot_channel
                )
                if len(voxel_df) == 0:
                    # Point too close to boundary
                    continue

                df = pd.DataFrame({'data': voxel_df['data'].values})
                df = self.quantify_fn(df, method=self.method)

                if self.method in df.columns:
                    self._cache[key] = df[self.method].iloc[0]
            except Exception as e:
                print(f"Warning: Quantification failed for point at t={key[0]}: {e}")

    def _update_plot(self):
        """Redraw the intensity vs time plot from cached values."""
        self.ax.clear()
        self._setup_axes()

        if self._cache:
            # Sort by time (first element of key tuple)
            sorted_items = sorted(self._cache.items(), key=lambda item: item[0][0])
            times = [key[0] for key, _ in sorted_items]
            intensities = [val for _, val in sorted_items]

            self.ax.plot(times, intensities, 'o-', color='green',
                         linewidth=1.5, markersize=6)

        self.fig.tight_layout()
        self.canvas.draw_idle()
