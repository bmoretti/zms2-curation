"""
Custom key bindings for napari viewer to enable easy navigation through 4D data.

This module provides intuitive key bindings for scrolling through time and z dimensions:
- r/f: Move up/down in z-axis
- d/g: Move backward/forward in time

These bindings use overwrite=True to avoid conflicts with napari's default point layer bindings.

Usage:
    from napari_keybindings import setup_navigation_keybindings

    viewer = napari.Viewer()
    setup_navigation_keybindings(viewer)
"""

import napari


def setup_navigation_keybindings(viewer: napari.Viewer):
    """
    Setup custom key bindings for 4D navigation in napari.

    Key bindings (override points layer defaults):
        r: Move up in z (decrease z index)
        f: Move down in z (increase z index)
        d: Move backward in time (decrease t index)
        g: Move forward in time (increase t index)

    Args:
        viewer: napari.Viewer instance
    """

    @viewer.bind_key('r', overwrite=True)
    def move_z_up(viewer):
        """Move up in z-axis (decrease z index)."""
        current_step = list(viewer.dims.current_step)
        # Assume dims order is (t, z, y, x) for 4D data
        if len(current_step) >= 2:  # At least t and z dimensions
            z_idx = 1  # z is the second dimension
            if current_step[z_idx] > 0:
                current_step[z_idx] -= 1
                viewer.dims.current_step = tuple(current_step)

    @viewer.bind_key('f', overwrite=True)
    def move_z_down(viewer):
        """Move down in z-axis (increase z index)."""
        current_step = list(viewer.dims.current_step)
        if len(current_step) >= 2:
            z_idx = 1
            z_max = viewer.dims.range[z_idx][1] - 1  # Max index
            if current_step[z_idx] < z_max:
                current_step[z_idx] += 1
                viewer.dims.current_step = tuple(current_step)

    @viewer.bind_key('d', overwrite=True)
    def move_time_backward(viewer):
        """Move backward in time (decrease t index)."""
        current_step = list(viewer.dims.current_step)
        if len(current_step) >= 1:
            t_idx = 0  # t is the first dimension
            if current_step[t_idx] > 0:
                current_step[t_idx] -= 1
                viewer.dims.current_step = tuple(current_step)

    @viewer.bind_key('g', overwrite=True)
    def move_time_forward(viewer):
        """Move forward in time (increase t index)."""
        current_step = list(viewer.dims.current_step)
        if len(current_step) >= 1:
            t_idx = 0
            t_max = viewer.dims.range[t_idx][1] - 1  # Max index
            if current_step[t_idx] < t_max:
                current_step[t_idx] += 1
                viewer.dims.current_step = tuple(current_step)

    print("\n" + "="*70)
    print("Custom navigation key bindings enabled:")
    print("  r/f: Navigate up/down in Z-axis")
    print("  d/g: Navigate backward/forward in Time")
    print("="*70 + "\n")


def setup_save_keybinding(viewer: napari.Viewer, save_callback):
    """
    Setup Ctrl+S key binding to save curation data.

    Args:
        viewer: napari.Viewer instance
        save_callback: Function to call when Ctrl+S is pressed (e.g., save_curation)
    """

    @viewer.bind_key('Control-s')
    def save_on_ctrl_s(viewer):
        """Save curation when Ctrl+S is pressed."""
        save_callback()

    print("Save key binding enabled: Ctrl+S")
