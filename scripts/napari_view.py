"""
Open the neuroglancer-prepared data (re-chunked image + segment pyramid) in napari,
for comparing napari's browsing speed on the same files.

    /hpc/mydata/bruno.moretti/envs/napari-latest/bin/python napari_view.py \
        --config ../configs/2025-04-17_wt.yaml [--t 200] [--no-labels]

Needs zarr >= 3 (the files are zarr v3 with sharding). Contrast comes from the
config's `neuroglancer.contrast`, so nothing is computed from the data on open
(napari-ome-zarr scans the data for contrast limits, which takes minutes here).
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tensorstore as ts
import yaml
import zarr

N_LEVELS = 4


def load_config(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent / path
    with open(path) as f:
        return yaml.safe_load(f)


class LazyArray:
    """
    Minimal array-like over a tensorstore view, for napari.

    napari only needs shape/dtype/ndim and __getitem__. Reading through
    tensorstore decodes the needed chunks in parallel; going through dask was
    ~4x slower per plane on this data.
    """

    def __init__(self, store):
        self.store = store
        self.shape = tuple(int(s) for s in store.shape)
        self.dtype = np.dtype(store.dtype.numpy_dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        view = self.store[idx]
        if isinstance(view, ts.TensorStore):
            return view.read().result()
        return view

    def __array__(self, dtype=None, copy=None):
        data = self.store.read().result()
        return data.astype(dtype) if dtype is not None else data


def pyramid(group_path, channel=None):
    """One LazyArray per resolution level (optionally one channel of TCZYX)."""
    levels = []
    for level in range(N_LEVELS):
        store = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file',
                         'path': str(Path(group_path) / str(level))}}).result()
        if channel is not None:
            store = store[:, channel]
        levels.append(LazyArray(store))
    return levels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--t', type=int, default=0, help='initial timepoint')
    parser.add_argument('--no-labels', action='store_true', help='skip the segmentation layer')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ng_cfg = cfg.get('neuroglancer') or {}
    raw = Path(cfg['data']['raw_data'])
    ng_dir = Path(ng_cfg.get('ng_dir') or raw.parent / 'ng')
    image = Path(ng_cfg.get('image') or raw)
    contrast = ng_cfg.get('contrast') or {}

    group = zarr.open_group(str(image), mode='r')
    scale = [tr for tr in group.attrs.asdict().get('ome', group.attrs.asdict())['multiscales'][0]
             ['datasets'][0]['coordinateTransformations'] if tr['type'] == 'scale'][0]['scale']
    tzyx_scale = [scale[0]] + scale[2:]

    viewer = napari.Viewer(title=f"{cfg['experiment']['name']} (napari, zarr v3)")
    for c, (name, colormap) in enumerate((('H2B', 'magenta'), ('MCP', 'green'))):
        viewer.add_image(pyramid(image, channel=c), name=name, multiscale=True,
                         colormap=colormap, blending='additive', scale=tzyx_scale,
                         contrast_limits=contrast.get(name, [0, 4000]))
    seg_path = ng_dir / 'segments.ome.zarr'
    if not args.no_labels and seg_path.exists():
        viewer.add_labels(pyramid(seg_path), name='nuclei', multiscale=True,
                          scale=tzyx_scale, opacity=0.3)
    viewer.dims.set_point(0, args.t * tzyx_scale[0])
    return viewer


if __name__ == '__main__':
    main()
    napari.run()
