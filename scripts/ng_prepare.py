"""
One-time conversion of a dataset into formats neuroglancer streams well.

Everything is written to <experiment>/ng/ (see ng_utils.ng_paths); the original
data is never modified.

Steps (run in this order; `segments` is the slow one, see ng_prepare.sbatch):
    python ng_prepare.py tracks   --config ../configs/2025-04-17_wt.yaml
    python ng_prepare.py spots    --config ../configs/2025-04-17_wt.yaml
    python ng_prepare.py segprops --config ../configs/2025-04-17_wt.yaml
    python ng_prepare.py segments --config ../configs/2025-04-17_wt.yaml --init
    python ng_prepare.py segments --config ../configs/2025-04-17_wt.yaml --t-start 0 --t-stop 20
    python ng_prepare.py images   --config ../configs/2025-09-26_dapt.yaml --init / --t-start ...

Outputs:
    tracks_index.parquet      track_id, t, z, y, x, parent_track_id (voxel coords)
    auto_spots.parquet        zms2 auto spots, auto_id = annotation id in the viewer
    auto_spots/               precomputed point annotations (T,Z,Y,X), related to nuclei
    segment_properties/       precomputed segment properties for every track_id
    segments.ome.zarr         OME-Zarr 0.5 (zarr v3, sharded) label pyramid
    image.ome.zarr            (only with `images`) rechunked image pyramid
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorstore as ts

import neuroglancer
import neuroglancer.write_annotations

from ng_utils import (
    load_config,
    ng_paths,
    ng_units_and_scales,
    read_ome_multiscales,
    spatiotemporal_axes,
    voxel_to_ng,
)


N_LEVELS = 4
SEG_CHUNK_ZYX = (64, 256, 256)
# Small blocks: a timepoint step only downloads the voxels near the displayed planes
# (measured on wt: 3.8 MB per step vs 107 MB with the original 64x512x512 chunks)
IMG_CHUNK_ZYX = (16, 64, 64)


# ============================================================================
# Tracks
# ============================================================================

def prepare_tracks(paths):
    print(f"Reading {paths['tracks_src']} ...")
    df = pd.read_csv(paths['tracks_src'],
                     usecols=['track_id', 't', 'z', 'y', 'x', 'parent_track_id'],
                     engine='pyarrow')
    df = df.sort_values(['track_id', 't'], kind='stable').reset_index(drop=True)
    paths['ng_dir'].mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths['tracks_index'], index=False)
    print(f"Wrote {len(df)} rows / {df['track_id'].nunique()} tracks -> {paths['tracks_index']}")


# ============================================================================
# Auto spots -> precomputed annotations
# ============================================================================

def annotation_coordinate_space(paths):
    axes = spatiotemporal_axes(paths['raw_image'])
    units, scales = ng_units_and_scales(axes)
    return neuroglancer.CoordinateSpace(names=axes['names'], units=units, scales=scales)


AUTO_SPOT_PROPERTIES = [
    neuroglancer.AnnotationPropertySpec(id='prob', type='float32',
                                        description='zms2 classifier probability'),
    neuroglancer.AnnotationPropertySpec(id='gauss3d_dog', type='float32',
                                        description='zms2 intensity (gauss3d_dog)'),
    neuroglancer.AnnotationPropertySpec(id='spot_id', type='uint32',
                                        description='zms2 spot_id (not unique)'),
    neuroglancer.AnnotationPropertySpec(id='fill_in', type='uint8', enum_values=[0, 1],
                                        enum_labels=['detected', 'fill-in']),
    neuroglancer.AnnotationPropertySpec(id='passed_filters', type='uint8', enum_values=[0, 1],
                                        enum_labels=['failed', 'passed']),
]


def load_auto_spots(spots_path):
    """zms2 spots pickle -> flat table with a unique auto_id (voxel coords)."""
    df = pd.read_pickle(spots_path)
    out = pd.DataFrame({
        'auto_id': np.arange(len(df), dtype=np.uint64),
        'spot_id': df['spot_id'].astype('int64').to_numpy(),
        't': df['t'].astype('int64').to_numpy(),
        'z': df['z'].astype('float64').to_numpy(),
        'y': df['y'].astype('float64').to_numpy(),
        'x': df['x'].astype('float64').to_numpy(),
        'nucleus_id': df['nucleus_id'].fillna(-1).astype('int64').to_numpy(),
        'prob': df['prob'].astype('float32').to_numpy(),
        'gauss3d_dog': df['gauss3d_dog'].fillna(0).astype('float32').to_numpy(),
        'fill_in_spot': df['fill_in_spot'].fillna(False).astype(bool).to_numpy(),
        'passed_filters': df['passed_filters'].fillna(False).astype(bool).to_numpy(),
    })
    return out


def prepare_spots(paths):
    print(f"Reading {paths['spots_src']} ...")
    spots = load_auto_spots(paths['spots_src'])
    paths['ng_dir'].mkdir(parents=True, exist_ok=True)
    spots.to_parquet(paths['auto_spots_table'], index=False)

    writer = neuroglancer.write_annotations.AnnotationWriter(
        coordinate_space=annotation_coordinate_space(paths),
        annotation_type='point',
        relationships=['nucleus'],
        properties=AUTO_SPOT_PROPERTIES,
    )
    pts = voxel_to_ng(spots[['t', 'z', 'y', 'x']].to_numpy())
    for row, p in zip(spots.itertuples(index=False), pts):
        writer.add_point(
            p, id=int(row.auto_id),
            prob=float(row.prob), gauss3d_dog=float(row.gauss3d_dog),
            spot_id=max(0, int(row.spot_id)), fill_in=int(row.fill_in_spot),
            passed_filters=int(row.passed_filters),
            nucleus=[int(row.nucleus_id)] if row.nucleus_id >= 0 else [],
        )
    print(f"Writing {len(spots)} annotations -> {paths['auto_spots']} ...")
    writer.write(paths['auto_spots'])
    set_annotation_bounds_to_image(paths['auto_spots'], image_shape_tzyx(paths['raw_image']))
    print("Done.")


def image_shape_tzyx(image_path):
    """Level-0 shape of the image without its channel axis."""
    arr = open_source_v2(Path(image_path) / '0')
    channel = spatiotemporal_axes(image_path)['channel_axis']
    return [int(s) for i, s in enumerate(arr.shape) if i != channel]


def set_annotation_bounds_to_image(ann_dir, shape):
    """
    Make the annotation bounds the image extent, [-0.5, n - 0.5) per dim.

    Neuroglancer shifts OME-Zarr data by half a voxel so voxel centres sit at
    integers, and only keeps that convention for the viewer if *every* layer has
    half-integer bounds. AnnotationWriter uses the tight (integer) point bounds,
    which would make the viewer snap t/z to voxel edges instead of centres.
    """
    info_path = Path(ann_dir) / 'info'
    info = json.loads(info_path.read_text())
    lower = [-0.5] * len(shape)
    upper = [n - 0.5 for n in shape]
    info['lower_bound'] = lower
    info['upper_bound'] = upper
    for level in info['spatial']:
        level['chunk_size'] = [u - l for l, u in zip(lower, upper)]
    info_path.write_text(json.dumps(info))


# ============================================================================
# Segment properties
# ============================================================================

SEGMENT_TAGS = ['has_parent', 'divides', 'has_auto_spots']


def build_segment_properties(tracks, spots):
    """Precomputed `neuroglancer_segment_properties` info dict for every track_id."""
    g = tracks.groupby('track_id')
    per = pd.DataFrame({
        't_start': g['t'].min(),
        't_end': g['t'].max(),
        'length': g.size(),
        'parent_track_id': g['parent_track_id'].first(),
    })
    sp = spots[spots['nucleus_id'] >= 0].groupby('nucleus_id')
    per['n_auto_spots'] = sp.size().reindex(per.index, fill_value=0)
    per['n_auto_passed'] = sp['passed_filters'].sum().reindex(per.index, fill_value=0)
    parents = set(tracks.loc[tracks['parent_track_id'] > 0, 'parent_track_id'].unique().tolist())

    tags = []
    for tid, row in zip(per.index.tolist(), per.itertuples()):
        t = []
        if row.parent_track_id > 0:
            t.append(0)
        if tid in parents:
            t.append(1)
        if row.n_auto_spots > 0:
            t.append(2)
        tags.append(t)

    def number(col, dtype, desc):
        return {'id': col, 'type': 'number', 'data_type': dtype, 'description': desc,
                'values': per[col].astype('int64').tolist()}

    return {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [str(i) for i in per.index.tolist()],
            'properties': [
                number('t_start', 'uint32', 'first frame of the track'),
                number('t_end', 'uint32', 'last frame of the track'),
                number('length', 'uint32', 'number of frames'),
                number('n_auto_spots', 'uint32', 'zms2 auto spots in this nucleus'),
                number('n_auto_passed', 'uint32', 'zms2 auto spots that passed filters'),
                number('parent_track_id', 'int32', 'parent track (-1 if none)'),
                {'id': 'tags', 'type': 'tags', 'tags': SEGMENT_TAGS,
                 'values': tags},
            ],
        },
    }


def prepare_segment_properties(paths):
    tracks = pd.read_parquet(paths['tracks_index'])
    spots = pd.read_parquet(paths['auto_spots_table'])
    info = build_segment_properties(tracks, spots)
    out = paths['segment_properties']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'info').write_text(json.dumps(info, separators=(',', ':')))
    print(f"Wrote properties for {len(info['inline']['ids'])} segments -> {out}")


# ============================================================================
# Sharded OME-Zarr 0.5 pyramids (labels and images)
# ============================================================================

def level_shape(shape_zyx, level):
    """Shape after `level` rounds of 2x stride/mean downsampling (ceil)."""
    s = np.asarray(shape_zyx, dtype=np.int64)
    for _ in range(level):
        s = (s + 1) // 2
    return tuple(int(v) for v in s)


def shard_shape(shape_zyx, chunk_zyx):
    """One shard per timepoint/channel: the full ZYX extent padded to chunk multiples."""
    return tuple(int(-(-s // c) * c) for s, c in zip(shape_zyx, chunk_zyx))


def zarr3_spec(path, shape, dtype, chunk, shard, dimension_names):
    itemsize = np.dtype(dtype).itemsize
    return {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': str(path)},
        'metadata': {
            'shape': list(shape),
            'data_type': np.dtype(dtype).name,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': list(shard)}},
            'chunk_key_encoding': {'name': 'default'},
            'fill_value': 0,
            'codecs': [{
                'name': 'sharding_indexed',
                'configuration': {
                    'chunk_shape': list(chunk),
                    'codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'blosc', 'configuration': {
                            'cname': 'zstd', 'clevel': 3, 'shuffle': 'shuffle',
                            'typesize': itemsize, 'blocksize': 0}},
                    ],
                    'index_codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'crc32c'},
                    ],
                    'index_location': 'end',
                },
            }],
            'dimension_names': list(dimension_names),
        },
    }


def init_pyramid(out_path, shape, dtype, leading_dims, chunk_zyx, ome_axes, level0_scale, name):
    """
    Create an empty sharded OME-Zarr 0.5 pyramid.

    Args:
        shape: full level-0 shape, e.g. (T, Z, Y, X) or (T, C, Z, Y, X)
        leading_dims: number of non-spatial leading dims (1 for TZYX, 2 for TCZYX)
        ome_axes: OME axes list (dicts), same length as shape
        level0_scale: OME scale list for level 0
    """
    out_path = Path(out_path)
    names = [a['name'] for a in ome_axes]
    lead = tuple(shape[:leading_dims])
    datasets = []
    for level in range(N_LEVELS):
        zyx = level_shape(shape[leading_dims:], level)
        full = lead + zyx
        chunk = (1,) * leading_dims + tuple(min(c, s) for c, s in zip(chunk_zyx, zyx))
        shard = (1,) * leading_dims + shard_shape(zyx, chunk[leading_dims:])
        ts.open(zarr3_spec(out_path / str(level), full, dtype, chunk, shard, names),
                create=True, delete_existing=True).result()
        f = 2 ** level
        scale = list(level0_scale[:leading_dims]) + [s * f for s in level0_scale[leading_dims:]]
        datasets.append({'path': str(level),
                         'coordinateTransformations': [{'type': 'scale', 'scale': scale}]})
    group = {
        'zarr_format': 3,
        'node_type': 'group',
        'attributes': {'ome': {'version': '0.5', 'multiscales': [{
            'name': name, 'axes': ome_axes, 'datasets': datasets}]}},
    }
    (out_path / 'zarr.json').write_text(json.dumps(group, indent=2))
    print(f"Initialized {out_path} with {N_LEVELS} levels, level-0 shape {shape}")


def downsample_labels(vol):
    """2x stride downsampling (keeps label ids exact)."""
    return vol[::2, ::2, ::2]


def downsample_mean(vol):
    """2x mean downsampling for intensity images (edges padded by replication)."""
    pad = [(0, s % 2) for s in vol.shape]
    v = np.pad(vol, pad, mode='edge') if any(p[1] for p in pad) else vol
    z, y, x = (s // 2 for s in v.shape)
    out = v.reshape(z, 2, y, 2, x, 2).mean(axis=(1, 3, 5), dtype=np.float32)
    return out.astype(vol.dtype)


def write_pyramid_timepoint(out_path, index, vol, downsample):
    """Write one (Z,Y,X) volume at leading `index` (e.g. (t,) or (t,c)) to all levels."""
    for level in range(N_LEVELS):
        if level > 0:
            vol = downsample(vol)
        arr = ts.open({'driver': 'zarr3',
                       'kvstore': {'driver': 'file', 'path': str(Path(out_path) / str(level))}},
                      open=True).result()
        arr[index].write(vol).result()


def open_source_v2(path):
    return ts.open({'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': str(path)}},
                   open=True).result()


def prepare_segments(paths, init, t_start, t_stop):
    src = open_source_v2(paths['segments_src'])
    out = paths['segments']
    if init:
        img_axes = read_ome_multiscales(paths['raw_image'])['axes']
        img_scale = spatiotemporal_axes(paths['raw_image'])['scales']
        axes = [a for a in img_axes if a.get('type') != 'channel']
        assert len(axes) == src.rank == 4, f"expected TZYX segments, got rank {src.rank}"
        init_pyramid(out, src.shape, np.uint32, 1, SEG_CHUNK_ZYX, axes, img_scale,
                     name='segments')
        return
    t_stop = min(t_stop if t_stop is not None else src.shape[0], src.shape[0])
    for t in range(t_start, t_stop):
        tic = time.time()
        vol = src[t].read().result()
        if vol.dtype != np.uint32:
            if vol.min() < 0:
                raise ValueError(f"negative label at t={t}")
            vol = vol.astype(np.uint32, copy=False) if vol.dtype.itemsize != 4 else vol.view(np.uint32)
        write_pyramid_timepoint(out, (t,), vol, downsample_labels)
        print(f"segments t={t} done in {time.time() - tic:.1f}s", flush=True)


def prepare_images(paths, init, t_start, t_stop):
    src_path = paths['raw_image']
    src = open_source_v2(Path(src_path) / '0')
    ms = read_ome_multiscales(src_path)
    out = paths['ng_dir'] / 'image.ome.zarr'
    if init:
        scale = [tr for tr in ms['datasets'][0]['coordinateTransformations']
                 if tr['type'] == 'scale'][0]['scale']
        assert src.rank == 5, f"expected TCZYX image, got rank {src.rank}"
        init_pyramid(out, src.shape, src.dtype.numpy_dtype, 2, IMG_CHUNK_ZYX,
                     ms['axes'], scale, name='image')
        # Keep channel names/colours for neuroglancer's omero parsing
        group = json.loads((out / 'zarr.json').read_text())
        zattrs = json.loads((Path(src_path) / '.zattrs').read_text())
        if 'omero' in zattrs:
            group['attributes']['ome']['omero'] = zattrs['omero']
            (out / 'zarr.json').write_text(json.dumps(group, indent=2))
        print(f"Set neuroglancer.image: {out} in the config to use it")
        return
    t_stop = min(t_stop if t_stop is not None else src.shape[0], src.shape[0])
    for t in range(t_start, t_stop):
        tic = time.time()
        for c in range(src.shape[1]):
            vol = src[t, c].read().result()
            write_pyramid_timepoint(out, (t, c), vol, downsample_mean)
        print(f"image t={t} done in {time.time() - tic:.1f}s", flush=True)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['tracks', 'spots', 'segprops', 'segments', 'images'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--init', action='store_true',
                        help='segments/images: create the empty pyramid (run once, first)')
    parser.add_argument('--t-start', type=int, default=0)
    parser.add_argument('--t-stop', type=int, default=None, help='exclusive')
    args = parser.parse_args()

    paths = ng_paths(load_config(args.config))
    if args.step == 'tracks':
        prepare_tracks(paths)
    elif args.step == 'spots':
        prepare_spots(paths)
    elif args.step == 'segprops':
        prepare_segment_properties(paths)
    elif args.step == 'segments':
        prepare_segments(paths, args.init, args.t_start, args.t_stop)
    elif args.step == 'images':
        prepare_images(paths, args.init, args.t_start, args.t_stop)


if __name__ == '__main__':
    main()
