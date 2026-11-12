"""
Shared helpers for the neuroglancer curation tools (ng_prepare.py, ng_curate.py).

Coordinate conventions
----------------------
- "voxel" coordinates: zms2 / napari convention, voxel i is centred at i.
  tracks.csv, spots_filled.pkl and the exported manual_points_df_*.csv use it,
  and so does the curated-spots table.
- "ng" coordinates: the viewer's global coordinates (annotations, position).
  Neuroglancer natively puts voxel i at [i, i+1), but it shifts OME-Zarr sources
  by half a voxel (OME places voxel centres at integers), so for our OME-Zarr
  images and segments, voxel i is centred at i here too. NG_OFFSET keeps this
  in one place.

Usage (CLI):
    python ng_utils.py import --config ../configs/2025-04-17_wt.yaml [--from DIR]
    python ng_utils.py export --config ../configs/2025-04-17_wt.yaml [--to DIR]
"""

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ============================================================================
# Config and paths
# ============================================================================

def load_config(config_path):
    """Load a YAML config; relative paths are resolved from the scripts/ dir."""
    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).parent / config_path
    with open(config_path) as f:
        return yaml.safe_load(f)


def ng_paths(config):
    """
    Resolve all paths used by the neuroglancer tools.

    The optional `neuroglancer:` config section can override:
      ng_dir   (default: <experiment>/ng, next to raw_data)
      image    (default: data.raw_data; e.g. a rechunked copy for dapt)
    """
    ng_cfg = config.get('neuroglancer') or {}
    raw = Path(config['data']['raw_data'])
    ng_dir = Path(ng_cfg.get('ng_dir') or raw.parent / 'ng')
    return {
        'image': Path(ng_cfg.get('image') or raw),
        'raw_image': raw,
        'segments_src': Path(config['data']['segments']),
        'tracks_src': Path(config['data']['tracks']),
        'spots_src': Path(config['data']['spots']) if config['data'].get('spots') else None,
        'ng_dir': ng_dir,
        'segments': ng_dir / 'segments.ome.zarr',
        'segment_properties': ng_dir / 'segment_properties',
        'auto_spots': ng_dir / 'auto_spots',
        'tracks_index': ng_dir / 'tracks_index.parquet',
        'auto_spots_table': ng_dir / 'auto_spots.parquet',
        'curated': ng_dir / 'curated_spots.parquet',
        'session': ng_dir / 'session.json',
        'export_dir': Path(config['output']['directory']),
    }


# ============================================================================
# OME-Zarr metadata
# ============================================================================

def read_ome_multiscales(zarr_path):
    """Return the first `multiscales` entry of an OME-Zarr group (v0.4 or v0.5)."""
    zarr_path = Path(zarr_path)
    if (zarr_path / 'zarr.json').exists():
        attrs = json.loads((zarr_path / 'zarr.json').read_text()).get('attributes', {})
        attrs = attrs.get('ome', attrs)
    else:
        attrs = json.loads((zarr_path / '.zattrs').read_text())
    return attrs['multiscales'][0]


def spatiotemporal_axes(zarr_path):
    """
    Names, units and level-0 scales of the non-channel axes of an OME-Zarr image.

    Returns:
        dict with 'names' (e.g. ['T','Z','Y','X']), 'units', 'scales' (level 0),
        and 'channel_axis' (index in the full axis list, or None).
    """
    ms = read_ome_multiscales(zarr_path)
    axes = ms['axes']
    scale = None
    for tr in ms['datasets'][0]['coordinateTransformations']:
        if tr['type'] == 'scale':
            scale = tr['scale']
    if scale is None:
        scale = [1.0] * len(axes)
    names, units, scales, channel_axis = [], [], [], None
    for i, (ax, s) in enumerate(zip(axes, scale)):
        if ax.get('type') == 'channel':
            channel_axis = i
            continue
        names.append(ax['name'])
        units.append(ax.get('unit', ''))
        scales.append(float(s))
    return {'names': names, 'units': units, 'scales': scales, 'channel_axis': channel_axis}


# Neuroglancer unit strings for OME unit names
OME_TO_NG_UNIT = {
    'second': 's', 'millisecond': 'ms', 'minute': 's', 'hour': 's',
    'micrometer': 'um', 'nanometer': 'nm', 'millimeter': 'mm', 'meter': 'm', '': '',
}
OME_TIME_FACTOR = {'minute': 60.0, 'hour': 3600.0}


def ng_units_and_scales(axes):
    """Convert OME units/scales to neuroglancer units/scales (minutes/hours -> s)."""
    units, scales = [], []
    for u, s in zip(axes['units'], axes['scales']):
        units.append(OME_TO_NG_UNIT.get(u, u))
        scales.append(s * OME_TIME_FACTOR.get(u, 1.0))
    return units, scales


# ============================================================================
# Coordinate conversion
# ============================================================================

# Offset between voxel-centre coordinates and viewer coordinates (0 for OME-Zarr sources)
NG_OFFSET = 0.0


def voxel_to_ng(coords):
    """zms2/napari voxel coordinates -> neuroglancer global coordinates."""
    return np.asarray(coords, dtype=np.float64) + NG_OFFSET


def ng_to_voxel(coords):
    """Neuroglancer global coordinates -> zms2/napari voxel coordinates."""
    return np.asarray(coords, dtype=np.float64) - NG_OFFSET


def ng_to_frame(t_ng):
    """Neuroglancer t position -> frame index (frame i is centred at i + NG_OFFSET)."""
    return int(np.floor(t_ng - NG_OFFSET + 0.5))


def frame_to_ng(frame):
    """Frame index -> neuroglancer t position at the centre of that frame."""
    return float(frame) + NG_OFFSET


def nearest_index(points_zyx, pos_zyx, scales_zyx=(1.0, 1.0, 1.0)):
    """
    Index of the point closest to pos (physical distance), or None if no points.

    Args:
        points_zyx: (N, 3) array
        pos_zyx: (3,) array, same coordinate convention as points
        scales_zyx: physical voxel size, so anisotropy is respected
    """
    points_zyx = np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3)
    if len(points_zyx) == 0:
        return None
    d = (points_zyx - np.asarray(pos_zyx, dtype=np.float64)) * np.asarray(scales_zyx)
    return int(np.argmin((d ** 2).sum(axis=1)))


# ============================================================================
# Tracks
# ============================================================================

class TrackIndex:
    """Fast per-track centroid lookup from tracks_index.parquet (voxel coords)."""

    def __init__(self, df):
        df = df.sort_values(['track_id', 't'], kind='stable').reset_index(drop=True)
        self.df = df
        ids = df['track_id'].to_numpy()
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        stops = np.r_[starts[1:], len(ids)]
        self._ranges = dict(zip(ids[starts].tolist(), zip(starts.tolist(), stops.tolist())))

    @classmethod
    def from_parquet(cls, path):
        return cls(pd.read_parquet(path))

    def __contains__(self, track_id):
        return int(track_id) in self._ranges

    def track(self, track_id):
        """Rows (t, z, y, x, ...) of one track, sorted by t."""
        start, stop = self._ranges[int(track_id)]
        return self.df.iloc[start:stop]

    def span(self, track_id):
        tr = self.track(track_id)
        return int(tr['t'].iloc[0]), int(tr['t'].iloc[-1])

    def group(self, track_ids):
        """Rows of several tracks (e.g. a nucleus split by tracking), sorted by t."""
        parts = [self.track(i) for i in track_ids if i in self]
        if not parts:
            raise KeyError(f'none of {list(track_ids)} in tracks')
        return pd.concat(parts).sort_values('t', kind='stable')

    def group_span(self, track_ids):
        g = self.group(track_ids)
        return int(g['t'].iloc[0]), int(g['t'].iloc[-1])

    def centroid(self, track_id, t):
        """(z, y, x) at frame t; uses the nearest frame if t is outside/missing.

        `track_id` may also be a list of ids treated as one nucleus.
        """
        if isinstance(track_id, (list, tuple, set)):
            tr = self.group(track_id)
        else:
            tr = self.track(track_id)
        ts = tr['t'].to_numpy()
        i = int(np.argmin(np.abs(ts - t)))
        return tr[['z', 'y', 'x']].to_numpy(dtype=np.float64)[i]


# ============================================================================
# Curated-spot table
# ============================================================================

CURATED_COLUMNS = ['annotation_id', 't', 'z', 'y', 'x', 'nucleus_id',
                   'status', 'source', 'auto_id', 'updated']
STATUSES = ['spot', 'rejected']
SOURCES = ['manual', 'auto']


def empty_curated_table():
    return pd.DataFrame({
        'annotation_id': pd.Series(dtype='string'),
        't': pd.Series(dtype='int64'),
        'z': pd.Series(dtype='float64'),
        'y': pd.Series(dtype='float64'),
        'x': pd.Series(dtype='float64'),
        'nucleus_id': pd.Series(dtype='int64'),
        'status': pd.Series(dtype='string'),
        'source': pd.Series(dtype='string'),
        'auto_id': pd.Series(dtype='Int64'),
        'updated': pd.Series(dtype='datetime64[ns, UTC]'),
    })


def normalize_curated_table(df):
    """Coerce a DataFrame to the curated-table schema (column order and dtypes)."""
    out = empty_curated_table()
    if len(df) == 0:
        return out
    df = df.copy()
    if 'annotation_id' not in df:
        df['annotation_id'] = [new_annotation_id() for _ in range(len(df))]
    if 'status' not in df:
        df['status'] = 'spot'
    if 'source' not in df:
        df['source'] = 'manual'
    if 'auto_id' not in df:
        df['auto_id'] = pd.NA
    if 'updated' not in df:
        df['updated'] = pd.Timestamp.now(tz='UTC')
    df = df[CURATED_COLUMNS]
    return df.astype(out.dtypes.to_dict()).reset_index(drop=True)


def new_annotation_id():
    return uuid.uuid4().hex


def read_curated_table(path):
    path = Path(path)
    if not path.exists():
        return empty_curated_table()
    return normalize_curated_table(pd.read_parquet(path))


def write_curated_table(df, path):
    """Atomically write the curated table (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_curated_table(df)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.chmod(tmp, 0o666 & ~_umask())  # mkstemp creates 0600; use normal permissions
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def import_manual_csvs(csv_dir):
    """Read napari-era manual_points_df_*.csv files into curated-table rows."""
    files = sorted(Path(csv_dir).glob('manual_points_df_*.csv'))
    if not files:
        return empty_curated_table()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['t'] = df['t'].round().astype('int64')
    df['nucleus_id'] = df['nucleus_id'].astype('int64')
    df['status'] = 'spot'
    df['source'] = 'manual'
    return normalize_curated_table(df[['t', 'z', 'y', 'x', 'nucleus_id', 'status', 'source']])


def row_to_annotation(row):
    """
    Curated-table row -> plain fields of a neuroglancer point annotation.

    Returns dict(id, point, segments, props) with props ordered as
    CURATED_PROPERTY_IDS (status, source, auto_id).
    """
    auto_id = row['auto_id']
    return {
        'id': str(row['annotation_id']),
        'point': voxel_to_ng([row['t'], row['z'], row['y'], row['x']]).tolist(),
        'segments': [[int(row['nucleus_id'])]] if row['nucleus_id'] >= 0 else [[]],
        'props': [STATUSES.index(row['status']), SOURCES.index(row['source']),
                  -1 if pd.isna(auto_id) else int(auto_id)],
    }


CURATED_PROPERTY_IDS = ['status', 'source', 'auto_id']


def annotation_to_row(ann_id, point, segments, props):
    """Inverse of row_to_annotation (without the 'updated' column)."""
    t = ng_to_frame(point[0])
    z, y, x = ng_to_voxel(point[1:4])
    seg = [int(s) for s in segments[0]] if segments and len(segments[0]) else []
    status, source, auto_id = (list(props) + [0, 0, -1])[:3]
    return {
        'annotation_id': str(ann_id), 't': t, 'z': float(z), 'y': float(y), 'x': float(x),
        'nucleus_id': seg[0] if seg else -1,
        'status': STATUSES[int(status)], 'source': SOURCES[int(source)],
        'auto_id': pd.NA if int(auto_id) < 0 else int(auto_id),
    }


def next_target_frame(t_now, span, undecided_frames, curated_frames):
    """
    Frame to jump to with the "next" action, searching forward from t_now and wrapping.

    Prefers frames with an undecided auto spot; once none are left, falls back to
    frames of the track with no curated entry at all.

    Returns (frame, reason) or (None, reason).
    """
    t0, t1 = span
    order = [t for t in range(t_now + 1, t1 + 1)] + [t for t in range(t0, t_now + 1)]
    undecided = set(undecided_frames)
    for t in order:
        if t in undecided:
            return t, 'undecided auto spot'
    curated = set(curated_frames)
    for t in order:
        if t not in curated:
            return t, 'frame without curated spot'
    return None, 'every frame of the track is curated'


def export_manual_csvs(df, out_dir):
    """
    Write one manual_points_df_{nucleus_id}.csv per nucleus (status == 'spot'),
    in the format read by process_manual_curations.py.

    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spots = df[df['status'] == 'spot']
    written = []
    for nid, g in spots.groupby('nucleus_id'):
        g = g.sort_values('t')
        out = pd.DataFrame({
            't': g['t'].astype('float64').to_numpy(),
            'z': g['z'].astype('float64').to_numpy(),
            'y': g['y'].astype('float64').to_numpy(),
            'x': g['x'].astype('float64').to_numpy(),
            'nucleus_id': np.full(len(g), int(nid), dtype='int64'),
        })
        path = out_dir / f'manual_points_df_{int(nid)}.csv'
        out.to_csv(path, index=False)
        written.append(path)
    return written


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_imp = sub.add_parser('import', help='import manual_points_df_*.csv into the curated table')
    p_imp.add_argument('--config', required=True)
    p_imp.add_argument('--from', dest='src', help='CSV directory (default: output.directory)')
    p_exp = sub.add_parser('export', help='export the curated table to manual_points_df_*.csv')
    p_exp.add_argument('--config', required=True)
    p_exp.add_argument('--to', dest='dst', help='output directory (default: output.directory)')
    args = parser.parse_args()

    paths = ng_paths(load_config(args.config))
    table = read_curated_table(paths['curated'])

    if args.cmd == 'import':
        src = Path(args.src) if args.src else paths['export_dir']
        new = import_manual_csvs(src)
        # Nuclei already in the table are left alone, so imports are idempotent
        new = new[~new['nucleus_id'].isin(table['nucleus_id'])]
        merged = pd.concat([table, new], ignore_index=True) if len(table) else new
        write_curated_table(merged, paths['curated'])
        print(f"Imported {len(new)} spots ({new['nucleus_id'].nunique()} nuclei) "
              f"from {src} -> {paths['curated']}")
    else:
        dst = Path(args.dst) if args.dst else paths['export_dir']
        written = export_manual_csvs(table, dst)
        print(f"Wrote {len(written)} files to {dst}")


if __name__ == '__main__':
    main()
