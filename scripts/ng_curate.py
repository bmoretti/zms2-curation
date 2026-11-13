"""
Neuroglancer viewer for curating MS2 spots, one nucleus (track) at a time.

    python ng_curate.py --config ../configs/2025-04-17_wt.yaml [--port 9000] [--data-port 9001]

Run ng_prepare.py first (tracks, spots, segprops, segments).

Neuroglancer's own controls do most of the work:
  [ / ]                 previous / next frame (t)
  , / .                 previous / next z slice
  double-click nucleus  show/hide it; the most recently shown nucleus is "active"
  Seg tab               search/sort nuclei by id, length, n_auto_spots, #tags
  ctrl+click            place a spot (the `curated` layer's point tool is always active)
  ctrl+right-click      select a spot; ctrl+alt+right-click deletes it; alt+drag moves it
  alt+= / alt+-         narrow / widen the z range in which spots are drawn in 2D

Added here:
  c        toggle follow-nucleus (recentre on the active nucleus when t changes)
  y / u    accept / reject the auto spot nearest the mouse at the current t
  space    jump to the next frame with an undecided auto spot (then: without any curated spot)
  m        merge the nucleus under the mouse into the active one (tracking split)
  ctrl+s   save now (edits also autosave after 2 s)

Curated spots are stored in <experiment>/ng/curated_spots.parquet; export them to
manual_points_df_*.csv with `python ng_utils.py export --config ...`.
"""

import argparse
import copy
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

import neuroglancer

import ng_server
from ng_server import ChunkFileHandler, FixedPortFileServer
from ng_utils import (
    CURATED_PROPERTY_IDS,
    SOURCES,
    STATUSES,
    TrackIndex,
    empty_curated_table,
    frame_to_ng,
    annotation_to_row,
    load_config,
    nearest_index,
    next_target_frame,
    ng_paths,
    ng_to_frame,
    ng_to_voxel,
    ng_units_and_scales,
    normalize_curated_table,
    read_curated_table,
    row_to_annotation,
    spatiotemporal_axes,
    voxel_to_ng,
    write_curated_table,
)


SAVE_DELAY_S = 2.0
BACKUP_EVERY_S = 600
BACKUPS_KEPT = 50
# Autosave refuses to drop more than this fraction of rows at once (ctrl+s overrides)
MAX_AUTOSAVE_LOSS = 0.2

# Colours are hex: CSS names are not full intensity ("green" is #008000, half brightness).
# gamma < 1 lifts dim signal (e.g. MS2 spots) without saturating bright regions as much.
IMAGE_SHADER = """
#uicontrol invlerp normalized(range=[{lo}, {hi}])
#uicontrol vec3 color color(default="{color}")
#uicontrol float gamma slider(min=0.2, max=2.0, step=0.05, default=1.0)
void main() {{
  emitRGB(color * pow(normalized(), gamma));
}}
"""

AUTO_SPOT_SHADER = """
#uicontrol float min_prob slider(min=0.0, max=1.0, default=0.0)
#uicontrol bool show_fill_in checkbox(default=true)
#uicontrol bool show_failed_filters checkbox(default=true)
void main() {
  bool fill_in = prop_fill_in() == 1u;
  bool passed = prop_passed_filters() == 1u;
  if (prop_prob() < min_prob || (fill_in && !show_fill_in) || (!passed && !show_failed_filters)) {
    discard;
  }
  setPointMarkerSize(14.0);
  setPointMarkerColor(vec4(0.0, 0.0, 0.0, 0.0));
  setPointMarkerBorderWidth(2.0);
  setPointMarkerBorderColor(passed ? (fill_in ? vec3(0.2, 0.6, 1.0) : vec3(0.0, 1.0, 1.0))
                                   : vec3(0.5, 0.5, 0.5));
}
"""

CURATED_SHADER = """
void main() {
  if (prop_status() == 1u) {
    setPointMarkerSize(8.0);
    setColor(vec3(0.6, 0.6, 0.6));
  } else {
    setPointMarkerSize(8.0);
    setColor(prop_source() == 1u ? vec3(1.0, 1.0, 0.0) : vec3(1.0, 0.5, 0.0));
  }
}
"""

# Navigation/display fields restored from session.json (JSON key -> ViewerState attribute)
SESSION_VIEW_KEYS = {
    'position': 'position',
    'crossSectionScale': 'cross_section_scale',
    'crossSectionDepth': 'cross_section_depth',
    'crossSectionOrientation': 'cross_section_orientation',
    'projectionScale': 'projection_scale',
    'projectionOrientation': 'projection_orientation',
    'layout': 'layout',
}

CURATED_PROPERTIES = [
    neuroglancer.AnnotationPropertySpec(id='status', type='uint8', default=0,
                                        enum_values=[0, 1], enum_labels=STATUSES),
    neuroglancer.AnnotationPropertySpec(id='source', type='uint8', default=0,
                                        enum_values=[0, 1], enum_labels=SOURCES),
    neuroglancer.AnnotationPropertySpec(id='auto_id', type='int32', default=-1,
                                        description='row of auto_spots.parquet (-1: manual)'),
]
assert [p.id for p in CURATED_PROPERTIES] == CURATED_PROPERTY_IDS


def _same_entry(a, b, tol=1e-3):
    """True if two curated rows are equal up to float32 rounding of coordinates."""
    if any(abs(float(a[k]) - float(b[k])) > tol for k in ('z', 'y', 'x')):
        return False
    if int(a['t']) != int(b['t']) or int(a['nucleus_id']) != int(b['nucleus_id']):
        return False
    if a['status'] != b['status'] or a['source'] != b['source']:
        return False
    return (pd.isna(a['auto_id']) and pd.isna(b['auto_id'])) or a['auto_id'] == b['auto_id']


# ============================================================================
# Curation app
# ============================================================================

class Curator:

    def __init__(self, config, args):
        self.paths = ng_paths(config)
        self.lock = threading.RLock()
        self.save_timer = None
        self.last_save = None

        print('Loading tracks and auto spots ...')
        self.tracks = TrackIndex.from_parquet(self.paths['tracks_index'])
        self.auto = pd.read_parquet(self.paths['auto_spots_table'])
        self.table = read_curated_table(self.paths['curated'])
        self.n_saved = len(self.table)
        self.last_backup = 0.0
        self.merges = self._load_merges()

        axes = spatiotemporal_axes(self.paths['raw_image'])
        units, scales = ng_units_and_scales(axes)
        self.dim_names = axes['names']
        self.scales_zyx = np.asarray(scales[1:])
        self.dimensions = neuroglancer.CoordinateSpace(names=self.dim_names, units=units,
                                                       scales=scales)
        self.active = None
        self.follow = True
        self.last_frame = None
        self.visible = []
        self.known_ids = set(self.table['annotation_id'])
        self.dirty = False

        # Data server: one root that contains the image and the prepared ng data
        root = Path(os.path.commonpath([self.paths['image'].parent, self.paths['ng_dir']]))
        self.data_root = root
        self.file_server = FixedPortFileServer(root, args.host, args.data_port)
        self.data_url = f'http://{args.public_host}:{self.file_server.port}/{self.file_server.token}'

        neuroglancer.set_server_bind_address(args.host, args.port)
        self.viewer = neuroglancer.Viewer()
        self.public_host = args.public_host
        self.contrast = (config.get('neuroglancer') or {}).get('contrast', {})

        self._setup_state()
        self._setup_actions()
        self._restore_session()
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed))
        self.update_status()

    # ------------------------------------------------------------------ setup

    def url(self, path):
        rel = Path(path).relative_to(self.data_root)
        return f'{self.data_url}/{rel.as_posix()}'

    def _setup_state(self):
        h2b_range = self.contrast.get('H2B', [200, 8000])
        mcp_range = self.contrast.get('MCP', [1500, 6000])
        with self.viewer.txn() as s:
            s.dimensions = self.dimensions
            # Show X/Y/Z in the panels; T gets a slider (and [ / ] keys)
            s.display_dimensions = [n for n in self.dim_names[::-1] if n != self.dim_names[0]]
            s.layers['H2B'] = neuroglancer.ImageLayer(
                source=f"zarr://{self.url(self.paths['image'])}",
                local_position=[0],
                shader=IMAGE_SHADER.format(lo=h2b_range[0], hi=h2b_range[1], color='#ff00ff'),
                blend='additive', opacity=1.0,
                # Use a coarser level once voxels are < 2 screen px: zoomed-out time
                # steps load ~4x less data (curation zoom still uses full resolution)
                cross_section_render_scale=2)
            s.layers['MCP'] = neuroglancer.ImageLayer(
                source=f"zarr://{self.url(self.paths['image'])}",
                local_position=[1],
                shader=IMAGE_SHADER.format(lo=mcp_range[0], hi=mcp_range[1], color='#00ff00'),
                blend='additive', opacity=1.0,
                # Use a coarser level once voxels are < 2 screen px: zoomed-out time
                # steps load ~4x less data (curation zoom still uses full resolution)
                cross_section_render_scale=2)
            s.layers['nuclei'] = neuroglancer.SegmentationLayer(
                source=[f"zarr://{self.url(self.paths['segments'])}",
                        f"precomputed://{self.url(self.paths['segment_properties'])}"],
                selected_alpha=0.15, not_selected_alpha=0.0, object_alpha=1.0,
                hover_highlight=True)
            s.layers['auto_spots'] = neuroglancer.AnnotationLayer(
                source=f"precomputed://{self.url(self.paths['auto_spots'])}",
                linked_segmentation_layer={'nucleus': 'nuclei'},
                filter_by_segmentation=['nucleus'],
                shader=AUTO_SPOT_SHADER)
            curated = neuroglancer.LocalAnnotationLayer(
                dimensions=self.dimensions,
                annotation_properties=CURATED_PROPERTIES,
                annotation_relationships=['nucleus'],
                linked_segmentation_layer={'nucleus': 'nuclei'},
                filter_by_segmentation=['nucleus'],
                shader=CURATED_SHADER).to_json()
            # Point tool active from the start (and it stays active after Esc), so
            # ctrl+click places a spot. The client only accepts the string form here.
            curated['tool'] = 'annotatePoint'
            s.layers['curated'] = neuroglancer.viewer_state.make_layer(curated)
            s.layers['curated'].annotations = [
                neuroglancer.PointAnnotation(**row_to_annotation(r))
                for _, r in self.table.iterrows()]
            s.selected_layer.layer = 'curated'
            s.selected_layer.visible = True
            s.layout = '4panel'
            s.cross_section_scale = 0.5
            # Zoom-relative depth of field for annotations in 2D panels: at this zoom,
            # spots within about +-1 z slice are drawn (adjust with alt+= / alt+-)
            s.cross_section_depth = 10
            self._color_curated_nuclei(s)

    def _setup_actions(self):
        actions = {
            'zms2-toggle-follow': self.toggle_follow,
            'zms2-accept': lambda st: self.review(st, 'spot'),
            'zms2-reject': lambda st: self.review(st, 'rejected'),
            'zms2-next': self.jump_next,
            'zms2-merge': self.merge,
            'zms2-save': lambda st: self.save(force=True),
        }
        for name, fn in actions.items():
            self.viewer.actions.add(name, fn)
        with self.viewer.config_state.txn() as cs:
            b = cs.input_event_bindings
            b.viewer['keyc'] = 'zms2-toggle-follow'
            b.viewer['keyy'] = 'zms2-accept'
            b.viewer['keyu'] = 'zms2-reject'
            b.viewer['space'] = 'zms2-next'
            b.viewer['keym'] = 'zms2-merge'
            b.viewer['control+keys'] = 'zms2-save'

    # ------------------------------------------------------------- persistence

    def _load_merges(self):
        p = self.paths['ng_dir'] / 'merges.json'
        if p.exists():
            return {int(k): [int(v) for v in vs] for k, vs in json.loads(p.read_text()).items()}
        return {}

    def _save_merges(self):
        p = self.paths['ng_dir'] / 'merges.json'
        p.write_text(json.dumps({str(k): v for k, v in self.merges.items()}, indent=1))

    def group_ids(self, nid):
        return [nid] + self.merges.get(nid, [])

    def _restore_session(self):
        p = self.paths['session']
        if not p.exists():
            return
        sess = json.loads(p.read_text())
        st = sess.get('viewer_state', {})
        with self.viewer.txn() as s:
            for key, attr in SESSION_VIEW_KEYS.items():
                if key in st:
                    setattr(s, attr, st[key])
            for layer in st.get('layers', []):
                name = layer.get('name')
                if name in s.layers and 'shaderControls' in layer:
                    s.layers[name].shader_controls = layer['shaderControls']
                if name == 'nuclei' and 'segments' in layer:
                    # '!id' marks a starred-but-hidden segment
                    s.layers['nuclei'].segments = [int(x) for x in layer['segments']
                                                   if not str(x).startswith('!')]
        self.active = sess.get('active')
        self.follow = sess.get('follow', True)
        self.visible = [int(x) for x in self.viewer.state.layers['nuclei'].segments]
        print(f'Restored session (active nucleus {self.active})')

    def _write_session(self):
        # to_json() returns the live state's dict, so copy before stripping annotations
        st = copy.deepcopy(self.viewer.state.to_json())
        for layer in st.get('layers', []):
            layer.pop('annotations', None)
        tmp = self.paths['session'].with_suffix('.tmp')
        tmp.write_text(json.dumps({'viewer_state': st, 'active': self.active,
                                   'follow': self.follow}))
        os.replace(tmp, self.paths['session'])

    def schedule_save(self):
        with self.lock:
            if self.save_timer is not None:
                self.save_timer.cancel()
            self.save_timer = threading.Timer(SAVE_DELAY_S, self.save)
            self.save_timer.daemon = True
            self.save_timer.start()

    def save(self, force=False):
        with self.lock:
            if self.dirty or force:
                n_new = len(self.table)
                if not force and n_new < (1 - MAX_AUTOSAVE_LOSS) * self.n_saved and self.n_saved >= 10:
                    msg = (f'NOT saved: curated spots would drop {self.n_saved} -> {n_new}. '
                           'Press ctrl+s in the viewer to accept; otherwise the file is kept.')
                    print(msg)
                    self.status(msg)
                    return
                self._backup()
                write_curated_table(self.table, self.paths['curated'])
                self.n_saved = n_new
                self.dirty = False
            self._write_session()
            self.last_save = time.strftime('%H:%M:%S')
        self.update_status()

    def _backup(self):
        """Copy the previous table to ng/backups/ (at most every BACKUP_EVERY_S)."""
        src = self.paths['curated']
        if not src.exists() or time.time() - self.last_backup < BACKUP_EVERY_S:
            return
        bdir = self.paths['ng_dir'] / 'backups'
        bdir.mkdir(exist_ok=True)
        shutil.copy2(src, bdir / f"curated_spots_{time.strftime('%Y%m%d-%H%M%S')}.parquet")
        for old in sorted(bdir.glob('curated_spots_*.parquet'))[:-BACKUPS_KEPT]:
            old.unlink()
        self.last_backup = time.time()

    # ---------------------------------------------------------- state changes

    def curated_from_state(self, state):
        rows = []
        for a in state.layers['curated'].annotations:
            segs = [[int(x) for x in g] for g in (a.segments or [])]
            rows.append(annotation_to_row(a.id, np.asarray(a.point), segs, list(a.props)))
        return rows

    def on_state_changed(self):
        self._track_active_nucleus(self.viewer.state)
        # Re-read: activating a nucleus may have moved t into its lifetime
        self._follow(self.viewer.state)
        self._sync_curated(self.viewer.state)

    def _track_active_nucleus(self, state):
        visible = [int(x) for x in state.layers['nuclei'].segments]
        # Segments merged into the active nucleus are part of it, not a new selection
        group = set(self.group_ids(self.active)) if self.active is not None else set()
        added = [v for v in visible if v not in self.visible and v not in group]
        self.visible = visible
        if added:
            self.activate(added[-1])
        elif self.active is not None and self.active not in visible:
            self.active = visible[-1] if visible else None
            self.update_status()

    def activate(self, nid):
        self.active = nid
        self.last_frame = None
        if nid not in self.tracks:
            self.status(f'nucleus {nid} is not in tracks_index')
        else:
            # Jump into the track's lifetime if the current frame is outside it
            t0, t1 = self.tracks.group_span(self.group_ids(nid))
            pos = self.viewer.state.position
            if pos is not None and not t0 <= ng_to_frame(pos[0]) <= t1:
                with self.viewer.txn() as s:
                    s.position = [frame_to_ng(t0), *s.position[1:]]
        self.update_status()

    def _follow(self, state):
        if state.position is None or len(state.position) < 4:
            return
        frame = ng_to_frame(state.position[0])
        if frame == self.last_frame:
            return
        self.last_frame = frame
        if not self.follow or self.active is None or self.active not in self.tracks:
            self.update_status()
            return
        zyx = voxel_to_ng(self.tracks.centroid(self.group_ids(self.active), frame))
        with self.viewer.txn() as s:
            s.position = [state.position[0], *zyx]
        self.update_status()

    def _sync_curated(self, state):
        """Mirror the `curated` layer into self.table and schedule a save if it changed."""
        rows = self.curated_from_state(state)
        relink = []
        for r in rows:
            if r['annotation_id'] in self.known_ids:
                continue
            # New annotation placed in the viewer: link it to the active nucleus
            if self.active is not None and r['nucleus_id'] != self.active:
                r['nucleus_id'] = self.active
                relink.append(r['annotation_id'])
        old = {r['annotation_id']: r for r in self.table.to_dict('records')}
        merged, changed = [], set(old) != {r['annotation_id'] for r in rows}
        for r in rows:
            prev = old.get(r['annotation_id'])
            if prev is not None and _same_entry(prev, r):
                merged.append(prev)  # keep full-precision coords and timestamp
            else:
                r['updated'] = pd.Timestamp.now(tz='UTC')
                merged.append(r)
                changed = True
        self.known_ids = {r['annotation_id'] for r in rows}
        if relink:
            with self.viewer.txn() as s:
                for a in s.layers['curated'].annotations:
                    if a.id in relink:
                        a.segments = [[self.active]]
        if changed:
            with self.lock:
                self.table = normalize_curated_table(pd.DataFrame(merged)) if merged \
                    else empty_curated_table()
                self.dirty = True
            with self.viewer.txn() as s:
                self._color_curated_nuclei(s)
            self.schedule_save()
            self.update_status()

    def _color_curated_nuclei(self, s):
        """Give nuclei with at least one curated entry a fixed colour in the Seg list."""
        curated = set(self.table['nucleus_id'].unique().tolist()) - {-1}
        s.layers['nuclei'].segment_colors = {int(n): '#ffaa00' for n in curated}

    # ----------------------------------------------------------------- actions

    def status(self, msg):
        with self.viewer.config_state.txn() as cs:
            cs.status_messages['zms2-msg'] = msg

    def update_status(self):
        parts = []
        if self.active is None:
            parts.append('no active nucleus (double-click one, or search in the Seg tab)')
        else:
            ids = self.group_ids(self.active)
            label = '+'.join(str(i) for i in ids)
            parts.append(f'nucleus {label}')
            if self.active in self.tracks:
                t0, t1 = self.tracks.group_span(ids)
                parts.append(f't {t0}-{t1}')
            mine = self.table[self.table['nucleus_id'] == self.active]
            n_spot = int((mine['status'] == 'spot').sum())
            n_auto = int(self.auto['nucleus_id'].isin(ids).sum())
            n_dec = int(mine['auto_id'].notna().sum())
            parts.append(f'{n_spot} spots, auto reviewed {n_dec}/{n_auto}')
        parts.append(f"follow {'on' if self.follow else 'off'} [c]")
        if self.last_frame is not None:
            parts.append(f'frame {self.last_frame}')
        parts.append(f'saved {self.last_save}' if self.last_save else 'not saved yet')
        with self.viewer.config_state.txn() as cs:
            cs.status_messages['zms2'] = ' | '.join(parts)

    def toggle_follow(self, action_state):
        self.follow = not self.follow
        self.last_frame = None
        self.on_state_changed()
        self.update_status()

    def _mouse_voxel(self, action_state):
        pos = action_state.mouse_voxel_coordinates
        if pos is None:
            return None
        return np.asarray(pos, dtype=np.float64)

    def review(self, action_state, decision):
        if self.active is None:
            self.status('no active nucleus')
            return
        pos = self._mouse_voxel(action_state)
        state = action_state.viewer_state
        frame = ng_to_frame((pos if pos is not None else state.position)[0])
        cand = self.auto[(self.auto['t'] == frame) &
                         self.auto['nucleus_id'].isin(self.group_ids(self.active))]
        if cand.empty:
            self.status(f'no auto spot for nucleus {self.active} at t={frame}')
            return
        ref = ng_to_voxel(pos[1:]) if pos is not None else ng_to_voxel(state.position[1:])
        i = nearest_index(cand[['z', 'y', 'x']].to_numpy(), ref, self.scales_zyx)
        spot = cand.iloc[i]
        ann_id = f"auto{int(spot['auto_id'])}"
        fields = row_to_annotation({
            'annotation_id': ann_id, 't': int(spot['t']), 'z': spot['z'], 'y': spot['y'],
            'x': spot['x'], 'nucleus_id': self.active, 'status': decision,
            'source': 'auto', 'auto_id': int(spot['auto_id'])})
        with self.viewer.txn() as s:
            anns = s.layers['curated'].annotations
            existing = [a for a in anns if a.id == ann_id]
            if existing:
                existing[0].props = fields['props']
            else:
                anns.append(neuroglancer.PointAnnotation(**fields))
            self.known_ids.add(ann_id)
        self.status(f"{decision} auto spot {int(spot['auto_id'])} at t={frame} "
                    f"(prob {spot['prob']:.2f})")
        if decision == 'rejected' or len(cand) == 1:
            self.jump_next(action_state)

    def jump_next(self, action_state):
        if self.active is None or self.active not in self.tracks:
            self.status('no active nucleus')
            return
        ids = self.group_ids(self.active)
        state = self.viewer.state
        frame = ng_to_frame(state.position[0])
        mine = self.table[self.table['nucleus_id'] == self.active]
        decided = set(mine['auto_id'].dropna().astype(int).tolist())
        auto = self.auto[self.auto['nucleus_id'].isin(ids)]
        undecided = auto.loc[~auto['auto_id'].isin(decided), 't'].tolist()
        target, reason = next_target_frame(frame, self.tracks.group_span(ids), undecided,
                                           mine['t'].tolist())
        if target is None:
            self.status(reason)
            return
        with self.viewer.txn() as s:
            s.position = [frame_to_ng(target), *s.position[1:]]
        self.status(f'-> t={target} ({reason})')

    def merge(self, action_state):
        if self.active is None:
            self.status('no active nucleus to merge into')
            return
        try:
            other = action_state.selected_values['nuclei'].value
        except (KeyError, AttributeError):
            other = None
        if hasattr(other, 'key'):
            other = other.key
        if not other or int(other) == self.active:
            self.status('hover over another nucleus to merge it into the active one')
            return
        other = int(other)
        group = self.merges.setdefault(self.active, [])
        if other not in group:
            group.append(other)
        self._save_merges()
        self.visible = sorted(set(self.visible) | {other})
        with self.viewer.txn() as s:
            for a in s.layers['curated'].annotations:
                if a.segments and len(a.segments[0]) and int(a.segments[0][0]) == other:
                    a.segments = [[self.active]]
            segs = [int(x) for x in s.layers['nuclei'].segments]
            if other not in segs:
                s.layers['nuclei'].segments = segs + [other]
        self.last_frame = None
        self.status(f'merged {other} into {self.active}')
        self.update_status()

    # ------------------------------------------------------------------- run

    def print_instructions(self):
        url = self.viewer.get_viewer_url()
        host = socket.getfqdn()
        url = url.replace(url.split('/')[2].split(':')[0], self.public_host, 1)
        port = url.split('/')[2].split(':')[1]
        print('\n' + '=' * 70)
        print(f'Viewer: {url}')
        print(f'From a laptop, first open a tunnel (replace <login> with your ssh login host):')
        print(f'  ssh -N -L {port}:{host}:{port} -L {self.file_server.port}:{host}:'
              f'{self.file_server.port} <login>')
        print('On an HPC desktop on this node, open the URL directly.')
        print('Press Ctrl+C here to save and quit.')
        print('=' * 70 + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--host', default='0.0.0.0', help='bind address')
    parser.add_argument('--port', type=int, default=9000, help='viewer port')
    parser.add_argument('--data-port', type=int, default=9001, help='data file server port')
    parser.add_argument('--image', help='image OME-Zarr to stream (overrides neuroglancer.image)')
    parser.add_argument('--read-ahead', type=int, default=ng_server.AHEAD, metavar='N',
                        help='timepoints the data server reads ahead into memory (0: off)')
    parser.add_argument('--log-requests', metavar='FILE',
                        help='append one line per data request (start, ms, status, bytes, '
                             'method, range, path) to FILE, for diagnosing slow loading')
    parser.add_argument('--public-host', default='localhost',
                        help='hostname the browser uses to reach this machine')
    args = parser.parse_args()

    ChunkFileHandler.prefetcher.offsets = (
        [d for d in range(1, args.read_ahead + 1)] + ([-1] if args.read_ahead else []))
    if args.log_requests:
        ChunkFileHandler.request_log = open(args.log_requests, 'a')
    config = load_config(args.config)
    if args.image:
        config['neuroglancer'] = dict(config.get('neuroglancer') or {}, image=args.image)
    curator = Curator(config, args)
    curator.print_instructions()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        curator.save()
        print('Bye.')


if __name__ == '__main__':
    main()
