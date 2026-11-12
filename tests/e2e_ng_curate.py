"""
End-to-end check of ng_curate.py in a headless Chrome (not collected by pytest).

Needs prepared data for the config (ng_prepare.py) and google-chrome. It works on
a temporary ng dir (symlinks to the prepared data plus a copy of the curated
table), so the real curated_spots.parquet and session.json are never touched.

    python tests/e2e_ng_curate.py --config configs/2025-04-17_wt.yaml --nucleus 187620 \
        [--screenshots DIR]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import neuroglancer  # noqa: E402
import neuroglancer.webdriver  # noqa: E402
import selenium.webdriver.common.action_chains  # noqa: E402,F401
from neuroglancer.viewer_config_state import ActionState  # noqa: E402

import ng_curate  # noqa: E402
from ng_utils import (  # noqa: E402
    load_config, ng_paths, ng_to_frame, read_curated_table, voxel_to_ng)

CHROME_ARGS = ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
PREPARED = ['segments.ome.zarr', 'segment_properties', 'auto_spots', 'tracks_index.parquet',
            'auto_spots.parquet']

failures = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        failures.append(msg)


def make_temp_ng_dir(paths):
    tmp = paths['ng_dir'] / '_e2e_tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for name in PREPARED:
        (tmp / name).symlink_to(paths['ng_dir'] / name)
    if paths['curated'].exists():
        shutil.copy(paths['curated'], tmp / 'curated_spots.parquet')
    return tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--nucleus', type=int, required=True,
                        help='a nucleus with auto spots (ideally already curated)')
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tmp = make_temp_ng_dir(ng_paths(cfg))
    cfg.setdefault('neuroglancer', {})
    cfg['neuroglancer'] = dict(cfg['neuroglancer'] or {}, ng_dir=str(tmp))
    paths = ng_paths(cfg)
    try:
        run(cfg, paths, args)
    finally:
        shutil.rmtree(tmp)
    print(f'\n{len(failures)} failure(s)')
    sys.exit(1 if failures else 0)


def run(cfg, paths, args):
    nid = args.nucleus
    orig = read_curated_table(paths['curated'])
    c = ng_curate.Curator(cfg, argparse.Namespace(
        host='127.0.0.1', port=0, data_port=0, public_host='localhost'))
    v = c.viewer

    def shot(name, wait=8):
        if args.screenshots:
            import PIL.Image
            time.sleep(wait)
            args.screenshots.mkdir(parents=True, exist_ok=True)
            img = v.screenshot(size=[1400, 900]).screenshot.image_pixels
            PIL.Image.fromarray(img).save(args.screenshots / f'{name}.png')

    def wait_saved():
        time.sleep(ng_curate.SAVE_DELAY_S + 1.5)

    with neuroglancer.webdriver.Webdriver(v, headless=True, print_logs=False,
                                          extra_command_line_args=CHROME_ARGS) as wd:
        wd.sync()

        # Showing a nucleus (what a double-click does) makes it active
        with v.txn() as s:
            s.layers['nuclei'].segments = [nid]
        time.sleep(2)
        st = v.state
        t0, t1 = c.tracks.span(nid)
        frame = ng_to_frame(st.position[0])
        check(c.active == nid, f'active nucleus = {c.active}')
        check(t0 <= frame <= t1, f'jumped into track span {t0}-{t1}: frame {frame}')
        check(np.allclose(st.position[1:], voxel_to_ng(c.tracks.centroid(nid, frame)), atol=0.01),
              'view centred on the nucleus')
        shot('01_activated', wait=15)

        # Native ']' (mouse over a panel) steps t; follow recentres
        with v.txn() as s:
            s.position = [float(t0), *s.position[1:]]
        time.sleep(1.5)
        panel = wd.driver.find_elements('css selector', '.neuroglancer-rendered-data-panel')[0]
        wd.action_chain().move_to_element(panel).send_keys(']').perform()
        time.sleep(2)
        st = v.state
        check(ng_to_frame(st.position[0]) == t0 + 1 and st.position[0] == t0 + 1,
              f"']' stepped to the next frame centre: {st.position[0]}")
        check(np.allclose(st.position[1:], voxel_to_ng(c.tracks.centroid(nid, t0 + 1)), atol=0.01),
              'follow after ]')

        # A point placed without a nucleus link gets linked to the active nucleus and saved
        p = voxel_to_ng([t0 + 1, *c.tracks.centroid(nid, t0 + 1)])
        with v.txn() as s:
            s.layers['curated'].annotations.append(neuroglancer.PointAnnotation(
                id='e2e_point', point=p.tolist(), segments=[[]], props=[0, 0, -1]))
        time.sleep(1)
        segs = [a.segments for a in v.state.layers['curated'].annotations if a.id == 'e2e_point'][0]
        check([[int(x) for x in g] for g in segs] == [[nid]], f'auto-linked: {segs}')
        wait_saved()
        tab = read_curated_table(paths['curated'])
        row = tab[tab.annotation_id == 'e2e_point']
        check(len(row) == 1 and row.iloc[0].t == t0 + 1 and row.iloc[0].nucleus_id == nid,
              'new point autosaved with frame and nucleus')
        kept = tab.merge(orig, on='annotation_id', suffixes=('', '_o'))
        check(len(kept) == len(orig) and (kept.y == kept.y_o).all() and (kept.z == kept.z_o).all(),
              'existing rows unchanged at full precision')

        # Review: reject then accept the auto spot nearest the mouse
        auto = c.auto[c.auto.nucleus_id == nid]
        a = auto.iloc[len(auto) // 2]
        mouse = voxel_to_ng([a.t, a.z, a.y + 1.0, a.x - 1.0])
        with v.txn() as s:
            s.position = [float(a.t), *s.position[1:]]
        time.sleep(1)
        st_json = v.state.to_json()
        aid = f'auto{int(a.auto_id)}'
        c.review(ActionState({'viewerState': st_json, 'mousePosition': mouse.tolist()}), 'rejected')
        time.sleep(1)
        ann = [x for x in v.state.layers['curated'].annotations if x.id == aid]
        check(len(ann) == 1 and list(ann[0].props)[:2] == [1, 1], 'reject recorded')
        check(ng_to_frame(v.state.position[0]) != int(a.t), 'jumped on after reject')
        c.review(ActionState({'viewerState': st_json, 'mousePosition': mouse.tolist()}), 'spot')
        time.sleep(1)
        ann = [x for x in v.state.layers['curated'].annotations if x.id == aid]
        check(len(ann) == 1 and list(ann[0].props)[:2] == [0, 1], 'accept replaces the reject')
        shot('02_review', wait=6)

        # Native point tool: active from the start, ctrl+click places, still active after Esc
        from selenium.webdriver.common.keys import Keys
        with v.txn() as s:
            s.layout = 'xy'
        time.sleep(4)

        def panel():  # re-query: the DOM is rebuilt when the selection panel opens
            return wd.driver.find_elements('css selector', '.neuroglancer-rendered-data-panel')[0]

        def ctrl_click(dx):
            n0 = len(v.state.layers['curated'].annotations)
            wd.action_chain().move_to_element_with_offset(panel(), dx, 5).pause(0.5) \
                .key_down(Keys.CONTROL).click().key_up(Keys.CONTROL).perform()
            for _ in range(10):  # poll: the state round-trip can take a moment
                time.sleep(0.5)
                anns = v.state.layers['curated'].annotations
                if len(anns) > n0:
                    break
            return [a for a in anns[n0:]]

        j = v.state.to_json()
        cur = [layer for layer in j['layers'] if layer['name'] == 'curated'][0]
        print('   diag: selectedLayer', j.get('selectedLayer'), 'tool', cur.get('tool'),
              'toolBindings', cur.get('toolBindings'), flush=True)
        placed = ctrl_click(5)
        check(len(placed) == 1 and [[int(x) for x in g] for g in placed[0].segments] == [[nid]],
              f'ctrl+click placed a spot linked to the nucleus ({len(placed)} new)')
        wd.action_chain().move_to_element(panel()).send_keys(Keys.ESCAPE).perform()
        time.sleep(1)
        again = ctrl_click(25)
        check(len(again) == 1, 'point tool still active after Esc')
        ids = {a.id for a in placed + again}
        with v.txn() as s:
            anns = s.layers['curated'].annotations
            for i in reversed([i for i, x in enumerate(anns) if x.id in ids]):
                del anns[i]
            s.layout = '4panel'
        time.sleep(1)

        # Custom key binding reaches Python
        f0 = c.follow
        wd.action_chain().send_keys('c').perform()
        time.sleep(1.5)
        check(c.follow != f0, 'c toggles follow')
        wd.action_chain().send_keys('c').perform()
        time.sleep(1)

        # Deleting in the viewer removes rows on disk
        with v.txn() as s:
            anns = s.layers['curated'].annotations
            for i in reversed([i for i, x in enumerate(anns) if x.id in ('e2e_point', aid)]):
                del anns[i]
        wait_saved()
        check(len(read_curated_table(paths['curated'])) == len(orig), 'deletions saved')
        sess = json.loads(paths['session'].read_text())
        check(sess['active'] == nid and not any(
                  'annotations' in layer for layer in sess['viewer_state']['layers']),
              'session.json has the active nucleus and no annotations')
        check(len(v.state.layers['curated'].annotations) == len(orig),
              'saving the session leaves the live layer intact')

        # Merge: another nucleus under the mouse joins the active one
        other = int(c.tracks.df.loc[c.tracks.df.track_id != nid, 'track_id'].iloc[0])
        c.merge(ActionState({'viewerState': v.state.to_json(),
                             'selectedValues': {'nuclei': {'value': str(other)}}}))
        time.sleep(1)
        check(c.merges.get(nid) == [other] and c.active == nid, f'merged {other} into {nid}')
        check(other in [int(x) for x in v.state.layers['nuclei'].segments], 'merged nucleus shown')
        check(json.loads((paths['ng_dir'] / 'merges.json').read_text()) == {str(nid): [other]},
              'merges.json written')

        # Guard: wiping the layer is not autosaved
        if len(orig) >= 10:
            with v.txn() as s:
                del s.layers['curated'].annotations[:]
            wait_saved()
            check(len(read_curated_table(paths['curated'])) == len(orig),
                  'mass deletion not autosaved')

    # Export of the untouched table reproduces napari-era CSV values
    out = paths['ng_dir'] / 'export'
    from ng_utils import export_manual_csvs
    written = export_manual_csvs(read_curated_table(paths['curated']), out)
    check(len(written) == orig[orig.status == 'spot'].nucleus_id.nunique(), 'one CSV per nucleus')
    if written:
        df = pd.read_csv(written[0])
        check(list(df.columns) == ['t', 'z', 'y', 'x', 'nucleus_id'], 'CSV columns')


if __name__ == '__main__':
    main()
