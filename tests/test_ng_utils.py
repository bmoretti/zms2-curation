"""Tests for the neuroglancer curation helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from ng_utils import (
    TrackIndex,
    annotation_to_row,
    next_target_frame,
    row_to_annotation,
    empty_curated_table,
    export_manual_csvs,
    frame_to_ng,
    import_manual_csvs,
    nearest_index,
    ng_to_frame,
    ng_to_voxel,
    ng_units_and_scales,
    normalize_curated_table,
    read_curated_table,
    spatiotemporal_axes,
    voxel_to_ng,
    write_curated_table,
)


# ============================================================================
# Coordinate conversion
# ============================================================================

class TestCoordinates:

    def test_ome_sources_need_no_offset(self):
        # neuroglancer shifts OME-Zarr by half a voxel, so voxel centres stay at integers
        np.testing.assert_allclose(voxel_to_ng([17, 194, 1295, 1775]), [17, 194, 1295, 1775])

    def test_round_trip(self):
        pts = np.array([[17.0, 194.0, 1295.1107477959392, 1775.4247367360936]])
        np.testing.assert_allclose(ng_to_voxel(voxel_to_ng(pts)), pts)

    @pytest.mark.parametrize('t_ng, frame', [(17.0, 17), (17.49, 17), (16.5, 17), (17.5, 18),
                                              (0.0, 0), (-0.4, 0)])
    def test_frame_from_ng_position(self, t_ng, frame):
        # matches neuroglancer's Math.round snapping for voxel-centre-at-integer dims
        assert ng_to_frame(t_ng) == frame

    def test_frame_round_trip(self):
        for f in (0, 1, 42, 399):
            assert ng_to_frame(frame_to_ng(f)) == f
            assert ng_to_frame(voxel_to_ng([f])[0]) == f


class TestNearestIndex:

    def test_empty_returns_none(self):
        assert nearest_index(np.zeros((0, 3)), [0, 0, 0]) is None

    def test_picks_closest(self):
        pts = [[0, 0, 0], [10, 10, 10], [5, 5, 5]]
        assert nearest_index(pts, [6, 6, 6]) == 2

    def test_respects_anisotropy(self):
        # 3 voxels in z (1 um each) is farther than 5 voxels in x (0.347 um each)
        pts = [[3, 0, 0], [0, 0, 5]]
        assert nearest_index(pts, [0, 0, 0], scales_zyx=(1.0, 0.347, 0.347)) == 1


# ============================================================================
# OME metadata
# ============================================================================

def _write_v04(path, scale):
    path.mkdir()
    (path / '.zattrs').write_text(json.dumps({'multiscales': [{
        'version': '0.4',
        'axes': [{'name': 'T', 'type': 'time', 'unit': 'second'},
                 {'name': 'C', 'type': 'channel'},
                 {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
                 {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
                 {'name': 'X', 'type': 'space', 'unit': 'micrometer'}],
        'datasets': [{'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': scale}]}],
    }]}))


class TestOmeAxes:

    def test_v04_drops_channel(self, tmp_path):
        _write_v04(tmp_path / 'img.zarr', [1.0, 1.0, 1.0, 0.347, 0.347])
        axes = spatiotemporal_axes(tmp_path / 'img.zarr')
        assert axes['names'] == ['T', 'Z', 'Y', 'X']
        assert axes['scales'] == [1.0, 1.0, 0.347, 0.347]
        assert axes['channel_axis'] == 1

    def test_v05_zarr_json(self, tmp_path):
        p = tmp_path / 'seg.ome.zarr'
        p.mkdir()
        (p / 'zarr.json').write_text(json.dumps({
            'zarr_format': 3, 'node_type': 'group',
            'attributes': {'ome': {'version': '0.5', 'multiscales': [{
                'axes': [{'name': 'T', 'type': 'time', 'unit': 'second'},
                         {'name': 'Z', 'type': 'space', 'unit': 'micrometer'}],
                'datasets': [{'path': '0', 'coordinateTransformations': [
                    {'type': 'scale', 'scale': [1.0, 2.0]}]}],
            }]}}}))
        axes = spatiotemporal_axes(p)
        assert axes['names'] == ['T', 'Z']
        assert axes['scales'] == [1.0, 2.0]
        assert axes['channel_axis'] is None

    def test_minutes_become_seconds(self):
        units, scales = ng_units_and_scales({'units': ['minute', 'micrometer'], 'scales': [2.0, 0.5]})
        assert units == ['s', 'um']
        assert scales == [120.0, 0.5]


# ============================================================================
# Tracks
# ============================================================================

@pytest.fixture
def tracks():
    return TrackIndex(pd.DataFrame({
        'track_id': [7, 7, 7, 3, 3],
        't': [10, 11, 13, 0, 1],
        'z': [1.0, 2.0, 4.0, 50.0, 51.0],
        'y': [0.0, 0.0, 0.0, 0.0, 0.0],
        'x': [0.0, 0.0, 0.0, 0.0, 0.0],
    }))


class TestTrackIndex:

    def test_span(self, tracks):
        assert tracks.span(7) == (10, 13)
        assert tracks.span(3) == (0, 1)

    def test_contains(self, tracks):
        assert 7 in tracks and 3 in tracks and 99 not in tracks

    def test_centroid_exact(self, tracks):
        np.testing.assert_allclose(tracks.centroid(7, 11), [2.0, 0.0, 0.0])

    def test_centroid_gap_uses_nearest(self, tracks):
        # t=12 is missing; 11 and 13 are equally near, argmin takes the first
        np.testing.assert_allclose(tracks.centroid(7, 12), [2.0, 0.0, 0.0])

    def test_centroid_outside_span_clamps(self, tracks):
        np.testing.assert_allclose(tracks.centroid(7, 100), [4.0, 0.0, 0.0])
        np.testing.assert_allclose(tracks.centroid(7, 0), [1.0, 0.0, 0.0])


# ============================================================================
# Curated table and CSV compatibility
# ============================================================================

NAPARI_CSV = """t,z,y,x,nucleus_id
17.0,194.0,1295.1107477959392,1775.4247367360936,187620
18.0,193.0,1300.0056473152708,1781.6182830666767,187620
"""


class TestCuratedTable:

    def test_normalize_fills_defaults(self):
        df = normalize_curated_table(pd.DataFrame({
            't': [1], 'z': [2.0], 'y': [3.0], 'x': [4.0], 'nucleus_id': [5]}))
        assert list(df.columns) == list(empty_curated_table().columns)
        assert df.loc[0, 'status'] == 'spot'
        assert df.loc[0, 'source'] == 'manual'
        assert pd.isna(df.loc[0, 'auto_id'])
        assert len(df.loc[0, 'annotation_id']) == 32

    def test_round_trip_parquet(self, tmp_path):
        df = normalize_curated_table(pd.DataFrame({
            't': [1, 2], 'z': [2.0, 3.0], 'y': [3.0, 4.0], 'x': [4.0, 5.0],
            'nucleus_id': [5, 5], 'status': ['spot', 'rejected'],
            'source': ['manual', 'auto'], 'auto_id': [pd.NA, 12]}))
        path = tmp_path / 'ng' / 'curated.parquet'
        write_curated_table(df, path)
        back = read_curated_table(path)
        pd.testing.assert_frame_equal(back, df)
        assert not list(path.parent.glob('*.tmp'))

    def test_written_file_is_not_private(self, tmp_path):
        path = tmp_path / 'curated.parquet'
        write_curated_table(empty_curated_table(), path)
        assert path.stat().st_mode & 0o044  # group/other can read (with a normal umask)

    def test_missing_file_gives_empty_table(self, tmp_path):
        assert len(read_curated_table(tmp_path / 'nope.parquet')) == 0

    def test_csv_import_export_round_trip(self, tmp_path):
        src = tmp_path / 'napari'
        src.mkdir()
        (src / 'manual_points_df_187620.csv').write_text(NAPARI_CSV)
        table = import_manual_csvs(src)
        assert len(table) == 2 and (table['nucleus_id'] == 187620).all()

        written = export_manual_csvs(table, tmp_path / 'out')
        assert [p.name for p in written] == ['manual_points_df_187620.csv']
        original = pd.read_csv(src / 'manual_points_df_187620.csv')
        exported = pd.read_csv(written[0])
        assert list(exported.columns) == ['t', 'z', 'y', 'x', 'nucleus_id']
        pd.testing.assert_frame_equal(exported, original)

    def test_export_skips_rejected(self, tmp_path):
        table = normalize_curated_table(pd.DataFrame({
            't': [1, 2], 'z': [0.0, 0.0], 'y': [0.0, 0.0], 'x': [0.0, 0.0],
            'nucleus_id': [5, 5], 'status': ['spot', 'rejected']}))
        written = export_manual_csvs(table, tmp_path)
        assert len(pd.read_csv(written[0])) == 1


class TestTrackGroups:

    def test_group_span_and_centroid(self, tracks):
        assert tracks.group_span([7, 3]) == (0, 13)
        np.testing.assert_allclose(tracks.centroid([7, 3], 1), [51.0, 0.0, 0.0])
        np.testing.assert_allclose(tracks.centroid([7, 3], 11), [2.0, 0.0, 0.0])

    def test_group_ignores_unknown_ids(self, tracks):
        assert tracks.group_span([7, 999]) == (10, 13)


# ============================================================================
# Annotation <-> table rows
# ============================================================================

class TestAnnotationRows:

    def test_round_trip_manual(self):
        row = normalize_curated_table(pd.DataFrame({
            't': [17], 'z': [194.0], 'y': [1295.11], 'x': [1775.42], 'nucleus_id': [187620]})).iloc[0]
        ann = row_to_annotation(row)
        assert ann['point'][0] == 17.0
        assert ann['segments'] == [[187620]]
        assert ann['props'] == [0, 0, -1]
        back = annotation_to_row(ann['id'], ann['point'], ann['segments'], ann['props'])
        assert back['t'] == 17 and back['nucleus_id'] == 187620
        assert back['status'] == 'spot' and back['source'] == 'manual'
        assert pd.isna(back['auto_id'])
        np.testing.assert_allclose([back['z'], back['y'], back['x']], [194.0, 1295.11, 1775.42],
                                   atol=1e-9)

    def test_round_trip_rejected_auto(self):
        row = normalize_curated_table(pd.DataFrame({
            't': [3], 'z': [1.0], 'y': [2.0], 'x': [3.0], 'nucleus_id': [5],
            'status': ['rejected'], 'source': ['auto'], 'auto_id': [42]})).iloc[0]
        ann = row_to_annotation(row)
        assert ann['props'] == [1, 1, 42]
        back = annotation_to_row(ann['id'], ann['point'], ann['segments'], ann['props'])
        assert (back['status'], back['source'], back['auto_id']) == ('rejected', 'auto', 42)

    def test_unlinked_annotation(self):
        back = annotation_to_row('a', [2.0, 1.0, 1.0, 1.0], [[]], [0, 0, -1])
        assert back['nucleus_id'] == -1

    def test_missing_props_use_defaults(self):
        back = annotation_to_row('a', [2.0, 1.0, 1.0, 1.0], None, [])
        assert (back['status'], back['source']) == ('spot', 'manual')


class TestNextTargetFrame:

    def test_prefers_undecided_after_current(self):
        assert next_target_frame(5, (0, 10), [2, 7], [])[0] == 7

    def test_wraps_around(self):
        assert next_target_frame(8, (0, 10), [2], [])[0] == 2

    def test_falls_back_to_uncurated(self):
        frame, reason = next_target_frame(0, (0, 3), [], [0, 1, 3])
        assert frame == 2 and 'without' in reason

    def test_all_done(self):
        assert next_target_frame(0, (0, 2), [], [0, 1, 2])[0] is None
