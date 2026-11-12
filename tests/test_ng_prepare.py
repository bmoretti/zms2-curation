"""Tests for ng_prepare.py on small synthetic data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

ts = pytest.importorskip('tensorstore')
pytest.importorskip('neuroglancer')

import ng_prepare  # noqa: E402
from ng_prepare import (  # noqa: E402
    build_segment_properties,
    downsample_labels,
    downsample_mean,
    init_pyramid,
    level_shape,
    load_auto_spots,
    set_annotation_bounds_to_image,
    shard_shape,
    write_pyramid_timepoint,
)

AXES_TZYX = [{'name': 'T', 'type': 'time', 'unit': 'second'},
             {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
             {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
             {'name': 'X', 'type': 'space', 'unit': 'micrometer'}]


class TestShapes:

    def test_level_shape_rounds_up(self):
        assert level_shape((369, 2199, 2842), 0) == (369, 2199, 2842)
        assert level_shape((369, 2199, 2842), 1) == (185, 1100, 1421)
        assert level_shape((369, 2199, 2842), 3) == (47, 275, 356)

    def test_level_shape_matches_stride(self):
        vol = np.zeros((7, 9, 5))
        for level in range(3):
            assert level_shape(vol.shape, level) == vol[::2 ** level, ::2 ** level, ::2 ** level].shape

    def test_shard_is_chunk_multiple(self):
        assert shard_shape((369, 2199, 2842), (64, 256, 256)) == (384, 2304, 3072)


class TestDownsample:

    def test_labels_keep_ids(self):
        vol = np.arange(4 * 4 * 4, dtype=np.uint32).reshape(4, 4, 4)
        out = downsample_labels(vol)
        assert out.shape == (2, 2, 2)
        assert set(out.ravel()) <= set(vol.ravel())

    def test_mean_odd_shape(self):
        vol = np.full((3, 5, 4), 10, dtype=np.uint16)
        out = downsample_mean(vol)
        assert out.shape == (2, 3, 2) and out.dtype == np.uint16
        assert (out == 10).all()

    def test_mean_values(self):
        vol = np.zeros((2, 2, 2), dtype=np.uint16)
        vol[0, 0, 0] = 80
        assert downsample_mean(vol)[0, 0, 0] == 10


class TestPyramid:

    def _open(self, path, level):
        return ts.open({'driver': 'zarr3',
                        'kvstore': {'driver': 'file', 'path': str(path / str(level))}}).result()

    def test_label_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ng_prepare, 'N_LEVELS', 3)
        out = tmp_path / 'seg.ome.zarr'
        shape = (2, 9, 20, 30)
        init_pyramid(out, shape, np.uint32, 1, (4, 8, 8), AXES_TZYX, [1.0, 1.0, 0.347, 0.347],
                     name='segments')
        rng = np.random.default_rng(0)
        vol = rng.integers(0, 1000, size=shape[1:], dtype=np.uint32)
        write_pyramid_timepoint(out, (1,), vol, downsample_labels)

        np.testing.assert_array_equal(self._open(out, 0)[1].read().result(), vol)
        np.testing.assert_array_equal(self._open(out, 2)[1].read().result(), vol[::4, ::4, ::4])
        assert (self._open(out, 0)[0].read().result() == 0).all()  # unwritten t

        meta = json.loads((out / 'zarr.json').read_text())['attributes']['ome']
        assert meta['version'] == '0.5'
        scales = [d['coordinateTransformations'][0]['scale'] for d in meta['multiscales'][0]['datasets']]
        assert scales[2] == [1.0, 4.0, 0.347 * 4, 0.347 * 4]
        arr_meta = json.loads((out / '0' / 'zarr.json').read_text())
        assert arr_meta['dimension_names'] == ['T', 'Z', 'Y', 'X']
        assert arr_meta['codecs'][0]['name'] == 'sharding_indexed'

    def test_image_with_channels(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ng_prepare, 'N_LEVELS', 2)
        out = tmp_path / 'img.ome.zarr'
        axes = AXES_TZYX[:1] + [{'name': 'C', 'type': 'channel'}] + AXES_TZYX[1:]
        init_pyramid(out, (1, 2, 6, 8, 8), np.uint16, 2, (4, 4, 4), axes,
                     [1.0, 1.0, 1.0, 0.26, 0.26], name='image')
        vol = np.full((6, 8, 8), 7, dtype=np.uint16)
        write_pyramid_timepoint(out, (0, 1), vol, downsample_mean)
        np.testing.assert_array_equal(self._open(out, 0)[0, 1].read().result(), vol)
        assert (self._open(out, 1)[0, 1].read().result() == 7).all()
        assert (self._open(out, 1)[0, 0].read().result() == 0).all()


@pytest.fixture
def tracks():
    return pd.DataFrame({
        'track_id': [1, 1, 1, 2, 2, 3],
        't': [0, 1, 2, 3, 4, 3],
        'z': 0.0, 'y': 0.0, 'x': 0.0,
        'parent_track_id': [-1, -1, -1, 1, 1, 1],
    })


@pytest.fixture
def spots():
    return pd.DataFrame({'nucleus_id': [1, 1, 2, -1], 'passed_filters': [True, False, True, True]})


class TestSegmentProperties:

    def test_values(self, tracks, spots):
        info = build_segment_properties(tracks, spots)
        inline = info['inline']
        assert info['@type'] == 'neuroglancer_segment_properties'
        assert inline['ids'] == ['1', '2', '3']
        props = {p['id']: p for p in inline['properties']}
        assert props['length']['values'] == [3, 2, 1]
        assert props['t_start']['values'] == [0, 3, 3]
        assert props['n_auto_spots']['values'] == [2, 1, 0]
        assert props['n_auto_passed']['values'] == [1, 1, 0]
        assert props['parent_track_id']['values'] == [-1, 1, 1]

    def test_tags(self, tracks, spots):
        props = {p['id']: p for p in build_segment_properties(tracks, spots)['inline']['properties']}
        tags = props['tags']['tags']
        named = [[tags[i] for i in v] for v in props['tags']['values']]
        assert named[0] == ['divides', 'has_auto_spots']
        assert named[1] == ['has_parent', 'has_auto_spots']
        assert named[2] == ['has_parent']

    def test_json_serializable(self, tracks, spots):
        json.dumps(build_segment_properties(tracks, spots))


class TestAutoSpots:

    def test_load_auto_spots(self, tmp_path):
        df = pd.DataFrame({
            'spot_id': [5, 5], 't': [0.0, 1.0],
            'z': pd.Series([3, 4], dtype=object), 'y': pd.Series([10, 11], dtype=object),
            'x': pd.Series([20, 21], dtype=object),
            'prob': np.array([0.5, 0.9], dtype=np.float32), 'gauss3d_dog': [1.0, np.nan],
            'nucleus_id': [7.0, np.nan], 'fill_in_spot': [False, True],
            'passed_filters': [True, False], 'data': [None, None]})
        df.to_pickle(tmp_path / 's.pkl')
        out = load_auto_spots(tmp_path / 's.pkl')
        assert out['auto_id'].tolist() == [0, 1]  # unique even though spot_id repeats
        assert out['nucleus_id'].tolist() == [7, -1]
        assert out['gauss3d_dog'].tolist() == [1.0, 0.0]
        assert out['z'].dtype == np.float64

    def test_bounds_are_half_integer_image_extent(self, tmp_path):
        d = tmp_path / 'ann'
        d.mkdir()
        (d / 'info').write_text(json.dumps({
            'lower_bound': [0, 3, 10, 20], 'upper_bound': [1, 4, 11, 21],
            'spatial': [{'key': 'spatial0', 'grid_shape': [1, 1, 1, 1],
                         'chunk_size': [1, 1, 1, 1], 'limit': 2}]}))
        set_annotation_bounds_to_image(d, [400, 369, 2199, 2842])
        info = json.loads((d / 'info').read_text())
        assert info['lower_bound'] == [-0.5] * 4
        assert info['upper_bound'] == [399.5, 368.5, 2198.5, 2841.5]
        assert info['spatial'][0]['chunk_size'] == [400, 369, 2199, 2842]
