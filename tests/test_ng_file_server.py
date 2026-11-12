"""Tests for the data file server used by ng_curate.py."""

import concurrent.futures
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

pytest.importorskip('neuroglancer')
requests = pytest.importorskip('requests')

import ng_server  # noqa: E402

DATA = bytes(range(256)) * 40  # 10240 bytes


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp('data')
    (root / 'a.zarr').mkdir()
    (root / 'a.zarr' / 'zarr.json').write_text('{"zarr_format": 3}')
    (root / 'a.zarr' / 'shard').write_bytes(DATA)
    outside = tmp_path_factory.mktemp('outside')
    (outside / 'secret').write_text('no')
    (root / 'link').symlink_to(outside, target_is_directory=True)
    srv = ng_server.FixedPortFileServer(root, '127.0.0.1', 0)
    yield f'http://127.0.0.1:{srv.port}/{srv.token}'
    srv.stop()


def test_full_file(server):
    r = requests.get(f'{server}/a.zarr/shard')
    assert r.status_code == 200 and r.content == DATA
    assert r.headers['Access-Control-Allow-Origin'] == '*'
    assert r.headers['Accept-Ranges'] == 'bytes'


def test_json_content_type(server):
    r = requests.get(f'{server}/a.zarr/zarr.json')
    assert r.json() == {'zarr_format': 3}
    assert r.headers['Content-Type'] == 'application/json'


@pytest.mark.parametrize('rng, expected, content_range', [
    ('bytes=0-99', DATA[:100], 'bytes 0-99/10240'),
    ('bytes=5000-5099', DATA[5000:5100], 'bytes 5000-5099/10240'),
    ('bytes=-16', DATA[-16:], 'bytes 10224-10239/10240'),
    ('bytes=10000-', DATA[10000:], 'bytes 10000-10239/10240'),
    ('bytes=10200-20000', DATA[10200:], 'bytes 10200-10239/10240'),
    ('bytes=-99999', DATA, None),  # suffix longer than the file -> whole file
])
def test_ranges(server, rng, expected, content_range):
    r = requests.get(f'{server}/a.zarr/shard', headers={'Range': rng})
    assert r.content == expected
    assert r.headers.get('Content-Range') == content_range
    assert r.status_code == (206 if content_range else 200)


def test_unsatisfiable_range(server):
    r = requests.get(f'{server}/a.zarr/shard', headers={'Range': 'bytes=20000-20100'})
    assert r.status_code == 416


def test_head(server):
    r = requests.head(f'{server}/a.zarr/shard')
    assert r.status_code == 200 and r.headers['Content-Length'] == str(len(DATA))
    assert r.content == b''


def test_missing_and_directory(server):
    assert requests.get(f'{server}/a.zarr/nope').status_code == 404
    assert requests.get(f'{server}/a.zarr').status_code == 404


def test_wrong_token(server):
    base, _ = server.rsplit('/', 1)
    assert requests.get(f'{base}/wrongtoken/a.zarr/shard').status_code == 404


def test_no_escape_with_dotdot(server):
    r = requests.get(f'{server}/a.zarr/%2E%2E/%2E%2E/etc/passwd')
    assert r.status_code in (403, 404)


def test_symlinks_inside_root_are_followed(server):
    # e2e tests serve a temp dir of symlinks to the prepared data
    assert requests.get(f'{server}/link/secret').text == 'no'


def test_parallel_range_reads(server):
    def read(i):
        r = requests.get(f'{server}/a.zarr/shard', headers={'Range': f'bytes={i}-{i + 9}'})
        return r.content == DATA[i:i + 10]
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        assert all(ex.map(read, range(0, 10000, 37)))


# ============================================================================
# Read-ahead along time
# ============================================================================

ts = pytest.importorskip('tensorstore')


@pytest.fixture
def sharded_image(tmp_path, monkeypatch):
    """A tiny 4-timepoint TZYX sharded zarr3 array laid out like ng_prepare's output."""
    import numpy as np
    import ng_prepare
    monkeypatch.setattr(ng_prepare, 'N_LEVELS', 1)
    axes = [{'name': n, 'type': t} for n, t in
            (('T', 'time'), ('Z', 'space'), ('Y', 'space'), ('X', 'space'))]
    out = tmp_path / 'img.ome.zarr'
    ng_prepare.init_pyramid(out, (4, 8, 16, 16), np.uint16, 1, (4, 8, 8), axes,
                            [1.0, 1.0, 1.0, 1.0], name='img')
    rng = np.random.default_rng(0)
    for t in range(4):
        ng_prepare.write_pyramid_timepoint(
            out, (t,), rng.integers(0, 1000, (8, 16, 16), dtype=np.uint16), lambda v: v)
    return out / '0'


def _read_index(shard_path, n):
    raw = shard_path.read_bytes()[-(n * 16 + 4):]
    import numpy as np
    return np.frombuffer(raw[:n * 16], dtype='<u8').reshape(n, 2)


def test_read_ahead_fetches_same_chunk_of_next_timepoints(sharded_image, tmp_path):
    cache = ng_server.FileCache()
    pf = ng_server.ShardPrefetcher(cache, ahead=2, behind=1, workers=2)
    shard = lambda t: sharded_image / 'c' / str(t) / '0' / '0' / '0'  # noqa: E731
    n = 2 * 2 * 2  # (8,16,16) shard / (4,8,8) inner chunks
    idx1 = _read_index(shard(1), n)
    off, nbytes = (int(v) for v in idx1[5])

    pf.on_read(str(shard(1)), off, nbytes)  # browser read inner chunk 5 at t=1
    pf.executor.shutdown(wait=True)

    for t in (2, 3, 0):  # ahead 2, behind 1
        o, b = (int(v) for v in _read_index(shard(t), n)[5])
        assert cache.contains((str(shard(t)), o, b)), f't={t} chunk 5 not prefetched'
        assert cache.get((str(shard(t)), o, b)) == shard(t).read_bytes()[o:o + b]
    # only chunk 5 was read ahead, not other chunks
    o, b = (int(v) for v in _read_index(shard(2), n)[4])
    assert not cache.contains((str(shard(2)), o, b))


def test_read_ahead_ignores_unsharded_and_out_of_range(sharded_image, tmp_path):
    cache = ng_server.FileCache()
    pf = ng_server.ShardPrefetcher(cache, ahead=3, behind=0, workers=2)
    plain = tmp_path / 'plain.bin'
    plain.write_bytes(b'x' * 100)
    pf.on_read(str(plain), 0, 10)  # not a zarr chunk path: no-op
    shard3 = sharded_image / 'c' / '3' / '0' / '0' / '0'
    o, b = (int(v) for v in _read_index(shard3, 8)[0])
    pf.on_read(str(shard3), o, b)  # last timepoint: nothing ahead
    pf.executor.shutdown(wait=True)
    assert all(k[0] == str(shard3) for k in cache._data)


def test_server_serves_prefetched_ranges_from_cache(sharded_image):
    root = sharded_image.parent.parent
    srv = ng_server.FixedPortFileServer(root, '127.0.0.1', 0)
    base = f'http://127.0.0.1:{srv.port}/{srv.token}/{sharded_image.parent.name}/0/c'
    try:
        idx0 = _read_index(sharded_image / 'c' / '0' / '0' / '0' / '0', 8)
        idx1 = _read_index(sharded_image / 'c' / '1' / '0' / '0' / '0', 8)
        o0, b0 = (int(v) for v in idx0[3])
        requests.get(f'{base}/0/0/0/0', headers={'Range': f'bytes={o0}-{o0 + b0 - 1}'})
        o1, b1 = (int(v) for v in idx1[3])
        key = (str(sharded_image / 'c' / '1' / '0' / '0' / '0'), o1, b1)
        import time
        for _ in range(50):
            if ng_server.ChunkFileHandler.cache.contains(key):
                break
            time.sleep(0.05)
        assert ng_server.ChunkFileHandler.cache.contains(key)
        r = requests.get(f'{base}/1/0/0/0', headers={'Range': f'bytes={o1}-{o1 + b1 - 1}'})
        assert r.status_code == 206
        assert r.content == (sharded_image / 'c' / '1' / '0' / '0' / '0').read_bytes()[o1:o1 + b1]
    finally:
        srv.stop()


# ============================================================================
# File descriptor cache
# ============================================================================

def _files(tmp_path, n, size=1000):
    paths = []
    for i in range(n):
        p = tmp_path / f'f{i}'
        p.write_bytes(bytes([i]) * size)
        paths.append(str(p))
    return paths


def test_fd_in_use_is_not_closed_by_eviction(tmp_path):
    import os
    paths = _files(tmp_path, 6)
    cache = ng_server.FileCache()
    cache.MAX_OPEN_FILES = 2
    entry = cache._acquire(paths[0])  # a read is using this descriptor
    for p in paths[1:]:  # other reads push it out of the LRU
        cache.read(p, 0, 10, store=False)
    assert os.pread(entry[0], 10, 0) == bytes([0]) * 10  # still file 0, still open
    cache._release(entry)
    cache.read(paths[1], 0, 10, store=False)  # now it may be closed
    assert len(cache._fds) <= 2


def test_short_read_raises_and_is_not_cached(tmp_path):
    paths = _files(tmp_path, 1, size=100)
    cache = ng_server.FileCache()
    with pytest.raises(OSError):
        cache.read(paths[0], 90, 20)
    assert not cache.contains((paths[0], 90, 20))


def test_many_files_parallel_reads_stay_correct(tmp_path):
    import random
    paths = _files(tmp_path, 40, size=5000)
    cache = ng_server.FileCache()
    cache.MAX_OPEN_FILES = 4

    def work(k):
        rnd = random.Random(k)
        for _ in range(200):
            i = rnd.randrange(40)
            off = rnd.randrange(0, 4900)
            if cache.read(paths[i], off, 100, store=False) != bytes([i]) * 100:
                return False
        return True
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        assert all(ex.map(work, range(32)))
