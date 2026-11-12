"""
Data file server for ng_curate.py: CORS, byte ranges, parallel reads, read-ahead in time.

neuroglancer reads our sharded zarr v3 arrays (one shard file per timepoint and
channel) as many small byte-range requests, one per inner chunk. On the cluster
filesystem each such read takes ~20 ms (p90 ~80 ms), and a time step at curation
zoom needs ~200 of them. So:

- reads run in a thread pool on cached file descriptors, in parallel;
- when the browser reads inner chunk i of the shard for time t, the same chunk i
  of the shards for t+1..t+AHEAD and t-1 is read into an in-memory cache, using
  each shard's index to find where chunk i is. The next time step is then mostly
  served from memory.
"""

import asyncio
import collections
import concurrent.futures
import email.utils
import json
import os
import re
import stat
import threading

import numpy as np
import tornado.httpserver
import tornado.httputil
import tornado.netutil
import tornado.web

import neuroglancer.random_token
from neuroglancer import async_util


AHEAD = 3             # timepoints read ahead
BEHIND = 1            # timepoints read behind (scrolling back)
CACHE_BYTES = 4 << 30
MAX_PENDING = 4000    # drop read-ahead work beyond this (e.g. while holding a key)
UNUSED = np.uint64(2**64 - 1)  # sharding index marker for an empty inner chunk

_CHUNK_PATH = re.compile(r'^(?P<array>.+)/c/(?P<t>\d+)(?P<rest>(?:/\d+)*)$')


def pread_all(fd, length, offset):
    parts, done = [], 0
    while done < length:
        b = os.pread(fd, length - done, offset + done)
        if not b:
            break
        parts.append(b)
        done += len(b)
    return b''.join(parts)


class FileCache:
    """
    Open file descriptors (LRU) plus an LRU of byte ranges, both thread-safe.

    A descriptor is reference-counted while a read uses it and is only closed once
    no read holds it: closing it mid-read would let the OS hand the same number to
    another file, and the read would return (and cache) that file's bytes.
    """

    MAX_OPEN_FILES = 512

    def __init__(self, max_bytes=CACHE_BYTES):
        self.max_bytes = max_bytes
        self._fds = collections.OrderedDict()  # path -> [fd, refcount]
        self._data = collections.OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _acquire(self, path):
        with self._lock:
            entry = self._fds.pop(path, None)
            if entry is None:
                entry = [os.open(path, os.O_RDONLY), 0]
            entry[1] += 1
            self._fds[path] = entry
            # Close least recently used descriptors that no read is using
            excess = len(self._fds) - self.MAX_OPEN_FILES
            for old_path in list(self._fds):
                if excess <= 0:
                    break
                old = self._fds[old_path]
                if old[1] == 0:
                    del self._fds[old_path]
                    os.close(old[0])
                    excess -= 1
            return entry

    def _release(self, entry):
        with self._lock:
            entry[1] -= 1

    def get(self, key):
        with self._lock:
            data = self._data.get(key)
            if data is not None:
                self._data.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return data

    def put(self, key, data):
        with self._lock:
            if key in self._data:
                return
            self._data[key] = data
            self._nbytes += len(data)
            while self._nbytes > self.max_bytes and self._data:
                self._nbytes -= len(self._data.popitem(last=False)[1])

    def contains(self, key):
        with self._lock:
            return key in self._data

    def read(self, path, offset, length, store=True):
        """Bytes [offset, offset+length) of path; raises OSError on a short read."""
        key = (path, offset, length)
        data = self.get(key)
        if data is None:
            entry = self._acquire(path)
            try:
                data = pread_all(entry[0], length, offset)
            finally:
                self._release(entry)
            if len(data) != length:
                raise OSError(f'short read: {len(data)} of {length} bytes at {offset} in {path}')
            if store:
                self.put(key, data)
        return data


class ShardPrefetcher:
    """Reads the same inner chunk of neighbouring timepoints' shards into the cache."""

    def __init__(self, cache, ahead=AHEAD, behind=BEHIND, workers=16):
        self.cache = cache
        self.offsets = [d for d in range(1, ahead + 1)] + [-d for d in range(1, behind + 1)]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._arrays = {}                          # array dir -> shard info or None
        self._indexes = collections.OrderedDict()  # shard path -> (n, 2) uint64, LRU
        self._lock = threading.Lock()
        self._pending = 0
        self._queued = set()

    # ---- metadata

    def array_info(self, array_dir):
        """Shard layout of a zarr v3 array whose first dimension is time, else None."""
        with self._lock:
            if array_dir in self._arrays:
                return self._arrays[array_dir]
        info = None
        try:
            meta = json.loads(open(os.path.join(array_dir, 'zarr.json')).read())
            codec = meta['codecs'][0]
            if codec['name'] == 'sharding_indexed':
                cfg = codec['configuration']
                shard = meta['chunk_grid']['configuration']['chunk_shape']
                inner = cfg['chunk_shape']
                n = int(np.prod([s // c for s, c in zip(shard, inner)]))
                crc = any(c['name'] == 'crc32c' for c in cfg.get('index_codecs', []))
                if shard[0] == 1 and cfg.get('index_location', 'end') == 'end':
                    info = {'n': n, 'index_bytes': n * 16 + (4 if crc else 0),
                            'n_t': meta['shape'][0]}
        except (OSError, ValueError, KeyError, IndexError):
            info = None
        with self._lock:
            self._arrays[array_dir] = info
        return info

    def shard_index(self, path, info):
        """(n, 2) array of (offset, nbytes) per inner chunk, cached; None if no shard."""
        with self._lock:
            idx = self._indexes.get(path)
            if idx is not None:
                self._indexes.move_to_end(path)
                return idx
        try:
            size = os.stat(path).st_size
        except OSError:
            return None
        off = size - info['index_bytes']
        # Same key the browser uses when it reads the index, so that read is a cache hit
        raw = self.cache.read(path, off, info['index_bytes'])
        idx = np.frombuffer(raw[:info['n'] * 16], dtype='<u8').reshape(info['n'], 2)
        with self._lock:
            self._indexes[path] = idx
            while len(self._indexes) > 256:
                self._indexes.popitem(last=False)
        return idx

    # ---- scheduling

    def on_read(self, path, offset, length):
        """Called (from a worker thread) after the browser read [offset, offset+length)."""
        m = _CHUNK_PATH.match(path)
        if m is None:
            return
        info = self.array_info(m['array'])
        if info is None:
            return
        idx = self.shard_index(path, info)
        if idx is None:
            return
        hit = np.flatnonzero((idx[:, 0] == offset) & (idx[:, 1] == length))
        chunk = int(hit[0]) if len(hit) else None  # None: the index itself, or other ranges
        t = int(m['t'])
        for dt in self.offsets:
            tn = t + dt
            if not 0 <= tn < info['n_t']:
                continue
            neighbour = f"{m['array']}/c/{tn}{m['rest']}"
            self._submit(neighbour, info, chunk)

    def _submit(self, path, info, chunk):
        key = (path, chunk)
        with self._lock:
            if key in self._queued or self._pending >= MAX_PENDING:
                return
            self._queued.add(key)
            self._pending += 1
        self.executor.submit(self._fetch, path, info, chunk)

    def _fetch(self, path, info, chunk):
        try:
            idx = self.shard_index(path, info)
            if idx is not None and chunk is not None:
                off, n = idx[chunk]
                if off != UNUSED and n > 0:
                    self.cache.read(path, int(off), int(n))
        except OSError:
            pass
        finally:
            with self._lock:
                self._pending -= 1
                self._queued.discard((path, chunk))


class ChunkFileHandler(tornado.web.RequestHandler):
    """
    Serves data files with CORS and byte ranges; reads run in a thread pool.

    (tornado's StaticFileHandler reads on the event loop, one request at a time,
    and hashes whole files for its ETag.)
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
    cache = FileCache()
    prefetcher = ShardPrefetcher(cache)
    request_log = None  # optional open file: one line per data request (--log-requests)

    def initialize(self, path):
        self.root = os.path.abspath(path)
        self.cache_hit = False

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with, range')
        self.set_header('Access-Control-Expose-Headers', 'content-range')
        self.set_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')

    def options(self, *args):
        self.set_status(204)
        self.finish()

    async def head(self, path):
        await self.get(path, include_body=False)

    async def get(self, path, include_body=True):
        # Symlinks are allowed (normpath, not realpath), '..' escapes are not
        abspath = os.path.normpath(os.path.join(self.root, path))
        if not (abspath == self.root or abspath.startswith(self.root + os.sep)):
            raise tornado.web.HTTPError(403)
        loop = asyncio.get_running_loop()
        try:
            st = await loop.run_in_executor(self.executor, os.stat, abspath)
        except OSError:
            raise tornado.web.HTTPError(404)
        if not stat.S_ISREG(st.st_mode):
            raise tornado.web.HTTPError(404)
        size = st.st_size
        start, end = 0, size
        range_header = self.request.headers.get('Range')
        if range_header:
            parsed = tornado.httputil._parse_request_range(range_header)
            if parsed is None:
                raise tornado.web.HTTPError(416)
            start, end = parsed
            if start is not None and start < 0:  # suffix range
                start = max(0, size + start)
            start = start or 0
            end = size if end is None else min(end, size)
            if start >= size or start >= end:
                self.set_header('Content-Range', f'bytes */{size}')
                raise tornado.web.HTTPError(416)
            if (start, end) != (0, size):
                self.set_status(206)
                self.set_header('Content-Range', f'bytes {start}-{end - 1}/{size}')
        self.set_header('Accept-Ranges', 'bytes')
        self.set_header('Content-Type', 'application/json' if abspath.endswith('.json')
                        else 'application/octet-stream')
        self.set_header('Content-Length', end - start)
        self.set_header('Last-Modified', email.utils.formatdate(st.st_mtime, usegmt=True))
        if include_body:
            ranged = (start, end) != (0, size)
            data = self.cache.get((abspath, start, end - start)) if ranged else None
            self.cache_hit = data is not None
            if data is None:
                try:
                    data = await loop.run_in_executor(
                        self.executor, self.cache.read, abspath, start, end - start, ranged)
                except OSError:
                    raise tornado.web.HTTPError(500)
            self.write(data)
            if ranged:
                self.prefetcher.executor.submit(self._safe_on_read, abspath, start, end - start)
        self.finish()

    @classmethod
    def _safe_on_read(cls, path, offset, length):
        try:
            cls.prefetcher.on_read(path, offset, length)
        except Exception:  # read-ahead is best effort; never break serving
            pass

    def on_finish(self):
        if self.request_log is not None:
            r = self.request
            self.request_log.write(
                f'{r._start_time:.3f}\t{r.request_time() * 1000:.1f}\t{self.get_status()}\t'
                f'{self._headers.get("Content-Length", 0)}\t{r.method}\t'
                f'{r.headers.get("Range", "")}\t{self.path_args[0] if self.path_args else ""}\t'
                f'{"hit" if self.cache_hit else ""}\n')
            self.request_log.flush()


class FixedPortFileServer(async_util.BackgroundTornadoServer):
    """Serves `static_dir` under a random token on a chosen port (0 = any)."""

    def __init__(self, static_dir, bind_address, port):
        super().__init__(daemon=True)
        self.static_dir = str(static_dir)
        self.bind_address = bind_address
        self.port = port
        self.token = neuroglancer.random_token.make_random_token()

    def _attempt_to_start_server(self):
        app = tornado.web.Application(
            [(rf'/{self.token}/(.*)', ChunkFileHandler, {'path': self.static_dir})])
        self.http_server = tornado.httpserver.HTTPServer(app)
        sockets = tornado.netutil.bind_sockets(port=self.port, address=self.bind_address)
        self.http_server.add_sockets(sockets)
        self.port = sockets[0].getsockname()[1]
