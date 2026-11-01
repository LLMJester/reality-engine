"""Run 5 dashboard server (read-only). python -m v3.dashboard5.server5 [--port 8766] [--mode real|mini|pilot]

GET /                 index.html (read fresh on every request, so page edits show on refresh)
GET /status.json      feed5.build(mode), rebuilt at most every 1 s; while a rebuild runs, other requests get the last one
                      at once (the mode can be switched with ?mode=mini on a pilot server)
feed5.py is re-imported when it changes on disk, so feed edits need no restart either.
"""
import argparse
import importlib
import json
import pathlib
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).resolve().parent
MIN_INTERVAL = 1.0


class Feed:
    def __init__(self, mode):
        self.mode, self.lock, self.cache, self.mtime, self.mod = mode, threading.Lock(), {}, None, None

    def _module(self):
        mt = (HERE / "feed5.py").stat().st_mtime
        if self.mod is None:
            from v3.dashboard5 import feed5
            self.mod, self.mtime = feed5, mt
        elif mt != self.mtime:
            cache = self.mod.CACHE
            self.mod = importlib.reload(self.mod)
            self.mod.CACHE = cache                       # keep the parsed records across a reload
            self.mtime = mt
        return self.mod

    def get(self, mode=None):
        mode = "real" if self.mode == "real" else (mode or self.mode)     # the real run's dashboard shows only it
        hit = self.cache.get(mode)
        if hit and time.time() - hit[0] < MIN_INTERVAL:
            return hit[1]
        if hit and not self.lock.acquire(blocking=False):
            return hit[1]                    # a rebuild is under way: answer at once with the last one, never queue
        if not hit:
            self.lock.acquire()
        try:
            hit = self.cache.get(mode)
            if hit and time.time() - hit[0] < MIN_INTERVAL:
                return hit[1]
            try:
                from v3.analysis5 import clean
                d = self._module().build(mode)
                d["switchable"] = self.mode != "real"       # the page offers the run selector only on a pilot server
                body = json.dumps(clean(d), default=str, allow_nan=False).encode("utf-8")
            except Exception:
                sys.stderr.write(traceback.format_exc())
                if hit:
                    return hit[1]
                body = json.dumps({"error": traceback.format_exc()[-1500:]}).encode("utf-8")
            self.cache[mode] = (time.time(), body)
            return body
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    feed = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/status.json":
            mode = (parse_qs(u.query).get("mode") or [None])[0]
            if mode not in (None, "real", "mini", "pilot"):
                mode = None
            self._send(200, self.feed.get(mode), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def make_server(port=8766, mode="real"):
    handler = type("H5", (Handler,), {"feed": Feed(mode)})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--mode", default="real", choices=("real", "mini", "pilot"))
    a = ap.parse_args(argv)
    srv = make_server(a.port, a.mode)
    sys.stderr.write(f"run 5 dashboard on http://127.0.0.1:{a.port}/ (mode {a.mode})\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
