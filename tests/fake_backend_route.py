#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# fake_backend_route.py: stands in for llama-server (--device HRX0, or -ngl 0 on
# the CPU; --port) in tests/laya_route_e2e.sh and laya_gguf_route.sh, so the Laya per-request
# router is tested without a GPU. It names the device it stands in for in every
# reply, so the test can see which backend a request reached.
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def device():
    if "--device" in argv:
        d = argv[argv.index("--device") + 1]
        return "hrx" if "HRX" in d else d.lower()
    if "-ngl" in argv and argv[argv.index("-ngl") + 1] == "0":
        return "cpu"
    sys.exit("fake_backend_route.py: neither --device nor -ngl 0 in " + " ".join(argv))


dev = device()
# the model file's stem, so a test serving several models (tests/laya_models_e2e.sh) sees which answered
model = argv[argv.index("-m") + 1].rsplit("/", 1)[-1].rsplit(".", 1)[0] if "-m" in argv else "fake"
port = None
for i, a in enumerate(argv):
    if a in ("--port", "-p") and i + 1 < len(argv):
        port = int(argv[i + 1])
        break
# engine#3130: `--port 0` REQUESTS an ephemeral port, it does not omit one. `assert port` is falsy for 0, so a launch
# that hands this fake `--port 0` (what a rail doing `serve --port 0` looks like once its children follow) died here
# before it could report anything, and the failure read as "the fake never started".
assert port is not None, "no port in " + " ".join(argv)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "device": dev})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n)) if n else {}
        text = "device:" + dev + " model:" + model
        if not req.get("stream"):
            msg = {"message": {"role": "assistant", "content": text}} if "chat" in self.path else {"text": text}
            # usage and timings in llama-server's shape (fixed numbers: the test checks they are relayed)
            self._json(200, {"id": "x", "model": "fake", "choices": [dict(index=0, finish_reason="stop", **msg)],
                             "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                             "timings": {"prompt_per_second": 400.0, "predicted_per_second": 50.0}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {"id": "x", "model": "fake", "choices": [{"index": 0, "delta": {"content": text}}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


time.sleep(0.5)  # a real backend takes a moment to load; serve must answer 503 meanwhile
_srv = ThreadingHTTPServer(("127.0.0.1", port), H)
# the line llama-server prints (tools/server/server.cpp:339), so a launcher that passed `--port 0` can read back the
# port this fake actually bound -- i.e. the fake behaves like the real backend one level down (engine#3130).
print("listening on http://127.0.0.1:%d" % _srv.server_address[1], file=sys.stderr, flush=True)
# engine#3130 / #3173: the line above is the child announcing a port; the FIRST ANSWER after it is what a readiness
# wait polls. A fake that answers within a millisecond leaves no gap between "bound" and "ready", so a launcher that
# should have refused the foreign line has already been satisfied and the refusal is never observable. FAKE_LLAMA_LOAD_S
# (seconds, default 0 = answer at once, unchanged) opens that gap the way a real llama-server's model load does.
_load_s = float(os.environ.get("FAKE_LLAMA_LOAD_S") or 0)
if _load_s > 0:
    time.sleep(_load_s)
_srv.serve_forever()
