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

# fake_backend_route.py: stands in for llama-server (--device Vulkan0/HRX0,
# --port) and zinc (-p) in tests/laya_route_e2e.sh, so the Laya per-request
# router is tested without a GPU. It names the device it stands in for in every
# reply, so the test can see which backend a request reached.
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def device():
    if "--device" in argv:
        d = argv[argv.index("--device") + 1]
        return "hrx" if "HRX" in d else "vulkan"
    return "zinc"  # zinc is the only backend started without --device


dev = device()
port = None
for i, a in enumerate(argv):
    if a in ("--port", "-p") and i + 1 < len(argv):
        port = int(argv[i + 1])
        break
assert port, "no port in " + " ".join(argv)


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
        text = "device:" + dev
        if not req.get("stream"):
            msg = {"message": {"role": "assistant", "content": text}} if "chat" in self.path else {"text": text}
            self._json(200, {"id": "x", "model": "fake", "choices": [dict(index=0, finish_reason="stop", **msg)]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {"id": "x", "model": "fake", "choices": [{"index": 0, "delta": {"content": text}}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


time.sleep(0.5)  # a real backend takes a moment to load; serve must answer 503 meanwhile
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
