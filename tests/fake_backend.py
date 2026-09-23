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
#
# fake_backend.py: stands in for llama-server in tests/smoke_serve.sh, so
# `1bit serve`'s proxy is tested without a GPU. Takes llama-server's arguments
# (-m, --port, anything else ignored), answers /health, and replies to the
# OpenAI routes with a fixed answer, streamed or not. The reply names the
# backend's own model id ("fake-backend-id") so serve's rename is visible.
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
WORDS = ["The", " capital", " of", " France", " is", " Paris", "."]


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
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not req.get("stream"):
            text = "".join(WORDS)
            msg = {"message": {"role": "assistant", "content": text}} if "chat" in self.path else {"text": text}
            self._json(200, {"id": "x", "model": "fake-backend-id", "choices": [dict(index=0, finish_reason="stop", **msg)]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in WORDS:
            chunk = {"id": "x", "model": "fake-backend-id", "choices": [{"index": 0, "delta": {"content": w}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")


time.sleep(0.5)  # a real backend takes a moment to load; serve must answer 503 meanwhile
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
