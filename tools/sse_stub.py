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
"""Device-free check of the probe's stream parser: a stub OpenAI-compatible server that streams the same words in
three different shapes (delta.content, delta.reasoning_content, message.content), plus a tools reply. If the probe
reports first_token_s and text for all three, the parser is right and any failure against FLM is FLM's shape, not mine."""
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer

SHAPE = "delta.content"
SHAPES = {"delta.content": lambda w: {"choices": [{"delta": {"content": w}}]},
          "delta.reasoning_content": lambda w: {"choices": [{"delta": {"reasoning_content": w}}]},
          "message.content": lambda w: {"choices": [{"message": {"content": w}}]}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not body.get("stream"):
            out = {"choices": [{"message": {"content": "stub", "tool_calls": None}}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for w in ["one ", "two ", "three"]:
            chunk = json.dumps(SHAPES[SHAPE](w)).encode()
            self.wfile.write(b"data: " + chunk + b"\n\n"); self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream the same three words in one SSE shape, for a parser under test.")
    ap.add_argument("port", type=int, help="loopback port to listen on")
    ap.add_argument("shape", choices=sorted(SHAPES), help="which envelope each chunk uses")
    a = ap.parse_args(argv)
    global SHAPE
    SHAPE = a.shape
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
