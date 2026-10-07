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

# fake_bridge_backend.py: stands in for the child llama-server in tests/bridge_e2e.sh, so
# the device bridge (docs/bridge.md) is tested with no GPU and no NPU. It takes
# llama-server's arguments, answers --help with the speculative types it supports (serve
# probes for `draft-external` before it launches the child), and on a chat request it does
# what the fork's draft-external type does: POST a draft request to --spec-external-addr
# and record what came back.
#
# FAKE_BRIDGE_EVENTS: the file it writes {"argv": [...], "addr": ..., "draft": [...]} to.
# FAKE_BRIDGE_NO_EXTERNAL=1: answer --help without draft-external, as a fork without the
# hook does.
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def value(flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


if "--help" in argv:
    types = ["none", "draft-simple", "draft-mtp", "draft-dflash", "ngram-simple"]
    if not os.environ.get("FAKE_BRIDGE_NO_EXTERNAL"):
        types.insert(3, "draft-external")
    print("--spec-type [" + "|".join(types) + "]")
    sys.exit(0)

port = int(value("--port"))
addr = value("--spec-external-addr") or ""
events = os.environ["FAKE_BRIDGE_EVENTS"]

# The child's own view of what serve told it. The bridge is asked exactly as the fork's
# draft-external implementation would ask it (docs/bridge.md, "The protocol").
prompt = [1, 2, 3]
draft = []
error = ""
if addr:
    body = json.dumps({"op": "draft", "seq": 0, "n_past": 3, "id_last": 4, "prompt": prompt,
                       "n_max": int(value("--spec-draft-n-max") or 4)}).encode()
    try:
        req = urllib.request.Request("http://" + addr + "/v1/draft", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            draft = json.loads(r.read()).get("draft", [])
    except Exception as e:  # the bridge is a side channel: a child that cannot reach it decodes normally
        error = str(e)

with open(events, "w") as f:
    json.dump({"argv": argv, "addr": addr, "draft": draft, "error": error}, f)


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
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        msg = {"message": {"role": "assistant", "content": "Paris"}, "index": 0, "finish_reason": "stop"}
        self._json(200, {"id": "x", "model": "fake-backend-id", "choices": [msg],
                         "timings": {"prompt_per_second": 500.0, "predicted_per_second": 20.0}})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
