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
# fake_spawn_exec.py: a stand-in for the agent-worker executor behind the PM's spawn_worker tool
# (--pm-spawn-url), for tests/pm_route.sh. Answers /health, records every proposal it is POSTed to
# $SPAWN_RECORD (one JSON per line) and:
#   - accepts any proposal whose worker is not "rejected-1", answering the fields the engine reads;
#   - refuses worker "rejected-1" with HTTP 400 and a reason, the way a real executor refuses a
#     proposal it will not run.
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])
RECORD = os.environ.get("SPAWN_RECORD")


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
        self._json(200 if self.path == "/health" else 404, {"ok": True})

    def do_POST(self):
        proposal = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if RECORD:
            with open(RECORD, "a") as f:
                f.write(json.dumps(proposal) + "\n")
        if self.path != "/propose":
            return self._json(404, {"error": "not found"})
        worker = proposal.get("worker", "")
        if worker == "rejected-1":
            return self._json(400, {"accepted": False,
                                    "reason": "model 'deepseek/deepseek-flash' is not on the ryzen allowlist "
                                              "(local providers only)",
                                    "detail": "deepseek/deepseek-flash (remote providers are rejected)"})
        self._json(200, {"accepted": True, "worker": worker, "box": proposal.get("box"),
                         "class": proposal.get("class"), "model": "local/fake-model",
                         "session": "pi-" + worker, "brief_path": "/tmp/fake-brief.md",
                         "ui_state_at_kickoff": "idle"})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
