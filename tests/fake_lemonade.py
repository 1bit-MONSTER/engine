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
# fake_lemonade.py PORT RECORD: a Lemonade stand-in for tests/pm_route.sh. Answers
# GET /api/v1/models and POST /api/v1/chat/completions as the expert the request names, and
# appends every chat request's model id and user text to RECORD (one JSON per line), so the
# test can check which expert the Project Manager asked for.
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, record = int(sys.argv[1]), sys.argv[2]


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
        self._json(200, {"object": "list", "data": [{"id": "Qwen2.5-Coder-7B-Instruct-1bit", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        user = next((m["content"] for m in req["messages"] if m["role"] == "user"), "")
        with open(record, "a") as f:
            f.write(json.dumps({"model": req.get("model"), "user": user, "max_tokens": req.get("max_tokens")}) + "\n")
        if req.get("model") == "broken-expert":
            self._json(500, {"error": {"message": "the expert fell over"}})
            return
        self._json(200, {"id": "exp", "object": "chat.completion", "model": req.get("model"),
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant",
                                                  "content": "EXPERT_ANSWER: void rev(char *s) { /* ... */ } /* tests follow */"}}]})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
