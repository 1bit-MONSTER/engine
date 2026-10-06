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
# fake_pm_backend.py: a llama-server stand-in for tests/pm_route.sh, playing the Project
# Manager (ZAYA). Takes llama-server's arguments (--port; the rest ignored), answers /health,
# and on /v1/chat/completions:
#   - a user message that mentions code, with tools offered and no tool result yet: a
#     delegate(expert="code", task=...) call. PM_CALL_FORM picks how the call is written:
#     "openai" (a tool_calls array, llama-server's parsed form) or "zyphra" (ZAYA's own
#     <zyphra_tool_call><function=delegate><parameter=..> text, which serve must parse itself);
#   - a tool result present: the final answer, which quotes the tool result so the test can see
#     the expert's answer (or the failure text) reach the client;
#   - anything else: a plain answer, no tool call.
# Every request body is appended to $PM_RECORD (one JSON per line) when that is set.
# /v1/completions answers a fixed text, so the test can see that route still forwards.
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
FORM = os.environ.get("PM_CALL_FORM", "openai")
RECORD = os.environ.get("PM_RECORD")

ZYPHRA = ('<zyphra_tool_call>\n<function=delegate>\n<parameter=expert>\ncode\n</parameter>\n'
          '<parameter=task>\n{task}\n</parameter>\n<parameter=context>\nC99, no libc beyond string.h\n</parameter>\n'
          '</function>\n')  # the stop sequence cuts the closing </zyphra_tool_call>


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
        self._json(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if RECORD:
            with open(RECORD, "a") as f:
                f.write(json.dumps({"path": self.path, "body": req}) + "\n")
        if self.path != "/v1/chat/completions":
            self._json(200, {"id": "x", "model": "fake-backend-id",
                             "choices": [{"index": 0, "text": "plain completion", "finish_reason": "stop"}]})
            return
        msgs = req["messages"]
        tool_results = [m for m in msgs if m.get("role") == "tool"]
        last_user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        if tool_results:
            text = "The expert answered:\n" + tool_results[-1]["content"]
            message = {"role": "assistant", "content": text}
            finish = "stop"
        elif req.get("tools") and ("code" in last_user.lower() or "function" in last_user.lower()):
            task = "Write " + last_user
            if FORM == "zyphra":
                message = {"role": "assistant", "content": ZYPHRA.format(task=task)}
                finish = "stop"
            else:
                message = {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_1", "type": "function",
                                           "function": {"name": "delegate",
                                                        "arguments": json.dumps({"expert": "code", "task": task,
                                                                                 "context": "C99"})}}]}
                finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "Hi there, I answered this myself."}
            finish = "stop"
        self._json(200, {"id": "x", "object": "chat.completion", "created": 1, "model": "fake-backend-id",
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


time.sleep(0.3)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
