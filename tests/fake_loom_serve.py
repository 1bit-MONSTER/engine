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
# fake_loom_serve.py: stands in for loom-serve in tests/loom_route.sh, the way tests/fake_backend.py stands in
# for llama-server. It keeps the parts of loom-serve's surface that `1bit serve`'s Loom route depends on
# (hrx-loom 0.1.0 and 0.2.0, written from its documented behaviour and observed answers; no loom-serve code):
#   - flags as --key=value: --model=CARD (a model.json; its model.id is the only model name served),
#     --listen=127.0.0.1:PORT, anything else accepted and ignored;
#   - GET /healthz and /readyz answer 200 at once; /health is 404 ("unknown service endpoint");
#   - GET /v1/status lists the card as "preparing" for 0.5 s, then "ready" (FAKE_LOOM_FAIL=1: "failed",
#     with an error message, and the process keeps running, as loom-serve does);
#   - POST /v1/chat/completions takes a fixed field set and answers 400 for any other field (0.1.0's rule;
#     0.2.0 accepts and passes them on, so the strict fake is the case where a leak fails loudly), 404 for another
#     model name; streamed replies are OpenAI chunks, a usage-only chunk only with stream_options.include_usage,
#     then [DONE]; no `timings` anywhere;
#   - no /v1/completions: 404;
#   - the body of the last chat request is written to $FAKE_LOOM_LAST, so a test can read what serve sent.
# FAKE_LOOM_IMPOSTOR: a detached process answers the port instead (tests/fake_backend.py's impostor).
# FAKE_LOOM_PORT_TAKEN_ONCE=MARKER: the first launch fails the way loom-serve does when its port is taken.
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def flag(name):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


card = flag("model")
model_id = json.load(open(card))["model"]["id"] if card else None
port = int(flag("listen").rsplit(":", 1)[1])
STARTED = time.time()
FIELDS = {"model", "messages", "stream", "tools", "max_tokens", "max_completion_tokens", "temperature", "stream_options",
          "store", "tool_choice", "enable_thinking", "model_options", "top_p", "top_k", "seed", "reasoning_effort",
          "preserve_thinking", "chat_template_kwargs", "media_io_kwargs"}
WORDS = ["The", " capital", " of", " France", " is", " Paris", "."]


def registration():
    if os.environ.get("FAKE_LOOM_FAIL"):
        return "failed"
    return "ready" if time.time() - STARTED > 0.5 else "preparing"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # The pinned loom-serve (hrx-loom 0.2.0) answers a request carrying "Expect: 100-continue" by closing the
    # connection without any response (measured: curl gets nothing in 0.7 ms; the same body without the header is
    # 200). cpp-httplib's client adds that header to every body >= 1024 bytes unless CPPHTTPLIB_EXPECT_100_THRESHOLD
    # is 0, so serve's Loom route answered 502 to every real prompt (engine#3158). http.server would send
    # "100 Continue" by default and hide that; behave like loom-serve instead.
    def handle_expect_100(self):
        self.close_connection = True
        return False

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg, kind="loom_request_error"):
        self._json(code, {"error": {"message": msg, "type": kind, "code": "INVALID_ARGUMENT"}})

    def do_GET(self):
        if self.path in ("/healthz", "/readyz"):
            self._json(200, {"status": "ready"})
        elif self.path == "/v1/status":
            row = {"object": "model", "id": model_id, "owned_by": "local",
                   "loom": {"kind": "text", "endpoint": "/v1/chat/completions", "registration": registration()}}
            if row["loom"]["registration"] == "failed":
                row["loom"]["error"] = {"code": 3, "message": "program/llm.loom:1:1: PARSE/003: unexpected token"}
            self._json(200, {"object": "list", "data": [row] if model_id else [], "server": {}})
        elif self.path == "/v1/models":
            data = [{"object": "model", "id": model_id}] if model_id and registration() == "ready" else []
            self._json(200, {"object": "list", "data": data})
        else:
            self._err(404, "unknown service endpoint", "loom_route_error")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if req.get("model") != model_id or registration() != "ready":
            self._err(404, "model is not registered", "loom_route_error")
            return
        if self.path != "/v1/chat/completions":
            self._err(404, "unknown service endpoint", "loom_route_error")
            return
        if os.environ.get("FAKE_LOOM_LAST"):
            with open(os.environ["FAKE_LOOM_LAST"], "w") as f:
                json.dump(req, f)
        for k in req:
            if k not in FIELDS:
                self._err(400, f"unsupported chat field '{k}'")
                return
        if "stream_options" in req and not req.get("stream"):
            self._err(400, "stream_options requires streaming")
            return
        usage = {"prompt_tokens": 12, "completion_tokens": len(WORDS) + 1, "total_tokens": 12 + len(WORDS) + 1,
                 "prompt_tokens_details": {"cached_tokens": 2}}
        if not req.get("stream"):
            self._json(200, {"id": "chatcmpl-1", "object": "chat.completion", "model": model_id,
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "".join(WORDS)}}],
                             "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": model_id}
        send(dict(base, choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        for w in WORDS:
            send(dict(base, choices=[{"index": 0, "delta": {"content": w}, "finish_reason": None}]))
            time.sleep(0.01)
        send(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]))
        if (req.get("stream_options") or {}).get("include_usage"):
            send(dict(base, choices=[], usage=usage))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


time.sleep(0.2)
if os.environ.get("FAKE_LOOM_PORT_TAKEN_ONCE"):
    marker = os.environ["FAKE_LOOM_PORT_TAKEN_ONCE"]
    if not os.path.exists(marker):
        open(marker, "w").close()
        print("runtime/src/iree/async/socket.c:195: FAILED_PRECONDITION; bind failed (errno 98)", file=sys.stderr, flush=True)
        sys.exit(1)
if os.environ.get("FAKE_LOOM_IMPOSTOR"):
    root = os.getpid()
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            def die_with(pid):
                while True:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        os._exit(0)
                    time.sleep(0.2)
            threading.Thread(target=die_with, args=(root,), daemon=True).start()
            with open(os.environ["FAKE_LOOM_IMPOSTOR"], "w") as f:
                f.write(str(os.getpid()))
            STARTED = 0  # ready at once: the impostor answers as a ready server would
            ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
        os._exit(0)
    while True:
        time.sleep(1)
srv = ThreadingHTTPServer(("127.0.0.1", port), H)
print(json.dumps({"event": "ready", "address": f"127.0.0.1:{port}", "models": 1 if model_id else 0}), file=sys.stderr, flush=True)
srv.serve_forever()
