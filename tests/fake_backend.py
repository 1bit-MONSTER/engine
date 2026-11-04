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
#
# GET /slots answers 200 like llama-server's, from "the main loop": once a request has asked for
# "fake_hang", the loop is wedged, so /slots never answers and the request gets no more bytes,
# while /health still answers 200 (what a hung GPU dispatch looks like; tests/watchdog_serve.sh).
import json, os, select, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
WORDS = ["The", " capital", " of", " France", " is", " Paris", "."]
WEDGED = threading.Event()


def _record_slot_action(action, body):
    """Write the slot action's request body where a test can read it (FAKE_SLOT_MARKER): the handoff's own
    contract is what tests/hybrid_route.sh and tests/hybrid_kvshare_route.sh assert, and it is not visible in
    the reply."""
    marker = os.environ.get("FAKE_SLOT_MARKER")
    if marker:
        with open(marker, "w") as f:
            json.dump({"action": action, "body": body}, f)


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
        elif self.path == "/metrics" and "--metrics" in sys.argv:
            body = b"# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total 12\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/slots":
            while WEDGED.is_set():
                time.sleep(1)
            self._json(200, [{"id": 0, "is_processing": False}])
        else:
            self._json(404, {"error": "not found"})

    def client_gone(self):
        r, _, _ = select.select([self.connection], [], [], 0)
        return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # what `--device hybrid` asks of its backend (app/serve_hybrid.h, tests/hybrid_route.sh): the
        # templated prompt, its ids (one per character here), and a slot restore that counts the
        # ids in the provider's state file (which the fake provider writes as one id per line)
        if self.path == "/apply-template":
            self._json(200, {"prompt": "".join(m.get("content", "") for m in req.get("messages", []))})
            return
        if self.path == "/tokenize":
            self._json(200, {"tokens": [ord(c) for c in req.get("content", "")]})
            return
        if self.path.startswith("/slots/"):
            slot, _, action = self.path[len("/slots/"):].partition("?action=")
            if action == "share":
                # the shared-KV protocol (fork 1bit-MONSTER/llama.cpp#164, engine ONEBIT_KV_SHARE=1):
                # connect to the engine's socket and send [uint32 length][JSON] with one fd, then report
                # the export id. No dma-buf here -- any readable fd proves the plumbing (the real one is
                # a dma-buf the provider mmaps). tests/hybrid_kvshare_route.sh
                path = str(req.get("socket", ""))
                import struct
                fd = os.open(__file__, os.O_RDONLY)  # a real fd the provider can pread()
                header = json.dumps({
                    "n_regions": 1,
                    "params": {"n_ctx": 4096, "n_batch": 2048, "n_ubatch": 512, "n_seq_max": 1,
                               "flash_attn": 1, "type_k": 1, "type_v": 1},
                    "regions": [{"offset": 0, "size": 4096, "layout": 1234567}],
                }).encode()
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.connect(path)
                    sock.sendmsg([struct.pack("<I", len(header)) + header],
                                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("<i", fd))])
                finally:
                    sock.close()
                    os.close(fd)
                _SHARE_ID = 7
                self._json(200, {"id_slot": int(slot), "share_id": _SHARE_ID, "n_regions": 1,
                                 "regions": [{"offset": 0, "size": 4096, "layout": 1234567}],
                                 "params": {"n_ctx": 4096, "n_batch": 2048, "n_ubatch": 512, "n_seq_max": 1,
                                            "flash_attn": 1, "type_k": 1, "type_v": 1},
                                 "timings": {"share_ms": 0.25}})
                return
            if action == "adopt":
                # the cells are only meaningful for the export they were written into
                if req.get("share_id") != 7:
                    self._json(400, {"error": {"message": "adopt must quote the share_id"}})
                    return
                toks = req.get("tokens") or []
                _record_slot_action("adopt", req)
                self._json(200, {"id_slot": int(slot), "share_id": 7, "n_adopted": len(toks), "n_read": 0,
                                 "timings": {"adopt_ms": 0.1}})
                return
            if action != "restore":
                self._json(400, {"error": {"message": "Invalid action"}})
                return
            # the file name is a bare name inside --slot-save-path, as llama-server requires
            # (fs_validate_filename): no directory part, no "..", nothing outside that directory
            name = str(req.get("filename", ""))
            slot_dir = sys.argv[sys.argv.index("--slot-save-path") + 1]
            # the path comes from the directory listing, never from the request string itself
            path = None
            for entry in sorted(os.listdir(slot_dir)):
                if entry == name:
                    path = os.path.join(slot_dir, entry)
            if path is None:
                self._json(400, {"error": {"message": "Invalid filename"}})
                return
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
            _record_slot_action("restore", dict(req, n_tokens_in_state_file=n))
            self._json(200, {"id_slot": int(slot), "filename": req["filename"], "n_restored": n, "n_read": 0,
                             "timings": {"restore_ms": 1.5}})
            return
        if req.get("fake_device_lost"):
            # what ggml-hrx logs after a GPU reset; the backend keeps running and fails every request
            print("graph_compute: launch HRX graph replay: hrx-system/runtime/src/iree/hal/drivers/amdgpu/system_event.c:129: "
                  "DATA_LOSS; AMDGPU hardware exception (reset type 0x00000001)", file=sys.stderr, flush=True)
            self._json(500, {"error": {"code": 500, "message": "Compute error.", "type": "server_error"}})
            return
        if req.get("fake_hang"):
            # streamed: two chunks, then nothing; not streamed: nothing at all
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for w in WORDS[:2]:
                    chunk = {"id": "x", "model": "fake-backend-id", "choices": [{"index": 0, "delta": {"content": w}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
            WEDGED.set()
            while True:
                time.sleep(1)
        # "fake_delay": N holds a non-streamed reply N seconds, noting in $FAKE_BACKEND_EVENTS
        # whether serve hung up first (it should once its own client has gone)
        delay = float(req.get("fake_delay", 0))
        if delay and not req.get("stream"):
            end = time.time() + delay
            while time.time() < end:
                if self.client_gone():
                    with open(os.environ["FAKE_BACKEND_EVENTS"], "a") as f:
                        f.write("cancelled\n")
                    return
                time.sleep(0.05)
            with open(os.environ["FAKE_BACKEND_EVENTS"], "a") as f:
                f.write("completed\n")
        if not req.get("stream"):
            text = "".join(WORDS)
            msg = {"message": {"role": "assistant", "content": text}} if "chat" in self.path else {"text": text}
            # llama-server's timings, fixed, for tools/bench.py (tests/bench_selftest.sh)
            timings = {"prompt_per_second": 500.0, "predicted_per_second": 20.0}
            # the slot the request was pinned to and whether it may reuse the slot's prompt
            # (what the hybrid handoff sets; tests/hybrid_route.sh reads them back)
            self._json(200, {"id": "x", "model": "fake-backend-id", "choices": [dict(index=0, finish_reason="stop", **msg)],
                             "timings": timings, "fake_id_slot": req.get("id_slot", -1),
                             "fake_cache_prompt": req.get("cache_prompt", None),
                             # the sampling fields serve forwards (tests/greedy_shortcut.sh)
                             "fake_top_k": req.get("top_k", None),
                             "fake_temperature": req.get("temperature", None)})
            # "fake_exit": N: the backend exits with status N shortly after this reply, on its own
            # (a crashed backend; serve must notice and not wait its SIGTERM grace for a dead pid)
            if "fake_exit" in req:
                threading.Timer(0.2, os._exit, [int(req["fake_exit"])]).start()
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


if os.environ.get("FAKE_BACKEND_LOG_LINES"):
    # what llama.cpp and ggml-hrx write to stderr that serve's /metrics reads (tests/metrics_serve.sh)
    for line in ("trim_graph_programs: HRX graph program cache over its limit (65 programs, limit 64): kept the current one, flush 1",
                 "trim_graph_programs: HRX graph program cache over its limit (65 programs, limit 64): kept the current one, flush 2",
                 "ggml_hrx: CPU fallback: GET_ROWS x2, SOFTPLUS x1"):
        print(line, file=sys.stderr, flush=True)
time.sleep(0.5)  # a real backend takes a moment to load; serve must answer 503 meanwhile
if os.environ.get("FAKE_BACKEND_IMPOSTOR"):
    # A process serve did not start answers on the port instead: this one detaches a server
    # (writing its pid to $FAKE_BACKEND_IMPOSTOR) and stays alive without listening itself.
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            with open(os.environ["FAKE_BACKEND_IMPOSTOR"], "w") as f:
                f.write(str(os.getpid()))
            ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
        os._exit(0)
    while True:
        time.sleep(1)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
