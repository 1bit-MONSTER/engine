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
"""End-to-end test of 1bit-server over HTTP, the way Lemonade drives it.

usage: server_e2e.py --server build/1bit-server --model M.gguf --golden tests/golden/qwen3-0.6b-capitals

Launches the server with Lemonade's flags, waits on /health, and checks
responses against the golden (HF transformers' greedy continuation) and the
OpenAI / llama-server response contract. Standard library only.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

FAILURES = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILURES.append(what)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
    def __init__(self, port):
        self.base = "http://127.0.0.1:%d" % port

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def post(self, path, body, raw=None):
        data = raw if raw is not None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def stream(self, path, body):
        """Returns (list of parsed SSE events, saw [DONE])."""
        req = urllib.request.Request(self.base + path, data=json.dumps(dict(body, stream=True)).encode(),
                                     headers={"Content-Type": "application/json"})
        events, done = [], False
        with urllib.request.urlopen(req, timeout=600) as r:
            check(r.headers.get("Content-Type", "").startswith("text/event-stream"), "stream content type is SSE")
            after_done = 0
            for line in r:
                line = line.decode().rstrip("\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    done = True
                    continue
                after_done += done
                events.append(json.loads(payload))
            check(after_done == 0, "no events after [DONE] (%d events)" % len(events))
        return events, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    meta = json.load(open(args.golden + "/meta.json"))
    port = free_port()
    # The flags Lemonade's HRX/llama.cpp recipes pass, with our device name.
    cmd = [args.server, "-m", args.model, "--ctx-size", "4096", "--device", "cpu", "--port", str(port),
           "--jinja", "--metrics", "--parallel", "1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    c = Client(port)
    try:
        # /health: 503 while loading, then 200 {"status":"ok"}.
        statuses, loading_ok, ready_body = set(), True, None
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                st, body = c.get("/health")
                statuses.add(st)
                if st == 200:
                    ready_body = body
                    break
                loading_ok &= st == 503 and body["error"]["message"] == "Loading model"
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(0.05)
        check(loading_ok, "/health is 503 Loading model while loading (seen: %s)" % sorted(statuses))
        check(ready_body == {"status": "ok"}, "/health turns 200 {\"status\": \"ok\"} when ready")

        st, body = c.get("/v1/models")
        check(st == 200 and body["object"] == "list" and body["data"][0]["object"] == "model", "/v1/models lists the model")

        # Greedy completion must equal HF transformers' greedy continuation.
        req = {"prompt": meta["prompt"], "max_tokens": meta["n_tokens"] - meta["n_prompt"], "temperature": 0}
        st, r = c.post("/v1/completions", req)
        check(st == 200, "/v1/completions 200")
        text = r["choices"][0]["text"]
        check(text == meta["continuation"], "greedy completion equals HF continuation: %r" % text)
        check(r["choices"][0]["finish_reason"] == "length", "finish_reason length at max_tokens")
        check(r["usage"]["prompt_tokens"] == meta["n_prompt"], "usage.prompt_tokens matches the HF tokenization")
        check(r["usage"]["completion_tokens"] == req["max_tokens"], "usage.completion_tokens")
        t = r["timings"]
        check(all(k in t for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_per_second", "cache_n")),
              "timings has the fields Lemonade reads")

        # Same request again: the prompt prefix is reused from the KV cache, same output.
        st, r2 = c.post("/v1/completions", req)
        check(r2["choices"][0]["text"] == text, "repeat request gives the same text")
        check(r2["timings"]["cache_n"] == meta["n_prompt"] - 1, "repeat request reuses the cached prefix (cache_n)")

        # Streaming gives the same text, then a final chunk with timings, then [DONE].
        events, done = c.stream("/v1/completions", dict(req, stream_options={"include_usage": True}))
        streamed = "".join(e["choices"][0]["text"] for e in events if e.get("choices"))
        check(streamed == text, "streamed completion equals non-streamed")
        finals = [e for e in events if e.get("choices") and e["choices"][0]["finish_reason"]]
        check(len(finals) == 1 and finals[0]["choices"][0]["finish_reason"] == "length" and "timings" in finals[0],
              "one final chunk with finish_reason and timings")
        check(events[-1].get("choices") == [] and events[-1]["usage"]["completion_tokens"] == req["max_tokens"],
              "include_usage adds a usage chunk")
        check(done, "stream ends with [DONE]")

        # Stop strings end generation before the match.
        st, r = c.post("/v1/completions", dict(req, stop=[" Rome"]))
        check(r["choices"][0]["text"] == meta["continuation"].split(" Rome")[0], "stop string truncates the output")
        check(r["choices"][0]["finish_reason"] == "stop", "finish_reason stop on a stop string")

        # Chat, thinking off: plain content, no reasoning.
        chat = {"messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}}
        st, r = c.post("/v1/chat/completions", chat)
        msg = r["choices"][0]["message"]
        check(st == 200 and msg["role"] == "assistant", "/v1/chat/completions 200")
        check("Paris" in msg["content"] and "<think>" not in msg["content"], "thinking off: answer in content: %r" % msg["content"])
        check("reasoning_content" not in msg, "thinking off: no reasoning_content")
        check(r["choices"][0]["finish_reason"] == "stop", "chat ends on the end-of-turn token")

        # Chat, thinking on: the <think> block goes to reasoning_content.
        think = dict(chat, max_tokens=512)
        del think["chat_template_kwargs"]
        st, r = c.post("/v1/chat/completions", think)
        msg = r["choices"][0]["message"]
        check(msg.get("reasoning_content", "") != "" and "<think>" not in msg["content"], "thinking on: reasoning split out")
        check("Paris" in msg["content"], "thinking on: answer in content: %r" % msg["content"][:80])
        events, done = c.stream("/v1/chat/completions", think)
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        check(deltas[0].get("role") == "assistant", "first chat chunk carries the role")
        check("".join(d.get("reasoning_content", "") for d in deltas) == msg["reasoning_content"],
              "streamed reasoning equals non-streamed")
        check("".join(d.get("content") or "" for d in deltas) == msg["content"], "streamed content equals non-streamed")
        check(done, "chat stream ends with [DONE]")

        # Errors.
        st, r = c.post("/v1/chat/completions", None, raw=b"{not json")
        check(st == 400 and r["error"]["type"] == "invalid_request_error", "bad JSON is 400")
        st, r = c.post("/v1/chat/completions", dict(chat, n=2))
        check(st == 400, "n=2 is 400")
        st, r = c.post("/v1/completions", {"prompt": "word " * 5000, "max_tokens": 1})
        check(st == 400 and r["error"]["type"] == "exceed_context_size_error", "over-long prompt is 400 exceed_context_size_error")
        st, r = c.get("/v1/nothing")
        check(st == 404, "unknown route is 404")
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
    if FAILURES:
        print("\nserver log:\n" + out)
        print("%d FAILED" % len(FAILURES))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
