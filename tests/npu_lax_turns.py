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
"""A three-turn chat against `1bit serve`, as an OpenAI client sends it (docs/npu-lax.md,
"Chat follow-ups"): every request carries the whole history, and each earlier answer goes
back without its think block, as Qwen3.6's chat template renders it. Per turn it prints
the prompt tokens, how many the server reused (usage.prompt_tokens_details.cached_tokens)
and from where, the tokens fed again, and the time to the first streamed token, measured
here from sending the request.

usage: tests/npu_lax_turns.py <base url> [--max-tokens 64] [--thinking] [--no-stream] [--min-reuse]
  --no-stream  plain requests (for servers without stream_options.include_usage): no time to
               first token, only the server's prompt time
  --min-reuse  exit 1 unless every follow-up reuses the history before its last answer: all
               of the previous prompt but its last assistant header (at most 8 tokens)
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

CONTEXT = (
    "Notes for a trip. The Loire valley lies south-west of Paris. Its chateaux were built from the "
    "fifteenth to the eighteenth century, when the French court moved between Amboise, Blois and "
    "Chambord. Chambord has 426 rooms and a double-helix staircase often credited to Leonardo da Vinci, "
    "who spent his last years at Clos Luce in Amboise. Chenonceau spans the river Cher on a gallery of "
    "arches; during the First World War it served as a hospital. Trains from Paris Montparnasse reach "
    "Tours in about an hour, and most chateaux are within forty kilometres of it. Bicycles can be hired "
    "in Tours, Amboise and Blois, and the Loire a Velo route follows the river for 900 km."
)
TURNS = [
    CONTEXT + "\n\nWhich chateau spans a river, and what was it used for in the First World War?",
    "How long is the train from Paris, and from which station?",
    "Suggest a two-day cycling plan using these notes. Keep it short.",
]


def post_stream(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft, text, usage, timings = None, "", None, {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if "error" in c:
                raise RuntimeError(c["error"])
            for ch in c.get("choices", []):
                d = ch.get("delta", {}).get("content", "")
                if d and ttft is None:
                    ttft = time.monotonic() - t0
                text += d
            if c.get("usage"):
                usage, timings = c["usage"], c.get("timings", {})
    return text, usage, timings, ttft, time.monotonic() - t0


def post_plain(url, body):
    body = dict(body, stream=False)
    body.pop("stream_options")
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        c = json.loads(r.read())
    return c["choices"][0]["message"]["content"], c["usage"], c.get("timings", {}), None, time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--min-reuse", action="store_true")
    a = ap.parse_args()
    messages, ok = [], True
    for i, q in enumerate(TURNS):
        messages.append({"role": "user", "content": q})
        body = {"messages": messages, "max_tokens": a.max_tokens, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": a.thinking}}
        try:
            text, u, tm, ttft, total = (post_plain if a.no_stream else post_stream)(a.url, body)
        except urllib.error.HTTPError as e:
            print(f"turn {i + 1}: HTTP {e.code}: {e.read().decode()[:300]}")
            return 1
        if u is None:
            print(f"turn {i + 1}: no usage chunk (stream_options.include_usage unsupported?)")
            return 1
        n, cached = u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"]
        print(f"turn {i + 1}: {n} prompt tokens, {cached} reused ({tm.get('cache', 'none')}), {n - cached} fed; "
              f"first token after {'-' if ttft is None else round(ttft, 2)} s (prompt {tm.get('prompt_ms', 0) / 1000:.2f} s "
              f"on the server); {u['completion_tokens']} generated, {total:.1f} s in all")
        answer = text.split("</think>")[-1].strip()  # the template drops the think block
        print(f"         {answer[:100]!r}")
        if a.min_reuse and i > 0 and cached < prev_n - 8:
            ok = False
            print(f"         FAIL: reused {cached}, less than the previous prompt ({prev_n}) but its assistant header")
        prev_n = n
        messages.append({"role": "assistant", "content": answer})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
