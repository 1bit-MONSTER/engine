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
"""e2e_bench.py — what a user waits for: time to first token, then the whole answer.

    tools/e2e_bench.py --url http://127.0.0.1:8080 [--prompt-tokens 512,2048,8192,16384]
                       [--max-tokens 256] [--runs 3] [--text FILE] [--label NAME] [--jsonl OUT]

Decode speed alone (tg128) leaves out the prompt. A request with P prompt tokens and N output
tokens takes

    end-to-end = TTFT + (N - 1) / decode speed         TTFT ~ P / prefill speed

and what the user gets out of it is the effective speed N / end-to-end. With long prompts
(agents, coding, RAG) TTFT is most of the wait, so prefill speed sets the effective speed.
The same idea as Artificial Analysis's "end-to-end response time" and "total response time for
100 output tokens"; this measures it against a running server instead of computing it.

Each request goes to /v1/completions (raw prompt, no chat template) with streaming, greedy
sampling, ignore_eos (exactly N tokens) and cache_prompt off; the prompt starts with a fresh
random number so no prefix cache applies. The prompt is cut from TEXT (default: this
repository's docs) to exactly P tokens with the server's /tokenize and /detokenize. Times are
measured at the client; the server's own prompt/decode timings are reported next to them.
"""
import argparse
import glob
import json
import os
import random
import statistics
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def post(url, body, stream=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600) if stream else json.load(urllib.request.urlopen(req, timeout=600))


def corpus(path):
    if path:
        return open(path, encoding="utf-8", errors="replace").read()
    parts = [open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(HERE, "..", "docs", "*.md")))]
    return "\n\n".join(parts)


def prompt_of(url, text, n_tokens):
    toks = post(url + "/tokenize", {"content": text})["tokens"]
    while len(toks) < n_tokens + 16:
        toks = toks + toks
    head = f"[{random.randrange(10**12)}]\n"
    n_head = len(post(url + "/tokenize", {"content": head})["tokens"])
    body = post(url + "/detokenize", {"tokens": toks[: n_tokens - n_head]})["content"]
    p = head + body
    return p, len(post(url + "/tokenize", {"content": p})["tokens"])


def one(url, model, prompt, n_out):
    body = {"prompt": prompt, "max_tokens": n_out, "stream": True, "temperature": 0, "ignore_eos": True,
            "cache_prompt": False, "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    t0 = time.perf_counter()
    first = last = None
    chunks, timings, usage = 0, {}, {}
    with post(url + "/v1/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            timings = ev.get("timings", timings)
            usage = ev.get("usage") or usage
            if any(c.get("text") for c in ev.get("choices", [])):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    n = usage.get("completion_tokens") or timings.get("predicted_n") or chunks
    e2e = last - t0
    return {"ttft_s": first - t0, "e2e_s": e2e, "n_out": n,
            "decode_tps": (n - 1) / (last - first) if last > first else 0.0,
            "effective_tps": n / e2e,
            "server_prompt_tps": timings.get("prompt_per_second"),
            "server_decode_tps": timings.get("predicted_per_second")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", help="model id to send (1bit serve with several models)")
    ap.add_argument("--prompt-tokens", default="512,2048,8192,16384")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--text", help="text to cut prompts from (default: docs/*.md)")
    ap.add_argument("--label", default="")
    ap.add_argument("--jsonl", help="append one line per prompt length")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    text = corpus(a.text)
    one(url, a.model, prompt_of(url, text, 64)[0], 8)  # warm-up: kernels, allocations
    print(f"{'label':<16} {'prompt':>7} {'out':>4} {'TTFT s':>8} {'decode t/s':>10} {'E2E s':>7} "
          f"{'effective t/s':>13} {'srv prefill t/s':>15}")
    for p_tok in (int(x) for x in a.prompt_tokens.split(",")):
        rs = []
        for _ in range(a.runs):
            prompt, n_p = prompt_of(url, text, p_tok)
            r = one(url, a.model, prompt, a.max_tokens)
            r["n_prompt"] = n_p
            rs.append(r)
        m = {k: statistics.median(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], (int, float))}
        srv = m.get("server_prompt_tps")
        print(f"{a.label:<16} {int(m['n_prompt']):>7} {int(m['n_out']):>4} {m['ttft_s']:>8.2f} {m['decode_tps']:>10.1f} "
              f"{m['e2e_s']:>7.2f} {m['effective_tps']:>13.1f} {srv if srv is None else round(srv):>15}", flush=True)
        if a.jsonl:
            with open(a.jsonl, "a") as f:
                f.write(json.dumps({"label": a.label, "runs": rs, "median": m}) + "\n")


if __name__ == "__main__":
    main()
