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
"""gauntlet.py: Inkling-2-Mini rows in the shape of A. B. Neelicattu's 21-22 Nov 2026 report, against one OpenAI chat endpoint.

Rows (each --runs times, single stream, greedy):
  code       short code prompt, up to 512 tokens out          -> decode tok/s
  reasoning  short reasoning prompt, up to 512 tokens out     -> decode tok/s
  sustained  code prompt, 2048 tokens out                     -> decode tok/s
  needle16k  ~16,817-token haystack with COBALT-FALCON-7429, 256 out -> PP = prompt_tokens / TTFT, TG = decode tok/s,
             recall checked
  k391       "17 * 23" greedy known answer (checked)
Decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token), completion_tokens from the server's usage
(speculative chunks carry several tokens). Power: hwmon power1_average sampled at 4 Hz in a thread; avg W over the
decode window, J/tok = integral(P dt) over that window / (completion_tokens - 1).

Vendored from the cl-loomfun lane's box harness (/data/dev/cl-loomfun/gauntlet.py, 22 Nov 2026) for the Hyperloom
Inkling workload; two changes: every run keeps its full reply in `text` (the workload's greedy-text gate compares
replies, a 160-character head is not enough to say where two replies part), and `--power auto` finds the amdgpu hwmon
by its name, because the hwmon index moves between boots.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals and returns them "
            "sorted. Include a docstring and three doctest examples.",
    "reasoning": "A train leaves city A at 9:00 travelling at 80 km/h toward city B, 360 km away. A second train "
                 "leaves B at 10:00 travelling at 100 km/h toward A. At what time do they meet, and how far from A? "
                 "Reason step by step.",
    "sustained": "Write a complete, well-commented Python module implementing a small in-memory key-value store with "
                 "TTL expiry, an LRU eviction policy, snapshot/restore to JSON, and a command-line interface. Write "
                 "all of the code; do not stop early.",
    "k391": "What is 17 * 23? Answer with the number only.",
}
NEEDLE = "COBALT-FALCON-7429"
HAY = ("In the archive of the northern observatory, each logbook entry records the weather, the instrument "
       "calibration, the names of the observers on duty, and any unusual sightings over the eastern ridge. ")


class Power(threading.Thread):
    def __init__(self, path, hz=4.0):
        super().__init__(daemon=True)
        self.path, self.dt, self.s, self.stop = path, 1.0 / hz, [], False

    def run(self):
        while not self.stop:
            try:
                self.s.append((time.perf_counter(), int(open(self.path).read()) / 1e6))
            except Exception:
                pass
            time.sleep(self.dt)

    def window(self, a, b):
        pts = [(t, w) for t, w in self.s if a <= t <= b]
        if len(pts) < 2:
            return None, None
        e = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(pts, pts[1:]))
        return e / (pts[-1][0] - pts[0][0]), e


def chat(url, model, content, max_tokens, timeout, extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage, text, chunks = None, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                ev = json.loads(d)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                dl = ch.get("delta") or {}
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    return {"t0": t0, "first": first, "last": last, "usage": usage, "chunks": chunks, "text": "".join(text)}


def haystack(n_words):
    w = (HAY * (n_words // len(HAY.split()) + 2)).split()[:n_words]
    mid = len(w) // 2
    return " ".join(w[:mid]) + " The vault access code is %s. " % NEEDLE + " ".join(w[mid:]) + \
        "\n\nWhat is the vault access code mentioned in the text above? Answer with the code only."


def find_amdgpu_power(root="/sys/class/hwmon"):
    """The amdgpu hwmon's power1_average, found by name; a path that does not exist when nothing matches (the sampler
    then records no power, which the result shows as null rather than a number from the wrong sensor)."""
    try:
        for h in sorted(os.listdir(root)):
            try:
                if open(os.path.join(root, h, "name")).read().strip() == "amdgpu":
                    return os.path.join(root, h, "power1_average")
            except OSError:
                continue
    except OSError:
        pass
    return os.path.join(root, "amdgpu-not-found", "power1_average")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rows", default="k391,code,reasoning,sustained,needle16k")
    ap.add_argument("--needle-tokens", type=int, default=16817)
    ap.add_argument("--power", default="auto", help="hwmon power file, or auto = the amdgpu hwmon's power1_average")
    ap.add_argument("--extra", default="{}", help="JSON merged into each request body")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", required=True)
    ap.add_argument("--meta", default="{}")
    a = ap.parse_args()
    extra, meta = json.loads(a.extra), json.loads(a.meta)
    if a.power == "auto":
        a.power = find_amdgpu_power()
    model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=60))["data"][0]["id"]
    pw = Power(a.power)
    pw.start()
    label = ""
    try:
        label = open(os.path.join(os.path.dirname(a.power), "power1_label")).read().strip()
    except Exception:
        pass
    recs = []
    chat(a.url, model, "Say ok.", 8, a.timeout, extra)  # warm-up / weight load, not measured
    needle_words = None
    for row in a.rows.split(","):
        runs = []
        if row == "needle16k":
            if needle_words is None:  # calibrate to the target prompt length with a 1-token request
                n = int(a.needle_tokens * 0.72)
                for _ in range(6):
                    r = chat(a.url, model, haystack(n), 1, a.timeout, extra)
                    pt = (r["usage"] or {}).get("prompt_tokens")
                    if not pt or abs(pt - a.needle_tokens) <= 40:
                        break
                    n = int(n * a.needle_tokens / pt)
                needle_words = n
        for i in range(a.runs):
            if row == "needle16k":
                content, mt = "[run %d] " % i + haystack(needle_words), 256
            else:
                content = ("[run %d] " % i) + PROMPTS[row]
                mt = {"code": 512, "reasoning": 512, "sustained": 2048, "k391": 600}[row]
            r = chat(a.url, model, content, mt, a.timeout, extra)
            u = r["usage"] or {}
            n = u.get("completion_tokens") or r["chunks"]
            ttft = (r["first"] - r["t0"]) if r["first"] else None
            dec = (n - 1) / (r["last"] - r["first"]) if r["first"] and r["last"] and r["last"] > r["first"] and n > 1 else None
            w, e = pw.window(r["first"], r["last"]) if r["first"] else (None, None)
            wq, eq = pw.window(r["t0"], r["last"]) if r["last"] else (None, None)
            run = {"run": i, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": n, "chunks": r["chunks"],
                   "ttft_s": ttft, "decode_tok_s": dec, "avg_w": w,
                   "j_per_tok": (e / (n - 1)) if e and n > 1 else None,
                   "avg_w_request": wq, "j_per_tok_request": (eq / n) if eq and n else None,
                   "pp_tok_s": (u.get("prompt_tokens") / ttft) if row == "needle16k" and ttft and u.get("prompt_tokens") else None,
                   "usage": u, "text_head": r["text"][:160], "text": r["text"]}
            if row == "needle16k":
                run["recall_ok"] = NEEDLE in r["text"]
            if row == "k391":
                run["answer_ok"] = "391" in r["text"]
            runs.append(run)
            print("%s %s run %d: n=%s ttft=%.3fs dec=%s W=%s J/tok=%s %s" % (
                a.arm, row, i, n, ttft or -1, dec and round(dec, 2), w and round(w, 1),
                run["j_per_tok"] and round(run["j_per_tok"], 3),
                {k: run[k] for k in ("recall_ok", "answer_ok", "pp_tok_s") if k in run and run[k] is not None}), flush=True)

        def agg(k):
            v = [x[k] for x in runs if x.get(k) is not None]
            return {"min": round(min(v), 3), "median": round(statistics.median(v), 3), "max": round(max(v), 3), "n": len(v)} if v else None
        rec = {"arm": a.arm, "row": row, "model": model, "runs": runs, "decode_tok_s": agg("decode_tok_s"),
               "pp_tok_s": agg("pp_tok_s"), "ttft_s": agg("ttft_s"), "avg_w": agg("avg_w"), "j_per_tok": agg("j_per_tok"),
               "avg_w_request": agg("avg_w_request"), "j_per_tok_request": agg("j_per_tok_request"),
               "power_source": "%s (%s) @4 Hz" % (a.power, label), "request_extra": extra}
        rec.update(meta)
        recs.append(rec)
        json.dump(recs, open(a.json, "w"), indent=1)
    pw.stop = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
