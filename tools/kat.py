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
"""kat: known-answer + determinism + drift checks against a running OpenAI-compatible server.

  kat.py run    --url http://127.0.0.1:8000 [--repeat 3] [--needle-words 3600] [--out report.json]
  kat.py record --url ... --out golden.json          store answers + per-token logprobs as a golden
  kat.py check  --url ... --golden golden.json        run, then compare against the golden

Built for `1bit serve` (any device) and llama-server. Every case is greedy (temperature 0, fixed
seed, thinking off). What it catches, mapped to bugs this engine has already shipped and fixed:
  known answer      wrong output (#309 needle miss, #319 multi-ubatch prompts, #369 4.7K EOS)
  empty / NaN       format-500, empty replies, NaN logprobs (#308, #313, #315)
  determinism       same request, different text (#140 decode-split race, #370 intermittent); every
                    repeat sends cache_prompt false, so each one runs the same (batched prefill) path
  cache path        the same request again with cache_prompt true after one that filled the cache: it
                    re-evaluates only the tail through the decode path. Kernel rounding differs between
                    the paths, so a near-tie may flip: that is a WARN with the first diverging token
                    and the fresh run's top-2 logprob margin there. It FAILS when the margin is over
                    --cache-margin (not a near-tie) or the cached run misses an answer the fresh run got
  prefix reuse      a second request reusing a cached long prefix answers wrongly (#318): the cache
                    check of prefix-reuse fills the cache with the needle prompt (same prefix) first
  varied lengths    a run of different prompt lengths (the #315 pattern; GTT growth, #hrx-program-cache)
  drift vs golden   first diverging token + logprob deltas after a pin bump or kernel change
A case the golden itself answers wrongly (a small model missing a needle) is not failed on the
expected answer in check mode; it must still match the golden's text (drift).
Exit code: 0 all pass, 1 any failure, 2 server unreachable.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

FILLER = ("The archive keeps weather notes from many small towns. Each entry lists the date, the wind, "
          "the clouds and a short remark by the person on duty. ")


def needle_prompt(words, needle, question, depth=0.5):
    n = max(1, words // len(FILLER.split()))
    parts = [f"[{i}] {FILLER}" for i in range(n)]
    parts.insert(int(len(parts) * depth), needle + " ")
    return "".join(parts) + "\n\n" + question


def cases(needle_words):
    long_q = "What is the secret code mentioned in the notes above? Answer with the number only."
    needle = "Note from the duty officer: the secret code is 7341."
    color = "Note from the duty officer: the box in the cellar is painted green."
    base = needle_prompt(needle_words, needle + " " + color, "")
    out = [
        {"name": "capital", "prompt": "What is the capital of France? Answer in one word.", "expect": ["Paris"],
         "max_tokens": 16},
        {"name": "arithmetic", "prompt": "What is 17 * 23? Answer with the number only.", "expect": ["391"],
         "max_tokens": 16},
        {"name": "code", "prompt": "Write a Python function add(a, b) that returns a + b. Reply with code only.",
         "expect": ["def add"], "max_tokens": 64},
        {"name": "needle", "prompt": base + long_q, "expect": ["7341"], "max_tokens": 16},
        # same long prefix, different question: served from the prompt cache on llama-server
        {"name": "prefix-reuse", "prompt": base + "What color is the box in the cellar? Answer in one word.",
         "expect": ["green", "Green"], "max_tokens": 16, "cache_fill": base + long_q},
    ]
    for i, frac in enumerate((0.05, 0.2, 0.45, 0.7)):
        w = max(40, int(needle_words * frac))
        out.append({"name": f"varied-{w}w", "prompt": needle_prompt(w, needle, long_q, depth=0.3),
                    "expect": ["7341"], "max_tokens": 16})
    return out


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(a, case, prompt=None, cache=False):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt or case["prompt"]}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": case["max_tokens"],
            "cache_prompt": cache, "chat_template_kwargs": {"enable_thinking": False}}
    if not a.no_logprobs:
        body.update({"logprobs": True, "top_logprobs": 3})
    t = time.time()
    try:
        r = post(a.url.rstrip("/") + "/v1/chat/completions", body, a.timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", "secs": time.time() - t}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}", "secs": time.time() - t}
    ch = (r.get("choices") or [{}])[0]
    text = (ch.get("message") or {}).get("content") or ""
    toks = []
    for tk in ((ch.get("logprobs") or {}).get("content") or []):
        top = [x.get("logprob") for x in (tk.get("top_logprobs") or [])]
        toks.append({"t": tk.get("token"), "lp": tk.get("logprob"), "top": top})
    usage = r.get("usage") or {}
    return {"text": text, "tokens": toks, "prompt_tokens": usage.get("prompt_tokens"), "secs": time.time() - t}


def margin_at(run, i):
    """Top-1 minus top-2 logprob of the fresh run at token i (None if unknown)."""
    toks = run.get("tokens") or []
    if i is None or i >= len(toks):
        return None
    top = [x for x in (toks[i].get("top") or []) if not bad_number(x)]
    return top[0] - top[1] if len(top) >= 2 else None


def cache_check(a, case, fresh):
    """Fill the prompt cache, ask again with cache_prompt true, compare to the fresh run.
    Returns (warns, fails, detail)."""
    fill = ask(a, case, prompt=case.get("cache_fill"), cache=True)
    cached = ask(a, case, cache=True)
    if "error" in fill or "error" in cached:
        return [], [f"cache path: {(fill.get('error') or cached.get('error'))}"], None
    d = compare(fresh, cached)
    if d["first_divergence"] is None:
        return [], [], {"first_divergence": None}
    i = d["first_divergence"]
    m = margin_at(fresh, i)
    detail = {"first_divergence": i, "fresh_margin": m, "cached_text": cached["text"]}
    msg = (f"cache path: cached request diverges from the fresh one at token {i} "
           f"(fresh top-2 margin {'n/a' if m is None else f'{m:.3f}'}): {cached['text'][:60]!r}")
    fresh_ok = any(e in fresh["text"] for e in case["expect"])
    cached_ok = any(e in cached["text"] for e in case["expect"])
    if fresh_ok and not cached_ok:
        return [], [msg + " and misses the answer the fresh run gave"], detail
    if m is not None and m > a.cache_margin:
        return [], [msg + f": not a near-tie (> {a.cache_margin})"], detail
    return [msg], [], detail


def bad_number(x):
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def compare(gold, cur):
    """First diverging token index and logprob deltas over the common prefix."""
    gt, ct = gold.get("tokens") or [], cur.get("tokens") or []
    first = None
    deltas = []
    for i, (g, c) in enumerate(zip(gt, ct)):
        if g["t"] != c["t"]:
            first = i
            break
        if not bad_number(g["lp"]) and not bad_number(c["lp"]):
            deltas.append(abs(g["lp"] - c["lp"]))
    if first is None and len(gt) != len(ct):
        first = min(len(gt), len(ct))
    return {"first_divergence": first, "max_lp_delta": max(deltas) if deltas else 0.0,
            "mean_lp_delta": sum(deltas) / len(deltas) if deltas else 0.0, "same_text": gold["text"] == cur["text"]}


def run(a, golden=None):
    try:
        urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=10).read()
    except Exception as e:  # noqa: BLE001
        print(f"kat: {a.url} not reachable ({e})")
        sys.exit(2)
    report = {"url": a.url, "repeat": a.repeat, "cases": [], "pass": True, "time": time.strftime("%FT%T%z")}
    for case in cases(a.needle_words):
        runs = [ask(a, case) for _ in range(a.repeat)]
        fails, warns = [], []
        g = golden.get(case["name"]) if golden is not None else None
        # the release pin itself answers this case wrongly: hold it to the golden text instead
        lenient = g is not None and g.get("expect_ok") is False
        for i, r in enumerate(runs):
            if "error" in r:
                fails.append(f"run {i}: {r['error']}")
                continue
            if not r["text"].strip():
                fails.append(f"run {i}: empty reply")
            if any(bad_number(t["lp"]) for t in r["tokens"]):
                fails.append(f"run {i}: NaN/inf logprob")
            if not any(e in r["text"] for e in case["expect"]):
                (warns if lenient else fails).append(f"run {i}: expected {case['expect']} got {r['text'][:80]!r}"
                                                     + (" (so did the golden)" if lenient else ""))
        texts = {r.get("text") for r in runs if "error" not in r}
        if len(texts) > 1:
            fails.append(f"nondeterministic: {len(texts)} different texts over {a.repeat} identical requests")
        cache = None
        if not a.no_cache_check and "error" not in runs[0]:
            w, f, cache = cache_check(a, case, runs[0])
            warns += w
            fails += f
        entry = {"name": case["name"], "prompt_tokens": runs[0].get("prompt_tokens"),
                 "text": runs[0].get("text"), "tokens": runs[0].get("tokens"),
                 "expect_ok": "error" not in runs[0] and any(e in runs[0].get("text", "") for e in case["expect"]),
                 "secs": [round(r["secs"], 2) for r in runs], "fails": fails, "warns": warns, "cache_path": cache}
        if golden is not None:
            if g is None:
                entry["drift"] = "no golden entry"
            elif "error" not in runs[0]:
                d = compare(g, runs[0])
                entry["drift"] = d
                if d["first_divergence"] is not None and d["first_divergence"] < a.min_divergence:
                    fails.append(f"drift: diverges from golden at token {d['first_divergence']}")
                if d["max_lp_delta"] > a.max_lp_delta:
                    fails.append(f"drift: logprob delta {d['max_lp_delta']:.3f} > {a.max_lp_delta}")
        report["cases"].append(entry)
        report["pass"] &= not fails
        st = "FAIL" if fails else "warn" if warns else "ok  "
        extra = ""
        if isinstance(entry.get("drift"), dict):
            dd = entry["drift"]
            extra = f" drift: first_div={dd['first_divergence']} max_dlp={dd['max_lp_delta']:.4f}"
        print(f"{st} {case['name']:16} ptok={entry['prompt_tokens']} secs={entry['secs']} "
              f"said={str(entry['text'])[:40]!r}{extra}")
        for f in fails:
            print(f"     - {f}")
        for w in warns:
            print(f"     ~ WARN {w}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["run", "record", "check"])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="kat", help="model field sent (1bit serve ignores it)")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--needle-words", type=int, default=3600, help="~1.3 tokens per word; 3600 is ~4.7K tokens")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--no-logprobs", action="store_true", help="for routes that answer logprobs with an error")
    p.add_argument("--golden")
    p.add_argument("--min-divergence", type=int, default=8, help="check: fail if text diverges before this token")
    p.add_argument("--max-lp-delta", type=float, default=0.5, help="check: fail above this |logprob| delta")
    p.add_argument("--cache-margin", type=float, default=1.0,
                   help="cache path: a divergence where the fresh run's top-2 logprob margin is above this fails")
    p.add_argument("--no-cache-check", action="store_true", help="skip the fresh-vs-cached comparison")
    p.add_argument("--out")
    a = p.parse_args()
    golden = None
    if a.mode == "check":
        if not a.golden:
            sys.exit("kat check: --golden FILE")
        golden = {c["name"]: c for c in json.load(open(a.golden))["cases"]}
    rep = run(a, golden)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    print("PASS" if rep["pass"] else "FAIL")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
