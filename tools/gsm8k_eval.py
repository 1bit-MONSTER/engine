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
"""gsm8k_eval.py — GSM8K accuracy of a served route, in the shape a leaderboard row needs.

    tools/gsm8k_eval.py --url http://127.0.0.1:18099 --n 40 --out gsm8k-27b-npu.json

Nothing about this tool is model-specific: it posts each GSM8K question to an
OpenAI-compatible `/v1/chat/completions` with greedy decoding, reads the number the model
wrote after `####` (the dataset's own answer format), and compares it to the gold answer.
The record it writes is what `bench/npu-leaderboard/README.md` asks for: the instrument
(exact prompt, decoding settings, token budget), the repeat count, and a per-question
transcript so a claimed accuracy can be re-graded offline without the device.

Why the transcript and not just a percentage: a reasoning model can be **truncated** by the
token budget, and a truncated answer is wrong for a reason that is not the model's maths.
Truncated completions are counted separately (`finish_reason == "length"`) and reported next
to the accuracy, so "78 % with 3 of 40 answers cut off" cannot be read as "78 %".

The dataset is fetched once from the HF datasets-server and cached as JSON; `--dataset-json`
reuses a cache and keeps a run offline. `--selftest` checks the answer extractor and the
scorer against known strings and needs no network and no device.

Exit codes: 0 ran, 1 the route answered with an error on some question (accuracy still
printed, the failing count named), 2 unusable arguments or an unreadable dataset.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the endpoint's maximum; a 40-question run is one page

# The prompt is part of the instrument: it is fixed here, printed into the record, and never
# adapted per model. A model that needs a different prompt needs a different record.
PROMPT = ("Solve this grade-school maths problem. Reason step by step, then write your final "
          "answer on the last line as\n#### <number>\n\nProblem: {q}")
ANSWER_RE = re.compile(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")


def extract(text):
    """The answer the model wrote after its last `####`, else the last number in the text."""
    hits = ANSWER_RE.findall(text or "")
    cand = hits[-1] if hits else None
    if cand is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")
        cand = nums[-1] if nums else None
    return norm(cand)


def norm(s):
    """A GSM8K answer as a comparable number: drop $ and thousands separators, keep decimals."""
    if s is None:
        return None
    s = str(s).strip().replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else repr(f)


def gold_of(answer_field):
    """The dataset's `#### N` answer, through the same normaliser as the model's."""
    m = ANSWER_RE.findall(answer_field or "")
    return norm(m[-1]) if m else norm((answer_field or "").split()[-1] if answer_field else None)


def fetch_rows(n, cache):
    """The first n questions, from the cache when it already holds them, else fetched.

    A cache shorter than n is topped up rather than silently shrinking the run: an accuracy
    reported over fewer questions than asked for is a different measurement.
    """
    rows = []
    if os.path.exists(cache):
        with open(cache) as f:
            rows = json.load(f)
        if len(rows) >= n:
            return rows[:n]
    while len(rows) < n:
        q = urllib.parse.urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                    "offset": len(rows), "length": min(PAGE, n - len(rows))})
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=60) as r:
                page = json.load(r)
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"gsm8k_eval: cannot read {DATASET} from the datasets-server: {e}")
        got = [x["row"] for x in page.get("rows", [])]
        if not got:
            break
        rows += got
    if not rows:
        raise SystemExit(f"gsm8k_eval: {DATASET}/{SPLIT} came back empty")
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(rows, f)
    return rows[:n]

def ask(url, prompt, max_tokens, timeout):
    """One greedy completion. Returns (text, finish_reason, completion_tokens, wall_s)."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
                       "seed": 1234, "stream": False}).encode()
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, f"error: {e}", 0, time.time() - t0
    wall = time.time() - t0
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return text, ch.get("finish_reason"), int((d.get("usage") or {}).get("completion_tokens") or 0), wall


def selftest():
    cases = [("blah\n#### 18", "18"), ("#### $1,024", "1024"), ("#### -3.5", "-3.5"),
             ("no hash but the total is 42 apples", "42"), ("#### 7\n#### 9", "9"), ("", None)]
    for text, want in cases:
        got = extract(text)
        assert got == want, f"{text!r}: extracted {got!r}, want {want!r}"
    assert gold_of("She sells 9 * 2 = 18.\n#### 18") == "18"
    assert norm("1,024") == "1024" and norm("72.0") == "72"
    # A truncated answer that happens to contain a number must still be scored, but the caller
    # reports it separately: the point is that scoring never silently treats it as a clean miss.
    assert extract("so the answer is 12 but I was cut off mid") == "12"
    print("gsm8k_eval selftest: ok (extractor, gold normaliser, truncation still scored)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", help="OpenAI-compatible route, e.g. http://127.0.0.1:18099")
    p.add_argument("--n", type=int, default=40, help="questions (the dataset order, not sampled)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip this many questions first, so one run can be split over several "
                        "device tickets and still cover a contiguous block")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=600, help="per-request seconds")
    p.add_argument("--dataset-json", default=os.path.expanduser("~/.cache/1bit/gsm8k-test.json"))
    p.add_argument("--out", help="write the transcript + summary here (JSON)")
    p.add_argument("--label", default="", help="free text for the record, e.g. the rung name")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.url:
        raise SystemExit("gsm8k_eval: --url is required (or --selftest)")
    if a.n < 1:
        raise SystemExit("gsm8k_eval: --n must be positive")
    if a.offset < 0:
        raise SystemExit("gsm8k_eval: --offset cannot be negative")

    rows = fetch_rows(a.offset + a.n, a.dataset_json)[a.offset:]
    res, errs = [], 0
    t_start = time.time()
    for i, r in enumerate(rows):
        q, gold = r["question"], gold_of(r["answer"])
        text, fin, ctok, wall = ask(a.url, PROMPT.format(q=q), a.max_tokens, a.timeout)
        if text is None:
            errs += 1
            res.append(dict(i=i, gold=gold, pred=None, correct=False, finish_reason=fin,
                            completion_tokens=0, wall_s=round(wall, 3), error=True))
            print(f"[{i + 1}/{len(rows)}] ERROR {fin}", flush=True)
            continue
        pred = extract(text)
        ok = pred is not None and pred == gold
        res.append(dict(i=i, gold=gold, pred=pred, correct=ok, finish_reason=fin,
                        completion_tokens=ctok, wall_s=round(wall, 3)))
        print(f"[{i + 1}/{len(rows)}] {'ok ' if ok else 'MISS'} gold {gold} pred {pred} "
              f"({fin}, {ctok} tok, {wall:.1f} s)", flush=True)

    n = len(res)
    ok = sum(1 for r in res if r["correct"])
    trunc = sum(1 for r in res if r["finish_reason"] == "length")
    toks = sum(r["completion_tokens"] for r in res)
    wall = time.time() - t_start
    summary = dict(label=a.label, url=a.url, n=n, correct=ok,
                   accuracy_pct=round(100.0 * ok / n, 2) if n else 0.0,
                   truncated=trunc, errors=errs, completion_tokens=toks,
                   wall_s=round(wall, 1),
                   s_per_token=round(wall / toks, 4) if toks else None,
                   prompt=PROMPT, max_tokens=a.max_tokens,
                   decoding="temperature 0, top_k 1, seed 1234, stream=false",
                   dataset=f"{DATASET}/{CONFIG}/{SPLIT}, questions {a.offset}-{a.offset + n - 1} "
                           "in dataset order")
    print(json.dumps(summary, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(summary=summary, questions=res), f, indent=2)
            f.write("\n")
        print(f"gsm8k_eval: wrote {a.out}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
