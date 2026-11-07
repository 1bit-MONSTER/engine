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
"""Pin the GSM8K scorer's contract (tools/gsm8k_eval.py, ctest gsm8k_eval).

An accuracy figure is the easiest number in this repository to produce wrongly and the hardest
to notice: a scorer that reads the wrong number out of a completion, or that counts a completion
cut off by the token budget as a clean miss, still prints a plausible percentage. This test
drives the extractor, the gold normaliser and the whole run loop against canned completions
(monkeypatched `ask`, so there is no route and no socket), and it re-derives the summary from the
written transcript so a summary that disagrees with its own per-question rows cannot pass.

No network and no device: the dataset is a temp cache written by the test, and `ask` never runs.
"""
import io
import json
import os
import sys
import tempfile
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gsm8k_eval as g  # noqa: E402

failures = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)


# 1. The answer extractor: the last `####` wins, $ and thousands separators are dropped, and a
#    completion with no `####` still yields its last number rather than counting as unparsed.
check("extract: the dataset's own format", g.extract("so\n#### 18") == "18")
check("extract: the LAST #### wins over an earlier one", g.extract("#### 7\n#### 9") == "9")
check("extract: $ and thousands separators are normalised", g.extract("#### $1,024") == "1024")
check("extract: a negative decimal survives", g.extract("#### -3.5") == "-3.5")
check("extract: no #### falls back to the last number", g.extract("that makes 42 apples") == "42")
check("extract: an empty completion has no answer", g.extract("") is None)
check("extract: a truncated answer is still scored",
      g.extract("the answer is 12 but I was cut off") == "12")
check("gold: the dataset field parses through the same normaliser",
      g.gold_of("9 * 2 = 18\n#### 18") == "18")
check("gold: a trailing 18.0 is the same answer as 18", g.norm("18.0") == g.norm("18") == "18")

# 2. A whole run against canned completions: 4 questions, one right, one wrong, one truncated
#    (wrong only because it was cut off), one route error. The summary must separate all four.
ROWS = [{"question": "q1", "answer": "#### 18"},
        {"question": "q2", "answer": "#### 3"},
        {"question": "q3", "answer": "#### 70"},
        {"question": "q4", "answer": "#### 5"}]
CANNED = {
    "q1": ("work work\n#### 18", "stop", 40),
    "q2": ("work work\n#### 4", "stop", 40),          # genuinely wrong
    "q3": ("so far 69 and then I ran out of", "length", 512),  # truncated, scored as a miss
    "q4": None,                                        # the route errored
}
calls = []


def fake_ask(url, prompt, max_tokens, timeout):
    q = prompt.split("Problem: ")[-1]
    calls.append((url, q, max_tokens))
    got = CANNED[q]
    if got is None:
        return None, "error: boom", 0, 0.5
    text, fin, tok = got
    return text, fin, tok, 1.0


g.fetch_rows = lambda n, cache: ROWS[:n]
g.ask = fake_ask
with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "run.json")
    argv = ["gsm8k_eval.py", "--url", "http://route.invalid:18099", "--n", "4",
            "--dataset-json", os.path.join(td, "cache.json"), "--out", out]
    old = sys.argv
    sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = g.main()
    finally:
        sys.argv = old
    d = json.load(open(out))
    s, qs = d["summary"], d["questions"]

check("run: the erroring question makes the exit code 1", rc == 1)
check("run: every question was asked once, in order", [c[1] for c in calls] == ["q1", "q2", "q3", "q4"])
check("run: the prompt carries the question and the #### instruction",
      "#### <number>" in g.PROMPT and "Problem: q1" in g.PROMPT.format(q="q1"))
check("run: one correct answer out of four", s["correct"] == 1 and s["n"] == 4)
check("run: accuracy is reported over all four, not over the answered three",
      s["accuracy_pct"] == 25.0)
check("run: the truncated completion is counted separately", s["truncated"] == 1)
check("run: the route error is counted separately", s["errors"] == 1)
check("run: the transcript keeps gold and prediction per question",
      [(q["gold"], q["pred"], q["correct"], q["finish_reason"]) for q in qs] ==
      [("18", "18", True, "stop"), ("3", "4", False, "stop"),
       ("70", "69", False, "length"), ("5", None, False, "error: boom")])
check("run: the summary is re-derivable from its own transcript",
      sum(q["correct"] for q in qs) == s["correct"]
      and sum(q["completion_tokens"] for q in qs) == s["completion_tokens"]
      and round(100.0 * sum(q["correct"] for q in qs) / len(qs), 2) == s["accuracy_pct"])
check("run: the instrument is in the record, not just in the code",
      s["prompt"] == g.PROMPT and s["max_tokens"] == 512
      and "temperature 0" in s["decoding"] and "test" in s["dataset"])

# 3. The cache rule: a cache that already holds n questions is used without any fetch, and a
#    shorter one is topped up rather than silently shrinking the run.
with tempfile.TemporaryDirectory() as td:
    cache = os.path.join(td, "c.json")
    json.dump(ROWS, open(cache, "w"))
    got = g.fetch_rows(2, cache)
check("cache: a long-enough cache answers without the network", len(got) == 2)

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
