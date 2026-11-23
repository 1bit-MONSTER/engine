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
"""hl_score.py: score and gate ONE Inkling-2-Mini Loom measurement for Hyperloom (InferenceSuite custom contract).

Reads what box_inkling.sh left (gauntlet.json, provenance.json, status) and writes the InferenceX-shaped report
Hyperloom 1.2.0 reads from $RESULT_DIR/inferencex_result.json: `output_throughput` (maximised) and `quality_gate`
(fail-closed: a report without `passed: true` is a rejected candidate).

Score: geometric mean of five medians -- code decode, reasoning decode, sustained-2048 decode, 16.8k-needle prefill
and 16.8k-needle decode (tok/s) -- the gauntlet's rows (~/briefs/INKLING-GAUNTLET.md), so a lever has to win on balance
and cannot buy decode with prefill.

Gate, three halves, all must pass:
  needle  every needle16k run recalls COBALT-FALCON-7429
  k391    every k391 run answers 391
  text    greedy replies agree with the reference's: the fraction of (row, run) replies whose first TEXT_CHARS
          characters are identical to the reference's must be >= TEXT_AGREE. Not byte-identity of whole replies:
          DSpark verification runs a different kernel shape from plain decode, so greedy text from d=3 and d=0 parts
          after a while on a few prompts (measured on this card, 22 Nov: 13 of 15 replies identical over the first 160
          characters), and an identity bar would reject speculative decoding itself.
There is NO logit (KLD) half: loom-serve 0.3.0 does not export logits for this card ("llm.logits": false). Every
report says so in quality_gate.kld. The reference is the first measurement of a state directory (Hyperloom's
baseline), keyed by card revision + GGUF hashes + loom-serve hash, so a reference is never compared across builds.

usage: hl_score.py --box DIR --state DIR --out FILE [--levers JSON]
       hl_score.py --self-test
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile

TEXT_CHARS = 160
TEXT_AGREE = 0.8
METRICS = (("code", "decode_tok_s"), ("reasoning", "decode_tok_s"), ("sustained", "decode_tok_s"),
           ("needle16k", "pp_tok_s"), ("needle16k", "decode_tok_s"))
TEXT_ROWS = ("k391", "code", "reasoning", "sustained", "needle16k")
UNIT = "tok/s (geomean of code, reasoning, sustained-2048 decode, 16.8k-needle PP and TG medians)"


def median_of(rows, row, key):
    for r in rows:
        if r.get("row") == row:
            v = [x[key] for x in r.get("runs", []) if isinstance(x.get(key), (int, float)) and x[key] > 0]
            return statistics.median(v) if v else None
    return None


def score(rows):
    vals = {"%s.%s" % m: median_of(rows, *m) for m in METRICS}
    if any(v is None for v in vals.values()):
        return 0.0, vals
    return math.exp(sum(math.log(v) for v in vals.values()) / len(vals)), vals


def texts(rows):
    out = {}
    for r in rows:
        if r.get("row") in TEXT_ROWS:
            for x in r.get("runs", []):
                if isinstance(x.get("text"), str):
                    out["%s/%d" % (r["row"], x.get("run", 0))] = x["text"]
    return out


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def gate(rows, reference, chars=TEXT_CHARS, agree=TEXT_AGREE):
    halves = {}
    byrow = {r.get("row"): r for r in rows}
    needle = [x.get("recall_ok") for x in byrow.get("needle16k", {}).get("runs", [])]
    halves["needle"] = {"passed": bool(needle) and all(v is True for v in needle), "runs": needle}
    k = [x.get("answer_ok") for x in byrow.get("k391", {}).get("runs", [])]
    halves["k391"] = {"passed": bool(k) and all(v is True for v in k), "runs": k}
    mine = texts(rows)
    if reference is None:
        halves["text"] = {"passed": bool(mine), "reference": "established by this measurement", "replies": len(mine)}
    else:
        ref = reference["texts"]
        keys = sorted(ref)
        same = [k_ for k_ in keys if k_ in mine and mine[k_][:chars] == ref[k_][:chars]]
        prefixes = {k_: common_prefix(mine.get(k_, ""), ref[k_]) for k_ in keys}
        frac = len(same) / len(keys) if keys else 0.0
        halves["text"] = {"passed": bool(keys) and frac >= agree, "agree": round(frac, 4), "bar": agree,
                          "chars": chars, "identical_heads": len(same), "replies": len(keys),
                          "missing": [k_ for k_ in keys if k_ not in mine],
                          "common_prefix_chars": prefixes, "reference_id": reference.get("id")}
    return {"passed": all(h["passed"] for h in halves.values()), "halves": halves,
            "kld": None, "kld_note": "not measured: loom-serve 0.3.0 exports no logits for this card"}


def reference_key(prov):
    blob = json.dumps({"card_revision": prov.get("card_revision"), "gguf": prov.get("gguf_sha256"),
                       "loom_serve": prov.get("loom_serve_sha256")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run(box, state, out, levers):
    box = pathlib.Path(box)
    status = (box / "status").read_text().strip() if (box / "status").exists() else "no status file"
    rep = {"framework": "custom", "workload_kind": "scriptable", "throughput_unit": UNIT, "output_throughput": 0.0,
           "bench_summary": {"status": status, "levers_requested": levers}}
    try:
        rows = json.loads((box / "gauntlet.json").read_text())
        prov = json.loads((box / "provenance.json").read_text())
    except (OSError, ValueError) as e:
        rep["quality_gate"] = {"passed": False, "reason": "unmeasured: %s (%s)" % (status, e)}
        return rep
    rep["bench_summary"]["provenance"] = prov
    s, vals = score(rows)
    rep["output_throughput"] = round(s, 3)
    rep["bench_summary"]["metrics"] = vals
    for name in ("avg_w", "j_per_tok"):
        rep["bench_summary"][name] = {r["row"]: (r.get(name) or {}).get("median") for r in rows}
    key = reference_key(prov)
    sdir = pathlib.Path(state) / key
    ref_file = sdir / "reference.json"
    reference = json.loads(ref_file.read_text()) if ref_file.exists() else None
    g = gate(rows, reference)
    if status != "ok":
        g["passed"] = False
        g["reason"] = "box status: " + status
    if s <= 0:
        g["passed"] = False
        g["reason"] = "a scored row is missing: %s" % [k for k, v in vals.items() if v is None]
    if reference is None and g["passed"]:
        sdir.mkdir(parents=True, exist_ok=True)
        ref = {"id": key + "-" + hashlib.sha256(json.dumps(texts(rows), sort_keys=True).encode()).hexdigest()[:8],
               "texts": texts(rows), "levers": prov.get("levers"), "score": rep["output_throughput"]}
        tmp = ref_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(ref, indent=1))
        os.replace(tmp, ref_file)
        g["halves"]["text"]["reference_id"] = ref["id"]
    g["reference_dir"] = str(sdir)
    rep["quality_gate"] = g
    return rep


def self_test():
    def row(name, runs):
        return {"row": name, "runs": runs}

    def fake(dec=50.0, text="The user wants", recall=True, ans=True):
        return [row("k391", [{"run": i, "decode_tok_s": dec, "text": text + " 391", "answer_ok": ans} for i in range(3)]),
                row("code", [{"run": i, "decode_tok_s": dec, "text": text + " code %d" % i} for i in range(3)]),
                row("reasoning", [{"run": i, "decode_tok_s": dec, "text": text + " reason"} for i in range(3)]),
                row("sustained", [{"run": i, "decode_tok_s": dec, "text": text + " module"} for i in range(3)]),
                row("needle16k", [{"run": i, "decode_tok_s": dec, "pp_tok_s": 2000.0, "text": "COBALT-FALCON-7429",
                                   "recall_ok": recall} for i in range(3)])]
    prov = {"card_revision": "a8354e77", "gguf_sha256": {"m.gguf": "e62d"}, "loom_serve_sha256": "1db8", "levers": {}}
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        def measure(rows, status="ok", p=prov):
            b = td / ("box%d" % len(list(td.iterdir())))
            b.mkdir()
            (b / "gauntlet.json").write_text(json.dumps(rows))
            (b / "provenance.json").write_text(json.dumps(p))
            (b / "status").write_text(status)
            return run(b, td / "state", None, {})

        base = measure(fake())
        check("baseline passes and establishes the reference", base["quality_gate"]["passed"]
              and base["quality_gate"]["halves"]["text"]["reference"].startswith("established"))
        check("score = geomean of the five medians", abs(base["output_throughput"]
                                                         - math.exp((4 * math.log(50) + math.log(2000)) / 5)) < 1e-2)
        same = measure(fake(dec=60.0))
        check("identical text passes against the reference", same["quality_gate"]["passed"]
              and same["quality_gate"]["halves"]["text"]["agree"] == 1.0)
        check("faster scores higher", same["output_throughput"] > base["output_throughput"])
        moved = measure(fake(text="Different opening"))
        check("all replies diverging in the head fails the text half", not moved["quality_gate"]["passed"]
              and not moved["quality_gate"]["halves"]["text"]["passed"])
        rows = fake()
        rows[1]["runs"][0]["text"] = "The user wantz a different one"   # 1 of 15 replies differs: 14/15 >= 0.8
        check("one diverging reply in fifteen stays inside the bar", measure(rows)["quality_gate"]["passed"])
        check("a lost needle fails", not measure(fake(recall=False))["quality_gate"]["passed"])
        check("a wrong 17*23 fails", not measure(fake(ans=False))["quality_gate"]["passed"])
        rows = fake()
        rows[2]["runs"] = []
        r = measure(rows)
        check("a missing scored row scores 0 and fails", r["output_throughput"] == 0.0 and not r["quality_gate"]["passed"])
        check("a box failure fails even with numbers", not measure(fake(), status="failed: gauntlet rc=1")["quality_gate"]["passed"])
        empty = td / "empty"
        empty.mkdir()
        r = run(empty, td / "state", None, {})
        check("no gauntlet output fails closed", r["quality_gate"]["passed"] is False and r["output_throughput"] == 0.0)
        other = dict(prov, loom_serve_sha256="ffff")
        r = measure(fake(text="Different opening"), p=other)
        check("a new loom-serve build gets its own reference, never the old one",
              r["quality_gate"]["halves"]["text"].get("reference", "").startswith("established"))
        check("every report says the KLD half is not measured", base["quality_gate"]["kld"] is None)
    for f in fails:
        print("FAIL:", f)
    print("hl_score self-test: %d checks failed" % len(fails) if fails else "hl_score self-test: ok")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box")
    ap.add_argument("--state")
    ap.add_argument("--out")
    ap.add_argument("--levers", default="{}")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.box and a.state and a.out):
        ap.error("--box, --state and --out are required")
    rep = run(a.box, a.state, a.out, json.loads(a.levers))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, a.out)
    g = rep["quality_gate"]
    print("[hl_score] output_throughput=%s gate=%s %s" % (rep["output_throughput"], "PASS" if g["passed"] else "FAIL",
                                                          json.dumps({k: v["passed"] for k, v in g.get("halves", {}).items()})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
