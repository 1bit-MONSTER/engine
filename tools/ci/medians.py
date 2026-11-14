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
"""medians.py -- report a median AND the spread, and refuse to call a difference real when it is not.

Engine#679 measured four identical decode runs on an idle box at 18,598 tokens and got
7.94 / 11.78 / 10.39 / 12.57 tok/s: a 1.58x spread on the SAME command. Every figure this lane
published before that was quoted to two or three decimals as if the spread did not exist, and
item 3a's "2.0x slower decode" turned out to be that spread showing up in the first rep.

So: report median, min, max and the ratio, and answer the only question a reader actually has --
are the two arms' observed ranges DISJOINT, or is the difference inside the noise? Disjoint ranges,
not disjoint medians: if the loser's maximum reaches into the winner's range then the loser produced
that value on its own.

    medians.py 7.94 11.78 10.39 12.57                  # one arm
    medians.py --a 7.94 11.78 10.39 12.57 --b 9.1 9.3 # two arms, verdict
    medians.py --a ... --b ... --lower                 # B is better when lower (ms/token)
    medians.py --self-test

A verdict of NOISE is not a failure of the measurement; it is the measurement telling you to
take more samples or to stop quoting the difference.
"""
import statistics
import json
import os
import sys


def spread(values):
    """(median, min, max, max/min) for a non-empty list of positive floats."""
    if not values:
        raise ValueError("no values")
    lo, hi = min(values), max(values)
    if lo <= 0:
        raise ValueError("values must be positive to express a spread as a ratio")
    return statistics.median(values), lo, hi, hi / lo


def verdict(a, b):
    """'A' / 'B' when one median is outside the other's observed range, else 'NOISE'.

    Disjoint ranges are the only evidence a handful of samples can give: if A's median sits
    inside B's min..max, then B produced that number by itself and the difference needs more
    runs before it means anything.
    """
    # A single sample has no range, so it can never be shown to be OUTSIDE the other arm's range:
    # [5.0] against [6.0] would read as a win on one run. It may still LOSE -- a point below the
    # other arm's min is below every one of its observations, which needs no spread to see.
    # (engine#2369 item 1: the previous version let a one-sample arm win.)
    _, alo, ahi, _ = spread(a)
    _, blo, bhi, _ = spread(b)
    # Disjoint ranges, not disjoint medians: if the loser's MAX reaches into the winner's
    # range, the loser produced that value on its own and the difference is not yet evidence.
    # (The first version compared medians and passed [5.0] against [5.0, 6.0] as "B" -- B had
    #  produced 5.0 itself. The self-test below is the arm that caught it.)
    if alo > bhi and len(a) >= 2:
        return "A"
    if blo > ahi and len(b) >= 2:
        return "B"
    return "NOISE"


def direction_word(winner, lower_is_better):
    """Which arm is better, spelled out. "A" alone is ambiguous: higher is better for tok/s and worse
    for ms/token, and a caller that reads the letter without the unit draws the wrong conclusion."""
    if winner == "NOISE":
        return "neither -- the ranges overlap"
    if lower_is_better:
        return f"{winner} (lower, and lower is better here)"
    return f"{winner} (higher, and higher is better here)"


def self_test():
    fails = 0

    def check(ok, what):
        nonlocal fails
        if not ok:
            fails += 1
            print(f"FAIL {what}")

    # the real data that produced this file: four identical runs, 1.58x apart
    med, lo, hi, ratio = spread([7.94, 11.78, 10.39, 12.57])
    check(abs(med - (10.39 + 11.78) / 2) < 1e-9, "median of four is the mean of the middle two")
    check(lo == 7.94 and hi == 12.57, "min and max are the observed ones")
    check(abs(ratio - 1.583) < 0.01, "the 1.58x spread of identical runs is reported as such")

    # two arms whose ranges overlap cannot be separated by these samples
    check(verdict([10.0, 11.0, 12.0], [10.5, 11.5]) == "NOISE", "overlapping ranges are noise")
    # an arm whose whole range sits above the other's is a real difference
    check(verdict([20.0, 21.0], [10.0, 11.0]) == "A", "disjoint above is A")
    check(verdict([10.0, 11.0], [20.0, 21.0]) == "B", "disjoint below is B")
    # touching ranges are not disjoint: B's max equals A's min means B can reach it
    check(verdict([20.0, 20.0], [19.0, 20.0]) == "NOISE", "a shared endpoint is not evidence")

    # a single sample has no range, so an arm whose range contains it cannot be distinguished from it
    check(verdict([5.0], [6.0]) == "NOISE", "two single samples are not evidence (no spread on either arm)")
    check(verdict([5.0, 5.1], [6.0]) == "NOISE", "one sample on either arm cannot win")
    check(verdict([5.0], [5.0, 6.0]) == "NOISE", "an arm whose range contains the single sample is not evidence")
    check(verdict([5.0], [6.0, 7.0]) == "B", "a single sample below a disjoint arm loses to it")

    check(ratio != 1.0, "sanity: the real data is not a flat line")

    # direction: the same verdict reads differently for tok/s and for ms/token
    # --bench reads the metric a cell actually means: a ttft cell's latency is ttft_s, not bench.py's `median`
    ttft = {"cell": "ttft2k", "median": 61.9, "reps": [{"ttft_s": 0.91, "tg_tok_s": 61.9}, {"ttft_s": 0.93, "tg_tok_s": 62.0}]}
    series, lower = bench_series(ttft)
    check(series == [0.91, 0.93] and lower, "a ttft cell's series is ttft_s, and lower is better")
    pp = {"cell": "pp512", "median": 2200.0, "reps": [{"ttft_s": 0.2, "pp_tok_s": 2200.0}, {"ttft_s": 0.21, "pp_tok_s": 2210.0}]}
    series, lower = bench_series(pp)
    check(series == [2200.0, 2210.0] and not lower, "a prefill cell's series is pp_tok_s, and higher is better")

    # an arm that measured nothing must not be compared: bench.py records `skip` and no median for a cell it could not
    # run, and tonight one arm produced exactly that (a compute fault on a 35B). The mode says "no cell measured in both
    # arms" and returns 1 rather than printing an empty table that reads like agreement.
    import tempfile as _tf
    _d = _tf.mkdtemp()
    _empty = os.path.join(_d, "empty.json")
    _full = os.path.join(_d, "full.json")
    with open(_empty, "w") as f:
        json.dump({"cells": [{"cell": "pp512", "skip": "no content streamed"}], "started": "x", "tag": "empty"}, f)
    with open(_full, "w") as f:
        json.dump({"cells": [{"cell": "pp512", "median": 700.0, "prompt_tokens": 512,
                              "reps": [{"ttft_s": 0.7, "pp_tok_s": 700.0}, {"ttft_s": 0.71, "pp_tok_s": 710.0}]}],
                   "started": "x", "tag": "full"}, f)
    check(bench_compare(_empty, _full) == 1, "an arm that measured nothing is not a comparison")

    check("higher" in direction_word("A", False), "default direction says higher")
    check("lower" in direction_word("B", True), "--lower says lower is better")
    check("neither" in direction_word("NOISE", True), "noise names no winner")

    # spread() refuses what it cannot express
    for bad in ([], [0, 1]):
        try:
            spread(bad)
        except ValueError:
            pass
        else:
            fails += 1
            print(f"FAIL spread({bad}) should raise")

    print(f"self-test: {'PASS' if fails == 0 else str(fails) + ' FAILED'}")
    return 0 if fails == 0 else 1


def bench_series(cell):
    """(series, lower_is_better) for one bench.json cell, in the metric the cell actually means.

    `ttft2k` runs with max_tokens=32, so bench.py's `median` for it is the DECODE tok/s measured after the first token;
    the latency is `ttft_median_s` and the per-rep latency is `reps[].ttft_s`. summary.py:88 documents this and its
    value() returns ttft_median_s - reading `median` for a ttft cell is how a latency row ends up wrong.
    """
    reps = cell.get("reps") or []
    if cell.get("cell", "").startswith("ttft"):
        return [r.get("ttft_s") for r in reps if r.get("ttft_s") is not None], True
    key = "tg_tok_s" if (cell.get("prompt_tokens") or 0) and len(reps) and reps[0].get("tg_tok_s") is not None \
        else "pp_tok_s"
    return [r.get(key) for r in reps if r.get(key) is not None], False


def bench_compare(path_a, path_b):
    """Per-cell winner between two bench.py arms, using THIS file's rule (disjoint ranges)."""
    def cells(path):
        data = json.load(open(path))
        return {c["cell"]: c for c in data.get("cells", []) if c.get("median") is not None or c.get("ttft_median_s")}, \
            data.get("started", "?"), data.get("tag", "?")

    a_cells, a_start, a_tag = cells(path_a)
    b_cells, b_start, b_tag = cells(path_b)
    common = [c for c in a_cells if c in b_cells]
    if not common:
        print(f"medians.py: no cell measured in both arms (A: {sorted(a_cells)}, B: {sorted(b_cells)})", file=sys.stderr)
        return 1
    print(f"A: {path_a}  (tag {a_tag}, started {a_start})")
    print(f"B: {path_b}  (tag {b_tag}, started {b_start})")
    print()
    print("| cell | A median [min,max] n | B median [min,max] n | winner | margin | why |")
    print("|---|---|---|---|---|---|")
    counts = {"A": 0, "B": 0, "NOISE": 0}
    for name in common:
        sa, lower = bench_series(a_cells[name])
        sb, _ = bench_series(b_cells[name])
        if len(sa) < 2 or len(sb) < 2:
            counts["NOISE"] += 1
            print(f"| {name} | n={len(sa)} | n={len(sb)} | NOISE | - | a single sample is not a range |")
            continue
        result = verdict(sa, sb)
        counts[result] += 1
        ma, la, ha, _ = spread(sa)
        mb, lb, hb, _ = spread(sb)
        if result == "NOISE":
            margin, why = "-", "ranges overlap"
        elif result == "A":
            margin, why = f"x{((mb / ma) if lower else (ma / mb)):.2f}", "ranges disjoint"
        else:
            margin, why = f"x{((ma / mb) if lower else (mb / ma)):.2f}", "ranges disjoint"
        print(f"| {name} | {ma:.3f} [{la:.3f},{ha:.3f}] {len(sa)} | {mb:.3f} [{lb:.3f},{hb:.3f}] {len(sb)} | "
              f"{result} | {margin} | {why} |")
    print()
    print(f"verdicts: A {counts['A']}, B {counts['B']}, NOISE {counts['NOISE']} ({len(common)} cells compared)")
    return 0


def main(argv):
    if "--help" in argv or "-h" in argv:
        print(__doc__.strip())
        return 0
    if "--self-test" in argv:
        return self_test()
    if "--bench" in argv:
        i = argv.index("--bench")
        paths = [a for a in argv[i + 1:] if not a.startswith("-")]
        if len(paths) != 2:
            print("medians.py: --bench takes exactly two bench.json paths", file=sys.stderr)
            return 2
        return bench_compare(paths[0], paths[1])
    lower_is_better = "--lower" in argv
    argv = [a for a in argv if a != "--lower"]
    def samples(tokens, flag):
        out = []
        for tok in tokens:
            try:
                out.append(float(tok))
            except ValueError:
                print(f"medians.py: {flag} expects numbers, got {tok!r}", file=sys.stderr)
                raise SystemExit(2)
        return out

    if "--a" in argv:
        i = argv.index("--a")
        j = argv.index("--b") if "--b" in argv else len(argv)
        a = samples(argv[i + 1 : j], "--a")
        b = samples(argv[j + 1 :], "--b") if "--b" in argv else None
    else:
        a, b = samples(argv, "args"), None
    if not a:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    med, lo, hi, ratio = spread(a)
    label = "A" if b is not None else "arm"
    print(f"{label}: median {med:.3f}  min {lo:.3f}  max {hi:.3f}  spread {ratio:.2f}x  (n={len(a)})")
    if b is None:
        return 0
    if not b:
        print("B: no samples", file=sys.stderr)
        return 2
    mb, blo, bhi, bratio = spread(b)
    print(f"B: median {mb:.3f}  min {blo:.3f}  max {bhi:.3f}  spread {bratio:.2f}x  (n={len(b)})")
    result = verdict(a, b)
    # a difference inside the observed spread is not a result, and the winner needs its direction
    print(f"verdict: {direction_word(result, lower_is_better)}" +
          ("" if result != "NOISE" else f" (A={a}, B={b} medians)") if result != "NOISE"
          else "verdict: NOISE -- the ranges overlap, take more samples")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
