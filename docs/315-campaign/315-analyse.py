#!/usr/bin/env python3
"""engine#315 pre-registered analyser.

Reads the campaign console produced by `315-paired.sh` and applies the classification fixed in
`docs/315-preregistration.md` — which differs from `issue-315-analysis.py` in one decisive way:

    Face B (invalid UTF-8 / JSON parse failure) is a FAULT face, not "harness-side truncation".
    The issue's own record characterises invalid-UTF-8 output as the second real defect, and
    "zero NaN" alone was explicitly ruled out as an acceptance result.

Console line grammar (produced by 315-paired.sh; the existing script's regex also matches it):

    pair <N> arm <c|l>: <verdict> nan=<n> req=<n> ret3=<n> parse=<n> gtt_start=<G>G utf8=<n> alloc=<n>

Classification:
    invalid        alloc>0, verdict INVALID, or the request phase produced nothing (req==0)
    out-of-regime  gtt_start < 95 GiB (gate) or > 110 GiB (invalid-upper bound)
    fault          nan>0 or hsa>0 or parse>0 or utf8>0 or verdict in {FAULT,TRUNCATED} or req<66
    clean          otherwise

Usage: 315-analyse.py <console> [<console>...] [--baseline-a 2/37] [--baseline-b 1/25]
Exit status 0 only if the acceptance criterion is met: >=60 in-regime clean arms in every
non-control cell AND zero faults among the in-regime arms.
"""
from __future__ import annotations

import math
import random
import re
import sys
from math import comb

ARM = re.compile(
    r"^\s*pair\s+(\d+)\s+arm\s+([cl]):\s+(\S+)\s+nan=(\d+)\s+req=(\d+)\s+ret3=(\d+)\s+parse=(\d+)"
    r"\s+gtt_start=(\d+)G(?:\s+utf8=(\d+))?(?:\s+alloc=(\d+))?")
GATE = 95          # GiB, pre-registered
UPPER = 110        # GiB, above this the detector fails to allocate (invalid, not faulted)
CELL_NAME = {"c": "control", "l": "fix"}
SEED = 20261007


def parse(path):
    arms = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ARM.match(line)
            if not m:
                continue
            pair, cell, verdict, nan, req, ret3, parse_, gtt, utf8, alloc = m.groups()
            arms.append(dict(pair=int(pair), cell=cell, verdict=verdict, nan=int(nan), req=int(req),
                             parse=int(parse_), gtt=int(gtt), utf8=int(utf8 or 0),
                             alloc=int(alloc or 0)))
    return arms


def classify(a):
    """Pre-registered classification. Order matters: invalid, then regime, then fault, then clean."""
    if a["alloc"] > 0 or a["verdict"] == "INVALID" or a["req"] == 0:
        return "invalid"
    if a["gtt"] < GATE or a["gtt"] > UPPER:
        return "oor"
    if (a["nan"] > 0 or a["verdict"] in ("FAULT", "TRUNCATED")
            or a["parse"] > 0 or a["utf8"] > 0 or a["req"] < 66):
        return "fault"
    return "clean"


def classify_old_rule(a):
    """The recorded study's rule, for comparability: Face B counted as harness-side, not faulted."""
    if a["nan"] > 0:
        return "fault"
    if a["verdict"] == "TRUNCATED" or a["req"] < 66 or a["parse"] > 0 or a["utf8"] > 0:
        return "trunc"
    return "clean"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rule_of_three(n):
    """95% upper bound on the rate when 0 events are observed in n trials."""
    return 3.0 / n if n else 1.0


def fisher_two_sided(a, nA, b, nB):
    N, K = nA + nB, a + b
    if K == 0 or K == N:
        return 1.0

    def prob(x):
        if x < 0 or K - x < 0 or x > nA or K - x > nB:
            return 0.0
        return comb(nA, x) * comb(nB, K - x) / comb(N, K)

    p0, tot = prob(a), 0.0
    for x in range(max(0, K - nB), min(nA, K) + 1):
        p = prob(x)
        if p <= p0 + 1e-12:
            tot += p
    return min(1.0, tot)


def p_rate_a_below_b(a, nA, b, nB, draws=200000):
    random.seed(SEED)
    hit = 0
    for _ in range(draws):
        if random.betavariate(1 + a, 1 + nA - a) < random.betavariate(1 + b, 1 + nB - b):
            hit += 1
    return hit / draws


def main(argv):
    paths = [a for a in argv if not a.startswith("--")]
    base_a, base_b = (2, 37), (1, 25)
    for arg in argv:
        if arg.startswith("--baseline-a"):
            base_a = tuple(int(x) for x in arg.split("=")[1].split("/"))
        if arg.startswith("--baseline-b"):
            base_b = tuple(int(x) for x in arg.split("=")[1].split("/"))

    arms = [a for p in paths for a in parse(p)]
    if not arms:
        print("no arms parsed — check the console path and the line grammar")
        return 2

    print(f"arms parsed: {len(arms)} from {len(paths)} console(s)\n")
    cells = {}
    for cell in sorted({a["cell"] for a in arms}):
        mine = [a for a in arms if a["cell"] == cell]
        counts = {"invalid": 0, "oor": 0, "fault": 0, "clean": 0}
        old = {"fault": 0, "trunc": 0, "clean": 0}
        for a in mine:
            counts[classify(a)] += 1
            old[classify_old_rule(a)] += 1
        inreg = counts["clean"] + counts["fault"]
        lo, hi = wilson(counts["fault"], inreg) if inreg else (0.0, 1.0)
        cells[cell] = dict(total=len(mine), inreg=inreg, **counts)
        print(f"cell {CELL_NAME.get(cell, cell)} ({cell}): scheduled={len(mine)} "
              f"in-regime={inreg} clean={counts['clean']} fault={counts['fault']} "
              f"invalid={counts['invalid']} out-of-regime={counts['oor']}")
        if inreg:
            rate = 100.0 * counts["fault"] / inreg
            print(f"    fault rate {counts['fault']}/{inreg} = {rate:.1f}%  95% CI {100*lo:.1f}-{100*hi:.1f}%")
        if counts["clean"]:
            ub = 100.0 * rule_of_three(counts["clean"])
            print(f"    if clean count stands: 95% upper bound on the true rate = {ub:.1f}% "
                  f"(rule of three over {counts['clean']} clean arms)")
        print(f"    under the recorded study's rule (Face B = harness-side): "
              f"clean={old['clean']} fault={old['fault']} trunc={old['trunc']}")

    if "c" in cells and "l" in cells:
        c, l = cells["c"], cells["l"]
        p = fisher_two_sided(c["fault"], c["inreg"] or 1, l["fault"], l["inreg"] or 1)
        pb = p_rate_a_below_b(c["fault"], c["inreg"] or 1, l["fault"], l["inreg"] or 1)
        print(f"\ncontrol vs fix: Fisher exact two-sided p = {p:.3f}; "
              f"P(rate_control < rate_fix) = {pb:.3f}")
        print(f"recorded baseline for reference: A {base_a[0]}/{base_a[1]} "
              f"({100*base_a[0]/base_a[1]:.1f}%), B {base_b[0]}/{base_b[1]} "
              f"({100*base_b[0]/base_b[1]:.1f}%)")

    fix = cells.get("l") or cells.get("c")
    ok = fix["clean"] >= 60 and fix["fault"] == 0
    print("\nacceptance (fix cell): >=60 in-regime clean arms AND zero faults on both faces")
    print(f"  clean={fix['clean']} (need 60)   faults={fix['fault']} (need 0)   -> "
          f"{'PASS' if ok else 'NOT MET'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
