#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""Re-cut the row-position census so no single arm or tensor family can carry the conclusion.

Why: the raw census reported 1,396 of 1,525 readings at row == 0, but the counts are dominated by one arm's
family of tiny ffn_moe_weights tensors (T=4, ne0=1). A conclusion drawn from a raw count would be a
conclusion about that family. This reports per ARM and per FAMILY, and separates the hidden-state tensors
(the ones the size finding is about: l_out-*, node_NNNN, norm, q/k/v, ffn_*) from the index/weight tensors
(ffn_moe_weights-*, ffn_moe_probs-*, ids), so the front-of-buffer claim has to survive on each.

Read-only; run on the box against the archived arm logs.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import sys

LINE = re.compile(
    r"name=(?P<name>\S+) op=\S+ bytes=\d+ nan=(?P<nan>\d+) of (?P<n>\d+) "
    r"ne0=(?P<ne0>-?\d+) first=(?P<first>-?\d+) row=(?P<row>-?\d+)")

INDEXISH = re.compile(r"^(ffn_moe_weights|ffn_moe_probs|ffn_moe_topk|ids|selected|inp_tokens|node_\d+ op=UNARY)")


def family(name: str) -> str:
    if name.startswith(("ffn_moe_weights", "ffn_moe_probs", "ffn_moe_topk")):
        return "index/weight"
    if name.startswith("l_out") or name.startswith("node_"):
        return "hidden-state"
    return "other"


def main() -> int:
    readings = []           # (arm, family, T, row, nan)
    for f in sorted(glob.glob(os.path.expanduser("~/wt/rt-det/*/server.log"))):
        arm = os.path.basename(os.path.dirname(f)).replace("-GLM-4.7-Flash-Q4_K_M", "")
        try:
            with open(f, errors="replace") as fh:
                txt = fh.read()
        except OSError:
            continue
        for m in LINE.finditer(txt):
            g = m.groupdict()
            n, ne0 = int(g["n"]), int(g["ne0"])
            if ne0 <= 0:
                continue
            T = n // ne0
            if T <= 1:
                continue
            readings.append((arm, family(g["name"]), T, int(g["row"]), int(g["nan"])))

    print("total multi-token row-field readings:", len(readings))
    if not readings:
        return 0

    by_family = collections.defaultdict(list)
    for arm, fam, T, row, nan in readings:
        by_family[fam].append((arm, T, row, nan))

    for fam in sorted(by_family):
        rs = by_family[fam]
        n = len(rs)
        arms = sorted({a for a, _, _, _ in rs})
        at_last = sum(1 for _, T, r, _ in rs if r == T - 1)
        at_zero = sum(1 for _, _, r, _ in rs if r == 0)
        front4 = sum(1 for _, _, r, _ in rs if r < 4)
        print(f"\n--- family={fam}  readings={n}  arms={len(arms)}")
        print(f"    row == T-1 (last token): {at_last} ({at_last/n*100:.1f}%)")
        print(f"    row == 0   (first token): {at_zero} ({at_zero/n*100:.1f}%)")
        print(f"    row <  4   (front of buffer): {front4} ({front4/n*100:.1f}%)")
        # per-arm, so one arm cannot carry it
        per_arm = collections.defaultdict(list)
        for a, T, r, _ in rs:
            per_arm[a].append((T, r))
        print("    per arm (rows seen):")
        for a in arms:
            rr = per_arm[a]
            rows_seen = sorted({r for _, r in rr})
            tmax = max(T for T, _ in rr)
            mark = "LAST-TOKEN" if any(r == T - 1 for T, r in rr) else ("front" if max(rows_seen) < 4 else "mixed")
            print(f"      {a[:30]:30s} n={len(rr):4d} Tmax={tmax:4d} rows={rows_seen[:6]} -> {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
