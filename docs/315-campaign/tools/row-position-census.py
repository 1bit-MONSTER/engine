#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""Tabulate every multi-token NaN reading that carries the extended fingerprint's row fields.

Why: the extended fingerprint prints first= and row= (the corrupted row index) and ne0= (the row width),
so T = n/ne0 and row/T is the corrupted row's position in the sequence. The row-size work established the
SIZE is one row; whether the row is the LAST token (row = T-1) has never been tabulated, and it decides
between a per-last-token mechanism (a row written once per decode step) and a front-of-buffer one.

Read-only; run on the box against the archived arm logs.
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys

LINE = re.compile(
    r"name=(?P<name>\S+) op=\S+ bytes=\d+ nan=(?P<nan>\d+) of (?P<n>\d+) "
    r"ne0=(?P<ne0>-?\d+) first=(?P<first>-?\d+) row=(?P<row>-?\d+)")


def main() -> int:
    rows = []
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
                continue          # a T=1 row carries no positional information
            rows.append((T, int(g["row"]), int(g["nan"]), g["name"], arm))
    rows.sort()
    print("multi-token NaN readings carrying row fields:", len(rows))
    if not rows:
        print("none: no archived arm used the extended fingerprint on a multi-token tensor")
        return 0
    hdr = "%5s %5s %9s %6s  %-24s %s" % ("T", "row", "nan", "row/T", "name", "arm")
    print(hdr)
    print("-" * len(hdr))
    for T, r, nan, name, arm in rows[:24]:
        print("%5d %5d %9d %6.2f  %-24s %s" % (T, r, nan, r / T, name[:24], arm[:24]))
    fr = [r / T for T, _, _, _, _ in rows]
    print("row/T: min=%.3f median=%.3f max=%.3f" % (min(fr), statistics.median(fr), max(fr)))
    print("row == T-1 (last token): %d of %d" % (sum(1 for T, r, _, _, _ in rows if r == T - 1), len(rows)))
    print("row == 0   (first token): %d of %d" % (sum(1 for T, r, _, _, _ in rows if r == 0), len(rows)))
    print("row < 4   (front of buffer): %d of %d" % (sum(1 for _, r, _, _, _ in rows if r < 4), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
