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
"""Read the row probe's fingerprint lines: which token row is NaN, and is it the same row throughout?

Why this exists: a NaN *count* cannot distinguish one token's row from many, and the answer decides the
mechanism (docs/315-mechanism.md section 47). The extended fingerprint prints `ne0=` (innermost extent),
`first=` (index of the first NaN) and `row=` (first/ne0 = the token index), so the question becomes whether
`row` is constant across the whole chain of named tensors and whether it equals the last row (token T-1).

Usage: row-readout.py <arm-dir-or-glob> [...] [--all]
  default: first NaN-bearing line per distinct node name, per arm
  --all:   every NaN-bearing line (in file order)
"""
from __future__ import annotations

import glob
import os
import re
import sys

LINE = re.compile(
    r"call=(?P<call>\d+) name=(?P<name>\S+) op=(?P<op>\S+) bytes=(?P<bytes>\d+) "
    r"nan=(?P<nan>\d+) of (?P<n>\d+)"
    r"(?: ne0=(?P<ne0>-?\d+) first=(?P<first>-?\d+) row=(?P<row>-?\d+))?"
    r" hash=(?P<hash>[0-9a-f]+)")

# GGML_HRX_SENTINEL=1 fills the logits allocation with payload NaN and the guard reports the match
# count as `sentinel=`. Read it (docs/315-STATUS.md records that it went unread for two days).
SENT = re.compile(r"sentinel=(?P<sent>\d+)")


def arms(paths):
    out = []
    for p in paths:
        for d in sorted(glob.glob(p)):
            log = os.path.join(d, "server.log")
            if os.path.isdir(d) and os.path.exists(log):
                out.append((os.path.basename(d.rstrip("/")), log))
    return out


def main(argv):
    show_all = "--all" in argv
    paths = [a for a in argv if not a.startswith("--")]
    found = arms(paths)
    if not found:
        print("no arm directories with a server.log matched")
        return 2
    for tag, log in found:
        lines = []
        with open(log, errors="replace") as fh:
            for row in fh:
                if "hrx-tensor-fp" not in row:
                    continue
                m = LINE.search(row)
                if m:
                    lines.append(m.groupdict())
        guard = 0
        sent_armed = 0
        sent_matched = 0
        with open(log, errors="replace") as fh:
            for r in fh:
                if "HRX returned NaN logits" in r:
                    guard += 1
                if "[hrx-sentinel] filled" in r:
                    sent_armed += 1
                m = SENT.search(r)
                if m:
                    sent_matched += int(m.group(1))
        # The sentinel field was armed for days and never read out (docs/315-STATUS.md), so it is
        # printed here beside every arm's verdict rather than left in the raw log. sentinel>0 means
        # the logits region still held the payload the fill wrote: the device never wrote it.
        sent_note = ("not armed" if sent_armed == 0 else
                     ("0 matches -> device wrote the region" if sent_matched == 0 else
                      f"{sent_matched} MATCHES -> reads are stale, never-written"))
        print(f"=== {tag[:44]}  guard_lines={guard}  fingerprint_lines={len(lines)}  "
              f"sentinel_armed={sent_armed}  sentinel={sent_note}")
        if not lines:
            print("    (no NaN-bearing fingerprint line: with NANONLY=1 that means no NaN in any named node)")
            continue
        seen = set()
        rows = set()
        for g in lines:
            if not show_all and g["name"] in seen:
                continue
            seen.add(g["name"])
            n, nan = int(g["n"]), int(g["nan"])
            if g["row"] is not None:
                row, ne0, first = int(g["row"]), int(g["ne0"]), int(g["first"])
                tokens = n // ne0 if ne0 else 0
                # A T=1 tensor (the decode's logits, or a decode graph's activations) has one row by
                # construction, so its row number carries no per-token information and must not enter the
                # consistency verdict - doing so would manufacture a false "different rows".
                if tokens > 1:
                    rows.add(row)
                tail = "" if tokens <= 1 or row < 0 else ("  <-- LAST ROW" if row == tokens - 1 else "")
                note = "  (T=1: single token, row is trivial)" if tokens <= 1 else ""
                print(f"    {g['name']:<22} {g['op']:<10} nan={nan:>7} of {n:<8} ne0={ne0:<6} "
                      f"first={first:<9} row={row:<5} of T={tokens}{tail}{note}")
            else:
                print(f"    {g['name']:<22} {g['op']:<10} nan={nan:>7} of {n:<8} (no row fields: "
                      f"this line predates the extension)")
        if rows:
            verdict = ("SAME ROW throughout the multi-token chain -> per-token state" if len(rows) == 1
                       else f"{len(rows)} DIFFERENT ROWS across multi-token tensors -> not one per-token state")
            print(f"    rows seen (multi-token tensors only): {sorted(rows)}  -> {verdict}")
        elif seen:
            print("    no multi-token tensor in this arm carried NaN (all affected tensors were T=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
