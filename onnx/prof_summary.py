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
"""Summarise an ORT profiling JSON (`ort_run --profile PREFIX`).

`ort_run` prints wall time per stage; this says which ops the stage is made of, so the next change is
chosen from evidence. ORT emits one record per node per Run, so the same node appears once per step —
the `run` column below splits them.

    python3 prof_summary.py /tmp/prof2048_2026-10-31_07-31-54.json [--top 12]
"""
import argparse
import collections
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="profiling JSON written by ort_run --profile")
    ap.add_argument("--top", type=int, default=10, help="rows per table")
    a = ap.parse_args()

    with open(a.trace) as f:
        events = json.load(f)

    nodes = [e for e in events if e.get("cat") == "Node"]
    if not nodes:
        print("no node events", file=sys.stderr)
        return 2

    # Group by Run: ORT's pid/tid stay constant, but the trace restarts its node list per Run, so the
    # event order is enough — a run boundary is where a node name repeats.
    seen = set()
    run, runs = 0, collections.defaultdict(list)
    for e in nodes:
        key = (e.get("name"), e.get("ts"))
        if key in seen:
            run += 1
            seen.clear()
        seen.add(key)
        runs[run].append(e)

    by_op = collections.Counter()
    per_run = collections.Counter()
    for r, evs in runs.items():
        total = sum(e.get("dur", 0) for e in evs)
        per_run[r] = total
        for e in evs:
            by_op[e.get("args", {}).get("op_name", "?")] += e.get("dur", 0)

    print(f"runs: {len(runs)}   nodes: {len(nodes)}")
    print("\nper run (ms):")
    for r in sorted(per_run):
        print(f"  run {r}: {per_run[r] / 1000.0:9.1f}  ({sum(1 for e in runs[r])} nodes)")

    grand = sum(by_op.values()) or 1
    print(f"\ntop ops over all runs (share of {grand / 1000.0:.0f} ms of node time):")
    for op, us in by_op.most_common(a.top):
        print(f"  {op:<22} {us / 1000.0:9.1f} ms  {100.0 * us / grand:5.1f}%")

    print(f"\ntop individual nodes:")
    for e in sorted(nodes, key=lambda e: -e.get("dur", 0))[: a.top]:
        op = e.get("args", {}).get("op_name", "?")
        print(f"  {e.get('dur', 0) / 1000.0:8.1f} ms  {op:<20} {e.get('name', '')[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
