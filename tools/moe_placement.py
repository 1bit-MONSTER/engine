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
"""Turn a `tools/moe_trace.cpp` trace into a per-layer expert placement table (engine#1702).

Why per layer, and why decode rows only: measured on 2026-11-09 (engine#1702), a **per-layer** co-activation placement
puts all of a decoded token's experts on one device for 44-60 % of tokens, against 0.2-0.6 % for a contiguous id split and
2.7-6.9 % for **one global** permutation. The global row is what fixes the shape of this file's output: one permutation is
worth almost nothing, so the plan is keyed by layer. The same measurement shows placement adds 5.4-6.9 % on batched prefill
while deduplication gives 90-94 %, so prefill rows ('P') are excluded: placement is a decode lever and this tool does not
repackage a prefill result as one.

Replay only. Nothing here touches a model, a tensor or the graph: the output is a JSON table a splitter or a streamer may
choose to consume, and an unused table changes nothing.

Usage:
    tools/moe_placement.py <trace> [-g GROUPS] [-o out.json] [--min-pairs N]
    tools/moe_placement.py --self-test

Trace format (tools/moe_trace.cpp, Writer::flush): `step<TAB>layer<TAB>P|D<TAB>id,id,...` and, on 'D' rows, two further
tab-separated predictions. Only the first four fields are read.
"""
import argparse
import json
import sys
from collections import defaultdict

# The one-device share this tool reports: the fraction of decode steps whose experts all fall in a single group. It is the
# number engine#1702 measured, so a plan's quality is comparable with the report that motivated it.
def one_device_share(plan, trace):
    """plan: {layer: [[ids in group 0], [ids in group 1], ...]}; trace: {layer: [[ids per step], ...]} -> (share, steps)."""
    same = total = 0
    for layer, steps in trace.items():
        groups_of_layer = plan.get(layer)
        if groups_of_layer is None:
            continue
        group_of = {e: i for i, group in enumerate(groups_of_layer) for e in group}
        for ids in steps:
            total += 1
            groups = {group_of.get(e, -1) for e in ids}
            if len(groups) == 1 and -1 not in groups:
                same += 1
    return (same / total if total else 0.0), total


def parse_trace(path):
    """{layer: [[ids], ...]} over DECODE rows only, in file order."""
    trace = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            _, layer, kind, ids = fields[0], fields[1], fields[2], fields[3]
            if kind != "D":
                continue
            try:
                layer = int(layer)
                row = [int(x) for x in ids.split(",") if x.strip()]
            except ValueError:
                continue
            if row:
                trace[layer].append(row)
    return dict(trace)


def coactivation(steps, n_experts):
    """Within-layer P(e2 | e1): counts[e1][e2] how often e2 was in the same step as e1."""
    counts = defaultdict(lambda: [0] * n_experts)
    for row in steps:
        for a in row:
            for b in row:
                if a != b:
                    counts[a][b] += 1
    return counts


def plan_layer(steps, n_experts, groups):
    """A balanced G-group order for one layer: greedy max-affinity chain, then a balance-preserving split.

    Deliberately not a full Kernighan-Lin: the report's plan uses KL over the same measure, and for the sizes that matter
    (256 or 512 experts) the greedy chain captures the clustering the share is computed from. The test pins the property
    that matters (co-activating experts end up in one group) rather than the exact partition, so a KL pass can replace this
    without changing the output's meaning.
    """
    counts = coactivation(steps, n_experts)
    seen = {e for row in steps for e in row}
    order = sorted(seen)
    if not order:
        return []
    # affinity e -> sum of co-activation with everything placed before it, greedily extended
    placed, remaining = [order[0]], set(order[1:])
    while remaining:
        last = placed[-1]
        nxt = max(remaining, key=lambda e: (counts[last][e], counts[e][last], -e))
        placed.append(nxt)
        remaining.discard(nxt)
    if groups <= 1:
        return placed
    # balance: chunk the chain, so a group is a contiguous run of strongly-affine experts
    per = (len(placed) + groups - 1) // groups
    return placed[: per * groups] if len(placed) > per * groups else placed


def group_of_plan(order, groups):
    per = (len(order) + groups - 1) // groups
    return [order[i * per :(i + 1) * per] for i in range(groups) if order[i * per :(i + 1) * per]]


def build(trace, groups, min_pairs=1):
    layers = {}
    for layer, steps in sorted(trace.items()):
        pairs = sum(len(r) * (len(r) - 1) for r in steps)
        if pairs < min_pairs:
            continue
        n_experts = max(max(r) for r in steps) + 1
        order = plan_layer(steps, n_experts, groups)
        if not order:
            continue
        layers[str(layer)] = {"groups": group_of_plan(order, groups), "decode_steps": len(steps),
                              "experts_seen": len(order)}
    return layers


def contiguous(trace, groups):
    """The control: a per-layer contiguous id split, which the report measured at 0.2-0.6 %."""
    plan = {}
    for layer, steps in trace.items():
        n = max(max(r) for r in steps) + 1
        per = (n + groups - 1) // groups
        plan[layer] = [list(range(i * per, min(n, (i + 1) * per))) for i in range(groups) if i * per < n]
    return plan


def self_test():
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what)
        fails += 0 if ok else 1

    # A synthetic trace with a KNOWN answer: clusters of four experts that never mix, ids deliberately far apart so a
    # contiguous split cannot fake the answer. G=2 is the report's own shape (HRX + NPU).
    def clusters(ids_per_cluster, reps=50):
        steps = []
        for _ in range(reps):
            for c in ids_per_cluster:
                steps.append(list(c))
        return {0: steps}

    # INTERLEAVED ids, which is the point: real co-activating experts are not contiguous in id space, so a contiguous
    # split cuts through every cluster. With ids 0-3 and 4-7 a contiguous G=2 split would score 100 % and prove nothing.
    two = clusters([[0, 2, 4, 6], [1, 3, 5, 7]])
    share, n = one_device_share({0: build(two, 2)["0"]["groups"]}, two)
    check(n == 100, f"two clusters, 2 groups: {n} decode steps counted")
    check(share > 0.99, f"G=2 puts each co-activating cluster on one device: share {share:.0%}")
    cshare, _ = one_device_share(contiguous(two, 2), two)
    check(cshare < 0.5, f"the contiguous control cannot: {cshare:.0%} (the report measured 0.2-0.6 %)")

    # Four interleaved clusters and four groups: every cluster fits one group.
    four = clusters([[0, 4], [1, 5], [2, 6], [3, 7]])
    fshare, _ = one_device_share({0: build(four, 4)["0"]["groups"]}, four)
    check(fshare > 0.99, f"G=4 with four clusters: share {fshare:.0%}")

    # A token whose experts span every group is on no single device, whatever the plan -- tested against an explicit plan
    # rather than one build() derived, so the arm fails if one_device_share ever stops reading group membership.
    spread = {0: [[0, 1, 2, 3]] * 10}
    sshare, _ = one_device_share({0: [[0], [1], [2], [3]]}, spread)
    check(sshare == 0.0, f"a token spanning all four groups is never on one device ({sshare:.0%})")
    together, _ = one_device_share({0: [[0, 1, 2, 3], [], [], []]}, spread)
    check(together == 1.0, f"... and the same token in one group is always ({together:.0%})")

    # Degenerate group counts must be exact, not approximate.
    check(len(build(two, 1)["0"]["groups"]) == 1, "G=1 is one group")
    total = sum(len(g) for g in build(two, 16)["0"]["groups"])
    check(total == 8, f"G=16 over 8 seen experts keeps all 8 ({total})")

    # Prefill rows must be ignored: the report says placement is not a prefill lever, so a tool that counted them would
    # inflate its own case.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".trace", delete=False) as f:
        f.write("0\t0\tP\t0,1,2,3\n0\t0\tD\t0,1,2,3\n")
        path = f.name
    try:
        parsed = parse_trace(path)
        check(parsed == {0: [[0, 1, 2, 3]]}, "only the decode row is read")
    finally:
        os.unlink(path)

    print(f"\nmoe_placement self-test: {fails} failure(s)")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("-g", "--groups", type=int, default=2, help="device groups (default 2: HRX + NPU)")
    ap.add_argument("-o", "--out", help="write the JSON table here (default: stdout)")
    ap.add_argument("--min-pairs", type=int, default=1, help="skip a layer with fewer than N co-activating pairs")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.trace:
        ap.error("a trace file is required (or --self-test)")
    if a.groups < 1:
        ap.error("--groups must be >= 1")
    trace = parse_trace(a.trace)
    if not trace:
        print("moe_placement: no decode rows in the trace (a 'P'-only trace is a prefill run, which placement does not "
              "help -- see engine#1702 §3)", file=sys.stderr)
        return 2
    table = {"schema": "1bit/moe-placement/1", "groups": a.groups, "layers": build(trace, a.groups, a.min_pairs)}
    text = json.dumps(table, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    plan = {int(l): v["groups"] for l, v in table["layers"].items()}
    share, n = one_device_share(plan, trace)
    cshare, _ = one_device_share(contiguous(trace, a.groups), trace)
    print(f"moe_placement: {len(table['layers'])} layer(s), {a.groups} group(s); one-device share {share:.1%} over {n} "
          f"decode steps (contiguous control {cshare:.1%})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
