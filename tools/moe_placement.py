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
2.7-6.9 % for **one global** permutation. Those shares are IN-SAMPLE (scored on the steps the plan was built from). On
steps the plan never saw it is 19-26 % for the second half of the same conversation and 7-14 % for a different prompt
(held_out_share() below, and `--score`), still 15-50x the contiguous split. The global row is what fixes the shape of this file's output: one permutation is
worth almost nothing, so the plan is keyed by layer. The same measurement shows placement adds 5.4-6.9 % on batched prefill
while deduplication gives 90-94 %, so prefill rows ('P') are excluded: placement is a decode lever and this tool does not
repackage a prefill result as one.

Replay only. Nothing here touches a model, a tensor or the graph: the output is a JSON table a splitter or a streamer may
choose to consume, and an unused table changes nothing.

Usage:
    tools/moe_placement.py <trace> [-g GROUPS] [-e EXPERTS] [-o out.json] [--min-pairs N]
    tools/moe_placement.py <trace> --score table.json
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


def chain_order(counts, seen):
    """A greedy max-affinity chain over the experts the sample fired: the starting order, and the order inside a group."""
    order = sorted(seen)
    if not order:
        return []
    placed, remaining = [order[0]], set(order[1:])
    while remaining:
        last = placed[-1]
        nxt = max(remaining, key=lambda e: (counts[last][e], counts[e][last], -e))
        placed.append(nxt)
        remaining.discard(nxt)
    return placed


# How many of the best candidates on each side a Kernighan-Lin step compares. The pair with the largest gain is almost
# always among the few nodes with the largest D on each side (Kernighan & Lin's own shortcut); the full product is
# 256 x 256 per step on a 512-expert layer and buys nothing measurable.
KL_CANDIDATES = 8


def kl_refine_pair(counts, ga, gb):
    """One Kernighan-Lin refinement of two equal-role groups, in place. Returns the co-activation weight it moved inside.

    The weight between two experts is how often they fired in the same step (symmetric). A pass tentatively swaps pairs,
    each node at most once, always taking the best available swap even when its gain is negative, and keeps the prefix of
    swaps with the largest total gain -- that is what lets it leave a local optimum a plain "swap while it helps" loop
    stops in. Passes repeat until one gains nothing. Sizes never change, so a balanced split stays balanced.
    """
    total = 0
    while True:
        in_a = set(ga)
        # D[e] = (weight to the other group) - (weight to its own group): what moving e alone would gain
        D = {}
        for e in ga:
            row = counts[e]
            D[e] = sum(row[x] for x in gb) - sum(row[x] for x in ga)
        for e in gb:
            row = counts[e]
            D[e] = sum(row[x] for x in ga) - sum(row[x] for x in gb)
        free_a, free_b = set(ga), set(gb)
        swaps, gains = [], []
        while free_a and free_b:
            top_a = sorted(free_a, key=lambda e: (-D[e], e))[:KL_CANDIDATES]
            top_b = sorted(free_b, key=lambda e: (-D[e], e))[:KL_CANDIDATES]
            gain, a, b = max((D[x] + D[y] - 2 * counts[x][y], -x, -y) for x in top_a for y in top_b)
            a, b = -a, -b
            swaps.append((a, b))
            gains.append(gain)
            free_a.discard(a)
            free_b.discard(b)
            # a goes to B and b goes to A: every free node's D moves by twice its weight to the pair
            ra, rb = counts[a], counts[b]
            for e in free_a:
                D[e] += 2 * ra[e] - 2 * rb[e]
            for e in free_b:
                D[e] += 2 * rb[e] - 2 * ra[e]
        best, best_k, run = 0, 0, 0
        for k, g in enumerate(gains, 1):
            run += g
            if run > best:
                best, best_k = run, k
        if best_k == 0:
            return total
        for a, b in swaps[:best_k]:
            ga[ga.index(a)] = b
            gb[gb.index(b)] = a
        total += best
        del in_a


def plan_layer(steps, n_experts, groups):
    """A balanced G-group order for one layer: a greedy max-affinity chain cut into G runs, refined by Kernighan-Lin.

    The chain alone is not enough, and that is measured rather than argued (engine#1702, the report's own traces, G=2):
    the chain split puts all of a decoded token's experts on one device for 34.8 % (Qwen3.6 code), 34.8 % (chat) and
    24.7 % (Flash-Next) of tokens, where the report's balanced Kernighan-Lin plan measured 60.0 %, 46.2 % and 44.0 %.
    A chain follows one neighbour at a time, so it strings clusters together and the cut lands wherever the count says.
    The refinement swaps experts between groups to move co-activation weight inside them; group sizes do not change.

    The returned order is group by group, and inside each group it is again a max-affinity chain, so neighbours in the
    file are experts that fire together -- the order a streamer's layout would want.
    """
    counts = coactivation(steps, n_experts)
    seen = {e for row in steps for e in row}
    placed = chain_order(counts, seen)
    if not placed:
        return []
    if groups <= 1:
        return placed
    parts = group_of_plan(placed, groups)
    if len(parts) > 1:
        moved = True
        while moved:   # every pair of groups, until a whole round moves nothing
            moved = False
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    if kl_refine_pair(counts, parts[i], parts[j]) > 0:
                        moved = True
    return [e for part in parts for e in chain_order(counts, part)]


def group_of_plan(order, groups):
    per = (len(order) + groups - 1) // groups
    return [order[i * per :(i + 1) * per] for i in range(groups) if order[i * per :(i + 1) * per]]


def fill_unseen(group_lists, total, groups):
    """Place the experts with no co-activation evidence, so the table covers the model and not just the sample.

    A decode sample does not fire every expert in every layer (on a 256- or 512-expert layer it is usually far from it),
    so a table built only from `seen` leaves the rest of the layer unplaced and says nothing -- a consumer would either
    skip them or index past the end. There is no affinity evidence for these, so the only defensible default is balance:
    round-robin onto the currently smallest group, with ties going to the earlier group so the output is deterministic.
    The count is reported per layer as `unseen`, so a layer where most of the plan is a default is visible rather than
    mistaken for a measurement.
    """
    placed = {e for g in group_lists for e in g}
    out = [list(g) for g in group_lists] + [[] for _ in range(groups - len(group_lists))]
    for e in range(total):
        if e not in placed:
            min(out, key=len).append(e)
    return [g for g in out if g]


def build(trace, groups, min_pairs=1, experts=None):
    layers = {}
    for layer, steps in sorted(trace.items()):
        pairs = sum(len(r) * (len(r) - 1) for r in steps)
        if pairs < min_pairs:
            continue
        seen_max = max(max(r) for r in steps) + 1
        if experts is not None:
            if min(min(r) for r in steps) < 0 or seen_max > experts:
                raise ValueError(
                    f"layer {layer} visits expert {max(max(r) for r in steps)}, but the model has {experts} "
                    f"(0..{experts - 1}): the trace and the model disagree, so a placement built from it would "
                    f"put an expert id the model does not have into a group")
            total = experts
        else:
            total = seen_max
        order = plan_layer(steps, total, groups)
        if not order:
            continue
        group_lists = fill_unseen(group_of_plan(order, groups), total, groups)
        layers[str(layer)] = {"groups": group_lists, "decode_steps": len(steps), "experts": total,
                              "experts_seen": len(order), "unseen": total - len(order)}
    return layers


def held_out_share(trace, groups, min_pairs=1, experts=None):
    """(share, steps): the table built from the FIRST half of each layer's decode steps, scored on the SECOND half.

    The share build() reports for its own trace is in-sample: the same steps chose the groups and are then counted. With
    a few hundred steps and 256 or 512 experts a partition can keep many of exactly those steps intact, so that number is
    an upper bound on what a consumer sees, not an estimate of it. Measured on engine#1702's own traces (G=2): in-sample
    51.1 % / 55.6 % / 52.3 %, held out 22.5 % / 26.3 % / 18.9 % (Qwen3.6 chat, code; Flash-Next), and a Qwen3.6 table
    built from the chat prompt scores 13.5 % on the code prompt. A placement is fixed in a model while prompts change, so
    the held-out figure is the one to quote.
    """
    first = {layer: steps[: len(steps) // 2] for layer, steps in trace.items() if len(steps) >= 2}
    second = {layer: steps[len(steps) // 2:] for layer, steps in trace.items() if len(steps) >= 2}
    if not first:
        return 0.0, 0
    layers = build(first, groups, min_pairs, experts)
    return one_device_share({int(l): v["groups"] for l, v in layers.items()}, second)


def load_table(path):
    """{layer: groups} from a table this tool wrote; ValueError naming what is wrong with anything else."""
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table, dict) or table.get("schema") != "1bit/moe-placement/1":
        raise ValueError(f"{path} is not a 1bit/moe-placement/1 table (schema {table.get('schema')!r})"
                         if isinstance(table, dict) else f"{path} is not a placement table")
    return {int(layer): entry["groups"] for layer, entry in table.get("layers", {}).items()}


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

    # A decode sample misses experts, and a table that placed only the seen ones would leave the rest of the layer
    # unplaced while saying nothing -- so every expert in range must appear EXACTLY once, and the ones with no evidence
    # must be counted rather than hidden.
    partial = {0: [[0, 1, 6, 7]] * 40}          # 2..5 never fire
    one = build(partial, 2)["0"]
    flat = [e for g in one["groups"] for e in g]
    check(sorted(flat) == list(range(8)) and one["unseen"] == 4,
          f"a partial sample still covers every expert in range once ({sorted(flat)} unseen {one['unseen']})")
    wider = build(partial, 2, experts=16)["0"]
    flat16 = [e for g in wider["groups"] for e in g]
    check(sorted(flat16) == list(range(16)) and wider["unseen"] == 12,
          f"--experts 16 covers 0..15 with 12 counted unseen ({len(flat16)} placed)")
    sizes = [len(g) for g in wider["groups"]]
    check(max(sizes) - min(sizes) <= 1, f"evidence-less experts are balanced across groups ({sizes})")
    try:
        build({0: [[0, 1, 7]] * 40}, 2, experts=4)
        check(False, "a trace that visits expert 7 of a 4-expert model is refused")
    except ValueError as e:
        check("expert 7" in str(e) and "4" in str(e), f"... refused by naming the id and the count ({str(e)[:40]}...)")

    # The partition, not just the chain (engine#1702). Two clusters of four plus a bridge pair (0, 1) that fires more
    # often than any pair inside a cluster: the max-affinity chain leaves cluster {0,2,4,6} for 1 at its first step and
    # then walks {3,5,7}, so cutting it in half splits BOTH clusters and keeps only the bridge steps (60 of 160). The
    # right plan keeps the clusters and cuts the bridge (100 of 160). Kernighan-Lin gets there by swaps of equal size.
    bridged = {0: [[0, 2, 4, 6]] * 50 + [[1, 3, 5, 7]] * 50 + [[0, 1]] * 60}
    b_groups = build(bridged, 2)["0"]["groups"]
    bshare, bn = one_device_share({0: b_groups}, bridged)
    check(abs(bshare - 100 / 160) < 1e-9,
          f"a bridge pair does not pull two clusters apart: share {bshare:.1%} of {bn} (the chain alone reaches 37.5%)")
    check(sorted(map(sorted, b_groups)) == [[0, 2, 4, 6], [1, 3, 5, 7]], f"... and the groups are the clusters ({b_groups})")
    check([len(g) for g in b_groups] == [4, 4], "... with the group sizes unchanged by the refinement")
    # refinement must never make a plan worse than the chain it started from, on a trace with no structure to find
    import random
    rng = random.Random(1702)
    noise = {0: [rng.sample(range(32), 4) for _ in range(300)]}
    counts = coactivation(noise[0], 32)
    chain_parts = group_of_plan(chain_order(counts, set(range(32))), 2)

    def inside(parts):
        return sum(counts[a][b] for part in parts for a in part for b in part)

    refined = build(noise, 2)["0"]["groups"]
    check(inside(refined) >= inside(chain_parts) and sorted(len(g) for g in refined) == [16, 16],
          f"on noise the refined plan holds at least the chain's co-activation ({inside(refined)} >= {inside(chain_parts)})")

    # In-sample is not held out. A layer whose clusters CHANGE half way: the table built from the first half is right
    # for the first half and wrong for the second, and held_out_share() must report the second, not the fit.
    shifted = {0: ([[0, 2, 4, 6]] * 25 + [[1, 3, 5, 7]] * 25) + ([[0, 1, 2, 3]] * 25 + [[4, 5, 6, 7]] * 25)}
    hs, hn = held_out_share(shifted, 2)
    check(hs == 0.0 and hn == 50, f"a table from the first half scores {hs:.0%} on a second half it never saw ({hn} steps)")
    steady = {0: ([[0, 2, 4, 6]] * 25 + [[1, 3, 5, 7]] * 25) * 2}
    hs2, _ = held_out_share(steady, 2)
    check(hs2 == 1.0, f"... and {hs2:.0%} when the second half fires the same clusters")

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

    # --score reads a table back by its schema, and refuses a file that is not one
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"schema": "1bit/moe-placement/1", "groups": 2, "layers": {"0": {"groups": [[0, 2, 4, 6], [1, 3, 5, 7]]}}}, f)
        good = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"schema": "something/else", "layers": {}}, f)
        bad = f.name
    try:
        rshare, _ = one_device_share(load_table(good), two)
        check(rshare == 1.0, f"a table read back from disk scores the trace it fits ({rshare:.0%})")
        try:
            load_table(bad)
            check(False, "a file with another schema is refused")
        except ValueError as e:
            check("something/else" in str(e), "a file with another schema is refused, naming the schema it has")
    finally:
        os.unlink(good)
        os.unlink(bad)

    print(f"\nmoe_placement self-test: {fails} failure(s)")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("-g", "--groups", type=int, default=2, help="device groups (default 2: HRX + NPU)")
    ap.add_argument("-o", "--out", help="write the JSON table here (default: stdout)")
    ap.add_argument("--min-pairs", type=int, default=1, help="skip a layer with fewer than N co-activating pairs")
    ap.add_argument("-e", "--experts", type=int,
                    help="the model's expert count per layer: the table then covers 0..N-1 and `unseen` says how many "
                         "were placed by balance alone. Default: the highest expert id the trace visits (a partial table)")
    ap.add_argument("--score", metavar="TABLE",
                    help="build nothing: score an existing table on this trace. A table scored on the trace it was "
                         "built from is in-sample; score it on another prompt's trace for what a consumer would see")
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
    if a.experts is not None and a.experts < 1:
        ap.error("--experts must be >= 1")
    if a.score:
        try:
            plan = load_table(a.score)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"moe_placement: cannot score {a.score}: {e}", file=sys.stderr)
            return 2
        missing = sorted(set(trace) - set(plan))
        if missing:
            print(f"moe_placement: {a.score} has no layer {missing[0]} ({len(missing)} of the trace's {len(trace)} "
                  f"layer(s) are not in the table): it was built for a different model or a shorter trace",
                  file=sys.stderr)
            return 2
        share, n = one_device_share(plan, trace)
        cshare, _ = one_device_share(contiguous(trace, len(next(iter(plan.values())))), trace)
        print(f"moe_placement: {a.score} on {a.trace}: one-device share {share:.1%} over {n} decode steps "
              f"(contiguous control {cshare:.1%})")
        return 0
    try:
        layers = build(trace, a.groups, a.min_pairs, a.experts)
    except ValueError as e:
        print(f"moe_placement: {e}", file=sys.stderr)
        return 2
    plan = {int(l): v["groups"] for l, v in layers.items()}
    share, n = one_device_share(plan, trace)
    hshare, hn = held_out_share(trace, a.groups, a.min_pairs, a.experts)
    # the two shares travel with the table, so a file found a week later says which number it was sold on
    table = {"schema": "1bit/moe-placement/1", "groups": a.groups, "layers": layers,
             "share_in_sample": round(share, 4), "share_held_out": round(hshare, 4), "held_out_steps": hn}
    text = json.dumps(table, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    cshare, _ = one_device_share(contiguous(trace, a.groups), trace)
    print(f"moe_placement: {len(table['layers'])} layer(s), {a.groups} group(s); one-device share {share:.1%} over {n} "
          f"decode steps IN-SAMPLE (contiguous control {cshare:.1%}); held out {hshare:.1%} over {hn} steps "
          f"(table from the first half of each layer's steps, scored on the second half)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
