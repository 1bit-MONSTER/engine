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
"""coactivation_replay — does co-activation expert placement pay on Strix Halo?

Reads a tools/moe_trace.cpp trace and answers three questions with one measure
(within-layer co-activation P(e2 in top-k | e1 in top-k)):

  a) NVMe streaming: with co-activated experts adjacent in the file, how many reads
     per token replace the per-expert reads the current LRU does?
  b) a two-device split (HRX + NPU): what share of a token's experts lands on one
     device under a co-activation placement, against the contiguous split?
  c) batched prefill at ub=512: the wire rows a dedup-style dispatch sends with and
     without that placement.

Method (no engine code, replay only):
  * co-activation counts from the trace's own routes (unordered pairs within a token,
    self pairs dropped);
  * placement = balanced Kernighan-Lin partition into G equal groups, then a greedy
    max-affinity chain inside each group to fix the file order;
  * streaming read count = per layer LRU misses (prompt warms the cache, decode
    counts), then runs of consecutive file slots among one token's misses = one read;
  * everything is compared against the unplaced control on the same trace.

Usage: coact.py TRACE.tsv [--caps 64,128,192] [--groups 2] [--quiet]
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------- trace reading
def read_trace(path):
    """Return (dec, pre, n_exp, n_layer). dec: list of {(layer): [experts]}, pre: list of steps,
    each a list of tokens, each a dict layer -> [experts]."""
    dec, pre = [], []
    step_rows = defaultdict(dict)   # step -> layer -> [row, ...]
    kind_of = {}
    n_exp = 0
    with open(path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            step, layer, kind = int(cols[0]), int(cols[1]), cols[2]
            ex = [int(x) for x in cols[3].split(",") if x]
            n_exp = max([n_exp] + [e + 1 for e in ex])
            step_rows[step].setdefault(layer, []).append(ex)
            kind_of[step] = kind
    for step in sorted(step_rows):
        layers = step_rows[step]
        kind = kind_of[step]
        if kind == "D":
            # one row per layer == one decoded token
            dec.append({l: rows[0] for l, rows in layers.items()})
        else:
            # a prompt batch: n tokens, one row each per layer
            # layers can disagree on row count (a layer with fewer tokens in the batch, or a
            # partial ubatch); take every (layer, row) that exists rather than assume a rectangle
            n = max(len(rows) for rows in layers.values())
            for j in range(n):
                pre.append({l: rows[j] for l, rows in layers.items() if j < len(rows)})
    n_layer = max((l for t in dec for l in t), default=0) + 1
    return dec, pre, n_exp, n_layer


# ------------------------------------------------------------- co-activation
def coact_matrix(tokens, n_exp, layer):
    """Symmetric co-selection counts for one layer. tokens: iterable of {layer: [experts]}."""
    c = np.zeros((n_exp, n_exp), dtype=np.float64)
    for t in tokens:
        ex = t.get(layer)
        if not ex:
            continue
        for i in range(len(ex)):
            for j in range(i + 1, len(ex)):
                a, b = ex[i], ex[j]
                if a == b:
                    continue
                c[a, b] += 1.0
                c[b, a] += 1.0
    return c


def kl_partition(w, groups, max_iter=300, seed=0):
    """Balanced Kernighan-Lin: split nodes into `groups` equal-size blocks maximizing
    intra-block weight. One best swap per pass, sizes preserved; deterministic given seed."""
    n = w.shape[0]
    rng = random.Random(seed)
    assign = np.zeros(n, dtype=np.int64)
    order = list(range(n))
    rng.shuffle(order)
    per = n // groups
    for g in range(groups):
        for k in order[g * per:(g + 1) * per]:
            assign[k] = g
    if per * groups < n:
        assign[per * groups:] = groups - 1

    diag = np.diag(w).copy()
    for _ in range(max_iter):
        best = (1e-9, None, 0, 0)
        for ga in range(groups):
            na = np.where(assign == ga)[0]
            if na.size == 0:
                continue
            for gb in range(ga + 1, groups):
                nb = np.where(assign == gb)[0]
                if nb.size == 0:
                    continue
                wab = w[np.ix_(na, nb)]
                saa = w[np.ix_(na, na)].sum(axis=1)
                sbb = w[np.ix_(nb, nb)].sum(axis=1)
                da = wab.sum(axis=1) - (saa - diag[na])
                db = wab.sum(axis=0) - (sbb - diag[nb])
                g = da[:, None] + db[None, :] - 2.0 * wab
                i = int(np.argmax(g))
                v = float(g.flat[i])
                if v > best[0]:
                    best = (v, (int(na[i // nb.size]), int(nb[i % nb.size])), ga, gb)
        if best[1] is None:
            break
        a, b = best[1]
        assign[a], assign[b] = best[3], best[2]
    return assign, np.bincount(assign, minlength=groups)


def greedy_order(w, members):
    """Order one group's experts so that high-affinity pairs are adjacent (greedy path)."""
    members = list(members)
    if len(members) <= 2:
        return members
    sub = w[np.ix_(members, members)]
    deg = sub.sum(axis=1)
    start = int(np.argmax(deg))
    order = [start]
    unused = set(range(len(members))) - {start}
    while unused:
        cur = order[-1]
        nxt = max(unused, key=lambda j: sub[cur, j])
        order.append(nxt)
        unused.discard(nxt)
    return [members[i] for i in order]


def placement(w, groups):
    """Balanced groups -> one flat permutation: group blocks, greedy chain inside each."""
    assign, sizes = kl_partition(w, groups)
    perm = []
    for g in range(groups):
        members = np.where(assign == g)[0]
        perm.extend(greedy_order(w, members))
    slot_of = {e: i for i, e in enumerate(perm)}
    return perm, slot_of, assign


# ------------------------------------------------------------------ policies
def lru_misses(stream, cap, warm=()):
    """stream: list of (layer, experts) in access order. One LRU per layer, `cap` slots.
    `warm` is replayed first (the prompt) without being counted, as the engine's own replay does."""
    caches = defaultdict(list)
    misses = 0
    total = 0
    n_warm = len(warm)
    for i, (layer, ex) in enumerate(list(warm) + list(stream)):
        c = caches[layer]
        counting = i >= n_warm
        for e in ex:
            hit = e in c
            if hit:
                c.remove(e)
                c.append(e)
            else:
                if len(c) >= cap:
                    c.pop(0)
                c.append(e)
            if counting:
                total += 1
                if not hit:
                    misses += 1
    return misses, total


def reads_with_adjacency(stream, cap, slot_of, warm=()):
    """Same LRU, but a token's misses that are consecutive in file order are one read."""
    caches = defaultdict(list)
    reads = 0
    misses = 0
    slots_read = 0   # experts' worth of bytes actually pulled (spans include unneeded neighbours)
    for layer, ex in warm:
        c = caches[layer]
        for e in ex:
            if e in c:
                c.remove(e)
                c.append(e)
            else:
                if len(c) >= cap:
                    c.pop(0)
                c.append(e)
    for layer, ex in stream:
        c = caches[layer]
        miss_slots = []
        for e in ex:
            if e in c:
                c.remove(e)
                c.append(e)
            else:
                misses += 1
                if len(c) >= cap:
                    c.pop(0)
                c.append(e)
                miss_slots.append(slot_of[e])
        if miss_slots:
            miss_slots.sort()
            # group the sorted slots into runs; a run of [i..j] reads j-i+1 experts, not just |run|
            runs = []
            start = prev = miss_slots[0]
            for sl in miss_slots[1:]:
                if sl == prev + 1:
                    prev = sl
                else:
                    runs.append((start, prev))
                    start = prev = sl
            runs.append((start, prev))
            reads += len(runs)
            slots_read += sum(b - a + 1 for a, b in runs)
    return reads, misses, slots_read


def split_stats(tokens, assign, groups, layer):
    """For one layer: tokens whose experts all land in one group, and mean groups touched."""
    one = 0
    n = 0
    touched = 0
    maj_share = []
    for t in tokens:
        ex = t.get(layer)
        if not ex:
            continue
        gs = {int(assign[e]) for e in ex}
        n += 1
        touched += len(gs)
        if len(gs) == 1:
            one += 1
        cnt = defaultdict(int)
        for e in ex:
            cnt[int(assign[e])] += 1
        maj_share.append(max(cnt.values()) / len(ex))
    if not n:
        return None
    return dict(tokens=n, one_group=one / n, groups_per_token=touched / n,
                majority_share=float(np.mean(maj_share)))


def contiguous_assign(n_exp, groups):
    per = n_exp // groups
    a = np.zeros(n_exp, dtype=np.int64)
    for g in range(groups):
        a[g * per:(g + 1) * per] = g
    if per * groups < n_exp:
        a[per * groups:] = groups - 1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--caps", default="64,128,192")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--ub", type=int, default=512, help="prefill batch to report (caps at the trace's own)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dec, pre, n_exp, n_layer = read_trace(args.trace)
    if not dec:
        sys.exit("trace has no decode rows")
    caps = [int(c) for c in args.caps.split(",") if int(c) <= n_exp]
    print(f"trace {os.path.basename(args.trace)}: {len(pre)} prompt tokens, {len(dec)} decoded, "
          f"{n_layer} layers, {n_exp} experts")

    # co-activation from every captured route (prompt + decode): the offline build sees both
    all_tokens = pre + dec
    coacts = {l: coact_matrix(all_tokens, n_exp, l) for l in range(n_layer)}

    # one global placement over the summed matrix (what a single file layout can do),
    # and per-layer placements (what a per-layer layout could do)
    wsum = sum(coacts.values())
    perm, slot_of, assign = placement(wsum, args.groups)
    layer_plans = {l: placement(coacts[l] + 1e-9 * wsum, args.groups) for l in range(n_layer)}

    stream = []
    for tok in dec:
        for l in sorted(tok):
            stream.append((l, tok[l]))

    print("\n== (a) NVMe streaming: reads per decoded token")
    print(f"{'cap/layer':>9} {"LRU hit%":>9} {'reads/tok base':>14} {'reads/tok placed':>16} "
          f"{'read drop':>9} {'bytes/needed':>12}")
    warm = [(l, tok[l]) for tok in pre for l in sorted(tok)]
    by_layer = defaultdict(list)
    for l, ex in stream:
        by_layer[l].append((l, ex))
    for cap in caps:
        misses, total = lru_misses(stream, cap, warm)
        base_reads = misses
        placed_reads = 0
        placed_slots = 0
        for l in range(n_layer):
            r, _, sl = reads_with_adjacency(by_layer[l], cap, layer_plans[l][1],
                                            [(l, ex) for (l2, ex) in warm if l2 == l])
            placed_reads += r
            placed_slots += sl
        ntok = len(dec)
        drop = 1 - placed_reads / max(1, base_reads)
        amp = placed_slots / max(1, misses)
        print(f"{cap:>9} {100*(1-misses/total):>8.1f}% {base_reads/ntok:>14.2f} "
              f"{placed_reads/ntok:>16.2f} {100*drop:>8.1f}% {amp:>11.2f}")

    print("\n== (b) two-device split (HRX + NPU), share of a token whose experts land on one device")
    print(f"{'placement':>26} {'per-layer mean':>15} {'global plan':>12} {'groups/token':>13}")
    per_layer_one = []
    for l in range(n_layer):
        s = split_stats(all_tokens, layer_plans[l][2], args.groups, l)
        if s:
            per_layer_one.append(s["one_group"])
    s_global = [split_stats(all_tokens, assign, args.groups, l) for l in range(n_layer)]
    s_global = [s for s in s_global if s]
    one_global = float(np.mean([s["one_group"] for s in s_global])) if s_global else float("nan")
    gpt = float(np.mean([s["groups_per_token"] for s in s_global])) if s_global else float("nan")
    print(f"{'co-activation (per layer)':>26} {100*np.mean(per_layer_one):>14.1f}%")
    print(f"{'co-activation (global)':>26} {one_global*100:>14.1f}% {one_global*100:>11.1f}% {gpt:>13.2f}")
    contig = contiguous_assign(n_exp, args.groups)
    s_contig = [split_stats(all_tokens, contig, args.groups, l) for l in range(n_layer)]
    s_contig = [s for s in s_contig if s]
    one_c = float(np.mean([s["one_group"] for s in s_contig]))
    print(f"{'contiguous (control)':>26} {100*one_c:>14.1f}% {100*one_c:>11.1f}% "
          f"{float(np.mean([s['groups_per_token'] for s in s_contig])):>13.2f}")

    print("\n== (c) batched prefill: wire rows a dedup dispatch saves at one batch")
    if pre:
        n_tok = min(args.ub, len(pre))
        batch = pre[:n_tok]
        print(f"batch of {n_tok} prompt tokens (trace has {len(pre)})")
        tot_pairs = 0
        rows_contig = 0
        rows_placed = 0
        uniq = 0
        for l in range(n_layer):
            pairs = sum(len(t.get(l, [])) for t in batch)
            u = len({e for t in batch for e in t.get(l, [])})
            tot_pairs += pairs
            uniq += u
            for t in batch:
                ex = t.get(l, [])
                if ex:
                    rows_contig += len({int(contig[e]) for e in ex})
                    rows_placed += len({int(layer_plans[l][2][e]) for e in ex})
        print(f"{'pairs (token,expert)':>26} {tot_pairs:>10}")
        print(f"{'unique experts':>26} {uniq:>10}  ({100*(1-uniq/max(1,tot_pairs)):.1f}% fewer than pairs)")
        print(f"{'wire rows, EP=2 contiguous':>26} {rows_contig:>10}  ({100*(1-rows_contig/max(1,tot_pairs)):.1f}% of pairs)")
        print(f"{'wire rows, EP=2 co-active/layer':>26} {rows_placed:>10}  "
              f"({100*(1-rows_placed/max(1,rows_contig)):.1f}% under contiguous)")

    # same for the decode stream, which is what a served token actually pays
    if dec:
        rows_dec_contig = rows_dec_placed = 0
        pairs_dec = 0
        for tok in dec:
            for l, ex in tok.items():
                pairs_dec += len(ex)
                rows_dec_contig += len({int(contig[e]) for e in ex})
                rows_dec_placed += len({int(layer_plans[l][2][e]) for e in ex})
        print(f"decode: {pairs_dec} (token,expert) pairs -> {rows_dec_contig} rows contiguous "
              f"({100*(1-rows_dec_contig/pairs_dec):.1f}% saved) vs {rows_dec_placed} placed "
              f"({100*(1-rows_dec_placed/pairs_dec):.1f}% saved)")


if __name__ == "__main__":
    main()
