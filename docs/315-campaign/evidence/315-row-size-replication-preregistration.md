<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: replicate the row-size prediction to the ≥5-arm threshold (315row2)

Written **before** the run, and before `315row` (4 rounds, tag `315row`, launched 2026-10-13T22:07:15)
has finished — so this is a genuine pre-registration and not a response to its outcome.

## Why a second run is pre-authorised

`315-row-size-test-preregistration.md` fixes its own readout: the prediction is **INVALID for the
prediction** unless **≥5 in-regime Face-A arms** result. The `315row` shape is **4 rounds × 3 cells =
12 arms, of which exactly 4 are eviction-cell arms** (`315-disc.sh` runs one `e` cell per round), and the
Face-A yield is a property of the eviction cell. So even at the record's **instrument-inflated 94 %
(16/17 in `315nan`)** the *expected* Face-A arm count from `315row` is **~3.8 — below the threshold.**

That is a defect in the first pre-registration's run shape, not in its prediction, and it is recorded here
rather than silently patched: the arithmetic above was available before `315row` launched and was not run.

## The run

Identical configuration to `315row`, differing **only** in the round count and the tag:

    BIN=~/wt/315fp-build/bin
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 8 315row2

**8 rounds ⇒ 8 eviction-cell arms.** At the recorded 94 % eviction-cell Face-A rate that is ~7.5 expected
Face-A arms, i.e. above the threshold with margin; at the *ambient* ~5 % rate it is ~0.4 and the run will
again be INVALID for the prediction, which is itself informative (it would mean `tail:16`'s yield was
session-specific, as `315-face-landscape.md` already suspects).

## This is not arm-shopping

The prediction, the readout, the byte tolerance and the three verdict branches are **unchanged** from
`315-row-size-test-preregistration.md`; only the number of eviction-cell arms is increased, and it is
increased **before** any `315row` result exists. `315row`'s Face-A arms are **pooled with** `315row2`'s when
the size/position table is computed — the first run's arms are not discarded. If the two runs disagree on
size or on the layer sequence, that disagreement is the finding and is reported as such.

## Readout (unchanged, restated so it cannot drift)

For each in-regime Face-A arm (`gtt_start ∈ [95,110]`, `nan≥1`), take the earliest `l_out-<N>` line in node
order; record `N` and `nan × 4` bytes.

| pooled observation across ≥5 in-regime Face-A arms | verdict |
|---|---|
| bytes cluster at ~8192 B (7936–8448) **while `N` differs** | prediction **confirmed** |
| bytes vary by >1 order of magnitude with `N` varying | prediction **falsified as stated** |
| `N` identical in every arm | moving-position half **falsified** (site fixed) |
| still <5 in-regime Face-A arms pooled | **INVALID for the prediction**; reported, never as confirmation |

## Confounds (carried, plus one new)

All confounds of the first pre-registration stand unchanged — instrument inflation, the `tail:16` window's
displacement-exposed head, the temporal caveat, and that four eviction arms are not a rate. One addition:
**pooling two runs across a session boundary** risks a session-specific reading masquerading as replication,
so the per-run table is printed **alongside** the pooled one and any run-to-run disagreement is stated.

No criterion changes.

---

## Placement note, written after `315row`'s round-1 arms but before its later rounds

`315row`'s round 1 read **arm q: OK** and **arm e: OK** — both `66/66` requests, `nan=0`, `parse=0`,
`utf8=0`, `http=0`, with the eviction arm at `gtt_start=98G` **in regime** and `side=1/1` (its forcer alive
for the whole arm, so the arm belongs to its cell). Neither faulted.

Two consequences, recorded here rather than after the run ends:

1. **The ≥5 threshold cannot be met by `315row` alone.** The run shape is 4 rounds × 1 eviction cell = **4
   eviction-cell arms**, so the maximum Face-A arm count it can produce is 4 — below the 5 the prediction
   requires. This run is therefore **INVALID for the prediction by construction**, whatever rounds 2–4 do,
   which is exactly why `315row2` was pre-registered above.
2. **A clean eviction arm is worth noting.** The record's `315nan` figure was 16/17 (94 %) faults in this
   cell under this config, so one clean eviction arm is a data point against a near-certain rate — but it is
   **one arm**, and the pool it would be compared against is instrument-inflated (`315-face-landscape.md`),
   so it is recorded as an observation and **not** as a rate, a contradiction, or a trend.

No criterion changes. Rounds 2–4 continue; their eviction arms are still the only arms that can produce
Face-A, and their results are pooled into the row-size table alongside `315row2`'s regardless of outcome.
