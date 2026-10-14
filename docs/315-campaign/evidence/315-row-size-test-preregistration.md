<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: test the fixed-size / moving-position prediction

Written **before** the run.

## The prediction under test

From `315-corruption-is-one-row-sized.md`, the hypothesis is:

> Under forced KFD eviction, **one hidden-state row's worth of bytes (~8 KiB) is corrupted at a position that varies
> between episodes**.

It has four supporting arms (7,936 / 7,936 / 8,192 bytes in three eviction arms at layers 0, 0, 1, plus a CPU arm
at 184 bytes that does not share it) and **no replication outside that session**. This run is the replication.

**Falsified by**: first-corruption sizes that are not ~8 KiB, or a size that varies while the position is fixed.

## The run

    BIN=~/wt/315fp-build/bin
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 4 315row

`tail:16` is the configuration with the **highest recorded Face-A yield** — 16 faults in 17 arms in `315nan`, against
1-in-3 for the name-list configs this session used. It matters here because the measurement needs many **Face-A**
arms, and the reduced name list produced three Face-B arms to every Face-A one.

**`l_out-<layer>` is inside every `tail:16` window** (it is the last node of its chunk), so the window reports the
first NaN-bearing layer *and* its count without a name list. The count in f32 × 4 gives the byte size directly,
which is the quantity the prediction is about.

## Readout, fixed in advance

For each **in-regime Face-A** arm (`gtt_start ∈ [95,110]`, `nan≥1`), take the **earliest `l_out-<N>` line in node
order** and record `N` and `nan × 4` bytes.

| observation across ≥5 in-regime Face-A arms | verdict |
|---|---|
| bytes cluster at ~8,192 B (±3 %, i.e. 7,936–8,448) **while `N` differs** | **prediction confirmed** — fixed size, moving position |
| bytes vary by more than an order of magnitude with `N` varying too | prediction **falsified as stated**; the fixed-size reading was a coincidence of one session |
| `N` is the same in every arm | the moving-position half is **falsified**; the site is fixed and §40's reading is back |
| fewer than 5 in-regime Face-A arms result | **INVALID for the prediction** — reported as such, never as confirmation |

Arms that fault as Face B, or fall outside the gate, are **not counted** in either direction and are listed
separately.

## Confounds, declared before the run

- **`tail:16` is the strongest inducer on record** (§39, 16/17). No rate is quoted from this run and no arm is an
  acceptance arm; the yield is being *used*, not measured.
- **The window's own head is displacement-exposed** — the earliest node of a `tail:16` window is the node most
  exposed to reuse by the window's later nodes. The earlier `315row` arm's first NaN-bearing stage was exactly such
  a node. So the count of the *earliest* line is the least trustworthy reading in each arm, which is precisely the
  quantity the prediction uses. **This is a real weakness of the test and is stated rather than buried**: it
  predicts a size, and the size is read from the least faithful line available. A confirmation would therefore be
  stronger than a refutation.
- **The temporal caveat from `315-origin-varies-reinstated.md` stands** — a live-to-end tensor can still be
  overwritten if the next chunk has run before the read. The predicted sizes are ~8 KiB; a systematically lags-by-
  one-chunk read would give a *later* layer's count, which for these arms is larger, so lag would inflate rather
  than manufacture the 8 KiB cluster.
- **Four eviction arms is not a rate.** The prediction is about size, not frequency.
