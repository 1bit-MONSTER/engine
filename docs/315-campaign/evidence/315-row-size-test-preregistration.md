<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
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

---

# Amendment (before the second attempt): the `tail:16` yield did not reproduce, so switch to the config that has

The first attempt is heading for the outcome the readout table already calls **INVALID**: after 4 of 8 arms,

    round 1 arm q: OK nan=0 req=66
    round 1 arm e: OK nan=0 req=66     <- the EVICTION cell, clean
    round 1 arm c: OK nan=0 req=66
    round 2 arm q: OK nan=0 req=66

**The eviction cell produced no fault at all**, where `315nan`'s recorded 16/17 came from the same `EXTRA_ENV`. So
the highest-yield configuration on record did not reproduce its own yield under this rig, and a run that cannot
produce Face-A arms cannot test a prediction about them. **This is reported as a failed attempt, not as a negative.**

**Second attempt: the configuration that has actually produced Face-A arms in this session.**

| configuration | rig | eviction-cell outcome |
|---|---|---|
| `315nan` (recorded) | `315-paired.sh`, two builds | 16/17 Face A |
| `315row2` (this attempt) | `315-disc.sh`, `tail:16` + probes | **0/1 — clean** |
| `315lout4` | `315-disc.sh`, 16 `l_out` names, no probes | **Face A ×2** (`r1e`, `r2e`), plus `r1c` Face A |
| `315inp2` | `315-disc.sh`, 10 `l_out`/`ffn_inp` names | Face B ×2, Face A ×1 |

The distinguishing feature of the configs that yield Face A is **the name list without the extra probes**, not
`tail:16`. Note also the dose observation already recorded: 64 names → Face B everywhere, 16 names → clean quiet
cell and Face A under eviction.

**The selector** (15 names, `l_out` only, no `ffn_inp`, no `COPY_PROBE`/`SENTINEL`/`LOGITS_FP`):

    l_out-0,l_out-1,l_out-2,l_out-3,l_out-4,l_out-10,l_out-15,l_out-20,l_out-25,l_out-30,\
    l_out-35,l_out-40,l_out-45,l_out-46,l_out-47

**Consecutive early layers 0–4**, which is what the size measurement needs — the first attempt could not have
distinguished a layer-1 origin from a layer-0 one even had it faulted — plus a spread to the tail at roughly the
`lout4` dose. **The readout table is unchanged**, including its INVALID clause.

**One thing this attempt cannot do that the first could:** `tail:16` covered *every* node in each chunk's last 16,
so it could have named a node the name list misses. The name list can only report the layers it names. Stated so the
substitution is not silently a narrowing.
