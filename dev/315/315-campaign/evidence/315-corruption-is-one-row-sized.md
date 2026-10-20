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

# The corruption is one hidden-state row's worth of bytes, at a position that moves

## The observation, in bytes rather than counts

The counts in `315-origin-varies-reinstated.md` look like different numbers until they are converted to bytes.
Every tensor here is 2048 wide (`ne0 = 2048`), so one row is `2048 × 4 = 8192` bytes:

| arm | cell | in regime | first corruption | **first bytes** | layer |
|---|---|---|---|---|---|
| `315lout4` r1e | eviction | 98G | `l_out-0` = 1,984 | **7,936** | 0 |
| `315lout4` r2e | eviction | 98G | `l_out-0` = 1,984 | **7,936** | 0 |
| `315inp2` r3e | eviction | 98G | `l_out-1` = 2,048 | **8,192** | 1 |
| `315lout4` r1c | CPU | 98G | `l_out-0` = 46 | 184 | 0 |

**Under the eviction trigger the first corrupted region is one row, to within 3 %** — 7,936, 7,936 and 8,192
bytes, in arms whose first-corrupted *layer* differs (0, 0, 1) and whose token counts differ (98, 135, 98).

And `7,936 = 8,192 − 256`, with **256 the alignment the record gives for the MoE tables**
(`dispatch-mul-mat-id-common.h:311-317`). So the two `l_out-0` arms are one row minus exactly one 256-byte block,
and the `l_out-1` arm is one full row.

The CPU-pressure arm does not share the size (184 bytes) — so **the size is a property of the eviction path**, the
same way the layer-0 signature was earlier.

## What this makes coherent

Three things that had been separate now line up:

1. **The size is fixed** (~one row) while **the position varies** (layer 0 in three arms, layer 1 in a fourth).
   A fixed-size, variably-placed corruption is what a **region-oriented** event produces — and the intervention is
   KFD eviction, which acts on memory regions, not on ops or layers.
2. **The row is the natural unit of the affected region.** One hidden-state row is exactly what a single
   row-oriented write moves: a `get_rows` result, or **one KV-cache slot's row** for one token. §40's own timing
   note already points at a slot — *"the first decoded token after `slot launch_slot_`"*.
3. **The spread then follows causally-masked attention.** A single corrupted row at position *k* is attended to by
   positions **≥ k** and by no earlier one, so the NaN fraction after the next attention layer should be about
   `(n − k)/n`. The observed growth is of that shape in every arm, and the implied *k* is consistent between
   consecutive layers **within** an arm (`r1e`: 34.7 % then 36.7 % ⇒ k ≈ 64 then ≈ 62; `r2e`: 5.2 % then 5.9 % ⇒
   k ≈ 128 then ≈ 127) while differing **between** arms. **The position moves; the size does not.**

## The hypothesis, stated so it can be wrong

> Under forced KFD eviction, **one hidden-state row's worth of bytes (~8 KiB) is corrupted at a position that varies
> between episodes**, and the all-NaN logits are that single row propagating through causally-masked attention to
> the end of the sequence and then through the head — Face A when the corruption is NaN, Face B when it is finite
> but wrong.

**Falsified by**: measuring the first corruption in ≥5 further in-regime Face-A arms and finding sizes that are not
~8 KiB, or finding the size varies while the position is fixed.

## What is *not* claimed, and the caveat that has to travel with it

- **Four arms, one session, and the same run supplied two of the identical numbers** — the withdrawal of the
  `1984` "constant" two commits ago applies here too. What is new is that the *bytes* agree while the *layer* does
  not, which is a different claim from "1984 is a constant" and is not yet replicated outside this session.
- **The temporal faithfulness caveat stands**: `ffn_inp-3` reading *fewer* NaN than `l_out-2` is impossible by
  propagation, so at least one reading in this session is not the graph's value. The per-arm spread curve and the
  byte sizes are therefore reliable as a *scale*, not as exact figures.
- **The KV-slot reading is a candidate, not a finding.** Nothing here distinguishes "one row of the residual
  stream" from "one KV slot's row" from "one row of some other row-oriented buffer"; the observations constrain the
  **size and motion**, not the identity of the buffer.
