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

# At the first corrupted tensor the NaN is one row at the FRONT of the buffer (row 0 or 1), never the last token

Read from the archived extended-fingerprint lines. No GPU time. This is the most specific positional result
the record has, and the method is stated because the naive version of it is confounded.

## The confound that had to be removed first

The fingerprint reports `first=` and `row=` = the **first** NaN's position, so a *spread* corruption biases
`row` toward 0 — reporting "row 0" would just mean "there is some NaN near the start". A raw census over all
1,525 multi-token row-field readings therefore cannot support any positional claim, and the first version of
this analysis (1,396 of 1,525 at row 0) was exactly that artifact.

The unbiased subset is the readings where **`nan == ne0`, i.e. exactly one row is NaN**: there the single
corrupted row *is* the first one, so `row` is its true position.

## The unbiased result

    12 readings, 4 arms, all ne0 = 2048 (one hidden-state row), T = 39, 172, 348

| quantity | value |
|---|---|
| rows observed | **{0, 1}** — nothing else |
| `row == 0` | **9 of 12** |
| `row == 1` | 3 of 12 |
| `row == T-1` (last token) | **0 of 12** |
| `row / T` | min 0.000, median 0.000, **max 0.003** |

    T=  39 ne0=2048 row=0 row/T=0.00  315fbar-p1c   (x3)
    T= 172 ne0=2048 row=0 row/T=0.00  315fbar-p2c   (x3)
    T= 348 ne0=2048 row=1 row/T=0.00  315fbar-p2l   (x3)
    T=  39 ne0=2048 row=0 row/T=0.00  315fbar-p4c   (x3)

So when the corruption is one row — which is the whole basis of the size finding — **that row is at the very
front of the buffer**, at three different sequence lengths and in four arms. It is **never** the last token,
which is the reading a per-decode-step mechanism would predict.

## The contrast that makes it a *mechanism* sketch rather than a curiosity

Across all 1,525 multi-token readings, the corrupted extent expressed in rows (`nan / ne0`) has
**median 348 rows** and maximum ~9,280; only **12 of 1,525** are exactly one row. So one row is the **onset**
reading, not the typical one. That is exactly what propagation from a front row predicts:

- corruption enters one row at the front at the first corrupted tensor (sometimes reported as row 1, so at
  the first or second token);
- causal attention lets every later token attend to that row, so the *next* tensor carries NaNs across
  hundreds of rows;
- by the head, every one of the 154,880 logits is NaN — Face A.

**Stated as a falsifiable hypothesis:** the corruption enters as a single hidden-state row at the front of
the buffer (token 0 or 1) and spreads forward through causally-masked attention, so the first corrupted
tensor shows one row anywhere in `{0, 1}` and the following tensors show NaN in every row after it.
Falsified by a one-row reading at a high row index, or by a first corrupted tensor whose NaN does not include
row 0 or 1, or by a downstream tensor with fewer rows corrupted than the onset one.

## What this does NOT establish

- **Not a cause.** It says where the corruption is, not what writes it. The producer probe — the one
  instrument that asks whether the device wrote the region at all — is pre-registered and queued, and this
  finding is what makes its question sharper: the region it fills is the **logits** region, whereas this
  result is about a **layer residual**, so the two are not the same buffer.
- **Not instrument-independent.** All 12 readings are from the `tail:16` + probes configuration
  (`315fbar`), which is the configuration whose onset layer is 41 and whose site moves with the read pattern.
  The front-of-buffer position must be re-measured under a name-list config before being called universal —
  the same caveat that applies to the size.
- **Not a row index in token space for every arm.** `ne0 = 2048` is the hidden width, so `row` is a token
  index only if the tensor's second dimension is the token axis; that holds for the residual stream but was
  not verified per tensor.
- **Small n.** 12 readings in 4 arms. The *strength* of the result is that all 12 agree and 0 of 12 sit at
  `T-1`; the weakness is that 12 is not many, and the four arms share a configuration and a rig.

## Reproducing

    grep -ahE "name=l_out-[0-9]+ |name=node_[0-9]+ " ~/wt/rt-det/*/server.log \
      | grep -aE "ne0=[0-9]+ first=[0-9]+ row=[0-9]+" \
      | awk '{ ... nan == ne0 ... }'   # the filter is the whole method

`dev/315/315-campaign/tools/row-position-census-by-family.py` prints the per-family and per-arm cut;
the `nan == ne0` filter is what makes the numbers above unbiased.
