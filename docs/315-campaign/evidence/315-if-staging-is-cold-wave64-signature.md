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

# If the staging path is cold: a 64-lane (wave64) granular signature, and what it would implicate

Two results from checking the archive while the A′/B′ campaign waits. The first is a measurement; the second
is a hypothesis with a number in it, and it is written down so it can be tested rather than admired.

## 1. The staging writeback path has never been observed being exercised

Checked across the whole record rather than taken from the earlier prose:

| search | result |
|---|---|
| `[hrx-wb-fp]` lines in any `~/wt/rt-det/*/server.log` | **0 files, 0 lines** |
| arms whose console sets `GGML_HRX_WRITEBACK_FINGERPRINT` | **none** |
| `host_staging` / `host-staging` mentions in any server log | **0** |

So the record's "**0 `[hrx-wb-fp]` records with the flag on over a complete 66/66 arm**" is the strongest
statement available, and it is a statement about an **instrument that produced nothing** — there is no arm on
this record in which the staging writeback has been *seen* to run. That is exactly the gap the A′/B′ amendment
closes: both arms now set the flag, so the campaign will produce the first evidence either way.

**Why it matters for the candidate:** the chain in
`315-candidate-host-staging-coherence.md` requires the staging path to be **warm**. If the A′/B′ arms show no
`[hrx-wb-fp]` lines, the chain has no first step in this configuration, the null is **structurally expected**,
and the candidate is dead for the disc rig. The amendment's second verdict row already says so.

## 2. If it is cold, the surviving signature has a 64-lane shape

Strip the candidate away and the measurements that stand are: **one `n_embd`-wide row**, at **row 0 or 1**
(12 of 12 unbiased readings), and the count being either `2048` or **`1984 = 2048 − 64`**.

That `− 64` has a natural reading on this hardware. **64 is the width of a wave64**, which is the wavefront
size on GFX11/RDNA, and the record already carries a wave64 defect of exactly this class:

> `third_party/hrx-system` … plus the **GFX11 wave64 lane-mask drain** backport (**#313 NaN**)

So the recurring `2048 − 64` is one hidden-state row **minus exactly one wave**, and the arithmetic is exact
rather than approximate:

    2048 = 32 × 64        a full row is 32 waves
    1984 = 31 × 64        the reading is 31 waves, and 1984 % 64 == 0 exactly
    2048 − 1984 = 64       one wave
    64 × 4 = 256 bytes    the same 256-byte block the earlier row-size work kept meeting

Every `nan=1984` reading on the record fits this: `1984 of 276480`, `1984 of 352256`, `1984 of 503808`,
`1984 of 200704` — and each extent divides exactly by 2048 (135, 172, 246, 98 rows), so the row width is
confirmed by divisibility rather than assumed. **One row of 32 waves, of which 31 are corrupted and exactly
one is not.**

That is a **wavefront-granular** corruption, not a byte-granular one. Combined with the position result (row 0
or 1, i.e. the **front** of the buffer, and the op wrote the rest correctly), the shape is:

> a row-oriented device write in which **31 of the row's 32 wave64s** hold the wrong value and one wave is
> left clean, at the **start** of the buffer.

Front-correct-rest-wrong would be a partial write; **front-wrong-rest-correct with a one-wave hole** is the
other way round, and points at lane masking — a wave's lanes not being written or not being drained — rather
than at a short write.

**Falsifiable, and cheaply:** the #313 backport is **already in the pin** (`98d05d94a9` carries it), so this is
*not* #313 itself — but it would predict (a) the corrupted elements to be **confined to whole waves** within the
row, one wave being clean, (b) the count to be `2048 − 64k` for small `k` (with `k = 1` the observed case),
and (c) a **lane-mask drain** visible when the arm is run with the wave64 drain path instrumented. (a) and (b)
are checkable against the existing `first=`/`row=` fields plus a per-element dump; (c) needs a run.

**Not claimed:** that this is the mechanism. It is a second candidate that the *number* suggests, recorded
before the A′/B′ result so that a cold result sends this in as a pre-registered test rather than a rescue
attempt. And if the A′/B′ shows the staging path is **warm** and B′ moves the rate, this becomes moot.
