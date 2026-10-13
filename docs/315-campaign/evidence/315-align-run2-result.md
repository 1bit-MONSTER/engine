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

# `315align` take 2 — six arms, all clean; the compaction ingredient was silently absent

Ran 15:28:54–15:41:48 with take 1's gaps closed in three respects and **one of them silently failed**. Both
the result and the defect are recorded, and the defect bounds what the result can support.

## The result

    arm a4096-e  align=4096 cell=e  180 requests  distinct=1  nan=0   gtt 99 GiB
    arm a256-e   align=256  cell=e  180 requests  distinct=1  nan=0   gtt 99 GiB
    arm a4096-q  align=4096 cell=q  180 requests  distinct=1  nan=0   gtt 81 GiB
    arm a256-q   align=256  cell=q  180 requests  distinct=1  nan=0   gtt 81 GiB
    arm b256-e   align=256  cell=e  180 requests  distinct=1  nan=0   gtt 81 GiB   <- repeat
    arm b4096-e  align=4096 cell=e  176 requests  distinct=1  nan=0   gtt 81 GiB   <- repeat

**Every arm: 180 (resp. 176) responses, all identical, zero NaN.** Including `align=256` in the eviction
cell, run twice.

## What take 2 fixed over take 1

| | take 1 | take 2 |
|---|---|---|
| GTT | 47 GiB | **99 GiB** during the cell-`e` arms (3 GLM neighbours composed first) |
| request shape | 120 short one-word messages | **180 × `/completion`, `cache_prompt=false`, `n_predict=96`** — the arms' own shape |
| cell `e` side load | `evictonly` alone | `evictonly 500` **+ a compaction loop** |
| arms | 4 | 6, with the 4096/256 cell-`e` contrast repeated |
| duration | 38 s | ~13 min |

## The defect: the compaction loop never wrote anything

`/proc/sys/vm/compact_memory` was **not writable** during the run (`compact_available=no`), and my loop was:

    while :; do echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || \
                 sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1; sleep 2; done

The first path fails (root-owned file). The second runs `sudo -n tee` **with no input on stdin**, so `tee`
writes an empty line rather than `1` — i.e. it never compacts anything. `sudo -n` itself works on this box
(verified afterwards: `sudo -n sh -c "echo 1 > /proc/sys/vm/compact_memory"` **succeeds**), so this was a
harness bug, not a permission limit.

**Consequence:** the prior #140 recipe's *second* ingredient was absent. That recipe recorded 1 fault in 5
runs at `align=256` **with** compaction, and explicitly noted the one faulty run coincided with foreign load
8-12 against clean repeats at 4-7 — so compaction and contention are not obviously optional.

## What the run does and does not support

- **Does support**: at `align=256` versus `align=4096`, in the arms' own request shape, at GTT 99 GiB, with
  the eviction forcer active, and over two repeats of the cell-`e` contrast — **the wrong-output signature
  did not appear in either** (all 180 responses identical, `nan=0`).
- **Does not support** retiring the FA path. The pre-registration's branch for "both clean" presumes the rig
  could have induced the event, and one of the two ingredients the only prior positive run used was missing.
  So this is **again closer to uninformative than to a null**, though it is a much stronger attempt than
  take 1: the shape, the GTT band and the forcer were all right.

## What a decisive version needs

Exactly one fix: make the compaction path actually write, e.g.

    while :; do sudo -n sh -c 'echo 1 > /proc/sys/vm/compact_memory'; sleep 2; done

and keep everything else from take 2. Given `sudo -n` works, that is a one-line change and a re-run of the
same six arms (~13 min).

## Bookkeeping

The harness bug is recorded here rather than silently patched, and the take-1 file's "what a decisive
version needs" list is now superseded by this one — the remaining gap is the compaction write, not the
regime, the shape or the arm count.

---

# Take 3 launched with the compaction write verified

The one-line fix from the section above was applied and the run launched at 15:48.

    start_comp(){ ... echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || \
                      sudo -n sh -c "echo 1 > /proc/sys/vm/compact_memory" >/dev/null 2>&1; ... }

**Verified before launch and again during the run**, rather than assumed:
`sudo -n sh -c "echo 1 > /proc/sys/vm/compact_memory"` returns **exit 0** — i.e. compaction is actually
triggered — and the loop process is live (`pgrep -f compact_memory` → the `while :` shell). GTT at the
launch was 98 GiB with 5 GLM neighbours resident.

Everything else is take 2's design unchanged: `align` 4096 vs 256, cells `q` and `e`, `evictonly 500`,
180 × `/completion` with `cache_prompt=false` and `n_predict=96` (the arms' own shape), and the cell-`e`
contrast repeated — six arms, ~13 min.

Result pending; recorded here so the launch conditions are durable rather than reconstructed later.
