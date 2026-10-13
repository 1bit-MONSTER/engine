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

# The id probe's first run: the ids are CLEAN, so the argsort candidate dies

The pre-registered readout (`315-idproducing-kernel.md`, branches fixed in advance), run for the first time
on 2026-10-13 at ~19:12, 4 pairs, instrument `ffn_moe_topk/argsort/weights/probs` anchored per layer 0..63.

## The faulting arm, and the control

    pair 1 arm l: FAULT nan=1 (Face A)  gtt_start=98G  req=14   <- in-regime, genuinely faulted
    pair 1 arm c: clean
    pair 2 arm l: clean

so the comparison is between a real faulting arm and clean arms from the same session.

## The readout

**`oor` (ids outside `[0, ne0)`) is ZERO in every line of every arm.** No out-of-range id was produced.

`ffn_moe_argsort` — the tensor that *must* be a clean permutation, since an argsort assigns each rank once —
by `dup_pairs`:

    dup_pairs    p1l (FAULT, nan=2)     p1c (clean, nan=0)
        0              18,719                 99,802
     2016              16,257                 16,257
        1               1,128                  9,786
        2                 145                    308
    50400                 156                    156

**The non-zero values appear identically in both arms** — `2016` exactly 16,257 times in each, `50400`
exactly 156 times in each. Those are baseline behaviour of the argsort, not a fault signature. The faulting
arm simply has fewer total lines (36,405 vs 126,309) because it **died early** after 2 NaN events, which is
what makes its small values look proportionally different.

`ffn_moe_topk` shows the same shape: `dup_pairs=6` at 533 occurrences per layer in **both** arms — topk
repeats experts normally, so that is not a signal either.

## The pre-registered reading

The first branch was *"`oor > 0` or `dup_pairs > 0` in a faulting arm → the id list is not a clean
permutation: the argsort's rank computation is the site"*. **It does not fire**: `oor = 0` always, and the
`dup_pairs` profile is the same in faulting and clean arms.

So the second branch applies:

> **ids clean while the matmuls are NaN → the argsort candidate dies, and the fault is downstream, in the
> table builders or in the matmul's descriptor use.**

**The id-producing/argsort family is therefore excluded as the site of the stale value.** What the fault
consumes is not a corrupted route id.

## What this leaves in scope

Downstream of the argsort, on the routed-FFN path: the **table builders** (`ffn_moe_weights` /
`ffn_moe_probs` construction) and the **matmul's descriptor use**. Both are engine-side and in scope, and
neither has been instrumented to the same depth as the ids now have.

## Caveats, stated rather than glossed

- **n is small**: one faulting arm (`nan=1`, `req=14`) against clean arms from the same session. The
  conclusion is a **one-sided exclusion** (the argsort did not produce bad ids in a faulting arm), not a
  measured null.
- The `dup_pairs` metric counts repeats **within a row**; a row is a token's expert selection. That is the
  right shape for a route id, but the metric would not catch a *swap* (two ids exchanged, still a valid
  permutation) — a distinction worth noting for anyone reusing this instrument.
- The probe is diagnostic: no criterion changes and no arm from it is eligible.
