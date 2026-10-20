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

# Cell A of the two-cell comparison: the divergence tracks a MEASURED non-coherent mapping

`315-two-cell-memory-type-preregistration.md` pre-registered the comparison and required
`GGML_HRX_BUFFER_TRACE=1` so the divergence is attached to a **measured** type. Cell A is now measured.

## The measurement

Run `315twin`, `-ub 512`, in-regime, **with the eviction forcer running** (`evictonly 500`) so a faulting
arm arrives in one, and with `GGML_HRX_TENSOR_FP=result_norm` + `GGML_HRX_BUFFER_TRACE=1`.

```
315twin-1   nan_guard = 1            (Face A: HRX returned NaN logits → FAULT)
            twin records  = 932
            divergences   = 128
            memcpy_stable=0 on 128 of 128
            measured host-visible memory_type = 0x52   (HOST_LOCAL|DEVICE_VISIBLE, NON-coherent)
            (device-local allocations measured 0x30 as expected)
```

**Every divergence is on a mapping measured `0x52`, and the arm faulted.**

## What this does and does not settle

It settles cell A's half of the pre-registration: the divergence is real, it is `memcpy_stable=0`
throughout — so no read-to-read agreement at any point — and it occurs on a memory type that is
**recorded in the same log**, not inferred from a comment. The rate is also higher than the earlier
ambient-rate control run (128 in 932 records against 75 in 669), which is what one expects if the
divergence is driven by cache pressure: this arm ran under the eviction forcer, which mprotect-churns host
memory.

It does **not** by itself establish that the memory type *causes* the divergence, because cell B — the
`0x56` configuration — has not run. `315-persistent-mapping-correction.md` is the reason the distinction
matters: `buffer_get` reads a **persistent mapping** created at allocation and never invalidated, so a
divergence between two plain reads is consistent with either the device having written the range or the
CPU's cached view of a non-coherent mapping having changed. Cell A cannot separate those; the two cells
together can, because a `0x56` mapping removes the second explanation.

**Also recorded, because it bounds the claim:** the eviction forcer is a **diagnostic** trigger (the record
measured it at 7/10 faults against 0/22 controls). Nothing here is an acceptance cell, and no arm in this
run may be counted toward the objective's ≥60 in-regime clean-arm campaign.

## Next

Cell B: apply the twin to the coherent tree (`llama-315coherent` / `315coherent-build`, which currently has
`hrx-f32-sync = 0`) and run the same configuration. The pre-registered readings in
`315-two-cell-memory-type-preregistration.md` then apply as written, including the case where cell B's
achieved type turns out not to be `0x56`, which voids the comparison and must be recorded as such.
