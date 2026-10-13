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

# Pre-registration: the sentinel test — did the device ever write the logits?

Written **before** the run. Frozen rig unchanged except `EXTRA_ENV`.

## The question this settles

Every Face-A fault in the record ends the same way:

    common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879],
                           contiguous=yes, continuing=no sentinel=0 (engine#315)

**All 154,880 logits NaN, contiguous.** Two mechanisms produce that exact line, and the campaign has never
separated them:

1. the head **computed** NaNs and wrote them;
2. the head (or its producer) **never wrote that region**, and what is being read is stale or unfilled memory.

The engine already ships the discriminator, and it has never been switched on. `ggml-hrx.cpp:384`:

    // engine#315 sentinel (GGML_HRX_SENTINEL): pre-fill the host-visible logits region with a payload
    // NaN (0x7FC0DEAD) so a faulting arm can distinguish "the device never wrote this region" from
    // "the device wrote NaNs". Only the 619520-byte (154880 x f32) logits allocation is touched, and
    // only when the env var is set, so a control arm without it is unaffected.
    if (host_visible && base != nullptr && size == 619520 && environment_flag_enabled("GGML_HRX_SENTINEL")) {
        uint32_t * words = reinterpret_cast<uint32_t *>(base);
        for (size_t i = 0; i < size / sizeof(uint32_t); ++i) { words[i] = 0x7FC0DEADu; }
        GGML_LOG_WARN("[hrx-sentinel] filled host-visible logits region size=%zu with 0x7FC0DEAD\n", size);

and the sampler guard already reports the match count as its `sentinel=` field. **Every recorded fault line
carries `sentinel=0` — but no recorded run ever set `GGML_HRX_SENTINEL`, so nothing filled that region and the
field has never meant anything.** It is a discriminator that was built, wired into the guard, and never armed.

## Readout, fixed in advance

With `GGML_HRX_SENTINEL=1`, in a faulting arm (`count=154880 of 154880`):

| `sentinel=` | meaning |
|---|---|
| **≈ 154880** | **the region was never written by the device.** The NaN being read is the sentinel payload itself, not a computation. Mechanism = the head's output never landed in that buffer. |
| **0** | the device **did** write the region, and wrote NaNs. Mechanism = a NaN computed upstream of the logits. |
| **0 < s < 154880** | mixed: part of the region was written, part was not. |

## Required liveness evidence, stated before the run

The fill happens **at allocation**. A `sentinel=0` reading is therefore only decisive if that buffer instance
was actually filled for the arm in question — so the run must show the engine's own
`[hrx-sentinel] filled host-visible logits region size=619520` line **before** the fault. Without that line, the
arm is **INVALID**, not a pass: an unfilled buffer and a device-written buffer are indistinguishable.

## Confounds, declared

- The sentinel fill is an intervention: it writes 619,520 bytes of host memory at allocation and changes what a
  never-written region contains. It can only affect arms that were already going to read unwritten memory.
- The fill takes the payload NaN, so it makes a *previously silent* failure mode **loud** rather than quiet — it
  cannot manufacture a fault, but it can change an arm's classification from "clean" to "fault" if the region
  was never written and previously held non-NaN stale data. Any such arm is recorded, not hidden.
- If the buffer is reallocated per request, later requests may not carry the fill; per-request reach is recorded.

No criterion changes. No arm from this is an acceptance arm.

---

# CORRECTION, before running anything: the sentinel was already armed — and the answer is in the record

**This pre-registration is wrong on its central premise.** It states the sentinel "has never been switched on"
and that "no recorded run ever set `GGML_HRX_SENTINEL`". Both are false, and the check that would have caught it
is a two-line grep of the campaign's own evidence files:

    evidence/315-instrumented-arm-hunt.md:27
    EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
               GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1" \
    bash ~/wt/315-paired.sh 20 2 2048 ~/wt/315instr3-build/bin ~/wt/315fp-build/bin 315nan

The discriminator was armed in the `315nan` session (2026-10-11), which produced 16/17 Face-A faults in regime.
**The run happened. What never happened is reading its `sentinel=` field.** The evidence file records the tally,
the rate (~94 % against the shipped 2/37), and the first NaN-bearing stage — and does not quote `sentinel=` once.

## The readout, extracted now from that run's own logs

**Liveness, required by the pre-registration's own terms — the fill actually happened:**

    20 of 21 arms carry:  [hrx-sentinel] filled host-visible logits region size=619520 with 0x7FC0DEAD

(for example `0.05.274.268 W`, `0.07.468.309 W`, `0.09.625.966 W` — at allocation, early in each arm). So the
requirement that would have made the arm INVALID is satisfied.

**The discriminator:**

    $ grep -h "NaN logits" ~/wt/rt-det/315nan-*/server.log | grep -oE "sentinel=[0-9]+" | sort | uniq -c
         17 sentinel=0

**All 17 faults report `sentinel=0`: not one of the 154,880 logits still held the payload.**

## What that decides

Branch **2** of the readout table above fires, and branch 1 is excluded:

- **The device wrote the logits region.** It did not leave it unwritten; the region is not stale and the
  "the sampler read memory the head never wrote" mechanism is **excluded for the logits tensor**, on a run that
  armed the discriminator and shows the fill.
- **The NaNs are computed and written by the device.** The all-`154880`-of-`154880` fault is a computed NaN, not
  a sentinel, not stale memory, not an unwritten page.

## How this composes with the rest of this stretch

- `norm` (RMS_NORM) reads **2048 of 2048 NaN**, and it is an **end-of-graph** tensor — the one class of reading
  this stretch showed *is* faithful under arena reuse. So the residual stream reaching the final norm is entirely
  NaN, and the head's NaN is downstream of that.
- The sentinel result closes the last "the device never wrote it" escape for the logits, which leaves the NaN
  being **produced upstream** of the final norm, in the FFN path (consistent with `315-instrumented-arm-hunt.md`'s
  own first-NaN-bearing stage: one `n_embd` row NaN from the last block's FFN onward).
- Device ASAN, enabled and initialized in the same campaign, reported **no** access violation in an in-regime
  faulting arm — so nothing in that class is implicated either. (Qualified: no positive control was run, so the
  sanitizer's own detection is asserted from its configuration and delivery path, not demonstrated.)

**Not decided:** whether *other* buffers were never written. The sentinel fills only the 619520-byte logits
allocation, so `sentinel=0` says nothing about the residual-stream or activation buffers. The "never wrote" family
is excluded **for the logits only**.

## The lesson, stated because it is this campaign's recurring failure

A discriminating instrument was built, armed, run, and its field never read — for two days, while the campaign
reasoned about the mechanism the field would have decided. This is the fifth instance of the same family
(a truncated sorted head; a line counter on newline-free output; a sum over streaming responses; a mis-indexed
`awk` field test; and now an unread field in an existing log). **The cheapest check is to grep the record before
building the experiment: it would have taken one command and saved writing this file twice.**
