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
