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

# The ASAN channel is genuinely live: wiring verified at every step

`315-asan-preregistration.md` requires that a null result be discarded as INVALID unless the sanitizer was
**provably active**. This records the proof, taken before any faulting arm is read. Every step is quoted from
source or from a running process, not inferred.

**1. The knob reaches the process.** Read from the running arm server's own environment
(`/proc/<pid>/environ`):

    EXTRA_ENV=GGML_HRX_LOG_DEVICE_EVENTS=1 HRX_HAL_SANITIZER=asan
    GGML_HRX_LOG_DEVICE_EVENTS=1
    HRX_HAL_SANITIZER=asan

**2. The knob sets the runtime feature flags.** `libhrx/src/libhrx/runtime.c:576`:

    const char* sanitizer = getenv("HRX_HAL_SANITIZER");
    if (!sanitizer || !sanitizer[0]) return iree_ok_status();
    runtime_features = IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
    if (strcmp(sanitizer, "asan") == 0) {
      runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN;
    } ...

**3. The runtime feature flags turn the sanitizer ON in the AMDGPU HAL.**
`runtime/src/iree/hal/drivers/amdgpu/logical_device.c:241-260`:

    static void ...apply_runtime_features(options, runtime_features) {
      options->feedback.enabled |= iree_any_bit_set(runtime_features, ..._FEEDBACK);
      options->asan.enabled     |= iree_any_bit_set(runtime_features, ..._ASAN);
      options->tsan.enabled     |= iree_any_bit_set(runtime_features, ..._TSAN);
    }
    ... iree_hal_amdgpu_logical_device_options_apply_runtime_features(
            options, create_params->runtime_features);

So `HRX_HAL_SANITIZER=asan` does set `asan.enabled` — it is not a knob that silently does nothing.

**4. The device accepts it.** The arms create their device and serve requests normally, so
`iree_hal_amdgpu_logical_device_options_verify_supported_features` passed with ASAN requested; device creation
did not fail.

**5. The sink is installed, before accelerator init.** `ggml-hrx.cpp:1371-1378`, inside
`create_registry_context()`, gated on `GGML_HRX_LOG_DEVICE_EVENTS` and called **before** `hrx_gpu_initialize(0)`
— which is what the API demands (`hrx_runtime_set_device_event_sink` returns `FAILED_PRECONDITION` if called
after accelerator initialisation). The registered callback, `log_hrx_device_event`, prints, for each ASAN
report:

    HRX ASAN: executable=%llu export=%u site=%llu access=%u address=0x%016llx length=%llu
              workgroup=(%u,%u,%u) workitem=(%u,%u,%u) shadow=0x%016llx value=0x%016llx dispatch=0x%016llx

**6. The report path exists in the deployed binary.** Every `libhrx.so` on the box carries the knob string (×2)
and the ASAN report path (×3).

**Conclusion:** the channel is live end to end, and it is device-side — the one class of attribution this
campaign has never had. A run that produces no `HRX ASAN` line is therefore a **meaningful** exclusion, not an
INVALID one: ASAN was on, the sink was installed, and nothing was reported.

## One arm lost to my own cleanup (recorded, not hidden)

The first arm (`315asan-r1q`) recorded **`gtt_start 47GiB`** — **out of regime**, against the gate's `[95,110]`.
Cause: I killed leftover servers to free the box *while the rig was composing*, so GTT fell away from the
composition the rig had projected (`gtt_before_arm=82G`). Per the standing rule an out-of-regime arm is never
evidence, in either direction, and this one is discarded. It is recorded because a silenced OOR arm is
indistinguishable from a clean one in a summary line, which is exactly the failure mode this campaign has
already been bitten by.

## Defect #12, carried

**Killing the top-level probe script does not stop the run.** `pkill -f 315-idprobe` matched and killed the
`315-idprobe.sh` wrapper only; the real chain — `315-attrib.sh` → `315-paired.sh` → the arm's `llama-server` —
kept running for another ~25 minutes, because the diagnostic mode sets `NOSTOP=1`. It was found only because it
pinned the box at 1577 % CPU, holding tctl at 91 °C and blocking the ASAN run's gate. Two lessons, both the shape
of this campaign's earlier harness defects: **kill the process group, not the script name**, and **a leftover can
block a later run's gate rather than merely slowing it**.

---

# The outcome: no device-level access violation in either face

Both ASAN runs are complete. Tallies, verbatim from the session logs:

**`315asan2`** — `EXTRA_ENV="GGML_HRX_LOG_DEVICE_EVENTS=1 HRX_HAL_SANITIZER=asan"`, rig otherwise untouched:

    round 1 arm q: FAULT  nan=1  req=6   transport=1            gtt_start=100G   <- Face A
    round 1 arm e: FAULT  nan=1  req=10  transport=1            gtt_start=100G   <- Face A
    round 1 arm c: FAULT  nan=0  req=17  http=1 s500=1 unparsed=2  gtt_start=100G   <- Face B

    $ grep -h "HRX ASAN" ~/wt/rt-det/315asan2-*/server.log | wc -l      ->  0
    $ grep -h "HRX ASAN" ~/wt/rt-det/315asan-*/server.log  | wc -l      ->  0   (the discarded OOR run)

**Three arms, both faces represented, every arm in regime, and not one sanitizer report.** The channel was live
(proved section by section above: the knob reached the process, set `asan.enabled`, and the device was created
with feedback, which requires a valid sink). So the readout is the pre-registered **branch 2**: the fault is not
an access violation of the class the AMDGPU HAL sanitizer detects — a buffer/dispatch-binding access leaving its
allocation, caught by pool redzones and shadow.

## What this does and does not exclude, stated at the strength the evidence supports

**Does:** no sanitizer-visible out-of-bounds buffer access accompanied either face in these arms.

**Does not**, and this is the whole of the caveat:

1. **No positive control.** The sanitizer's own detection was never demonstrated on this device — the argument
   that it *would* fire rests on its configuration and delivery path, both verified, not on an induced violation
   being caught. A null result with no positive control bounds the family only as far as the sanitizer's own
   functioning goes.
2. **Granularity.** It validates at buffer/dispatch-binding granularity. An access that leaves its *row* but stays
   inside the allocation is not an ASAN event — and a gather reading a wrong-but-in-bounds row would produce
   exactly the finite-but-wrong distribution this campaign calls Face B.
3. **n = 3.** Two Face-A arms and one Face-B arm.

## Where this leaves the two faces

| face | excluded | still open |
|---|---|---|
| **A** (NaN logits) | stale/never-written logits region (the engine's own sentinel: `sentinel=0`, fill present in 20 of 21 arms); sanitizer-visible OOB | a NaN **computed** upstream of the final norm — the residual stream is all-NaN at the head, and *which* node produced it is unattributable with the host-side instrument (arena slot reuse) |
| **B** (format failure) | same two | a finite but **wrong** distribution at the sampler; the arm's own text is valid UTF-8 and begins mid-essay, i.e. a corrupted opening |

Both faces therefore point at the same unresolved question: **a value is wrong at a consumer, and no
host-side read can be trusted to say where it was first written.** Settling it needs a read whose fidelity does
not depend on the allocator's reuse — per-node at production, or device-side validation of the consuming op.
