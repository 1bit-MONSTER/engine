<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: device-ASAN with the HRX device-event sink (never run in this campaign)

Written **before** the run. The frozen rig (`~/wt/315-disc.sh`, three cells, `GLO=95`/`GHI=110`) is unchanged
except for `EXTRA_ENV`.

## Why this is the experiment the campaign is missing

Every instrument used so far reads **host** memory and reports after the fact, which this stretch showed is not
tensor-faithful for mid-graph tensors (`315-which-readings-are-faithful.md`). The runtime ships a **device-side,
source-attributed** channel that has never been switched on here:

- **`GGML_HRX_LOG_DEVICE_EVENTS=1`** — `ggml-hrx.cpp:1371-1378` installs `log_hrx_device_event` as the HRX
  device-event sink **before** `hrx_gpu_initialize(0)`. The sink is already written to print, for every ASAN
  report: `executable=… export=… site=… access=… address=0x… length=… workgroup=(…) workitem=(…) shadow=0x…
  value=0x… dispatch=0x…`
- **`HRX_HAL_SANITIZER=asan`** — `libhrx/src/libhrx/runtime.c:576` sets
  `IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN | …_FEEDBACK`, i.e. device address sanitizer plus feedback.

Both exist in **every** `libhrx.so` on the box (checked: `HRX_HAL_SANITIZER` ×2, the ASAN report path ×3 in each
build). **Neither the env var nor the sink appears anywhere in the campaign record or in any driver/probe
script** — the channel is implemented, wired, and has never been used.

It matters because it is the only channel that can attribute a fault to a **kernel**: `export_ordinal` names the
export, `executable_id` the executable, and `fault_address`/`access_kind`/`access_length` the access.

## Hypothesis

In-regime faults include, or are preceded by, an **out-of-bounds buffer access performed by a specific kernel**,
which the AMDGPU HAL's device address sanitizer can report.

## Readout, fixed in advance

1. **An `HRX ASAN:` line in a faulting arm** → the mechanism is an out-of-bounds access. The line names the
   kernel (`executable`, `export`), the access (`address`, `length`, `access`), the site (`site`) and the
   sanitizer's view (`shadow`, `value`). **That is the named mechanism**, and it is falsifiable by inspection of
   the named kernel.
2. **No `HRX ASAN:` line, fault still occurs** → the fault is **not** a dispatch-visible out-of-bounds access and
   the sanitizer does not see it. A genuine exclusion of the OOB family.
3. **Neither report nor fault** → **INVALID**, not a pass. The intervention may have masked the fault; per the
   campaign's standing rule an arm that does not exercise the fault never counts as evidence of absence.

## Confounds, declared before the run

- **ASAN is an intervention.** It adds shadow memory and per-dispatch validation, and may move the fault rate
  (the record already holds that the fingerprint instrument *raises* it, §39). No arm here is an acceptance arm.
- **Granularity.** The sanitizer validates buffer/dispatch bindings; it is not a per-memory-operation checker and
  can miss an out-of-bounds access that stays inside a mapped buffer. A null result therefore bounds the OOB
  family *as the sanitizer defines it*, not "no OOB anywhere".
- **Liveness.** A null result is only meaningful if the sanitizer was provably active. If the run shows neither
  an ASAN report nor any other evidence the feature was accepted, the outcome is **INVALID**, not a pass.
- Device creation may simply **fail** with the sanitizer on; that is itself a recorded finding, not an error to
  be discarded.

## What would be recorded either way

Whichever branch fires, the run is recorded with its arm tally, its `gtt_start`s, and the exact `EXTRA_ENV` used.
No criterion changes. No arm eligible.
