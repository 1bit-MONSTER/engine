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

# The `tail:16` run: what it settled, and why it was stopped

Recorded because a run that is stopped early leaves a gap in the record unless the reason is written down.

## What it was for

`315-glm-never-dispatches-the-changed-kernel.md` and
`315-f32-twin-result-divergence-on-the-faulting-arm.md` both left the same instrument gap: the logits node
`result_output` — the tensor the NaN guard reports — produced **zero** fingerprint records in the arms that
faulted, because the name filter is an exact per-comma-item match and the name list did not reach it. The
fix recorded there was to select by **position** (`tail:<N>`), which bypasses the name filter.

## What it settled

**The position selector works.** With `GGML_HRX_TENSOR_FP=tail:16` the logits node is covered in every arm:

| arm | verdict | `result_output` records | divergences |
|---|---|---:|---:|
| `315tail-1` | clean | 165 | 0 |
| `315tail-2` | clean | 417 | 0 |
| `315tail-3` | clean | 402 | 0 |
| `315tail-4` | clean | 843 | 0 |
| `315tail-5` | clean | 54 | 0 |
| `315tail-6` | (in flight when stopped) | 831 | 0 |

So the instrument gap is closed and the twin does fingerprint the logits — but **six arms produced no fault**,
so "does the logits read diverge in a faulting arm" is still unanswered. That is a statement about this run,
not about the model: the earlier, lighter runs faulted at arm 2 and arm 3.

## Why it was stopped

Because a better-aimed experiment was available and this one is expensive. `tail:16` fingerprints sixteen
nodes with three reads each per graph — including the 620 KB logits tensor — which makes each arm heavy
enough that its detector phase ran long and its arms came back truncated (`TimeoutError`) rather than
faulting. Two runs in a row produced truncated arms.

Meanwhile `patch-writeback-read-probe.py` tests the **named mechanism** directly rather than a downstream
symptom: it reports `graph_replay_state.has_pending_host_writebacks()` at the engine's async logits read, and
that probe adds **no** per-tensor hashing, so arms stay light and a fault should arrive sooner. If
`pending_writebacks=1` appears at the read in a faulting arm, the candidate is confirmed; if it is 0
throughout, the candidate is dead and
`315-what-the-synchronise-drains.md`'s remaining list is where the hunt goes.

That run (`315wb`) was launched with `GGML_HRX_WRITEBACK_READ_PROBE=1`, `-ub 512`, and **no** tensor
fingerprint, so that the only instrumented thing in the arm is the probe.
