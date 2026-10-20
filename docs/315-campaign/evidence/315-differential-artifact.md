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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 task-3 differential artefact: the HRX backend between the recorded clean lower bound
     (f2099e9b7) and the pin the in-regime arms run (e44c9d01a). Read-only git work on strixhalo; two
     commands, both quoted with their exact output. -->

# engine#315 — the differential artefact: what changed in `ggml/src/ggml-hrx` between the clean bound and the pin

```sh
ssh strixhalo 'cd ~/wt/llama-315copy && git rev-parse --short HEAD; git rev-parse --short f2099e9b7'
# e44c9d01a
# f2099e9b7

ssh strixhalo 'cd ~/wt/llama-315copy && git diff --stat f2099e9b7..HEAD -- ggml/src/ggml-hrx/'
#  .../flash_attention_decode_split_f32_f16_wmma.loom | 104 ++++++++++++++++++++-
#  1 file changed, 102 insertions(+), 2 deletions(-)

ssh strixhalo 'cd ~/wt/llama-315copy && git log --oneline f2099e9b7..HEAD -- \
  ggml/src/ggml-hrx/dispatch_registration/common/dispatch-gated-mul-mat-id.cpp \
  ggml/src/ggml-hrx/dispatch_registration/common/dispatch-mul-mat-id-common.h \
  ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom \
  ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/mul_mat_id_swiglu_f32_f32_wmma.loom'
# (no output)
```

**One file in the whole backend differs, and it is the decode-split flash-attention kernel**: +102/-2 in
`flash_attention_decode_split_f32_f16_wmma.loom`, the same change already rate-tested against the always-stage
build (5.4% vs 4.0%, Fisher p = 1.000) and closed by mechanism-doc sections 26/34/36.

## Why this file exists

It is the constraint that decides which of #315's two live candidate families can be a *change* and which can
only be a *pre-existing amplifier*:

* **The routed-FFN path is byte-identical between the clean bound and the faulting pin.** The dispatch
  registration (`dispatch-gated-mul-mat-id.cpp`, `dispatch-mul-mat-id-common.h`), the id-producing kernel
  (`small_rows_f32.loom`), and the fused consumer (`mul_mat_id_swiglu_f32_f32_wmma.loom`) have no commits in the
  window. So the routing-tensor site that `315row-p1c` measured cannot have been *introduced* by the pin move.
  Either it is older than `f2099e9b7` (in which case the clean label was about rate, not about absence, and the
  same site must be reachable on the clean bound too), or the pin move is what makes it fire.
* **The only thing the pin move did change in this backend is the decode-split attention kernel.** That keeps the
  FA change in the causal frame even though its kernel-level checks passed - and it sharpens the one caveat
  already on the record: section 38 established that GLM runs the **row-major MLA** layout, which invalidates the
  "GLM geometry" label on the race sweeps that cleared this kernel (sections 26, 34: 40/40 masked-V cases,
  2250 + 2000 race reps). Those sweeps were run at shapes that are not GLM's MLA geometry, so "the pin is
  race-free" is established for the shapes tested, not for the shape this model actually runs.

## The artefact set this completes, against task-3's contract

| task-3 item | state |
|---|---|
| ISA/IR diff of the changed dispatch/kernel | **this file**: for the routed-FFN path the diff is empty; for the FA path it is `task3-fa-kernel-ir.txt` (103 `+` / 6 `-`) with the named site `template.def<@ggml.flash_attention.decode_split.produce_partials.active>` |
| `rocprofv3` trace, faulting vs clean in-regime arm | **not done as rocprofv3**; the engine's own queue-event profiler is used instead and its output is recorded in `evidence/315-queue-profile.md` (612 dispatches, 519 barriers, 92 unbarriered, 1 block, for a 24-token greedy decode). It is a *clean* profile, not a faulting-vs-clean pair: the trace of a faulting arm is still owed |
| REA (or documented equivalent) on libhrx/loomc and the HIP code objects | **partly done**: the shipped `libhrx.so` is not stripped, and its symbol/flag inventory (`pm-salvage/issue-315/CANDIDATE-SITE-WAIT-BARRIER-DEFER.md`) is the documented equivalent; REA's own provider failed with an unretained log |

The one item this file adds beyond the table is the **shape gap** in the FA clearance, which is the strongest
remaining question about the pin-move window.
