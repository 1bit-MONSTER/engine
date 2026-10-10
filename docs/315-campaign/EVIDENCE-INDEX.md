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

     engine#315: evidence index. Maps every objective criterion, every claim the mechanism document
     makes, and every instrument or tool, to the path that backs it - so the completion audit (and
     task-7's write-back) can check evidence rather than prose. Sections refer to
     docs/315-mechanism.md. -->

# engine#315 evidence index

## Objective criteria

| criterion | status | evidence |
|---|---|---|
| named, falsifiable mechanism for **both** faces, with a discriminating experiment | **not met** | the search and everything it eliminated is sections 24-38; what survives is listed below |
| forward-only change landed on a branch/PR | **not met** | no change is justified yet; the one candidate found statically was closed as not-applicable (section 37 + `evidence/315-glm-dispatch-inventory.md`) |
| pre-registered in-regime campaign, >=60 arms, zero faults both faces, with CI and power statement | **not run** | tooling frozen and audited (`315-campaign.sh`, `315-paired.sh`, `315-detector.py`); invocation in section 27; acceptance statistic and its four freeze notes in `315-preregistration.md` section 8 |
| no regression: engine CI green, dense control and MUL_MAT_ID op-test unchanged | **partially baselined** | `evidence/315-engine-ci-baseline.md` (copyright check, pin gate, ci.yml python tests, ctest registry tests all run locally at the pin); the gcc-15 build + ctest were *not* run |
| `KNOWN-ISSUES.md` updated or retired with evidence paths | **not updated** | it still carries the superseded "the logits buffer is never written" story that section 21 falsifies; section 31 records what the update has to correct |

## What was established

| claim | section | evidence path |
|---|---|---|
| Face A is a compute-side write of varied-payload NaNs; the LM head is exonerated (its input is already NaN) | 23 | `~/wt/rt-det/315hunt2-p6l-GLM-4.7-Flash-Q4_K_M/{server.log,detector.json}` |
| both faces fire on the first decoded token of a newly launched task (5/5 comparable arms, 18/19 pooled) | 23, 24 | the same logs, plus the older `identical.json` arms |
| the shipped pin's in-regime Face-A rate is 5.4% (2/37), always-stage 4.0% (1/25) | 31 | the rate study quoted in `KNOWN-ISSUES.md` |
| the sentinel instrument could not have decided "never written" (it fills once per allocation) | 25 | `ggml-hrx.cpp` `hrx_buffer_init`, plus the faulting arm's single `[hrx-sentinel]` line |
| the FA masked-row / tile-visibility direction is dead | 26 | `evidence/315-fa-masked-v-regime.md` (40/40, `non-finite outputs 0`) |
| every kernel-level test at the pin passes; two targets do not compile (pre-existing) | 27 | `evidence/315-hrx-tests-baseline.md` |
| GLM runs the **row-major MLA** layout, never the transposed-V transient | 38 | `dispatch-flash-attention.cpp` `prepare_flash_attention_value` |
| GLM never dispatches `q6_k_token1_shortlist`, so section 37's defect is not its mechanism | 37 | `evidence/315-glm-dispatch-inventory.md` |
| the pin is race-free at the October shape and at the regime's shapes (2250 + 2000 reps) | 34, 36 | `evidence/315-fa-race-pin-sweep.md` |
| the two faces never co-occur (23 Face-A, 7 Face-B, 0 both, 108 clean of 142 arms) | 42 | `evidence/315-face-landscape.md` |
| Face A travels with finite nondeterminism; Face B does not | 42 | the same file (`identical_distinct` distribution) |
| neither instrument can fabricate NaN (copy probe only fprintfs; sentinel is size-gated to 619520) | 41 | `ggml-hrx.cpp:181-199, 380-391` in the fp build's tree |
| the faulting frame is a T=1 decode whose final norm is all-NaN, the first NaN frame a prefill at block 41 | 40 | the `p10l` arm's `server.log` lines 1713-1868 |
| the transient-reuse guard is default-on and is the packer that runs for decode programs | 35 | `GGML_HRX_LOG_TRANSIENT_REUSE=1` output quoted in section 35 |

## What is eliminated (with the cost it took)

Sections 26, 27, 32, 33, 34, 36, 37, 38 - coherence and ordering (their own campaign), FA masked rows, all
kernel standalone logic, the AQL block-boundary asymmetry, intra-dispatch unfenced global handoffs, the
October FA race, the `exact_output` access defect. Section 29 is the consolidated table.

## Instruments and tools

| item | path | state |
|---|---|---|
| nan-dump fingerprint patch (`tail:<N>`, `NANONLY`, `op=`) | `patches/315-nan-dump.md`, `tools/gen-nan-dump.py` | applied to `~/wt/llama-315copy`, built into `~/wt/315fp-build` |
| non-finite masked-V test extension | `patches/315-fa-masked-v-nonfinite.md`, `tools/gen-fa-masked-nonfinite.py` | built and run (40 cases) |
| binding-access audit | `tools/audit-binding-access.py` | run over all four manifests (522 bindings, 2 findings) |
| FA race sweeps | `tools/fa-sweep.sh`, `tools/fa-variant-profile.py` | run (2250 + 2000 reps) |
| MLA race harness | `tools/glm-mla-race.cpp` | compiled on strixhalo, **not run** (needs the GPU) |
| cheap probe | `315-cheap-probe.sh` | run; one fault in eight runs at `align=256 + compaction` |
| instrumented arm hunt | `315-paired.sh` + the §27 command | ran; record in `evidence/315-instrumented-arm-hunt.md` (17 fault / 1 clean, and a control arm that faulted) |
| face landscape | (post-hoc parse of the arm logs) | `evidence/315-face-landscape.md` |
| acceptance tooling | `315-campaign.sh`, `315-paired.sh`, `315-detector.py`, `315-analyse.py` | frozen; four freeze notes in `315-preregistration.md` section 8 |
| acceptance rule self-test | `tools/test-analyse.py` | 10/10 cases pass (gate edges, Face-B-as-fault, rate over in-regime arms rather than clean arms) |
| attribution + chain probes | `315-attrib.sh`, `315-chain.sh`, `315-chain2.sh` | diagnostic only, never acceptance-eligible. `315-chain.sh` ran: pair 1 faulted both cells with a 60-name fingerprint and **zero** NaN lines, which is the direct Face-B measurement; `315-chain2.sh` is staged for the input-side case |
| visibility falsifier | `315-graphsync.sh` | queued: `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` (a stream sync after every replay), no other instrument |
| row probe | `315-rowprobe.sh` | queued: builds the extended fingerprint (`ne0=`/`first=`/`row=`) into a new dir and reads the per-token MoE chain (section 47) |
| Face-B probe (not run) | `315-faceb.sh` | written then dropped in favour of the visibility falsifier; its question is answered by the chain probe's zero-NaN result |

## Arms and logs

`~/wt/rt-det/<tag>-p<N><c|l>-GLM-4.7-Flash-Q4_K_M/` on strixhalo (server.log, detector.json, run.log).
Note the glob must be `*-*l-*`, not `*-l-*`. Campaign arms would be `315fix-*`; none exist yet.
