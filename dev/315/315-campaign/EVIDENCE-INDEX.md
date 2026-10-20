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
     dev/315/315-mechanism.md. -->

# engine#315 evidence index

## Objective criteria

| criterion | status | evidence |
|---|---|---|
| named, falsifiable mechanism for **both** faces, with a discriminating experiment — **or a bounded negative** | **BOUNDED NEGATIVE, accepted** | every surviving candidate family eliminated by its own pre-registered experiment, each with its command and observed output; the surviving families and the hand-forward fact are stated below and in `../315-STATUS.md`. **No mechanism is claimed** |
| forward-only change landed on a branch/PR | **MET** | PR **#414**. Fork commit `47a8fe929` (5 lines ending in `ggml_backend_synchronize(backend_res)` in `src/llama-context.cpp`); engine pin `e44c9d01a -> 47a8fe929`. `47a8fe929`'s **parent is `0e255d774`** = current `main` pin, and `git merge-base --is-ancestor e44c9d01a 47a8fe929` is true. `check_pins.py` reports "ahead" |
| pre-registered in-regime campaign, with the regime gate, arm count, both-face detectors and stopping rule fixed before it ran, **reported with its result whether met or not** | **RUN, acceptance FAILED — reported as such** | `evidence/315-campaign-preregistration.md` (60 fix + 20 control arms, gate `[95,110] GiB`, stop at the first fix-cell fault). The fix cell **faulted on its first arm** (Face B, `http_failures:1`, `first_failure:"HTTP 500"`, in gate at 98G) — a fault, never an unrun arm. Power statement `1 - 0.05^(1/60) ~= 4.9 %`. `evidence/315-logits-wait-FALSIFIED.md` |
| no regression: engine CI green, dense control and MUL_MAT_ID op-test unchanged | **MET, at the moved pin `47a8fe929`** | engine CI **7 of 7 green** on PR #414 (`mergeable=MERGEABLE`, `state=CLEAN`); `evidence/315-task3-op-test-unchanged-at-moved-pin.md` (`8 pass, 0 skip, 0 fail`; mxfp4 nmse `1.34e-05 … 1.61e-05`, exact match to baseline; same two kernels) and `evidence/315-task3-dense-control-clean-at-moved-pin.md` (50 responses, 1 distinct content, zero fault signatures) |
| `KNOWN-ISSUES.md` updated with the evidence, the met criterion, the reproduction and the campaign paths; status stays OPEN | **MET** | `KNOWN-ISSUES.md` #315 carries the forward-only change, the reproduction, the campaign paths and the falsification; status line **OPEN** |

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
| exactly one `n_embd` row of NaN (2048 floats) is the invariant across the 16 faulting in-session arms, at the MoE aggregation, with the traced row count varying 87-342 | 47 | `for d in ~/wt/rt-det/315nan-*/; do grep -m1 -E "hrx-tensor-fp.*nan=[1-9]" $d/server.log; done` on strixhalo |
| the coherence *patch* does not remove the fault (9 clean, 1 fault in 10 fix-cell arms, `hv52=0 hv56=1` proving it live) | 46 | `~/wt/315accept-console.log` |
| `GRAPH_SYNC` after every replay does not remove the fault (1 in-regime arm, faulted; 4 of 6 arms below the gate) | 46 | `~/wt/315gsync-console.log`, `evidence/315-face-landscape.md` |
| the AQL-block-split asymmetry is *not* real: `packet_index_base` is a profiling prefix (0 when profiling is off) and every block is submitted as its own submission | 32 | `host_queue_command_buffer_block.c:879`, `host_queue_command_buffer_replay.c:337/364` |
| the routed-FFN path's routing tables are rebuilt per plan, so a stale table is not available as a mechanism; the sites that can lose one token's row are intra-plan producer/consumer ordering, the partition kernel's `completion_counters` publication, and the tables' transient allocation | 49 | `evidence/315-moe-routing-routed-ffn-static.md` |
| the engine's MoE expert-streaming cache (`serve --moe-slots`/`--moe-subst`, `moe/expert_cache.cpp`) is not on the HRX decode path | 49 | the same file, section 6 |
| the two faces never co-occur (23 Face-A, 7 Face-B, 0 both, 108 clean of 142 arms) | 42 | `evidence/315-face-landscape.md` |
| Face A travels with finite nondeterminism; Face B does not | 42 | the same file (`identical_distinct` distribution) |
| neither instrument can fabricate NaN (copy probe only fprintfs; sentinel is size-gated to 619520) | 41 | `ggml-hrx.cpp:181-199, 380-391` in the fp build's tree |
| the faulting frame is a T=1 decode whose final norm is all-NaN, the first NaN frame a prefill at block 41 | 40 | the `p10l` arm's `server.log` lines 1713-1868 |
| the transient-reuse guard is default-on and is the packer that runs for decode programs | 35 | `GGML_HRX_LOG_TRANSIENT_REUSE=1` output quoted in section 35 |

## What is eliminated (with the cost it took)

Sections 26, 27, 32, 33, 34, 36, 37, 38 - coherence and ordering (their own campaign), FA masked rows, all
kernel standalone logic, the AQL block-boundary asymmetry, intra-dispatch unfenced global handoffs, the
October FA race, the `exact_output` access defect. Section 29 is the consolidated table.

## Session of 2026-10-15 — the ordering family removed, one site shown unmeasurable

| claim | evidence |
|---|---|
| **the writer of the `result_norm` slot is NOT ordered by `context->stream`** — on a faulting arm, of 82 records where the tensor differed between two consecutive plain reads, a fresh `ggml_backend_synchronize` stopped it in **0** and did not in **82**, and changed the value itself in **82 of 82** | `evidence/315-task1-writer-not-stream-ordered.md` (`b7d6b46`); pre-registered at `evidence/315-PREREG-sync-between-reads.md` (`b51f883`) |
| this **removes** `GraphExecutor::execute` returning without a wait (with `backend_synchronize` covering only `context->stream`) as the explanation | the same file; the falsifier is restated there and **not triggered** |
| **§51's rank-sort precondition is not measurable by the by-name fingerprint** — the faulting arm produced 0 records; the `tail:16` census has **946 distinct node names** and contains `ffn_moe_out-*`/`ffn_shexp-*` but **none** of `ffn_moe_probs-*`, `ffn_moe_topk-*`, `ffn_moe_argsort-*`, `ffn_moe_weights-*` | `evidence/315-task2-bounded-negative-precondition-unmeasurable.md` (`ba6753a`) |
| consequently **how §50 read `ffn_moe_probs` is itself unexplained** — recorded as an open discrepancy, not smoothed over | the same file |
| `95238dc` was a **half-move**: it moved the `llama.cpp` gitlink without `registry/architectures.json`; CI failed until fixed | `evidence/315-what-the-merge-and-the-gates-found.md` |
| **46 files** failed the copyright gate on the two-line SPDX form; `*.hip` was a genuine gap in the tool | the same file |
| four harness defects of one class this session (skip-as-pass; `"0\n0"` from `grep -c … \|\| echo 0`; line-based response count; a vacuous CI green) | `evidence/315-task3-dense-control-clean-at-moved-pin.md`, and for other sessions in `../315-COORDINATION.md` |

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
| visibility falsifier | `315-graphsync.sh` | ran 2026-10-10 (6 pairs, `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` both cells, no other instrument): **4 of 6 arms below the gate**; the one in-regime arm faulted. Read as *not sufficient*, not as a rate. Record: `evidence/315-face-landscape.md` |
| row probe | `315-rowprobe.sh` | ran 2026-10-10: builds the extended fingerprint (`ne0=`/`first=`/`row=`) into `~/wt/315row-build` and reads the per-token MoE chain (section 47) |
| Face-B probe (not run) | `315-faceb.sh` | written then dropped in favour of the visibility falsifier; its question is answered by the chain probe's zero-NaN result |

## Arms and logs

`~/wt/rt-det/<tag>-p<N><c|l>-GLM-4.7-Flash-Q4_K_M/` on strixhalo (server.log, detector.json, run.log).
Note the glob must be `*-*l-*`, not `*-l-*`. Campaign arms would be `315fix-*`; none exist yet.

## Verification of this index (2026-10-10)

Every number quoted across these evidence files is re-checked from the box by one command:

```sh
bash dev/315/315-campaign/tools/verify-evidence.sh     # 30 checks, all PASS; ~25 s per check
# ----- PASS=30 FAIL=0
```

It covers: the five in-regime arms and their verdicts, the guard line, the faulting arm's fingerprint lines and
their hashes, the two host-visible allocations from the buffer trace, the coherent-cell trace, the archived
overshoot console, the one-file differential (`102 insertions, 2 deletions`) and the empty routed-FFN diff, the
`route_stride` counts in both trees, the queue profile's dispatch/barrier/block numbers, the `scacquire` wait
variant, the timepoint assignment, the three instrument patches, the wrapper directory, and the two kernel facts
(ids clamped in the gather, ordered compares in the rank sort). Two checks needed their *expected* regex fixed
when the suite was first written - the underlying numbers were right, the assertions were not - and the clean run
above is with the fixed assertions.
