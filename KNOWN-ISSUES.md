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

# Known issues

## engine#370 — intermittent MUL_MAT f32 m=1 n=1 k=2048 error on `common.mul_mat.f32_f32_decode`

**Status: OPEN, root cause not fixed in this repository.** The kernel lives in the llama.cpp fork
(`motifs/dequant.loom`, fork tip e44c9d01a4d5), which is not vendored here, so no engine-side change can fix it.

**Affected:** `test-backend-ops -b HRX0 -o MUL_MAT`, case `MUL_MAT(type_a=f32,type_b=f32,m=1,n=1,k=2048)`,
kernel `common.mul_mat.f32_f32_decode`. Observed on strixhalo 2026-10-09 (hrx-system 98d05d94, performance mode):
1 failure in 40 runs on the unmodified fork tip (ERR 0.0037); 1 failure in 28 full-suite runs at the quant stack
tip (ERR 0.0102); 0 failures in 40 further MUL_MAT runs and 10/10 running the case alone.

**Assessment:** pre-existing, not introduced by the quant PRs. The error magnitude and ~1-in-40 rate point to a
race or an uninitialised read in the shared decode path rather than rounding. Fix belongs in the fork; bump the
pin once it lands. Treat a single failure of this case as a flake, not a regression.

## engine#315 — intermittent all-NaN logits and format-500 on GLM-4.7-Flash under GTT pressure

**Status: OPEN — mechanism NAMED and replicated; no forward fix landed yet.** The issue stays open until a fix lands and is verified (criteria 2-5). Its first criterion — a named, falsifiable mechanism for both faces with a discriminating experiment — was met on 2026-10-13 by a pre-registered three-cell intervention: **forced eviction of in-flight KFD work causes the fault on both faces at 70 % vs 0/22 control arms, p = 3.6e-5**. See "What remains unknown" for the experiment, and note what is still missing: the specific handoff that reads stale data.

Owner acceptance, recorded 2026-10-07: the owner authorised landing this entry in writing — *"you have gh
authorization and do what needs to be done"* — on the basis that no forward fix is available. The limitation text
below was drafted by the assistant; the owner has not edited it and may amend or withdraw it at any time.

**Affected:** `GLM-4.7-Flash-Q4_K_M` on `HRX0`, `llama-server`, at high GTT occupancy (measured from
`gtt_start` ~83 GB up to ~110 GB). Intermittent under ambient pressure — the regime-controlled rate is
**~5-7 % per in-regime arm** (see the rate table below) — but **reproducible on demand** when a separate
process forces KFD eviction/restore: **7 faults in 10 in-regime arms (70 %)** while matched control cells
stay at 0/22.

**Rate, corrected 2026-10-13.** The earlier "roughly 10-20 % per run" was a pooled figure and it mixed a
deliberate high-rate reproduction rig with ordinary runs. Decomposed from the arm ledger's own per-tag table,
the `315nan` session alone contributed 18 of the 33 pooled in-regime faults from 19 arms (94.7 %); excluding
it the pool is 18.1 %. The two measurements taken under a *stated* regime agree with each other instead:

| measurement | arms | regime | faults | rate |
|---|---|---|---|---|
| shipped-pin study (this file, 2026-10-09) | 37 | 102 GiB | 2 | **5.4 %** |
| `315ctl1` cwsr=1 fresh-boot control (2026-10-13) | 27 | 97-98 GiB | 2 | **7.4 %** |

Fisher exact two-sided p = 1.000, i.e. no difference detectable. **So the working rate for this configuration is
~5-7 % per in-regime arm**, with both faces represented. Any experiment sized against 10-20 % or against the
pooled 26 % will be under-powered: at 7 % a clean block of 20 arms occurs by chance ~21 % of the time. Re-measure
the rate at the configuration under test and size from that number.

### Reproduction

3 detached neighbour `llama-server` processes at default context drive GTT into the ~100 GB band, then one
detector server is exercised with 8 identical greedy requests separated by different-length distractors,
followed by 50 mixed-length requests. Under pressure the logits buffer is never written, and the sampler guard
reports:

```
count=154880 of 154880, contiguous=yes
```

`154880` is the model's vocabulary width — the whole LM-head output. The guard fires on the consumer side,
which is why it reports the shape and not the producer.

### Both failure faces

The same underlying defect — output data that is wrong or unwritten under memory pressure — surfaces in two
places depending on where it lands:

1. **All-NaN logits.** The logits buffer is never written; the sampler guard fires with
   `count=154880 of 154880, contiguous=yes`. Example: `v315g-c3` (control arm).
2. **Format-500.** The model emits text the server rejects:
   `{"error":{"code":500,"message":"The model produced output that does not match the expected Content-only
   format"}}`. Example: `v315k-l1` (legacy arm), `identical: 1 distinct of 8` then mixed `10 of 50`, `nan=0`.

A NaN-only count records the second face as a clean run, which is what makes it dangerous as a metric. Scoring
must be: an arm is **valid only if it completes `mixed: 50 of 50 ok`**; an arm that aborts early has reduced
exposure and is excluded, not counted as clean.

**When each face appears, with timestamps (2026-10-13).** Both faces were captured in the same 20-pair control
run and their logs give the sequence rather than a verdict, which sharpens what a mechanism must explain:

    Face A  (p16c)  0:10.67  model loaded / listening
                    0:11.11  first request (task 0) launched
                    0:11.28  HRX returned NaN logits: count=154880 of 154880, [0..154879], contiguous=yes
                    0:11.28  sampling.cpp:620 aborts -> backtrace, process dies (105-line log vs a clean arm's 485)
                    -> the FIRST request, 165 ms of compute in, the WHOLE vocabulary, process does not survive

    Face B  (p8l)   0:40.20  printing 96-token eval timings, 21 completions already OK
                    1:31.42  Content-only parse rejects one output -> HTTP 500
                    -> ~90 s in, after 21 good completions, ONE degenerate output, process survives

So Face A is an **initialisation-adjacent whole-buffer** failure on the first request, and Face B is a **late,
partial content** failure inside an otherwise working server. That is a materially different pair of signatures
from "intermittent corruption somewhere in a run", and it means the defect is already present at
load-completion - it does not need a long generation phase to appear. It does NOT follow that a warmup-targeted
experiment is cheaper: the model load is ~10-15 min and dominates every arm regardless of how many requests
follow, so a warmup-only arm costs about the same as a full one while yielding half the arms.

### Levers tested and excluded

- **`GGML_HRX_TRANSIENT_REUSE=legacy`** — no effect. 15-arm matched study, alternating control and legacy
  inside one box state: **0 NaN in 15 control vs 1 in 14 legacy.** A 7-arm paired confirmation scored by
  completion rather than NaN: **0 NaN in 7 arms**, control 3 clean / 1 truncated, legacy 2 clean / 1 truncated.
  Truncation is not one-sided, and the unpaired aggregates on both sides are noise.
- **Placement advice on the transient arena's committed backing** — cannot be a fix. `iree_hal_buffer_params_t`
  has no placement member (`usage`, `access`, `type`, `queue_family_affinity`, `min_alignment`;
  `runtime/src/iree/hal/buffer.h:646`), and the only advice API is advisory by contract: *"incorrect hints may
  reduce performance but will not cause incorrect behavior"* (`iree/hal/allocator.h:495`). The HRX transient
  path never calls it, but a hint that cannot cause incorrect behaviour cannot resolve one either.
- **The staged host writeback path** — excluded. `download_prepared_host_staging` gates on `has_download`,
  which needs `staging.download = access.write && !binding.graph_input` **and** a binding with non-null
  `host_data` — set only for non-directly-bindable *host* allocations. No graph node writes into a host tensor
  on this path, so `PendingHostWriteback` is inert: 11 valid arms, 0 `HRX host writeback` lines, with the
  reporting path proven to fire under `=poison`.
- **The direct device-to-host copy not being covered by the wait** — falsified.
  `stream_synchronize_sleeping` (`runtime/hrx-sleeping-wait.cpp:127`) flushes **before** it waits, so the copy
  recorded by `backend_get_tensor_async` is submitted and then awaited.
- **Pinning `f2099e9b7` (the `1bit/amd-core-our-hrx` cell)** — not available as a fix.
  `git merge-base --is-ancestor f2099e9b7 26330cc20` is **true**: it is a strict ancestor of the pin
  `engine@main` already ships (`third_party/llama.cpp @ 26330cc20490a547f94c9e5c960bf93063453151`), so adopting
  it would be a revert, which this issue's constraint excludes. Its `322/322 MUL_MAT` result is therefore **not**
  a working configuration that the shipped pin lost — whatever makes that cell clean is already in `26330cc20`,
  which faults. The regression is in what the `1bit/hrx-vulkan-patched` merge added on top, and `f2099e9b7` is a
  known-clean lower bound to bisect from.
- **The `TRANSIENT_REUSE` knob is not a lever for this fault** beyond the arms above: truncation is not one-sided,
  and the unpaired aggregates on both sides are noise.
- **The sleeping wait returning before completion** — cleared: it falls through to `hrx_stream_wait` and returns
  that status.

### Traps this issue has already sprung

- **Wrong device.** Results are only meaningful with an explicit provider/device; a run against the wrong backend
  produces plausible-looking numbers that answer a different question.
- **Vacuous pass.** Zero NaN can be achieved by failing differently — an arm that aborts early logs no NaN. Both
  faces above must be counted, plus complete-arm counts.
- **Stale incremental builds.** Several earlier measurements on this issue were retracted as build artifacts: a
  nested build directory that had previously compiled a *different* `ggml-hrx` carried ~1.3 MB of stale objects.
  Fresh build directories with the engine's own `ExternalProject` arguments are required.
- **Silent arm loss.** `rt-det-direct.sh` takes the box lock on fd 9 and passes it to the `nohup`'d server, so an
  abnormal arm death leaves an orphan holding the flock and every later arm wedges without running. One-word fix
  at the launch site: `... 2>&1 9>&- &`.

Three more traps found on 2026-10-13, each of which cost time or nearly produced a wrong conclusion:

- **Raw per-pid eviction counts are meaningless for a FAULTING arm.** A faulting arm's server can be orphaned
  (`ppid=1`) and outlive its own arm, so it accumulates evictions for as long as it survives. The faulting arm
  `p8l` shows **2,798** raw evictions — the highest in its run — against **203** when correctly bounded to the arm
  window, while the clean-arm ceiling is 559. A raw `grep` therefore reports the opposite of the truth. Always use
  the arm-bounded **pre-fault** column (`315-evict-per-arm.py`), never a raw per-pid count.
- **A Face-B fault can be invisible to `grep`.** The parser's rejected text contains invalid UTF-8, which makes
  `grep` treat the log as binary and return *nothing* for a line that is present. Use `grep -a`, or trust the
  detector's `first_failure`/`http_failures`. Related: the scoring tool's own face tally keys on parse/utf8
  counters, so a Face-B fault recorded as `http_failures` lands in its "unattributed" column — `detector.json` is
  authoritative, not the console and not the face tally.
- **A pooled rate is not a rate for a configuration.** Pooling a deliberate high-rate repro rig (`315nan`, 18
  faults in 19 arms) with ordinary runs produced the "~26 %" that several instruments were powered against, when
  the two regime-controlled measurements agree at 5-7 %. See the corrected rate table above.
- **A rollback path is not a rollback.** The `cwsr_enable=0` drop-in was removed for a control run and its verbatim
  bytes had never been recorded — only a path and an md5, and the file was gone. Capture the CONTENT of any
  environment change into the repository, not a reference to it.

### What remains unknown

**The trigger is NAMED and REPLICATED (2026-10-13): forced eviction of in-flight KFD work causes the
fault, on BOTH faces.** A pre-registered three-cell experiment (`docs/315-campaign/evidence/`,
upstream pre-registration commit `36f360a`, branches fixed **before** the run) isolates the variable:

| cell | side load | result |
|---|---|---|
| `e` | `~/wt/evictonly 500` — launches **no kernels**, only `mprotect` churn over `hipHostRegister`'d memory, forcing KFD evict/restore | **7 FAULT / 10 arms = 70 %** |
| `q` | quiet (no side load) | 0 FAULT / 10 arms |
| `c` | CPU pressure only, pinned to CCD1, no GPU | 0 FAULT / 12 arms |

Fisher exact two-sided **p = 3.6e-5**; rule-of-three upper bound on the control rate (22 arms, 0 faults)
= 13.6 %. All seven faults in-regime at `gtt_start = 98 GiB`, zero OOR in either run. The intervention
is verified rather than assumed: `evictonly` produces **4,134 `Q_EVICT` / 4,168 `Q_RESTORE` in 25 s**
against **0 / 0 in a 25 s quiet window** (the issue's own `kfd_smi_watch`). Because it launches no
kernels, the effect cannot be compute, thermal or scheduling — and the CPU-pressure cell independently
rules those out across 12 arms.

**Both faces appear under the same intervention**, in the same cells, same day, same build:

- **Face A** ×4 — `nan=1`, `count=154880 of 154880, contiguous=yes`, the sampler guard aborts the server.
- **Face B** ×3 — `nan=0`, `http=1 s500=1 unparsed=2`, the parser rejects the output and returns HTTP 500
  while the server survives.

So a single intervention that isolates eviction from every other factor tested produces **both** faces,
which is what "one mechanism, two observables" predicts.

**What is still NOT named: which handoff reads the stale data.** The static leg showed that the
engine#123/#140-class handoff in GLM's path is correctly fenced in the current pin (`barrier<global>`
then `barrier<workgroup>` between the reduce's stores and the q8 pack, plus the completion-counter
handshake), so the corrupted handoff is elsewhere and is unidentified. The **trigger** is named; the
**code site** is not.

**Correction to an earlier reading in this issue.** A per-arm *correlation* on the cwsr=1 control found
that faulting arms did **not** show more evictions than clean arms (203 against a clean mean of 254.5).
That is not in conflict: a rate-correlation and an intervention are different instruments. The
intervention shows eviction **causes** the fault; the correlation shows that under ambient pressure the
per-arm eviction *count* does not predict which arm faults. This also explains why an ambient-rate
control could not surface a mechanism at all: at the measured ~5-7 % the event rate is too low for any
correlation to see, which is why the intervention had to force the rate up first.

The fault reproduces on the shipped pin — `pair 2 arm c: FAULT-NaN nan=1 req=34 gtt_start=102G`, build
`10604 = 3971f49bf`, `parse=0` ruling out the format-500 mode — so the shared-memory tile-visibility change does
not eliminate it. The rate study is complete, and its per-arm regime gate and truncation accounting are reported
here — both halves ran at `gtt_start = 102 GiB` on all 70 arms, and parse-500 truncations are counted separately,
never scored as clean arms:

```
A  shipped fix  (10604 = 3971f49bf)   40 scheduled, 40 in-regime, 37 complete
   clean 66/66: 35   NaN faults: 2 (both parse=0, inside the gate)   rate 2/37 = 5.4%  (95% CI 0.7-18.2%)   parse-500: 3
B  always-stage (10605)               30 scheduled, 30 in-regime, 25 complete
   clean 66/66: 24   NaN faults: 1                                    rate 1/25 = 4.0%  (95% CI 0.1-20.4%)   parse-500: 5
Fisher exact two-sided p = 1.000;  P(rate_A < rate_B) = 0.460
```

So the tile-visibility pin did not measurably change the pressure-dependent all-NaN rate, and the fault is still
live on the shipped pin. The reading is bounded, not directional: both intervals contain the ~10% figure that
motivated the change and neither excludes "no change" (and that ~10% comparator itself rested on 1 fault in 15
arms, 95% CI ~0.2-32%, so the lower rate needs no environmental explanation). No runtime-side ordering gap on the
device-to-host path could be named, and no forward-pin fix candidate
survives (the only one proposed is an ancestor of the shipped pin, i.e. a revert).

**Candidate families closed by measurement and source reading (2026-10-13).** The block-split candidate
recorded here previously is WITHDRAWN: the engine's own profiler measures GLM's decode program as **one block with
zero block-boundary gaps**, at the default and at both alternate `HRX_AQL_BLOCK_SIZE` values, so the
"919 + 760 dispatches, two 128 KB blocks, 1,399 barriers per token" figure this file carried is not reproduced —
it is either a different workload or wrong. The following were also closed on 2026-10-13, and
`docs/315-STATUS.md` UPDATE 7 carries the full evidence paths:

- **The CWSR/preemption framing.** A fresh-booted `cwsr_enable=1` control (20 pairs, 27 in-regime arms) faulted
  **2/27 = 7.41 % with BOTH faces** (`p8l` Face B via HTTP 500; `p16c` Face A via `nan=1`), on opposite cells.
  A fresh boot at cwsr=1 therefore faults, so the 70-arm cwsr=0 clean campaign cannot be attributed to CWSR, and
  "cwsr=0 prevents it" is not the mechanism.
- **Eviction/buffer-movement as a rate mechanism.** On the arm-bounded pre-fault measure (the causal one — a fault
  cannot cause evictions that precede it) faulting arms show **no excess**: 203 evictions for the sampled faulting
  arm against a clean-arm mean of 254.5 (n=25, clean floor 117), and one faulting arm sat at 16. The predicted
  excess is absent, so the eviction count is not a discriminator for this fault.
- **The engine's own #123/#140 stale-buffer class, in GLM's path.** Read to conclusion: the two handoffs that
  class fixed carry **both** the store-wait and the LDS fence in GLM's kernel
  (`loom-libs/ops/...flash_attention_decode_split_f32_f16_wmma.loom:1170-1178`, plus the completion-counter
  handshake at `:1158-1180`), and four further hazard hypotheses were rejected by reading the code rather than
  asserted (`masked_value_stage` is per-subgroup so its subgroup barrier suffices; `probability_stage` is
  workgroup-wide but workgroup-barriered; `mul_mat_id_swiglu` writes a global it never reads; the q6_k handoff is
  cross-dispatch). The consumer's own read path was already cleared (`hsa_signal_wait_scacquire`, timepoint
  advanced after submit) and so was dispatch ordering.

**What that leaves as the honest position:** no candidate on the record survives its own discriminating test,
while the fault itself is real, reproducible, and now well characterised — at **~5-7 %** per in-regime arm, on
**both** faces, at **both** cwsr settings, on two different binaries, at 97-102 GiB. The single most useful
lesson for whoever picks this up: **every instrument on this issue was powered against a rate that turned out to
be wrong by ~4x**, so size any new experiment from a freshly measured rate, and prefer a design that RAISES the
event rate before testing a mechanism — which is what the #123/#140 work did with a separate process forcing queue
evictions, and what none of this issue's instruments did.

**Pin note (2026-10-09).** `engine@main` has since moved twice: first to `third_party/llama.cpp e44c9d01a4d5`
(#347), then to `0e255d774032` (#373, the fork tip after the NVFP4 / Q2_0 / IQ GET_ROWS merges), both with
`third_party/hrx-system 98d05d94`. That is the same runtime the study above ran against, and `26330cc20490` is an
ancestor of both, so the measurements above still describe what ships. Two things learned on the way:
a bump that moved `hrx-system` alone to `4ba76c18eafe` (2026-10-08) left `main` unbuildable with
`ONEBIT_HRX=ON` (the pinned `ggml/src/ggml-hrx/loom-jit.cpp` still used the LoomC API that revision removed; reverted
in #363, and #365 now holds `hrx-system` when LoomC symbols go missing); and with the port of `loom-jit.cpp`
(fork #97) that revision compiles but returns an immediate end-of-text on a 4.7K-token prompt where `98d05d94`
answers (#369), so `hrx-system` stays at `98d05d94` and moving it needs a multi-ubatch prompt check, not a symbol
check. The #373 pin passed that check: needle 20/20 on Qwen3-0.6B.

### Instrumentation available for the bisect (implemented, not yet built)

On `f2099e9b7`: a probe on the route the logits actually take (`backend_get_tensor_async` →
`hrx_stream_copy_buffer`) that names the producing graph `value`. It recovers the `ValueId` by searching the
retained `GraphProgram` value maps for the downloaded tensor pointer (`GraphProgram::graph()` exposes the
`ValueMap`; each `Value` holds `const ggml_tensor * tensor`), records at copy time, and scans after the
flush-and-wait that publishes the data, with a `=poison` mode so the reporting path can be proven to fire. It
compiles (single TU). It is deliberately unbuilt and unrun: relinking rebuilds every TU that includes
`backend-context.h`, and that CPU inside a GTT-bound measurement is the confound that already invalidated one run
of this experiment.
