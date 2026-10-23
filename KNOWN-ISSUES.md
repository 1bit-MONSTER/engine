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

The investigation notes and evidence cited below as `notes/315/...` are kept in a private repository.

**Status: RESOLVED 2026-10-22, closed.** The trigger is named (the CWSR context save area living in a
pageable SVM range on a CWSR-on box, which KFD eviction can migrate out from under a preempted queue), the
repair is shipped (`1bit serve` places the save area in GTT when `amdgpu.cwsr_enable=1`; engine PR #454,
pins in #455), and the acceptance soak this entry required **passed on 2026-10-18/19**: 37 in-band arms,
9 h 03 min in band, **0 graded faults**, every graded arm's save area traced as two GTT BOs with zero SVM
lookups. See "UPDATE 2026-10-19 - the CWSR-on soak passed" at the end of this entry for the numbers, the
bound, and what the result does not show. The earlier "no limitation is recorded" status is kept below for
the record; it is superseded, not deleted.

Owner acceptance, recorded 2026-10-07: the owner authorised landing this entry in writing — *"you have gh
authorization and do what needs to be done"* — on the basis that no forward fix is available. The limitation text
below was drafted by the assistant; the owner has not edited it and may amend or withdraw it at any time.

**Affected:** `GLM-4.7-Flash-Q4_K_M` on `HRX0`, `llama-server`, at high GTT occupancy (measured from
`gtt_start` ~83 GB up to ~110 GB). Raises at roughly 10-20 % per run; not deterministic and not reproducible on
demand, which is why it is recorded as a limitation rather than fixed.

### Reproduction

3 detached neighbour `llama-server` processes at default context drive GTT into the ~100 GB band, then one
detector server is exercised with 8 identical greedy requests separated by different-length distractors,
followed by 50 mixed-length requests. Under pressure the decoder produces corrupted logits, and the sampler
guard reports:

```
count=154880 of 154880, contiguous=yes, sentinel=0
```

`154880` is the model's vocabulary width — the whole LM-head output — and `sentinel=0` says every one of those
NaNs was **written by the compute**: not one is the sentinel payload the instrumentation pre-fills the logits
allocation with. The earlier reading here, that "the logits buffer is never written", is therefore falsified, and
it was unsafe from the start: the sentinel only ever fills the 619520-byte logits allocation (`ggml-hrx.cpp`,
`size == 619520`), so it says nothing about the tensors where the corruption is born. The guard fires on the
consumer side and reports the shape, not the producer — the producer is upstream of the final norm, which the
nan-dump fingerprint (notes/315/315-campaign/evidence/315-instrumented-arm-hunt.md) established by finding the first
NaN-bearing tensors in the FFN path.

### Both failure faces

The same underlying defect — one token's hidden row corrupted during a prefill — surfaces in two places
depending on what the corrupted memory holds (NaN, or stale finite values):

1. **All-NaN logits (Face A).** One token's hidden row is already NaN by the time the MoE aggregation adds run, so
   that token's KV slot is poisoned and the next decode attending to it hands the output norm an all-NaN hidden
   state: `count=154880 of 154880`. Across sixteen faulting arms the NaN count at the first affected tensor is
   **2048 in fifteen of them** — exactly one `n_embd` row — while the batch size varies four-fold (T of 87, 98,
   135, 140, 172, 306, 342), and the first affected node is an unnamed partial expert sum
   (`build_moe_ffn` names only the final aggregate). Mechanism, evidence and the discriminating experiment:
   notes/315/315-mechanism.md sections 40, 44, 47.

   **Correction, 2026-10-10.** That "one row" figure is a property of the instrument that measured it, not yet a
   property of the fault. Those arms used `GGML_HRX_TENSOR_FP=tail:16` (a window over the last sixteen graph
   nodes). Arm `315row-p1c`, in regime at `gtt_start=98G`, measured by *node name* instead and found the
   corruption in the routed-FFN **routing** tensors with the whole batch already NaN (`l_out-4`: 20480 of 20480
   floats, all ten tokens' rows; `ffn_moe_up-4`: 61440 of 61440), the earliest NaN-bearing node being
   `ffn_moe_weights-4` — the `ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` route-weight gather — while
   `ffn_moe_probs-4` carried none. Until the same instrument is run over the same arms, the safe statement is
   that the corruption involves the MoE routing tensors and that the row extent is instrument-dependent.
   Evidence: notes/315/315-campaign/evidence/315-row-p1c-inregime-mechanism.md.
2. **Format-500 (Face B).** The same event with *finite* contents. A faulting request whose 60-name tensor
   fingerprint included `norm`, `result_norm` and `result_output` logged **no NaN at all**, and the server rejected
   its output as `unparsed Content-only output: the}:}:}甚;` — garbage; the paired control arm produced a
   repetition loop, `a gaps gaps gaps ...`. The sampler is drawing from a corrupted distribution, the text is
   degenerate, and the failure surfaces at the server's chat-template parser instead of at the sampler guard.
   Evidence: notes/315/315-campaign/evidence/315-face-landscape.md.

A NaN-only count records the second face as a clean run, which is what makes it dangerous as a metric. Scoring
must be: an arm is valid only if it completes the request phase in full (66 of 66 requests); an arm that aborts
early has reduced exposure and is excluded, not counted as clean. Both faces must count — NaN logits through the
guard, format failures through HTTP status, UTF-8 and JSON parse failures — and a third signal is needed for finite
corruption that still parses: the detector's eight identical greedy requests must agree (`identical_distinct == 1`),
because no clean arm among the 142 recorded has ever been non-deterministic while all seven non-deterministic arms
were Face-A faults (`notes/315/315-preregistration.md` section 8, freeze note 2026-10-10).

### State as of 2026-10-10 (this section supersedes the mechanism text above wherever they differ)

Diagnostic record: `notes/315/315-mechanism.md` sections 42-51; evidence under `notes/315/315-campaign/evidence/`.

**Where the fault sits now.** The routed FFN's *routing* tensors, in the MoE path:
`ffn_moe_argsort` / `ffn_moe_topk` (the top-k ids, `src/llama-graph.cpp:1930-1935`) feed
`ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` and `ggml_mul_mat_id`. In arm `315row-p1c` the gather was NaN for two
whole tokens while the ids had never been instrumented at all, which is the measurement gap this entry now
carries as its most important limitation.

**A candidate with a mechanism, not yet a finding.** `ggml_argsort_rows_f32`
(`kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom:111`) computes each element's rank with *ordered*
comparisons (`cmpf ogt`/`olt`/`oeq`) plus an index tie-break. That is a total order for finite input, so the ranks
are a bijection and every output slot is written; **for a row holding two or more NaNs it is not** — all three
comparisons are false, two elements take the same rank, and one slot in `[0, cols)` is never written and keeps
whatever the buffer held before. A stale `i32` in the top-k view is an arbitrary expert id. The two consumers
disagree about it: the gather clamps the id (`small_rows_f32.loom:157+`), while the expert-table builders
consume it with `index.assume` *ranges* (`mul_mat_id_swiglu_f32_f32_wmma.loom:47-53`), not checks. Full reading:
`evidence/315-id-producing-kernel.md`.

**Falsified in this round** (each with its own evidence path). Arm counts below are **re-derived from the
run consoles** (`notes/315/315-campaign/tools/lever-census.py`, all arms at `gtt_start` 97-98 GiB, i.e. inside
the frozen `>= 95 GiB` gate), not from prose:

| candidate | run | control cell | fix cell | reading | evidence |
|---|---|---|---|---|---|
| memory coherence is the carrier (`HOST_COHERENT` on host-visible buffers) | `315accept` / face-landscape | — | **9 clean + 1 fault / 10** | not sufficient | `evidence/315-face-landscape.md` |
| consumer-side ordering is the carrier (`GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1`, a stream sync after every replay) | `315gsync2` | 1 fault / 4 | 0 / 4 | no change (Fisher p = 1.00) | `evidence/315-face-landscape.md` |
| ordering the instrument's own reads (`ggml_backend_synchronize` before each read) | `315row` vs `315fp2` | 1 / 1 (`315row` c) | 1 / 4 (`315fp2` l) | no change (p = 1.00) | `evidence/315-fp2-ordered-fingerprint.md` |
| forced AQL execution barrier on **every** packet (`HRX_FORCE_BARRIERS=1`, env-gated patch in `aql_block_processor.c`) | `315fbar` | 3 faults / 4 | 1 / 4 | no change | `315fbar-console.log` |
| flash attention is the carrier (`-fa off`) | `315nofa` | 1 / 4 | 2 / 4 | fault persists | `315nofa-console.log` |
| `HRX_COMMAND_BUFFER_MODE=pm4` | `315pm4` | 3 OK + 1 fault / 4 | 4 / 4 arms failed at `req=1` | **unusable**: every fix arm failed the *first* request with HTTP 500 - not a fault-rate comparison at all | `315pm4-console.log` |
| the expert/partition tables go stale | static | — | — | both are rebuilt by producer dispatches inside the same plan | `evidence/315-moe-routing-routed-ffn-static.md` section 3 |
| the fused path loses `route_stride` | static | — | — | that defect is already fixed in every tree that faults | the same file, section 7 |
| the engine's MoE expert-streaming cache | static | — | — | Vulkan-only; the arms never enter it | the same file, section 6 |

**Two arm counts in the 2026-10-10 issue comment were transcription errors, corrected here against the
consoles.** That comment gives the forced-barrier control as `2 / 4` and the `GRAPH_SYNC` fix cell as
`0 / 2`; `315fbar-console.log` records control **3 faults / 4** (pairs 1, 2, 4 fault; pair 3 clean) and
`315gsync2-console.log` records the fix cell as **0 faults / 4** (not 2). Neither correction changes any
reading: the barrier still shows no improvement, and the sync cell is still clean against a faulting
control. The comment's command section cites these same consoles as its source, so the consoles govern.

**Withdrawn (kept for its own record) - was "still untested, and next".** The ids themselves: instrumented
as `i32` with the name match anchored (the previous filter was `std::strstr(list, name)`, so `-41` also
matched `-4` and only layers 4 and 41 were covered). `notes/315/315-campaign/315-idprobe.sh` builds that probe
and runs four in-regime arms, reporting per row the first eight ids, the count outside `[0, n_expert)` and
whether the row's ids repeat. Duplicate or out-of-range ids make the argsort kernel the site; a clean
permutation kills it.

**This is no longer the next step.** The correction directly below supersedes it twice over: the by-name
instrument that would have produced these ids reads a transient arena slot's last writer (so no post-graph
by-name read can name a producer on this backend), and the `oor`/`dup_pairs` ids diagnostic was itself
**mis-specified** - `oor` was compared against `ne0`, which for the top-k *view* is `k`, not the expert
count, so essentially all of its ~253,000 output lines are flagged and none is usable. The probe exists and
runs; its output cannot answer the question it was built for. Do not treat this paragraph or
`315-idprobe.sh` as an untested lead, and do not cite `evidence/315-id-producing-kernel.md` as a live result.

**Correction, evening 2026-10-10 — this supersedes the two paragraphs directly above.** Both the routing-tensor
candidate and the by-name instrument that produced it are now withdrawn, and the ordering family is closed:

* the fingerprint reads the MoE intermediates out of one **host-visible, non-coherent 44 MB arena**
  (`size=44048640 memory_type=0x52`) through a plain `memcpy` (`ggml_backend_tensor_get` short-circuits for host
  buffers), and, worse, a **post-graph read by name returns that arena slot's last writer** - proven inside one
  arm by arithmetic, since `ffn_out-0` reads `nan=7048 of 1048576` while `l_out-0` reads `nan=4963`, and
  `l_out = ffn_out + attn_out` forbids `l_out` from having fewer NaNs. So no post-graph by-name instrument can
  name a producer on this backend, and the `oor`/`dup_pairs` ids diagnostic was mis-specified besides (`oor`
  against `ne0`, which for the top-k *view* is `k`, not the expert count);
* **four in-regime toggles, all null, every arm at `gtt_start=98 GiB`**: `GRAPH_SYNC` after each replay
  (control 1 fault / 4, fix 0 / 4); `ggml_backend_synchronize` before the instrument's reads (1/1 vs 1/4);
  `HRX_FORCE_BARRIERS=1` forcing an AQL barrier on every packet (3/4 vs 1/4); and `-fa off` (1/4 vs 2/4, the
  fault persists). `HRX_COMMAND_BUFFER_MODE=pm4` is not a lever at all - every fix arm failed the first
  request with HTTP 500;
* **the ordering/visibility super-family is closed at source as well as by run.** Both halves have a
  citation, not just a symbol name:

| claim | evidence |
|---|---|
| the host read of the logits is ordered: the accessor synchronises before returning | `src/llama-context.cpp:3728-3730` - `llama_get_logits_ith` calls `ctx->synchronize()` as its first statement |
| the wait is an **acquire** wait, not a relaxed poll | `runtime/src/iree/hal/drivers/amdgpu/semaphore.c:329` - `iree_hsa_signal_wait_scacquire(..., HSA_WAIT_STATE_BLOCKED)`, inside the epoch-wait loop that returns via `iree_hal_amdgpu_semaphore_epoch_is_reached` |
| the waited timepoint covers the **last** submission, so a stream wait cannot miss it | `semaphore.c:219` - `iree_async_semaphore_advance_timeline(base_semaphore, new_value, frontier)` runs in `iree_hal_amdgpu_semaphore_signal` *before* `dispatch_timepoints`; the public advance is `libhrx/src/libhrx/stream.c:240` (`hrx_stream_advance_timeline`) |
| the barrier bit is decided from **encoded flags only**, and the driver explicitly refuses to infer memory hazards | `runtime/src/iree/hal/drivers/amdgpu/aql_block_processor.c:103-116` (`..._packet_has_barrier`: execution-barrier flag, final flag, first-packet wait count, first-packet inline acquire scope) and `:119-126` (`..._packet_control`, containing the comment *"Do not infer memory hazards here by walking command operands."*) |

  Read together: the consumer's own path ends in a blocking acquire wait that is covered by the last
  submission's timepoint, and the packet barrier is a recorded property the driver will not infer. Forcing
  that bit on everywhere (`HRX_FORCE_BARRIERS=1`) changed the outcome in neither direction. That is why the
  family is closed for *this* fault rather than merely untested;
* **what survives**: the fault is real and common in regime (178 arms on the box, **131 in-regime, 34 faulting
  in-regime**, both faces, never in the same arm), the guard line is exactly
  `count=154880 of 154880 ... contiguous=yes`, and the one ordered by-name capture put the corruption at
  **layer 0 of a T=512 prefill**, whole-batch, cascading to a fully NaN residual stream by layer 1;
* **the best-matching lead, stated as a lead and not as a mechanism: the same *window* the engine's own #123/#140
  record describes, opening somewhere our levers do not reach.** The historical instance is written down in
  `docs/hrx.md` (section "MoE router expert-id fault fix" and the `00adc2b` section), and it is the only
  record in the tree whose mechanism explains *both* faces in one sentence and *the GTT dependence at all*:
  * a producer wave writes **global** memory, a consumer wave reads it back, and the barrier between them was
    `kernel.barrier<workgroup>` - which fences **LDS only** (no store wait, no `buffer_gl0_inv`). A consumer
    could therefore quantize the output slot's previous contents: *"The f32 output was right and `next_q8` was
    wrong, and NaN when the stale bytes decoded as NaN"* - wrong-but-finite (Face B) and NaN (Face A) from one
    defect, decided by what the stale bytes happened to hold. That is the same two-observables-one-event shape
    this issue measures;
  * *"The window opens when a dispatch is preempted, and any process's KFD queue eviction (page compaction,
    KSM, memory pressure) does that, which is why the rate followed those settings"* - a mechanism that
    explains the GTT dependence, which no other candidate in this issue does.

  **Both historical fixes are in the pin, verified as ancestors of `0e255d774` (`git merge-base --is-ancestor`):**
  `895d63f` (#123, the router argmax sentinel) and `00adc2b` (#140, `barrier<global>` then `barrier<workgroup>`
  in every decode-split `reduce_fused` variant). So the historical defect is fixed, and what is *not* closed is
  the same window opening somewhere else - most plausibly under memory pressure itself, where the preemption
  is induced by the driver rather than by an adversarial test. Today's static audit over the whole `loom-libs/ops`
  corpus and the forced-barrier run close this mechanism's **intra-dispatch** half for this model, but neither
  reaches a window opened by the driver's own eviction.

**The two instruments that could name it, and neither is built yet.** Named here so the next attempt starts from
this state rather than re-deriving it:

1. **Read a value at the point of production** - inside the dispatch, or from a non-transient copy taken at the
   producer. This is the instrument the backend forces: a post-graph, by-name read cannot work on this backend
   (it returns a transient arena slot's last writer, proven by arithmetic in an arm), so an in-dispatch read or a
   producer-side copy is the only route left.
2. **Correlate a faulting arm with the driver's own eviction/preemption events** - KFD SMI eviction records and
   the amdgpu preemption path, aligned to the arm's own timeline. This is the instrument that tests the lead
   above directly, and it needs an in-regime **faulting** arm to correlate; the profiled sessions so far produced
   clean arms only (see the next bullet), so its faulting half is still owed.

Neither instrument is a fix, and no limitation entry is recorded: a mechanism is not named, so `#315` stays OPEN
by this issue's own rule.

**No mechanism is proven, so no limitation entry is recorded**, and `#315` stays OPEN - a documented limitation is
this issue's own rule only *after* the mechanism is named. The next instrument has to read a value at the point of
production (inside the dispatch, or from a non-transient copy at the producer) or correlate a faulting arm with
the driver's own eviction/preemption events; both are named in the issue comment of 2026-10-10 evening.

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

### What remains unknown

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
survives (the only one proposed is an ancestor of the shipped pin, i.e. a revert). The open candidates are in the
kernel/binding family — narrowed as of 2026-10-10 to the routed-FFN routing tensors and the id-producing kernel,
with the two 128 KB AQL command blocks GLM's decode splits into (919 + 760 dispatches, 1,399 barriers per token)
still unexamined as an *ordering* question. See "State as of 2026-10-10" above.

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

### SUPERSEDED 2026-10-10: instrumentation for the bisect (implemented, not built — and now known to be unreachable)

On `f2099e9b7`: a probe on the route the logits actually take (`backend_get_tensor_async` →
`hrx_stream_copy_buffer`) that names the producing graph `value`. It recovers the `ValueId` by searching the
retained `GraphProgram` value maps for the downloaded tensor pointer (`GraphProgram::graph()` exposes the
`ValueMap`; each `Value` holds `const ggml_tensor * tensor`), records at copy time, and scans after the
flush-and-wait that publishes the data, with a `=poison` mode so the reporting path can be proven to fire. It
compiles (single TU). It is deliberately unbuilt and unrun: relinking rebuilds every TU that includes
`backend-context.h`, and that CPU inside a GTT-bound measurement is the confound that already invalidated one run
of this experiment.

---

## UPDATE 2026-10-13 — the condition is now named and it is `gtt_start`; the read-path fix is REFUTED

This supersedes the "no mechanism is proven" line at the top of this entry. It does **not** retire the
entry: the cause site is still not located, so the issue remains OPEN.

### The condition that governs the fault: measured, and it is occupancy, not the instrument

A census of every recorded arm on the box (all `315*-console.log` under `~/wt`, 1,675 lines, every line
matching `arm [qelc]: (OK|FAULT|OOR|INVALID)` parsed for cell, `gtt_start`, `gtt_now`) pools to:

```
    gtt_start=97G:  OK=150  FAULT=8    rate =  5.1%
    gtt_start=98G:  OK=145  FAULT=64   rate = 30.6%
```

**BUT THIS POOLED SPLIT IS CONFOUNDED AND MUST NOT BE READ AS A PROPERTY OF THE LEVEL.** Compared
within a single console — one session, one instrumentation — it does not survive:

```
    315ctl1 (NO forcer)   97G: 10 OK / 1 FAULT = 9.1%    98G: 15 OK / 1 FAULT = 6.2%   <- no level effect
    315disc2 (WITH forcer) 97G: 12 OK / 0 FAULT = 0.0%    98G:  1 OK / 5 FAULT = 83.3%
```

The whole pooled effect is carried by the forcer run, i.e. the 98G population is dominated by forcer
arms and the **forcer**, not the level, produced those faults. Within-cell breakdowns do not rescue it,
because cells are not independent of instrument (cell `e` is *defined* by having the forcer).

What is supported: the forcer causes faults (0/22 -> 7/10 pooled, arm A 4/4, arm B faulting with the fix
on); and individual arms have drifted into a high band and faulted while doing so (arm A quiet r2
97->102G; arm B cell `c` 97->106G). Isolated observations, not rates. The level question is open and is
what the pre-registered `315gttstep` run exists to test. See
`notes/315/315-campaign/evidence/315-gtt-start-is-the-discriminator.md`, whose correction section carries the
within-console numbers.

This reconciles four observations that previously did not fit together: the 0/70 campaign (run at 97G,
the 5% band, where ~3.6 faults were expected and 0 observed, P~3%) is a genuine low-band measurement and
never evidence of absence; cell `e`'s near-total fault rate is because the forcer drives arms straight
through the high band; the quiet- and CPU-cell faults occur at 98G starts or by drifting there; and the
falsified per-arm eviction-*count* correlation (faulting mean 109.5 vs clean 254.5) failed because a
count of evictions is not a measure of band occupancy.

Evidence: `notes/315/315-campaign/evidence/315-gtt-start-is-the-discriminator.md`,
`315-faults-track-gtt-drift.md`, `315-ab1-cell-c-fault-drift.md`, `315-0of70-has-no-eviction-cell.md`.

### The forced-eviction intervention, and the fix it motivated

Forcing KFD evict/restore via `~/wt/evictonly` raises the rate from 0/22 to 7/10 arms (Fisher exact
two-sided p = 3.6e-5), producing **both** faces under one intervention — the trigger-level mechanism.

The candidate repair that intervention motivated — routing `buffer_get`'s bare `memcpy` of the
non-coherent `0x52` logits mapping through `hrx_synchronous_d2h` — has been **REFUTED by its own
pre-registered A/B** (`fc7ffd5`, `notes/315/315-campaign/evidence/315-ab-preregistration.md`):

```
    cell e (forced eviction, the test cell)    arm A (knob OFF): 0 OK / 4 FAULT
                                               arm B (knob ON):  FAULT in round 1 (Face B) and round 2 (Face B)
```

The knob was confirmed live by reading the arm server's own `/proc/<pid>/environ` during the arm. The
refutation is not dismissible: the one objection that would make it vacuous — that `hrx_synchronous_d2h`
is itself an incoherent read — was checked at source and disproved, the runtime's staging pool being
created `HOST_LOCAL | HOST_COHERENT | DEVICE_VISIBLE`
(`host_queue_staging.c:427`, `315-ab1-fix-path-is-coherent.md`).

**The prepared patch must NOT be landed.** It is preserved at
`notes/315/315-campaign/patches/315-coherent-read-fix.patch` for the record only.

### Consequence for this entry's acceptance criterion

The criterion used here has been ">=60 in-regime arms, zero faults". That criterion is **satisfiable
without testing the defect**: 60 quiet arms run at the ambient ~5% rate while the fault lives elsewhere
entirely.

**An earlier version of this paragraph said the gate should move to "the high band (>=98G)". That is also
wrong, and is corrected here.** The cell-separated tally across every recorded console shows occupancy alone
does not do it:

    cell q (QUIET, no side load)          cell e (evictonly forcer)
      gtt_start=44G   0%  (n=4)             gtt_start=44G   0%  (n=4)
      gtt_start=97G   5%  (n=39)            gtt_start=98G  95%  (n=43)
      gtt_start=98G   0%  (n=2)

The quiet cell is **0% at 98G** against 5% at 97G — so `gtt_start` does not govern the fault, and a band-based
gate would still admit arms that cannot fault.

**The gate has to be the intervention cell, not a band.** The fault lives in cell `e`: ~95% under forced KFD
eviction against ~5% ambient, a 19x contrast on the same workload at the same level. Any claim that #315 is
gone must therefore show zero faults **in the forced-eviction cell** — and must report the ambient
(quiet-cell) rate alongside it, so that a clean result cannot be produced by simply not applying the
trigger. That is what makes the criterion falsifiable rather than satisfiable by omission.

### Families closed by discriminating experiment (2026-10-13)

Six candidate families are closed by a pre-registered or source-level test; a seventh (the FA
alignment class) is now **excluded as sufficient** on the frozen rig — an earlier attempt to exclude it
rested on a parse error and was retracted (`315-galign-CORRECTION-my-parse-was-wrong.md`), and this run
replaces it:

| family | disposition | evidence |
|---|---|---|
| the logits **read path** — routing `buffer_get`'s bare `memcpy` through `hrx_synchronous_d2h` | **REFUTED** by its own pre-registered A/B: cell `e` faulted 0 OK / 4 FAULT with the knob off and 0 OK / 2 FAULT with it on, both Face B, knob verified live in the arm server's `/proc` environ | `315-ab1-arm-B-interim.md`, `315-ab1-fix-path-is-coherent.md` |
| the **host side** — every host access to the non-coherent `0x52` mapping | **CLOSED**: all four accesses enumerated (`buffer_memset`, `buffer_set`, `buffer_get`, `buffer_clear`) and none is on the decode path | `315-write-side-unaddressed.md` |
| the **runtime's AQL submission policy** | **CLOSED**: `aql_block_processor.c` states in code that the runtime does not infer memory hazards, consuming only compiler-set packet flags, so it never derives a producer/consumer pair | `315-device-hunt-boundary.md` |
| the **graph reorder** (`GGML_HRX_GRAPH_ORDER`) | **REFUTED** by a pre-registered A/B: cell `e` faulted 0 OK / 4 FAULT in BOTH arms (reorder on and off), 100% either way | `315-gord1-final.md` |
| the **deferred host writeback** (`add_host_writeback` queued, flushed at `mark_stream_synchronized`) | **CLOSED by mechanism**: the queue is never populated, and the gate is `binding.host_data != nullptr` while the logits binding resolves to `nullptr` (allocation host-visible, binding device-addressable) | `315-deferred-writeback-gap.md` |
| the **FA alignment/dispatch** class (`GGML_HRX_FA_PARTIAL_ALIGN`, the #140 family) | **not required**: on the FROZEN rig (the driver that reproduces the trigger), arm A (align 4096) and arm B (align 256) gave IDENTICAL tallies — cell `e` faulted **3 of 3 arms in both**, at `gtt_start=98G` with the forcer live. So the layout the #140 oracle used to expose the divergence is not required for the fault. One-sided (3 arms per alignment cannot bound a rate difference), the same strength as the coherence and `GRAPH_SYNC` rows — not a measured null. | `315-align-frozen-result.md` |
| the **id-producing / argsort** family (route ids corrupted → tables consume a stale id) | **NOT excluded — unmeasurable with the current instrument, for a now-named reason.** The instrument read the **correct** tensors (`ffn_moe_argsort` = the argsort result, i32, `ggml.c:5347`; `ffn_moe_topk` = a `view_4d` of it) and two independent read paths (plain memcpy and synchronising `hrx_synchronous_d2h`) returned **identical** values — so neither the naming nor the read path was the defect. The cause is **arena slot reuse**: the MoE path lives in a single reused `TransientArena`, the fingerprint reads after the whole graph has run, and the allocator reassigns the slot of any tensor whose lifetime has ended — so an i32 rank tensor reads back as a later float tensor's data. This also downgrades the strongest recorded id lead (`ffn_moe_weights-4`: gather output NaN with finite source) to **indistinguishable from a read artifact**. Only end-of-graph tensors are faithful; the trustworthy statement is that the residual stream is **entirely NaN at the head**. | `315-idprobe-readpath-refuted.md`, `315-which-readings-are-faithful.md` |

| **where the corruption is born** (the search's location, not its cause) | **NARROWED to layer 0 or earlier — the first faithful localization on the record.** `l_out-<layer>` is the last node of its per-layer graph chunk, so it is the least displacement-exposed reading available, and in the reduced-selector session **all three Face-A arms carry NaN at `l_out-0`**: 1,984 of 200,704 (98 tokens) and **1,984 of 276,480 (135 tokens) — the same absolute count at different sizes** — growing monotonically to saturation by layer 15, after which `l_out-15…45` are all the **identical hash** `ba52bebcb7a7d383`. A clean quiet arm in the same session shows zero NaN at every layer. `1984 = 2048 − 64 = 7936 bytes = one 8192-byte row less its first 256 bytes`, and 256 is the alignment the record gives for the MoE tables. This **contradicts** `315-mechanism.md` §40's *"born at block 41, not inherited from block 0"* — a tension the `315row` file had already flagged and left unresolved — and it means the per-node leads at layer 41 and the head have been downstream of an already-saturated stream. **Not a named mechanism:** `ffn_inp-0` (layer 0's post-attention residual, faithfully readable by lifetime) and the embedding are still to be separated. | `315-nan-origin-is-layer-0.md`, `315-lifetime-is-the-faithfulness-rule.md` |
| **device-side access violations** (an out-of-bounds read/write by a specific kernel) | **not detected, but qualified.** Device ASAN was enabled via `HRX_HAL_SANITIZER=asan` (`runtime.c:576` → `options->asan.enabled`) with the engine's existing device-event sink installed by `GGML_HRX_LOG_DEVICE_EVENTS=1` (`ggml-hrx.cpp:1371`, before `hrx_gpu_initialize`). In-regime Face-A arms (`gtt_start=100G`, `nan=2`) produced **no ASAN report**. Wiring verified at every step, but **no positive control was run**, so this bounds the family only as far as the sanitizer's own detection goes. | `315-asan-wiring-verified.md`, `315-asan-preregistration.md` |
| **the logits region holding stale / never-written memory** (Face A read of an unfilled buffer) | **EXCLUDED by the engine's own sentinel, recovered from an existing run.** `GGML_HRX_SENTINEL=1` fills the 619520-byte logits allocation with payload NaN `0x7FC0DEAD`, and the sampler guard reports the match count as `sentinel=`. The `315nan` session armed it (fill line present in **20 of 21 arms**) and **all 17 fault lines read `sentinel=0`** — the device wrote the region, with NaNs. The fill was never read out at the time. | `315-sentinel-preregistration.md` |### Still open

The **cause** — which handoff reads stale data after an eviction — is not located, now after closing six
families. What IS established is the trigger, quantified on the quiet cell (the one cell with no instrument
confound): ambient **~5%** against **~95%** under forced KFD eviction, an order-of-magnitude contrast on the
same workload at the same GTT level (`315-quiet-vs-forcer-by-level.md`). That file also retires an earlier
band-based gate recommendation — the quiet cell's rate at 98G was first put at 0% on n=2, but a **second**
quiet-cell fault at 98G overturned that leg (1 in 3 arms pooled; rule-of-three on 0/2 is ~78%, so "zero" was
never supportable from it). The load-bearing figure is the cross-cell contrast at matched level, not the
quiet rate. So the gate must be the **intervention cell**, not a band. The surviving candidates are the ones §55/§56 left open (executor-level cross-dispatch ordering,
or a different class), and the MoE id-producing-kernel family under active work by another agent.

The **`GGML_HRX_FA_PARTIAL_ALIGN`** discriminator — the #140 FA layout/dispatch class, which §57 of
`315-mechanism.md` names as the cheapest never-applied lever and pre-commits to abandoning the FA path if
the rate does not move — is pre-registered at `315-align-preregistration.md`, not yet run.

### On the fix

**No forward-only fix is currently landable**, and the reason is structural rather than an absence of
ideas: the engine validates **program structure** (`verify_command_program`,
`validate_program_constant_access`) and never **memory contents**; the only content check is
`std::isnan(logits[i])` in the sampler, Face-A only. Nothing on the replay → restore → consume path can
distinguish bytes the device wrote from a pre-eviction stale copy, so a repair would need the runtime to
guarantee restore fidelity (out of scope: the driver/runtime lane) or would be a Face-A-only workaround the
entry's own rule forbids recording. Full argument: `315-no-fix-available.md`.

The prepared read-path patch is refuted and must not be
landed. The one defensive mitigation in the tree (`GGML_HRX_NAN_CONTINUE`, scrubbing NaNs to `-INFINITY`)
addresses Face A only and is a **workaround, not a repair** — and this entry's own rule is that a
documented limitation is acceptable only *after* the mechanism is named, so no limitation entry is
recorded.

The controlled confirmation of the `gtt_start` boundary (cell fixed, start stepped 97/98/99) is
pre-registered at `notes/315/315-campaign/evidence/315-gttstep-preregistration.md`, not yet run.

---

## UPDATE 2026-10-14 — the site is instrument-relative, the size is not, and Face B was never measurable

Session evidence: `notes/315/315-campaign/evidence/315-instrument-independent-summary.md` and the nine documents it
cites; all read from the archived arm logs at pin `e44c9d01a`, no new arms. Nothing above is edited.

### The condition is named (as UPDATE 2026-10-13 says) but the SITE is not a property of the defect

Three configurations already on the record separate the two candidate variables:

| condition | selector | probes | first NaN at |
|---|---|---|---|
| A — `315size` | 15-name `l_out` list | none | **layer 0** |
| B — `315lout4`, `315inp2` | `l_out` name list | `COPY_PROBE`+`SENTINEL`+`LOGITS_FP` | **layer 0** |
| C — `315nan`, `315fbar` | **`tail:16`** | same probes | **layer 41** |

A vs B varies the probes and the onset does **not** move; B vs C varies the selector and it **does**
(0 → 41). So the **read pattern** sets the onset, and `hrx-producer` shows `graph n_nodes=9`, making
`tail:16` read **every** node of a subgraph (~9 per chunk) against ~1–3 for a name list. That also explains
why `tail:16`'s recorded 16/17 yield did not reproduce: reading every node is a different regime, not a
failed replication. **Consequence for this entry: a fix aimed at a layer is unsound, because the layer moves
with the instrument.** Any "the corruption is at layer N" statement in this entry or its evidence is
instrument-relative unless the read pattern is fixed.

### What survives across the instruments: one hidden-state row, at the front of the buffer

- **Size.** At the first corrupted tensor the count is one `n_embd`-wide row: `2048` floats = **8,192 bytes**,
  or `1984` = **7,936 bytes** = one row minus one 256-byte alignment block. Five of seven non-saturated onset
  readings fall in 7,936–8,192 B across three configurations. `nan = 2048` *exactly* in 18 of 19 arms is a
  property of condition C, not of the defect.
- **The count is a complete read, not a truncation.** In all four arms where the `hrx-finite-fp` channel
  printed, `finite + nan` equalled the tensor extent **exactly** (77,824+2,048=79,872; 350,208+2,048=352,256;
  710,656+2,048=712,704).
- **Position: row 0 or 1.** In the unbiased subset — readings where `nan` is *exactly* one row, so the
  reported row is the true position and not the first of several — **12 readings across 4 arms are all at
  row 0 or 1** (`row/T` ≤ 0.003), at `T = 39, 172, 348`, and **never the last token (0 of 12)**. Across all
  1,525 multi-token readings the corrupted extent is **median 348 rows**, so one row is the onset and the
  rest is causally-masked propagation to the full head.
- **The `l_out-N` per-layer read is not a per-layer measurement.** `l_out-42 ≡ l_out-44` and
  `l_out-43 ≡ l_out-45` return byte-identical, 25 %-NaN content in the `315fbar` cold cell — aliased reads
  from arena slot reuse. Scope: sound only for **unsaturated** reads; the stride-3 family in `315lout4` is
  fully saturated and is **not** evidence. The unnamed `node_NNNN` reads do not alias (only
  `node_3304 ≡ node_3305`), so **two readings with the same hash are one reading**.

### Face B has never been recorded with an instrument that can see it

- **No arm on the record carries both a fingerprint and Face-B evidence.** The fingerprint's only gate is
  `if (fp_nan_only && nan == 0) continue;`, so an arm with fingerprint lines has NaN and an arm with Face B
  has none — the two faces are **mutually exclusive in the instrument's output** under `NANONLY`. Every
  Face-B statement in this entry therefore rests on arm-level verdicts, with no tensor-level measurement
  behind it.
- **Its signature is otherwise invariant**: `nan=0 http=1 s500=1 unparsed=2` in all 15 Face-B arms, in **all
  three cells including the quiet cell** (`side=0/0`, three runs) and even one out-of-regime arm — so Face B
  is not eviction-specific either. Counting subtleties: `unparsed=2` is a *log-line* count for one event,
  `http=1 s500=1` is the same failure counted twice, and `req < 66` alone forces `FAULT`, so a `FAULT` verdict
  is not by itself content evidence.
- **Invalid UTF-8 has never been observed.** Of 3,200 request records and 267 greedy-output files, **zero**
  bodies are invalid UTF-8. The three apparent cases are 32-byte *excerpt* truncations (`utf8_ok = True` and
  `json_ok = True` on the full bodies); the 34 `utf8_ok = False` records are transport failures with no body.
  The recorded Face B is **Content-only / non-JSON output answered with HTTP 500**. A "third manifestation"
  hypothesis (wrong but well-formed output at HTTP 200) was tested and **withdrawn** — clean quiet-cell arms
  reach the same maximum repetition ratio.

### Two harness defects found and one fixed

- **The acceptance harness could not attribute Face B** — and it failed twice over: its condition was
  `parse > 0 or utf8 > 0`, which are **0 in every Face-B arm**, and `315-paired.sh` computed
  `http`/`transport`/`s500`/`unparsed` and used them in its **verdict** but never **printed** them. Both fixed
  in `notes/315/315-campaign/315-campaign.sh` and `315-paired.sh`, verified end-to-end on both branches (a real
  Face-B signature now attributes; an all-zero fault still does not). **Any future campaign must detect
  Content-only/HTTP 500 — a UTF-8-validity check would never fire and would pass a faulting arm as clean.**
- **The "rule of three" bound prints probabilities above 100 %** for `n < 3` (`300 %` at n=1). Correct at the
  criterion's n=60 (5 %), so the intended reading is unaffected; not changed, because deciding what an
  under-powered campaign should say is a criterion question.

### An instrument that has never been enabled, and one that cannot reach the site

- **`GGML_HRX_PRODUCER_PROBE`** fills a host-visible region with its own payload `0x7FC0DEAD` **before** each
  hooked graph and classifies **after a drained stream**, reporting `sentinel`/`nan`/`finite` and the NaN's
  **bit pattern**. The archive contains **zero** `[hrx-producer] armed` and **zero** `[hrx-producer] anomaly`
  lines — it has never once been enabled — and **no arm has ever logged a NaN payload** (the only 8-hex
  patterns in the logs are copy-probe `host_ptr=` addresses). Pre-registered as `315prod`; queued.
- **It cannot reach the site this entry now points at.** Its predicate (`is_host` **and** 256 KiB–8 MiB)
  admits exactly one tensor, the **logits** region; layer residuals print `host=0` and can never be armed. A
  **device-buffer** variant is proposed, pre-registered, and its patch written and anchor-verified against a
  real source copy (two defects caught in review; **compile NOT claimed**) at
  `notes/315/315-campaign/evidence/315-device-buffer-probe-proposal.md` and
  `notes/315/315-campaign/tools/patch-device-producer-probe.py`.

### On the fix — unchanged conclusion, one new reason

**No forward-only fix is landable**, as above, and now for an additional reason independent of the
content-validity gap: the site is instrument-relative, so a change aimed at "layer 41" would be aimed at a
read artifact. The instrument-independent fact available to a fix is the **size** (approximately one
hidden-state row, 7,936–8,192 B) and the position (front of the buffer) — both of which describe *what* is
corrupted without saying *what writes it*, since no instrument on the record observes the writing.

**Nothing here retires this entry.** The defect is not gone: Face A's one-row front-of-buffer corruption is
measured in 18 of 19 arms, and Face B's Content-only/HTTP-500 signature appears in 15 arms including control
cells. What has changed is that the mechanism cannot be named with the instruments that exist, and the site
in every earlier reading was an artifact of the instrument that produced it.

### 2026-10-15 — the mechanism is named and a forward-only fix is landed; the acceptance campaign is running

**Status line unchanged: OPEN**, because the acceptance criterion is met by the campaign below and its
result is not in yet. What has changed is that the mechanism is now named, which is the precondition this
entry's own rule sets for anything to be recorded.

**The mechanism.** `llama_context::decode` obtains the logits through an **asynchronous** read that this
backend never completes, and the decode's own completion wait is commented out on a premise that is false
here. Four links, each at file:line:

1. `src/llama-context.cpp:1483-1489` extracts the logits with `ggml_backend_tensor_get_async(...)`.
2. `ggml/src/ggml-backend.cpp:268-280` **skips its own `ggml_backend_synchronize`** when the backend
   implements `get_tensor_async` — and HRX does.
3. `ggml-hrx.cpp:454-473` (`backend_get_tensor_async`) enqueues `hrx_stream_copy_buffer` on the stream and
   **returns without waiting**. Measured, not assumed: the engine's read is
   `name=result_output bytes=619520 host_dst=0x7fed46100000 registered=1` — the registered-host-buffer
   branch, which has no wait at all.
4. `src/llama-context.cpp:2087`, at the end of `decode`:
   `// wait for the computation to finish (automatically done when obtaining the model output)` followed by
   `//synchronize();`. The premise is false for a backend that implements the async read, because obtaining
   the output *is* links 1-3. The two other `synchronize()` calls in `decode` do not cover this path
   (one is guarded by `if (!embd_seq.empty())` — embeddings; the other is in `encode`, not `decode`).

So `llama_decode` returns with the copy — and possibly the graph — still in flight, and the sampler reads
the destination: the **previous** token's logits (finite, plausible, wrong → the format-500 body) or a
region **not yet written** (→ all-NaN logits). Intermittency is whether the copy has completed.

This explains the levers already falsified on this record without contradicting any of them: coherence is
irrelevant (§13 — a missing wait is not a memory-visibility property), the earlier sync cell missed it
because it synchronized at the end of `graph_compute`, *before* the copy is issued, and the read-ordering
work ordered the instrument's reads rather than the engine's.

**The fix, forward-only.** Fork commit `47a8fe929` on `1bit-MONSTER/llama.cpp` branch
`1bit/315-logits-read-wait` — wait for the async logits read where it is issued — and the engine pin on
`1bit/315-preregistration` moves `e44c9d01a → 47a8fe929` (`95238dc`), with
`git merge-base --is-ancestor e44c9d01a 47a8fe929` true. It is not a revert.

**The campaign.** Pre-registered before it ran
(`notes/315/315-campaign/evidence/315-campaign-preregistration.md`): 60 in-regime **fix** arms plus 20
interleaved **control** arms on the unfixed build, same harness, same regime gate `[95,110] GiB`, one
window, instruments off, stopping at the first fix-cell fault. Zero faults in 60 arms gives a one-sided
95 % upper bound of `1 − 0.05^(1/60) ≈ 4.9 %`. A clean fix cell with a control cell that never faults is
reported as **under-powered, not passed**.

**Evidence paths:** `notes/315/315-campaign/evidence/315-MECHANISM-unwaited-async-logits-read.md` (the mechanism
and its falsifier), `…-campaign-preregistration.md`, `…-which-binary-is-which.md` (cell F =
`~/wt/315logwait-build` at `47a8fe929`; cell C = `~/wt/315fixlog-build` at `e44c9d01a`; both with
`hrx-f32-sync` and `hrx-wb-read` = 0).

### 2026-10-15 (later) — the logits-wait mechanism is FALSIFIED by its own campaign; the fix stays landed

The mechanism recorded above was tested by the pre-registered campaign it named, and it failed on the first
fix arm. The full record is `notes/315/315-campaign/evidence/315-logits-wait-FALSIFIED.md`.

```
pair 1 arm c (control, e44c9d01a): OK    req=66  gtt_start=98G
pair 1 arm l (fix,     47a8fe929): FAULT req=9   gtt_start=98G
STOP: a fix-cell arm faulted or was invalid — acceptance requires zero faults
```

The fix arm's `detector.json` reads `http_failures: 1`, `first_failure: "HTTP 500"`, `transport_failures: 0`,
and its `server.log` carries the **Face B** signature verbatim — *"the model produced output that does not
match the expected Content-only…"* — with invalid UTF-8 immediately before it and `HRX returned NaN`
appearing zero times. In gate at `gtt_start 98GiB`. It is counted as a fault, not an invalid arm, because
the truncation is a *consequence* of the defect rather than an independent harness death.

**So: the acceptance criterion is NOT MET, and the unwaited-async-logits-read mechanism is removed as the
cause.** The prediction was universal — *the faults stop* — and one in-regime fault refutes it, as one
counter-example refuted the deferred-writeback candidate. It is not evidence that the wait is wrong; it is
evidence that the wait is **not sufficient**, so the unwaited read is not the mechanism. No earlier
candidate is revived: coherence (§13), the deferred writeback, and the read-path families all stay closed.

The fix commit `47a8fe929` and the pin move `e44c9d01a → 47a8fe929` **remain landed** — a correct
forward-only change for the reason its comment gives, and the objective's criterion for a forward-only
change is satisfied by it. What is not satisfied is the mechanism and the campaign.

**Standing after this:** no named mechanism. Of the candidates that survived the campaign, one has since been
removed by its own pre-registered experiment and one has been shown unmeasurable by the current instrument.

* **The stream-ordering explanation for the `result_norm` divergence is REMOVED.**
  `notes/315/315-campaign/evidence/315-task1-writer-not-stream-ordered.md` ran the pre-registered
  `315-PREREG-sync-between-reads.md` on a faulting arm (Face A, 312 records) and landed on its **second**
  pre-declared row: of 82 records where `result_norm` differed between two consecutive plain reads, a fresh
  `ggml_backend_synchronize` stopped the difference in **0** and did not stop it in **82** — and changed the
  value itself in all 82. So the writer is **not ordered by `context->stream`**, which removes
  `GraphExecutor::execute` returning without a wait (with `backend_synchronize` covering only
  `context->stream`) as the explanation. Recorded at `b7d6b46`.
* **§51's rank-sort precondition is UNMEASURED and not measurable by the by-name fingerprint.**
  `notes/315/315-campaign/evidence/315-task2-bounded-negative-precondition-unmeasurable.md` (`ba6753a`) — the
  faulting arm produced zero records because `ffn_moe_probs-*` is not among the 946 node names this build's
  fingerprint walks. This is a negative about the instrument, not about the site.

**Surviving family, stated as surviving:** an open writer of the `result_norm` slot that is *not* ordered by
`context->stream` — the reused-arena-slot family in
`notes/315/315-campaign/evidence/315-what-the-synchronise-drains.md` — plus the unmeasured §51 site. The measured,
unexplained fact handed forward is that `result_norm` changes between two consecutive identical plain reads in
a faulting arm, at two different addresses, with no pending host writeback at the engine's logits read, and
with a fresh synchronise failing to stop it.

**No regression (criterion 4), measured at the moved pin `47a8fe929` and not the old one:** engine CI is
**green, 7 of 7** on `1bit-MONSTER/engine` PR #414 (`mergeable=MERGEABLE`, `state=CLEAN`); the HRX suite and
the `MUL_MAT_ID` op-test are **unchanged** (`8 pass, 0 skip, 0 fail`, `mxfp4` nmse **1.34e-05 … 1.61e-05**, an
exact match to the recorded baseline, same two kernels); and the dense-model control (Qwen3-8B, dense, not MoE)
is **clean and unchanged** (50 responses, 1 distinct content, zero fault signatures). Evidence:
`notes/315/315-campaign/evidence/315-task3-op-test-unchanged-at-moved-pin.md` and
`315-task3-dense-control-clean-at-moved-pin.md`.

**CI did not pass on arrival.** It required fixing two pre-existing defects — a half-move (`95238dc` moved the
`llama.cpp` gitlink without `registry/architectures.json`) and 46 files missing the full copyright notice —
recorded in `notes/315/315-campaign/evidence/315-what-the-merge-and-the-gates-found.md`, with no gate loosened.

### 2026-10-18 — candidate 1: the CWSR save area in GTT (CWSR off stays the shipped default)

On a CWSR-on boot, the same binary ran the #315 composition (GLM-4.7-Flash Q4_K_M, 4 neighbour
servers, 16 GiB ballast, `gtt_start` 104 GiB) with and without
`HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1`. With them, every queue's CWSR save
area is a KFD GTT BO instead of an SVM range (checked per run with kprobes): **46 runs, 0 faults**
(42 clean in band; 4 excluded for heat or another agent's NPU job, all four OK). Without them:
**16 runs, 4 faults**. Fisher exact two-sided p = 0.0043. 0 of 42 bounds the fault rate at 6.9 % or
less (one-sided 95 %); it does not show zero.

`1bit serve` now sets the three variables for its HRX backends when amdgpu runs with CWSR on
(`--hrx-cwsr-placement`, default `auto`; docs/hrx.md "CWSR save-area placement"). With CWSR off it
changes nothing.

**CWSR off (`amdgpu.cwsr_enable=0`) stays the shipped default** (the soak below does not change it; the
default is the owner's decision). The soak this paragraph required ran on 2026-10-18/19 and passed -
see the closing section.

---

## UPDATE 2026-10-22 — the CWSR-on soak passed, the condition is confirmed, #315 is closed

Acceptance soak run 2026-10-18 15:56 (boot) to 2026-10-19 04:52 (box time) on strixhalo, on a **CWSR-on**
boot (`amdgpu.cwsr_enable=1`, stock amdgpu, kernel `7.3.0-rc4-1bit`), with the engine build deciding the
placement itself (no hand-set environment), then restored to CWSR-off. Build: engine `9228e5a`
(llama.cpp `8d9ce2c1a`, hrx-system `82804197b` with the `HRX_ALLOW_NO_SVM` gate), `1bit` sha256
`ccbf5991f6c4…`. Composition: 4 `1bit serve` neighbours of GLM-4.7-Flash (`-c 2048`) + a sized GTT ballast
(no kernels, no queues) + one graded `1bit serve` GLM-4.7-Flash (`-c 8192 -fa on`) reloaded per arm; every
GPU process is a `1bit serve` of that build, so every HRX queue got its placement through the engine.

| item | value |
|---|---|
| in-band time | **32,575 s = 9 h 03 min** (band = graded arm `gtt_start` in [100,108] GiB) |
| in-band arms | **37**, every one `verdict=OK`, `in_band=1` (0 out of band, 0 INVALID) |
| `gtt_start` per arm | 101-102 GiB; `gtt_peak` 110-111 GiB; `avail_min` 7-8 GiB |
| graded detector | 66 requests per arm (`~/wt/315-detector.py`), **37 arms, 0 failures** (`det_http_failures=0 det_utf8_failures=0 det_parse_failures=0 transport_failures=0`, `identical_distinct=1` every arm) |
| graded chat | 20 greedy `/v1/chat/completions` per arm, **740 requests, 0 bad, 0 mismatch** |
| background load | 4,791 neighbour requests, **2 bad** (both HTTP 502, isolated; the next request to the same server answered 200; no NaN, no 500, no backend exit, no watchdog kill - recorded, not graded) |
| NaN aborts | **0** (`HRX returned NaN` absent from every server log) |
| HSA faults / backend exits / watchdog kills / arm hangs | **0 / 0 / 0 / 0** |
| dmesg | page faults **0**, ring timeouts **0**, GPU resets **0**, OOM **0**; Tctl peak 83 C, 6 gate pauses, 0 freezes |
| **placement, graded arms** | **37 of 37**: `csa_gtt=2 csa_svm=0 kbg_fail=0`, engine line `HRX: CWSR is on; CWSR save area placed in GTT (engine#315)` in every serve log, llama-server child's environ exactly `HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1` |
| neighbours | `csa_gtt=2 csa_svm=0 gtt_line=1` for all four; their rows carry `env_ok=-1` because the environment snapshot raced the exec, and a stale `zaya.pids` file from an earlier attempt raised `DRIFT` in `counters.txt` (the run's own stop flag never fired; a fresh counter pass prints `placement_drift=0`) |

**Against the control, same boot, same composition.** Phase B (the hand-set environment, `PHASE-B-GO.md`)
faulted **4 of 16** runs without the placement; this soak faulted **0 of 37** arms with it (engine-decided).
Fisher exact, two-sided: **p = 0.0062**. The pre-registered criterion was >= 6 h in band and >= 42 arms
with zero faults; the run reached 9 h 03 min and stopped at its 9 h wall cap with 37 arms, so per the
orchestrator ruling of 2026-10-19 01:30 it is graded **PASS (cap-limited)**, with the arm count reported.

**What this does not show.** 0 of 37 bounds the per-arm fault rate at **7.8 %** (one-sided 95 %), it does
not make the fault impossible; the optional positive control (`--hrx-cwsr-placement=off` on the same boot)
was not run, so the rate contrast rests on Phase B rather than on a same-session control; and the soak ran
with the ballast rather than ZAYA1-8B alongside (ZAYA was dropped on 2026-10-18 19:19 for memory headroom -
GTT 112 GiB left 4 GiB MemAvailable and tripped the guard on otherwise-clean arms).

**CWSR-off remains the shipped default.** The repair makes a CWSR-on box safe; whether to ship CWSR on is a
separate owner decision and nothing here recommends it. CWSR-off evidence on the record: 11 and 10 in-regime
arms in two campaigns, 0 faults (`~/wt/315cwsr0-console.log`, `~/wt/315cwsr60-console.log`).

Evidence on the box: `~/wt/315soak/{SUMMARY,STATE.md,PROVENANCE,counters.txt,arms/}`,
`~/wt/315soak/gate/PASS` (placement gate), `~/wt/315-PATCH-STATUS.md` §11; the plan it followed is
`~/wt/315c1/SOAK-GO.md`.

### Addendum 2026-10-23 — hw-gate on real hardware, the thermal threshold, and what is still unmeasured

Three things the closing section above leaves implicit, recorded here so the entry can be read without
the issue thread.

**1. hw-gate (the six-check correctness gate) was run on this fix's PR, on strixhalo.** `hw_gate` is
`skipped` for ordinary PR events by design (it runs on the `hw-gate` label, on `bump-hrx/*`, or by hand),
so it was dispatched by hand on PR #518's head. Run
<https://github.com/1bit-MONSTER/engine/actions/runs/39892647547>: the logits-pin under test was
`cc65151` with llama.cpp `af04c9f22b97` and hrx-system `82804197bcbe`, 338 s, 0 amdgpu ring timeouts,
Tctl 60 °C at the end. **The run was cancelled when the box went down** (it is unreachable as this is
written) part-way through the sixth model, so its verdict is "not completed", not "failed on a check".
Everything that did run passed:

| check | result |
|---|---|
| build | PASS |
| ops | PASS — **695/695** tests (MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, GET_ROWS, ROPE, RMS_NORM, SOFT_MAX) |
| fa-decode-q8 | PASS — 6/6 (K q8_0 × V f16 and q8_0; Hq/Hkv 16/8 and 32/8; n_kv 1031, 2048, 4103) |
| kat | PASS — qwen3-0.6b, zaya1-8b, qwen3.8-27b, qwen3-coder-30b-a3b, qwen3-4b-iq4xs, qwen3-0.6b-kq8 (9 cases each) |
| needle | PASS — qwen3-0.6b 7,082-token, zaya1-8b 7,403-token, qwen3-0.6b-kq8 7,082-token (all answered 7341) |
| placement | PASS — all on HRX in every graph: 170 / 193 / 152 / 99 / 99 / 122 graphs (largest 986 / 6,449 / 3,655 / 3,030 / 1,266 / 1,154 nodes) |
| kld qwen3-0.6b | PASS — mean KLD 0.000003, p99 0.00004, **same top 99.90 %** |
| kld zaya1-8b | PASS — mean KLD 0.007729, p99 0.11945, **same top 98.33 %** |
| kld qwen3.8-27b | PASS — mean KLD −0.000000, p99 0.00004, **same top 100.00 %** |
| kld qwen3-coder-30b-a3b | PASS — mean KLD 0.000912, p99 0.01119, **same top 98.97 %** |
| kld qwen3-4b-iq4xs | PASS — mean KLD 0.000002, p99 0.00005, **same top 99.85 %** |
| model qwen3-0.6b-kvq8 | **cut off** — rc 143 (SIGTERM) from the cancellation, not a check failure; its placement line reads 1322 nodes, 0 off HRX |

**2. The thermal threshold in the soak was 96 °C, not 93 °C.** The #315 soak followed phase B's rules:
detector/chat drivers pause at 92 °C, resume at ≤ 84 °C, and only ≥ 96 °C aborts (`~/wt/315soak/STATE.md`,
2026-10-18 17:19, an orchestrator ruling after an arm was lost to 92-93 °C). The measured peak across the
whole run was **83 °C**, with 6 gate pauses, 0 freezes and 0 aborts, so no event came near either
threshold — but the stated 93 °C rule was not the one in force and that is recorded rather than smoothed
over.

**3. What is still unmeasured, and what it would take.** None of these are hidden behind the PASS above:

| gap | why | what would close it |
|---|---|---|
| GLM-4.7-Flash's own **KLD / same-top** | no stored base logits exist for this model on the box (the gate has bases only for qwen3-0.6b, qwen3-4b-iq4xs, qwen3-coder-30b-a3b, qwen3.8-27b, zaya1-8b), and producing one costs 1-2.5 GB on a disk that was at 14 GB free with other agents' work resident | add `glm-4.7-flash` to `MODEL_TABLE` with a stored base at the pin, then `hw-gate.sh --phases kld --models glm-4.7-flash` |
| GLM **throughput** at this composition | the box went down before the `llama-bench` run | `llama-bench -m GLM-4.7-Flash-Q4_K_M.gguf -dev HRX0 -ngl 99 -fa on -p 512,2048 -n 128 -r 3` on the same build |
| the **intervention cell** (forced KFD eviction, `~/wt/evictonly`) re-run **with** the fix | the #315 regime needs ~100 GiB of GTT with 4 GLM-4.7-Flash neighbours, which needs the services-off window the owner gates (`~/soff.sh`, BOX-CONTROL.md queue item), and the box is down | one services-off window: composition as in `~/wt/315soak/soak.sh` plus the forcer, ≥ 8 fresh arms, 0 faults expected under CWSR off |
| a **trace of KFD migrating the save area** during a faulting run | the record traces the save area's **allocation class** (`kfd_queue_buffer_get` → GTT vs `kfd_queue_buffer_svm_get` → SVM, `~/wt/315c1/{csa.bt,alloc.bt}`), not the eviction event itself | a kprobe on the KFD eviction/restore path keyed by the CSA range, on a CWSR-on boot with the forcer |

On the regression test, precisely: the fix's own commit (#454) added `tests/hrx_cwsr_test.cpp`, whose
`auto + CWSR on` case asserts the three variables and the GTT line on **any** box (it feeds
`decide_cwsr_placement` directly), and `tests/hrx_cwsr_route.sh`, whose CWSR-on block is skipped when
`amdgpu.cwsr_enable` reads 0 — i.e. on the shipped configuration the unit test covers the decision and the
end-to-end route test does not. A `ONEBIT_CWSR_ENABLE_FILE` seam that would have let the route test cover
it on a CWSR-off box was written and **withdrawn**: CodeQL flags an environment-derived path reaching
`fopen` (`cpp/path-injection`, alert 29), and this repo does not loosen a gate to land a test.

Status of the documentation itself: this entry and the docs change ship in **PR #518, open and awaiting
the orchestrator's merge** (auto-merge is off by rule); #315 is closed on the strength of the soak above.
