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

## engine#315 — intermittent all-NaN logits and format-500 on GLM-4.7-Flash under GTT pressure

**Status: OPEN. No limitation entry is recorded.** The earlier status line here read "accepted as a
documented limitation"; that is superseded. This issue's own rule is that a documented limitation is
acceptable only *after* the mechanism is named, and no mechanism is proven - so nothing is recorded and
the issue stays open until the criterion below is met. See the correction at the end of this entry.

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
nan-dump fingerprint (docs/315-campaign/evidence/315-instrumented-arm-hunt.md) established by finding the first
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
   docs/315-mechanism.md sections 40, 44, 47.

   **Correction, 2026-10-10.** That "one row" figure is a property of the instrument that measured it, not yet a
   property of the fault. Those arms used `GGML_HRX_TENSOR_FP=tail:16` (a window over the last sixteen graph
   nodes). Arm `315row-p1c`, in regime at `gtt_start=98G`, measured by *node name* instead and found the
   corruption in the routed-FFN **routing** tensors with the whole batch already NaN (`l_out-4`: 20480 of 20480
   floats, all ten tokens' rows; `ffn_moe_up-4`: 61440 of 61440), the earliest NaN-bearing node being
   `ffn_moe_weights-4` — the `ggml_get_rows(ffn_moe_probs, ffn_moe_topk)` route-weight gather — while
   `ffn_moe_probs-4` carried none. Until the same instrument is run over the same arms, the safe statement is
   that the corruption involves the MoE routing tensors and that the row extent is instrument-dependent.
   Evidence: docs/315-campaign/evidence/315-row-p1c-inregime-mechanism.md.
2. **Format-500 (Face B).** The same event with *finite* contents. A faulting request whose 60-name tensor
   fingerprint included `norm`, `result_norm` and `result_output` logged **no NaN at all**, and the server rejected
   its output as `unparsed Content-only output: the}:}:}甚;` — garbage; the paired control arm produced a
   repetition loop, `a gaps gaps gaps ...`. The sampler is drawing from a corrupted distribution, the text is
   degenerate, and the failure surfaces at the server's chat-template parser instead of at the sampler guard.
   Evidence: docs/315-campaign/evidence/315-face-landscape.md.

A NaN-only count records the second face as a clean run, which is what makes it dangerous as a metric. Scoring
must be: an arm is valid only if it completes the request phase in full (66 of 66 requests); an arm that aborts
early has reduced exposure and is excluded, not counted as clean. Both faces must count — NaN logits through the
guard, format failures through HTTP status, UTF-8 and JSON parse failures — and a third signal is needed for finite
corruption that still parses: the detector's eight identical greedy requests must agree (`identical_distinct == 1`),
because no clean arm among the 142 recorded has ever been non-deterministic while all seven non-deterministic arms
were Face-A faults (`docs/315-preregistration.md` section 8, freeze note 2026-10-10).

### State as of 2026-10-10 (this section supersedes the mechanism text above wherever they differ)

Diagnostic record: `docs/315-mechanism.md` sections 42-51; evidence under `docs/315-campaign/evidence/`.

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
run consoles** (`docs/315-campaign/tools/lever-census.py`, all arms at `gtt_start` 97-98 GiB, i.e. inside
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
matched `-4` and only layers 4 and 41 were covered). `docs/315-campaign/315-idprobe.sh` builds that probe
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
`docs/315-campaign/evidence/315-gtt-start-is-the-discriminator.md`, whose correction section carries the
within-console numbers.

This reconciles four observations that previously did not fit together: the 0/70 campaign (run at 97G,
the 5% band, where ~3.6 faults were expected and 0 observed, P~3%) is a genuine low-band measurement and
never evidence of absence; cell `e`'s near-total fault rate is because the forcer drives arms straight
through the high band; the quiet- and CPU-cell faults occur at 98G starts or by drifting there; and the
falsified per-arm eviction-*count* correlation (faulting mean 109.5 vs clean 254.5) failed because a
count of evictions is not a measure of band occupancy.

Evidence: `docs/315-campaign/evidence/315-gtt-start-is-the-discriminator.md`,
`315-faults-track-gtt-drift.md`, `315-ab1-cell-c-fault-drift.md`, `315-0of70-has-no-eviction-cell.md`.

### The forced-eviction intervention, and the fix it motivated

Forcing KFD evict/restore via `~/wt/evictonly` raises the rate from 0/22 to 7/10 arms (Fisher exact
two-sided p = 3.6e-5), producing **both** faces under one intervention — the trigger-level mechanism.

The candidate repair that intervention motivated — routing `buffer_get`'s bare `memcpy` of the
non-coherent `0x52` logits mapping through `hrx_synchronous_d2h` — has been **REFUTED by its own
pre-registered A/B** (`fc7ffd5`, `docs/315-campaign/evidence/315-ab-preregistration.md`):

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
`docs/315-campaign/patches/315-coherent-read-fix.patch` for the record only.

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

### Still open

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
pre-registered at `docs/315-campaign/evidence/315-gttstep-preregistration.md`, not yet run.
