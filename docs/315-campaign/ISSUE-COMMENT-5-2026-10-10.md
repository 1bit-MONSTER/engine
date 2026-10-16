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
## The ordering family is closed by four null toggles in regime; no mechanism is named yet, and #315 stays OPEN

This is the consolidation of today's in-regime work on `strixhalo` (pin `e44c9d01a4d5`). It reports a **negative
result honestly**: every lever that could have made an ordering or visibility defect produce this fault has now
been toggled in regime, arm by arm, and **none of them changes the outcome**. That falsifies the whole
ordering/visibility super-family and leaves the fault in the *computed* class, where the instruments available
today cannot name the producer.

### 1. The toggles, with their arm counts

Every run below: same harness (`315-paired.sh` through `315-attrib.sh`), same model, same composition
(`REGIME_LO=80`), arms alternating inside one session where a control cell exists, and **every arm at
`gtt_start = 98 GiB`** — inside the frozen pre-registration's `>= 95 GiB` gate. "Inducer" means both cells carried
`GGML_HRX_TENSOR_FP=tail:16 NANONLY=1`, which is what makes a difference visible in a handful of arms (it raises
the rate; no rate here is an acceptance figure).

| lever toggled | control cell | fix cell | reading |
|---|---|---|---|
| `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` (a stream sync after every graph replay) | 1 fault / 4 arms | 0 / 2 | no change (Fisher p = 1.00) |
| `ggml_backend_synchronize` before the instrument's reads (patched build vs the previous build) | 2 / 5 (`315row`) | 1 / 5 (`315fp2`) | no change (p = 1.00) |
| forced AQL execution barrier on **every** packet (`HRX_FORCE_BARRIERS=1`, env-gated patch in `aql_block_processor.c`) | 2 / 4 | 1 / 2 | no change |
| `-fa off` (take the model off the flash-attention path entirely) | 1 / 2 | 2 / 2 | fault persists |
| `HOST_COHERENT` on host-visible buffers (earlier today, its own campaign) | — | 1 fault in 10 | not sufficient |
| `HRX_COMMAND_BUFFER_MODE=pm4` | 3 OK + 1 fault / 4 | 0 OK + 4 arms at `req=1` | **unusable**: every PM4 arm failed the *first* request with HTTP 500, so it is not a fault-rate comparison at all |

### 2. The family is closed at source as well, not only by the runs

* the host read path is already ordered: `llama_get_logits_ith` synchronises (`src/llama-context.cpp:3729`), the
  wait ends in **`iree_hsa_signal_wait_scacquire`** (`runtime/src/iree/hal/drivers/amdgpu/semaphore.c:329`), and
  `stream->timepoint` is advanced *after* each submission, so an `hrx_stream_wait` waits on the last one;
* the packet barrier bit is decided by `..._packet_has_barrier` (`aql_block_processor.c:103-116`) and the driver
  explicitly does **not** infer memory hazards (`:126`) - and forcing that bit on everywhere changed nothing;
* the intra-dispatch variant of the same class was audited statically over the whole `loom-libs/ops` corpus: the
  three kernels that hand work across waves through global memory are exactly the three carrying
  `barrier<global>`, and the quantized-matmul partials are a two-dispatch reduction whose ordering the host owns.

### 3. Why "name the producing value" is not obtainable with the instruments I have

Two independent findings, each proven from the arms themselves, each sufficient on its own:

1. **The fingerprint reads a host-visible, non-coherent arena with a plain `memcpy`.** The MoE intermediates live
   inside one `44048640`-byte `host_visible=1 memory_type=0x52` allocation (buffer trace), and
   `ggml_backend_tensor_get` short-circuits for host buffers - no iface call, no sync. Ordering those reads
   (patch applied, verified in the build) changed nothing about the fault (row 2/5 vs fp2 1/5), so the reads are
   ordered now and the numbers below still stand: **the locator was never the problem.**
2. **A post-graph read by name returns the transient arena's last writer.** In the faulting arm
   `ffn_out-0` reads `nan=7048 of 1048576` while `l_out-0` reads `nan=4963 of 1048576` - impossible, because
   `l_out = ffn_out + attn_out` requires `l_out`'s NaN set to contain `ffn_out`'s. The only explanation is that
   the slot an earlier tensor occupied has since been reused by a later node, so reading it "by name" returns
   somebody else's bytes.

Both mean the same thing for task-2's wording: a *post-graph, by-name* instrument cannot name the producing graph
value on this backend, and that is a property of the backend, not of the fault. What remains trustworthy is
therefore narrow, and it is what I would carry forward: the guard's own line, the non-transient tail
(`norm`, `result_norm`, `result_output`), and the arm verdicts.

### 4. What is established, at the resolution that survives

* **The fault is real, in regime, and common enough to study.** 178 arm directories on the box, **131 in-regime,
  34 faulting in-regime**, across a dozen independent sessions, at `gtt_start` 98-99 GiB. Both faces appear and
  never in the same arm: `nan=1` (Face A, the sampler guard) and `http=1` (Face B, the server's own 500).
* **The exact symptom, reproducible:**
  `2.53.659.050 E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no sentinel=0`.
* **Coarse localisation:** in the one in-regime arm where the fault was fingerprinted with ordered reads at
  `gtt_start=98G`, the corruption is already present at **layer 0** of a **T = 512 prefill** and cascades to a
  fully NaN residual stream by layer 1 (`l_out-1 nan=1048576 of 1048576`, all 512 rows). Whole-batch, not one row.
  Layer 0's attention output is not instrumented, so the onset is "layer 0's FFN or its input".
* **The two faces are one event with two observables**, as the failure-mode table above and the earlier Arm
  landscape show: what the corrupted memory holds decides which one you see.

### 5. The lead that matches the symptom best, stated as a lead

`docs/hrx.md` in this repo records the historical instance of exactly this class (engine#123/#140): a producer
wave writing **global** memory, a consumer wave reading it back, with the barrier fencing **LDS only** - no store
wait, no `buffer_gl0_inv` - so the consumer could quantize the slot's previous contents. Its own words are the
reason I am not dropping this line: *"the f32 output was right and `next_q8` was wrong, and NaN when the stale
bytes decoded as NaN"* - **Face B and Face A in one sentence** - and *"the window opens on dispatch preemption,
which is why the rate followed memory-pressure settings"*, which is the only mechanism I have seen that explains
the GTT dependence at all.

That fix is in the pin, and today's audit + forced-barrier run close its intra-dispatch and host-recorded halves
for this model. What is *not* closed is the same window opening somewhere else - most plausibly under memory
pressure itself (eviction/migration of a buffer a kernel is reading), which no userspace lever in the inventory
reaches.

### 6. Remedy: none, and by this issue's own rule a "documented limitation" cannot substitute yet

A limitation entry is acceptable **once the mechanism is proven**; it is not proven. So `#315` stays **OPEN**, and
what it now carries is a much smaller search space plus two falsified instruments. The next instrument that can
actually name the producer is one that reads a value **at the point of production** (inside the dispatch, or from
a non-transient copy made at the producer) - and, for the pressure link, a trace that correlates a faulting arm
with the driver's own eviction/preemption events (two profiled in-regime sessions are running as I write this;
the first produced 8 clean arms, so the faulting half of that trace is still owed).

### 7. Commands

```sh
ssh strixhalo 'grep -E "^  pair" ~/wt/{315row,315gsync2,315fbar,315nofa,315pm4}-console.log'   # the toggles above
ssh strixhalo 'grep -m1 "HRX returned NaN logits" ~/wt/rt-det/315fp2-p2l-*/server.log'        # the guard line
ssh strixhalo 'grep -m1 "size=44048640" ~/wt/rt-det/315accept-p1c-*/server.log'               # the host-visible arena
bash docs/315-campaign/tools/verify-evidence.sh                                               # 30 checks, all PASS
bash docs/315-campaign/315-forcebarrier.sh                                                    # the barrier toggle (re-runnable)
```

---

## Addendum, 2026-10-12: task-4 was completed against a change this issue had already falsified

Recorded because the goal's task list shows **task-4 "Implement the forward-only fix" as complete**, and
under the goal's own rules a completed task is immutable, so this note is the only way to correct it.

**What was done.** A forward-only branch `1bit/315-coherence-fwd` (`6cb7cfbfa`) was pushed to the fork: the
pin `0e255d774` plus `ggml-hrx: make host-visible buffers coherent unconditionally`
(memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT for host-visible buffers). It satisfies the *plumbing* half of
the task-4 contract - `git merge-base --is-ancestor 0e255d774 6cb7cfbfa` is true, the diff is 1 file / 7
insertions / 3 deletions, and it touches no arena or reuse knob.

**Why it is not the fix, and must not be read as progress.** It is the same change as `fb4dff7bb`
(`engine-315/host-visible-coherence`), which this document records as falsified in its own pre-registered
campaign - section 13 of `docs/315-mechanism.md`, and the first row of the "Falsified in this round" table
above: *"memory coherence is the carrier | `HOST_COHERENT` patch live (`hv52=0 hv56=1`), still 9 clean + 1
fault in 10"*. The mechanism note warns about exactly this trap: *"The fork has a forward-only engine#315
change, and it is not the answer."*

**The process error, stated plainly, so it is not repeated.** The change was accepted on the strength of its
*shape* - forward-only, minimal, mergeable - without checking it against the falsified-lever record. The
contract's own words are *"Re-read the issue's falsified-levers list and confirm the claimed mechanism is not
one of them; if it is, the fix is rejected."* The falsified-lever check that was run searched for arena and
transient-reuse knobs and did not test coherence itself, which is the one lever that mattered.

**Consequence for the remaining tasks.** task-5 (the ≥60-arm campaign) has no fix to verify, task-6 has no
revision to regress, and task-7's "retire the entry" is forbidden while no mechanism is proven. Nothing in
sections 1-7 above is changed by this addendum: the ordering/visibility family remains closed, the fault
remains in the *computed* class, and the next instrument remains the one named in section 6.

**Recommendation.** Delete `origin/1bit/315-coherence-fwd`. It is mergeable and therefore a standing trap for
the next agent; leaving it invites this same error.
