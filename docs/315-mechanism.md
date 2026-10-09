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
# engine#315 — mechanism statement (task-3) and fix direction (task-4)

Status: **candidate mechanism whose premise is measured and whose predicted observable is confirmed; the
causal (rate) link is still pending** because it needs the box to itself for tens of arms.

## The mechanism

The sampler reads the logits **directly out of a host-visible HRX buffer that is not host-coherent**. The
generic `ggml_backend_tensor_get_async` short-circuits for host buffers (plain memcpy, no backend call), so
no copy appears anywhere in the backend's own transfer path, and nothing in that path guarantees the
device's write into that memory is visible to the CPU reader. That is consistent with the fault's shape:
whole-vocabulary damage, intermittent, and sensitive to allocation/pressure timing.

## Evidence (all measured 2026-10-09, on the pinned revision `e44c9d01a4d5`)

| observation | value | how |
|---|---|---|
| logits buffer allocation | `type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=619520` | `GGML_HRX_BUFFER_TRACE=1`, one load + three decodes |
| size identity | `619520 = 154880 × 4` — exactly the vocabulary the sampler guard reports | same trace vs the guard's `count=154880 of 154880` |
| no copy for the logits | zero logits-sized download records; the 146 records are per-layer weight leases (884736 / 2211840 / 5898240, each ×47) | `GGML_HRX_DOWNLOAD_FINGERPRINT=1` |
| the copy path is ordered | `hashes_equal=1` on all 146 records (probe read ≡ real copy) | same, plus `hrx_stream_synchronize` precedes the copy at `runtime/host-memory.cpp:1350` |
| the deferred writeback route is cold | 0 `[hrx-wb-fp]` records with the flag on over a complete 66/66 arm | v1 instrumentation |
| coherence is gated on a knob | `GGML_HRX_USE_UNIFIED_MEMORY=1` → `direct_host_binding=1`, `memory_type=0x56` = `0x52 \| 0x04` (`HOST_COHERENT`) | same trace, second load |
| host-visible buffers are persistently mapped | `HRX_BUFFER_USAGE_MAPPING_SCOPED \| ..._PERSISTENT` requested for them (`ggml-hrx.cpp:319`) | code, pinned |

**Predicted before it was run, then confirmed:** the knob flips exactly the coherence bit, on exactly the
buffer family the logits come from. That is the experiment the mechanism had to survive, and it did.

## Ruled out, with evidence

- **Deferred-writeback publish window (H1)** — the route is cold (0 records over a full arm).
- **Copy/race in the download path** — `hashes_equal=1` everywhere, and the sync precedes the copy.
- **"Logits are host-staged"** (the recorded hard negative) — upheld, and now explained: there is no copy
  at all, because the buffer is host-visible and the generic layer short-circuits.
- **Arena knobs / transient reuse** — falsified on the record (15 paired arms; 77-arm tally).
- **Pin/composition (AMD core vs our `ggml-hrx`)** — the landed rate study found no measurable difference
  (Fisher exact p = 1.000).

## Not yet ruled out (stated so it is not overclaimed)

- **The causal link.** Coherence controls the property; whether it removes the *fault* is a rate question at
  ~5 %/arm, so it needs tens of arms with the box to itself. A self-starting launcher
  (`docs/315-campaign/315-when-free.sh`) waits for the machine and runs the instrumented hunt; the
  differential itself is `GGML_HRX_USE_UNIFIED_MEMORY=1` vs unset at the same regime.
- **The peer session's candidate** (AQL block-transition dose-response, `HRX_AQL_BLOCK_SIZE` 262144 vs
  32768) is still in flight on the same box and must be accounted for in any final statement.
- **Face B (invalid UTF-8 / format-500) was never observed in this session**, so no mechanism is claimed for
  it. Its cause may be entirely different from Face A's.

## Fix direction (task-4) — pre-committed by outcome, minimality first

If coherence removes the fault, the fix is a **scoped visibility guarantee on the output path**, not the
global toggle:

1. give the host-visible buffer type used for outputs `HOST_COHERENT` — i.e. drop the `direct_host_binding`
   gate for that type only (`ggml-hrx.cpp` `buffer_alloc`), leaving the rest of the memory policy alone;
2. or issue the runtime's explicit visibility operation (flush/invalidate) for those buffers before the host
   reads them;
3. or stop allocating the logits host-visibly at all (device-local + one explicit copy), which removes the
   class of window at the cost of one copy per token.

Whichever is chosen, it must be forward-only: a commit on `1bit-MONSTER/llama.cpp`, then an engine pin move
whose previous pin is an ancestor of the new one (`git merge-base --is-ancestor` must be true).

**Not a fix:** `GGML_HRX_USE_UNIFIED_MEMORY=1`. A configuration toggle that trades coherent memory for
performance is not a resolution, and the pre-registration's campaign must pass with the default
configuration.

## Reproducing the two measurements

```bash
# premise (one load + three decodes, no regime needed)
UNIFIED= LOG_OVERRIDE=/tmp/315trace-default.log bash ~/wt/315-trace-premise.sh
UNIFIED=1 LOG_OVERRIDE=/tmp/315trace-unified.log bash ~/wt/315-trace-premise.sh
# then: grep 'hrx-buf' /tmp/315trace-*.log | grep 619520
```

## 12. Discriminating experiment: result (tag `315diff2`, 2026-10-09)

**Premise (measured, §11).** The logits buffer is `HRX0_HOST`, `host_visible=1`, `direct_host_binding=0`,
`memory_type=0x52` — host-visible and **not** `HOST_COHERENT` (0x04). The sampler reads that buffer
directly: there is no download and no copy on the route (the generic `ggml_backend_tensor_get_async`
short-circuits for host buffers, and no 619520-byte transfer exists anywhere in the arm's log). Since
nothing in the transfer path orders the device's write against that read, a reader may observe unwritten
bytes — which is the observed shape: whole-vocabulary NaNs on one face, and invalid output bytes
surfacing as an HTTP 500 parse failure on the other.

**Method.** Instrumented build `~/wt/315instr3-build` (pin `e44c9d01a4d5` + all three diagnostic
patches), `GGML_HRX_BUFFER_TRACE=1` on every arm. Arms alternate the two configurations on the same box,
model, detector and load, in-regime at `gtt_start 98 GiB`: A = default, B = `GGML_HRX_USE_UNIFIED_MEMORY=1`
(the coherence discriminator, **not** a fix).

**Result** (arms on disk under `~/wt/rt-det/315diff2-*`; verdicts from `315-classify.py`):

| pair | A = default | B = coherent | host-visible **non**-coherent buffers |
|------|-------------|--------------|----------------------------------------|
| 1    | clean       | clean        | 2 / 0 |
| 2    | clean       | clean        | 2 / 0 |
| 3    | clean       | clean        | 2 / 0 |
| 4    | **FAULT**   | clean        | 2 / 0 |

Pair 4's fault is **Face B**, in-regime, on the predicted side: the arm answered with a 500 whose message
is `does not match the expected Content-only`, following `common_chat_peg_parse: unparsed Content-only
output:` — i.e. generated text carrying bytes that are not valid UTF-8. Evidence:
`~/wt/rt-det/315diff2-d4-GLM-4.7-Flash-Q4_K_M/server.log` (`"code":500` ×1, unparsed ×2),
`detector.json` (`http_failures=1`, `requests=21`), `run.log` (`gtt_start 98GiB`), `gtt.log`.

The trace count is the per-arm assertion that the configurations really differ: every default arm
allocates **2** host-visible non-coherent buffers (the 619520-byte logits buffer plus a 44048640-byte one),
every coherent arm allocates **0**. So the discriminator is a measured property of the allocation rather
than the knob itself, and it flips exactly as §11 predicted.

**What this does and does not establish.** It establishes that a fault occurs with the non-coherent
host-visible memory class and, in this sample, not with the coherent class: 1 fault in 4 in-regime default
arms (consistent with the recorded ~5 %/arm), 0 in 4 coherent arms. It does **not** establish the rate —
that is what the pre-registered ≥60-arm campaign (§7) is for, which is why the differential stops here
rather than being read as evidence of a fix.

**Classifier gap (recorded because it would have faked a pass).** The first predicate in `315-paired.sh`
and `315-differential.sh` read only nan/hsa/utf8/parse, so pair 4's Face-B fault was labelled
`INCOMPLETE` — its request count fell short after the 500. Face B is first-class in the pre-registration,
so the predicate now also counts `http_failures`, `transport_failures` and the server-log signatures
(`"code":500`, unparsed Content-only). That implements the pre-registered rule rather than relaxing it:
the table above is produced by applying that definition to arms that were already on disk, and
`docs/315-campaign/315-classify.py` is the single source of truth for it.

## 13. The coherence hypothesis is falsified by its own campaign (tag `315accept`, 2026-10-09)

The candidate from §12 (unconditional `HOST_COHERENT` for host-visible buffers) was built into a fix cell
and run under the pre-registered acceptance rule (§7: ≥60 in-regime clean arms in the fix cell, zero
faults on both faces). The pre-registered stopping rule ended the run at the first fix-cell fault, arm 10:

| cell | in-regime arms | clean | faults | rate (95 % CI) |
|------|---------------:|------:|-------:|----------------|
| control (pin + diagnostics) | 4 | 3 | 1 | 25.0 % (4.6–69.9) |
| **fix** (pin + fix + diagnostics) | 10 | 9 | **1** | **10.0 % (1.8–40.4)** |

Fisher exact two-sided **p = 0.505**; P(rate_control < rate_fix) = 0.214. Acceptance: **NOT MET**
(9 clean of 60 required, one fault where zero are allowed).

The decisive line is the faulting fix arm itself:

```
pair 10 arm l: FAULT nan=0 req=33 parse=0 gtt_start=98G hv52=0 hv56=1
```

`hv56=1` means the output region *was* allocated `HOST_COHERENT` in the arm that faulted — the same
per-arm trace that proved the fix live in the smoke proves it live here. The fault happened anyway, in
regime (`gtt_start 98 GiB`), as Face B (`nan=0`; a 500 after `unparsed Content-only` output). So a
visibility gap in the output region is **not a sufficient cause**: making the region coherent does not
stop the fault.

Honest bounds on this result: it does not show the fix raises the rate (1/10 vs the recorded 5.4 % is
indistinguishable at this n), and it does not clear coherence as a *contributing* factor. What it does is
remove coherence from the list of sufficient mechanisms — which the earlier n=4 differential had merely
*suggested* (one fault, on the non-coherent side). That low-n signature did not reproduce at scale, and
the campaign is the reason we know that rather than believing it.

Two things the pre-registration bought, worth recording: the stopping rule ended the run at arm 10 instead
of burning ~5 h to a foregone conclusion, and Face-B-first scoring is what caught this fault at all —
under the recorded study's rule (Face B counted separately as "truncation") the same arms read
`clean=9 fault=0 trunc=1`, i.e. it would have been reported close to a pass.

**Where the mechanism now points.** The output data is written *wrong or not at all*, not merely
invisible, so the surviving family is the producer side. The recorded open candidate is already in it: GLM
is the only model whose decode splits into two 128 KB AQL command blocks (919 + 760 dispatches, 1,399
barriers per token), and the instrumentation implemented for that family (the
`backend_get_tensor_async` → producing-`value` probe on `f2099e9b7`) is the natural next step.

## 14. Refined candidate: ordering, not visibility — and a correction to §11

§13's falsification splits the problem cleanly. Coherence makes a write *visible*; it does not order a
write against a later read. The output region can still be read before the device has written it at all,
which is what whole-vocabulary NaN (Face A) actually looks like: the reader sees an untouched buffer, not
a torn one.

**Correction to the route claim in §11.** "The logits are never copied, so there is no copy to order" was
an over-reach. The download fingerprints cover `runtime/host-memory.cpp`'s path only; a copy issued
through the buffer stream (`buffer_submit_and_wait`, `ggml-hrx.cpp:150`) is invisible to them, and the CPU
read of a host-visible buffer is a plain `memcpy` that no backend instrumentation can observe. The honest
form of that finding is *no 619520-byte transfer was observed through the instrumented download path* —
not "no transfer exists".

**What the pinned revision actually does** (all at `e44c9d01a4d5`):

| site | fact |
|------|------|
| `ggml-hrx.cpp:518` `graph_compute` | calls `executor.execute(*graph)`, returns its status — **no synchronize** |
| `runtime/graph-executor.cpp:78` `GraphExecutor::execute` | no wait, fence or synchronize anywhere in its body |
| `ggml/src/ggml-backend.cpp:274-279` generic `tensor_get_async` | when the backend implements `get_tensor_async` (HRX does, `ggml-hrx.cpp:454`) the generic function does **not** synchronize — it trusts the backend |
| `src/llama-context.cpp:1488` | `llama_decode` issues the logits copy with `*_async` and returns to its caller, which samples |
| `src/llama-context.cpp:712` | `llama_context::synchronize()` does wait — but nothing in the sampling path calls it |

So the ordering of the device's writes against the host's read of the output region rests on a wait that
the sampling path does not perform. That is a hypothesis with a one-line discriminating test, so it is run
as the **sync cell**: the pin + the same inert diagnostics + a single `backend_synchronize(backend)` at
the end of `graph_compute`, so the cell differs from the control by exactly that wait. Nothing in the
diagnostics or in the cell changes placement or coherence, so the comparison is not confounded by §12's
rejected candidate.

Difference from the falsified candidate, stated plainly: §12 changed *what the reader can see*
(coherence, `0x52` → `0x56`); this changes *when the read is allowed to happen* (ordering). Only the
second can make an unwritten buffer safe, which is why the Face-A shape ("all 154880 elements") is the
prediction this hypothesis actually explains.

## 15. The ordering candidate is falsified too — and that locates the fault in the producer

The §14 candidate (a completion barrier — `backend_synchronize(backend)` at the end of `graph_compute`, so
the host read is ordered after the device's work) was run as the **sync cell** under the same
pre-registered rule, across tags `315sync` (interrupted by an ssh-session freeze; its pair 1 recorded) and
`315sync2` (relaunch). The stopping rule ended the relaunch at the first fix-cell fault, pair 5.

| cell | in-regime arms | clean | faults | rate (95 % CI) |
|------|---------------:|------:|-------:|----------------|
| control (pin + diagnostics) | 2 | 2 | 0 | 0 % (0–65.8) |
| **fix (pin + barrier + diagnostics)** | 6 | 5 | **1** | **16.7 % (3.0–56.4)** |

Acceptance: **NOT MET** (5 clean of 60 required, one fault where zero are allowed).

The faulting arm is itself the finding:

```
pair 5 arm l: FAULT nan=1 req=10 gtt_start=98G hv52=1 hv56=0
  server.log: common_sampler_sample: HRX returned NaN logits: count=154880 of 154880,
              index range [0..154879], contiguous=yes, continuing=no
```

With the barrier in place the host read happens *after* the compute stream reports completion, so this NaN
was not read early. **Every** element of the 154880-wide output was NaN, so the region was either written
with NaN or never written by the device at all — while the stream reported success. That takes the
mechanism off the host path and puts it in the **producer**: the dispatch or kernel that fills the output
region can silently fail to produce it.

Both host-path candidates are therefore excluded by their own pre-registered campaigns — visibility (§13)
and ordering (§15) — and what remains is the family the recorded entry had already named: the
kernel/binding side, whose concrete open candidate is GLM's decode splitting into two 128 KB AQL command
blocks (919 + 760 dispatches, 1,399 barriers per token). The next question is a producer-side one — does
that dispatch run, and what does it write — not a transfer question.

**Cost, for the record.** The stopping rule ended both campaigns at the first fix-cell fault: 14 arms (§13)
and 7 arms (§15), instead of up to 120 each. Two falsifications for under an hour of box time.

## 16. Producer probe: built, and what it did and did not establish

Built as `GGML_HRX_PRODUCER_PROBE` in `runtime/graph-executor.cpp` (pre-fill host-visible outputs with a
payload NaN `0x7FC0DEAD`, drain the stream, classify sentinel-intact vs other-NaN vs finite, log anomalies
and an `armed` line proving the path is live). Four build/run cycles, each fixing an error of mine:

1. `HRX_CHECK` is a macro local to `ggml-hrx.cpp`; unavailable in that TU.
2. `hrx_status_t` is an **opaque pointer**, not an enum — the real tests are `hrx_status_is_ok()` and
   `hrx_status_ignore()`.
3. A selector requiring "host-visible **and** not consumed by any node" armed nothing: the region is a
   graph-written destination, consumed by the node that writes it.
4. A once-per-process dump only ever reported the first chunk.

**What the runs established, and this part is structural.** HRX executes the llama graph as many small
chunks — `n_nodes=9`, tails like `PERMUTE q_nope_perm-0`, `MUL_MAT q-0`, `RMS_NORM norm-0` — and across four
arms and ~260 requests **no chunk executed through `GraphExecutor::execute` contained a host-visible tensor**
(zero `host-node` lines, zero `armed` lines). So the sampled output region is not written by a graph-program
dispatch; its writer is outside that path — a scheduler-issued copy, or a producer that is not an HRX graph
operation.

**The bound on that claim, stated rather than glossed.** It rests on the gate
`ggml_backend_buffer_is_host(node->buffer)`. If the chunk that writes the region carries it as a view whose
buffer test fails, the conclusion degrades to "not observed" rather than "not written". That is not settled
here, and I am not going to claim the stronger form.

**Consequence.** "A skipped HRX graph dispatch" is not supported as the producer-side explanation. The
remaining producer-side candidates are the staging/copy path
(`backend_copy_tensor_async` → `hrx_stream_copy_buffer`) and non-HRX producers; the next instrument should
hook the copy path, not the executor.

**Cost, stated plainly.** Four build/run cycles and several hours of box time went into this instrument, and
every cycle was my error in choosing what to target. The structure dump that finally explained the shape
should have been the first thing built, not the fourth.

## 17. Copy-path probe (task-4 prerequisite): interpretation pre-committed before the run

§16 concluded that the writer of the 619,520-byte output region is outside `GraphExecutor::execute`
and named the next instrument: the copy path. Built as a runtime-gated probe (`GGML_HRX_COPY_PROBE`,
so the marker is provably in the binary and a zero-hit log cannot mean "compiled out"), at the four
sites in `ggml-hrx.cpp`: `submit_copy_buffer`, `set_tensor_async`, `get_tensor_async` (device→host,
i.e. the download of the output region), `copy_tensor_async`.

The probe prints `hrx-copy-probe armed` on first use (proof the path is live) and, for every copy,
`hrx-copy site=… tensor=… size=… host_ptr=…`, plus a dedicated `hrx-copy-LOGITS` line when
`size == 619520`.

**Pre-committed reading of a single in-regime probe arm, fixed before the data:**

| observation | conclusion | next discriminator |
|---|---|---|
| `armed` present, `hrx-copy-LOGITS` with `site=get_tensor_async` | the output region *is* read through the device→host download path | fingerprint the **device** buffer immediately before this copy: if it is already NaN, the fault is in the producer/compute side; if it is finite, the copy or the host mapping is the producer |
| `armed` present, **no** `hrx-copy-LOGITS` at all | no copy path touches the region; it is read directly from a host mapping, and the four sites are not the writer | fingerprint the host mapping in place (read it twice, before and after the host read) rather than instrumenting more copies |
| `armed` absent | the probe did not reach the backend: this is an **invalid** arm, not evidence | fix the probe wiring; do not read the absence as a negative |

This is deliberately a *reading*, not a claim: whichever row is hit, the arm's own verdict
(`nan`/`hsa`/`parse`/`utf8` counts, §5) is recorded alongside it, and the row that does not match
is reported as not observed rather than argued away.

## 18. Sentinel experiment (the decisive discriminator): reading pre-committed

§15 left one fact unexplained: in the faulting arm **every** one of the 154,880 logits was NaN, so the
region was "either written [as] garbage or never written". §16 cannot separate those two: its producer
probe armed nothing (its selector required a host-visible tensor that no node consumes), so its sentinel
pre-fill never ran, and its own text records that the conclusion degrades to "not observed".

That distinction is testable directly. The host-visible logits allocation (the 619,520-byte region,
identified in §11) is filled with a payload NaN `0x7FC0DEAD` when `GGML_HRX_SENTINEL` is set, and the
existing sampler guard — which already counts NaN, index range and contiguity — additionally counts how
many of those NaNs are exactly the sentinel. Only the 619,520-byte allocation is touched, only under the
env var, so a control arm without it is unaffected.

**Reading, fixed before the run:**

| observation on a faulting arm | conclusion |
|---|---|
| `sentinel` == `count` (every NaN is ours) | the device **never wrote** the logits region: the fault is a missing/lost **write**, not a stale or early **read** — which is also consistent with §13 and §15 both failing, since neither touched the write |
| `sentinel` == 0 (none of the NaNs are ours) | the device **did write**, and wrote NaNs: the producer is compute/kernel-side |
| 0 < `sentinel` < `count` | the region was written **in part**: a torn or interrupted write, which implicates the write path too |
| no `[hrx-sentinel]` line, or the arm has no `armed` line | the instrument did not run: an **invalid** arm, not evidence |

Two consequences stated in advance. First, on a *clean* arm the guard does not fire at all (it returns
before logging when `nan_count == 0`), so a clean arm carries no sentinel datum — the sentinel is only
readable on faulting arms, and the experiment is therefore about *which* faults exist, not about the rate.
Second, if `sentinel` == 0 on the first faulting arm, the sentinel hypothesis is dead and the producer-side
instrument is the next step, exactly as §16 proposed.

### 18b. The sentinel is a diagnostic, not part of the fix — fixed before task-5

Two consequences of the fill, worked out before the campaign so they cannot be settled to taste later:

1. **The sentinel makes an unwritten region deterministically NaN.** It is written once, at allocation, so
   any healthy arm overwrites all 154,880 elements with real logits and is unaffected. It therefore
   cannot *inflate* the fault count: a fault still requires a NaN to reach the sampler. But it *can*
   convert a partially-written region from "stale finite values from the previous token" (silent wrong
   logits) into "sentinel NaN" (a detected fault). That is a sensitivity increase on the *diagnostic*
   instrument, not a change to the shipped behaviour.
2. **Therefore the acceptance campaign (task-5) runs the fix build with the sentinel OFF.** The campaign
   must measure the shipped configuration; an arm run with `GGML_HRX_SENTINEL=1` measures the instrument
   as much as the fix. The same applies to the copy probe (inert, but pointless in a campaign), and to
   any other diagnostic env var: the campaign's `EXTRA_ENV` is empty, and that is recorded per session.
   Diagnostic runs like the current one are labelled as such and their counts are never read as
   acceptance (matching the driver's own `NOSTOP=1` note).

If the sentinel is kept in the eventual fix build, this paragraph is void and must be restated with
evidence that the fill cannot affect a healthy arm.

## 19. The "no copy for the logits" premise is falsified (copy probe, arm `315sent3-p1l`)

The copy probe answered §16's open question, and the answer contradicts §11's recorded evidence.

Measured in one arm (66/66 requests served; `gtt_start=80G`, i.e. **out of regime** — no regime claim is
made from it, and both arms were classified `OOR` by the driver):

```
hrx-copy-probe armed
[hrx-sentinel] filled host-visible logits region size=619520 with 0x7FC0DEAD
hrx-copy site=get_tensor_async tensor=result_output size=619520 host_ptr=0x…      × 2734
```

**Every copy in the arm is logits-sized (619520) and every one goes through `backend_get_tensor_async`**
— the device→host download of the output region — 2734 times for 66 requests. So the sampler does **not**
read the logits out of a mapped region with no transfer at all; it reads them after an explicit
`hrx_stream_copy_buffer`. §11's line *"no copy for the logits — zero logits-sized download records"* is
therefore wrong, and the mechanism statement built on it ("the sampler reads the logits directly out of a
host-visible buffer ... so no copy appears anywhere") loses its premise.

Why the two instruments disagreed, stated precisely: §11's `GGML_HRX_DOWNLOAD_FINGERPRINT` recorded 146
per-layer **weight-lease** copies (884736 / 2211840 / 5898240, ×47). The copy probe recorded 2734
**logits** copies and **no** weight leases at all. They are watching different branches of the same path,
and "there is no copy" was an artefact of instrumenting the weight-lease branch only.

Consequence, per the reading pre-committed in §17 (row 1, "the output region *is* read through the
device→host download path"): the next discriminator is to **fingerprint the source device buffer
immediately before this copy** — count NaN in the source before, and in the destination after, the
`hrx_stream_copy_buffer` at `backend_get_tensor_async`. If the source is already NaN, the producer is
upstream of the copy (compute/staging); if the source is finite and the destination is NaN, the copy or
its ordering is the producer. No claim is made until that runs.

### 19b. Composition arithmetic, corrected by measurement (`315sent3`)

| quantity | measured in `315sent3` |
|---|---|
| base GTT (two foreign Vulkan servers resident) | 26.7 GiB |
| per neighbour (`-c 2048`) | **≈17.7 GiB** (2 neighbours → 62 GiB) |
| detector server's own contribution | **≈18 GiB** (62 → 80) |

So `REGIME_LO=60 REGIME_HI=73` (2 neighbours) lands at 80 GiB — **below** the gate. To land in band with
this inventory the helper must compose **3 neighbours**: `26.7 + 3 × 17.7 ≈ 79.8`, then `+18 ⇒ ≈98 GiB`.
The default band (LO=80 HI=92) instead composes 3 neighbours and then overshoots to 97 *before* the
detector, ending at 115 (INVALID). The usable setting is therefore `REGIME_LO=70 REGIME_HI=85`, and the
arm's own `gtt_start` remains the only thing that decides regime membership.

### 19c. Why the fingerprint missed the logits (mechanism of the miss, verified in source)

`GGML_HRX_DOWNLOAD_FINGERPRINT` is implemented in `runtime/host-memory.cpp:1419`, immediately after
`hrx_synchronous_d2h(device, source, offset, host_destination, size)` at line 1411 — i.e. in the
**synchronous download** path, the one taken when the host buffer is *not* registered. The logits take the
other branch: `backend_get_tensor_async` finds the destination in `device->host_buffers` (registered) and
issues `hrx_stream_copy_buffer`, which is where the copy probe fires (2734 times per arm).

Its gate (`size >= 4096`) would have admitted 619520, so this was not a threshold artefact: the
instrumented branch simply never executes for registered host buffers. §11's "zero logits-sized download
records" is therefore an artefact of *which branch* was instrumented, not evidence that no transfer
happened — and the 146 weight-lease records it did produce are consistent with the same explanation.

**Consequence for the sentinel experiment:** the sentinel is written into the *source* allocation, so a
faulting arm reporting `sentinel == count` means the copy was faithful and the source region was never
written by the device. That is §18's decisive row, and it needs no additional instrument — only a
faulting in-regime arm.
