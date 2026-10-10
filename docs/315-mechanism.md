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

## 20. Face B captured in-regime (arm `315hunt-p1l`, `gtt_start=98 GiB`)

The sentinel hunt's **first probe arm faulted**, in regime, and it was Face B rather than Face A:

```
pair 1 arm c: OK    nan=0 req=66 gtt_start=98G
pair 1 arm l: FAULT nan=0 req=15 gtt_start=98G utf8=0 parse=0
detector.json: requests=15 ok=14 http_failures=1 first_failure="HTTP 500" truncated=true
server.log   : common_chat_peg_parse: unparsed Content-only output: .,。_./ <U+FFFD>, exactly,…
               srv: got exception {"code":500,"message":"…does not match the expected Content-only format"}
probe        : armed=True sentinel_filled=True copies=931, ALL get_tensor_async/result_output/619520
               NaN lines: 0
```

Three things follow, and they separate the faces:

1. **Face B is not an unwritten logits region.** `nan=0`, the guard never fired, and every one of the
   931 copies is a clean logits download. The logits were written and finite; the *text* was garbage, so
   the corruption is **upstream of the logits** (intermediates, KV, weights or a binding) — a different
   producer from Face A's, and one that no instrument built so far observes.
2. **Face B is common in-regime.** It appeared on the first in-regime probe arm (1 of 1), where the
   recorded Face-A NaN rate is ~5 %/arm. The campaign's bar (≥60 clean arms, zero faults on both faces)
   therefore stands or falls on Face B, and the supplied "identical: 1 distinct of 8" signature matches
   the legacy example in `KNOWN-ISSUES.md` (`v315k-l1`), so this is the known second face, reproduced.
3. **It looks stable, not racy, within an arm.** `identical_distinct = 1`: eight identical greedy
   requests produced the same garbage. A per-request race would be expected to vary.

**Reading pre-committed for the next experiment (repeat the same in-regime probe arm 3×, compare the
garbage text for an identical prompt):**

| observation | conclusion |
|---|---|
| the same garbage text across arms | a **stable wrong value or binding** (deterministic corruption) ⇒ bisect the inputs to prompt-eval, not the timing |
| different garbage text each arm | a **transient/race** corruption ⇒ the next instrument must be per-arm and timing-side |
| the arm comes back clean | Face B is load-sensitive rather than deterministic; the repeat rate must be measured before any determinism claim |

No mechanism is claimed for either face yet: Face A's sentinel row has not been observed (no NaN arm in
regime), and Face B's producer is unidentified.

### 20b. Face B did not reproduce in three consecutive in-regime arms

Repeating the same in-regime probe arm 3× (`315det`, `REGIME_LO=70 HI=85`, `gtt_start=98G`, instrument
live) gave **4/4 clean arms** — 66/66 requests each, no fault on either face:

```
pair 1 arm c: OK  nan=0 req=66 gtt_start=98G
pair 1 arm l: OK  nan=0 req=66 gtt_start=98G
pair 2 arm l: OK  nan=0 req=66 gtt_start=98G
pair 3 arm l: OK  nan=0 req=66 gtt_start=98G
```

Reading, per the row pre-committed in §20: *"the arm comes back clean ⇒ Face B is load-sensitive rather
than deterministic; the repeat rate must be measured before any determinism claim."* So the
"stable wrong value or binding" reading is **not** supported, and no determinism claim is made.

In-regime probe-arm tally so far: **1 fault in 4** (`315hunt-p1l` Face B; `315det-p1l/p2l/p3l` clean), all
of them with the instrument live and the composition at `gtt_start=98G`. That is the first rate datum for
Face B, from a diagnostic run — it is **not** an acceptance result and not comparable to the recorded
5.4 % Face-A tally, because it counts a different face.

**Operational consequence recorded for the next run:** `315-paired.sh` stops the whole session at the
first fix-cell fault, so with Face B at this rate a Face-A hunt keeps being terminated by a Face B arm
before reaching a NaN arm. The Face-A hunt must therefore run with `NOSTOP=1` — the driver's own
documented mode for diagnostic runs — and count faces separately afterwards, never as acceptance.

## 21. DECISIVE: the logits ARE written, with NaN — "never written" is falsified

The sentinel experiment ran on a real in-regime Face-A arm (`315fa-p2l`, `gtt_start=98G`, `nan=1`,
`req=34/66`). The result rejects the recorded claim and names the producer side:

```
HRX returned NaN logits: count=154880 of 154880, index range [0..154879],
                         contiguous=yes, continuing=no sentinel=0        <- NOT ONE NaN is ours
[hrx-sentinel] filled host-visible logits region size=619520 …          <- fill ran (1 line)
hrx-copy site=get_tensor_async tensor=result_output size=619520         <- 1667 copies, all logits
```

`sentinel = 0` with `count = 154880` means the region the sampler read was **overwritten by the device
with NaN**, faithfully delivered by a copy path that is known-good (hashes_equal=1 on every record, and
the probe shows every copy is logits-sized and registered-buffer). Consequences, in order of force:

1. **`KNOWN-ISSUES.md`'s "the logits buffer is never written" is false.** That sentence was inferred from
   all-NaN output; the sentinel is the experiment that separates "unwritten" (sentinel survives) from
   "written as NaN" (sentinel overwritten). It came back **written as NaN**.
2. **Face A's producer is compute-side.** Whatever computes the LM-head input produces NaN under pressure,
   and the delivery path is fine.
3. **This is why both landed fixes failed.** §13 (unconditional `HOST_COHERENT`) and §15 (a completion
   barrier before the host read) are remedies for a *missing* or *ill-timed write*. There is no missing
   write: the write happened, with NaN in it. Two independent falsifications now have one explanation.
4. **It also makes the two faces one family.** Face B is *wrong but finite* output (§20); Face A is the
   same "wrong data under pressure" at NaN magnitude. Neither is a delivery defect.

**Not yet named:** *which* compute stage manufactures the NaN. §23 takes this to the tensor level: the
LM head is exonerated (its input is already NaN) and the fault is pinned to the first decoded token of a
newly launched task, but the stage that manufactures it is still open.

## 22. State of play and remaining work (handoff, 2026-10-09)

### Established, with evidence paths

| # | finding | evidence |
|---|---|---|
| 1 | **The logits are copied device→host once per token** (2734 copies in one arm, all 619520 bytes, all `get_tensor_async`/`result_output`). §11's "no copy appears anywhere" is false; its zero-record fingerprint sat in the `hrx_synchronous_d2h` branch, not the registered-host-buffer branch. | `315sent3-p1l`, `315hunt-p1l`, §19/§19c |
| 2 | **Face A: the device writes NaN.** All 154,880 logits NaN with `sentinel=0` — not one is our payload. `KNOWN-ISSUES.md`'s "the logits buffer is never written" is falsified; the producer is compute-side, which is why the coherence (§13) and ordering (§15) fixes failed: both address a missing/late write, and the write is present. | `315fa-p2l` (in-regime, `gtt_start=98G`), §21 |
| 3 | **Face B is not an unwritten region either.** `nan=0`, 931 faithful logits copies, then `unparsed Content-only output` → HTTP 500. Wrong-but-finite data, upstream of the logits. Intermittent: 1 fault in 4 in-regime probe arms, and 3/3 repeats clean, so not deterministic. | `315hunt-p1l`, `315det-*`, §20/§20b |
| 4 | **Instruments built, proven live, and working**: copy probe (`armed`), logits sentinel (`filled`, 1 line), per-call logits hash (2732 lines/arm), per-tensor fingerprint (8271 lines/arm). | `315fp2-*l-*` |
| 5 | **Composition arithmetic, measured**: ≈17.7 GiB per neighbour (`-c 2048`), ≈18 GiB for the detector; `REGIME_LO=70 REGIME_HI=85` composes 3 neighbours → `gtt_start≈98 GiB` in band. The driver's `<neighbours>` argument is inert (parsed, never used). | §12, §12b, §19b |

### Remaining, in dependency order

1. **Name the specific compute stage.** *Partly answered by §23*: the fingerprint ran in a faulting arm
   and the head is exonerated — the head's input is already NaN — and every Face-A fault on the record sits
   on the first decoded token of a newly launched task. What is still open is the stage that manufactures
   the NaN inside that first-decode graph; the next measurement is the fingerprint on that graph's inputs
   (the activation handed to the first decode, and the KV it reads).
2. **Fix (task-4)** — cannot be written honestly before step 1 names the producer. Candidate directions
   already on the record: §66 option 3 (device-local output + one explicit copy), or fixing the stage
   that manufactures the NaN/garbage.
3. **Campaign (task-5)**: `~/wt/315-campaign.sh` is written and syntax-checked; it runs the pre-registered
   shape with `EXTRA_ENV` **empty** (§18b — the instruments must be off for acceptance), target 60 clean
   fix-cell arms, cap 80 pairs. Note `issue-315-analysis.py` still needs the §11 Face-B correction before
   its output is an acceptance result; the campaign script therefore reports from the driver's own per-arm
   lines, which already implement §5.
4. **Regression (task-6)**: `test-backend-ops -o MUL_MAT_ID` must be built with `-DLLAMA_BUILD_TESTS=ON`
   from the revision under test (baseline 108/108, fail=0, supported=110), plus the dense-model control.
5. **Write-back (task-7)**: forward-only pin move on `1bit/315-preregistration` (ancestor checks against
   both `e44c9d01a`, this branch's pin, and `26330cc20`, `main`'s shipped pin), then PR + `KNOWN-ISSUES.md`.

### Working inventory (so none of this has to be rediscovered)

| item | path |
|---|---|
| engine worktree / branch | `~/wt315-prereg`, `1bit/315-preregistration` (engine repo `~/projects/engine`) |
| fork source under instrumentation | `~/wt/llama-315copy` (pin + diagnostics + copy probe + sentinel + both fingerprints) |
| builds | `~/wt/315instr3-build` (control), `~/wt/315copy-build`, `~/wt/315fp-build` (instruments) |
| readers | `~/wt/315-probe-read.py`, `~/wt/315-logits-fp.py`, `~/wt/315-faceb-compare.py`, `~/wt/315-fp-run.sh`, `~/wt/315-campaign.sh` |
| driver + helper (pre-registered) | `~/wt/315-paired.sh`, `~/wt/315-regime.sh`, `~/wt/315-detector.py` |
| arm logs | `~/wt/rt-det/<tag>-p<N><c|l>-GLM-4.7-Flash-Q4_K_M/` (note: glob must be `*-*l-*`, not `*-l-*`) |

**Host note (cost one detour on 2026-10-09):** every `~/wt/...` path in the table above is on **strixhalo**
(`ssh strixhalo`, 192.168.50.110), not on the host that holds this engine worktree (`ryzen`, where `~/wt`
exists but holds none of the 315 artefacts). The engine worktree `~/wt315-prereg` is the one path that is
local. Nothing was ever lost; the working set only looked missing from the wrong box.

## 24. Two faces, one place — and one hypothesis eliminated offline (2026-10-09)

Everything below was derived from logs already on the box; no arm was needed, so none of it cost box time.

### Corrected: the length/tile-alignment test, redone on comparable arms

**This subsection corrects the first version of itself.** The first pass pooled the 18 arms that raised
`HRX returned NaN logits`, which mixes two different harnesses and request shapes: the pre-registered
detector harness (`detector.json`, `-c 2048`, all 2026-10-09) and the earlier study's `identical.json` arms
(2026-10-05..08, `-c 8192`). Prompt length as a function of request index is only defined inside one
harness, so the pool's lengths were meaningless and the "no clustering modulo 16" claim it produced was an
artifact. Redone inside the detector harness only.

Comparable arms: 122. The request shape is genuinely fixed - 106 arms share the identical 66-request
prompt-length prefix and the rest are that same prefix truncated at their fault, and all 66 request indices
have an unambiguous token length across arms. Face-A faults in this harness: **5**, and their faulting
request, prompt length and locality are:

| arm | faulting request | prompt tokens | mod 16 | lines after slot launch |
|---|---|---|---|---|
| `315fa-p2l` | 34 | 10 | 10 | 3 |
| `315fa-p5l` | 34 | 10 | 10 | 3 |
| `315fa-p7c` | 34 | 10 | 10 | 1 |
| `315hunt2-p6l` | 19 | 810 | 10 | 10 |
| `315sync2-p5l` | 10 | 172 | 12 | 2 |

So the faults do **not** scatter: 4 of 5 have `prompt_tokens mod 16 == 10` (three of them the same 10-token
request), 1 has 12. Only 3 of the 66 requests are `== 10` and 7 are `== 12`, i.e. the observed lengths fall
in a 10-of-66 (~15%) class. Under a uniform null that is p ~ 0.2% - **but it is post hoc**: the modulus and
the residue were chosen after seeing the data, and 16/32/64 were all tried, so it is a candidate, not a
result. What makes it worth keeping is that it is independently motivated by the kernel below (a
partially-filled last key tile is exactly where the shared-memory visibility predicate is read) and that it
is cheap to test prospectively:

* **Prediction.** In the detector harness, faults occur only at requests whose prompt length is
  `mod 16 in {10, 12}`; request 45 (458 tokens) is the one untested member of the `== 10` class, so it is the
  cheapest single check. A fault at any other residue falsifies the whole tile-alignment story.
* The three same-request-34 faults, the varied faulting request, and the fact that the lengths span 10 to 810
  tokens all say the trigger is not the request's *content*; alignment is a property of how the KV length
  lands in the key tiles, not of the prompt text.

### Both faces fire at the same place

Face A, comparable arms only: all 5 detector-harness faults sit 1-10 log lines after a `slot launch_slot_`
(table above), i.e. on the first decoded token of the newly launched task. The 13 further Face-A arms from the
earlier `identical.json` harness behave the same way (1-3 lines after a launch), which is why the pooled
pass saw 18 of 18; the pooled count is corroboration, not the primary evidence. Face B: of the 7
**detector-harness** arms whose detector reports an HTTP failure with no NaN, 4 emit their marker 6 lines
after a launch, the same boundary. The other 3 emit it deep inside a generation - but that marker is written
when the response is *parsed*, not when the bad data was produced, so its position is not a production-time
locator and those 3 neither support nor contradict the boundary reading.

### Hypothesis (falsifiable, both faces)

The first decode of a newly launched task consumes state that its own prefill produced — KV rows and the
attention mask — through the decode-split FA path, and in a fraction of launches it reads a region whose
contents are not yet valid. What the region happens to hold decides the face: a NaN pattern yields all-NaN
logits (Face A), stale-but-finite contents yield a wrong token stream that fails to parse (Face B). One
mechanism, two severities, one location.

Why here first: `ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom`
is the kernel whose tile-visibility logic already had to be changed for engine#314, both of whose commits
record that the change was **never verified against the authoritative masked-KV tool** and that the original
failure "does not currently reproduce on demand" (section 22 host note; the commits are ancestors of this
branch's pin `e44c9d01a`, i.e. the #314 fix is *in* and these faults remain). A masked-key visibility defect
in that kernel is exactly a first-decode, data-dependent, intermittent fault.

One fact from that history that matters for task-4: the pin carries **both** #314 candidates, and the merge
`26330cc20` took the *shared-memory* variant (`f9ff06675`, 102 lines: `all_visible` becomes "all 16 lanes of
this tile see a visible key", read from the tile's own shared-memory entries) **over** the always-stage
fallback (`cf4dfd801`, 6 lines: `all_visible = false`, `any_visible = true`, so every key tile takes the
staging path and never issues a collective). So the conservative path exists, is authored, and is *not* what
runs. If outcome 1 of the list below holds, that fallback is a forward-only candidate fix already in the tree,
which would make task-4 a small, reviewable change rather than a new kernel.

### Pre-predicted outcomes of the next arm

Run on the first decode of a faulting arm, fingerprinting the attention side by name — `attn_inp_kq_mask` is a
real node name (`src/llama-graph.cpp:40`) — with `NANONLY` set, plus the `tail:<N>` sweep from
`315-campaign/patches/315-nan-dump.md`:

1. **Mask clean, attention output NaN** -> the defect is inside the decode-split FA kernel (visibility or
   mask handling). This is the hypothesis.
2. **Mask itself NaN or wrongly `-inf`** -> the defect is in mask construction at the prefill->decode
   boundary, not in the kernel.
3. **Attention side clean, first NaN at `norm`** -> hypothesis **falsified**; the producer is in the
   layer/FFN path and the FA kernel is exonerated.
4. **Face B reproduces with the instruments on** -> its logits hash must diverge from the clean reference at
   exactly the first call of the new task, not drift; anything else falsifies the stale-read half of the
   mechanism.

### Where the source audit landed (unverified by measurement)

Reading the pin's own copy of the decode-split kernel narrows the mechanism to one array and one predicate,
but proves nothing on its own:

* a full block is 64 keys (`scf.for %key_tile = [%c0 to %c64 step %c16]`, four 16-key tiles), and the
  `all_visible` / `any_visible` predicates are now derived by loading the tile's 16 entries from
  `%visibility_view : view<64xf32>` at `key_tile_ordinal * 16` and testing `> %negative_large`;
* the non-staging fast path is taken only when `all_visible`, i.e. only when every key of that tile is
  genuinely visible - which is the right condition, *provided the array is current*;
* so the failure that fits every observation is a **stale or partially-written visibility array**: the
  predicate then reports `all_visible` for a tile whose keys are in fact masked, the kernel takes the fast
  path, and the masked-out V rows are consumed. Masked rows in a freshly reset slot are exactly where a NaN
  (Face A) or plausible-looking garbage (Face B) would come from, on the first decode after a task boundary,
  intermittently, and independently of GTT pressure.

Unverified: nothing here shows the array is stale, and the array's producer (prologue vs host-side setup)
has not been traced. The discriminating measurement is the one already listed: if the mask/visibility input
is stale at the first decode while the source mask is clean, this is right; if the first NaN instead appears
with a clean visibility array on the fast path, the predicate logic itself is wrong; if the attention side is
clean throughout, the whole FA direction is falsified and the producer moved to the layer/FFN path.

None of this is run yet: the box has been held across four consecutive goal turns by a foreign `repro336`
GLM-4.7-Flash session (7 resident servers at the last check, GTT 102.8 GiB, 1-2 GB RAM free), and the regime
gate needs `gtt_start` in [95, 110] GiB after composing neighbours.

### What the next measurement costs, and one lead that could make it cheaper
The pre-registered shape costs ~3.5-4.6 min per arm plus regime composition, and the fault rate is ~4%
per arm, so a faulting arm with the fingerprint live is a few arms' worth of box time, not one. Acceptance
(task-5) must keep the declared gate; **diagnosis does not** - it only needs a fault, and the fault is not
obviously a property of GTT pressure:

* the 5 comparable Face-A faults all sit at `gtt_start 98GiB`, i.e. inside the gate, so the gate is where the
  rate was measured;
* but `~/wt/rt-det/rtf-GLM-4.7-Flash-Q4_K_M` (2026-10-05, the earlier `identical.json` harness, `-c 8192`,
  one server and no neighbours) raised the same `HRX returned NaN logits` error at `gtt_start 23197143040`
  bytes = **21.6 GiB**.

That is one observation in a non-comparable harness, so it is a lead, not a licence to drop the regime: what
it suggests is that a lone server (~22 GiB GTT, no composition) can fault, which would let the *diagnostic*
arm run whenever ~22 GiB is free instead of needing the box emptied. Worth one deliberate check before
trusting it, and it does not change anything about the acceptance campaign.

## 23. The producer, at tensor resolution (2026-10-09, `315hunt2-p6l`)

Arm `315hunt2-p6l` is the first faulting arm to carry the per-tensor fingerprint
(`GGML_HRX_TENSOR_FP=result_norm,result_output`; copy probe, sentinel and `GGML_HRX_LOGITS_FP=100000` also
live; in-regime, `gtt_start 98GiB`). It faulted on request 19 of 66
(`HRX returned NaN logits: count=154880 of 154880 ... sentinel=0`), detector
`transport_failures=1`, `first_failure=RemoteDisconnected`.

Per-call record, faulting `graph_compute` (fp call 103644):

| node | bytes | NaN | hash |
|---|---|---|---|
| `norm` (final norm output) | 8192 | **2048 of 2048** | `89ade87e7a93fb83` |
| `result_norm` (head input) | 8192 | **2016 of 2048** | `77c1ef05f75f6127` |
| `result_output` (logits) | 619520 | **154848 of 154880** | `f8b83526883036a3` |

Across all 1091 fingerprint calls in the arm: 1089 of 1090 full-size calls have `norm nan=0`, and in every
one of those clean calls `norm` and `result_norm` are **byte-identical**. The fault call is the only call in
the arm where the two differ. Conclusions:

1. **The head is exonerated.** Its input is already NaN, so the head amplifies a corrupt activation rather
   than manufacturing the NaN. The producer is upstream of the final norm output.
2. **The NaN is arithmetic, not ours.** FNV-1a over 154,880 copies of the sentinel payload `0x7FC0DEAD` is
   `79889fb2391c6c83`; the fault call's logits hash is `91a04b03818e7083`, and no constant NaN payload
   reproduces it. So the logits are NaNs of *varying* payload — the sentinel did not leak into the region
   and was not broadcast.
3. **The host read is not ordered against the whole write.** `result_output` held 32 finite floats at the
   fingerprint read and the sampler's later read of the same region reported 154,880 of 154,880 NaN —
   128 bytes landed between the two reads. Same for the `norm`/`result_norm` pair (2048 vs 2016 NaN at the
   same instant).
4. **The fault is one token, at a task boundary.** Sampler calls 1085 and 1086 are clean, 1087 is all-NaN:
   no drift. And in **18 of 18** arms on the record that raise `HRX returned NaN logits`, the error is 1-3
   log lines after `slot launch_slot_ ... processing task` (the instrumented arm: 10 lines, with exactly
   one `hrx-logits-fp` line between the launch and the error — the faulty one). Every Face-A fault fires on
   the **first decoded token of a newly launched task**, i.e. in the first graph that consumes the state
   handed over from that task's prefill. (§24 corrects this count: it pooled two harnesses, and on the 5
   comparable detector-harness faults the locality is the same — 1-10 lines after a launch.)

Unexplained, worth keeping: the single call 103549 in this arm reported `bytes=0` for all three names
(`hash=14650fb0739d0383`, the FNV offset basis, i.e. an empty read); it is the only such call in 1091 and it
sits between the task launch and the fault.

Status: the fault is localised to one graph, one token, at a prefill→decode boundary — with the head and the
sentinel eliminated as explanations. The next measurement is to fingerprint that first-decode graph's own
inputs, i.e. the nodes ahead of the final norm. The instrument for it is written but **not built**:
`GGML_HRX_TENSOR_FP=tail:<N>` selects nodes by position (so no tensor name has to be guessed) and
`GGML_HRX_TENSOR_FP_NANONLY=1` keeps only the NaN-bearing lines, and each line now carries `op=` too. See
`315-campaign/patches/315-nan-dump.md` and `315-campaign/tools/gen-nan-dump.py`; it applies cleanly to
`~/wt/llama-315copy`. It is deliberately not built yet, because building it into `315fp-build` would replace
the binary that produced the evidence above.

Not done, and why: the strixhalo box was held across both 2026-10-09 evening goal turns by a foreign
`repro336` GLM-4.7-Flash session (`~/repro336/build/bin/llama-server`, ports 20101-20103 and 19656; 4 resident
servers, GTT 103-111 GiB). The pre-registered regime needs `gtt_start` in [95, 110] GiB after composing
neighbours, which cannot be reached while those hold the box, and the arm driver deliberately does not kill
servers that are not ours. They were left alone.

## 25. Research pass: what the fork already contains, and one instrument that does not measure what it claims

Read-only work: no arm was run, and none of this needed the box.

### The sentinel cannot support the inference drawn from it

`GGML_HRX_SENTINEL`'s fill lives inside `hrx_buffer_init` — the 619520-byte host-visible branch of
`ggml/src/ggml-hrx/ggml-hrx.cpp`. It runs once per buffer *allocation*, at load. In the faulting arm
`315hunt2-p6l` there is exactly one `[hrx-sentinel] filled` line, at 0.44.216 s, and the fault lands at
2.01.889 s after ~1087 tokens have written that same region. So `sentinel=0` at the fault only says that some
earlier token overwrote the pattern: it cannot separate "the device never wrote this token" from "the device
wrote it". Section 22 finding #2's clause "not one is our payload, so the write is present" over-reads it.

The conclusion survives on a sounder basis. Sampler call 1086 (the previous token) was finite (`nan=0`,
hash `f38cc8b5b54dbb8f`) and 1087 is `nan=154880`; and no constant NaN payload reproduces the fault's logits
hash (FNV-1a over 154,880 copies of one payload is not `91a04b03818e7083`). A region still holding the
previous token's logits would read `nan=0`; a region still holding the sentinel would read one constant
payload. Both are excluded, so the faulting graph did write all 154,880 elements as NaN. Face A is a
compute-side write; that half of section 22 stands, for a different reason than the one recorded.

### What the fork already contains

* **The masked-row leak in these kernels is a known, fixed family.** `e6187096b` (2026-10-02,
  `1bit/hrx-fa-masked-v`) is titled "masked KV rows reach flash-attention P*V as V = +0 (identical requests
  gave different logits)": P is not exactly zero in F16, so multiplying it by the V of a *masked* key made
  the output depend on the sign and value of whatever an earlier request (or a rejected draft) had left past
  the sequence end. The fix zeroes masked V rows per tile in both FA kernels and adds
  `tests/test-hrx-fa-masked-v` (27 cases; masked K/V rewritten five ways must give bit-identical output, NMSE
  against CPU <= 5e-4).
* **That fix is already in the pin.** `e6187096b` is an ancestor of `e44c9d01a`, as are *both* #314 commits;
  the pin's kernel carries the comment "Key tiles with every key visible read V directly, fully masked tiles
  use a zero V fragment, and mixed tiles stage V with masked rows zeroed", and `tests/test-hrx-fa-masked-v.cpp`
  is in the tree. The family's first instance is therefore closed in the revision we ran.
* **#314's predicate rewrite is not a regression, on the source.** `%visibility_view` is LDS written by
  subgroup 0 (`view.store %lane_mask, %visibility_view[%lane]`), published before a `kernel.barrier<workgroup>`
  of the same scope, and the 16 new reads test the same predicate the vote used (`cmpf ogt, mask,
  %negative_large`, cf. `%lane_key_visible`). "The predicate is wrong, so masked rows went unzeroed" is not
  supported by reading it.
* **The fork has a forward-only engine#315 change, and it is not the answer.** `fb4dff7bb`
  (`engine-315/host-visible-coherence`, 2026-10-09 02:47) is "ggml-hrx: make host-visible buffers coherent
  unconditionally" - 7 lines, `memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT` - and the pin *is* its ancestor,
  so it is forward-only. But section 13 already falsified coherence as a sufficient mechanism with its own
  campaign, so this is not the fix to move to.
* **Nothing post-pin on either remote fixes an FA, visibility or NaN defect.** The only other post-pin
  `ggml-hrx` commit is `30d6d3b44` (loom-jit runtime globals). The fix for the remaining fault has to be
  written, not found.

### The mechanism that survives, and a discriminator that costs minutes

The predicate that decides whether a tile's masked rows are zeroed is computed from **one mask value per
key** (`%lane_mask` staged into `%visibility_view`, then `cmpf ogt` against `%negative_large`), while the
accumulation it gates covers **several query rows** (`masked_score0..3` all add that same `%lane_mask`). Where
the rows of a tile do not share one mask - the sequence end falling inside the tile, or any multi-row decode
whose rows end at different keys - the predicate can be right for one row and wrong for the accumulating one,
so a tile that is masked for that row is judged all-visible and read directly, which by the Oct 2 design does
**not** zero masked rows. Those rows hold the previous request's data past the sequence end, so a finite
stale row gives a small wrong contribution (Face B: wrong-but-finite text, HTTP 500) and a non-finite stale
row gives `tiny_P * NaN = NaN` (Face A: all 154,880 logits NaN). Intermittent, history-dependent, and it
lands on the first decode of a new task, because that is where the sequence end falls inside a tile.

Stated as a hypothesis with two falsifiers: if `test-hrx-fa-masked-v` passes with masked K/V rewritten as
`NaN`/`+inf`/`-inf` (not only `+0`/`-0`/`+x`/`-x`/random) *and* against a KV tail primed by a previous
request, this route is dead; and section 24's outcome 3 still kills the whole FA direction if the faulting
arm's `attn_inp_kq_mask` fingerprint is clean.

What makes this worth running first is cost. The discriminator already exists - 27 kernel cases, seconds on
the device - instead of 60 in-regime arms. It needs extending in exactly the two directions the hypothesis
names: non-finite masked V rows, and decode-split cases with 2..15 query rows whose masks end at different
keys. That is a test edit, not a campaign.

If it holds, the fix is to stop letting the predicate decide whether rows are zeroed: re-land `cf4dfd801`'s
constant predicates (`all_visible = false`, `any_visible = true`) as a fresh commit on the pin, which sends
every tile down the staged path where masked rows are zeroed. Forward-only (the pin stays an ancestor), at the
cost of the staging loop on every tile - the trade the #314 merge declined.

## 26. The FA masked-row direction is falsified at the regime's shape (2026-10-09)

This closes the direction sections 24 and 25 opened. It cost minutes, not a campaign, and it was
pre-registered before it ran.

**What ran.** The fork's own `tests/test-hrx-fa-masked-v`, extended two ways - non-finite masked V rows
(section 25) and regime-shaped cases - on the fork branch `engine-315/fa-masked-v-nonfinite`
(`c2114574d`, `ec0175971`) built on `26330cc20`, with `ggml/src/ggml-hrx/` byte-identical to the pin
`e44c9d01a` at that moment. Evidence: `315-campaign/evidence/315-fa-masked-v-regime.md`.

The environment is part of the result: without `IREE_HAL_AMDGPU_LIBHSA_PATH` the HRX device does not
register at all and the test prints `no HRX0 device, skipped` - the same trap the regime script's own
comment records for the neighbours.

**Result: 40 of 40 cases pass**, every one reporting `differing 0/8, non-finite outputs 0`, on both
kernels (`ggml_flash_attention_decode_split_f32_f16_wmma_next_q8` and `ggml_flash_attention_f32_f16_wmma`),
including the 13 added regime-shaped cases: `cells=2048` at `visible` 10, 11, 150, 172, 173, 810, 811, 2047
and 512/810/868 with 2-15 query rows, plus the prefill boundary at 16 rows.

**Reading, from the table pre-registered in `patches/315-fa-masked-v-nonfinite.md`:** "all cases pass,
non-finite outputs 0 -> the decode-split and prefill kernels do not consume masked V rows in any of the
covered shapes; the hypothesis is dead and the producer is elsewhere". So it is:

* section 24's "#314's predicate rewrite reintroduced the Oct 2 masked-row leak" - falsified;
* section 25's "the predicate is one mask value per key while the accumulation covers several rows" -
  falsified;
* the "re-land `cf4dfd801`'s constants" fix candidate is **withdrawn**. It would change the kernel and fix
  nothing, which is exactly the kind of change the objective forbids.

**What this does not exclude, and where the probes now point.** The test says the kernel is clean *given
clean inputs*. It says nothing about whether the FA inputs are themselves non-finite - the mask, or the K/V
that the task's own prefill wrote - and nothing about the rest of the decode graph (norms, the routed
FFN/MoE, the LM head). Those are the surviving loci, and they need an arm-level fingerprint
(`attn_inp_kq_mask` by name, or the `tail:<N>` sweep), not a kernel test.

**One practical finding worth keeping.** That run happened at 103 GiB of GTT held by a foreign session: the
kernel test needs almost no device memory. So kernel-level and compile-level iteration is available even
while the box looks full; only arm-level work needs the regime, and it is the only thing still waiting.

## 27. Kernel-level tests are exhausted, and all of them pass

With section 26 the kernel routes are closed, so the whole HRX test suite was run at the pin's kernel on the
same cheap footing (evidence and command: `315-campaign/evidence/315-hrx-tests-baseline.md`):

| test | result |
|---|---|
| `test-hrx-fa-masked-v` (extended) | 40 cases passed, `differing 0/8, non-finite outputs 0` |
| `test-hrx-decode-stride` | 72 cases, 0 failures |
| `test-hrx-moe-split` | all cases OK |
| `test-hrx-mul-mat-id-k32` | all cases OK |
| `test-hrx-buffer` | 5 passed, 0 failed |
| `test-hrx-attention-sink` | disjoint output accepted, aliasing refused |
| `test-hrx-hadamard` | PASS |
| `test-hrx-mxfp4` | 7 rows x 256 values bit-exact |

Two do not compile at this revision, both pre-existing and unrelated to the masked-V work:
`test-hrx-ops.cpp` (`fatal error: dispatch_registration/common/dispatch-activation-publication.h file not
found`) and `test-hrx-loom-jit.cpp` (API drift against the HRX revision in use). They are recorded here
because task-6's "engine CI green" claim has to account for them, and because a CI result that silently skips
two test targets is not a green suite.

**What this settles.** No kernel's standalone behaviour exposes the fault: not the FA kernels (40 cases
including the regime's own `cells=2048` shapes), not the KV handoff (`decode-stride`), not the routed FFN
(`moe-split`) and not `MUL_MAT_ID`. The producer therefore lives in model-level state under the regime - the
allocation/streaming path that only exists when the box is loaded, or the *content* of the mask and K/V at
fault time - and both of those need arms, not tests.

That also removes an easy trap: a "fix" aimed at a kernel's logic now has nothing to stand on. The earlier
`cf4dfd801` candidate is withdrawn (section 26), and nothing in the suite suggests a replacement.

**Cheap footing is worth keeping.** Every test above ran at 103 GiB of GTT held by a foreign session, i.e.
kernel-level and compile-level work needs almost no device memory. Arm-level work is the only thing that
needs the regime gate, and it is now the only thing outstanding.

## 28. A cheap reproducer at the pin, found while the box was otherwise unusable

Sections 26-27 left the producer at model level and the box unable to run GLM arms. The engine's own
`docs/hrx.md` documents the historical instance of exactly this class - engine#123/#140: a producer wave
writes global memory, a consumer wave reads it back, and the barrier between them fenced **LDS only**
(`barrier<workgroup>`, no store wait, no `buffer_gl0_inv`), so the consumer could quantize the previous
contents of the slot; "the f32 output was right and `next_q8` was wrong, and NaN when the stale bytes decoded
as NaN". The window opens on dispatch preemption, which is why the rate followed memory-pressure settings.
That fix is in the pin. Its repro recipe was not tried again until now.

Recipe: the study's own stress knob plus forced queue evictions, on a model small enough that no regime is
needed. `GGML_HRX_FA_PARTIAL_ALIGN=256`, `-fa on`, Qwen3-0.6B on HRX0, 180 identical greedy requests in one
server process, with `sudo -n tee /proc/sys/vm/compact_memory` looping in the background. Harness:
`315-campaign/315-cheap-probe.sh`; evidence: `315-campaign/evidence/315-cheap-repro.md`.

| configuration | runs | outcome |
|---|---|---|
| default align, quiet | 1 | clean, 180/180 identical |
| default align + compaction | 2 | clean, 180/180 both |
| align=256, quiet | 1 | clean, 180/180 |
| **align=256 + compaction** | **5** | **1 faulty (95 distinct outputs of 180), 4 clean (180/180)** |

**Correction to the first version of this section.** It presented the table above as a clean 2x2 and called
the configuration a reproducer. Five runs of the faulty cell do not support that: it fired once (95 distinct
outputs, tokens 0/2/7/23 = 2/7/35/121 distinct logits hashes) and was clean four times at 180 requests each.
The other three cells have one or two runs each and were always clean, so "needs both the layout knob and the
evictions" is a reading of single runs, not an established result. What is established is narrower and still
worth having: **a preemption-sensitive wrong-output event is possible at this pin**, it happened at align=256
with compaction running, its onset was the first decoded token, and every sampled logit was finite. The
runs: `315align256` (faulty), `315r256-1`, `-2`, `-3`, `-4`, `-5` (clean), plus `315a256-quiet`,
`315fa-ctl`, `315fa-compact`, `315r4096-1`, `315ctl-nocompact`.

Why it is not controllable yet: the one faulty run happened while the foreign session on the box was busy
(load average 8-12); the four clean repeats ran while it was quieter (load 4-7). That fits the class's own
description - the window opens on queue preemption, and page compaction alone may not be enough without real
contention - but it is a correlation over five runs, not a measurement. Making it a tool means creating the
contention deliberately (a second HRX workload of our own running against the same device while the probe
runs) rather than borrowing the neighbour's.

**Follow-up, 2026-10-10:** the contention was never observed. A watcher was left measuring the neighbouring
session's own `/proc` CPU counters for 1200 s and saw no device work at all - the `repro336` session holds the
shared box lock (`~/.cache/lax-decode/box.lock`, fd 9, inherited by its three `llama-server` children) while
sitting idle. So the box is *memory*-occupied, not compute-occupied, which is a refinement of the blocker: the
regime gate is blocked by their resident GTT, not by contention. The paired probe run did not happen either -
my watcher invoked `~/wt/cheap-probe.sh` while the harness had been copied to `/tmp/cheap-probe.sh`, so the
sample was lost to a path bug, recorded here rather than papered over. Given section 34 (2250 clean FA reps on
the pin) the cheap-probe path is not worth re-running without a specific hypothesis to test with it.

Three things follow, at the strength the evidence actually supports:

1. A wrong-output event of this class is reachable at the pin without GLM, without the 98 GiB regime and
   without neighbours - but rarely, and not yet on demand.
2. Its onset in the faulty run was the first decoded token, the same locality the arms show for engine#315
   (section 23).
3. All sampled logits stayed finite (`nan=0`), so that shape is Face B; nothing here has produced Face A.

The knob is not merely misaligning a buffer - it changes which kernels the dispatcher selects
(`GGML_HRX_LOG_DISPATCH=1`): at align=256 `common.flash_attention_decode_split_next_q8` never runs and
`kquant.mul_mat_add.decode_f32 (fused)` does; at the shipped align=4096 it is the other way round. The
registry's own skip knob separates them: with `GGML_HRX_DISABLE_DISPATCH=kquant.mul_mat_add.decode_f32`
the fault still happened (68 distinct outputs of 180, that same evening, one run), so the matmul-add fusion
is not the carrier. What is left is the flash-attention substitution - at align=256 the single-token decode
is served by `common.flash_attention_f32_f16_wmma`, the prefill-shaped kernel, instead of the decode-split
one. That kernel has no global-to-global handoff to blame (its only stored-and-loaded buffer is the mask
input, written by a separate test-only helper: auditing its views by root buffer shows `%score_stage` as LDS
and nothing else), so if it is the carrier, the defect is not the missing global fence of section 28's
opening paragraph but something else in how that kernel is dispatched or parameterised for a decode shape.

What this does and does not say. It does **not** show that #315's in-regime GLM fault is this instance: the
knob is a documented stress knob, not the shipped default, this is a different model and shape, and the event
fired once in five attempts. What it does say is narrower: a wrong-output, preemption-sensitive event of this
family was possible at the pin on 2026-10-09, at a non-default layout, starting at the first decoded token.
That is a lead, not a reproducer - but the probe itself costs a minute and does not need the box to be free,
which no other artefact in this investigation can say.

### The arm probe is built and verified, so the next arm needs no setup

Sections 26-27 pushed the surviving locus to model-level state under the regime, which only an arm can see.
That arm's instrument is now in the vehicle: `~/wt/llama-315copy` carries the additional nan-dump patch
(`315-campaign/patches/315-nan-dump.md`, `GGML_HRX_TENSOR_FP=tail:<N>` + `GGML_HRX_TENSOR_FP_NANONLY=1`,
lines carrying `op=`), and `~/wt/315fp-build` was rebuilt from it - the build succeeded and
`GGML_HRX_TENSOR_FP_NANONLY` is present in the produced `libggml-hrx.so`, alongside the older
`hrx-tensor-fp`, `hrx-sentinel` and `hrx-copy-probe` markers.

The next diagnostic arm is therefore:

```
REGIME_LO=70 REGIME_HI=85 CONTROL_EVERY=999 NOSTOP=1 \
EXTRA_ENV="GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000 \
           GGML_HRX_TENSOR_FP=tail:24 GGML_HRX_TENSOR_FP_NANONLY=1" \
  bash ~/wt/315-paired.sh 15 2 2048 ~/wt/315instr3-build/bin ~/wt/315fp-build/bin 315nan
```

It logs only the NaN-bearing nodes of each graph's tail, with their op, so a faulting arm names the stage
that first goes NaN instead of restating the four already-falsified levers.

Provenance, stated plainly because it is a change to a binary that produced earlier evidence: `315fp-build`
is the *diagnostic* vehicle (`315fix-build` and `315instr3-build` are the acceptance cells and are untouched),
and both new modes are opt-in - with `GGML_HRX_TENSOR_FP` unset or set to a name list, behaviour is exactly
as before. Section 23's evidence is the raw log `~/wt/rt-det/315hunt2-p6l-*/server.log`, which is unchanged.

### How to run that test

The patch is written, generated from the live tree and verified to apply:
`315-campaign/patches/315-fa-masked-v-nonfinite.md` (generator `315-campaign/tools/gen-fa-masked-nonfinite.py`).
To run it once the box is free:

1. `git apply -p1` that diff to a branch of the fork at the pin `e44c9d01a`;
2. build the one target in the **tests-ON build dir that already exists**, `~/wt/v315-build`
   (`LLAMA_BUILD_TESTS=ON`), and run `<dir>/bin/test-hrx-fa-masked-v`. Every case prints
   `masked-row variants differing N/8, non-finite outputs M`.

Two facts make that cheap rather than a rebuild:

* `ggml/src/ggml-hrx/` - the kernel, the test file and the whole backend directory - is **identical** between
  `26330cc20` (the revision `v315-build` was configured against, `~/wt/llama-hybrid`) and the pin
  `e44c9d01a`; `git diff --stat 26330cc20 e44c9d01a -- ggml/src/ggml-hrx/ tests/test-hrx-fa-masked-v.cpp` is
  empty. So the library that build links against already is the pin's kernel, and the patch applies cleanly to
  that tree as well.
* the remaining work is one translation unit plus a link, not a campaign.

The only resource it needs is RAM (1-7 GB free at the last check, on a box whose foreign session holds
114 of 122 GB) and CPU - **not** GTT headroom, and not the regime gate. So it can run as soon as the shared
box has a little memory spare, before the box is anywhere near empty, which is what makes it the right next
move instead of the 60-arm campaign.

The engine pin itself is declared in `registry/architectures.json` (`"llama.cpp (hrx)"`), which is the line a
forward-only pin move has to change for task-4/task-7.

### Already checked, at zero box cost

`/home/bcloud/tmp/bump-hrx/llama` on `ryzen` is a fork checkout at **the pin `e44c9d01a`**, and the build
directories `~/wt/loom-dflash2-ownfile/build-hrx` and `~/wt/loom-gdn-multiseq/build-hrx` carry
`compile_commands.json` entries for `tests/test-hrx-fa-masked-v.cpp`. That is the exact project compile
command (local SDK `amdclang++`, `-Wall -Wextra -Wpedantic -Wcast-qual`, HRX include paths), so the patch can
be type-checked on `ryzen` without touching strixhalo:

* the patched file compiles clean - exit 0, **no output at all**, i.e. no warnings under those flags;
* the check is real, not a no-op: appending a deliberate syntax error to the file makes the same command exit
  1 with `error:` diagnostics, and restoring it returns to exit 0.

So the only thing left for this discriminator that needs the box is *running* it, and running it needs RAM and
CPU - not GTT, not the regime gate. Compile-level iteration costs no box time at all.

## 29. What the day eliminated, and what is left

Consolidation, so the next reader (or auditor) does not have to reassemble it from eighteen sections. Every
line below is a measurement or a read of the tree, not a preference.

### Eliminated, with the cost it took

| lever | status | evidence |
|---|---|---|
| host-visible coherence is the mechanism (the recorded task-3 story) | **falsified** | its own campaign, section 13; and section 21/22: the region *is* written |
| the ordering fix is the mechanism | **falsified** | section 15, that fix failed the same test |
| "the logits buffer is never written" | **falsified** | section 21; section 25 adds that the sentinel could not have decided it either way (it fills once per allocation) |
| the FA kernels consume masked V rows, incl. #314's predicate rewrite | **falsified** | section 26: 40/40 cases on the device, both kernels, regime shapes, `non-finite outputs 0` |
| any kernel's standalone logic | **falsified as a locus** | section 27: 9 runnable HRX tests pass, incl. decode-stride, moe-split, MUL_MAT_ID |
| the #123/#140 global-fence class explains the cheap-path event | **not supported** | section 28: the kernel serving the decode at align=256 has no global-to-global handoff at all |
| `kquant.mul_mat_add` fusion carries it | **falsified** | section 28: the event happened with it disabled |
| arena knobs, transient reuse | falsified earlier | already on the record, per the objective |

### What survives

1. **Both faces fire on the first decoded token of a newly launched task.** 5 of 5 comparable detector-harness
   faults sit 1-10 lines after a `slot launch_slot_`; the pooled set adds 13 more from the earlier harness with
   the same locality (section 23). Face B's marker sits at the same boundary in 4 of its 7 arms.
2. **Face A is a compute-side write.** The faulting graph wrote all 154,880 logits as NaN with varying
   payloads, while the previous token was finite and no constant payload reproduces the hash (section 25).
   The LM head is exonerated: its own input is already NaN (section 23).
3. **The event needs preemption.** Compaction alone never did it; the one time it happened, the neighbouring
   session was busy; a later attempt with our own `llama-bench` contention did not. So the window is real and
   the trigger is not yet under our control (section 28, corrected).
4. **The probe is cheap.** A minute, no GLM, no regime, no neighbours; kernel-level and compile-level work
   needs almost no device memory and ran fine at 103 GiB of foreign GTT.

### What remains, and what each needs

* **An arm with the nan-dump instrument** (`GGML_HRX_TENSOR_FP=tail:<N>` + `NANONLY`, built and verified in
  `315fp-build`) - names the first NaN-bearing node in a faulting graph. Needs the regime gate, so needs the
  box all but empty. Command is in section 27.
* **A trigger we control for the cheap path** - a fault rate of 1 in 5 runs is not a reproducer. The two
  candidates left are a contention source that actually perturbs KFD queues (our own bench did not) and a
  longer-KV shape closer to the arms.
* **The fix (task-4) stays unwritten on purpose.** With the FA direction dead, the coherence branch falsified
  and the fence class not applicable, there is nothing left that a minimal patch could be aimed at. Writing one
  now would be guessing, which is what the objective forbids.

### One conflict the user should see

The task list has **task-3 complete** with the mechanism "host-visible non-coherent output region read directly
with no ordering on the route". Sections 13, 21, 22 and 26 falsify that mechanism, and the later work did not
replace it with an equally concrete alternative. Completed tasks are immutable under the goal's own rules, so
this is not being edited here: it means tasks 4-7 would be validating a change aimed at a mechanism the record
no longer supports, and the tree should probably be restructured (the user's call, via `/goal-tweak`).

## 30. The engine's own gates, run at the pin

Task-6 needs "engine CI green" and task-7 needs a forward-only pin move. Both of those gates can be exercised
without a GPU, so the baseline and the gates themselves are now on the record:
`315-campaign/evidence/315-engine-ci-baseline.md`.

Run in `~/wt315-prereg` at `third_party/llama.cpp = e44c9d01a`: `tools/copyright.py --check` exits 0 (after
`b777bdc` - it was failing on my own three `evidence/*.txt` files and the copied probe script, i.e. the branch
would have failed CI on artefacts of this investigation); `tools/check_pins.py 6d5e470` and `beec5e6` both exit 0
with `third_party/llama.cpp: ... -> e44c9d01a is ahead`; the three `ci.yml` python tests pass; and the two ctest
registry tests pass.

Three things that matter for the remaining tasks:

* **The forward-only gate works here and is machine-enforced** - but only over **submodule gitlinks**. The hrx
  revision in `registry/architectures.json` is not tied to the gitlink by anything in `tools/` or `tests/`, so a
  change that moves one and not the other passes CI while leaving the tree inconsistent. Task-7 has to move both,
  by hand.
* **A #315-motivated pin move already shipped and did not fix it.** `6d5e470` ("pin llama.cpp to the
  shared-memory FA tile visibility (engine#314/315)") took the pin from `cf4dfd801` to `26330cc20`, and sections
  26-27 falsify that whole locus. So the eventual change cannot be "bump to the next AMD pair" - it has to be a
  new fork commit that is a descendant of `e44c9d01a`.
* **The regression half is still untested**: the `gcc-15` build of `onebit` plus the 14 test targets and `ctest`
  have not been run at all, because there is nothing to regress against yet. That is the one piece of task-6 that
  will cost real time when the fix exists. (`g++-15` is installed on `ryzen` - 15.2.0 - but all nine submodules are
  uninitialised in this worktree, so the build starts with `git submodule update --init`.)

## 31. Reconciling with KNOWN-ISSUES.md: what was already measured

The issue's own entry is a substantive record, and reading it end to end changes what sections 24-30 are worth.
Three things in it are load-bearing here, and one of them is a measurement that closes a candidate this session
had floated.

**The rate study is already done, in-regime, on both revisions.** From `KNOWN-ISSUES.md`:

```
A  shipped fix  (10604 = 3971f49bf)   40 scheduled, 40 in-regime, 37 complete
   clean 66/66: 35   NaN faults: 2 (both parse=0, inside the gate)   rate 2/37 = 5.4%  (95% CI 0.7-18.2%)   parse-500: 3
B  always-stage (10605)               30 scheduled, 30 in-regime, 25 complete
   clean 66/66: 24   NaN faults: 1                                    rate 1/25 = 4.0%  (95% CI 0.1-20.4%)   parse-500: 5
Fisher exact two-sided p = 1.000;  P(rate_A < rate_B) = 0.460
```

So the **always-stage variant that section 26 floated as a fix candidate has already been measured**: 62 arms
across the two cells, no reduction. The withdrawal is therefore corroborated by measurement and not only by the
kernel test - and it also fixes the number task-5's power statement is really about: the shipped pin's Face-A rate
in-regime is **5.4%**, so 60 clean arms bounding the rate below ~5% is a real improvement claim, not a formality.

**The issue text still carries the superseded Face-A story.** It says "the logits buffer is never written" and
quotes the sampler guard. Section 21 falsified that (the faulting graph wrote all 154,880 logits as NaN, with
varying payloads) and section 25 showed the sentinel could not have decided it either way. Task-7 has to correct
that paragraph, not append to it.

**One structural lead in the issue that sections 24-30 never mention:** "GLM is the only model whose decode splits
into two 128 KB AQL command blocks (919 + 760 dispatches) with 1,399 barriers per token", filed under the
kernel/binding family. That is a concrete, model-specific structural difference at exactly the layer the fault
lives in - model-level state, first decode of a task - and it is a better lead than anything left in the kernel
tests. It should be the first thing the next mechanism attempt checks: where the two blocks split, and what
ordering the boundary does or does not preserve.

**Also on the record and not to be redone:** the staged host-writeback path is excluded (11 valid arms, 0 `HRX host
writeback` lines, reporting path proven under `=poison`); the direct device-to-host copy *is* covered by the wait;
`GGML_HRX_TRANSIENT_REUSE=legacy` has no effect; pinning `f2099e9b7` would be a revert; and the issue lists four
traps its own history sprang - the last of which, "stale incremental builds", is worth stating against this
session's own work: `315fp-build` was rebuilt *incrementally* after the nan-dump patch, which is a source change
plus relink rather than a stale object, but any arm used for acceptance must come from a fresh build directory with
the engine's own `ExternalProject` arguments (section 27 records the diagnostic rebuild as such).

**The pre-existing probe that names the producing graph `value`** (implemented on `f2099e9b7`, deliberately
unbuilt because relinking there recompiles every TU that includes `backend-context.h`, and that CPU confounds a
GTT-bound measurement) is the task-2 instrument. It is complementary to the nan-dump fingerprint built here: the
fingerprint says which node goes NaN first, that one says which graph value produced the downloaded bytes.

### Following the AQL-block lead, and where it stops

Section 31 names the strongest untested lead: GLM is the only model whose decode splits into two 128 KB AQL
command blocks. I traced it as far as the constant, then stopped before it became an internals expedition:

* the block size is `IREE_HAL_AMDGPU_AQL_PROGRAM_DEFAULT_BLOCK_SIZE = 128 * 1024` in
  `runtime/src/iree/hal/drivers/amdgpu/aql_program_builder.h` (hrx-system), described as "default usable bytes per
  command-buffer block", with `IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE = 256`;
* a program is a *linked* chain (`iree_hal_amdgpu_aql_program_t` has `first_block`, `block_count`,
  `max_block_aql_packet_count`, `max_block_kernarg_length`), and the builder has a flag whose comment is "the
  current block has recorded its initial barrier packet count" - so barrier packets are counted inside the block
  budget, which is the arithmetic that makes a barrier-heavy model (GLM: 1,399 per token) the one that
  overflows;
* **`iree_hal_amdgpu_aql_program_block_next` has only three callers: the definition, the profiler, and the
  validator.** Nothing in the submission path walks blocks with it. So the boundary handling - where the blocks
  are copied into the AQL ring and what ordering, if any, is established between them - lives in a function I have
  not found yet, and that function is the next thing to read. The question to ask it is narrow: when a program
  spans two blocks, does the second block's record inherit the first block's completion ordering, and what happens
  to a global-memory producer/consumer pair that the split falls between?

This is a lead with a named file and a named next question, not a finding. It is recorded here because it is
directly testable against GLM's own shape and because it is the only structural difference on the record that is
specific to the affected model.

## 32. The block-boundary asymmetry - the leading candidate, and what would settle it

Following section 31's next step into `aql_program_block` and the AQL emitter turns up a specific asymmetry. All
of it is in hrx-system, `runtime/src/iree/hal/drivers/amdgpu/`:

1. **A split is a pure capacity event.** `aql_program_builder_split_block` (`aql_program_builder.c:306`) appends a
   `BRANCH` terminator, finalizes the current block and begins the next. It carries across exactly two things: the
   `HAS_PENDING_EXECUTION_BARRIER` flag and, if that flag is set, the pending barrier's **acquire** scope. Nothing
   else - notably not `initial_barrier_packet_count`, which `begin_block` resets to 0 and `finalize_block` then sets
   to the whole block's packet count unless the block set `HAS_INITIAL_BARRIER_PACKET`.
2. **The packet barrier bit is decided by a *logical* index.** `..._packet_has_barrier` (`aql_block_processor.c:105`)
   sets the AQL barrier bit when the packet carries `PACKET_FLAG_EXECUTION_BARRIER`, or `PACKET_FLAG_FINAL`, or
   `logical_packet_index == 0` with a waiting submission, or `logical_packet_index == 0` with a non-NONE inline
   acquire scope. And `logical_packet_index = processor->packets.index_base + packet_index` (line 128).
3. **`index_base` and the FINAL flag are per block, keyed on zero.** The submission site
   (`host_queue_command_buffer_block.c:300-345`) builds the processor per block with `.index_base =
   packet_index_base`, and sets `IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET` **only when
   `packet_index_base == 0`**. Identically, the acquire scope is raised to the submission's `inline_acquire_scope`
   only for `logical_packet_index == 0` (lines 132-143).

Read together: the submission-level ordering machinery - the inline acquire scope, the wait-barrier condition and
that FINAL flag - is attached to the **first block** of a program. A **second block gets none of it**; its packets
carry a barrier bit only if their own command recorded an execution dependency. Whether that is a defect or a
deliberate, correct design depends on semantics I have not verified (program order in the ring may already supply
the ordering), but it has exactly the shape this issue needs: it is **only reachable by a model whose command
program exceeds 128 KB**, and GLM is the only one on the record (919 + 760 dispatches, 1,399 barriers per token);
it concerns completion ordering rather than data layout; and a preemption window would be what turns a benign
overlap into a stale read.

**What would settle it, cheaply and without the regime.** Two steps, both driver-level:

1. Read the rest of the submission path and answer one question in code: when a program spans two blocks, what
   makes the second block's first payload packet wait for the first block's last dispatch? If the answer is "the
   branch terminator's semantics" or "a recorded dependency that is always present", the candidate dies here.
2. If it does not die in code, build the discriminating program: exceed 128 KB deliberately, put a
   global-memory producer at the end of block 1 and its consumer at the start of block 2 with **no** recorded
   dependency, and check the block-2 read against the block-1 write - first with the queue idle, then with
   compaction forcing queue evictions, which section 28 showed is how this family's window opens. One test,
   seconds, no model, no regime, no neighbours.

Explicitly **unverified**: I have not shown that the second block's packets are missing an ordering they need,
only that the submission-level bookkeeping is attached to the first block alone. Recording it as the leading
candidate rather than a finding is deliberate - the two steps above are what make it a mechanism or kill it.

### Correction: step 1 kills the asymmetry I described

I ran step 1, and the asymmetry is not real. The `packet_index_base` that reaches
`..._write_command_buffer_block` is **not** a block ordinal: the only call site
(`host_queue_command_buffer_block.c:879`) passes `profile_queue_device_prefix_packet_count`, which is
`queue_device_event ? 1u : 0u` - a profiling prefix, 0 whenever profiling is off. The replay loop
(`host_queue_command_buffer_replay.c:337/364`) submits **each block as its own submission** via
`iree_hal_amdgpu_host_queue_submit_command_buffer_block`, which takes no packet-index argument at all.

So `packet_index_base == 0` holds for every block, the first-packet conditions in `..._packet_has_barrier`
(`logical_packet_index == 0` with a waiting submission or an inline acquire) and the `FINAL_PAYLOAD_PACKET` flag
are evaluated **per block**, not once per program, and the "second block gets none of the submission-level
machinery" reading was wrong. This is why it was labelled a candidate rather than a finding, and it is recorded
as a wrong turn rather than deleted, because the next attempt should not spend the same hour on it.

What is left of the lead, stated as questions rather than a story:

* what sets `PACKET_FLAG_FINAL` on payload packets, and whether the first payload packet of a *second* block
  ends up carrying a barrier bit (the flag named FINAL is set from `packet_index_base == 0`, which is 0 for every
  block, so either every block's packets are barrier-bearing - which would serialize the queue and cannot be the
  design - or the flag is consumed somewhere I have not read);
* the replay loop consumes the wait resolution after the first block and submits the rest with an empty signal
  list, so the second block's ordering against the *first* block rests entirely on packet barrier bits rather than
  on the resolution;
* and the 128 KB split point in GLM's stream is still unexamined: whether it happens to fall between a producer
  and its consumer is an empirical question about that model's command stream, not a code property.

The AQL-block lead is therefore **open, not established**, and it is no longer the leading candidate on the
strength of the asymmetry - that reason is gone.

## 33. Static audit of the kernel family: no GLM-relevant unfenced global handoff

With the FA masked-row route dead (section 26), all kernels passing (section 27) and the AQL asymmetry cleared
(section 32), the remaining kernel-side possibility was the #123/#140 class itself: a producer wave writing
**global** memory that a consumer wave inside the *same dispatch* reads back, fenced only at LDS scope. That is
invisible to the host-side dependency recorder, so a static audit is the right instrument for it.

Method (`/tmp/scan_global_handoffs.py`, results below): for every `loom-libs/ops/*.loom`, resolve each
`buffer.view`/`buffer.assume.alignment` chain to its root buffer, find root buffers that are both stored *and*
loaded inside the file, and report each one next to the kernel's `barrier<global>` count.

Result, abbreviated to the handoffs that matter:

| kernel | handoff buffers | `barrier<global>` | reading |
|---|---|---|---|
| `flash_attention_decode_split_f32_f16_wmma` | `partial_max/sum`, `counter_scratch`, `visibility_stage`, ... | **13** | fenced - this is where #140's fix lives |
| `mul_mat_symmetric_i4_wmma` | `partial_na`, `scratch`, `output_na` | **18** | fenced |
| `rmsnorm_binary_f32` | `scratch` | **1** | fenced |
| `mul_mat_q6_k_packed_f16_wmma` | `partial_values_noalias`, `partial_ids_noalias` | 0 | **different dispatches**: one launch stores them (837/838), a second launch loads them (881/882), so the ordering belongs to the host-side recorder, not to a kernel fence |
| `mul_mat_id_swiglu_f32_f32_wmma`, `moe_routing_tables`, `mul_mat_q5_k_q8_plane`, `small_rows_f32`, `attention_sink_f32`, `hadamard_f32` | `*_stage`, `%scratch`, `%data`, `%route_stage` | 0 | LDS by the `_stage` convention, or scratch used within one wave |

So there is **no GLM-relevant kernel with an intra-dispatch global handoff and no global fence**: the three
kernels that do hand work across waves through global memory are exactly the three that carry `barrier<global>`,
and the quantized-matmul partials that look alarming at first glance turn out to be a two-dispatch reduction
whose ordering the host side owns.

Limits of this audit, stated so it is not over-read: the LDS/global split is inferred from naming (`_stage`, and
`_noalias`/`_na` for parameters) plus the template parameter lists, not from the memory space itself, so a few
rows (`%scratch`, `%data`, `%q_shared`/`%s_shared`, `%expert_table`) are "unclassified" rather than proven LDS;
the audit covers intra-dispatch handoffs only, by construction; and it is static - it says nothing about whether
the host-side recorder actually records every producer/consumer pair it should, which is the one ordering
question still standing after section 32.

## 34. The standalone FA-race harness: the pin is race-free at every recorded shape

Unblocking by REA is what found this: the Oct 5 working set in `~/.cache/kvnan/` contains a purpose-built
standalone reproducer the earlier investigation left behind.

* `fa-race.cpp` / `fa-race` - builds one `FLASH_ATTN_EXT` (D=DV=128, Hq=16, Hkv=8, F32 precision) on HRX0, runs
  it `reps` times **with identical inputs**, and reports every run whose output differs from run 0, grouped by
  `(query tile, head, 16-channel block)`. Usage `fa-race <T> <cells> <visible> <reps> [seed] [device]`; a bad
  run prints `run 14: 256 differ (112 nonfinite): [qtile 0 head 9 ch 96..111: 256]`, the summary is
  `T=... cells=... visible=... reps=... bad_runs=N`.
* the Oct 5 logs (`fa-race1.log`, `fa-race2.log`, `bis1-4.log`, `h336.log`) show it reproducing a real race:
  `bad_runs` 2-11 per 250 at `T=190 cells=4864 visible=4798`, with 64-144 of the differing values **non-finite**,
  and 0 at `T=192`, `T=512`, `T=64` and at `cells=4608`. So the trigger was a specific shape, and `T=190` is not
  a multiple of the 16-token query tile.
* `h336.log` is an A/B the earlier session read as a fix: four rounds of 250 reps with the h336 libraries gave
  `bad_runs=0 0 0 0` where the control gave `4 11 4 2`, at an identical checksum (`1.190266`); `e2e.log` adds
  24/24 clean model-level runs.

**What the pin does at the same shapes, measured today** (harness run against `~/wt/315fp-build`, the
`e44c9d01a` tree):

| T | cells | visible | reps | checksum | bad_runs |
|---|---|---|---|---|---|
| 190 | 4864 | 4798 | 250 | **1.190266** | **0** |
| 192 | 4864 | 4798 | 250 | -5.256695 | 0 |
| 1 / 16 | 4864 | 4798 | 250 | -2.715973 / 14.680580 | 0 / 0 |
| 1 / 2 / 8 / 16 / 190 | 2048 | 810 | 250 | -7.353318 / -7.709488 / 35.336398 / 41.561140 / 146.350555 | 0 |

2250 reps across nine shapes - including the one that raced in October and the regime's own (`cells=2048`,
`visible=810`) - and **not one bad run**. The checksum at `T=190 cells=4864 visible=4798` is exactly the October
value (`1.190266`), so the same kernel and the same path are exercised; it is simply deterministic now. The race
was real, shape-specific, and closed by the time of the pin.

Two cautions this adds to the artifact set:

* **Do not read the Oct 5 `b*/c*/e*/s*` tag matrix as a bisection.** Extracting and disassembling one cache entry
  per tag (`/tmp/fa_variant_profile.py`, using the validated `llvm19` objdump) shows `b0..b3`, `c0..c3`, `e0..e3`,
  `s0..s2` and `jit190`/`jit190c`/`jit190d` are the **same compiled kernel** instruction for instruction (12
  `s_barrier`, 86 `s_waitcnt`, 3 `s_waitcnt_vscnt`, 4 `global_store`, 164 `global_load`), and those logs' own
  `md5sum jit190*/*.bin` lines agree. Their very different `bad_runs` therefore came from outside the kernel -
  environment or which libraries happened to be loaded - not from code. Only `jit190w` (292 `s_waitcnt`),
  `jith336` (76 `s_waitcnt`, 1 `s_waitcnt_vscnt`), `jitmf` and the `T=192` build differ at all.
* **The harness works and is cheap**: a second or two per 250 reps, no regime, no model, a few MB of GTT. It
  links only `libggml`/`libggml-base` and loads the HRX backend at runtime, so any build can be pointed at by
  setting `LD_LIBRARY_PATH` to that build's `bin` plus its `hrx/src/ggml-hrx-deps-build/libhrx/src/libhrx` and
  `.../loom/binding/c`, with `IREE_HAL_AMDGPU_LIBHSA_PATH` set and a fresh `GGML_HRX_JIT_CACHE_DIR`. Its RUNPATH
  still names the October build directory, which no longer exists - hence
  `libhrx.so.0: cannot open shared object file` unless that environment is supplied.

**This closes the FA-race lead for the pin**, as sections 26, 27 and 33 closed the others: what raced in October
does not race now, and no shape I can construct does. What remains is not a kernel property - it is the regime
itself (real memory pressure and the preemptions it causes) or the full GLM graph, which is exactly what the
in-regime arms exercise and what standalone harnesses cannot reproduce by construction.

## 35. The transient-reuse guard: the last live code candidate, and what would settle it

The pin contains `ggml/src/ggml-hrx/dispatch/transient-reuse-guard.{h,cpp}` and it is **on by default**. Its own
header explains the design: the stock packer hands a freed arena range to the next value that fits, "often the
value written by the very next command", which the recorder turns into a write-after-read dependency and an
execution barrier. The guard reuses a freed range "only for a value whose first command already depends, through
the program's own data dependencies, on every command that used the range before" - that is, it deliberately
**suppresses** the ordering that would otherwise keep a reader away from a range still in use.

That makes it the last live code candidate, for the right reasons: it is recent (present in the Oct 5 baseline,
absent from the older `loom-kvnan-h336src`), default-on, it removes ordering, and a wrong suppression produces
exactly the two faces - stale-but-finite bytes are wrong output (Face B), non-finite bytes are NaN logits
(Face A) - intermittently, with a preemption-sized window.

**It is live in the faulting configuration, measured.** With `GGML_HRX_LOG_TRANSIENT_REUSE=1`, a 0.6B decode run
logs:

```
ggml-hrx transient reuse: 312 commands, 478 values, arena 331776 bytes guarded, 331776 unguarded
ggml-hrx transient reuse: 396 commands, 478 values, arena 421888 bytes guarded, 421888 unguarded
ggml-hrx transient reuse: 368 commands, 506 values, arena 602112 bytes guarded, 194816 unguarded
```

So the guard is the packer that runs for decode programs; in the third program it *inflates* the arena threefold
because it declines reuses the stock packer would take (its fallback thresholds are 16 MiB unguarded, or guarded
> 2x unguarded + 8 MiB; neither trips at these sizes).

**What already cuts against it:** the record's `GGML_HRX_TRANSIENT_REUSE=legacy` study - 15 arms, "0 NaN in 15
control vs 1 in 14 legacy", then a 7-arm paired confirmation scored by completion with 0 NaN - says the fault
appears with the stock packer too, so the guard is not necessary for it. That is weak evidence in both
directions: n=14 at the recorded ~5% rate, and the study exercised the knob rather than the guard's dependency
test.

The discriminator is a matched A/B at one box state, both sides at >=60 arms, with
`GGML_HRX_LOG_TRANSIENT_REUSE=1` to record which packer actually ran - i.e. a third cell in the task-5 campaign
rather than a separate experiment. **I tried the cheap version today and it is inconclusive:** three runs of 180
requests with the guard on at `align=256 + compaction` - the only configuration that has ever fired on this
harness - gave 1 distinct output each, so the event did not fire at all; the three `legacy` runs then failed to
launch because of my own quoting bug (`$EXTRA command` instead of `env $EXTRA command`). The event count is now
**1 in 8** runs of the configuration that produced it. That is recorded rather than re-run until it fires,
because comparing a rare event across six runs is not evidence either way.

## 36. Side quest: three more closes and one open edge

With the regime blocked, the side quest was to attack the fault's own shapes and kernel family with the cheap
standalone methods, and to finish the reads left open in sections 32-35.

**1. The FA path is clean at GLM's own geometry.** `fa-race` used Hq=16/Hkv=8; the recompiled sweep used
Hq=32/Hkv=4 (taken from `~/wt/lt/cfg`). **That label is wrong and section 38 corrects it** - GLM-4.7-Flash is MLA
(`deepseek2`): 20 query heads against a single latent, 256-wide MLA key/value, 576-wide cache entries. So this
sweep is evidence about the Qwen-shaped geometry, not about GLM's; it is still 2000 clean reps, just not at the
shape it claims. The sweep's numbers:

| T | cells | visible | reps | checksum | bad_runs |
|---|---|---|---|---|---|
| 1 / 16 / 190 | 2048 | 810 | 250 | 13.916653 / 55.761970 / 1000.327255 | 0 / 0 / 0 |
| 1 / 16 | 2048 | 2048 | 250 | 16.731105 / 74.238626 | 0 / 0 |
| 190 | 2048 | 2000 | 250 | -231.562139 | 0 |
| 1 / 190 | 4864 | 4798 | 250 | 11.450974 / -125.324899 | 0 / 0 |

2000 reps, eight shapes, zero bad runs - and `T=1`/`T=16` at `cells=2048` is the first-decode shape the arms
actually fault in. Combined with section 34 that is 4250 clean reps across both geometries.

**2. The masked-row result holds under repetition.** The section 26 suite was single-shot per case; race hunting
needs repetition. Fifty consecutive runs of the extended `test-hrx-fa-masked-v` (40 cases x 8 masked-row
variants, including NaN/inf tails, at `cells=2048` and `visible` 10/172/810) gave **0 failing runs and 0
non-finite outputs** - 2000 case-executions with garbage in the masked rows, all deterministic and finite.

**3. `moe_routing_tables.loom` resolves to safe**, closing the one "unclassified" row from section 33's audit.
`@ggml_moe_build_expert_table` puts **one workgroup per expert** (`kernel.workgroup.id<x>`), and that workgroup
builds its expert's assignment rows alone, carrying its scan prefix across the (token, route) loop iterations
(`kernel.workgroup.scan<addi>` + `reduce` + loop-carried `matched_base`). All stores into `expert_table` come
from the single workgroup that owns the row, and the reader is a **second dispatch**, so there is no
intra-dispatch handoff to fence and no wave can see a partial row. Ordering between the two dispatches is the
host recorder's, as in section 33's quantized-matmul partials.

**4. The transient-reuse guard's algorithm reads as structurally sound - the open edge is its input.** The pin's
`transient-reuse-guard.cpp` builds, per command, the set of commands it depends on, keyed by
`(binding.origin, value.value)`: a read takes the last writer (RAW), a write takes the readers (WAR), and the
rows are closed transitively into bitsets (`precedes(before, command)`). The reuse decision then requires the
candidate's first command to follow every previous user of the freed range. Two things that would break it are
not in this algorithm but upstream of it: whether `Command::bindings` is **complete** for every command (an
indirect dispatch's arguments, a host-staging buffer, a completion counter - anything the command touches that
is not listed as a binding yields no edge, and the guard would then vouch for a reuse whose ordering the
recorder skips), and whether program order plus the recorder's barriers really is execution order for every pair
the bitset says it is. So the guard stays the last live candidate, but the place to look next is the recorder's
binding construction, not the packer.

## 37. A concrete defect in the access the host is told: `exact_output` is read but declared write-only

Following section 36's "look at what the host is told", the corpus manifests were audited against the kernels
they describe. `kernels/<family>/manifest.json` gives every export a parallel `bindings` / `binding_access` list,
and that access is what the host consumes - `command-program.cpp` copies it onto the command
(`command_binding.access = definition->bindings[binding_index].access`) and `transient-reuse-guard.cpp` reasons
from it (`writes(binding.access)`, `reads(binding.access)`) to decide which ranges may be reused.

Method (tools/audit-binding-access.py, run over all four manifests): resolve each .loom function's
`buffer.view` / `buffer.assume.*` chains **positionally** to their root parameters - the positional part matters,
see the correction below - then check every declared binding against whether that function stores to it, loads
from it, or both. Reported as dangerous: `read` declared for a binding the body stores, or `write` declared for a
binding the body loads, because both leave the host without an edge it needs.

**Result: 2 findings out of 522 bindings.** The second is the one that matters:

| export | binding | declared | body |
|---|---|---|---|
| `ggml_mul_mat_q6_k_packed_selected_refine_token1` | `exact_output` | **`write`** | **loads** (`%value = view.load %exact_view[%tid]`) |
| `ggml_quantize_f32_symmetric_i4_k64_plane` | `scales` | `write` | loads |

**What the kernel does.** `mul_mat_q6_k_packed_f16_wmma.loom:1185` is the single-token *selected refine* wrapper: it
applies the inner `selected_refine.body` template, then after a workgroup barrier each publishing lane loads the
exact value it computed and scatters it into the output row of the candidate it came from:

```loom
%candidates_noalias, %exact_noalias, %output_noalias = buffer.assume.noalias %candidates, %exact_output, %output
%exact_view = buffer.view %exact_noalias[%base] : buffer -> view<64xf32>
...
%value = view.load %exact_view[%tid] : view<64xf32> -> f32
view.store %value, %output_view[%candidate] : f32, view<[%bounded_output_size]xf32>
```

So `exact_output` is written by the inner template **and read by the wrapper**, while the manifest tells the host
it is write-only. The registration is real and on the decode path: `dispatch-mul-mat.cpp:78` binds this export as
`kMulMatQ6KPackedSelectedRefineToken1Kernel` and `:376`'s `match_q6_k_token1_shortlist_dispatch` runs it as part of
`"common.mul_mat.q6_k_token1_shortlist"` - the big-vocabulary head path (approximate shortlist, then exact logits
for the selected candidates), which is the shape a 154,880-token vocabulary needs.

**Consequence, stated at the strength the reading supports.** With `exact_output` declared write-only, the
reuse guard's access map records only `last_writer` for that key and *clears* its readers. A range handed to a
later value therefore looks free of the read this kernel actually performs, so the guard can approve a reuse
whose ordering the recorder then omits, and a later writer can run while this lane's load is in flight. The read
then returns whatever the range held - stale-but-finite bytes scatter a wrong logit, non-finite bytes scatter a
NaN. Under preemption the window that matters is the load's own latency, which is the same memory-pressure
sensitivity the issue describes.

**What this does not explain, and I am not claiming it does.** Section 23's decisive arm had the *final norm
output* at 2048 of 2048 NaN, so in that fault the head's input was already NaN: a defect that corrupts a few
scattered candidate logits cannot be the producer of *that* face. What this finding fits is the **Face B**
profile - wrong-but-finite data becoming text the server rejects - and it is a *code-level* defect rather than
another correlation, on the decode path, with a one-word fix.

**The fix and its discriminator.** The fix is a single manifest word: `exact_output` should be `read_write`, and
`scales` in the quantize export likewise (or its load should be shown to be vestigial). That is a fork-side
change, so it lands as a new commit in the fork and then a forward-only engine pin move - exactly the deliverable
shape the objective asks for. The discriminator is cheap and does not need the regime: re-run the 0.6B + compaction
probe (or the arm campaign) before and after, and re-run this audit to confirm zero under-declarations. What has
to be shown before the mechanism claim is complete is that the reuse actually happens for this range in a faulting
run - which the `GGML_HRX_LOG_TRANSIENT_REUSE=1` numbers from section 35 can be made to answer, since they already
report which packer and which arena sizes a program used.

**The open question that decides whether this matters: does GLM take the shortlist path at all?** The matcher gates
it hard (`dispatch-mul-mat.cpp:376`): weight type `Q6_K`, `token_count == 1`, `input_size % 256 == 0` and
`input_size <= 8192`, **`output_size >= 65536`** and `output_size >= 16 * input_size`, the projection output has no
consumers in the graph index, and the input has an `has_embedded_external_concat_projection_ancestor`. GLM-4.7-Flash
satisfies the size conditions trivially (154,880 outputs against a 2,048-wide input, Q6_K head in Q4_K_M,
single-token decode), which is presumably why the path exists - but the last two conditions are about graph shape,
not size, and **the 0.6B dispatch logs I have do not show `common.mul_mat.q6_k_token1_shortlist` at all** (they show
`q6_k_token1_final_projection_q8`, `q6_k_aligned_skinny`, `q6_k_i8_prepacked` and `q6_k_prefill_wave32`). So the
claim that GLM runs this kernel is a plausible inference from the gates, not a measurement, and settling it needs
one GLM server started with `GGML_HRX_LOG_DISPATCH=1` - which needs the box's GTT, i.e. the same blocker as the
arms. Recorded here so the first thing done after the box frees is a dispatch log, not another hypothesis.

**Resolved, same day: GLM never dispatches that path, so this is not the mechanism.** The box freed (GTT 26.5 GiB,
lock free), so within minutes of writing the paragraph above I ran exactly that server and captured the
inventory - one GLM-4.7-Flash instance at the pin with `GGML_HRX_LOG_DISPATCH=1`, two short completions,
teardown. Its q6_k dispatches are `q6_k_aligned_skinny`, `q6_k_i8_prepacked`, `q6_k_prefill_wave32` and
`q6_k_token1_final_projection_q8`; **`q6_k_token1_shortlist` does not appear**, so
`ggml_mul_mat_q6_k_packed_selected_refine_token1` is not on this model's decode path. The full 35-name
inventory is `315-campaign/evidence/315-glm-dispatch-inventory.md`.

So section 37 stands as a *defect* - a binding the host is told is write-only while the kernel reads it - but it
is **closed as not-applicable to engine#315**: the kernel it affects is never dispatched for the failing model.
The one-word manifest fix would still be correct on its own merits, but it is no longer part of this issue's fix,
and it is not a candidate mechanism. This is the second time a plausible static finding has been killed by one
cheap measurement, which is the right ratio: the audit cost minutes and the dispatch log cost minutes, and both
were written down before they were believed.

**Correction to this audit's first version**, recorded because the first numbers looked far more dramatic: it
found 93 under-declarations, then 42 after restricting to per-function bodies. Both were artifacts of parsing
`buffer.assume.noalias` lines. Those lines are **positional multi-assignments**
(`%a, %b, %c = buffer.assume.noalias %x, %y, %z`), and a single-pair regex pairs the last left-hand name with
the first right-hand name - mapping `%out_na` to `%weight` instead of `%output` and inventing writes into
read-only weights. Pairing the lists positionally drops the count to the 2 above, both of which were then
checked by hand against the kernel body. The lesson is the usual one: a static audit that finds dozens of
defects in widely exercised code is more likely to be wrong than the code.

## 38. GLM's real attention geometry is MLA - and that invalidates the "GLM geometry" label on my sweeps

While the arms ran I read the model's own metadata instead of trusting a config file, and it matters:

| key | value |
|---|---|
| arch | `deepseek2` (**MLA**, multi-head latent attention) |
| `attention.head_count` / `head_count_kv` | **20** / **1** |
| `attention.key_length` | 576 (= `kv_lora_rank` 512 + rope 64) |
| `attention.key_length_mla` / `value_length_mla` | 256 / 256 |
| `attention.q_lora_rank` / `value_length` | 768 / 512 |
| `rope.dimension_count` | 64 |
| `expert_count` / `expert_used_count` / `expert_feed_forward_length` | 64 / 4 / 1536 (1 dense leading block, 47 blocks) |
| `vocab_size` | 154,880 |

**So sections 34 and 36 are mislabelled.** `fa-race` (Hq=16/Hkv=8) and my GLM-geometry recompilation (Hq=32/Hkv=4,
taken from `~/wt/lt/cfg`) are both **not** GLM's attention shape: GLM has 20 query heads against a single MLA
latent, with 256-wide per-head MLA key/value and a 576-wide cache entry. The sweeps' clean results stand as
results about *those* shapes; they are not evidence about GLM's, and the earlier session's `lt/cfg` was for a
different model. Section 27's kernel suite is unaffected in the same way: `test-hrx-fa-masked-v` uses Hq=16/Hkv=8
with head 128, so its 27 cases (and my 40) are not MLA either.

That re-opens a cheap, well-defined probe rather than a vague one: an **MLA-shaped** attention race hunt. The
shape is now known from the model itself, the harness pattern exists (`fa-race.cpp`: build one `FLASH_ATTN_EXT`,
run it N times with identical inputs, compare bit-for-bit), and the FA dispatch for this model at the pin is
`common.flash_attention_f32_f16_wmma` - whose source carries an explicit MLA branch ("MLA (deepseek2/GLM) stores
V with the QK ... such a V keeps the row-major layout"). Nothing I have run to date exercises that branch, which
makes it the most obviously untested code on the failing model's path.

It also puts the `test-hrx-fa-masked-v` family's value in perspective: those tests are good evidence about the
non-MLA kernels, and they say nothing about the branch GLM actually takes.

**What GLM's decode attention actually is, from the fork's own graph code.** `llama-graph.cpp:2442` calls
`ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, ...)` on the **latent** tensors (the 576-wide key = 512
`kv_lora_rank` + 64 rope, the 512-wide value), and immediately afterwards (`:2449-2454`) does the MLA de-absorption:

```cpp
if (v_mla) {
    cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
    cur = ggml_mul_mat(ctx0, v_mla, cur);      // broadcast across n_tokens
}
```

So GLM's decode attention is two steps, and **neither is covered by anything run in this investigation**: a flash
attention over a 576/512 latent with 20 query heads and one latent "KV head", and then a small `v_mla` matmul that
projects the 512-wide latent attention output into the per-head value space. `fa-race` (Hq 16/Hkv 8, D=DV=128) and
the masked-V suite (Hq 16/Hkv 8, head 128) exercise neither shape, which is why their clean results - while
they stand - cannot be read as "GLM's attention is clean".

The probe that follows from this is concrete, and the harness pattern exists: adapt `fa-race.cpp` to the MLA
tensors (k 576-wide, v 512-wide, `n_head` 20, one latent KV head, `cells`/`visible` at the regime's values,
`-c 2048` and up), add the `reshape_4d` + `v_mla` `mul_mat` step, and compare bit-for-bit across reps. The cheaper
half of it is extending `test-hrx-fa-masked-v`'s `Case` constants to the MLA dims (head 256/512, Hq 20, Hkv 1),
which would at least drag the MLA branch of `flash_attention_f32_f16_wmma` into a test that already exists and
already checks masked-row insensitivity.

**The harness is written and compiled, waiting for the GPU.** `/tmp/glm-mla-race.cpp` on strixhalo (also kept at
`315-campaign/tools/glm-mla-race.cpp`) builds the MLA decode graph exactly as the fork does - `flash_attn_ext` on
k 576 / v 512 with 20 query heads and one latent KV head, then `reshape_4d(cur, 512, 1, 20, T)` and
`mul_mat(v_mla, cur)` - runs it `reps` times with identical inputs and reports every run that differs, grouped by
`(token, head, 16-channel block)`. Usage: `glm-mla-race <T> <cells> <visible> <reps> [seed] [device]`, built with
the same one-line `g++` against `~/wt/llama-315copy/ggml/include` and `~/wt/315fp-build/bin`. It has **not been
run**: the in-regime instrumented arm hunt owns the box, and a second job would perturb exactly the queue and
preemption conditions the regime gate depends on.

**What the registration does with GLM's layout, for whoever runs it.** `dispatch-flash-attention.cpp`'s
`prepare_flash_attention_value` decides between two V layouts: it transposes V into a
`common.flash_attention.transposed_value` transient (dispatching `ggml_copy_transpose_f16`) only when V's rows are
contiguous (`nb[1] == columns * 2`) **and** `query_token_count >= 256` and `key_value_token_count >= 512` and
several extent/alignment conditions hold. MLA stores V with the QK head-size stride, so its rows are wider than
`columns` and GLM **always keeps the row-major layout** - no transpose, no transient, the kernel takes the real row
stride. The file also names the constraint that makes 576 legal: `576 = DeepSeek-V2/V3 / GLM-4.7-Flash MLA
key_length; 592 is the LDS ceiling`. Two consequences for the probe: at the regime's shapes (`cells=2048`,
`visible~810`, 1-190 query rows) GLM is on the row-major MLA path, and for single-token decode with a long KV the
dispatcher moves to the *decode-split* kernel - whose MLA instantiation, like the row-major MLA branch itself, has
never been exercised by any test in this investigation (my masked-V suite runs the non-MLA Hq16/Hkv8 shape, and the
short-prompt dispatch inventory did not reach the KV threshold where the decode-split kernel engages).

## 39. The instrumented arm hunt: a stage gets a name, and the instrument turns out to be an inducer

The nan-dump fingerprint was put in front of a faulting arm (17 scored pairs and one control before the run's own
8400 s timeout; `docs/315-campaign/evidence/315-instrumented-arm-hunt.md` is the record). Two results, in order of
importance.

**The instrument moved the fault rate.** 16 of 17 fix arms faulted and the single control arm faulted too - 94%
against the recorded 5.4%, Fisher two-sided p = 1.2e-10, every fault Face A, every arm inside the gate at
`gtt_start=98G`, no Face-B event in the whole run. Both candidate instruments are host-side interventions on exactly
the buffers this defect concerns (`GGML_HRX_SENTINEL` fills the logits allocation with a NaN payload;
`GGML_HRX_TENSOR_FP=tail:16` adds 16 device-to-host reads per `graph_compute`), and the control build does not
implement the fingerprint yet faulted with it set. So this run **measures the instrument, not the regime**. That is
also the run's most useful by-product: a ~94% trigger is a reproducer, and bisecting the env set (fingerprint off,
sentinel off, both off) is now the cheapest way to learn what actually gates this fault. Nothing above the bisection
is a mechanism, and section 29's elimination table is unchanged by this run.

**The stage that does have a name.** In the first NaN-bearing call, one token's `n_embd` row is NaN from the last
block's FFN outward: `ffn_shexp-41` (MUL_MAT, shared expert) 2048 of 352256, `ffn_swiglu-41` (GLU) 2048, the three
`ADD`s that follow 1984 each, and the final `norm-42` 2048. The next call's `FLASH_ATTN_EXT` carries it forward as a
consumer. This agrees with section 23 - producer upstream of the final norm, LM head exonerated - and narrows the
producer to the FFN path. The `tail:16` window begins after `ffn_norm-41`, so it cannot yet separate "the FFN's input
row was already NaN" from "the FFN matmul fabricated it from clean input": `tail:64`, or a name list over
`ffn_norm-41`/`inp_attn-41`, decides it, and that arm must carry the attribution control from step 1 so it is not
read as a second instrument-induced stage.

Face B produced nothing in this run - 0 of 17 arms - so it is untouched by these instruments, and the cheap probe's
Face-B-flavoured 95/180 (section 36) remains the only positive Face-B observation on the record.

## 40. The frame that failed, the frame that first carried NaN, and the one-bit test between them

Two facts from the faulting arm's own log (`p10l`), which replace the reading I wrote in section 39.

**The graph that raised Face A is `call=51774`, and the fingerprint's last three lines are that graph, in order:**
`norm` (RMS_NORM) -> `result_norm` (MUL, `bytes=8192` = 2048 f32, i.e. a **T=1 decode step**) with 2048 of 2048 NaN ->
`result_output` (MUL_MAT, `bytes=619520` = 154880 logits) with 154880 of 154880 NaN -> two lines later, line 1868,
`common_sampler_sample: HRX returned NaN logits: count=154880 of 154880`. So the visible fault is one decode token
whose final norm is *entirely* NaN: the hidden state handed to it was already NaN.

**The first NaN-bearing segment is `call=51764`, a prefill-shaped graph** (tensors of 1409024 bytes = 352256 f32 =
~86-172 tokens at this model's 2048-wide rows), and its window covers **block 41's FFN and block 42's attention**:
`ffn_shexp-41` (MUL_MAT, 2048), `ffn_swiglu-41` (GLU, 2048), `ffn_moe_out-41` (ADD, 1984), `ffn_out-41` and
`l_out-41` (ADD, 1984), then `norm-42`, `q-42`/`q_nope-42`/`q_nope_perm-42` (5120/5056). Every earlier call's window
was clean, so the corruption is *born* at block 41, not inherited from block 0. The model is also **not** 42 layers:
`norm-42` is block 42's own norm and `q-42` its 20x512 MLA query, while the output norm is the unnumbered `norm`
above. Block 41 of ~47 is where this frame starts.

**Why the shape matters.** A prefill whose *sampled* row is clean can still carry a NaN row inside it - the prefill
samples its last token, so a corrupted row elsewhere in the window costs nothing at that point. That row is written
to its token's KV slot, and the next decode that attends to it hands the output norm an all-NaN hidden state. That is
the timing the fault has shown all along (section 23: the first decoded token after `slot launch_slot_`), and it is a
shape both faces can share, with Face B the milder version in which only part of a decode's logits go NaN and the
sampler emits invalid UTF-8.

**The chain, from the pin's own builder** (`src/models/deepseek.cpp:130-176`, GLM is the deepseek2 builder):

    ffn_inp-<il>  = ADD(attn_out, inpSA)      <- the block's residual input
    ffn_norm-<il> = RMS_NORM(ffn_inp)
    ffn_moe_out-<il> = MoE(build_moe_ffn(ffn_norm))      ffn_shexp-<il> = shared expert(ffn_norm)
    ffn_out-<il>  = ADD(ffn_moe_out, ffn_shexp)
    l_out-<il>    = ADD(ffn_out, ffn_inp)

so a name list settles causality in one arm: if `l_out-40`, `ffn_inp-41` and `ffn_norm-41` carry NaN, the block's
input was already poisoned (its attention branch, or further back); if they are clean while `ffn_shexp-41`,
`ffn_swiglu-41` and `ffn_moe_out-41` are not, the FFN branch fabricated it. `GGML_HRX_TENSOR_FP` accepts a
comma-joined name substring list as well as `tail:<N>`, so this is an env change, not a rebuild:

    GGML_HRX_TENSOR_FP=l_out-40,ffn_inp-41,ffn_norm-41,ffn_shexp-41,ffn_swiglu-41,ffn_moe_out-41,ffn_out-41,l_out-41,
                       attn_norm-42,ffn_inp-42,ffn_norm-42

**The caveat that must not be skipped.** Those counts are not row-aligned across tensors (2048 inside a tensor whose
row is 1536; 1984 = 2048-64; 5056 = 5120-64), and 2048 floats = 8192 bytes is exactly the extent of the NaN payload
the sentinel instrument writes. So "which node reads NaN" may be tracking where that payload was *read* rather than
where the defect is born. That is the question the attribution session is on the box answering right now, and it is
why the chain probe above is written down but not yet run.

## 41. Correction: neither instrument can fabricate the NaN the fingerprint found

Section 40 closed with a caveat that the stage list might be tracking an instrument's payload rather than the
defect. That caveat is falsified, from the instrument source itself (`~/wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp`,
the tree `~/wt/315fp-build` was configured against - `CMakeCache.txt` `ggml_SOURCE_DIR`):

- the copy probe (lines 181-199) is **read-only**: it writes two lines of `fprintf` per site and checks
  `size == 619520`. It never touches a buffer's contents.
- the sentinel (lines 380-391) writes its `0x7FC0DEAD` payload only when
  `host_visible && base != nullptr && size == 619520 && environment_flag_enabled("GGML_HRX_SENTINEL")`. Its own
  comment says "Only the 619520-byte (154880 x f32) logits allocation is touched ... so a control arm without it is
  unaffected." That is the logits allocation alone - not `ffn_shexp-41` (1409024 bytes), not `ffn_swiglu-41`
  (1056768), not `result_norm` (8192).

So every NaN count the fingerprint reported for a non-logits tensor is a **computed** NaN, and the logits' own NaN is
computed too: the fault line's `sentinel=0` says no sentinel word survived there, i.e. the LM head overwrote the
region and wrote NaNs into it. Section 40's stage list stands as evidence; what the instruments cannot be exonerated
from is *timing*. `GGML_HRX_TENSOR_FP=tail:16` issues 16 device-to-host reads per `graph_compute` - each one an extra
synchronisation on the same stream as the real logits read - and the copy probe issues unbuffered `fprintf`s from the
submit path. That is a plausible way to move a 5.4% window to 94% **without fabricating anything**, and it is what
the attribution session on the box is measuring. A rate that is high *under instrumentation* is therefore a
reproducer rather than a disqualification, and the chain probe in section 40 can run on its own merits: no instrument
can fake which node first carries a computed NaN.

## 42. The structural relationship between the two faces, from all 142 recorded arms

`docs/315-campaign/evidence/315-face-landscape.md` reads every arm log on the box and changes what a unifying
mechanism has to look like.

**The faces never co-occur: 23 Face-A arms, 7 Face-B arms, 0 arms with both, 108 fully clean.** Whatever the
defect is, in 142 draws the two observables are mutually exclusive. Any claim that they are "the same event seen
by two builds" has to survive that; the guard's own switch (`GGML_HRX_NAN_CONTINUE`) is one way to test it,
which is what `315-faceb.sh` does.

**Face A travels with finite nondeterminism, Face B does not.** `identical_distinct` is 1 in 135 arms and >1 in
exactly 7 - and all 7 are Face-A arms. So a Face-A arm contains both NaN logits *and* different finite outputs
for identical greedy requests, i.e. nondeterministic corruption rather than a wrong answer. The Face-B arms are
not divergent, and their `idn=1` covers the 6-7 identical requests that completed before they aborted, so Face
B's generation was consistent up to the failure.

**Face B shows no dependence on the gate** (7/113 in-band vs 0/29 out; Fisher p ~ 0.3), and what it certainly
is, is the server's own chat-template parse failing - "Content-only" is a parser name (`common/chat.cpp:812`).
That makes Face B a *text*-level failure, and it makes the open question precise: was that text produced from
NaN logits (in which case the two faces are one mechanism separated by the guard's abort-vs-continue choice), or
from finite-but-corrupted generation (in which case Face B needs its own mechanism, downstream of the LM head)?

One caveat is recorded with the landscape rather than quietly dropped: the single out-of-band Face-A event
(`315nan-p20l`, gtt=44) happened *after* I SIGKILLed another arm's server in that session, so it is ambiguous and
is not evidence that Face A occurs outside the regime.

## 43. Two constraints on the work that remains

**Mechanics of a forward-only fix.** `tools/check_pins.py <base>` takes every submodule whose gitlink differs
between `<base>` and HEAD and asks GitHub's compare API whether the new commit is *ahead* of the old one; a pin
that is behind or has diverged fails. So the fork-side commit has to descend from the engine's current pin
(`third_party/llama.cpp = e44c9d01a4d5110cecaf45472a717f46cd1abbf9`), not be rebased onto a different base - a
fix prepared on top of main's shipped pin (`26330cc20`) would be rejected as sideways. The engine-side change is
then the gitlink plus `registry/architectures.json`, and the CI baseline already recorded the gap that nothing
ties that registry pin to the gitlink, so both have to be moved by hand and stay consistent.

**The box budget does not fit the campaign.** The `315nan` session ran 06:08:04 to 09:37:17 for 22 arms (21
scored plus the control), i.e. **9.5 minutes per arm**, dominated by loading GLM into the in-regime band and by
the pre-registered 66-request workload. A 60-arm campaign is therefore ~9.5 hours of box time by itself. Roughly
6 hours of the 12-hour budget are already spent and the two queued probes add ~2.3, which leaves ~3 - so the
acceptance campaign cannot be run inside the declared budget even with the mechanism and the fix in hand. This is
a constraint on the objective, not a finding about the defect, and it needs a decision from the user: extend the
box budget, or accept that the campaign is delivered outside it. The per-arm cost is intrinsic to the registered
workload (the 8x96-token identical block plus 50 mixed requests at GTT pressure), so shortening it would change
the measurement rather than make acceptance harder, which the freeze rules forbid.

## 44. Face B is not a second defect: it is the same corruption seen after the parser

The line immediately before all seven Face-B 500s is `common_chat_peg_parse: unparsed Content-only output:`
followed by the model's own text, and that text is degenerate in two flavours: symbol soup with invalid bytes
(`a..........* it on ${... and: it itns it that'~><es x case a a~ it`), and a 12-character fragment
(`We A ( the).`). In `315accept-p1c` the offender is inside the identical block - request 13 of 16 - while
`identical_distinct` was 1 over the requests that succeeded, so the same prompt produced normal output and then
this fragment. The event is anomalous, not a property of the prompt, which rules out "the model legitimately
failed a strict format" for that arm.

**So the two faces are one mechanism with two observables.** The guard that raises Face A is unconditional and
silent when `nan_count == 0`, and no Face-B arm logged a guard line: those requests' logits contained no NaN.
Sampling degenerately from NaN-free logits is what a *finite* corruption looks like - soup, or a premature stop -
and it reaches the sampler, produces text, and fails later, at the parser. The observable therefore depends on
what the corrupted memory happened to hold: NaNs (Face A, caught at the sampler) or stale finite values (Face B,
caught at the parser). Section 40's stage - block 41's FFN during a prefill - is the *producer* for both.

**Consequence for the acceptor.** A rule that only counts NaN and server failures cannot see finite corruption
that still parses, so the fault rule now includes `identical_distinct > 1` (the 8 identical greedy requests
disagreeing). On the recorded population that never fires without another signal: 135 of 142 arms are
deterministic, and the 7 exceptions are exactly the Face-A arms. Freeze note of 2026-10-10 in
`315-preregistration.md` section 8; analyser and its self-test updated (`tools/test-analyse.py`, 13/13).

**What it does not yet settle.** No Face-B arm had the logits fingerprint on, so an offending request's logits
have never been fingered. `315-faceb.sh` turns it on and can test the strong form: the eight identical greedy
requests must log the same logits hash, so a corrupted sample has to appear as exactly one hash differing from
its seven siblings. Until that is read, "Face B is corrupted logits" rests on the degenerate text plus the
absence of NaN, not on a measurement of the logits themselves.

## 45. Correction: the visibility/ordering eliminations lean on an inference that only covers the logits buffer

Sections 21, 22 and 29 dismissed the coherence (§13) and ordering (§15) fixes with the sentence "both address a
missing or late write, and the write is present". That "write is present" came from the fault line's `sentinel=0`.
Section 41 then established what the sentinel actually covers: `GGML_HRX_SENTINEL` writes its payload only when
`size == 619520`, i.e. the logits allocation and nothing else. So `sentinel=0` is evidence about **the LM head's
output buffer only**. It is not evidence about the tensors where the NaN is born - `ffn_shexp-41` is 1409024 bytes,
`ffn_swiglu-41` is 1056768, `result_norm` is 8192, and none of them is ever sentinel-filled. The inference does not
extend to the producer, and sections 21/29 used it as though it did.

What actually survives from those sections:

- **§13's own campaign is untouched by this** and remains the strongest statement available: a cell with the
  coherence bit forced on (`GGML_HRX_USE_UNIFIED_MEMORY=1`, `memory_type` `0x52` -> `0x56`, verified in the trace)
  still faulted, so coherence is **not sufficient**. That is direct experimental evidence, independent of any
  sentinel argument.
- What §13 does not establish, in its own words, is coherence being "cleared as a *contributing* factor".
- The same applies to ordering: "not sufficient" was shown; "not contributing" was inferred, and the inference is
  the part that does not hold.

**Why this matters now, and what it revives.** The producer is block 41's FFN during a prefill (§40), and today's
Face-B evidence is that the *same* event yields stale **finite** values where Face A yields NaN (§44). A reader
observing a buffer's *previous* contents - finite stale data - rather than its intended contents is the signature
of a visibility or ordering failure at that crossing, with the value class decided by whatever the buffer held
before (NaN from a sentinel/uninitialised state, or stale floats). So the visibility/ordering family is a live
*contributing* hypothesis for the producer tensors, constrained by §13 to be insufficient on its own.

**Next discriminator, in order.** (1) The chain probe now on the box: whether block 41's *input* tensors
(`l_out-40`, `ffn_inp-41`, `ffn_norm-41`) already carry NaN or only its FFN outputs do. An input-side NaN puts the
corruption at a boundary *before* the block, which is where a visibility/ordering defect would sit; an output-side
NaN puts it inside the FFN's own compute. (2) Only then a coherence/ordering variant scored with the multi-face
rule including `identical_distinct`, since a NaN-only or 500-only score cannot see the finite half of this defect.

## 46. Face B measured, the coherence bit cleared, and what is left of the visibility family

Three results from 2026-10-10, all in `docs/315-campaign/evidence/315-face-landscape.md`.

**Face B is now measured, not inferred.** The chain-probe session (`315chain`) faulted its first pair in both cells
with a 60-name fingerprint spanning the residual stream, both attention branches, the whole FFN/MoE chain and the
output norm. The fix-cell arm logged `http_failures 1` with the parser rejecting `the}:}:}甚;`, and the control arm
rejected a repetition loop, `a gaps gaps gaps gaps ...`. Both logged **zero** fingerprint lines: no NaN anywhere,
logits included. So Face B is a *finite* corruption, and its text flavours - garbled, or a repetition loop - are what
near-degenerate logits produce under greedy sampling. My NaN-only fingerprint is structurally blind to this half;
localising it needs hashes across identical requests, not NaN counts.

**The coherence bit is cleared as the mechanism.** The `315accept` session tested the *patch*, not the knob:
`315fix-build` (unconditional `HRX_MEMORY_TYPE_HOST_COHERENT`) against `315instr3-build`, and the driver's own
`hv52`/`hv56` columns prove the patch was live in the scored cell (`hv56=1 hv52=0` versus `hv52=1 hv56=0`). It gave
**9 clean and 1 Face-B fault in 10 fix-cell arms** - the same shape as the recorded baseline. Section 13's conclusion
survives its own re-reading with a stronger intervention: coherence is not sufficient.

**What that leaves, and the run now queued to test it.** The logits allocation is host-visible and *not* coherent,
so a host read depends on an invalidate/sync landing after the device's write. If the family is right at all, the
defect is that *ordering*, not the coherence attribute - which is exactly what `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC`
exercises by calling `hrx_stream_synchronize` after every graph replay
(`command-program-executor.cpp:2034`). `315-graphsync.sh` is queued on the box: six in-regime arms, no other
instrument, compared against the no-knob arms already on disk. Zero faults there means a missing or early
consumer-side sync is the mechanism and a scoped variant of it is the minimal forward-only fix; an unchanged rate
means the corruption is computed - a kernel or shape defect - and the visibility family is dead for the producer.

**One record gap to carry forward.** Every `control_bin` on disk is `315instr3-build`, whose tree has 233 insertions
of its own, so the control cell has always been a candidate rather than a baseline. A stock-pin control has never
been measured; building one from a clean worktree at the pin is a prerequisite for the campaign's control cell and
for any "unchanged relative to baseline" claim in task-6.
