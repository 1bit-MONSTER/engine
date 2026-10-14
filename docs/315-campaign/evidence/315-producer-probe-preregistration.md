<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Pre-registration: the producer-side sentinel/payload test (`315prod`) — the one instrument that answers the buffer question

Written **before** the run. This is the experiment the record's surviving lead has asked for since
`315-cause-boundary-final.md`: *"read a value at the point of production"*. The instrument exists, is compiled
into a binary, and **has never been enabled in an arm** — no `[hrx-producer] armed` and no
`[hrx-producer] anomaly` line exists anywhere in the archive.

## Why this is different from every previous attempt

Every earlier attempt read a tensor **from the host, after the graph** — which the record established is
untrustworthy (arena slot reuse, aliased `l_out` reads with a 2-layer stride, and now an instrument-dependent
onset layer). This instrument does not read afterwards: it **writes a known pattern into the region before the
graph runs, then classifies after a drained stream**. It reports, per hooked graph:

    [hrx-producer] armed   op=%s name=%s bytes=%zu words=%zu
    [hrx-producer] anomaly op=%s name=%s words=%zu sentinel=%zu nan=%zu finite=%zu first_nan=0x%08x

`sentinel` counts words still holding the probe's own payload `0x7FC0DEAD`; `nan` counts genuine NaNs (with
the **bit pattern of the first**); `finite` counts written finite values. So it answers the question the engine
cannot: **did the device write the bytes the consumer reads?**

## The instrument, read at source

`~/wt/llama-315probe/ggml/src/ggml-hrx/runtime/graph-executor.cpp`:

- `producer_probe_enabled()` — gate is `GGML_HRX_PRODUCER_PROBE` (non-empty, not `"0"`).
- `producer_probe_target(t)` — a tensor is targeted if it is **host-visible** and its size is in
  **256 KiB … 8 MiB** (the logits region's 619,520 B is inside; the comment records that size, not topology, is
  the right selector because an earlier "not consumed by any node" filter **armed nothing at all**).
- **Only graphs containing a host-visible tensor are hooked**, and each is reported (a once-per-process dump
  only ever saw the first chunk, which contains none).
- `producer_probe_fill(graph)` runs **before** execution: it writes `0x7FC0DEAD` over every target.
- `producer_probe_scan(stream, graph, armed)` runs **after** execution and **only on a drained stream**
  (`hrx_stream_synchronize` first, with an explicit source comment that reading an undrained buffer "would
  report the sentinel as never written and invent the wrong conclusion").
- The payload `0x7FC0DEAD` is deliberately distinguishable from a plain quiet NaN `0x7FC00000`.

**Availability, checked, no build required:** `~/wt/315probe-build/bin/libggml-hrx.so.0.18.0` contains all
eight probe strings, including `GGML_HRX_PRODUCER_PROBE` and the `first_nan=0x%08x` format.

## The run

    BIN=~/wt/315probe-build/bin
    EXTRA_ENV="GGML_HRX_PRODUCER_PROBE=1"
    ~/wt/315-disc.sh 8 315prod

Deliberately minimal: **no other instrumentation**, so the run is not a new induced regime. The probe is the
only addition, and it is the run's whole purpose.

## Readout, fixed in advance

For each arm, from `[hrx-producer]` lines in `server.log`:

| observation | verdict |
|---|---|
| `anomaly … sentinel>0`, `first_nan=0x7FC0DEAD` on a faulting arm | **the device never wrote that region** — the consumer reads pre-fill content. This is the runtime-restore-fidelity gap, observed at the point of production, and it makes the fault a *visibility/staleness* defect rather than a compute defect |
| `anomaly … sentinel=0`, `nan>0`, `first_nan=0x7FC00000` | the region **was** written and the NaN is an ordinary quiet NaN from arithmetic — the fault is upstream of the buffer, and the "never written / restored copy" family is excluded |
| `anomaly … sentinel=0`, `nan>0`, `first_nan` any other value | the region was written, and the payload starts an identification of the producer (compare against `0x7FC0DEAD`, `0x7FC00000`, `0xFFC00000`, `0x7FFFFFFF`) |
| `armed …` present but **no `anomaly` line** in a faulting arm | the region was written cleanly on the hooked graph while the fault came from elsewhere — **the buffer family is excluded for that arm** |
| **no `armed` line at all** | the size/is-host selector armed nothing; **INVALID for this question**, reported as such, and the size band is widened in a follow-up rather than the empty output being read as a clean result |
| no faulting arm in the run | **INVALID for this question**; the run measured nothing about the fault |

**The decisive pair is the first two rows.** They separate "the consumer is reading memory the device never
wrote" from "the device wrote a wrong value", which is precisely the distinction
`315-cause-boundary-final.md` names as what a repair requires and the engine cannot currently make.

## Confounds and limits, declared before the run

- **The probe itself writes to the target before each hooked graph.** That is the measurement, but it means a
  region the op only *partially* writes will show leftover sentinel words without the region being un-visible.
  A non-zero `sentinel` is therefore evidence of *unwritten words*, which must be read together with `words`
  to see whether the gap is the whole region or a tail.
- **A faulting arm must still occur.** The instrument does not induce faults; if the run produces none, the
  run is INVALID (above).
- **`first_nan` is the first NaN in word order**, not necessarily the first one written; it identifies a
  payload, not a site.
- **No rate.** The probe is a per-graph classifier; nothing here is a denominator, and no arm is an acceptance
  arm.
- The recorded sentinel work (`GGML_HRX_SENTINEL`, 20 of 21 arms armed, all 17 fault lines `sentinel=0`) is
  **not contradicted** by this run's premise: that instrument filled the region once and the fault lines were
  the *sampler's* guard, whereas this one fills **before every hooked graph** and classifies **after a drained
  stream**, so it can catch a single graph that fails to write. The two are different instruments with
  different granularity, and a disagreement between them is itself informative.

## Evidence this pre-registration rests on

    grep -naE "hrx-producer" ~/wt/llama-315probe/ggml/src/ggml-hrx/runtime/graph-executor.cpp
    strings ~/wt/315probe-build/bin/libggml-hrx.so.0.18.0 | grep -E "hrx-producer|PRODUCER_PROBE"
    grep -ah "hrx-producer] anomaly\|hrx-producer] armed" ~/wt/rt-det/*/server.log | wc -l   # = 0

---

## Amendment, written BEFORE the run: the probe alone would probably yield no fault

The run above specifies `GGML_HRX_PRODUCER_PROBE=1` alone, on the reasoning that the probe should not be
compounded with other instrumentation. Checking the yields before launching shows that would likely waste the
run:

| config | recorded fault yield |
|---|---|
| ambient (no instrumentation) | ~5 %/arm (`docs/315-mechanism.md`; 2/37 in `315-instrumented-arm-hunt.md`) |
| 15-name `l_out` list (`315size`) | **faulted in 4 of its 9 arms**, all three cells |
| `tail:16` + probes (`315nan`) | 16/17 — but did **not** reproduce in `315row` |

At 8 rounds there are **8 eviction-cell arms**, so at the ambient rate the expected fault count is ~0.4 and the
likely outcome is *no faulting arm* → INVALID. The instrument cannot manufacture the fault it is meant to
classify.

**Amended run**, decided before launching and recorded here rather than after seeing any result:

    BIN=~/wt/315probe-build/bin
    EXTRA_ENV="GGML_HRX_PRODUCER_PROBE=1 \
               GGML_HRX_TENSOR_FP=l_out-0,l_out-1,l_out-2,l_out-3,l_out-4,l_out-10,l_out-15,l_out-20,\
    l_out-25,l_out-30,l_out-35,l_out-40,l_out-45,l_out-46,l_out-47 \
               GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 8 315prod

The 15-name list is the **fault-yield enabler**, not part of the measurement: the probe's target is chosen by
size and host-visibility, so it does not depend on the fingerprint at all. Declared plainly because it is a
confound — the run is therefore *"the name-list regime plus the producer probe"*, not the ambient regime — and
a positive sentinel result from it is a statement about **that** configuration. If a follow-up at the ambient
rate is wanted, it needs far more than 8 arms and is a separate pre-registration.

**Nothing else changes**: the six verdict branches, the decisive sentinel/`first_nan` pair, the declared
confounds, and the INVALID clauses all stand as written above.

### One more thing to check at readout, present in the design

Because the probe hooks **only graphs containing a host-visible tensor**, and `hrx-producer`'s own comment
records that HRX executes the llama graph as ~9-node chunks, the probe will report on the logits chunk and not
on the layer chunks. So a positive result localises the *logits region*, and says nothing directly about
`node_2948`'s buffer — stated here so the result is not over-read.

---

## Second amendment, still BEFORE any result: the table misread the source's counters

Implementing the table as code (so the verdict is computed rather than chosen) caught an error **in the table
itself**. The source loop is:

    if (bits == kProbeSentinel) { ++sentinel; }
    else if (std::isnan(fp[w])) { ++nan; if (first_nan == 0) first_nan = bits; }
    else { ++finite; }

**The probe's own payload is counted in `sentinel` ALONE** — it is the first branch, so a surviving payload
word never increments `nan` and **never sets `first_nan`**. `first_nan` is therefore always the first
*genuine* (non-sentinel) NaN. The table above asserts `sentinel>0` **and** `first_nan = 0x7FC0DEAD` as its
"never written" test, which the source makes **impossible to satisfy**: a surviving sentinel yields
`sentinel>0, nan=0`, and `first_nan` is then unset and prints as `0x00000000`.

Corrected readout, with the source's exact partition `sentinel + nan + finite == words` asserted:

| observation | verdict |
|---|---|
| `sentinel > 0` **and** `nan == 0` | **NEVER_WRITTEN** — payload words survive and there is no genuine NaN: the device did not write this region. Consumer reads pre-fill content ⇒ visibility/staleness defect |
| `sentinel > 0` **and** `nan > 0` | **PARTIAL_UNWRITTEN** — the region was partly written; size the gap against `words` before concluding anything |
| `sentinel == 0`, `nan > 0`, `first_nan = 0x7FC00000` | **WRITTEN_QUIET_NAN** — written, ordinary arithmetic NaN ⇒ the never-written/restored-copy family is excluded |
| `sentinel == 0`, `nan > 0`, `first_nan` any other value | **WRITTEN_OTHER_PAYLOAD** — written; the payload identifies the producer |
| no `anomaly` line, `armed` present | **WRITTEN_CLEAN** |
| no `armed` line | **SELECTOR_ARMED_NOTHING** — INVALID for the question; widen the band, do not read as clean |

`PARTIAL_UNWRITTEN` is a new branch and it is the one the earlier table wrongly collapsed into "never
written": a handful of leftover payload words in an otherwise-written region is **not** evidence that the
region went unwritten, and reading it as such would have manufactured the exact conclusion the experiment is
supposed to test.

**Ordering note, kept from before:** the payload test is no longer a special case at all — the payload never
appears in `first_nan`, so the decisive distinction is simply `nan > 0` versus `nan == 0` given `sentinel > 0`.

The classifier is `docs/315-campaign/tools/producer-probe-readout.py`, self-tested 7/7 including the partition
identity, so the table is exercised rather than asserted.

### Two more readout facts worth having before the arms land

- **`anomaly` requires `sentinel > 0 || nan > 0`.** A hook that fired, wrote the region correctly and produced
  no NaN prints **no anomaly line at all** — so "armed but no anomaly" is a *clean* reading, not a missing
  measurement, and `SELECTOR_ARMED_NOTHING` is the only genuine no-measurement case.
- **`armed` prints once per process** (`armed_logged` is a `static bool`), so its absence across an arm is
  meaningful only if the process never armed a target; several arms may share one `armed` line's absence.

Nothing else changes: the run config, its declared confound (the name list is the fault-yield enabler), the
hooked-graph limitation (logits chunk only), and every INVALID clause stand as written.
