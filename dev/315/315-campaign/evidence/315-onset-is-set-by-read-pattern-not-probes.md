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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The onset layer is set by the READ PATTERN, not by the probes — and it is already answered on the record

This is the first result in the #315 work that **does not need a new run**: three conditions already exist in
the archived arms, and they separate the two candidate variables.

## The three conditions, read from `EXTRA_ENV` in the consoles

| condition | arms | selector | probes | onset layer | evidence |
|---|---|---|---|---|---|
| **A** | `315size` | 15-name `l_out` list | **none** | **0** | `l_out-0 nan=1984 of 276480` (r2e), `1984 of 503808` (r1e), `1790 of 20480` (r2q) |
| **B** | `315lout4`, `315inp2` | `l_out` name list | `COPY_PROBE` + `SENTINEL` + `LOGITS_FP` | **0** | `315lout4-r1e l_out-0 nan=1984 of 200704`; `315lout4-r1c l_out-0 nan=46 of 20480`; `315inp2-r3e` first line `l_out-1` (so `l_out-0` clean) |
| **C** | `315nan`, `315fbar` | **`tail:16`** (all ~9 nodes/chunk) | same three probes | **41** | `node_2948 nan=2048` exactly, 18 of 19 arms |

A and B share a selector and differ in probes: **the onset does not move (layer 0 either way).** B and C share
the probes and differ in selector: **the onset moves (0 → 41).**

**Therefore the read pattern is the variable that sets the onset layer, and the probe set is not.** That is the
`A vs C` dose test hypothesised in `315-onset-layer-is-config-dependent.md`, answerable from existing data —
and it answers *in favour of* the hypothesis: onset is a function of how many tensors the instrument reads
(~1–3 with a name list, ~9 with `tail:16` over a 9-node subgraph).

## What this means for the mechanism question

- **Neither layer 0 nor layer 41 is "the" site of the defect.** Both are where a particular *instrument* first
  sees NaN. Under a name list the corruption is present at layer 0's output; under `tail:16` layer 0 is clean
  and the first NaN is at layer 41. Since `tail:16` reads *more* and still does not see layer-0 NaN, the read
  pattern is not merely hiding the earlier site — it is **changing which layers carry NaN**.
- This is the record's "instrumentation is an inducer" finding, now localised: the dose does not just select
  the face (Face A vs Face B) or the rate, it **selects the layer**.
- **Any claim of the form "the corruption is at layer N" is a claim about an instrument** unless the read
  pattern is held fixed, which is why the earlier revisions of this thread kept collapsing.
- What survives as instrument-independent is the **size**: one `n_embd`-wide row, `2048` floats, with
  `finite + nan = extent` exactly — and even that was measured under condition C and should be re-measured
  under A before being called universal.

## A correction to the peer's comparison, because it drove a decision

Commit `0ed67c2` describes `315lout4` as *"disc rig, 16 `l_out` names, **no probes**"* and concludes that the
configs which yield Face A are *"the name list **without** the extra probes"*. The console shows otherwise:

    extra_env='GGML_HRX_COPY_PROBE=1 GGML_HRX_SENTINEL=1 GGML_HRX_LOGITS_FP=100000
               GGML_HRX_TENSOR_FP=l_out-0,l_out-3,...,l_out-47 GGML_HRX_TENSOR_FP_NANONLY=1'

`315lout4` **did** carry all three probes. So the amended run's selector (a 15-name `l_out` list with the
probes retained) is very close to `315lout4`, which is the configuration that already worked — the switch is
therefore likely still correct in outcome, but its stated reason ("without the probes") is not what the logs
show, and the same table's inference that the probes suppress the yield is unsupported.

## What the running `315cfg` is now worth

`315cfg` runs A′ (name list, no probes) against B (name list, probes) — the contrast that the table above
already answers. It remains a **formal** pre-registered test of that half, and it is worth letting finish for
that reason, but **its result will not be news**, and the interesting arm is the `A vs C` pair, which the
archive has already supplied. If `315cfg` is thermally blocked past its cap, abandoning it loses little.

## Caveats

- `315size` and `315lout4` ran on a **thermally saturated** box (`thermal_pauses` to 29, `stopped_frac` to
  0.82), so conditions A and B are not as clean as they look; C's arms came from a quieter box.
- `315inp2-r3e`'s first line is `l_out-1`, which is onset ≤1, not 0 — it is the only B-adjacent arm that does
  not start at `l_out-0`, and it may simply have had a clean `l_out-0`.
- All three conditions are arm-level single readings; none has a rate.

## `315cfg` was STOPPED, with the reason, before producing any arm

I launched `315cfg` (A′ = name list without probes, B = same list with probes) at 03:49 on 2026-10-14. It
never produced an arm. Two things happened in the following six minutes:

1. I found that **condition B already exists on the record** as `315lout4` (and `315inp2`) — the comparison
   `315cfg` was built to make is answered above, from arms already in the archive.
2. It spent the whole time in the **gate wait** — `tctl` 92–94 °C, then once cooled to 75 °C the gate stayed
   shut on `load1 = 8.6–11`, above the required 4, from foreign workloads. Its cap is 2 h.

It was stopped at 03:55 with no arm run, and the box released (lock verified free, no orphaned neighbours —
the one leftover `sleep 30` holding the inherited lock fd was killed).

**This is a stop, not a pruning.** Nothing was discarded, no arm was excluded after the fact, and no result is
being withheld: the run produced no result to withhold. The reason is recorded here so it is clear that the
`A vs B` half of the onset question is answered **from the archive** and not from this run, and so that a
reader does not later look for `315cfg*` arms that do not exist.

**What is lost:** a same-session, back-to-back A′-vs-B control. The archive comparison is cross-session and
partly cross-rig, which is weaker — but both A and B read **layer 0**, so a probe effect would have to be
masked by session drift to be missed, and that is stated rather than dismissed.
