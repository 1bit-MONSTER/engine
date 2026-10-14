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
