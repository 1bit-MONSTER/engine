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
# The engine as a Hyperloom custom workload

[Hyperloom](https://github.com/AMD-AGI/Hyperloom) (MIT) optimises an inference stack by proposing a
configuration, measuring it, and keeping it when it is both faster and still correct. Its
`--framework custom` route (scriptable, server-less) runs one script per round and reads one JSON report.
This directory is that script for the engine's HRX route, written for **Hyperloom 1.2.0** with Hyperloom
on the orchestration host (ryzen) and the measurement on the Strix Halo box (gfx1151), which is the 1bit OS
appliance: no compiler, the engine payload at `/opt/1bit-package/1bit`, several lanes sharing one GPU.

| File | Runs on | What it is |
| --- | --- | --- |
| `custom_radeon8060s.sh` | orchestration host | The entrypoint 1.2.0 resolves by board name (`--gpu-type radeon8060s`). Validates the round's levers (names and values), ships `box_bench.sh` + `benchmark.py`, runs ONE `box-hold gpu` ticket over ssh (`../box_ticket.sh`), and turns the report written on the box into `$RESULT_DIR/$RESULT_FILENAME.json`. |
| `box_bench.sh` | box, inside `box-hold gpu timeout` | RAM room (declared need + 16 GB floor, watchdog) and GTT room, then `benchmark.py` against the installed payload. |
| `benchmark.py` | box | The measurement: `llama-bench` pp512/tg128, `1bit serve` code/prose/short tok/s, the KLD gate, the greedy-text gate, the InferenceX-shaped result. |
| `levers.json` | Hyperloom | The knob grid EXPLORE sweeps before the model proposes anything (see "Levers"). |

The Inkling-2-Mini Loom-route workload is `../inkling-loom/`; both share `../box_ticket.sh`.
`hrx2kineto.py` and `hrx-tracelens.sh` stay in `tools/hyperloom/`: the TraceLens bridge (HRX dispatch
profile -> Kineto trace -> roofline).

## What changed for 1.2.0

* No `HYPERLOOM_BENCHMARK_BACKEND=bypass`, no Magpie, no Ray: 1.2.0 removed them and measures every round
  through InferenceSuite. The old `custom_strixhalo.sh` (gpu-hold on the old dev box) is gone; the entrypoint
  is the board-named `custom_radeon8060s.sh`.
* The measurement runs on the box over ssh as one bounded `box-hold gpu` ticket (held part
  `HL_TICKET_MIN`, default 40 minutes, capped at the fleet's 45); Hyperloom never runs on the box.
* Threshold-named pins (`*_MAX`, `*_MIN`) are refused by 1.2.0: the KLD bar is the constant `KLD_BAR` in
  `benchmark.py` (**0.01**, the owner's bar since 22 Nov 2026; it was 1e-4), `HL_GATE_KLD_BAR` overrides it
  for a hand run only.

## Prerequisites (on the box)

* The engine payload (default `HL_PACKAGE=/opt/1bit-package/1bit`): `1bit` at its top and the HRX
  llama.cpp tools under `hrx/llama/bin` -- the layout `benchmark.py` reads as `HL_BUILD`. Its bundled
  `libhsa-runtime64.so.1` becomes `IREE_HAL_AMDGPU_LIBHSA_PATH`.
* The weights (`--model`, a path on the box), plus `HL_MTP_HEAD` / `HL_DFLASH_GGUF` for the drafting levers.
* `wiki.test.raw` for the KLD gate (`HL_WIKI`, default `$HL_BOX_DIR/wiki.test.raw`).

## The launch line (orchestration host)

```sh
python -m hyperloom optimize -v \
    --framework custom \
    --framework-path <disposable public engine checkout> \
    --benchmark-scripts-dir <tracked checkout>/tools/hyperloom/custom \
    --model /data/dev/cl-hyperloom/models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --gpu-type radeon8060s --tp 1 --max-hours 4 \
    --extra-env HL_DFLASH_GGUF=/data/dev/cl-hyperloom/models/Qwen3.8-27B-DFlash2-q8_0.gguf
```

`--extra-env` pins are part of the measurement contract: a variant may add keys but must not overwrite one.
`--benchmark-scripts-dir` must be a *tracked* checkout, never an untracked copy inside the framework path
(integrating a patch there runs `git clean -fd`). The framework path is the code Hyperloom may patch; a
patch to it is only measured once it is in the payload on the box, so with this setup FRAMEWORK_AGENT
rewrites are proposals for a PR, not measured candidates.

## What one measurement is

1. `llama-bench -dev HRX0 -p 512 -n 128 -r 3`: pp512 and tg128 from one model load.
2. `1bit serve` on a free port: greedy 256-token chats on code/prose/short, median of 3 each.
3. `llama-perplexity` KLD against the base logits (`-c 512 --chunks 8 -b 512 -ub 512`, the
   `scripts/hw-gate.sh` recipe), and the three serve replies compared with the baseline's.
4. Score = geometric mean of pp512, tg128 and serve code tok/s, so a lever has to win on balance.
   With `HL_LONGCTX` set, the score is instead the long-context decode rate (see Levers).

The gate is fail-closed and has two halves; a candidate that fails either is rejected:

* **KLD**: mean KL divergence against the base logits recorded at the baseline must be at most
  `KLD_BAR` = **0.01** (the owner's bar since 22 Nov 2026, the hardware gate's own default; it was 1e-4).
* **Greedy text**: the three replies must be byte-identical to the baseline's.

Both references are recorded by the first measurement (the baseline) in `HL_STATE`
(default `$HL_BOX_DIR/state/hrx-<model-stem>` on the box), outside the checkout. Every result names the model file it measured
in `bench_summary.model` (resolved path and its SHA-256 — the same `sha256_head` fingerprint that keys the
references, so a result and a reference can be matched by hash), and every *compared* result names that state
in `quality_gate.reference` (baseline model fingerprint and levers, plus the SHA-256 of `kld-base.kld`
and `texts.json`) — the directory is per-box and the base logits are regenerated when missing, so the
same KLD mean can read `PASS` against a reference the report never named. At the old 1e-4 bar the batch levers
(`-ub 1024`, `-b 2048`: KLD 4.03e-4, +4-5 %) were rejected on 30 Oct; 0.01 admits rounding-level changes and
still rejects a broken one.

Every result also records the conditions it was measured in, in `bench_summary.box`: the die temperature
(`tctl_c`, found by the hwmon channel's `Tctl` label, since the index moves between boots) with `tctl_c_min` /
`tctl_c_max` over samples taken at each phase boundary and `tctl_samples` saying how many were readable, the
box-wide GTT in use at start (`gtt_used_gib` of `gtt_total_gib`) and `mem_available_gib` at start. The score is
measured on a shared box: the same configuration produced 44.69 with a quiet die and 31.68 with 85 GB in use
and 37 GB of swap, and both passed the gate, which is about the output and not about comparability. Each field
is `null` where the box does not expose it, so an unreadable sensor is never read back as a clean arm. An arm
killed mid-run (the 2026-11-01 thermal abort at 93 C) still writes no result — the fields describe arms that
finished, and a killed arm is reported by its absence.

## Levers

`levers.json` seeds EXPLORE; the model can still propose its own. Each entry sets environment that
`benchmark.py` turns into flags:

| Lever | Flags | Notes |
| --- | --- | --- |
| `HL_B`, `HL_UB` | `-b`, `-ub` | batch / micro-batch. `llama-bench` and the KLD leg only: `1bit serve` has its own parser and rejects them (`unknown option -ub`), setting batching per route itself. |
| `HL_FA` | `-fa on\|off\|auto` | flash attention, on both legs. `off` reaches serve and then fails at decode (`decode() failed: Compute error`, HTTP 500) on this pin — a real finding, reported as a rejected candidate. |
| `HL_CTK`, `HL_CTV` | `-ctk`, `-ctv` | KV cache types (`f16`, `q8_0`, ...), `llama-bench`/KLD only (serve has no such flag). A quantized cache implies `-fa on`: without it the bench dies in `graph_compute ... error -1`. |
| `HL_MTP_MAX` | `--no-mtp` / `--mtp-max N` | A GGUF with its own MTP head drafts **by default** on HRX (`1bit serve` reports it); `0` turns drafting off, `N` caps the draft length, unset keeps serve's default (draft length 3 via its recipe). `HL_MTP_HEAD` names an external head instead. Because the default *drafts*, `bench_summary.drafting.requested` is recorded explicitly — `null` means no lever was set and the pin's default applied, `{"HL_MTP_MAX": "0"}` means a deliberately non-drafting measurement — and `drafting.serve` carries what the server itself reported (`spec_type`, the draft `recipe`, and the accepted/generated counts with their mean accepted length), and its `status` says which of the four cases the row is in — `drafted`, `not-drafted` (the serve said there was no spec implementation, which is what `HL_MTP_MAX=0` produces), `no-evidence` (log read, nothing either way) or `unreadable` (log could not be read, i.e. nobody knows) — so a lost log can never serialise as a non-drafting measurement. |
| `HL_DFLASH`, `HL_DFLASH_GGUF` | `--dflash <drafter>` | block-diffusion drafter; exclusive with `--mtp`. `HL_DFLASH_GGUF` names the drafter file (default `dflash.gguf`, so relative to the run's cwd) and this model has two quants of it, so the arm's identity is the *pair*: `bench_summary.drafting` reports the resolved path and sha256 of the file the measurement really used, and of `HL_MTP_HEAD` when that is the head in use. Before that, a DFlash arm recorded `{"HL_DFLASH": "1"}` and the drafter was recoverable only from `serve.log`. |
| `HL_PARALLEL` | `--parallel N` | HRX runs one gated-delta-net sequence at a time, so this is expected flat on Qwen3.8 |
| `HL_GTT_NEED_GIB`, `HL_GTT_RULE_GIB` | `box_bench.sh`, inside the ticket | **Not levers: admission.** The engine's guardrail (`ggml_hrx`, engine #680) refuses a load whose device buffers do not fit, *after* the device is ours; on 2026-11-08 that cost two claims as `llama-bench failed (rc=1, rows=[])` / score 0.0, which reads like a broken workload. `box_bench.sh` refuses the ticket with a named reason unless `GTT in use + HL_GTT_NEED_GIB <= HL_GTT_RULE_GIB` (21 GiB -- `1bit serve` prints `need 20.21 GiB` for the 27B -- and 95 GiB). Inside `box-hold gpu` no other lane loads the GPU, so the check runs once, after the grant. |
| `GGML_HRX_*` | environment only | the knobs `docs/hrx.md` names (`Q8_PREFILL_RELAX`, `GRAPH_PROGRAM_CACHE`, ...) |
| `HL_CTX` | `-c` | context the serve leg is started at (default 8192). Not a lever: without it serve sizes the KV cache for the model's own context and aborts on its own fit check whenever the box is busy. Raise it for a long-context workload; `HL_LONGCTX` must fit inside it (`HL_CTX >= HL_LONGCTX + HL_LONGTOK`), and a run that does not fit is refused before anything loads. |
| `HL_LONGCTX`, `HL_LONGTOK` | `-c` + the long request | **engine#679 metric.** `HL_LONGCTX=<tokens>` adds one long request to the serve leg, on a deterministic filler prompt sized at this filler's measured ~4.85 characters/token, and makes **its decode rate the score**; `HL_LONGTOK` is how many tokens it **asks** for (default 128). The long leg is the only request that sets `ignore_eos`, so the decode runs the requested length instead of stopping at EOS early — that makes the rate's denominator the one we asked for. It is a request to the backend, not a guarantee from it, so the result records both sides: `longctx.decode_tokens_requested`, `longctx.decode_tokens` (what came back) and `longctx.decode_complete`, and `throughput_unit` names the count the score was divided by (`tok/s (decode at 18762-token ctx, 512 decoded)`, or `…, 278 of 512 requested)` when the decode stopped short) — two rows differing only in that count are not the same measurement. The short legs never pin it: their replies are the greedy-text gate's inputs. The short legs and the gate's code are otherwise unchanged, and `bench_summary.longctx` reports the context the server actually saw, the decode count and the prefill rate. The long reply is deliberately **never** compared by the greedy-text gate — it depends on the context, not on the lever — and that holds even when the reference was established by a long-context run and therefore carries a `long` reply: the gate compares the short legs only (`gated_text_keys()`), so a short-prompt run is never gated on a key it does not produce. |

Every serve leg is also started with `--placement-report`, which is the second measurement
condition rather than a lever: `GGML_HRX_REPORT=2` alone does nothing, because llama-server drops the
library's INFO lines below `-lv 4` and only that flag lifts them, so before it `bench_summary.placement`
was `null` in every artifact and a graph that left HRX was indistinguishable from one nobody looked
at (`docs/serve.md`, engine#390). It reports; `GGML_HRX_STRICT=0` (set for the whole measurement)
keeps it from refusing anything. Numbers from before it was added were measured at serve's default
verbosity and are not directly comparable with numbers after it.

## Box etiquette

* Every measurement is one `box-hold gpu` ticket (`../box_ticket.sh`), so lanes never share the GPU; the
  held part is capped by `timeout` at `HL_TICKET_MIN` minutes (default 40, refused above 45), and stopping
  the round kills the remote ticket by its id.
* The job declares its RAM need (`HL_RAM_NEED_GB`, default 32) and is refused below need + 16 GB; a
  watchdog stops it if MemAvailable falls under 16 GB (the box's own memory guard would kill it anyway).
* `HL_BUDGET_SEC` (default 780) bounds `benchmark.py` itself so it finishes inside the ticket.
* `HL_REPS` (default 3 — the repetition count behind the pp512/tg128 medians, recorded as
  `bench_summary.llama_bench_reps`), `HL_TOKENS`, `HL_WIKI`, `HL_STATE` are the other knobs.
