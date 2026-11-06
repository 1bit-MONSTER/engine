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
configuration, measuring it, and keeping it when it is both faster and still correct. Upstream
1.1.x has a supported route for code it has no framework definition for: `--framework custom` on
the scriptable (server-less) path, with `--framework-path` naming the checkout it may patch and
`--benchmark-scripts-dir` naming the directory that holds the entrypoint. This directory is that
entrypoint for the engine's HRX route on Strix Halo (gfx1151).

The old route here (`hl_patch_strixhalo.py` teaching an installed 1.0.0 package about `strixhalo`,
plus `hl_cli_login.py` and a `launch.sh` wrapper) is gone: upstream now knows the board
(`--gpu-type radeon8060s`), and nothing in this repository patches Hyperloom itself any more.

| File | What it is |
| --- | --- |
| `custom_radeon8060s.sh` | The name upstream resolves (`custom_<runner-type>.sh`, runner type `radeon8060s`). Forwards to the workload below. |
| `custom_strixhalo.sh` | The entrypoint: takes one claim on the box's GPU FIFO (`gpu-hold -l hyperloom -t 20`) and runs one measurement. Builds nothing. |
| `benchmark.py` | The measurement: `llama-bench` pp512/tg128, `1bit serve` code/prose/short tok/s, the KLD gate, the greedy-text gate, and the InferenceX-shaped result Hyperloom reads. |
| `levers.json` | The knob grid EXPLORE sweeps before the model proposes anything (see "Levers"). |

`hrx2kineto.py` and `hrx-tracelens.sh` stay in `tools/hyperloom/`: they are the TraceLens bridge
(HRX dispatch profile -> Kineto trace -> roofline), used by the profile leg and by hand.

## Prerequisites

* A **built** framework checkout. Hyperloom commits KEEP candidates into `--framework-path`, so
  that checkout is disposable and its build lives outside it:

  ```sh
  git worktree add -b hl/run ~/wt/hl-engine origin/main
  git -C ~/wt/hl-engine submodule update --init --recursive third_party/llama.cpp third_party/hrx-system
  cmake -B ~/wt/hl-engine-build -G Ninja -S ~/wt/hl-engine -DCMAKE_BUILD_TYPE=Release \
      -DONEBIT_HRX=ON -DONEBIT_HRX_TOOLCHAIN=/opt/rocm-therock \
      -DONEBIT_HRX_LIBHSA=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
  cmake --build ~/wt/hl-engine-build -j 4 --target 1bit llama-bench llama-perplexity llama-server
  ```

  The build root is `HL_BUILD` (default `~/wt/hl-engine-build`): `1bit` sits at its top, and the
  HRX llama.cpp tools (`llama-bench`, `llama-perplexity`) under `hrx/llama/bin`, which is what the
  `llama_hrx` ExternalProject target writes. `HL_BIN` / `HL_ONEBIT` override those two paths.
* The weights plus a drafting sidecar (optional but the levers that use them need them):
  the model under `--model`, `HL_MTP_HEAD` (default `<...>/MTP/mtp-*.gguf`) and `HL_DFLASH_GGUF`.
* `IREE_HAL_AMDGPU_LIBHSA_PATH` pointing at the HSA runtime. The entrypoint defaults it to the
TheRock's copy and `HL_LIBHSA` overrides it; without it the HRX backend registers **no device**
("Available devices: (none)") and every leg fails.
* `HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1` on any box whose libhsa page-faults
  an HRX allocation (engine #715, the nightly libhsa). The entrypoint defaults all three to those
  values and an explicit setting still wins, so one candidate can measure SVM on if that is wanted.
* `wiki.test.raw` for the KLD gate (`HL_WIKI`, default `~/models/qwen38/wikitext-2-raw/wiki.test.raw`).
* A Hyperloom install (upstream, unpatched), `HYPERLOOM_BENCHMARK_BACKEND=bypass`, and Ray on a
  free port (`RAY_ADDRESS=127.0.0.1:6390`: 6379 is taken by a docker redis on this box).

## The launch line

```sh
export HYPERLOOM_BENCHMARK_BACKEND=bypass RAY_ADDRESS=127.0.0.1:6390
export MODEL=$HOME/models/qwen38/UD-Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
FRAMEWORK=$HOME/wt/hl-engine
$HOME/hyperloom-ws/.venv-2026-10-30/bin/python -m hyperloom optimize -v \
    --framework custom \
    --framework-path "$FRAMEWORK" \
    --benchmark-scripts-dir "$FRAMEWORK/tools/hyperloom/custom" \
    --model "$MODEL" \
    --gpu-type radeon8060s --max-hours 4 \
    --extra-env HL_BUILD="$HOME/wt/hl-engine-build" \
    --extra-env HL_DFLASH_GGUF="$HOME/models/qwen38/DFlash2/Qwen3.8-27B-DFlash2-q8_0.gguf"
```

`--extra-env` pins are part of the measurement contract: a variant may add keys but must not
overwrite one, so the build path and the sidecar paths belong there, not in a shell export.

The LLM agents run on the DeepSeek API through the Anthropic-compatible endpoint, env only:

```sh
export ANTHROPIC_AUTH_TOKEN=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.pi/agent/auth.json")))["deepseek"]["key"])')
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic CLAUDE_MODEL=deepseek-v4-pro
unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN
```

Upstream's preflight writes that endpoint and key into `~/.claude/config.json` of whatever `HOME`
it runs under. On a shared box that is the CLI redirect for everyone, so run with a **sandboxed
`HOME`** that holds a `.gitconfig` (Hyperloom commits KEEPs into the checkout, so `git commit`
must work there) and let preflight configure the sandbox's own `~/.claude`.

Two directories, deliberately: `--benchmark-scripts-dir` must be a *tracked* copy of these files
(a checkout of this repository), never an untracked copy inside the framework path — integrating a
patch there runs `git clean -fd` and would delete it.

## What one measurement is

1. `llama-bench -dev HRX0 -p 512 -n 128 -r 3`: pp512 and tg128 from one model load.
2. `1bit serve` on a free port: greedy 256-token chats on code/prose/short, median of 3 each.
3. `llama-perplexity` KLD against the base logits (`-c 512 --chunks 8 -b 512 -ub 512`, the
   `scripts/hw-gate.sh` recipe), and the three serve replies compared with the baseline's.
4. Score = geometric mean of pp512, tg128 and serve code tok/s, so a lever has to win on balance.
   With `HL_LONGCTX` set, the score is instead the long-context decode rate (see Levers).

The gate is fail-closed and has two halves; a candidate that fails either is rejected:

* **KLD**: mean KL divergence against the base logits recorded at the baseline must be at most
  `HL_GATE_KLD_MAX` (default `1e-4`, the owner's bar — much tighter than the hardware gate's 0.01).
* **Greedy text**: the three replies must be byte-identical to the baseline's.

Both references are recorded by the first measurement (the baseline) in `HL_STATE`
(default `~/lb/hl-gate/<model-stem>`), outside the checkout. Every result names the model file it measured
in `bench_summary.model` (resolved path and its SHA-256 — the same `sha256_head` fingerprint that keys the
references, so a result and a reference can be matched by hash), and every *compared* result names that state
in `quality_gate.reference` (baseline model fingerprint and levers, plus the SHA-256 of `kld-base.kld`
and `texts.json`) — the directory is per-box and the base logits are regenerated when missing, so the
same KLD mean can read `PASS` against a reference the report never named. At 1e-4, a lever that
legitimately changes rounding (KV cache types, flash-attn, micro-batch) will usually fail the KLD half — that is
the bar as specified, and `HL_GATE_KLD_MAX` moves it without touching code.

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
| `GGML_HRX_*` | environment only | the knobs `docs/hrx.md` names (`Q8_PREFILL_RELAX`, `GRAPH_PROGRAM_CACHE`, ...) |
| `HL_CTX` | `-c` | context the serve leg is started at (default 8192). Not a lever: without it serve sizes the KV cache for the model's own context and aborts on its own fit check whenever the box is busy. Raise it for a long-context workload; `HL_LONGCTX` must fit inside it (`HL_CTX >= HL_LONGCTX + HL_LONGTOK`), and a run that does not fit is refused before anything loads. |
| `HL_LONGCTX`, `HL_LONGTOK` | `-c` + the long request | **engine#679 metric.** `HL_LONGCTX=<tokens>` adds one long request to the serve leg, on a deterministic filler prompt sized at this filler's measured ~4.85 characters/token, and makes **its decode rate the score**; `HL_LONGTOK` is how many tokens it decodes (default 128). The short legs and the gate's code are unchanged, and `bench_summary.longctx` reports the context the server actually saw, the decode count and the prefill rate. The long reply is deliberately **never** compared by the greedy-text gate — it depends on the context, not on the lever — and that holds even when the reference was established by a long-context run and therefore carries a `long` reply: the gate compares the short legs only (`gated_text_keys()`), so a short-prompt run is never gated on a key it does not produce. |

Every serve leg is also started with `--placement-report`, which is the second measurement
condition rather than a lever: `GGML_HRX_REPORT=2` alone does nothing, because llama-server drops the
library's INFO lines below `-lv 4` and only that flag lifts them, so before it `bench_summary.placement`
was `null` in every artifact and a graph that left HRX was indistinguishable from one nobody looked
at (`docs/serve.md`, engine#390). It reports; `GGML_HRX_STRICT=0` (set for the whole measurement)
keeps it from refusing anything. Numbers from before it was added were measured at serve's default
verbosity and are not directly comparable with numbers after it.

## Box etiquette

* Every measurement runs inside `gpu-hold -l hyperloom -t 20`, so the box's fair FIFO is held for
  exactly the measurement and the watchdog aborts at the critical die temperature.
* The measurement assumes it is the only GPU work in flight: it does not wait for a cool die first,
  so do not raise the hold above the measurement's own worst case. `HL_BUDGET_SEC` (default 780)
  bounds the script itself so it finishes inside the hold.
* `HL_REPS` (default 3 — the repetition count behind the pp512/tg128 medians, recorded as
  `bench_summary.llama_bench_reps`), `HL_TOKENS`, `HL_WIKI`, `HL_STATE`, `HL_GPU_HOLD_MINUTES` are the other
  knobs.
