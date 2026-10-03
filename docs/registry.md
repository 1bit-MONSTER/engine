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
# Model registry and HF census

Step 5 of the port ([PORTING.md](PORTING.md)): which Hugging Face models this engine can
run, kept current every day. Two numbers are reported, and they are never added together:

- **Mapped:** a backend's own code accepts the model's architecture.
- **Checked:** a model of that architecture loaded, answered and streamed through
  `1bit serve` on Strix Halo (`tests/serve_e2e.sh`).

## The registry

`registry/architectures.json` maps each HF architecture (a config's `architectures[0]`,
such as `Qwen3ForCausalLM`) to its GGUF architecture and the backends that accept it.
`tools/registry_build.py` generates it from the pinned sources. None of it is typed in by
hand:

| Backend | Accepts the architecture when |
|---|---|
| HF -> GGUF | a `@ModelBase.register(...)` class in our HRX fork's llama.cpp converter names it |
| `hrx` | the GGUF architecture is in our HRX fork's `src/llama-arch.cpp`; the same llama-server runs it on the CPU (`--device cpu`), so `hrx` also means the CPU route |
| `zinc` | ZINC's `parseArchitecture` accepts the GGUF architecture |
| `npu` | the fast lane's model type: `qwen3` (the lane kernels are built for Qwen3-0.6B's shapes) |

The `vulkan` column (the upstream llama.cpp release the engine built for Vulkan) left with that
build in RFC #213 stage 3. With it went the HF class names only the upstream
converter registers: 33 classes, 23 of them for architectures the HRX fork does not run (among
them `qwen4exp`, `zamba`, `zamba2`, `blackmamba`) and 10 aliases of architectures it does run
(DFlash drafters, `exaone-moe`, `nemotron_h_moe`), which a GGUF converted elsewhere still runs on.

On 2026-10-03 (RFC #213 stage 3): 290 HF architectures mapped, hrx 290, zinc 63, npu 7. Mapped means a backend's code accepts the architecture; the census below reports how many were checked. `registry/architectures.json` always holds the current counts and the pins they come from: every pin bump regenerates it (`scripts/registry-regen.sh`, run by the bump workflows), and CI's `registry_pins` test fails a pin that moved without it.

**Reviewed gaps.** Some unmapped architectures only look like a supported family. [Architecture gaps](arch-gaps.md) records why each of them is not an alias. `registry/significant.json` lists those classes, and `tools/registry_build.py --check-gaps` (ctest `registry_gaps`) fails if one becomes mapped without a recorded reason.

## Checked models

`tools/registry_check.py <1bit> --models <dir>` runs `tests/serve_e2e.sh` for every row of
`registry/check_models.tsv` and records the result in `registry/checked.json`, failures
included. The architecture recorded is the one the backend loads: the GGUF
`general.architecture`, or the NPU directory's `model_type`. Checked on Strix Halo
2026-09-25, engine `8c2805d` (hrx rows at the llama.cpp pin `96f6b89`):

| GGUF architecture | Model | hrx | zinc | npu |
|---|---|---|---|---|
| `qwen3` | Qwen3-0.6B | pass | pass | pass |
| `qwen2` | Qwen2.5-7B-Instruct | pass | fails: crashes at load | |
| `qwen3moe` | Qwen3-Coder-30B-A3B | pass | pass | |
| `qwen35moe` | Qwen3.6-35B-A3B Q8_0 | pass | pass | |
| `deepseek2` | GLM-4.7-Flash | pass | not mapped | |
| `minicpm` | MiniCPM4-8B | pass | not mapped | |
| `llama` | MiniCPM5-1B | pass | fails: empty reply | |

(The vulkan rows, which passed on the same date, were dropped from `registry/check_models.tsv` and
`registry/checked.json` with the Vulkan build, as were the Zyphra Zamba rows, which ran on Vulkan
only.)

Both ZINC failures come from upstream ZINC, pinned at `3a35e76`:
- **Qwen2.5-7B crashes at load (exit 136, SIGFPE in RADV).** ZINC sizes its DMMV shaders'
  shared-memory input buffer with the model's largest dimension. Qwen2.5-7B's
  intermediate size is 18944, which needs 75,776 bytes, more than a workgroup's 64 KiB.
- **MiniCPM5-1B gives an empty reply.** Its chat template opens with `{{- bos_token }}`,
  which ZINC's built-in ChatML renderer drops. Without `<s>`, the model ends the turn at
  once. llama.cpp renders the `<s>`, and the model answers.

Both are fixed on `bong-water-water-bong/zinc` branch `fix/lds-clamp-and-template-bos`
(`5453c19`). With it, all five ZINC rows pass and ZINC's own tests pass 635/635. The engine
keeps pinning upstream ZINC until the fixes are there.

GLM-4.7-Flash and Qwen3-Coder-30B-A3B failed on HRX until the llama.cpp pin `96f6b89` (#95).
The pin fixes four things:
- HRX0 now claims flash attention for MLA heads (576) and at full context (262144).
- The path without flash attention now compiles.
- HRX0 no longer claims a sigmoid MoE router.
- HRX0's expert matmul (MUL_MAT_ID) now reads each token's expert ids at their real row
  stride. It used to read them packed, so every token of a batched prompt after the first
  went to the wrong experts. Qwen MoE models were immune, because their router dispatch
  supplies the stride.

## The census

`tools/census.py` walks every page of the HF API's text-generation listing (with each
model's config inline) and counts models by architecture. `registry/census.json` keeps the
counts and the coverage read against the registry and the checked results. The
`census.yml` workflow runs daily at 05:17 UTC. It rebuilds the registry from the pins,
sweeps HF, and opens a PR when anything changed.

The census of 2026-09-27 (the first full sweep ran 2026-09-25), with `registry/census.json` holding the latest:

| | models | share of those with an architecture |
|---|---|---|
| Text-generation models on HF | 415,695 | |
| With an architecture in their config | 332,726 (2,616 architectures) | |
| Mapped | 315,701 | 94.88% |
| Checked | 214,033 | 64.33% |

| Backend | Mapped | Checked |
|---|---|---|
| hrx | 94.78% | 64.32% |
| zinc | 69.83% | 10.28% |
| npu | 9.29% | 9.29% |

"Checked" counts every model whose architecture has a passing model. It does not mean each
of those models was run. The unmapped architectures with the most models are
`Step1MoEForCausalLM` (2,882), `OPTForCausalLM` (2,097), `ParlerTTSForConditionalGeneration`
(1,586) and `GPTNeoForCausalLM` (1,565). `tools/census.py --pr-body` lists the top ten.

## The watch

The sweep is a snapshot, so a class that appears *tomorrow* moves the mapped share
quietly. `tools/census_watch.py` polls the newest text-generation models (createdAt
descending), reads each one's architecture class from the config inline in the listing,
and compares it against `registry/architectures.json` — the same read `census.py`'s
coverage() makes, so "uncovered" here means "not in the number the census reports". No
per-model fetch and no compiled probe: the registry is generated data, keyed by the HF
architecture class the census counts.

Exit 0 means every newest model is mapped; **exit 1 is the alert** — a class arrived that
no backend accepts. One kind of unmapped class is reported without the alert: a class whose
models all load through custom modeling code (`auto_map` in the config, so Transformers has
no implementation of it either) and that fewer than 3 uploaders publish. Research one-offs
like that arrive every day. It alerts once 3 uploaders use it, and a reviewed class
(`registry/significant.json`) always alerts.

The watch runs daily at 04:30 UTC (ahead of the 05:17 sweep) in two places:
- GitHub Actions (`.github/workflows/census-watch.yml`, [RFC #246](https://github.com/1bit-MONSTER/engine/discussions/246)), which files one deduped
  `census-watch` issue. Only its alert job holds `issues: write`, and that job runs no
  repository code.
- the systemd timer on the development box (`scripts/1bit-census-watch.{service,timer}`, from
  the `~/census-main` worktree that tracks `origin/main`), which marks its unit failed and
  logs every run to `~/.1bit/logs/census-watch-*.log`.

A class the watcher flags is one of two things, and the run says which:

- **new class** — map it in `registry/architectures.json`, but only when a pinned
  backend's code really accepts the architecture (rule 3 in CONTRIBUTING.md: recognized
  is not verified).
- **reviewed, not an alias** — already in `registry/significant.json`
  ([arch-gaps.md](arch-gaps.md)); it needs real engine support, and if a backend starts
  accepting it, the reason is recorded as `mapped_ok`.

Gated repos (no config without a token) are reported as unverifiable and never fail the
run; derivatives (a quantized GGUF, a LoRA adapter) have no config by design and are
skipped, because the release they derive from carries the config and is what gets checked.

## Commands

```
tools/registry_build.py            # rebuild registry/architectures.json from the pins
tools/registry_build.py --check    # exit 1 if it is stale
tools/registry_check.py build/1bit --models ~/models   # run the checks on Strix Halo
tools/census.py                    # full HF sweep (about 420 pages)
tools/census.py --report           # recompute coverage from the saved counts
tools/census_watch.py              # newest models vs the registry; exit 1 on an uncovered class
scripts/census-watch.sh            # the same, with the log file and the origin/main refresh
```
