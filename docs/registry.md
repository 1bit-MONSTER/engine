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
| HF -> GGUF | a `@ModelBase.register(...)` class in llama.cpp's converter names it (upstream pin, then the HRX fork) |
| `vulkan` | the GGUF architecture is in the upstream pin's `src/llama-arch.cpp` |
| `hrx` | the GGUF architecture is in our HRX fork's `src/llama-arch.cpp` |
| `zinc` | ZINC's `parseArchitecture` accepts the GGUF architecture |
| `npu` | the fast lane's model type: `qwen3` (the lane kernels are built for Qwen3-0.6B's shapes) |

On 2026-09-26: 323 HF architectures mapped, vulkan 323, hrx 300, zinc 64, npu 1. Mapped means a backend's code accepts the architecture; the census below reports how many were checked. `registry/architectures.json` always holds the current counts and the pins they come from: every pin bump regenerates it (`scripts/registry-regen.sh`, run by the bump workflows), and CI's `registry_pins` test fails a pin that moved without it.

**Reviewed gaps.** Some unmapped architectures only look like a supported family. [Architecture gaps](arch-gaps.md) records why each of them is not an alias. `registry/significant.json` lists those classes, and `tools/registry_build.py --check-gaps` (ctest `registry_gaps`) fails if one becomes mapped without a recorded reason.

## Checked models

`tools/registry_check.py <1bit> --models <dir>` runs `tests/serve_e2e.sh` for every row of
`registry/check_models.tsv` and records the result in `registry/checked.json`, failures
included. The architecture recorded is the one the backend loads: the GGUF
`general.architecture`, or the NPU directory's `model_type`. Checked on Strix Halo
2026-09-25, engine `8c2805d` (hrx rows at the llama.cpp pin `96f6b89`):

| GGUF architecture | Model | vulkan | hrx | zinc | npu |
|---|---|---|---|---|---|
| `qwen3` | Qwen3-0.6B | pass | pass | pass | pass |
| `qwen2` | Qwen2.5-7B-Instruct | pass | pass | fails: crashes at load | |
| `qwen3moe` | Qwen3-Coder-30B-A3B | pass | pass | pass | |
| `qwen35moe` | Qwen3.6-35B-A3B Q8_0 | pass | pass | pass | |
| `deepseek2` | GLM-4.7-Flash | pass | pass | not mapped | |
| `minicpm` | MiniCPM4-8B | pass | pass | not mapped | |
| `llama` | MiniCPM5-1B | pass | pass | fails: empty reply | |

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

The first full sweep, 2026-09-25:

| | models | share of those with an architecture |
|---|---|---|
| Text-generation models on HF | 415,414 | |
| With an architecture in their config | 332,565 (2,610 architectures) | |
| Mapped | 310,221 | 93.28% |
| Checked | 213,456 | 64.18% |

| Backend | Mapped | Checked |
|---|---|---|
| vulkan | 93.28% | 64.18% |
| hrx | 93.19% | 64.18% |
| zinc | 69.69% | 10.28% |
| npu | 9.29% | 9.29% |

"Checked" counts every model whose architecture has a passing model. It does not mean each
of those models was run. The unmapped architectures with the most models are
`Step1MoEForCausalLM` (2,882), `OPTForCausalLM` (2,097), `ParlerTTSForConditionalGeneration`
(1,586) and `GPTNeoForCausalLM` (1,565). `tools/census.py --pr-body` lists the top ten.

## Commands

```
tools/registry_build.py            # rebuild registry/architectures.json from the pins
tools/registry_build.py --check    # exit 1 if it is stale
tools/registry_check.py build/1bit --models ~/models   # run the checks on Strix Halo
tools/census.py                    # full HF sweep (about 420 pages)
tools/census.py --report           # recompute coverage from the saved counts
```
