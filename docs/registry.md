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

At the current pins: 265 HF architectures, vulkan 265, hrx 246, zinc 44, npu 1.

**Reviewed gaps.** Some unmapped architectures only look like a supported family. [Architecture gaps](arch-gaps.md) records why each of them is not an alias. `registry/significant.json` lists those classes, and `tools/registry_build.py --check-gaps` (ctest `registry_gaps`) fails if one becomes mapped without a recorded reason.

## Checked models

`tools/registry_check.py <1bit> --models <dir>` runs `tests/serve_e2e.sh` for every row of
`registry/check_models.tsv` and records the result in `registry/checked.json`, failures
included. The architecture recorded is the one the backend loads: the GGUF
`general.architecture`, or the NPU directory's `model_type`. Checked on Strix Halo
2026-09-25, engine `8c2805d` (hrx rows at `c19b066`):

| GGUF architecture | Model | vulkan | hrx | zinc | npu |
|---|---|---|---|---|---|
| `qwen3` | Qwen3-0.6B | pass | pass | pass | pass |
| `qwen2` | Qwen2.5-7B-Instruct | pass | pass | fails: no answer | |
| `qwen3moe` | Qwen3-Coder-30B-A3B | pass | pass | pass | |
| `qwen35moe` | Qwen3.6-35B-A3B Q8_0 | pass | pass | pass | |
| `deepseek2` | GLM-4.7-Flash | pass | fails: compute error | not mapped | |
| `minicpm` | MiniCPM4-8B | pass | pass | not mapped | |
| `llama` | MiniCPM5-1B | pass | pass | fails: no answer | |

GLM-4.7-Flash on HRX fails with HTTP 500 "Compute error." (#95). Its attention heads (576)
are wider than HRX0's flash attention takes (512), so llama.cpp runs the model without flash
attention, a path HRX0 cannot compute. HRX0 also claims the model's sigmoid MoE router
nodes, which only its softmax top-8 router dispatch can run. Qwen3-Coder-30B-A3B failed the
same way until `--device hrx` defaulted the context to 32768: at the model's full 262144,
HRX0 declines flash attention too.

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
| hrx | 93.19% | 63.91% |
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
