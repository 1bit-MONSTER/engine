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
# Vulkan from upstream llama.cpp

`1bit serve --device vulkan` runs a `llama-server` built from upstream
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), pinned to its latest
release in `third_party/llama.cpp-vulkan`, plus the few upstream pull requests we
carry until upstream merges them ("Carried upstream PRs" below). It is a separate tree from
`third_party/llama.cpp`, which stays on the llama.cpp + hrx-system pair AMD
tests together for HRX ([hrx.md](hrx.md)).

Two pins, because the two routes stay current in different ways:

| Route | Source | Pinned to | Moved by |
|---|---|---|---|
| `--device vulkan` | `third_party/llama.cpp-vulkan` (1bit-MONSTER/llama.cpp `1bit/vulkan-upstream`) | ggml-org's latest release + carried upstream PRs | `bump-llama-vulkan.yml`, daily |
| `--device hrx` | `third_party/llama.cpp` + `third_party/hrx-system` | AMD's tested pair (ROCm/ggml-staging-automation) | `bump-hrx.yml`, daily |

A new model architecture reaches the Vulkan route the day upstream releases it,
without waiting for AMD's pair to move. Qwen3.8-Flash-Next (`qwen4exp`) is the
case that prompted this: upstream added it on 2026-08-27, and AMD's pinned
llama.cpp (`f1a0aca141de`) cannot load it.

## Carried upstream PRs

`1bit/vulkan-upstream` is ggml-org's release plus these commits. On each new release
`bump-llama-vulkan.yml` rebases them onto it; one that upstream has merged becomes
empty and drops out, and one that no longer applies stops the bump for a hand rebase.
The commit the engine pinned before is kept as the tag `vulkan-upstream-<sha12>`.

| Upstream PR | What it adds | Carried as |
|---|---|---|
| [ggml-org#28243](https://github.com/ggml-org/llama.cpp/pull/28243) (Daniel Han, Ryan Monsurate) | Qwen3.8-Flash-Next's NextN/MTP draft head (`--spec-type draft-mtp`), and the fix for `-md` loading the target model instead of the draft file | `62484fba` on v0.5.0 |

Reviewed line by line before it was pinned: it touches only the qwen4exp model, its
converter, and two lines of the shared speculative decoding (the `-md` path fix, and
KV sharing kept to gemma4-assistant drafts, as v0.5.0 already did). A draft head can
only change speed: the target model checks every drafted token.

## Build

Needs the Vulkan headers, loader and `glslc`.

```
git submodule update --init --depth 1 third_party/llama.cpp-vulkan
cmake -B build -G Ninja -DONEBIT_VULKAN=ON
cmake --build build
```

`ONEBIT_VULKAN` and `ONEBIT_HRX` build side by side: `--device vulkan` then uses
the upstream build and `--device hrx` the HRX build. Without `ONEBIT_VULKAN`,
`--device vulkan` falls back to the HRX build's Vulkan backend, as before.
`ONEBIT_LLAMA_SERVER` still overrides both.

## Verified (Strix Halo)

v0.5.0 `7fe450e19305`:

| Model | Result |
|---|---|
| `tests/serve_e2e.sh` (Qwen3-0.6B Q4_K_M) | PASS (`serve_e2e_vulkan_upstream`) |
| Qwen3.8-27B UD-Q4_K_XL | "The capital of France is Paris.", 12.2 tok/s |
| Qwen3.8-Flash-Next UD-Q4_K_XL (`--ctx-size 8192`) | "The capital of France is Paris.", 22.0 tok/s; the HRX pin fails to load it |

v0.5.0 + ggml-org#28243 (`62484fba`), llama-server built from the branch:

| Model | Result |
|---|---|
| Qwen3-0.6B Q4_K_M | "The capital of France is Paris.", 315-331 tok/s (unchanged) |
| Qwen3.8-27B UD-Q4_K_XL + `--mtp` (Q4_0 head) | 35.0 / 28.1 / 32.2 tok/s code / prose / short; draft acceptance 188/200, 171/252, 6/6, the same as v0.5.0 |

Large models need `--ctx-size`: without it llama-server allocates the KV cache
for the model's full trained context (262,144 tokens for Qwen3.8), which does not
fit next to Flash-Next's 104 GiB of weights.

Measured quant sweet spots for Qwen3.8 on this route are on the
[wiki](https://github.com/1bit-MONSTER/engine/wiki).

## ZAYA1 (Zyphra), from our llama.cpp

[ZAYA1](https://huggingface.co/Zyphra/ZAYA1-8B) is Zyphra's MoE: 8.8B parameters, 760M active per
token. Every layer runs CCA attention, where q and k pass through short convolutions over time,
then a top-1 router that picks one of 16 experts or a skip expert. Upstream llama.cpp has no ZAYA:
its draft (ggml-org/llama.cpp#23112) was closed on 2026-09-05. ZAYA lives in our llama.cpp,
`third_party/llama.cpp` (`1bit-MONSTER/llama.cpp`, [PR #2](https://github.com/1bit-MONSTER/llama.cpp/pull/2)),
and runs there on Vulkan, HRX and ROCm (HIP). `1bit serve --device vulkan` reads the GGUF's
`general.architecture`, and runs an architecture that only our llama.cpp has on that build's
`Vulkan0` even when the upstream build is present; `--device hrx` runs it on `HRX0`.

```
python third_party/llama.cpp/convert_hf_to_gguf.py <Zyphra/ZAYA1-8B> --outtype f16 --outfile zaya1-8b-f16.gguf
build/hrx/llama/bin/llama-quantize zaya1-8b-f16.gguf zaya1-8b-Q4_K_M.gguf Q4_K_M
1bit serve -m zaya1-8b-Q4_K_M.gguf --device vulkan --ctx-size 8192
```

Convert with this converter: it writes the grouped convolution's weights tap-major, which the
graph expects, so GGUFs made by other converters do not load.

Verified on Strix Halo (Radeon 8060S):

| Check | Result |
|---|---|
| Against transformers' `ZayaForCausalLM` in FP32, 96 teacher-forced positions | the F16 GGUF picks the same top token at 95 (the other is a 0.07-nat tie; transformers' own BF16 run matches FP32 at 91) |
| `test-llama-archs -a zaya` | Vulkan and ROCm match the CPU (NMSE 9.6e-8 and 8.1e-14); save and reload bit-exact |
| `tests/serve_e2e.sh`, Q4_K_M | PASS on `--device vulkan` and `--device hrx` |

Q4_K_M (5.17 GiB), by device:

| Device | Prefill (pp512) | Decode (tg128) | Perplexity* |
|---|---|---|---|
| Vulkan0 | 3,437 tok/s | 93.0 tok/s | 21.57 |
| ROCm0 | ~2,400 tok/s | 61.5 tok/s | 21.78 |
| HRX0 | 1,175 tok/s | 25.5 tok/s | 21.55 |

F16 on Vulkan0: 1,138 tok/s prefill, 45.9 tok/s decode, perplexity 20.59.

\* 512-token chunks over this repository's docs (PORTING.md, hrx.md, README.md), for comparing
quants and devices, not models. ROCm's Q4 matmuls quantize activations to 8 bits, hence its
slightly higher figure.

Vulkan decodes fastest, so `auto` and `--device vulkan` stay the default for ZAYA. The engine's
`--device rocm` server is built from ROCmFPX's tree, which has no ZAYA; the ROCm numbers above are
our llama.cpp built with `GGML_HIP=ON` for gfx1151.

What it took, besides the port: CCA's grouped convolution runs as one batched matmul per tap
(as one small matmul per group it held Vulkan decode at 50 tok/s), and the graph avoids what
HRX and ROCm lacked: copies into part of a state row, concatenating strided views, `l2_norm`,
and a two-tap `ssm_conv` kernel on ROCm.
