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
| [ggml-org#21412](https://github.com/ggml-org/llama.cpp/pull/21412) (Echo Labs) | Zyphra's Zamba2 (Mamba-2 layers plus shared transformer blocks, per-layer LoRA adapters merged at conversion) and its converter | `5ce79a56` |
| ours: `vulkan: SSM_SCAN for d_state 64` | Vulkan's Mamba-2 scan had pipelines only for a 128/256 state, so Zamba2-2.7B/7B (state 64) ran the scan on the CPU in every layer; two test-backend-ops cases with their shapes | `a32f0d6f` |
| ours: `vulkan: SSM_SCAN for Mamba-1` | Vulkan's scan only handled Mamba-2, so every Mamba-1 layer (Mamba, Falcon-Mamba, Zamba v1, BlackMamba) ran on the CPU; a per-channel kernel with the state in registers (d_state 16), three test-backend-ops cases | `adb2af5d` |
| ours: `model: Zamba v1` | Zyphra's Zamba-7B-v1: Mamba-1 layers split into two heads (per-head x_proj/dt_proj, B/C as scan groups) plus one shared transformer block, stored once; converter de-interleaves in_proj | `5319c0e6` |
| ours: `gguf-py: LlamaHfVocab scores from the merge ranks` | upstream converter bug: models with only a SentencePiece-style BPE `tokenizer.json` (Zamba, Zamba2, and others) got every token score -1000, so merges were picked arbitrarily (Zamba-7B-v1: 318/1500 wikitext lines tokenized like HF); now -rank of the producing merge: 1500/1500 | `ee6e4e0d` |
| ours: `model: BlackMamba` | Zyphra's BlackMamba (1.5B, 2.8B): Mamba-1 layers alternating with a Switch MoE (sigmoid top-1 router with bias, gated erf-GELU experts), biased final LayerNorm; the converter reads the Megatron checkpoint and uses the built-in GPT-NeoX vocab | `f6261222` |

Each upstream PR is reviewed line by line before it is pinned. #28243: it touches only the qwen4exp model, its
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

v0.5.0 + #28243 + #21412 + the d_state-64 scan (`a32f0d6f`); Zamba2 converted with the carried converter (`convert_hf_to_gguf.py --outtype q8_0`), test-backend-ops SSM_SCAN 14/14 on Vulkan0:

| Zamba2 (Q8_0, `llama-bench -p 256 -n 64`) | pp256 | tg64 | chat |
|---|---|---|---|
| 1.2B-instruct | 3166 | 89.8 | "The capital of France is Paris." |
| 2.7B-instruct | 1541 (was 614) | 45.2 (was 4.3: scan on the CPU) | correct |
| 7B-Instruct (2 shared blocks, RoPE) | 523 (was 29) | 17.1 (was 2.1) | correct, 16.7 tok/s |

Mamba-1 scan (`adb2af5d`), `state-spaces/mamba-370m-hf` F16, `llama-bench -p 256 -n 64`: pp256 228 → **6736** tok/s, tg64 25.8 → **173.5** tok/s; greedy completions identical to the CPU-scan build; test-backend-ops SSM_SCAN 17/17.

Zamba v1 (`ee6e4e0d`), `Zyphra/Zamba-7B-v1` Q8_0: teacher-forced top-1 vs transformers FP32 **96/96** on three prompts; greedy continuations identical to the reference's; pp256 543, tg64 14.9 tok/s; 2 graph splits (all compute on Vulkan0: each layer's two-head scan runs on the Mamba-1 Vulkan kernel from `adb2af5d`). GGUFs of Zamba or Zamba2 converted before `ee6e4e0d` have the broken token scores: reconvert them.

BlackMamba (`f6261222`), `Zyphra/BlackMamba-1.5B`: teacher-forced top-1 vs Zyphra's own PyTorch code (FP32, CPU) **96/96** on three prompts, identical greedy text and prompt tokens; F16 pp256 7180 / tg64 250 tok/s, Q8_0 pp256 10136 / tg64 **369** tok/s. No HF `architectures` in its config.json: the converter routes on `mamba_moe_layers`.

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

### ZAYA1-74B-preview

[ZAYA1-74B-preview](https://huggingface.co/Zyphra/ZAYA1-74B-preview) is the same design at 74.8B
parameters and 60 layers. It alternates sliding-window layers (a 4,097-token window, rope theta 1e4)
with full-attention layers (theta 1e7). The converter writes the window, the per-layer pattern and
the second rope base ([llama.cpp #11](https://github.com/1bit-MONSTER/llama.cpp/pull/11)).
ZAYA1-8B has no sliding layers and is unchanged: its Q4_K_M perplexity matches to four decimals,
and its F16 still matches transformers FP32 at 95 of 96 teacher-forced positions.

```
python third_party/llama.cpp/convert_hf_to_gguf.py --remote Zyphra/ZAYA1-74B-preview --outtype q8_0 --outfile zaya1-74b-Q8_0.gguf
build/hrx/llama/bin/llama-quantize --allow-requantize zaya1-74b-Q8_0.gguf zaya1-74b-Q4_K_M.gguf Q4_K_M
1bit serve -m zaya1-74b-Q4_K_M.gguf --device vulkan --ctx-size 8192
```

`--remote` streams the safetensors from Hugging Face, so the 150 GB BF16 checkpoint never has to
fit on disk. The Q8_0 file is 79.7 GB, and requantizing it to Q4_K_M needs `--allow-requantize`.
GGUFs made with `--remote` before llama.cpp #14 have no chat template, because the download
skipped `chat_template.jinja`. Reconvert them, or serve them with `--chat-template-file`.

Verified on Strix Halo, HF revision `bc8eb466`, Q4_K_M (42.6 GiB, 45.7 GB):

| Check | Result |
|---|---|
| `tests/serve_e2e.sh`, `--device vulkan --ctx-size 8192` | PASS |
| Chat with the model's own template (code, prose, a one-line fact) | coherent answers |
| `llama-bench`, Vulkan0 | prefill (pp512) 505.8 tok/s, decode (tg128) 35.4 tok/s |

There is no reference check for the 74B: a 74B FP32 transformers run needs about 300 GB of
memory, which does not fit on Strix Halo.

### ZAYA1-base and ZAYA1-reasoning-base: legacy checkpoints

ZAYA1-base, ZAYA1-reasoning-base and Zyphra's `*-legacy` repos keep the original Megatron-style
checkpoint:
- attention and MoE are separate half-layers;
- sizes live in per-layer lists;
- experts are stored one by one.

transformers cannot load them. The converter maps them to the transformers layout itself
([llama.cpp #14](https://github.com/1bit-MONSTER/llama.cpp/pull/14)). Converting
`Zyphra/ZAYA1-8B-legacy` gives a GGUF whose 1,283 tensors are byte-identical to the one from
`Zyphra/ZAYA1-8B`, and the model metadata is identical too.
