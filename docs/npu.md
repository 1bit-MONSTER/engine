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
# NPU engine

The NPU engine is in progress: correctness first, speed next. Its kernels, built from source,
match an fp64 reference on Qwen3-0.6B, 1.7B and 4B. The layer kernel and lm head reach cosine
similarity 0.9999 or better at every tested position, and greedy decoding matches the reference
token for token.
They are not merged into this repository yet. The release carries the NPU runtime, compiled into
`1bit`, and the `npu` backend of Lemonade's `onebit` recipe; until the kernels land, a release
package serves GGUF models on HRX, not NPU models ([releases.md](releases.md#the-npu-engine)).

The NPU runtime and the NPU kernels are proprietary and closed source
([releases.md](releases.md#private-components)). They ship only as compiled binaries: the runtime
is compiled into `1bit`, and the kernels ship as compiled, encrypted artifacts that `1bit` verifies
and decrypts in memory at load (a wrong key fails cleanly, and nothing phones home). Their source,
build and method are not in this repository or in any release. This page covers the public parts:
the model format, how `1bit serve` runs an NPU model, and how to build.

## A model directory

```
<model dir>/
  model.q4nx             weights (safetensors layout, Q4NX tiles: "Q4NX chunks" below)
  config.json            Hugging Face config: dimensions, rope_theta, eos_token_id
  tokenizer.json         Hugging Face tokenizer
  npu/                   the model's NPU kernels
```

`1bit serve -m <model dir>` serves such a directory on the NPU behind the OpenAI-compatible API
([serve.md](serve.md)); inside Lemonade, that is what its `onebit` backend runs (`--onebit npu`).

### Q4NX chunks

Q4NX is [FastFlowLM](https://github.com/ROCm/FastFlowLM)'s NPU model format (MIT), and its models on
Hugging Face (`FastFlowLM/*-NPU2`) are Q4NX directories. A Q4NX weight is a grid of 32-row x
256-column tiles, one chunk each, row-major over the grid. The chunk kind is the last dimension of
the tensor's shape: `[tiles, 5120]`, or `[tile rows, tile cols, chunk bytes]` (`npu/q4nx.h` has the
byte layouts).

| Chunk | Kind | Weight | Seen in |
|---|---|---|---|
| 5120 B | q4_1: bf16 scale and zero per 32 columns of a row | `code * scale + zero` | Qwen3, Llama, Gemma, Phi models |
| 4736 B | Q4_K: u8 scale and min per 32 columns, bf16 `S`, `M` per row | `S * scale * code + M * min` | Qwen3.5-4B projections |
| 8704 B | Q8: bf16 scale per 32 columns, int8 codes | `d * code` | Qwen3.5-4B `lm_head` and embedding |

**Q8 has two code layouts under the same chunk size.** Qwen3.5-4B's lm_head (tied to the
embedding) stores code (row r, column k) at byte `k*32 + r`. Qwen3.6-35B-A3B's Q8 tensors
(untied lm_head, projections) store it at `(r/16)*4096 + (k/32)*512 + (k%32)*16 + r%16`.
Both containers report `flm_version` 1.0.3, and neither carries metadata that tells them
apart. Decoding tile (0, 0) of each lm_head and comparing with a GGUF of the same model
gives correlation 0.9998 with the right layout and about 0.01 with the other. `npu/q4nx`
decodes the first layout.

`npu/q4nx.h` decodes all three, and repacks Q4_K into q4_1 (exact apart from rounding
`S * scale` and `M * min` to bf16), so code that reads q4_1 runs Q4_K weights unchanged. Q8 does
not fit q4_1. `tests/npu_q4nx_test.cpp` checks the decoders bit for bit against reference decoders
on real Qwen3.5-4B and Qwen3-0.6B tiles (committed in `tests/golden/q4nx`), and the repack against
its rounding bound.

### From a GGUF

`tools/gguf_to_q4nx.py pack <model.gguf> <out dir> --tokenizer <dir>` writes a model
directory's `model.q4nx` and `config.json` from a GGUF (qwen3 only for now), and copies
`tokenizer.json` from `--tokenizer`. It writes every projection and `lm_head` as q4_1
tiles and the norms and embedding as bf16. A tied `lm_head` is made from `token_embd`.

| GGUF type | To q4_1 | Error |
|---|---|---|
| Q4_0 | scale `d`, zero `-8d`, codes unchanged | bf16 rounding of scale and zero |
| Q4_1 | scale `d`, zero `m`, codes unchanged | bf16 rounding of scale and zero |
| Q4_K | per 32-column sub-block: scale `d*sc`, zero `-dmin*mn`, codes unchanged (a Q4_K super-block is one tile row's 256 columns) | bf16 rounding of scale and zero |
| Q5_K, Q6_K, Q8_0, F16, BF16, F32 | dequantized, then min/max q4_1 per 32 columns | lossy |

`tools/gguf_to_q4nx.py verify <model.gguf> <out dir>` decodes every tensor it wrote and
compares it with the GGUF, dequantized. For the exact types, every weight must lie
within the bf16 rounding bound. The requantized tensors show their error.

On `Qwen3-0.6B-Q4_K_M.gguf` (unsloth), all 311 tensors pass. The Q4_K tensors have 0
weights over the bound. The Q6_K tensors (half of `v_proj` and `down_proj`, and
`lm_head` from the Q6_K embedding) requantize to cosine 0.9969 and relative RMS error
0.079. Served on the NPU, the repacked model decoded 24/24 greedy tokens the same as a CPU
reference of the repacked weights, with every step's logits at cosine 0.99995 or better.

`1bit serve -m <model.gguf> --device npu` repacks a GGUF itself, caches the result under
`~/.cache/1bit/q4nx/`, and serves it. GGUFs of six architectures (Qwen2.5, Qwen3 MoE,
Qwen3.6-35B-A3B, MiniCPM4/5, GLM-4.7-Flash) answer correctly this way, on a generic per-op path
that is not yet fast (#179, #181, #209). On a release, GGUFs belong on HRX.

**Declared scales are checked.** The repack (`scripts/repack_gguf.py`) fails with `rc=1` if the
GGUF declares a known-critical scale-like value that the repacked `config.json` drops:
`embedding_scale`, `residual_scale`, `logit_scale`, `scale_emb`, `scale_depth`,
`dim_model_base`, `final_logit_softcapping`, `attn_logit_softcapping`. MiniCPM4-8B decoded
punctuation soup because three of them were dropped, and no argmax-level check can see the
third, which moves only the probabilities. Any other scale-like key (`*scale*`, `*softcap*`,
`*mup*`) is warned about, not fatal. `scripts/check_repack_config.py` runs the same scan on its
own, with a negative control.

A cached directory is checked again on every hit, with the same metadata-only scan, and it is
rebuilt if the scan fails or if its `repack-stamp.txt` (converter path and commit, architecture,
repack script) names a different converter than this build would use (`repack_gguf.py
--check-stamp <dir>`: exit 0 current, 1 stale, 2 cannot tell). If the checker or the converter
cannot be found, serve warns once per process and reuses the directory, so a missing script
cannot stop serving. A Q4NX directory served as it is (`1bit serve -m <Q4NX dir>`), such as a
published FastFlowLM model, is trusted as published.

## Serving

The 1bit NPU engine runs the model on the XDNA 2 NPU. With a release package that carries the NPU
kernels (dense Qwen3: 0.6B, 1.7B and 4B):

```
./1bit serve -m <Qwen3 NPU model dir> --port 8000
```

In Lemonade, the `onebit` recipe's `npu` backend serves the same directories (`--onebit npu`).
The model directory holds the weights only; the kernels ship inside the package.

- **Tokenizer:** `npu/tokenizer.cpp` reads `tokenizer.json` directly (PCRE2 for the
  pre-tokenizer regex). Added tokens with `special: false`, such as Qwen3's `<think>` and
  `</think>`, stay whole, as Hugging Face `tokenizers` does.
  `tests/data/qwen3_tokenizer_golden.json` holds 14 cases encoded by `tokenizers` 0.23.1, and
  all 14 match, ids and round trip.
- **Endpoints:** `/v1/health` (503 while loading), `/v1/models`, `/v1/chat/completions`
  (streaming or not) and `/v1/completions`.
- **Prompt:** ChatML, Qwen2/Qwen3's template for text messages.
- **Decoding:** greedy, with a 1.1 repetition penalty over the last 64 ids by default.
  `repetition_penalty` in the request overrides it, and 1 gives plain greedy.
- **One request at a time:** the model has one KV cache.
- **Repeatable:** every generation starts from a zeroed KV cache (#400), so identical greedy
  requests give identical text, whatever ran before them.

## Add-on routes

The engine keeps a hook for closed-source NPU add-ons (`npu/private_route.h`). Qwen3.6-35B-A3B
(`model_type` `qwen3_5_moe`) runs on the NPU through one, experimentally; it is not in this
repository and not in a release.

- `-DONEBIT_NPU_PRIVATE=<add-on checkout>` (with `-DONEBIT_NPU=ON`) builds that checkout's
  `addons/` into `1bit`. It must define the target `onebit_npu_private`, which holds
  `register_private_addon()`; `1bit` calls it at startup.
- The add-on registers a route per `model_type` (whether it can serve a directory, and a
  session that generates from token ids) and may register `1bit` subcommands. `1bit serve`
  uses the route for that model type; `--npu-opt KEY=VALUE` passes options through to it.
- A route can be `optional`: an opt-in route for a `model_type` the engine also serves. When it
  declines a directory (its `unavailable()` gives a reason), the engine serves the directory
  exactly as it would without the add-on.
- Without the add-on, `1bit serve -m <35B dir> --device npu` exits saying the route is not part
  of this build.

An add-on may also register an encoder for the Laya router (`laya/encoder.h`,
[laya.md](laya.md#the-encoder-on-the-npu-add-on)). `tests/npu_private_route_test.cpp` (ctest
`npu_private_route`, in CI) checks the registry and that message.

## The runtime

The engine reaches the NPU runtime only through its own public header, `npu/npu_rt_api.h`: the
calls the engine's NPU code makes, each class an opaque handle. The runtime itself is not in this
repository; a build from this repository alone has no NPU lane. `ONEBIT_NPU=ON` takes one of:

```
# the public binary drop (docs/releases.md, "The public binary drop")
cmake -B build -G Ninja -DONEBIT_HRX=ON -DONEBIT_NPU=ON -DONEBIT_BINARY_DROP=<drop dir or URL>
# a prebuilt runtime library that implements npu/npu_rt_api.h
cmake -B build -G Ninja -DONEBIT_NPU=ON -DONEBIT_NPU_RT_LIB=<path>/libnpu_rt_api.a
```

- `ONEBIT_NPU_HOST` (the model reader, Q4NX, packing, the tokenizer) needs no runtime and builds
  everywhere, including public CI, where its tests run.
- The release builds the runtime from a pinned commit named in `config/npu-runtime.config`
  ([releases.md](releases.md#private-components)).

The NPU needs the `amdxdna` kernel driver. `third_party/xdna-driver` pins upstream
[amd/xdna-driver](https://github.com/amd/xdna-driver) as the driver version the engine is checked
against (the kernel's own copy also works); nothing from the pin is built into or shipped with
the release. `.github/workflows/bump-xdna.yml` moves the pin when run by hand, because a driver
move means rebuilding the module on Strix Halo and checking the NPU there first.
