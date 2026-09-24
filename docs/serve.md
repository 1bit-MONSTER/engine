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
# `1bit serve`: the engine behind one OpenAI-compatible API

The engine runs inside Lemonade ([lemonade.md](lemonade.md)): Lemonade runs
`1bit serve` as a backend, the way it runs `llama-server`. `1bit serve` exposes
nothing but an OpenAI-compatible API, so it also works on its own with any
OpenAI client.

```sh
1bit serve -m <model> [--port 8000] [--host 127.0.0.1]
           [--device auto|npu|vulkan|hrx|rocm|zinc|mlx] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]
           [--prefill-device hrx] [--prefill-min-tokens N] [--lean]
           [--mtp HEAD.gguf] [--mtp-max N] [--mtp-p-min P]
```

One model per process:

| Endpoint | |
|---|---|
| `GET /health`, `GET /v1/health` | 503 while the model loads, then 200 |
| `GET /v1/models` | the one model (`--alias`, else the file or directory name) |
| `POST /v1/chat/completions` | streamed (SSE) or not |
| `POST /v1/completions` | |

## Where the model runs

| Model | `--device` | Runs on |
|---|---|---|
| NPU model directory (`model.q4nx` + `npu/`, docs/npu.md) | `auto`, `npu` | the NPU fast lane, in process |
| `.gguf` | `auto`, `vulkan` | the upstream llama.cpp build's llama-server on `Vulkan0` (docs/vulkan.md); without `ONEBIT_VULKAN`, the HRX build's |
| `.gguf` | `hrx` | the HRX build's llama-server on `HRX0` (docs/hrx.md) |
| `.gguf` | `vulkan --prefill-device hrx` | the HRX build's llama-server decoding on `Vulkan0`, long prompt prefixes prefilled on `HRX0` over one shared KV cache (docs/hrx.md, "Prefill on HRX, decode on Vulkan") |
| ROCmFP4 `.gguf` | `auto`, `vulkan` with `--lean` | the lean (ROCmFPX) build's llama-server on `Vulkan0` (docs/lean.md) |
| `.gguf` | `rocm` | the ROCm build's llama-server on `ROCm0` (ROCmFPX's tree, `ONEBIT_LEAN_ROCM`); ROCmI4 files take its W4A4 path (docs/lean.md) |
| `.gguf` | `zinc` | this build's ZINC (Vulkan, ROCm or CUDA, whichever it was built for; docs/zinc.md) |
| Hugging Face id | `mlx` | lemon-mlx-engine's server, on Apple Silicon (docs/apple.md) |

`chat_template_kwargs.enable_thinking: false` works on every device. The GPU
backends apply the model's own chat template. The NPU route emits what Qwen3's
template does: an empty think block after the assistant prefix.

For a `.gguf` the engine starts that server as a private child on a loopback
port and forwards the OpenAI routes to it, streaming included. Replies carry
the served model name. For ZINC, which rejects foreign model ids, requests go
out without `model`. `auto` means Vulkan for GGUF, the fastest measured device
for standard quants (docs/hrx.md), until the Laya router (docs/laya.md) makes
that choice per request.

HRX needs TheRock's HSA runtime: the distro `libhsa` rejects gfx1151's
PM4-emulation probe, and then HRX registers no device. `serve` sets
`IREE_HAL_AMDGPU_LIBHSA_PATH` itself unless you did. It uses `--hrx-libhsa`,
else the build's copy, else the first one under `/opt/rocm-therock`.

The child binaries default to this build's (`-DONEBIT_HRX`, `-DONEBIT_ZINC`),
then `$ONEBIT_LLAMA_SERVER` / `$ONEBIT_ZINC`, then `llama-server` / `zinc` on PATH.

## Multi-token prediction (`--mtp`)

`--mtp <head.gguf>` turns on llama-server's `draft-mtp` speculative decoding: the model's
own MTP head (Unsloth ships it as `MTP/mtp-<model>-*.gguf`) drafts tokens on the same
device, and the model checks them in one batch. `--mtp-max N` caps the draft length;
`--mtp-p-min P` drafts a token only when the head is at least that sure of it.
It works on the llama.cpp devices (`vulkan`, `hrx`, `rocm`).

Measured through `1bit serve` on Strix Halo, Qwen3.8-27B UD-Q4_K_XL with its Q4_0 MTP
head, decode tok/s on three chat prompts (code / prose / short), 2026-09-24:

| Route | Without `--mtp` | With `--mtp` |
|---|---|---|
| `--device vulkan` (upstream pin) | 12.2 / 12.0 / 12.0 | **35.0 / 28.3 / 28.9** |
| `--device rocm` | 11.8 / 11.8 / 12.2 | 38.8 / 20.6 / 16.9 |
| `--device rocm --mtp-max 3` | | 27.1 / 19.2 / 19.9 |

Vulkan with MTP is the pick: 2.4-2.9x on every prompt, same file, same accuracy (a
drafted token is kept only when the model agrees). ROCm edges it on code only.

The full sweep on Vulkan (draft length x `--mtp-p-min`, code / prose / short; 12.2 / 12.3 / 12.3
without MTP):

| Setting | tok/s | Drafts accepted |
|---|---|---|
| `--mtp` (draft length 3, the default) | 35.9 / 28.2 / 28.8 | 94% / 68% / 75% |
| `--mtp-max 4 --mtp-p-min 0.5` | 38.1 / 28.0 / 26.9 | 93% / 71% / 76% |
| **`--mtp-max 6 --mtp-p-min 0.5`** | **42.0** / 27.3 / 25.2 | 84% / 60% / 72% |
| `--mtp-max 8` | 21-25 / 13-20 / 10-20 | 76-88% / 39-65% / 33-78% |

Use the default for chat and `--mtp-max 6 --mtp-p-min 0.5` for code (3.4x): code is
predictable enough to keep long drafts, prose is not. Draft length 8 collapses on every
prompt. Adding n-gram drafting to MTP gains nothing. Small-active MoE models are the
opposite case: on Qwen3-Coder-30B-A3B (3B active, 88 tok/s on Vulkan) every draft model
tried (Qwen3 0.6B / 1.7B / 4B) was slower than no drafting, even at 82-87% acceptance.

## Many requests at once (`--parallel`)

`--parallel N` gives llama-server N slots (`-np N`): requests that arrive together decode
together, one read of the weights per step for all of them (continuous batching, what
vLLM is built on). `--ctx-size` is split across the slots, so size it for all of them.

Total decode tok/s with 1-16 simultaneous 256-token requests, 16 slots, 2026-09-24:

| Requests | Qwen3.8-27B UD-Q4_K_XL, Vulkan | ... ROCm | Qwen3-Coder-30B-A3B Q4_K_M, Vulkan | ... ROCm |
|---|---|---|---|---|
| 1 | 11.8 | 11.6 | 85.7 | 67.6 |
| 2 | 21.7 | 20.1 | 125.8 | 100.2 |
| 4 | 36.9 | 30.2 | 182.4 | 140.3 |
| 8 | **50.9** | 35.6 | **228.2** | 200.4 |
| 16 | 42.9 | **68.0** | 202.9 | **318.9** |

Vulkan peaks at 8 requests and falls back at 16; ROCm keeps scaling to 16, where it
serves 5.8x (27B) and 3.7x (Coder) the best single stream. Use `--device vulkan` for up
to about 8 concurrent users, `--device rocm --parallel 16` beyond that. Two backends
decoding at once, by comparison, add 18% (docs/lean.md).

## Verified (Strix Halo, 2026-09-23)

`tests/serve_e2e.sh` with Qwen3-0.6B: the Q4_K_M GGUF for the GPU devices and the Q4NX model directory for the NPU. The test checks `/health` 200,
`/v1/models`, a chat that answers "Paris." under the served name, and
streaming:

| Device | Result |
|---|---|
| `npu` | PASS (33 SSE chunks): Qwen3-0.6B NPU model directory on the fast lane |
| `vulkan` | PASS (32 SSE chunks) |
| `hrx` | PASS (32 SSE chunks), with no environment set up |
| `zinc` | PASS (6 SSE chunks) |

ctest runs these as `serve_e2e_<device>` when configured with
`-DONEBIT_SERVE_TEST_GGUF=<gguf>` (and `serve_e2e_mlx` on macOS with
`-DONEBIT_MLX_SERVER`). `smoke_serve` runs everywhere, CI included:
`tests/fake_backend.py` stands in for llama-server. It checks the proxy (`/health`,
`/v1/models`, the reply rename, SSE relay) and that no backend outlives a
SIGKILLed `serve`.
