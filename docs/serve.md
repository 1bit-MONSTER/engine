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
           [--device auto|npu|vulkan|hrx|zinc|mlx] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]
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
