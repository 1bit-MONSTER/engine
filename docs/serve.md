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

The engine is a backend that a host launches, the way Lemonade launches
`llama-server`. It exposes nothing but an OpenAI-compatible API. (The embedded
Lemonade of step 1 is on its way out; see PORTING.md.)

```sh
1bit serve -m <model> [--port 8000] [--host 127.0.0.1]
           [--device auto|npu|vulkan|hrx|zinc] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH]
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
| `.gguf` | `auto`, `vulkan` | this build's llama-server on `Vulkan0` |
| `.gguf` | `hrx` | this build's llama-server on `HRX0` |
| `.gguf` | `zinc` | this build's ZINC (Vulkan, ROCm or CUDA, whichever it was built for; docs/zinc.md) |

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

`tests/serve_e2e.sh` with Qwen3-0.6B Q4_K_M. The test checks `/health` 200,
`/v1/models`, a chat that answers "Paris." under the served name, and
streaming:

| Device | Result |
|---|---|
| `vulkan` | PASS (32 SSE chunks) |
| `hrx` | PASS (32 SSE chunks), with no environment set up |
| `zinc` | PASS (6 SSE chunks) |

ctest runs these as `serve_e2e_<device>` when configured with
`-DONEBIT_SERVE_TEST_GGUF=<gguf>`.
