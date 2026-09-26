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

# ONNX Runtime GenAI

`1bit serve --device onnx` serves ONNX Runtime GenAI models: a directory with `genai_config.json`, `model.onnx` and the tokenizer, the format of Lemonade's ONNX models. The engine runs [ryzenai-server](https://github.com/lemonade-sdk/ryzenai-server) (MIT, pinned in `third_party/ryzenai-server`) as a child process, the way it runs llama-server, and forwards the OpenAI API to it. A directory with a `genai_config.json` goes to `--device onnx` on its own:

```sh
cmake -B build -DONEBIT_ONNX=ON          # builds ryzenai-server into build/onnx/
build/1bit serve -m ~/models/oga/Qwen2.5-0.5B-Instruct-cpu-onnx
```

## Where it runs

The execution mode comes from the model's `genai_config.json`:

| mode | runtime | how to get it |
| --- | --- | --- |
| CPU | Microsoft's ONNX Runtime GenAI 0.11.2 + ONNX Runtime 1.23.2 (MIT) | `scripts/build-onnx.sh`: fetched pinned, sha256-checked |
| GPU (WebGPU over Vulkan) | the same GenAI, with ONNX Runtime 1.23.2 built with its WebGPU provider | `ONNX_WEBGPU=1 scripts/build-onnx.sh`, see [the GPU on Linux](#the-gpu-on-linux-webgpu) |
| NPU, hybrid (NPU + iGPU) | AMD's Ryzen AI Software (the Vitis AI execution provider) | its own licence and download; set `RYZENAI_INSTALL_PATH` to it before building |

The engine does not ship Ryzen AI Software. With it installed, the same build serves Lemonade's `-npu` and `-hybrid` ONNX models.

## Build

`scripts/build-onnx.sh <prefix>` (run by `-DONEBIT_ONNX=ON`):
- **Libraries.** It fetches the two MIT releases, or uses `$RYZENAI_INSTALL_PATH`.
- **Build.** It builds ryzenai-server against them.
- **Output.** It puts the binary and the libraries it loads in `<prefix>`. The binary carries `DT_RPATH $ORIGIN`, so ONNX Runtime GenAI finds `libonnxruntime.so` beside it when it loads it at run time.

`ONEBIT_ONNX_SERVER` or `--device onnx` with another `ryzenai-server` on `PATH` overrides the build's.

## The GPU on Linux: WebGPU

What DirectML is on Windows, ONNX Runtime's WebGPU execution provider is on Linux: it runs on
Dawn, which drives the Radeon through Vulkan (RADV), so it needs no ROCm. Microsoft publishes no
native Linux build of it (its Python wheel links it statically), so `ONNX_WEBGPU=1` builds ONNX
Runtime 1.23.2 from source, pinned to its commit, with `--use_webgpu`. The build needs Node.js
(`NODE_DIR` if `node` is not on `PATH`), takes about 20 minutes and about 6 GB. It places Dawn's
licence and ONNX Runtime's third-party notices beside the library.

```sh
ONNX_WEBGPU=1 scripts/build-onnx.sh build/onnx
```

A model runs on the GPU when its `genai_config.json` asks for `webgpu`, which is what ONNX Runtime
GenAI's model builder writes with `-e webgpu`:

```sh
python onnxruntime-genai/src/python/py/models/builder.py -i <HF checkpoint> -o <out> -p int4 -e webgpu
```

Measured on Strix Halo (Radeon 8060S), 2026-09-26, 128 tokens through `ryzenai-server`:

| Model (int4, built with the v0.11.2 builder) | WebGPU decode | CPU decode |
| --- | --- | --- |
| Qwen3-4B | 52.9 tok/s | 26.8 tok/s |
| Qwen2.5-0.5B-Instruct | 164 tok/s | 120 tok/s |

- **Placement.** ONNX Runtime put 292 of Qwen2.5-0.5B's nodes on WebGPU and 6 (attention-mask shape
  arithmetic) on the CPU.
- **The GPU was in use.** During generation the process had `/dev/dri/renderD128` open and the
  Radeon Vulkan driver loaded.
- **Same answers.** Both devices gave the same text.
- **`tests/serve_e2e.sh` on `--device onnx`.** PASS on the WebGPU build (Qwen2.5-0.5B-Instruct).
- **Qwen3's thinking.** ryzenai-server does not pass `enable_thinking` to the chat template, so
  Qwen3 thinks first; `/no_think` in the prompt turns it off.

## Measured

On Strix Halo, 2026-09-25, with `amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx` (CPU, int4) and `1bit serve -m <dir>` (no `--device`):
- **Chat.** "The capital of France is Paris." and a list of three colors, at 125–129 tok/s decode.
- **Streaming.** A reply streamed as SSE.
- **Model list.** `/v1/models` names the model with device `onnx`.
- **Shutdown.** Stopping `1bit serve` stopped the backend.

llama-server's extra routes (`/tokenize`, `/slots`, …) answer 501 on `--device onnx`.

## Next

- **Windows.** Done for the CPU: `scripts/build-windows.sh` builds `ryzenai-server.exe` ([Windows](windows.md)). The GPU on Windows would need ONNX Runtime's DirectML builds, whose `DirectML.dll` is under Microsoft's redistributable licence rather than MIT; the WebGPU provider (above) also runs on Windows, over D3D12, without it.
- **`-DONEBIT_ONNX_WEBGPU`.** A CMake switch for the WebGPU build, so `-DONEBIT_ONNX=ON` can build it without the environment variable.
- **The Ryzen AI NPU.** On a machine with Ryzen AI Software installed.
