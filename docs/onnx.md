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
| NPU, hybrid (NPU + iGPU) | AMD's Ryzen AI Software (the Vitis AI execution provider) | its own licence and download; set `RYZENAI_INSTALL_PATH` to it before building |

The engine does not ship Ryzen AI Software. With it installed, the same build serves Lemonade's `-npu` and `-hybrid` ONNX models.

## Build

`scripts/build-onnx.sh <prefix>` (run by `-DONEBIT_ONNX=ON`):
- **Libraries.** It fetches the two MIT releases, or uses `$RYZENAI_INSTALL_PATH`.
- **Build.** It builds ryzenai-server against them.
- **Output.** It puts the binary and the libraries it loads in `<prefix>`. The binary carries `DT_RPATH $ORIGIN`, so ONNX Runtime GenAI finds `libonnxruntime.so` beside it when it loads it at run time.

`ONEBIT_ONNX_SERVER` or `--device onnx` with another `ryzenai-server` on `PATH` overrides the build's.

## Measured

On Strix Halo, 2026-09-25, with `amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx` (CPU, int4) and `1bit serve -m <dir>` (no `--device`):
- **Chat.** "The capital of France is Paris." and a list of three colors, at 125–129 tok/s decode.
- **Streaming.** A reply streamed as SSE.
- **Model list.** `/v1/models` names the model with device `onnx`.
- **Shutdown.** Stopping `1bit serve` stopped the backend.

llama-server's extra routes (`/tokenize`, `/slots`, …) answer 501 on `--device onnx`.

## Next

- **Windows.** `ryzenai-server.exe` in `scripts/build-windows.sh`: ONNX Runtime GenAI's Windows builds include DirectML, the GPU on Windows.
- **The Ryzen AI NPU.** On a machine with Ryzen AI Software installed.
