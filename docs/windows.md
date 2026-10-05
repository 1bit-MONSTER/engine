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

# Windows

`1bit.exe` runs on Windows 10 and later (x64): `1bit serve` with GGUF models on the CPU (`--device cpu`, which `auto` picks) and ONNX Runtime GenAI models (`--device onnx`, through the `ryzenai-server.exe` Lemonade installs), and `1bit route` (Laya). It is cross-built on Linux. There is no GPU route on Windows for now: the engine's GPU route is HRX, which has no Windows build yet, and the engine builds no Vulkan (RFC #213 stage 3). There is no NPU lane on Windows yet.

```sh
scripts/build-windows.sh <out-dir>     # ~2 min on Strix Halo once the downloads are cached
```

The script writes:
- **`<out-dir>/1bit.exe`** (3.2 MB). It imports only `KERNEL32`, `WS2_32` and the UCRT that ships with Windows 10.
- **`<out-dir>/llama-server.exe`**, the engine's llama.cpp pin (`third_party/llama.cpp`, the tree the HRX build uses) built for Windows with `GGML_HRX=OFF` and `GGML_VULKAN=OFF`, for the CPU. Until RFC #213 stage 3 it came from the upstream Vulkan pin (`third_party/llama.cpp-vulkan`), now removed.

The engine builds no ONNX server for Windows. `--device onnx` uses the `ryzenai-server.exe` that Lemonade installs for its `ryzenai-server` backend (Ryzen AI's ONNX Runtime GenAI, [ONNX Runtime](onnx.md)): `1bit.exe` looks under `$LEMONADE_CACHE_DIR`, else `%USERPROFILE%\.cache\lemonade`, in `bin\ryzenai-server\npu`, then for `ryzenai-server.exe` on `PATH`.

Keep them together: `1bit.exe serve -m model.gguf` finds `llama-server.exe` next to itself. `1bit.exe serve -m <dir with genai_config.json>` finds Lemonade's `ryzenai-server.exe`. `--llama-server PATH`, `ONEBIT_LLAMA_SERVER` and `ONEBIT_ONNX_SERVER` name other ones.

## What it downloads

Every download is pinned and checked against its sha256:

| input | why |
| --- | --- |
| llvm-mingw 20260922 (clang 23, UCRT) | the engine is C++26; distribution MinGW GCC 13 is too old |
| PCRE2 10.48 | the tokenizer, linked statically |

Host tools: `cmake`, `git` and `curl`.

## What changed for Windows

- **Backends.** `1bit serve` starts its backends with `CreateProcess`, inside a job object that kills them when its last handle closes. So a backend dies with `1bit.exe` for any reason, as `PR_SET_PDEATHSIG` does on Linux; the Linux and macOS paths are unchanged. Windows has no SIGTERM, so a backend stops with `TerminateProcess`.
- **Model files.** Model and safetensors files are mapped through `npu/file_map.{h,cpp}` (mmap on POSIX, `CreateFileMapping` on Windows).
- **cpp-httplib.** It is built at the Windows 10 API level, with its non-blocking DNS lookup off, because MinGW's headers lack `GetAddrInfoExCancel` and the engine only ever connects to 127.0.0.1.
- **ONNX.** Until 2026-10-04 the script also cross-built `ryzenai-server.exe` against Microsoft's ONNX Runtime GenAI 0.11.2 and ONNX Runtime 1.23.2 `win-x64` releases. Lemonade ships `ryzenai-server.exe` itself, so the engine now uses that one.

## Tested

On Strix Halo under Wine 11:
- **CPU route (2026-10-01).** The package built without Vulkan. `llama-server.exe` imports only `KERNEL32` and the UCRT. `1bit.exe serve -m Qwen3-0.6B-Q4_K_M.gguf` with no `--device` chose `cpu` and started `llama-server.exe` beside it. It answered "The capital of France is Paris." (70 tok/s, on the CPU) and "Three colors are: red, blue, and green." Stopping it left no `llama-server.exe` behind.
- **Vulkan route (2026-09-25, before the GPU route went).** The same model answered at 132 tok/s on the Radeon 8060S through Wine's Vulkan, and `taskkill /F /IM 1bit.exe` left no `llama-server.exe` behind.
- **ONNX route (2026-09-25, with the engine's own `ryzenai-server.exe`, since replaced by Lemonade's).** `1bit.exe serve -m <Qwen2.5-0.5B CPU ONNX dir>` (no `--device`) chose `onnx`, started `ryzenai-server.exe` beside it and answered "The capital of France is Paris." (`ryzenai-server.exe` alone: 61 tok/s under Wine). After `taskkill /F`, nothing was left running.
- The Linux build of the same sources passes its tests.

It has not run on a real Windows machine yet.

## Next

- **ONNX on the Windows GPU (DirectML)**, once the `DirectML.dll` licence question is settled. The Ryzen AI NPU through Ryzen AI Software (its own licence, installed by the user).
- **The NPU lane on Windows XRT.**
