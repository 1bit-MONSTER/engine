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

`1bit.exe` runs on Windows 10 and later (x64): `1bit serve` with the Vulkan backend and with ONNX Runtime GenAI models (`--device onnx`, on the CPU), `1bit route` (Laya) and `1bit comfy`. It is cross-built on Linux. There is no NPU lane, HRX, ROCm or ZINC on Windows yet.

```sh
scripts/build-windows.sh <out-dir>     # ~2 min on Strix Halo once the downloads are cached
```

The script writes:
- **`<out-dir>/1bit.exe`** (3.2 MB). It imports only `KERNEL32`, `WS2_32` and the UCRT that ships with Windows 10.
- **`<out-dir>/llama-server.exe`**, the Vulkan pin (`third_party/llama.cpp-vulkan`) built for Windows. It also imports `vulkan-1.dll`, from the GPU driver.
- **`<out-dir>/ryzenai-server.exe`** (the ONNX backend, [ONNX Runtime](onnx.md)), with Microsoft's `onnxruntime-genai.dll`, `onnxruntime.dll` and `onnxruntime_providers_shared.dll` beside it.

Keep them together. `1bit.exe serve -m model.gguf` finds `llama-server.exe` next to itself, and `1bit.exe serve -m <dir with genai_config.json>` finds `ryzenai-server.exe`. `--llama-server PATH`, `ONEBIT_LLAMA_SERVER` and `ONEBIT_ONNX_SERVER` name other ones.

## What it downloads

Every download is pinned and checked against its sha256:

| input | why |
| --- | --- |
| llvm-mingw 20260922 (clang 23, UCRT) | the engine is C++26; distribution MinGW GCC 13 is too old |
| PCRE2 10.48 | the tokenizer, linked statically |
| Vulkan-Headers `vulkan-sdk-1.4.357.0` | llama.cpp's Vulkan backend |
| the Vulkan loader's `vulkan-1.def`, same tag | the import library, made with `llvm-dlltool`: no Vulkan SDK or Windows machine needed |
| SPIRV-Headers, same tag | llama.cpp's Vulkan backend |
| ONNX Runtime GenAI 0.11.2 `win-x64` and ONNX Runtime 1.23.2 `win-x64` (Microsoft, MIT) | ryzenai-server.exe (CPU). The DirectML builds would add the GPU, but `DirectML.dll` is under Microsoft's redistributable licence, not MIT, so they are not fetched. |

Host tools: `cmake`, `git`, `curl`, a native C/C++ compiler (llama.cpp's shader generator runs on the build machine) and `glslc`.

## What changed for Windows

- **Backends.** `1bit serve` starts its backends with `CreateProcess`, inside a job object that kills them when its last handle closes. So a backend dies with `1bit.exe` for any reason, as `PR_SET_PDEATHSIG` does on Linux; the Linux and macOS paths are unchanged. Windows has no SIGTERM, so a backend stops with `TerminateProcess`.
- **Model files.** Model and safetensors files are mapped through `npu/file_map.{h,cpp}` (mmap on POSIX, `CreateFileMapping` on Windows).
- **cpp-httplib.** It is built at the Windows 10 API level, with its non-blocking DNS lookup off, because MinGW's headers lack `GetAddrInfoExCancel` and the engine only ever connects to 127.0.0.1.
- **`1bit comfy`.** It spawns `comfyui_cpp.exe` and waits for it, since there is no exec in place on Windows.
- **llama.cpp workaround.** The pinned llama.cpp uses `std::function` in `ggml-vulkan-types.h` without including `<functional>`. libc++ does not include it transitively, so the script adds it from the command line.
- **ryzenai-server's CMake** is used unmodified through a wrapper project:
  - the wrapper clears its MSVC-only `/SUBSYSTEM:CONSOLE` link flag (MinGW links console programs by default);
  - a one-line `Wbemidl.h` covers MinGW's lowercase `wbemidl.h`;
  - the MinGW-built exe links Microsoft's MSVC-built `onnxruntime-genai.dll` through its import library.

## Tested

On Strix Halo under Wine 11, whose Vulkan goes to the host's RADV driver, so on the real Radeon 8060S. Measured 2026-09-25:
- `1bit.exe serve -m Qwen3-0.6B-Q4_K_M.gguf` found `llama-server.exe` beside it, answered "The capital of France is Paris." (132 tok/s) and "Three colors: green, red, and blue.", and streamed a reply as SSE.
- `taskkill /F /IM 1bit.exe` left no `llama-server.exe` behind.
- `1bit.exe serve -m <Qwen2.5-0.5B CPU ONNX dir>` (no `--device`) chose `onnx`, started `ryzenai-server.exe` beside it and answered "The capital of France is Paris." (`ryzenai-server.exe` alone: 61 tok/s under Wine). After `taskkill /F`, nothing was left running.
- The Linux build of the same sources passes its tests.

It has not run on a real Windows machine yet.

## Next

- **ONNX on the Windows GPU (DirectML)**, once the `DirectML.dll` licence question is settled. The Ryzen AI NPU through Ryzen AI Software (its own licence, installed by the user).
- **The NPU lane on Windows XRT.**
