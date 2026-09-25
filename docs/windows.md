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

`1bit.exe` runs on Windows 10 and later (x64): `1bit serve` with the Vulkan backend, `1bit route` (Laya) and `1bit comfy`. It is cross-built on Linux. There is no NPU lane, HRX, ROCm or ZINC on Windows yet.

```sh
scripts/build-windows.sh <out-dir>     # ~2 min on Strix Halo once the downloads are cached
```

The script writes two single, static executables:
- **`<out-dir>/1bit.exe`** (3.2 MB). It imports only `KERNEL32`, `WS2_32` and the UCRT that ships with Windows 10.
- **`<out-dir>/llama-server.exe`**, the Vulkan pin (`third_party/llama.cpp-vulkan`) built for Windows. It also imports `vulkan-1.dll`, from the GPU driver.

Keep the two side by side: `1bit.exe serve -m model.gguf` finds `llama-server.exe` next to itself. `--llama-server PATH` or `ONEBIT_LLAMA_SERVER` names another one.

## What it downloads

Every download is pinned and checked against its sha256:

| input | why |
| --- | --- |
| llvm-mingw 20260922 (clang 23, UCRT) | the engine is C++26; distribution MinGW GCC 13 is too old |
| PCRE2 10.48 | the tokenizer, linked statically |
| Vulkan-Headers `vulkan-sdk-1.4.357.0` | llama.cpp's Vulkan backend |
| the Vulkan loader's `vulkan-1.def`, same tag | the import library, made with `llvm-dlltool`: no Vulkan SDK or Windows machine needed |
| SPIRV-Headers, same tag | llama.cpp's Vulkan backend |

Host tools: `cmake`, `git`, `curl`, a native C/C++ compiler (llama.cpp's shader generator runs on the build machine) and `glslc`.

## What changed for Windows

- **Backends.** `1bit serve` starts its backends with `CreateProcess`, inside a job object that kills them when its last handle closes. So a backend dies with `1bit.exe` for any reason, as `PR_SET_PDEATHSIG` does on Linux; the Linux and macOS paths are unchanged. Windows has no SIGTERM, so a backend stops with `TerminateProcess`.
- **Model files.** Model and safetensors files are mapped through `npu/file_map.{h,cpp}` (mmap on POSIX, `CreateFileMapping` on Windows).
- **cpp-httplib.** It is built at the Windows 10 API level, with its non-blocking DNS lookup off, because MinGW's headers lack `GetAddrInfoExCancel` and the engine only ever connects to 127.0.0.1.
- **`1bit comfy`.** It spawns `comfyui_cpp.exe` and waits for it, since there is no exec in place on Windows.
- **llama.cpp workaround.** The pinned llama.cpp uses `std::function` in `ggml-vulkan-types.h` without including `<functional>`. libc++ does not include it transitively, so the script adds it from the command line.

## Tested

On Strix Halo under Wine 11, whose Vulkan goes to the host's RADV driver, so on the real Radeon 8060S. Measured 2026-09-25:
- `1bit.exe serve -m Qwen3-0.6B-Q4_K_M.gguf` found `llama-server.exe` beside it, answered "The capital of France is Paris." (132 tok/s) and "Three colors: green, red, and blue.", and streamed a reply as SSE.
- `taskkill /F /IM 1bit.exe` left no `llama-server.exe` behind.
- The Linux build of the same sources passes its tests.

It has not run on a real Windows machine yet.

## Next

- **An ONNX Runtime GenAI backend:** Lemonade's ONNX and hybrid models, and the Ryzen AI NPU on Windows through the Vitis AI execution provider.
- **The NPU lane on Windows XRT.**
