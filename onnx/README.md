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
# onnx/ — ONNX Runtime in-process (plugin EP + `1bit ort-info`)

Optional, off by default. `-DONEBIT_ORT=ON` links the pinned ONNX Runtime into
the engine and adds `1bit ort-info`; OFF builds contain no ORT code path.

## The pin

| item | value |
|---|---|
| ONNX Runtime | **1.23.2** (OrtApi v23) |
| package | `libonnxruntime1.23` / `libonnxruntime-dev` `1.23.2+dfsg-6ubuntu1` (Ubuntu 26.04 *resolute*) |
| library | `/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23.2` |
| sha256 | `9c8f6112e9dcf09ed4e621a1d0a7b6ba49575c7d09f27a652d24a22c8564b13c` |
| headers | `/usr/include/onnxruntime/` (same package, so header and binary are one version) |
| discovery | `pkg-config libonnxruntime` — a vendored build that ships `libonnxruntime.pc` is found the same way |
| licence | MIT, `THIRD_PARTY_NOTICES.md` |

Verify the pin on a new box:

```sh
sha256sum /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23.2
pkg-config --modversion libonnxruntime
```

## Build and run

```sh
# standalone probe: no HIP, no submodules, seconds
cmake -S onnx -B onnx/build && cmake --build onnx/build -j
ctest --test-dir onnx/build --output-on-failure
./onnx/build/ort_info

# in the engine (adds the `1bit ort-info` subcommand)
cmake -S . -B build -DONEBIT_ORT=ON   # plus the usual engine flags
cmake --build build --target onebit
./build/1bit ort-info
```

Expected shape of the output:

```
onebit ort-info
  ortv    : 1.23.2
  api     : v23 (compile-time ORT_API_VERSION, negotiated above)
  lib     : /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23
  builtin : 3
    [0] DnnlExecutionProvider
    [1] XnnpackExecutionProvider
    [2] CPUExecutionProvider
  eproster: 1 (partition order; first EP gets the node, no tie-break)
    [0] CPUExecutionProvider   vendor=Microsoft  device=CPU
```

`builtin` is what the shipped `.so` was compiled with; `eproster` is what a
session can actually partition to, **in priority order**. Registration order is
strict priority with no tie-break, so our plugin EP must be registered first and
CPU last.

## Why a probe and not a session

`GetApi(ORT_API_VERSION)` returning non-null is the whole ABI check: it fails
loudly when the header we compiled against is newer than the loaded library,
which is the one mistake that otherwise corrupts memory instead of erroring.
The roster comes from `GetEpDevices`, the same call the session's partitioner
uses.

## Files

- `ort_info.cpp` — `onebit_ort_info_main()`, the probe body (no `main`, so the
  single binary can dispatch to it).
- `ort_info_main.cpp` — standalone `main()` for the `ort_info` target/ctest only.
- `CMakeLists.txt` — works both as an in-tree subdirectory and standalone.
