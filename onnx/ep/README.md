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
# libonebit_ep.so — 1bit plugin execution provider for ONNX Runtime

A plugin EP, not a fork: ONNX Runtime 1.23.2 loads this `.so` at runtime through
`RegisterExecutionProviderLibrary`, so ORT itself is never patched, rebuilt or shipped by us.
Every ORT entry point is resolved from the `OrtApi`/`OrtEpApi` structs handed over at load time.

## What it runs today

float32 `MatMul` (rank 2 and 3, batched, `B` rank-2 broadcast), through the single kernel seam in
`onebit_ep_kernel.cpp`. Everything else falls through to whatever EP follows in the session's
registration order.

The seam has two backends. The host loop is always there. With `-DONEBIT_EP_NPU=ON` the multiply
first goes to the NPU through `npu/ort/onebit_npu_gemm.*`, which drives the same `I8Ctx` sequence
the forward route uses (`packB` int8 per column, activations per row at `amax/127`, one full ELF
per projection shape, dequant `Cm*as*gs`) — not a second GEMM. A refusal (device lock held by
another process, no kernel for this `K x N`, any XRT failure) is written to the trace as
`npu_fallback k=… n=… why=…` and the host loop keeps the session working; the graph never fails
because the device was busy.

## The NPU backend (deliverable 3)

Two NPU trees are live and the option takes either, because they do not share a layout or a
runtime:

```sh
# the private runtime tree (what the shipped engine builds against on strixhalo)
cmake -S onnx -B build -DONEBIT_EP_NPU=ON \
  -DONEBIT_NPU_ENGINE_INCS="$W/npu/forward/engine-src;$W/npu" \
  -DONEBIT_NPU_RT_DIR=/path/to/npu_rt      # holds npu_rt.cpp + npu_rt_api.cpp

# in this repo the NPU tree is npu/ (I8Ctx in npu/forward/engine-src) and the runtime
# still comes from ONEBIT_NPU_RT_DIR: -DONEBIT_EP_NPU=ON alone is a hard error.
```

The kernel is built for one `K` and one `N`, so the device needs a matching full ELF:

```sh
export ONEBIT_EP_NPU_ELF=/path/full_i8_O_K1024_N1024_m16.elf
export ONEBIT_EP_NPU_M=16 ONEBIT_EP_NPU_K=1024 ONEBIT_EP_NPU_N=1024
ctest --test-dir build -R onebit_npu_gemm_probe --output-on-failure   # skips if the device is busy
ONEBIT_EP_PROOF_M=4 ONEBIT_EP_PROOF_K=1024 ONEBIT_EP_PROOF_N=1024 \
  ./build/ep/onebit_ep_proof ./build/ep/libonebit_ep.so
```

`onebit_ep_proof` prints `backend=npu|host`; the tolerance follows it (1e-5 host, 2e-2 for the
int8 device path, where the difference is quantisation noise). The device lock is
`/tmp/1bit-npu-device.lock`, taken **non-blocking** and held for the process: if the owner's
forward serve has it, this EP does not queue behind it, it uses the host.

## Building and testing

The plugin needs no ORT at link time — only the headers, via pkg-config `libonnxruntime`:

```sh
cmake -S onnx -B onnx/build && cmake --build onnx/build -j8
ctest --test-dir onnx/build --output-on-failure
```

`onebit_ep_proof` builds a two-op graph (`MatMul` → `Relu`) through the ORT model editor API and
runs it twice: once on the CPU EP, once with this plugin registered **first** and the CPU EP
**last**. The MatMul must be ours, the numbers must match the CPU EP to < 1e-5 relative, and the
library's own trace must show the compute callback running. `onebit_ep_proof_cpu_fallback` is the
negative control: with nothing claimed, the CPU EP takes the MatMul and the library must stay idle,
so the match in the first test can only come from our kernel.

## Knobs (debug only)

| variable | effect |
|---|---|
| `ONEBIT_EP_TRACE=1` | append one line per callback the host calls (registration, capability, compute) to `/tmp/onebit_ep_trace.<pid>.log` — this is the ownership evidence the proof reads. The path is fixed on purpose: a `.so` other processes load must not open a path an environment variable names |
| `ONEBIT_EP_OPS=<comma list>` | ops to claim; default `MatMul`. Empty falls back to the default, so use e.g. `none` to claim nothing |
| `ONEBIT_EP_VERBOSE=1` | same as the trace, on stderr |

## Where ORT and this plugin meet (the sharp edges, v1.23.2)

- `ReleaseEpFactory` **must return `OrtStatus*`**. `EpLibraryPlugin::Unload()` wraps that return
  value in `ToStatusAndRelease` and dereferences it; a `void` signature makes ORT read whatever is
  in `RAX` as an `OrtStatus` and segfault inside `ReleaseEnv`.
- The data-transfer object must be non-null even when the EP is CPU-only: ORT calls it unguarded.
- Allocator memory must come from an `OrtDeviceAllocator` and be 256-byte aligned (MLAS and other
  EPs use aligned loads); a malloc'd 16-byte-aligned buffer faults.
- Registration order is priority order with no tie-break, so the CPU EP goes last.
- `OrtEpDevice` handles are invalidated by `UnregisterExecutionProviderLibrary`, so sessions that
  used them must be gone first.
