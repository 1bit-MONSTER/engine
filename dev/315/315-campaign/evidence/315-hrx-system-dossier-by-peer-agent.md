<!--
ANNOTATION (2026-10-13): this dossier's section 4.5 and its architecture diagram claim the `_spt` dispatch
works via a GOT-indirect thunk ("JMP qword ptr [0x0032a3e0] ... resolves through a GOT slot that is patched").
THAT MECHANISM IS WRONG. A larger decompilation effort (~/amd-oss/hrx-re/) shows 0x31d8a0 is a .plt stub, no
hipMalloc_spt export exists, and the real mechanism is name-based dlsym(symbol + "_spt"). See
315-dossier-spt-claim-corrected.md in this directory. The dossier is preserved for its other findings; this
claim is not to be cited.
-->
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
# HRX System — Full Technical Dossier

**Target:** `github.com/ROCm/hrx-system`
**Commit:** `a5b9a7e` — "[Loom] Preserve native producers across packet consumers (#1641)"
**Dossier date:** 2026-10-13
**Method:** source analysis + full local build + Ghidra 12.1.4 static analysis via REA 6.4.0 + live execution on AMD Radeon RX 9070 XT (gfx1201)
**Host:** ryzen, Ubuntu, ROCm SDK `~/sdk-gfx1201-devel` (clang 23.0.0git), 16 cores, 45 GB RAM

Every claim below is tagged:

- **[OBS]** Observed directly — file contents, tool output, or program behavior
- **[INF]** Inferred from structure or evidence, not directly demonstrated
- **[GAP]** Not determined

---

## 0. What this system is

HRX System is an **alternative implementation of HIP** — a drop-in replacement for ROCm's `libamdhip64.so` — built on a vendored fork of the **IREE runtime**, with its own compiler (**Loom**) and its own NPU/GPU native-access library (**libamdf**).

It is **not** an official ROCm component. The repo's own banner says so:

> "This is early-access runtime infrastructure. Expect API, packaging, and deployment details to move while the fork is brought up. This is not an official component within the ROCm stack."

### The three-layer story

```
                    ┌──────────────────────────────────────────┐
   Application ────▶│  libamdhip64.so   (540 HIP symbols)      │
   (ROCm headers)   │  libhrx/src/binding/hip/                 │
                    └────────────────┬─────────────────────────┘
                                     │  hipXxx() → iree_hip_*() → iree_hal_streaming_*()
                    ┌────────────────▼─────────────────────────┐
                    │  IREE HAL  (runtime/src/iree/hal)        │
                    │  drivers: amdgpu │ task │ xdna │ vulkan  │
                    └───────┬──────────────────────┬───────────┘
                            │ HSA (dlopen)         │ libamdf
                    ┌───────▼────────┐    ┌────────▼──────────┐
                    │ libhsa-runtime │    │ libamdf (NPU/XDNA)│
                    │  (ROCm SDK)    │    │ amdf_query_api()  │
                    └────────────────┘    └───────────────────┘
```

Separately, **Loom** is the compile tier: it emits gfx1201 machine code (`libloomc.so`) that the runtime loads as an ordinary HSA code object. Loom and the runtime are coupled **only** by artifact format, not by ABI.

**[OBS]** Component line counts (excluding `.git` and `build/`):

| Component | Files | Lines | Role |
|---|---:|---:|---|
| `loom` | 7,683 | 1,719,774 | Compiler: own IR, passes, linker, multi-target codegen |
| `runtime` | 2,692 | 854,710 | Vendored IREE fork (HAL, VM, task system) |
| `libhrx` | 285 | 146,102 | HRX C ABI + HIP compatibility binding |
| `libamdf` | 721 | 125,317 | Native GPU/NPU access (Vulkan-style layered API) |
| `build_tools` | 516 | 72,350 | Bazel↔CMake, CI, toolchains |
| `docs` | 90 | 31,248 | Original AMD hardware reference (PM4/AQL/SDMA/XDNA) |
| `experimental` | 81 | 7,848 | XDNA CTS/benchmarks |
| **Total** | **12,116** | **~2.96 M** | |

---

## 1. Build — verified working

**[OBS]** The documented CMake path was used verbatim with the local SDK. **Configure succeeded first try; build completed 7042/7042 steps with 0 errors.**

```bash
cmake -S . -B build/hrx-system -GNinja \
  -DIREE_ROCM_PATH=/home/bcloud/sdk-gfx1201-devel \
  -DIREE_ROCM_DEPENDENCY_MODE=package \
  -DCMAKE_C_COMPILER=$SDK/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HAL_DRIVER_AMDGPU=ON \
  -DIREE_HAL_AMDGPU_TARGETS='gfx1201'
```

**[GAP]** `libamdf` did **not** build — it needs `AMDF_BUILD` + `AMDF_FAMILY_XDNA`, which the default configure does not enable. The NPU path is therefore **unverified by execution** in this dossier; it is described from source only.

**[OBS]** Build system split: **Bazel is the source of truth** for the build graph; **CMake is the packaging/install/embedding build**, generated from Bazel by `build_tools/bazel_to_cmake/`. There are **no CMakePresets** in the tree.

---

## 2. Shipped artifacts

**[OBS]** `cmake --install --component HrxPublicDist` produced exactly:

```
bin/hrx-info                              (8.4 MB ELF PIE)
include/hrx/hrx_runtime.h                 C ABI (1,378 lines)
include/hrx/hrx_runtime_cxx.h             RAII helpers (header-only)
include/loomc/*.h                         Loom C API (47 headers)
lib/libhrx.so.0.1.0      → .so.0 → .so    8.8 MB  SONAME libhrx.so.0
lib/libamdhip64.so.7.9999.0 → .so.7 → .so 12 MB  SONAME libamdhip64.so.7
lib/libloomc.so.0.1.0    → .so.0 → .so    48 MB  Loom compiler library
lib/cmake/hrx/…, lib/cmake/loomc/…        package configs
```

**Notable:** `libloomc.so` (the Loom compiler) ships inside the **public** install component — the compiler is a supported embedding surface, not an internal tool.

**[OBS]** All three libraries need only `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2`. Neither `libhrx.so` nor `libamdhip64.so` has a `DT_NEEDED` on `libhsa-runtime64.so` — HSA is loaded at runtime via `dlopen`. This is why the drop-in replacement works without a ROCm install prefix.

---

## 3. `libhrx` — the public C ABI

**[OBS]** 155 `HRX_API` declarations in `hrx_runtime.h`. Export control is an **anonymous** version script:

```ld
{ global: hrx_*; local: *; };
```

**[OBS] Verified in the built binary by two independent methods** — `nm -D` and REA's pwntools ELF adapter (`ev_9574038c…`):

- **Exactly 155 defined global symbols, all `hrx_*`**
- **Zero leakage** — 2,351 symbols are `STV_HIDDEN`

The API is organized as capability namespaces over a unified device model:

| Namespace | Purpose |
|---|---|
| `hrx_gpu_*` / `hrx_cpu_*` | Accelerator lifecycle (`initialize`, `device_count`, `device_get`) |
| `hrx_device_*`, `hrx_allocator_*` | Device properties, allocation |
| `hrx_stream_*`, `hrx_queue_*`, `hrx_semaphore_*`, `hrx_fence_*` | Async execution |
| `hrx_buffer_*`, `hrx_buffer_view_*` | Memory |
| `hrx_virtual_*`, `hrx_physical_*` | Virtual memory management |
| `hrx_graph_*`, `hrx_graph_exec_*`, `hrx_capture_*` | Graph capture/replay |
| `hrx_runtime_set_device_event_sink` | Event sink (ABI-versioned envelopes) |

**[OBS]** `HRX_VERSION_MAJOR/MINOR/PATCH = 0/1/0`. `hrx_runtime_cxx.h` is **header-only** (templates/inline only) so it adds **no** exported symbols.

> **[GAP]** `hrx_runtime.h` contains **no NPU/XDNA symbols**. The README's "GPU, NPU, and CPU" claim is delivered by `libamdf` (§6), not by `libhrx`.

---

## 4. `libamdhip64.so` — the HIP compatibility layer

This is the headline component. **It works.**

### 4.1 ABI surface — measured, not claimed

**[OBS]** Export control is a **named-version** linker script, `libhrx/src/binding/hip/amdhip64.map`:

> "HIP libamdhip64 symbol version script for HRX. Version nodes are generated from TheRock nightly libamdhip64.so exports."

**[OBS] Per-node symbol counts in the built binary:**

| Version node | Symbols | | Version node | Symbols |
|---|---:|---|---|---:|
| `hip_4.2` | 255 | | `hip_6.2` | 14 |
| `hip_4.3` | 27 | | `hip_6.4` | 11 |
| `hip_4.4` | 15 | | `hip_6.5` | 3 |
| `hip_4.5` | 42 | | `hip_7.1` | 19 |
| `hip_5.0` | 5 | | `hip_7.2` | 24 |
| `hip_5.1` | 31 | | `hip_7.14` | 15 |
| `hip_5.2` | 19 | | **`HRX_0.1`** | **2** |
| `hip_5.3` | 38 | | | |
| `hip_5.5` | 7 | | **Total** | **540** |
| `hip_5.6` | 5 | | | |
| `hip_6.0` | 5 | | | |
| `hip_6.1` | 3 | | | |

Plus `libamdhip64.so.7` as the BASE node. The `HRX_0.1` node adds HRX extensions: `hipHALDeinit`, `hipHRXSetDeviceEventSink`.

**[OBS]** `SONAME = libamdhip64.so.7`, file version `7.9999.0` — a deliberate sentinel claiming HIP 7 while signaling "HRX build."

### 4.2 Implementation coverage — exact accounting

**[OBS]** Reconciled three ways (export table vs. source definitions):

| Source | Symbols | Meaning |
|---|---:|---|
| `api.c` (28,899 lines) | **370** | Real implementations |
| `abi_stubs.c` (2,379 lines) | **139** | Loader compat stubs |
| `library.c`, `linker.c`, `vmm.c`, … | **31** | Specialized implementations |
| **Total exported** | **540** | |

**~69% of the HIP ABI is genuinely implemented.** The remaining 31% **loads successfully** but returns `hipErrorNotSupported` — 89 explicit `hipErrorNotSupported` returns in `abi_stubs.c`.

The stub file states its intent:

> "Loader compatibility stubs for HIP ABI entry points HRX does not implement. These keep binaries linked against upstream libamdhip64 loadable while preserving a loud unsupported result if one of these paths is executed."

**[OBS]** 39 `_spt` (stream-per-thread) variants are exported.

### 4.3 Declaration-vs-export gap

**[OBS]** 555 symbols are declared in `amdhip64.map`; **540** are actually exported. The 14 absent are all accounted for:

| Missing symbol | Reason |
|---|---|
| `_Z24hipExtModuleLaunchKernel…`, `_Z24hipHccModuleLaunchKernel…` | C++ mangled; superseded by `hipExtModuleLaunchKernel` |
| `hipGLGetDevices`, `hipGraphicsGLRegister*` (×3) | OpenGL interop — not implemented |
| `hipGraphAddExternalSemaphores{Signal,Wait}Node` | Not implemented |
| `hipOccupancyMaxActiveClusters`, `hipOccupancyMaxPotentialClusterSize` | Not implemented |
| `hipMemMapArrayAsync` | Not implemented |
| `__hipGetPCH`, `__hipRegisterSurface` | Legacy |
| `hipRegisterTracerCallback`, `hipGetCmdName` | Not implemented |
| `hipHRXSetArrayLeaseObserverForTesting` | Test-only; only in the `amdhip64_array_lease_test` variant |

**No undeclared symbols are exported** (`EXPORTED BUT NOT DECLARED` = 0).

### 4.4 Ghidra static analysis — the call chain, proven from the binary

**[OBS]** Ghidra 12.1.4 analysis of `libamdhip64.so.7.9999.0` (sha256 `46471d7a…`):
**5,311 procedures, 11,476 strings, 42 segments.** Evidence `ev_006567d1…`

**[OBS] `hipMemset` decompiled — every callee name recovered from the binary alone:**

```c
hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
  token = iree_hip_error_state_begin();
  if ((ulong)token >> 0x20 != 0) return token.fatal_result;
  if (sizeBytes == 0) hVar2 = hipSuccess;
  else if (dst != NULL) {
    context = NULL;
    hVar2 = iree_hip_ensure_context(&context);
    if (hVar2 == hipSuccess) {
      if (!iree_hip_context_invalidate_visible_captures(context)) {
        status = iree_hal_streaming_memory_memset(
            context, dst, sizeBytes, &value_local, 1, context->default_stream);
        if (status == 0) {
          status = iree_hal_streaming_stream_synchronize(context->default_stream);
          ...
```

This is a **complete, independent confirmation** of the architecture: `hipMemset → iree_hal_streaming_memory_memset → HAL`. It also reveals implementation structure not visible in the header:
- an **error-state token** protocol (`begin`/`publish`) with a fatal-result fast path
- **explicit stream-capture invalidation** on every mutating call
- status-code special-casing (`& 0x1f == 5` → `hipErrorInvalidValue`) before falling back to the general mapping

**[OBS] `hipFree`** — 258 bytes, 20 basic blocks, 9 callees. Notably it dispatches between device and host free:

```
iree_hal_streaming_memory_free_device   ← standard device pointer
iree_hal_streaming_memory_host_flags     ← probe
iree_hal_streaming_memory_free_host      ← host-registered pointer
```

**[OBS] `hipMemcpy`** — 436 bytes, **31 basic blocks**, 10 callees, including a dedicated kind resolver and three cross-context fast paths:
`iree_hip_resolve_memcpy_kind`, `iree_hip_try_cross_context_h2d`, `iree_hip_try_cross_context_d2h`, `iree_hip_try_managed_d2d`.

**[OBS] Ghidra correctly recovered exact signatures from the binary** (matching source):

| Function | Recovered signature | Bytes | Blocks |
|---|---|---:|---:|
| `hipMalloc` | `hipError_t(void**, size_t)` | 247 | 14 |
| `hipMemcpy` | `hipError_t(void*, void*, size_t, hipMemcpyKind)` | 436 | 31 |
| `hipMemset` | `hipError_t(void*, int, size_t)` | — | 15 |
| `hipFree` | `hipError_t(void*)` | 258 | 20 |
| `hipGetDeviceProperties` | `hipError_t(hipDeviceProp_tR0000*, int)` | — | 3 |
| `hipLaunchKernel_spt` | `hipError_t(void*, dim3, dim3, …)` | — | 3 |

### 4.5 The `_spt` dispatch mechanism — found in the binary

**[OBS]** Ghidra reported `hipMalloc` as **ambiguous**, with two entries: `0x19d6e0` and `0x31d8a0`. Analysis resolved it:

| Address | Size | Disassembly | Meaning |
|---|---:|---|---|
| `0x19d6e0` | 247 B | full body, 7 callees | real `hipMalloc` |
| `0x31d8a0` | **6 B** | `JMP qword ptr [0x0032a3e0]` | **GOT-indirect thunk** |

Same for `hipMemcpy` (`0x31d8c0` → `JMP [0x0032a3f0]`).

**[OBS]** The source confirms the design — `hipGetProcAddress` prefers the `_spt` symbol when `IREE_HIP_API_LOOKUP_PER_THREAD_DEFAULT_STREAM` is set, and version-selects legacy aliases:

```c
if (strcmp(symbol, "hipGetDeviceProperties") == 0)
  lookup_symbol = hip_version >= R0600_VERSION ? "hipGetDevicePropertiesR0600"
                                               : "hipGetDevicePropertiesR0000";
```

So the plain symbol name resolves through a GOT slot that is patched to the per-thread-stream variant. **This is the same runtime-patching mechanism upstream ROCm HIP uses**, reproduced faithfully.

### 4.6 Status mapping

**[OBS]** `status_conversion.c` maps IREE → HIP exhaustively:

| IREE status | HIP error |
|---|---|
| `INVALID_ARGUMENT`, `OUT_OF_RANGE`, `INCOMPATIBLE` | `hipErrorInvalidValue` |
| `RESOURCE_EXHAUSTED` | `hipErrorOutOfMemory` |
| `NOT_FOUND` | `hipErrorNotFound` |
| `PERMISSION_DENIED` | `hipErrorInvalidContext` |
| `UNIMPLEMENTED` | `hipErrorNotSupported` |
| `UNAVAILABLE` | `hipErrorNotReady` |
| `FAILED_PRECONDITION` | `hipErrorNotInitialized` |
| `DATA_LOSS` | `hipErrorLaunchFailure` |
| `ABORTED` | `hipErrorIllegalAddress` |

> **Defect [OBS]:** `PERMISSION_DENIED → hipErrorInvalidContext` and `RESOURCE_EXHAUSTED → hipErrorOutOfMemory` produce misleading diagnostics. When `/dev/kfd` access was denied, the user saw `HSA_STATUS_ERROR_OUT_OF_RESOURCES` ("failed to allocate resources … needs to spawn threads") and later `hipErrorOutOfMemory` for what was purely an EACCES. See §8.

---

## 5. `runtime/` — the vendored IREE fork

**[OBS]** `runtime/src/iree/` is the **IREE runtime source tree checked directly into this repo** — not a submodule (no `.gitmodules`), not renamed. Every file carries `Copyright <year> The IREE Authors` with years spanning 2020–2026. The Bazel module is `hrx` and there is **no `bazel_dep(name="iree")`**.

**[INF]** Best described as an **in-tree fork / vendored overlay** of IREE. The single-commit clone means the exact upstream revision is **not recoverable** from git metadata.

Subsystems: `base/` (platform/threading), `hal/` (driver framework), `task/` (multithreaded scheduler), `vm/` (bytecode VM), `async/`, `net/`, `io/`, `tools/`, `testing/`.

### 5.1 HAL drivers

**[OBS]** Registered drivers and their configuration:

| Driver | Source | Backing |
|---|---|---|
| `amdgpu` | `hal/drivers/amdgpu/` | **HSA/ROCR directly** |
| `task` | `hal/drivers/task/` | CPU, IREE task system (**default**) |
| `xdna` | `hal/drivers/amd/` | NPU via **libamdf** |
| `vulkan`, `webgpu` | `hal/drivers/…` | Supported, not enabled here |

**[OBS]** The AMDGPU driver explicitly bypasses HIP:

> "It does not route through HIP or the legacy ROCm HAL driver."

### 5.2 The actual GPU submission path

**[OBS]** Work reaches the GPU as **AQL packets** — confirmed by the source that submits it:

| Step | Location |
|---|---|
| Reserve ring slot | `util/aql_ring.h:185-203` — atomic fetch-add on `write_dispatch_id` |
| Emit packet body | `util/aql_emitter.h` |
| **Publish** | `util/aql_ring.h:255-262` — **atomic store-release of header+setup** |
| **Ring doorbell** | `util/aql_ring.h:277-296` — MMIO store to `hardware_doorbell_ptr` after `host_write_fence()`; falls back to `hsa_signal_store_screlease()` |
| Orchestration | `host_queue.c:96-99` — `aql_ring_commit(...)` then `aql_ring_doorbell(...)` |

**[OBS]** HSA is bound dynamically: `util/libhsa.c:288-293` tries `libhsa-runtime64.so.1` then `libhsa-runtime64.so`, resolving every entry point through `iree_dynamic_library_lookup_symbol`. Queue creation uses `hsa_amd_queue_create_desc_t` + `hsa_amd_queue_create`; code objects load via `hsa_executable_create_alt` → `hsa_executable_load_agent_code_object` → `hsa_executable_freeze`.

> **This is why the local ROCm SDK mattered:** the system `/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1` lacks `hsa_amd_queue_create`, so HRX correctly refused to use it. Pointing at the SDK provided it.

### 5.3 Target selectors and arch dispatch

**[OBS]** `gfx942`, `gfx1151`, `gfx1201` serve four purposes:
1. **Build-time**: which HSACO device-library blobs get embedded (`IREE_HAL_AMDGPU_TARGETS`). Architectures not built in **fail at runtime**.
2. **Runtime matching**: `target/identity_catalog.inl` maps exact → generic (`gfx1201 → gfx12-generic`), and `executable.c` preflights agent/ISA compatibility.
3. **Load-time derivation**: `target/code_object.c` recovers the processor from the HSACO ELF machine flags.
4. **Selection priority**: exact target = 100, generic = 50.

### 5.4 Loom coupling

**[OBS]** There are **no Loom types in `runtime/src`**. Loom emits a standard HSACO; the runtime loads it through the ordinary ELF code-object path. Dependency direction is strictly **Loom → HAL executable format**. `loom/config/execute/` declares Loom execution substrates named `iree_hal`, `iree_hal_amdgpu`, `iree_hal_vulkan`.

---

## 6. `libamdf` — the NPU/GPU native layer

**[OBS]** Undocumented in the root README. Self-described as:

> "a small, portable C library for native AMD GPU and NPU access … with executable loading, command construction, scheduling, and memory policy owned by the calling runtime, typically a hardware abstraction layer (HAL)."

**Explicitly must not depend on `runtime/`, `loom/`, or `libhrx/`** — those may consume libamdf, never the reverse.

### 6.1 API design

**[OBS]** A single entry point exports the whole ABI:

```c
amdf_query_api(min_version, max_version, amdf_api_t** out_api);   // ABI v9
```

- Windows `.def` exports **exactly one symbol**: `amdf_query_api`
- Linux hides everything else via `C_VISIBILITY_PRESET hidden`
- `amdf_api_t` is a **49-member function-pointer table** with append-only growth, `structure_size`/`type`/`next` extension chains
- Extensions: `AMDF_EXTENSION_GPU = 2` (6 methods), `AMDF_EXTENSION_XDNA = 1` (10 methods)

### 6.2 Hardware coverage

**[OBS]** Three engine kinds: `AMDF_ENGINE_KIND_GPU`, `AMDF_ENGINE_KIND_XDNA`, plus CPU participation via system-memory scopes (no synthetic CPU device).

XDNA targets mapped by PCI ID in `src/xdna/device_profile.c`:

| PCI ID | Target | Architecture |
|---|---|---|
| `0x1502/0x00` | `amd.xdna.phoenix` | AIE2 |
| `0x17f0/0x10` | `amd.xdna.strix` | AIE2P (NPU4) |
| `0x17f0/0x11` | `amd.xdna.strix_halo` | AIE2P (NPU5) |
| `0x17f0/0x20` | `amd.xdna.krackan` | AIE2P (NPU6) |

Platform seams: Linux KFD/DRM (`src/gpu/umd/kfd`, `src/xdna/umd/drm`), Windows WDDM/MCDM (`src/gpu/umd/wddm`, `src/xdna/umd/mcdm`).

> **`amdf` is never expanded anywhere in the repo.** **[GAP]** Most consistent reading with the "one fabric" language in `docs/fabric.md` is "AMD Fabric," but the source never states it. Treat as a guess.

### 6.3 Documentation

**[OBS]** Six design docs: `fabric.md`, `discovery.md`, `memory.md`, `performance.md`, `gpu.md`, `xdna.md`. `performance.md` is unusual and notable — it specifies **method-level cost contracts** (which paths may allocate, lock, syscall, lazily initialize, or poll) rather than just functional behavior.

**[OBS]** CTS covers GPU (PM4, SDMA, AQL, recipes, peer, interop), XDNA, and cross-device recipes, with required-case enforcement (`--amdf_require_test=Suite.Case`) and a design rule that **a failed oracle must still reach completion/retirement before storage reuse**.

> **[GAP]** libamdf was not built or executed in this investigation.

---

## 7. `loom` — the compiler

**[OBS]** The largest and most novel component (1.72 M lines). Self-described as:

> "a source-first native compiler substrate for asynchronous device-oriented programs."

### 7.1 What Loom is and is not

**Is:** a complete compiler — own IR (High/Low ISA), passes, linker, specializer, scheduler, register allocator, and multi-target emitters, embedded via a C ABI (`libloomc.so`, 47 public headers).

**Is not** [OBS]:
- **not** a runtime (it emits artifacts; loading is the caller's job)
- **not** a dispatch router over HIP/BLAS/Triton/CK/AITER — the README rejects this explicitly
- **not** LLVM/MLIR-based — grep for `llvm` in `loom/src` finds only license headers
- **not** a Python FFI binding — Python is an authoring DSL and generator; the embedding surface is C

### 7.2 Ownership split

**[OBS]** A deliberate and unusual doctrine from `loom/AGENTS.md`:

> "**Python owns** source-of-truth data that already lives naturally in Python: dialect declarations, descriptor schemas, target facts, vendor machine-readable ISA data, generated contract rows, spelling tables… **C owns** the shipping compiler surface: structs, enums, public and private headers, semantic comments, function declarations, hot-path interpreters, lowering mechanics…"

> "Generated files are not design documents. A generator that emits a public struct or enum for a backend table is usually hiding the contract in the wrong place."

### 7.3 Targets

**[OBS]** `target/arch/`: `amdgpu`, `amd/xdna`, `spirv`, `x86`, `wasm`, `vm`, `cmd`. Status per the README:

| Target | Status |
|---|---|
| AMDGPU HSACO | "Most established native target path" |
| SPIR-V/Vulkan | Working, hardening |
| x86, Wasm | Real early paths |
| Portable command programs | Target-neutral artifacts |

### 7.4 Verification: Loom actually compiles to gfx1201

This is the strongest single result in this dossier. **[OBS]** Given a 12-line Loom kernel, `loom-compile` emitted a real GPU code object:

```
$ loom-compile kernel.loom --format=amdgpu-hsaco --target=amdgpu:gfx1201 --output=copy_i32.hsaco
→ 9,168 bytes, exit 0
```

`file` reports: **"ELF 64-bit LSB shared object, AMD GPU architecture version 1"**, OS/ABI **"AMD HSA"**, Machine `EM_AMDGPU` (0xe0), `e_flags = 0x4E`.

**[OBS]** `e_flags = 0x4E` decodes via the repo's own `build_tools/amdgpu/elf_machine_map.inl:54`:
```
IREE_AMDGPU_ELF_MACHINE_TARGET(0x04eu, "gfx1201", false, false)
```
**Exactly the requested target — and exactly the local GPU.**

**[OBS]** Symbols: `copy_i32` (52-byte `FUNC` in `.text`) and `copy_i32.kd` (64-byte kernel descriptor in `.rodata`) — the canonical HSA structure.

**[OBS]** Disassembled with the SDK's `llvm-objdump`:

```asm
    .amdgcn_target "amdgcn-unknown-amdhsa-unknown-gfx1201"
copy_i32:
    s_load_b128 s[4:7], s[0:1], 0x0     ; one 128-bit load of BOTH buffer descriptors
    v_lshlrev_b32 v0, 2, v0             ; lane * 4  (i32 stride)
    s_wait_kmcnt 0x0
    global_load_b32 v1, v0, s[4:5]      ; input[lane]
    s_wait_loadcnt 0x0
    global_store_b32 v0, v1, s[6:7]     ; output[lane]
    s_wait_storecnt 0x0
    s_endpgm
```

This is **hand-quality output**: descriptors coalesced into a single 128-bit scalar load, index arithmetic vectorized, and every memory operation fenced with the *narrowest correct* `s_wait_*cnt` variant rather than a blanket wait. LLVM's own disassembler confirms the target string `gfx1201`.

**Loom is a real, working compiler producing production-grade RDNA4 code.**

### 7.5 Tools

**[OBS]** `loom-compile`, `loom-link`, `loom-opt`, `loom-check` (golden tests), `loom-format`, `loom-import-cxx`, `iree-test-loom`, `iree-benchmark-loom`, `iree-run-loom`, plus Python CLIs (`loom-compile-report`, `loom-migrate`).

**[OBS]** Source formats: `.loom` (text IR), `.loombc` (bytecode), `.loom-test` (golden tests), artifact manifests (JSON), compile reports (JSON). Correctness (`check.case`) and performance (`check.benchmark`) policies live **inside the source file**.

---

## 8. Live execution on gfx1201 — end-to-end verification

### 8.1 Device discovery

**[OBS]** After correcting the HSA path (§8.4):

```
HRX (HIP Runtime Extended) v0.1.0
GPU accelerator: 1 device
  [0] AMD Radeon RX 9070 XT (Node 1) (gfx1201)
CPU accelerator: 1 device
  [0] CPU 0 (task)
```

### 8.2 GPU smoke test

**[OBS]** `hrx-info --device=gpu:0 --test=all`:
```
Opening gpu:0...
  Allocate 1 MiB buffer: OK
  Fill buffer: OK
  Copy buffer: OK
  Verify data: OK
  Release: OK
All tests PASSED
```

### 8.3 The drop-in replacement — proven

**[OBS]** A test program compiled against **real ROCm SDK headers** (not HRX's), linked `-lamdhip64`, then executed with HRX's library first on the search path:

```
hipGetDeviceCount -> 1 err=no error
prop err=no error name=AMD Radeon RX 9070 XT gcnArch=gfx1201 major=12 minor=0
hipMalloc -> 0x700988400000 err=no error
hipMemset -> no error
hipMemcpy D2H -> no error first=ab      ← data verified
hipFree -> no error
hipRuntimeGetVersion -> no error v=70051831     (= 7.5.18.31)
hipDriverGetVersion -> v=70051831
```

**[OBS]** `ldd` confirms the loaded library:
```
libamdhip64.so.7 => /home/bcloud/work/hrx-system/build/hrx-install/lib/libamdhip64.so.7
```

**An application built against upstream ROCm headers and linking real HIP runs correctly on HRX's reimplementation, on real RDNA4 hardware, including device property queries, memory allocation, fill, and verified device→host copy.**

### 8.4 CTS — 79/79

**[OBS]** `ctest -R 'libhrx/'` in the installed test tree:

```
100% tests passed, 0 tests failed out of 79
Label Time Summary:
  runtime-resource=amd-gpu  = 41.42 sec*proc (32 tests)
  libhrx/cts                = 29.31 sec*proc (25 tests)
  libhrx/src/binding/hip    =  0.91 sec*proc (27 tests)
  libhrx/src/libhrx         = 12.05 sec*proc (9 tests)
```

32 tests gated on real AMD GPU hardware — including `hip_module_execution_test`, `hip_module_library_execution_test`, `hip_occupancy_test`, `hip_event_timing_test`, `hrx_cts_virtual_memory_gpu`, and `hip_blocking_printf_stress_test`. All pass.

### 8.5 Passthrough / interception

**[OBS]** Four interceptor libraries build (`libhip_intercept`, `libhip_logging`, `libhip_noop`, `libhip_buffer_tracer`) implementing a proxy: load in place of HIP, `dlopen` the real backend from `HIP_PASSTHROUGH_BACKEND_LIB`, resolve all symbols into a function table, and optionally layer an interceptor from `HIP_PASSTHROUGH_INTERCEPTOR`.

**[OBS]** Verified live: with `libhip_intercept.so` forwarding to HRX's `libamdhip64.so`, the drop-in test passes identically — the proxy is transparent.

> **Minor defect [OBS]:** the `libhip_logging` interceptor loaded and forwarded correctly but **wrote no log file**, despite `HIP_LOG_FILE`/`HIP_LOG_LEVEL` being honored in `hip_logging.c:569,580`. Untriaged.
>
> **[OBS]** These four libraries are **not installed by any component** — only `include/passthrough/hip_function_table.h` is. They are build-tree developer tools.

### 8.6 The environment bug — and a misleading error message

This cost real investigation time and is worth recording.

**[OBS]** Timeline:
1. 10:58–11:04 — all GPU tests pass.
2. **11:05:18** — `/dev/kfd` and `/dev/dri/*` are **recreated** (mtime proves it).
3. A **GDM greeter session** (`gdm-greeter`, seat0) appears and logind grants it the device ACL; `bcloud`'s sessions lose theirs.
4. All GPU access begins failing.

**[OBS]** Device state: `/dev/kfd 660 root:render`, ACL granted only to `gdm-greeter`; user `bcloud` was **not** in `render` or `video`. Direct `open()`:
```
/dev/kfd -> Permission denied
/dev/dri/renderD128 -> Permission denied
```

**[OBS]** HRX's reaction:
```
GPU accelerator: unavailable
(libhsa.c:499: RESOURCE_EXHAUSTED; [hsa_init] HSA_STATUS_ERROR_OUT_OF_RESOURCES:
 The HSA runtime failed to allocate the necessary resources. This error may also
 occur when the HSA runtime needs to spawn threads or create internal OS-specific events.)
```

**This message is actively misleading** — it describes a resource-allocation failure when the true cause was `EACCES` on a device node. Per §4.6, HRX maps it onward to `hipErrorOutOfMemory`. Both messages point away from the actual problem.

**[OBS] Proof of diagnosis** — running with device access restored:
```
$ sudo hrx-info
GPU accelerator: 1 device
  [0] AMD Radeon RX 9070 XT (Node 1) (gfx1201)
$ sudo hrx-info --device=gpu:0 --test=all
All tests PASSED
```

**[OBS]** This also **retroactively explains** the single CTS failure seen at 11:05 (`hip_blocking_printf_stress_test`, `synchronize_result = 719 = hipErrorLaunchFailure`) — a device-permission revocation mid-run, not an HRX defect. Confirmed by the clean 79/79 re-run.

**Remedy applied:** `sudo usermod -aG render,video bcloud` — verified recorded (`render:x:990:ollama,lemonade,bcloud`). Requires a **new login session** to take effect.

---

## 9. `docs/` — the hidden asset

**[OBS]** **31,248 lines across 90 files** of original AMD hardware reference — assembled, per its own disclaimer, "through best-effort reverse engineering and distillation of publicly available repositories and documentation, together with our own experiments… uses no confidential or non-public vendor material."

Coverage: PM4 (23 files), SDMA (23), AQL (10), GPU core (14 — queues, scheduling, cooperative, clusters, context-save, notifications, WDDM), XDNA (5), CPU/GPU/NPU interop (5), recipes (5).

Breadth is notable: CWSR context-save sizing, CDNA5 clusters and TDM tensor descriptors, `COMPUTE_STATIC_THREAD_MGMT_SE*` affinity, `ALLOC_QUEUE_GWS`, metadata-prefetch rings, KFD event ages, GPU cache residency, RDMA write visibility, and all **six directed CPU/GPU/NPU handoffs**.

The interop chapter is careful about its claims:

> "These arrows name programming flows, not a claim of coherent peer access on every device."

---

## 10. CI, packaging, testing

**[OBS]** Nine workflows: `build_core_{linux,windows}`, `ci_{iree,libhrx}`, `presubmit`, `release_core_linux`, `test_core_linux_gpu{,_source}`, `docs`.

**[OBS]** Release artifacts (per README): `hrx-public-*`, `hrx-public-deps-*`, `hrx-tests-*`, `hrx-rocm-buildenv-*` for both Linux and Windows, as `.tar.zst` with companion environment scripts.

**[OBS]** Local presubmit is hook-driven with three profiles (`default`, `paranoid`, `ci`), via `python dev.py {bazel,cmake} {setup,hook,precommit,presubmit,fix}`. Hooks apply mechanical fixups to staged files only and **stop before tests** if the index changed — preserving partial-staging intent.

**[OBS]** The installed test tree is relocatable (2,924 tests total across all components) and runs outside the build directory; `HRX_TEST_TMPDIR` redirects per-test scratch.

---

## 11. Consolidated findings

### Strengths

1. **The HIP replacement genuinely works** — built against stock ROCm headers, ran with verified data on RDNA4 (§8.3).
2. **ABI discipline is rigorous** — 540 versioned exports across 19 real HIP version nodes, zero symbol leakage in `libhrx.so`, zero undeclared exports (§4.1, §3).
3. **Failure is honest** — unimplemented HIP entry points are stubs that return `hipErrorNotSupported`, so binaries load rather than fail to resolve (§4.2).
4. **Loom is a real compiler** — emits optimal gfx1201 assembly, LLVM-verified (§7.4).
5. **Architecture is verified three independent ways** — source, Ghidra decompilation, and live execution all agree on `hipXxx → iree_hal_streaming_* → HAL` (§4.4).
6. **31k lines of original AMD hardware reference** — a substantial, reusable artifact in its own right (§9).
7. **Clean component boundaries** — `libhrx`/`libamddf`/`loom`/`runtime` have enforced dependency directions (§6, §7.2).

### Defects and gaps

| # | Finding | Severity | Evidence |
|---|---|---|---|
| 1 | **Misleading error for device permission denial.** EACCES on `/dev/kfd` surfaces as `HSA_STATUS_ERROR_OUT_OF_RESOURCES` → `hipErrorOutOfMemory`/`hipErrorInvalidContext`. No path suggests a permissions problem. | Medium — real debugging cost | §8.6 |
| 2 | **14 declared-but-unexported symbols** — GL interop, cluster occupancy, `hipMemMapArrayAsync`, `hipGraphAddExternalSemaphores*`, test hooks. Silent ABI holes vs. the map file. | Low–Medium | §4.3 |
| 3 | **`hipErrorNotSupported` for 139/540 (26%) of the ABI.** Broad but incomplete. | Expected, but quantify before adoption | §4.2 |
| 4 | **`libhip_logging` writes no log** despite honoring its env vars. | Low | §8.5 |
| 5 | **Passthrough libraries are never installed** by any component — build-tree only. | Low | §8.5 |
| 6 | **NPU path unverified** — `libamdf` requires `AMDF_BUILD`+`AMDF_FAMILY_XDNA`, not enabled by default. The README's NPU claim rests on unbuilt code. | Coverage gap | §1, §6 |
| 7 | **`amdf` never expanded** anywhere in the repo. | Cosmetic | §6.2 |
| 8 | **Unknown upstream IREE revision** — single-commit clone, no lock entry; fork provenance not reproducible. | Maintainability | §5 |
| 9 | **`libhrx` exposes no NPU API**, though the README implies GPU/NPU/CPU unity. NPU lives only in `libamdf`. | Documentation | §3 |

### Not determined

- Whether `hipGetDevicePropertiesR0600` returns fully correct values for every field (only `name`, `gcnArchName`, `major`, `minor` verified).
- Behavior under multi-GPU, fork, and multithread scenarios (upstream CTS README lists these as untested).
- Any performance comparison vs. ROCm HIP — this dossier did not benchmark.
- The 2,924-test tree was run only for the `libhrx/` subset (79 tests).

---

## 12. Reproduction commands

```bash
# Build
cd ~/work/hrx-system
cmake -S . -B build/hrx-system -GNinja \
  -DIREE_ROCM_PATH=/home/bcloud/sdk-gfx1201-devel \
  -DIREE_ROCM_DEPENDENCY_MODE=package -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_C_COMPILER=$SDK/llvm/bin/clang -DCMAKE_CXX_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_ASM_COMPILER=$SDK/llvm/bin/clang -DCMAKE_AR=$SDK/llvm/bin/llvm-ar \
  -DCMAKE_RANLIB=$SDK/llvm/bin/llvm-ranlib \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_HAL_AMDGPU_TARGETS='gfx1201'
cmake --build build/hrx-system
cmake --install build/hrx-system --prefix build/hrx-install --component HrxPublicDist
cmake --install build/hrx-system --prefix build/hrx-tests    --component HrxTestsDist

# Device access (required; see §8.6)
sudo usermod -aG render,video $USER   # then re-login
export LD_LIBRARY_PATH=$PWD/build/hrx-install/lib:/home/bcloud/sdk-gfx1201-devel/lib

# Verify GPU + smoke test
./build/hrx-install/bin/hrx-info
./build/hrx-install/bin/hrx-info --device=gpu:0 --test=all

# CTS
HRX_TEST_TMPDIR=/tmp/hrx-tests ctest \
  --test-dir build/hrx-tests/share/hrx-system/tests -R 'libhrx/' --output-on-failure

# Loom compile
build/hrx-system/loom/src/loom/tools/loom-compile/loom-compile \
  loom/docs/examples/guide/checks-and-benchmarks/kernel.loom \
  --format=amdgpu-hsaco --target=amdgpu:gfx1201 --output=/tmp/copy_i32.hsaco
/home/bcloud/sdk-gfx1201-devel/llvm/bin/llvm-objdump -d \
  --arch-name=amdgcn --triple=amdgcn--amdhsa /tmp/copy_i32.hsaco

# ABI surface
nm -D --defined-only build/hrx-install/lib/libamdhip64.so.7.9999.0 | wc -l   # 540
nm -D --defined-only build/hrx-install/lib/libhrx.so.0.1.0  | wc -l          # 155 hrx_*
```

**REA evidence records** (Ghidra 12.1.4, REA 6.4.0):
- `ev_9574038c…` — `libhrx.so` layout: 4,404 symbols, 155 exported `hrx_*`, 2,351 hidden
- `ev_40db2022…` — `libamdhip64.so` layout: 540 exports, all `hip*`, 0 other
- `ev_482a27b4…` — `hrx-info` layout
- `ev_006567d1…` — `libamdhip64.so` Ghidra overview: 5,311 procedures, 11,476 strings

---

## Appendix: architecture in one diagram

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │ APPLICATION — compiled against stock ROCm headers                      │
 │   hipMalloc hipMemcpy hipLaunchKernel hipModuleLoad ... hipGetDeviceProp│
 └───────────────────────────────┬────────────────────────────────────────┘
                                 │  ABI: 540 symbols, 19 version nodes
 ┌───────────────────────────────▼────────────────────────────────────────┐
 │ libamdhip64.so  (12 MB,  SONAME .so.7,  NEEDED: libc+libm only)        │
 │                                                                         │
 │   real impls (370)          │  compat stubs (139)                      │
 │   api.c (28,899 lines)      │  abi_stubs.c → hipErrorNotSupported      │
 │                             │  library.c, linker.c, vmm.c (31)         │
 │                                                                         │
 │   every call:  iree_hip_error_state_begin()                             │
 │                iree_hip_ensure_context()                                │
 │                iree_hip_context_invalidate_visible_captures()           │
 │                iree_status_to_hip_result()                              │
 │   GOT thunks:  hipMalloc → JMP [GOT] → hipMalloc_spt  (per-thread)      │
 └───────────────────────────────┬────────────────────────────────────────┘
                                 │  iree_hal_streaming_*
 ┌───────────────────────────────▼────────────────────────────────────────┐
 │ IREE HAL  (runtime/src/iree/hal)  — vendored IREE fork                  │
 │   amdgpu driver │ task (CPU, default) │ xdna │ vulkan │ webgpu          │
 └──────────┬──────────────────────────────┬──────────────────────────────┘
            │ AQL rings (dlopen HSA)       │ libamdf (1 symbol exported)
 ┌──────────▼──────────────┐   ┌───────────▼────────────────────────────┐
 │ libhsa-runtime64.so     │   │ libamdf  — amdf_query_api(ABI v9)      │
 │  hsa_amd_queue_create   │   │  GPU: KFD (Linux) / WDDM (Windows)     │
 │  doorbell MMIO store    │   │  XDNA: DRM (Linux) / MCDM (Windows)    │
 │  store-release publish  │   │  AIE2 / AIE2P  (NPU4/5/6)              │
 └──────────┬──────────────┘   └────────────────────────────────────────┘
            │ gfx1201 AQL dispatch
 ┌──────────▼─────────────────────────────────────────────────────────────┐
 │ AMD Radeon RX 9070 XT — RDNA4,  wave32/64,  HSACO code objects         │
 └──────────▲─────────────────────────────────────────────────────────────┘
            │  .hsaco ELF (EM_AMDGPU 0xe0, e_flags 0x4E = gfx1201)
 ┌──────────┴─────────────────────────────────────────────────────────────┐
 │ LOOM  (libloomc.so, 48 MB) — independent compiler, coupled by format    │
 │   .loom source + check.case/check.benchmark → IR → passes → link →      │
 │   specialize → Low ISA → schedule → allocate → emit                     │
 │   Python owns facts/descriptors; C owns the shipping compiler           │
 │   Verified: copy_i32 → 8-instr gfx1201 kernel, LLVM-objdump confirmed   │
 └────────────────────────────────────────────────────────────────────────┘
```
