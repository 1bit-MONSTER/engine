// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// onebit_npu_gemm.h — the plugin EP's NPU MatMul, behind one call.
//
// This is not a second GEMM: it drives the same context the forward route uses
// (I8Ctx from npu/forward/engine-src, one full ELF per projection shape, int8 weights
// packed per column by packB, int8 activations quantised per row at amax/127,
// dequant Cm*as*gs). The forward route's npu_gemm_rows() is the model-bound
// version of the same sequence; a plugin EP has no model, so it feeds this.
//
// Device rules (brief, TORTEP.md): the EP never evicts the owner's forward
// serve. The lock is taken NON-BLOCKING and held for the life of the process;
// if someone else holds it the NPU path is simply unavailable and the caller
// keeps the CPU kernel. An ORT session must never block behind another process.
#pragma once

#include <cstdint>
#include <string>

namespace onebit::npu {

// True when the NPU path can run. When false, `why` says what is missing
// (lock held, no ELF configured, XRT device open failed, ...).
bool available(std::string* why);

// True when an ELF is configured for this (K, N). Answered from configuration
// alone — no device is opened and the device lock is not taken — so the EP can
// ask while it is deciding which nodes to claim.
bool has_kernel(int64_t k, int64_t n);

// C[M,N] = A[M,K] @ B[K,N], all float32 row-major, B[K][N].
// False means the NPU path could not run at all — the caller must fall back to
// its CPU kernel and must not claim the device did the work.
//
// `stats`, when not null, is filled with a measurement hook for the caller's profiler (ONEBIT_EP_QDQ_PROF):
// how many calls reached a kernel, how many of those had to re-pack the weight, and the wall time that went
// to `packB` against the time that went to `go_rows` (quantise + upload, both syncs, launch, wait, readback,
// dequant — one number, because those steps are the runtime's own and are timed inside it, not here).
// The float leg's cost per call is otherwise unattributable: the per-design probe times the same function
// with synthetic operands and a warm weight cache, so a graph that re-packs or that pays for real 32-row
// activation quantisation looks identical to it. A plain call passes null and pays nothing.
struct F32Stats {
    long calls = 0;       // calls that reached a kernel
    long wpacks = 0;      // packB runs, i.e. weight-cache misses (see Q8RawSplit::wpacks: equal to calls on a
                          // model graph, because one packed weight per (K, N) is shared by 28-56 weights)
    double wpack_ms = 0;  // packB time (one column-wise quantisation of a whole weight per miss)
    double rows_ms = 0;   // go_rows time: quantise+upload, both syncs, launch, wait, readback, dequant
};

bool gemm_f32(const float* a, int64_t m, int64_t k, int64_t n, const float* b,
              float* c, std::string* err, F32Stats* stats = nullptr);

// The A8W8 QDQ path: int8 activations and int8 weights straight into the kernel's
// input and weight BOs, exact int32 accumulators out (C[m][n], row major).
//
// Different from gemm_f32 on purpose: no float anywhere. gemm_f32 quantises the
// weights with packB (amax/127) and the activations per row, because the forward
// route has floats; a quantised graph already has the int8 bytes and the scales,
// so this writes both BOs verbatim and hands the raw accumulators back — the
// caller applies the graph's own scales, zero points and rounding, so that
// arithmetic exists in exactly one place (onnx/ep/onebit_ep_kernel.cpp) and a
// device run and a host run of the same graph agree to the byte.
// False means the device could not run it (no ELF for this K x N, device busy,
// XRT failure); the caller keeps its host loop and must not claim the device.
//
// `split`, when not null, is filled with where that call's wall time went, in ms. It is a measurement
// hook for the caller's own profiler (onnx/ep/onebit_ep_kernel.cpp, ONEBIT_EP_QDQ_PROF): a plain call
// passes null and pays one branch per section, not a clock read.
struct Q8RawSplit {
    // Weight-cache misses -- and on a real transformer graph this equals the call count, by design rather than by
    // accident: the cache holds ONE packed weight per (K, N) kernel, while Qwen3-0.6B has 196 weights across five
    // shapes (28 for q/o/down, 56 for k/v and gate/up), so successive calls of one shape evict each other. Measured
    // with this counter: 1764 packs for 1764 calls, 771 ms of a 7646 ms device arm.
    long wpacks = 0;
    double wpack_ms = 0;     // weight pack into the BO + its upload
    double aupload_ms = 0;   // activation memset/memcpy + A readback sizing, per launch block
    double async_ms = 0;     // A flush (sync_A) per launch block
    double launch_ms = 0;    // launch + wait_kernel: the device's own time, as the host sees it
    double readback_ms = 0;  // C invalidation + the block's copy back to `c`
};

bool gemm_q8_raw(const int8_t* a, int64_t m, int64_t k, int64_t n, const int8_t* b,
                 int32_t* c, std::string* err, Q8RawSplit* split = nullptr);

}  // namespace onebit::npu
