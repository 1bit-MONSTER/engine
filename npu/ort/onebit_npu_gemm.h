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
bool gemm_f32(const float* a, int64_t m, int64_t k, int64_t n, const float* b,
              float* c, std::string* err);

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
bool gemm_q8_raw(const int8_t* a, int64_t m, int64_t k, int64_t n, const int8_t* b,
                 int32_t* c, std::string* err);

}  // namespace onebit::npu
