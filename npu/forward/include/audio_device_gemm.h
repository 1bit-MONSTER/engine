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
//
// audio_device_gemm.h -- one staged projection of an audio encoder, on the NPU (engine#1783).
//
// Why this exists as a header and not as code inside the audio executor: the device side of a projection is the TARGET's job
// (`NpuTarget::gemv` takes f32 activations and f32 back, and does the per-row int8 quantisation, the b8 pack and the
// dequantisation itself), so what the executor needs is the small, testable decision layer around it -- which shape may go to
// the device, what the caller's tensor is versus what the kernel writes, and what to say when it cannot. That is a pure
// function over a `NpuTarget&`, so it is driven against a fake target with no device, the same shape as every other rule here.
//
// Two facts it encodes, both learned from measurements on this box (engine#1783):
//  * the kernel is named at the ENGINE'S STAGED width -- K a multiple of 256, N of 512 -- so the kernel writes N padded
//    (whisper-small's O is N768 named N1024). The caller's tensor is the model's width, so the result is de-padded here rather
//    than by every caller; forgetting that is how a padded column becomes a plausible wrong activation.
//  * a `Plan` that loads has told you NOTHING about its designs: `Plan::designs_agree`/`check_artifacts` both return true when
//    the plan names no designs. So the shape this header trusts is the one the CALLER states (role/K/N taken from the pack), and
//    it refuses a shape the engine's grid cannot name instead of deriving one.

#pragma once

#include <string>
#include <vector>

#include "audio_kernel_roles.h"
#include "target.h"   // ::npu::target::{NpuTarget, GemvArgs, Caps, Op} (target.h opens npu::target at global scope)

namespace onebit::audio {

namespace nput = ::npu::target;

// One projection as the ENCODER states it (the model's own K and N, not the staged widths).
struct DeviceProjection {
    int         role = 0;        // index into the target's design table
    int         layer = -1;      // layer in flight, for residency and profiling
    int         k = 0;           // the model's contracted width
    int         n = 0;           // the model's output width (the CALLER's tensor width)
    int         rows = 0;        // M: rows to run in one launch
    const float* x = nullptr;    // M x k row-major f32
    int         a_k = 0;         // the activation's real width per row (0 = k)
    const char* tag = nullptr;   // container tag, for the target's log
};

// One projection on the device: `out` receives rows x n (the CALLER's width), de-padded from the staged write.
// Returns false and fills `why` for: a shape the engine's grid cannot name, a row count the target cannot launch,
// a target without gemv, a target failure, or an output whose size is not rows x staged_n (which is the target
// disagreeing with the plan, and is never repaired here).
inline bool device_projection(nput::NpuTarget& target, const DeviceProjection& p, std::vector<float>* out,
                             std::string* why) {
    if (out == nullptr) {
        if (why != nullptr) *why = "no output vector";
        return false;
    }
    if (p.x == nullptr || p.k <= 0 || p.n <= 0 || p.rows <= 0) {
        if (why != nullptr) *why = "a projection needs x, rows > 0, k > 0 and n > 0";
        return false;
    }
    if (!target.supports(nput::Op::gemv)) {
        if (why != nullptr) *why = std::string("the target \"") + target.name() + "\" does not implement gemv";
        return false;
    }
    const ProjectionShape shape{p.k, p.n};
    if (role_of_shape(shape, 0) == std::string("unknown") && p.tag == nullptr) {
        // Not a refusal: the caller states the role index directly. But an unnamed shape with no tag is undebuggable, and
        // this header's job is to keep the device log answerable.
        if (why != nullptr) *why = "an unnamed projection needs a tag so the target's log can name it";
        return false;
    }
    const ProjectionShape staged_shape = staged(shape);
    const nput::Caps caps = target.caps();
    const bool wide = p.rows > caps.max_rows;
    if (wide && p.rows > caps.prefill_rows) {
        if (why != nullptr)
            *why = "rows=" + std::to_string(p.rows) + " is more than the target launches in one call (prefill_rows=" +
                   std::to_string(caps.prefill_rows) + ", max_rows=" + std::to_string(caps.max_rows) + ")";
        return false;
    }

    nput::GemvArgs a;
    a.role = p.role;
    a.layer = p.layer;
    a.M = p.rows;
    a.K = staged_shape.k;
    a.N = staged_shape.n;
    a.x = p.x;
    a.a_k = p.a_k > 0 ? p.a_k : p.k;
    a.rows = wide;
    a.tag = p.tag;

    std::vector<float> got;
    std::string err;
    if (!target.gemv(a, got, err)) {
        if (why != nullptr) *why = err.empty() ? "the target refused the projection" : err;
        return false;
    }
    const size_t want = size_t(p.rows) * size_t(staged_shape.n);
    if (got.size() != want) {
        if (why != nullptr)
            *why = "the target wrote " + std::to_string(got.size()) + " floats, the plan's staged shape is " +
                   std::to_string(p.rows) + " x " + std::to_string(staged_shape.n) + " = " + std::to_string(want) +
                   " (the target and the plan disagree; this is not repaired here)";
        return false;
    }
    // De-pad: the kernel's rows are staged_n wide, the caller's tensor is n wide. Drop the padding columns per row rather
    // than returning the staged buffer, which would hand the caller a row whose stride is not its own width.
    out->assign(size_t(p.rows) * size_t(p.n), 0.0f);
    for (int r = 0; r < p.rows; ++r)
        for (int c = 0; c < p.n; ++c) (*out)[size_t(r) * size_t(p.n) + size_t(c)] = got[size_t(r) * size_t(staged_shape.n) + size_t(c)];
    return true;
}

}  // namespace onebit::audio
