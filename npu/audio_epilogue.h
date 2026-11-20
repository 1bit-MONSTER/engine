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
// npu/audio_epilogue.h -- the elementwise tail of a projection, applied per element where the product is written
// (engine#1783). The whisper encoder spells it as separate ONNX nodes after a MatMul:
//   bias       Add(x, b[n])
//   gelu       Div(y, c_div) -> Erf -> Add(., c_one) -> Mul(c_half, .) -> Mul(y, .)
//   residual   Add(r[m][n], y)
// Each step here is the same single IEEE operation on the same two values that the node runs (broadcast_apply's
// x + y, x * y, x / y; Erf's std::erf), in the graph's order, so a folded output is bit-identical to the node-by-node
// one. Float + and * are exactly commutative, so the operand order inside one step does not change a bit. The device
// dequant applies it (npu/target/xdna/rows_host.h), and so does the host when a target did not.
#ifndef ONEBIT_NPU_AUDIO_EPILOGUE_H
#define ONEBIT_NPU_AUDIO_EPILOGUE_H

#include <cmath>
#include <cstdint>

namespace onebit::audio {

struct Epilogue {
    const float* bias = nullptr;   // per output column; null = no bias step
    bool gelu = false;             // the five-node erf GELU, with the graph's own constants
    float gelu_div = 0.0f, gelu_one = 0.0f, gelu_half = 0.0f;
    const float* res = nullptr;    // residual, row m at res + m * res_ld; null = no residual step; ignored with gelu
    int64_t res_ld = 0;
    bool any() const { return bias != nullptr || gelu || res != nullptr; }
};

inline float epilogue_one(const Epilogue& e, float v, const float* res_row, int64_t n) {
    if (e.bias != nullptr) v = v + e.bias[n];
    if (e.gelu) {
        float t = v / e.gelu_div;
        t = std::erf(t);
        t = t + e.gelu_one;
        t = e.gelu_half * t;
        v = v * t;
    }
    // GELU and residual are exclusive (the planner in audio_exec.cpp never sets both: whisper's fc1 has the GELU, o and fc2
    // the residual), and kept exclusive here: a residual add straight after the GELU's last multiply is the one place in
    // this tail where -mfma under GCC's default -ffp-contract=fast could fuse a multiply and an add into one FMA, which the
    // node path (two nodes, the product stored in between) never does.
    else if (res_row != nullptr) v = res_row[n] + v;
    return v;
}

// Rows [m0, m1) of `out` (row stride ldo, N columns), in place. Residual rows are indexed like `out`'s.
inline void epilogue_rows(const Epilogue& e, int64_t m0, int64_t m1, int64_t N, float* out, int64_t ldo) {
    if (!e.any()) return;
    for (int64_t m = m0; m < m1; ++m) {
        float* o = out + m * ldo;
        const float* rr = e.res != nullptr ? e.res + m * e.res_ld : nullptr;
        for (int64_t n = 0; n < N; ++n) o[n] = epilogue_one(e, o[n], rr, n);
    }
}

}  // namespace onebit::audio

#endif
