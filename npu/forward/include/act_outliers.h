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
// act_outliers.h -- the outlier-channel split for the int8 FFN GEMMs (engine#2536 ffn_down, engine#2576 gate/up).
//
// The i8 GEMM path quantises each activation row with ONE scale, amax/127. On Qwen2.5-3B the ffn_down input rows have
// amax/rms of 38 (median over the 36 layers) to 100 (layer 35, one channel at 1200), so a typical channel keeps 1-3 int8
// levels. That turned "July 20, 1969" into "1 lick this out 1969" on both AMD Qwen2.5-3B packs. The weights were
// fine (the f32 host GEMM on the same container answered "969,"), and so were the kernels (they matched the host i8 arm).
//
// The split, LLM.int8-style: the channels of a row with |x| > tau * rms(row), at most kmax of them, leave the int8
// activation (zeroed, so the scale is set by the rest) and their x_c * W[:, c] is added on the host in f32 from the
// layer's own weight column. A row with no such channel is untouched, so a model without outliers runs the plain path.
//
// This header holds the parts that need no device: picking the channels, and gathering one 256-wide column block of a
// tile tensor so a column can be decoded without decoding the whole tensor. q4nx_forward.cpp owns the GEMM wiring.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace onebit::npu {

// The defaults, in one place so act_outliers_test can assert them (engine#2585 review): tau 20 (docs/npu.md has the
// table; 40 left Llama-2-7b-chat 1.7.1 without its EOS once gate/up were covered) and at most 16 channels per row.
constexpr float kActOutlierTauDefault = 20.0f;
constexpr int kActOutlierKmaxDefault = 16;

// NPU_ACT_OUTLIER_TAU as read: unset gives the default, "0" turns the split off.
inline float act_outlier_tau_from(const char* env) { return env ? (float)std::atof(env) : kActOutlierTauDefault; }
// NPU_ACT_OUTLIER_KMAX as read: unset gives the default; anything below 1 is clamped to 1.
inline int act_outlier_kmax_from(const char* env) {
    if (!env) return kActOutlierKmaxDefault;
    const int k = std::atoi(env);
    return k < 1 ? 1 : k;
}

struct ActOutlier {
    int row;   // activation row
    int ch;    // input channel (the weight column)
    float x;   // its value, before it is zeroed
};

// Append the outliers of the M rows of A (row stride `stride`, real width `w`) to `out`: per row, the channels with
// |x| > tau * rms(row), keeping the kmax largest. tau <= 0 picks nothing.
inline void act_outlier_pick(const float* A, int M, int stride, int w, float tau, int kmax,
                             std::vector<ActOutlier>& out) {
    if (!A || M <= 0 || w <= 0 || tau <= 0.0f || kmax <= 0) return;
    std::vector<std::pair<float, int>> c;
    for (int m = 0; m < M; m++) {
        const float* r = A + (size_t)m * (size_t)stride;
        double ss = 0.0;
        for (int k = 0; k < w; k++) ss += double(r[k]) * double(r[k]);
        const float thr = tau * (float)std::sqrt(ss / (double)w);
        c.clear();
        for (int k = 0; k < w; k++)
            if (std::fabs(r[k]) > thr) c.push_back({std::fabs(r[k]), k});
        if ((int)c.size() > kmax) {
            std::partial_sort(c.begin(), c.begin() + kmax, c.end(),
                              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
            c.resize((size_t)kmax);
        }
        for (const auto& e : c) out.push_back({m, e.second, r[e.second]});
    }
}

// Copy the tiles of column block `tc` (every tile-row's tile tc) of a [TR x TC] row-major grid of `cb`-byte tiles into
// `buf`, in tile-row order: the tensor a [TR*32, 256] decode of that block reads.
inline void act_outlier_gather_block(const uint8_t* raw, int TR, int TC, int cb, int tc, std::vector<uint8_t>& buf) {
    buf.resize((size_t)TR * (size_t)cb);
    for (int tr = 0; tr < TR; tr++)
        std::memcpy(&buf[(size_t)tr * (size_t)cb], raw + ((size_t)tr * (size_t)TC + (size_t)tc) * (size_t)cb, (size_t)cb);
}

// Add the split-out channels back on the host: for every pick, out[row][out_off + n] += x * col(ch)[n], n < rows (and
// within N). `out_off` is where this tensor's outputs start in the GEMM's N: 0 for a single projection, gate_rows for
// the `up` half of a fused [gate | up] design (engine#2576).
template <class ColFn>
inline void act_outlier_correct(float* out, int N, const std::vector<ActOutlier>& picks, int out_off, int rows,
                                ColFn col) {
    if (!out || N <= 0 || out_off < 0 || out_off >= N) return;
    const int nn = std::min(rows, N - out_off);
    for (const auto& pk : picks) {
        const float* c = col(pk.ch);
        if (!c) continue;
        float* y = out + (size_t)pk.row * (size_t)N + (size_t)out_off;
        for (int n = 0; n < nn; n++) y[n] += pk.x * c[n];
    }
}

}  // namespace onebit::npu
