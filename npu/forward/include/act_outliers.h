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
// act_outliers.h -- the outlier-channel split for the int8 ffn_down GEMM (engine#2536).
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
#include <cstring>
#include <utility>
#include <vector>

namespace onebit::npu {

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

}  // namespace onebit::npu
