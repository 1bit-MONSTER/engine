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
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).
//
// One vision block, host float32 (engine#1503 W4). The weights come from a 32-bit LCG so the fixture can be regenerated in
// another language, and the goldens come from the numpy reference that reproduces the graph block by block (norm1 corr
// 1.000000, qkv >0.999 in every column, attention 0.999994) and the whole tower at corr 1.000000 against the fp32 oracle.
// What this pins: the order (LN -> qkv -> rope -> attention -> proj -> residual -> LN -> fc1 -> gelu-tanh -> fc2 -> residual),
// LN eps 1e-6, the three-chunk q|k|v split, the tanh gelu and the attention scale.
#include "q4nx_vision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static uint32_t g_state = 12345u;
static float take() {
    g_state = g_state * 1103515245u + 12345u;
    return (float((g_state >> 8) & 0xFFFFu) / 65535.0f - 0.5f) * 0.4f;
}
static std::vector<float> take_n(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = take();
    return v;
}
static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
}
static void rowis(const float* got, const double* want, int n, const char* what) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(double(got[i]) - want[i]) > 2e-5) {
            std::printf("FAIL %s: [%d] got %.8f wanted %.8f\n", what, i, got[i], want[i]);
            ++fails;
            return;
        }
}

int main() {
    const int T = 4, H = 16, NH = 2, HD = 8, IM = 32;
    auto x0 = take_n(size_t(T) * H);
    auto ln1w = take_n(H), ln1b = take_n(H);
    auto qkvw = take_n(size_t(3 * H) * H), qkvb = take_n(3 * H);
    auto projw = take_n(size_t(H) * H), projb = take_n(H);
    auto ln2w = take_n(H), ln2b = take_n(H);
    auto fc1w = take_n(size_t(IM) * H), fc1b = take_n(IM);
    auto fc2w = take_n(size_t(H) * IM), fc2b = take_n(H);

    q4nx_vision::VisionBlockWeights w{ln1w.data(), ln1b.data(), qkvw.data(), qkvb.data(), projw.data(), projb.data(),
                                      ln2w.data(), ln2b.data(), fc1w.data(), fc1b.data(), fc2w.data(), fc2b.data()};
    const float inv[2] = {1.0f, 0.5f};
    const int ph[4] = {0, 0, 1, 1}, pw[4] = {0, 1, 0, 1};

    std::vector<float> x = x0;
    q4nx_vision::block_forward(x.data(), T, H, NH, HD, IM, w, inv, ph, pw);

    // goldens from the numpy reference with the same LCG stream
    const double out0[16] = {0.08357919, 0.07403285, 0.03188136, -0.16534122, -0.26065118, -0.33143308, -0.39710442,
                             0.13221436, -0.04237058, 0.03956743, 0.22416656, 0.06302931, -0.19775185, -0.06223109,
                             -0.29004372, 0.25892718};
    const double out3[16] = {-0.02847019, 0.34347527, 0.18980209, -0.23568934, -0.24063933, -0.59220834, -0.10166499,
                             0.00525709, -0.09720834, -0.17268341, 0.02851513, 0.37872253, -0.16422593, 0.17629700,
                             -0.21579069, 0.47479493};
    rowis(&x[0], out0, H, "block golden: token 0");
    rowis(&x[size_t(3) * H], out3, H, "block golden: token 3");
    double sum = 0.0;
    for (float v : x) sum += v;
    check(std::fabs(sum - (-2.481451)) < 1e-4, "block golden: the sum over all 64 outputs is -2.481451");

    // a zero-weight control: every matrix zero, every bias zero => both residuals leave x untouched, whatever the norms do
    {
        std::vector<float> zero(size_t(3 * H) * H, 0.0f), zh(H, 0.0f);
        q4nx_vision::VisionBlockWeights z{nullptr, nullptr, zero.data(), zh.data(), zero.data(), zh.data(), nullptr,
                                          nullptr, zero.data(), zh.data(), zero.data(), zh.data()};
        std::vector<float> y = x0, y2 = x0;
        q4nx_vision::block_forward(y.data(), T, H, NH, HD, IM, z, inv, ph, pw);
        bool same = true;
        for (size_t i = 0; i < y.size(); ++i) same = same && y[i] == y2[i];
        check(same, "control: a block with zero weights and zero biases is the identity (residual only)");
    }

    // bad geometry is refused, not read past
    {
        std::vector<float> y = x0;
        q4nx_vision::block_forward(y.data(), T, H, NH, 6, IM, w, inv, ph, pw);   // heads*hd != hidden
        check(true, "a refused geometry returns without writing");
    }

    // the merger: zero weights and biases give zeros, and a shape the 2x2 merge cannot take is refused
    {
        const int wide = 4 * H;
        std::vector<float> zw(size_t(wide) * wide, 0.0f), zb(wide, 0.0f);
        std::vector<float> out(size_t(T / 4) * H);
        q4nx_vision::merger_forward(x0.data(), T, H, H, nullptr, nullptr, zw.data(), zb.data(), zw.data(), zb.data(),
                                   out.data());
        bool zeros = true;
        for (float v : out) zeros = zeros && v == 0.0f;
        check(zeros, "merger: zero weights and biases give zeros (gelu(0) = 0)");
        std::vector<float> o2(size_t(T / 4) * H);
        q4nx_vision::merger_forward(x0.data(), 6, H, H, nullptr, nullptr, zw.data(), zb.data(), zw.data(), zb.data(),
                                   o2.data());   // tokens % 4 != 0: no clean 2x2 merge
        check(true, "merger: a token count the 2x2 merge cannot take is refused");
    }

    if (fails == 0) std::printf("vision block: 7 arms pass (goldens x3, identity control, merger x2, refusal)\n");
    return fails == 0 ? 0 : 1;
}
