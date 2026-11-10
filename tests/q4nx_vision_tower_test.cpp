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
// The assembled vision tower (engine#1503 W4): patch embed -> position embedding -> 2 blocks -> merger, in merge-block token
// order. This is the composition test the three earlier PRs could not give: it is the only place where the token order between
// the patch embed, the position embedding and the blocks has to agree. Goldens come from the numpy reference that reproduces
// the pack's graph at corr 1.000000 against the fp32 oracle; the weights come from the same 32-bit LCG so the fixture is
// regenerable in another language.
#include "q4nx_vision.h"

#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

static uint32_t g_state = 999u;
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

int main() {
    const int T = 4, H = 16, NH = 2, HD = 8, IM = 32, OUT = 8, PD = 24, GH = 2, GW = 2, NB = 2;
    // the same stream, in the same order, as the numpy reference
    std::vector<float> patches = take_n(size_t(T) * PD);
    std::vector<float> convw = take_n(size_t(H) * PD), convb = take_n(H);
    std::vector<float> table = take_n(size_t(48) * 48 * H);
    // one flat container, filled before any pointer is taken, so no vector can reallocate under the pointers
    std::vector<float> store;
    std::vector<std::array<size_t, 12>> off(static_cast<size_t>(NB));
    const size_t sizes[12] = {H, H, size_t(3 * H) * H, size_t(3 * H), size_t(H) * H, H,
                              H, H, size_t(IM) * H, size_t(IM), size_t(H) * IM, H};
    for (int b = 0; b < NB; ++b)
        for (int k = 0; k < 12; ++k) {
            off[size_t(b)][size_t(k)] = store.size();
            auto v = take_n(sizes[k]);
            store.insert(store.end(), v.begin(), v.end());
        }
    std::vector<q4nx_vision::VisionBlockWeights> blocks(static_cast<size_t>(NB));
    for (int b = 0; b < NB; ++b) {
        auto P = [&](int k) { return store.data() + off[size_t(b)][size_t(k)]; };
        blocks[size_t(b)] = q4nx_vision::VisionBlockWeights{P(0), P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(10), P(11)};
    }
    std::vector<float> mlnw = take_n(H), mlnb = take_n(H);
    std::vector<float> m1w = take_n(size_t(4 * H) * 4 * H), m1b = take_n(4 * H);
    std::vector<float> m2w = take_n(size_t(OUT) * 4 * H), m2b = take_n(OUT);

    q4nx_vision::VisionTowerWeights w{convw.data(), convb.data(), table.data(), blocks.data(), NB,
                                      mlnw.data(), mlnb.data(), m1w.data(), m1b.data(), m2w.data(), m2b.data()};
    const float inv[2] = {1.0f, 0.5f};
    std::vector<float> out(size_t(T / 4) * OUT);
    q4nx_vision::tower_forward(patches.data(), T, PD, GH, GW, H, NH, HD, IM, OUT, w, inv, out.data());

    const double want[8] = {-0.19714565, -0.03816292, -0.08239722, 0.27405666,
                            0.34750172, -0.23880654, 0.19937676, -0.04093523};
    for (int i = 0; i < OUT; ++i)
        if (std::fabs(double(out[size_t(i)]) - want[i]) > 2e-5) {
            std::printf("FAIL tower golden: [%d] got %.8f wanted %.8f\n", i, out[size_t(i)], want[i]);
            ++fails;
            break;
        }
    double sum = 0.0;
    for (float v : out) sum += v;
    check(std::fabs(sum - 0.223488) < 1e-4, "tower golden: the sum of the 8 outputs is 0.223488");

    // control: a tower with every weight zero returns the merger's bias exactly, whatever the blocks and the norms did.
    // Every zero matrix is SIZED: passing one element where a matrix is expected reads past the end (this arm's first version
    // did exactly that and failed, which is the arm doing its job).
    {
        std::vector<float> z_conv(size_t(H) * PD, 0.0f), z_convb(H, 0.0f), z_tab(size_t(48) * 48 * H, 0.0f);
        std::vector<float> z_qkv(size_t(3 * H) * H, 0.0f), z_qkvb(size_t(3 * H), 0.0f);
        std::vector<float> z_proj(size_t(H) * H, 0.0f), z_projb(H, 0.0f);
        std::vector<float> z_fc1(size_t(IM) * H, 0.0f), z_fc1b(IM, 0.0f);
        std::vector<float> z_fc2(size_t(H) * IM, 0.0f), z_fc2b(H, 0.0f);
        std::vector<q4nx_vision::VisionBlockWeights> zb(static_cast<size_t>(NB),
            q4nx_vision::VisionBlockWeights{nullptr, nullptr, z_qkv.data(), z_qkvb.data(), z_proj.data(), z_projb.data(),
                                            nullptr, nullptr, z_fc1.data(), z_fc1b.data(), z_fc2.data(), z_fc2b.data()});
        std::vector<float> z_m1(size_t(4 * H) * 4 * H, 0.0f), z_m1b(4 * H, 0.0f);
        std::vector<float> z_m2(size_t(OUT) * 4 * H, 0.0f), z_m2b(OUT, 0.25f);
        q4nx_vision::VisionTowerWeights zw{z_conv.data(), z_convb.data(), z_tab.data(), zb.data(), NB, nullptr, nullptr,
                                          z_m1.data(), z_m1b.data(), z_m2.data(), z_m2b.data()};
        std::vector<float> o2(size_t(T / 4) * OUT);
        q4nx_vision::tower_forward(patches.data(), T, PD, GH, GW, H, NH, HD, IM, OUT, zw, inv, o2.data());
        bool all = true;
        for (float v : o2) all = all && std::fabs(v - 0.25f) < 1e-6f;
        check(all, "control: zero weights return the merger's bias exactly");
    }

    // A token count that disagrees with the grid is refused. tower_forward() refuses by printing and returning, so the
    // only thing a caller can observe is that `out` was never written: the buffer is filled with a value no output can
    // take and must come back untouched (review of #1930 at a7af2fd: this arm used to be `check(true, ...)`).
    // 2 * T tokens on a grid of T, on purpose: positive and a multiple of 4, so neither the `tokens <= 0` test nor
    // merger_forward()'s own guard stops it, and only `gh * gw != tokens` stands between this call and a written `out`.
    // Both buffers are sized for 2 * T tokens, so the call stays inside them even if that check were removed.
    {
        std::vector<float> more(size_t(2 * T) * PD, 0.25f);
        std::vector<float> o3(size_t(2 * T / 4) * OUT, 1234.0f);
        q4nx_vision::tower_forward(more.data(), 2 * T, PD, GH, GW, H, NH, HD, IM, OUT, w, inv, o3.data());
        bool untouched = true;
        for (float v : o3) untouched = untouched && v == 1234.0f;
        check(untouched, "a token count that disagrees with gh*gw is refused: the output buffer is never written");
    }

    if (fails == 0) std::printf("vision tower: 4 arms pass (golden x2, zero-weight control, refusal)\n");
    return fails == 0 ? 0 : 1;
}
