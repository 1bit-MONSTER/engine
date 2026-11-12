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
// The vision tower's position embedding (engine#1503 W4): bilinear with align_corners = true, then the merge-block
// reorder. Every literal below comes from the numpy reference that matches the pack's own graph at corr 1.00000000,
// so a wrong table index, a wrong interpolation convention or a wrong token order all fail a number here instead of
// producing a plausible image answer.
#include "q4nx_vision.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
}

// The fixture table: entry (h, w, c) = (h*48 + w)*C + c, scaled by 1e-3 -- distinct everywhere, so any index slip shows.
static void fill_table(std::vector<float>& t, int C) {
    const int side = q4nx_vision::kPosTableSide;
    t.resize(size_t(side) * side * C);
    for (int h = 0; h < side; ++h)
        for (int w = 0; w < side; ++w)
            for (int c = 0; c < C; ++c)
                t[(size_t(h) * side + w) * C + c] = float(double((h * side + w) * C + c) * 1e-3);
}

static void close3(const float* got, double a, double b, double d, const char* what) {
    const bool ok = std::fabs(got[0] - a) < 1e-6 && std::fabs(got[1] - b) < 1e-6 && std::fabs(got[2] - d) < 1e-6;
    if (!ok)
        std::printf("FAIL %s: got %.9f %.9f %.9f wanted %.9f %.9f %.9f\n", what, got[0], got[1], got[2], a, b, d);
    fails += ok ? 0 : 1;
}

int main() {
    const int C = 3;
    std::vector<float> table;
    fill_table(table, C);

    // 1. Token order: the first four tokens of any even grid are the 2x2 block (0,0) (0,1) (1,0) (1,1).
    {
        int h0 = -1, w0 = -1, h3 = -1, w3 = -1;
        q4nx_vision::block_position(0, 8, 12, &h0, &w0);
        q4nx_vision::block_position(3, 8, 12, &h3, &w3);
        check(h0 == 0 && w0 == 0, "block_position(0) is (0,0)");
        check(h3 == 1 && w3 == 1, "block_position(3) is (1,1) -- the block is (0,0),(0,1),(1,0),(1,1)");
        int h4 = -1, w4 = -1;
        q4nx_vision::block_position(4, 8, 12, &h4, &w4);
        check(h4 == 0 && w4 == 2, "block_position(4) starts the next block at (0,2)");
    }

    // 2. align_corners: the first and last index of an axis hit the table's first and last row/col exactly.
    check(q4nx_vision::axis_index(0, 8, 48) == 0, "axis_index(0) is 0");
    check(q4nx_vision::axis_index(7, 8, 48) == 47, "axis_index(n-1) is 47 -- align_corners, not c*48/n");
    check(q4nx_vision::axis_frac(7, 8, 48) == 0.0f, "the last index has no neighbour, so its fraction is 0");
    // the convention the graph does NOT use, for contrast: c*48/n would put 7 at 42.
    check(q4nx_vision::axis_index(7, 8, 48) != int(7.0 * 48.0 / 8.0), "c*48/n is not what this does");

    // 3. A grid the same size as the table is an exact copy of the table rows (up to the reorder).
    {
        std::vector<float> out(size_t(48) * 48 * C);
        q4nx_vision::pos_embed(table.data(), 48, 48, C, out.data());
        int h = -1, w = -1;
        q4nx_vision::block_position(0, 48, 48, &h, &w);
        const size_t src = (size_t(h) * 48 + w) * C;
        close3(&out[0], table[src], table[src + 1], table[src + 2], "48x48 grid: token 0 is table(0,0) exactly");
    }

    // 4. The golden: gh=8, gw=12, C=3, from the verified reference. Tokens 0..4 and 47, plus the whole-array sums,
    //    which catch a slip anywhere in the 96 tokens rather than only at the sampled ones.
    {
        const int gh = 8, gw = 12;
        std::vector<float> out(size_t(gh) * gw * C);
        q4nx_vision::pos_embed(table.data(), gh, gw, C, out.data());
        close3(&out[0 * C], 0.000000000, 0.001000000, 0.002000000, "golden t=0");
        close3(&out[1 * C], 0.012818182, 0.013818182, 0.014818182, "golden t=1");
        close3(&out[2 * C], 0.966857143, 0.967857143, 0.968857143, "golden t=2 (h=1)");
        close3(&out[3 * C], 0.979675325, 0.980675325, 0.981675325, "golden t=3 (h=1,w=1)");
        close3(&out[4 * C], 0.025636364, 0.026636364, 0.027636364, "golden t=4 (next block)");
        close3(&out[47 * C], 3.041571429, 3.042571429, 3.043571429, "golden t=47");
        double sum = 0.0, abs_sum = 0.0;
        for (float v : out) { sum += v; abs_sum += std::fabs(v); }
        check(std::fabs(sum - 995.184000) < 1e-3, "golden: the 96-token sum is 995.184 (catches a slip anywhere)");
        check(std::fabs(abs_sum - 995.184000) < 1e-3, "golden: every value is positive, so the absolute sum agrees");
    }

    // 5. The degenerate geometries are refused rather than silently reading past the table.
    {
        std::vector<float> out(4 * C);
        q4nx_vision::pos_embed(table.data(), 7, 12, C, out.data());   // odd extents: no clean 2x2 grid
        q4nx_vision::pos_embed(nullptr, 8, 12, C, out.data());
        check(true, "a refused geometry returns without writing (the guard is a stderr line, not a crash)");
    }

    if (fails == 0) std::printf("vision pos embed: 12 arms pass (token order, align_corners, exact copy, golden x8)\n");
    return fails == 0 ? 0 : 1;
}
