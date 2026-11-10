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
// npu_vision_plan_test
//
// The image plan (npu/vision_plan.h, engine#1788) against transformers' MiniCPMV4_6ImageProcessor 5.22.2: the methods
// find_best_resize, get_refine_size and get_sliced_grid, called as the processor's _preprocess calls them, for every
// pair of the 59 sizes below (3,481 images). The eight numbers of each plan are folded into one FNV-1a hash, compared
// with the hash the reference produces for the same walk; sixteen plans are also written out so a failure says where.
#include "../npu/vision_plan.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace vis = onebit::npu::vision;

int g_fail = 0, g_arms = 0;
void check(bool ok, const std::string& what) {
    ++g_arms;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL %s\n", what.c_str());
    }
}

const int kSizes[] = {1,    2,    3,    7,    13,   14,   27,   28,   55,   56,   57,   100,  111,  224,  300,
                      336,  392,  419,  420,  447,  448,  449,  476,  500,  503,  504,  600,  640,  672,  700,
                      720,  768,  800,  896,  900,  960,  1000, 1024, 1080, 1200, 1280, 1344, 1440, 1536, 1600,
                      1792, 1920, 2000, 2048, 2160, 2560, 3000, 3024, 3840, 4032, 4096, 5000, 6000, 8000};

// {height, width, {source h, source w, slice rows, slice cols, refine h, refine w, slice h, slice w}}, from the reference
struct Row {
    int h, w, plan[8];
};
const Row kRows[] = {
    {448, 448, {448, 448, 0, 0, 0, 0, 0, 0}},
    {1, 1, {448, 448, 0, 0, 0, 0, 0, 0}},
    {100, 100, {448, 448, 0, 0, 0, 0, 0, 0}},
    {449, 448, {448, 448, 2, 1, 672, 616, 336, 616}},
    {480, 640, {392, 504, 1, 2, 560, 784, 560, 392}},
    {720, 1280, {336, 616, 2, 3, 784, 1512, 392, 504}},
    {1080, 1920, {336, 616, 2, 4, 896, 1792, 448, 448}},
    {1920, 1080, {616, 336, 4, 2, 1792, 896, 448, 448}},
    {3024, 4032, {392, 504, 3, 3, 1176, 1512, 392, 504}},
    {224, 2000, {168, 1344, 1, 4, 280, 2688, 280, 672}},
    {5000, 300, {1848, 112, 9, 1, 5544, 336, 616, 336}},
    {896, 896, {448, 448, 2, 2, 896, 896, 448, 448}},
    {1344, 1344, {448, 448, 3, 3, 1344, 1344, 448, 448}},
    {600, 800, {392, 504, 2, 2, 784, 1008, 392, 504}},
    {8000, 8000, {448, 448, 3, 3, 1344, 1344, 448, 448}},
    {14, 8000, {56, 10304, 0, 0, 0, 0, 0, 0}},
};
const uint64_t kReferenceHash = 0x8480087394bf204fULL;  // the walk below, by the reference
const int kReferenceMaxRows = 716;                      // the most text rows any of the 3,481 plans gives

void eight(const vis::ImagePlan& p, int out[8]) {
    out[0] = p.source.height;
    out[1] = p.source.width;
    out[2] = p.slice_rows;
    out[3] = p.slice_cols;
    out[4] = p.refine_height;
    out[5] = p.refine_width;
    out[6] = p.slice.height;
    out[7] = p.slice.width;
}

}  // namespace

int main() {
    for (const Row& r : kRows) {
        int got[8];
        eight(vis::plan_image(r.h, r.w), got);
        bool same = true;
        for (int i = 0; i < 8; ++i) same = same && got[i] == r.plan[i];
        check(same, std::to_string(r.h) + " x " + std::to_string(r.w) + ": source " + std::to_string(got[0]) + " x " +
                        std::to_string(got[1]) + ", grid " + std::to_string(got[2]) + " x " + std::to_string(got[3]) + ", refine " +
                        std::to_string(got[4]) + " x " + std::to_string(got[5]) + ", slice " + std::to_string(got[6]) + " x " +
                        std::to_string(got[7]) + " is the reference's plan");
    }

    uint64_t h = 0xcbf29ce484222325ULL;
    int max_rows = 0, pairs = 0;
    bool shapes_ok = true;
    for (int height : kSizes)
        for (int width : kSizes) {
            const vis::ImagePlan p = vis::plan_image(height, width);
            int v[8];
            eight(p, v);
            for (int x : v)
                for (int b = 0; b < 4; ++b) h = (h ^ uint64_t((uint32_t(x) >> (8 * b)) & 0xff)) * 0x100000001b3ULL;
            if (p.total_rows() > max_rows) max_rows = p.total_rows();
            ++pairs;
            // what the tower needs of every piece: both patch extents multiples of 4, and at least one row
            const auto ok = [](const vis::PiecePlan& q) {
                return q.grid_h > 0 && q.grid_w > 0 && q.grid_h % 4 == 0 && q.grid_w % 4 == 0 && q.height == q.grid_h * 14 &&
                       q.width == q.grid_w * 14 && q.rows == q.grid_h * q.grid_w / 16;
            };
            shapes_ok = shapes_ok && ok(p.source) && (p.slice_rows == 0 || ok(p.slice)) && p.slice_rows * p.slice_cols <= 9 &&
                        p.pieces() == 1 + p.slice_rows * p.slice_cols;
        }
    std::printf("  %d image sizes: hash 0x%016llx (reference 0x%016llx), most text rows %d\n", pairs, (unsigned long long)h,
                (unsigned long long)kReferenceHash, max_rows);
    check(h == kReferenceHash, "all 3,481 plans are the reference's (one hash over every number)");
    check(max_rows == kReferenceMaxRows, "the largest plan gives the reference's row count");
    check(shapes_ok, "every piece of every plan is a multiple of 4 patches in both extents, and at most 9 slices");

    {
        const vis::ImagePlan p = vis::plan_image(1080, 1920);
        check(p.pieces() == 9 && p.source.rows == 6 * 11 && p.slice.rows == 64 && p.total_rows() == 66 + 8 * 64,
              "1080 x 1920: one 24 x 44 source (66 rows) and eight 32 x 32 slices (64 rows each), 578 rows");
    }
    for (const auto& [hh, ww] : std::vector<std::pair<int, int>>{{0, 100}, {100, 0}, {-1, 5}, {1 << 21, 10}}) {
        std::string why;
        try { vis::plan_image(hh, ww); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("cannot be planned") != std::string::npos, std::to_string(hh) + " x " + std::to_string(ww) + " is refused");
    }
    std::printf("vision plan: %d checks, %d failed\n", g_arms, g_fail);
    return g_fail ? 1 : 0;
}
