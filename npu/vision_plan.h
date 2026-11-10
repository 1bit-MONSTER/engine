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
// Where a MiniCPM-V 4.7 image goes before the vision tower sees it (engine#1788): the sizes, the slice grid, and how
// many rows each piece becomes on the text side. Arithmetic only; no pixel is touched here.
//
// The model looks at an image twice: once whole, resized to about `scale_resolution`^2 pixels, and, when the image is
// larger than that, again as a rows x cols grid of slices cut from a second, larger resize. Every piece is a multiple
// of 4 patches in both extents, because the tower merges 2 x 2 twice (npu/vision_tower.h), and becomes
// (height/patch/4) * (width/patch/4) rows.
//
// This is transformers' MiniCPMV4_6ImageProcessor (the image processor MiniCPM-V 4.7 loads), followed step by step,
// including the places where the obvious arithmetic gives a different answer:
//   - `round()` there is Python's: halves go to the EVEN neighbour (2.5 -> 2), which std::round does not do;
//   - `int()` truncates, and the width is computed from the already truncated height;
//   - the slice grid tries one piece fewer and one more than area / scale^2, and a tie goes to more rows;
//   - an image that is not sliced is scaled UP to the resolution; one that is sliced has its whole view scaled down only.
// (The per-slice size, refine / grid, is an exact division: the refine size was just rounded to a multiple of the grid.)
// tests/npu_vision_plan_test.cpp holds that implementation's answers for 3,481 image sizes.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::npu::vision {

struct PiecePlan {
    int height = 0, width = 0;  // pixels the piece is resized (source) or cut (slice) to
    int grid_h = 0, grid_w = 0; // patches
    int rows = 0;               // rows on the text side: grid_h/4 * grid_w/4
};

struct ImagePlan {
    PiecePlan source;                       // the whole image
    int slice_rows = 0, slice_cols = 0;     // 0 x 0: the image is not sliced
    int refine_height = 0, refine_width = 0; // the resize the slices are cut from (slice_rows x slice_cols equal pieces)
    PiecePlan slice;                        // every slice: they are all the same size
    // Everything the tower is run on, in the order the reference feeds it: the source, then the slices row by row.
    int pieces() const { return 1 + slice_rows * slice_cols; }
    int total_rows() const { return source.rows + slice_rows * slice_cols * slice.rows; }
};

namespace plan_detail {

// Python's round(): to nearest, halves to even. The default floating-point rounding mode is that rule.
inline int64_t py_round(double v) { return int64_t(std::nearbyint(v)); }

// max(round(length / divisor) * divisor, divisor)
inline int64_t ensure_divide(double length, int64_t divisor) {
    const int64_t r = py_round(length / double(divisor)) * divisor;
    return r > divisor ? r : divisor;
}

inline void best_resize(double height, double width, int scale, int patch, bool allow_upscale, int64_t* out_h, int64_t* out_w) {
    if (height * width > double(scale) * double(scale) || allow_upscale) {
        const double aspect = width / height;
        const double h = double(int64_t(double(scale) / std::sqrt(aspect)));  // int(): truncation
        height = h;
        width = double(int64_t(h * aspect));
    }
    const int64_t unit = int64_t(patch) * 4;  // two 2 x 2 merges
    *out_w = ensure_divide(width, unit);
    *out_h = ensure_divide(height, unit);
}

}  // namespace plan_detail

// The plan for a height x width image. The defaults are the model's (preprocessor_config.json: max_slice_nums 9; the
// image processor's scale_resolution 448; the tower's 14-pixel patch). Throws on a size that is not an image or that
// the arithmetic cannot hold.
inline ImagePlan plan_image(int height, int width, int max_slice_nums = 9, int scale_resolution = 448, int patch = 14) {
    using namespace plan_detail;
    if (height <= 0 || width <= 0 || height > (1 << 20) || width > (1 << 20))
        throw std::runtime_error("an image of " + std::to_string(height) + " x " + std::to_string(width) + " pixels cannot be planned");
    if (max_slice_nums < 1 || max_slice_nums > 64 || scale_resolution < 4 * patch || scale_resolution > 8192 || patch < 1)
        throw std::runtime_error("the image plan's parameters are out of range");
    ImagePlan p;
    const double H = height, W = width;

    // the slice grid: the factorisation of about (area / scale^2) pieces whose shape is closest to the image's
    const double log_ratio = std::log(W / H);
    const double ratio = W * H / (double(scale_resolution) * double(scale_resolution));
    const int64_t ceil_ratio = int64_t(std::ceil(ratio));
    const int multiple = int(ceil_ratio < max_slice_nums ? ceil_ratio : max_slice_nums);
    if (multiple > 1) {
        int best_rows = 1, best_cols = 1;
        double min_error = INFINITY;
        for (int n = multiple - 1; n <= multiple + 1; ++n) {
            if (n == 1 || n > max_slice_nums) continue;
            for (int rows = 1; rows <= n; ++rows) {
                if (n % rows != 0) continue;
                const int cols = n / rows;
                const double error = std::fabs(log_ratio - std::log(double(cols) / double(rows)));
                if (error < min_error || (error == min_error && rows > best_rows)) {
                    best_rows = rows;
                    best_cols = cols;
                    min_error = error;
                }
            }
        }
        p.slice_rows = best_rows;
        p.slice_cols = best_cols;
    }
    const bool sliced = p.slice_rows > 0;

    const auto piece = [&](int64_t h, int64_t w) {
        if (h > (int64_t(1) << 24) || w > (int64_t(1) << 24)) throw std::runtime_error("the image plan does not fit");
        PiecePlan q;
        q.height = int(h);
        q.width = int(w);
        q.grid_h = q.height / patch;
        q.grid_w = q.width / patch;
        q.rows = (q.grid_h / 4) * (q.grid_w / 4);
        return q;
    };
    int64_t sh = 0, sw = 0;
    best_resize(H, W, scale_resolution, patch, !sliced, &sh, &sw);  // a small image is scaled UP to the resolution
    p.source = piece(sh, sw);

    if (sliced) {
        const int64_t refine_w = ensure_divide(W, p.slice_cols), refine_h = ensure_divide(H, p.slice_rows);
        int64_t bh = 0, bw = 0;
        best_resize(double(refine_h) / double(p.slice_rows), double(refine_w) / double(p.slice_cols), scale_resolution, patch, true,
                    &bh, &bw);
        p.slice = piece(bh, bw);
        p.refine_height = p.slice.height * p.slice_rows;
        p.refine_width = p.slice.width * p.slice_cols;
    }
    return p;
}

}  // namespace onebit::npu::vision
