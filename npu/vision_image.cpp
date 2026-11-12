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
#include "vision_image.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace onebit::npu::vision {

namespace {

double cubic(double x) {
    const double a = -0.5;
    if (x < 0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Taps {
    std::vector<int> first;       // per output index: the first input index
    std::vector<int> count;       // and how many taps
    std::vector<int32_t> weight;  // count taps each, `stride` apart
    int stride = 0, precision = 0;
};

Taps taps_for(int in, int out) {
    const double scale = double(in) / double(out);
    const double fs = scale > 1.0 ? scale : 1.0, support = 2.0 * fs, inv = 1.0 / fs;
    Taps t;
    t.stride = int(std::ceil(support)) * 2 + 1;
    t.first.resize(size_t(out));
    t.count.resize(size_t(out));
    std::vector<double> w(size_t(out) * t.stride, 0.0);
    double wmax = 0.0;
    for (int i = 0; i < out; ++i) {
        const double center = scale * (i + 0.5);
        int lo = int(center - support + 0.5);
        if (lo < 0) lo = 0;
        int hi = int(center + support + 0.5);
        if (hi > in) hi = in;
        const int n = hi - lo;
        double total = 0.0;
        for (int j = 0; j < n; ++j) {
            w[size_t(i) * t.stride + j] = cubic((j + lo - center + 0.5) * inv);
            total += w[size_t(i) * t.stride + j];
        }
        for (int j = 0; j < n; ++j) {
            if (total != 0.0) w[size_t(i) * t.stride + j] /= total;
            if (w[size_t(i) * t.stride + j] > wmax) wmax = w[size_t(i) * t.stride + j];
        }
        t.first[size_t(i)] = lo;
        t.count[size_t(i)] = n;
    }
    // the largest precision at which the biggest weight still fits a signed 16-bit integer
    for (int p = 0; p < 22; ++p) {
        if (int64_t(0.5 + wmax * double(int64_t(1) << (p + 1))) >= (1 << 15)) break;
        t.precision = p + 1;
    }
    t.weight.resize(w.size());
    for (size_t k = 0; k < w.size(); ++k) {
        const double v = w[k] * double(int64_t(1) << t.precision);
        t.weight[k] = int32_t(v < 0 ? -0.5 + v : 0.5 + v);
    }
    return t;
}

inline uint8_t clip8(int64_t v) { return uint8_t(v < 0 ? 0 : (v > 255 ? 255 : v)); }

}  // namespace

std::vector<uint8_t> resize_bicubic_u8(const uint8_t* rgb, int height, int width, int out_height, int out_width) {
    if (!rgb || height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0 || out_height > (1 << 16) || out_width > (1 << 16))
        throw std::runtime_error("cannot resize a " + std::to_string(height) + " x " + std::to_string(width) + " image to " +
                                 std::to_string(out_height) + " x " + std::to_string(out_width));
    // width first
    std::vector<uint8_t> mid;
    const uint8_t* src = rgb;
    if (out_width != width) {
        const Taps t = taps_for(width, out_width);
        const int64_t half = t.precision > 0 ? int64_t(1) << (t.precision - 1) : 0;
        mid.resize(size_t(height) * out_width * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < out_width; ++x) {
                const int32_t* w = t.weight.data() + size_t(x) * t.stride;
                const uint8_t* p = rgb + (size_t(y) * width + size_t(t.first[size_t(x)])) * 3;
                int64_t acc[3] = {half, half, half};
                for (int j = 0; j < t.count[size_t(x)]; ++j)
                    for (int c = 0; c < 3; ++c) acc[c] += int64_t(p[size_t(j) * 3 + c]) * w[j];
                for (int c = 0; c < 3; ++c) mid[(size_t(y) * out_width + x) * 3 + c] = clip8(acc[c] >> t.precision);
            }
        src = mid.data();
    }
    if (out_height == height) return src == rgb ? std::vector<uint8_t>(rgb, rgb + size_t(height) * width * 3) : mid;
    // then height
    const Taps t = taps_for(height, out_height);
    const int64_t half = t.precision > 0 ? int64_t(1) << (t.precision - 1) : 0;
    std::vector<uint8_t> out(size_t(out_height) * out_width * 3);
    const size_t row = size_t(out_width) * 3;
    for (int y = 0; y < out_height; ++y) {
        const int32_t* w = t.weight.data() + size_t(y) * t.stride;
        for (size_t k = 0; k < row; ++k) {
            int64_t acc = half;
            for (int j = 0; j < t.count[size_t(y)]; ++j) acc += int64_t(src[(size_t(t.first[size_t(y)]) + j) * row + k]) * w[j];
            out[size_t(y) * row + k] = clip8(acc >> t.precision);
        }
    }
    return out;
}

std::vector<Piece> preprocess(const uint8_t* rgb, int height, int width, const ImagePlan& plan) {
    if (plan.source.height <= 0 || plan.source.width <= 0 || plan.source.grid_h <= 0 || plan.source.grid_w <= 0)
        throw std::runtime_error("the image plan is empty");
    // one piece: the (h, w) window of an interleaved image at (y0, x0), as normalised planes
    const auto planes = [](const std::vector<uint8_t>& img, int img_w, int y0, int x0, const PiecePlan& q) {
        Piece p;
        p.grid_h = q.grid_h;
        p.grid_w = q.grid_w;
        p.pixels.resize(size_t(3) * q.height * q.width);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < q.height; ++y)
                for (int x = 0; x < q.width; ++x)
                    p.pixels[(size_t(c) * q.height + y) * q.width + x] =
                        (float(img[(size_t(y0 + y) * img_w + size_t(x0 + x)) * 3 + c]) - 127.5f) / 127.5f;
        return p;
    };
    std::vector<Piece> out;
    out.push_back(planes(resize_bicubic_u8(rgb, height, width, plan.source.height, plan.source.width), plan.source.width, 0, 0,
                         plan.source));
    if (plan.slice_rows > 0) {
        if (plan.refine_height != plan.slice.height * plan.slice_rows || plan.refine_width != plan.slice.width * plan.slice_cols)
            throw std::runtime_error("the image plan's slices do not tile its refine size");
        const std::vector<uint8_t> refine = resize_bicubic_u8(rgb, height, width, plan.refine_height, plan.refine_width);
        for (int r = 0; r < plan.slice_rows; ++r)
            for (int c = 0; c < plan.slice_cols; ++c)
                out.push_back(planes(refine, plan.refine_width, r * plan.slice.height, c * plan.slice.width, plan.slice));
    }
    return out;
}

}  // namespace onebit::npu::vision
