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
// npu_vision_image_test
//
// The image path before the tower (npu/vision_image.h, engine#1788).
//
// resize_bicubic_u8 against torch 2.13's torch.nn.functional.interpolate(uint8, mode="bicubic", antialias=True): for
// each case the image below is resized by both, and the reference's bytes are held here as an FNV-1a hash of the whole
// output plus three sample bytes. The image is integer arithmetic, so it is the same picture in any language:
//   byte(y, x, c) = (x*(7+c) + y*(13-c) + ((x*y) % 31)*5 + c*29 + ((x/9 + y/5) % 2)*90) & 255
// preprocess(): the pieces, their order, and that each value is (byte - 127.5) / 127.5 of the resized image.
#include "../npu/vision_image.h"
#include "../npu/vision_tower.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

std::vector<uint8_t> image(int h, int w) {
    std::vector<uint8_t> v(size_t(h) * w * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                v[(size_t(y) * w + x) * 3 + c] =
                    uint8_t((x * (7 + c) + y * (13 - c) + ((x * y) % 31) * 5 + c * 29 + ((x / 9 + y / 5) % 2) * 90) & 255);
    return v;
}

uint64_t fnv(const std::vector<uint8_t>& b) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint8_t v : b) h = (h ^ v) * 0x100000001b3ULL;
    return h;
}

// {height, width, out height, out width, hash of the reference's output, {byte (0,0,0), byte (oh/2, ow/3, 1), byte (oh-1, ow-1, 2)}}
struct Case {
    int h, w, oh, ow;
    uint64_t hash;
    int sample[3];
};
const Case kCases[] = {
    {37, 53, 56, 112, 0x183132e60e013115ULL, {0, 52, 214}},     // up in both
    {480, 640, 392, 504, 0xa3482dec42670b72ULL, {3, 169, 175}},  // down in both: a 480 x 640 image's whole view
    {97, 211, 448, 448, 0x71de32691a84dd1fULL, {0, 202, 255}},   // up in one, up more in the other
    {300, 200, 168, 112, 0xd94cee2b35fef077ULL, {9, 201, 197}},
    {64, 64, 56, 56, 0x909ccbe6090e462dULL, {2, 161, 119}},      // a slight shrink
    {1000, 30, 224, 56, 0x1a51666a837eb165ULL, {25, 137, 139}},  // down 4.5x in one, up in the other
    {50, 50, 50, 112, 0x5bf628c4db3c6e45ULL, {0, 71, 79}},       // height unchanged: that pass is skipped
    {56, 56, 56, 56, 0x97d7e9db2a2df6b3ULL, {0, 127, 58}},       // nothing to do: the image comes back
    {3, 5, 56, 56, 0xfe99d14c4912f969ULL, {0, 56, 162}},         // fewer input pixels than taps
    {1, 1, 448, 448, 0x9053309dc853d325ULL, {0, 29, 58}},
    {480, 640, 560, 784, 0x4c864ee63453d9d4ULL, {0, 91, 158}},   // the same image's refine size (1 x 2 slices)
};

}  // namespace

int main() {
    for (const Case& c : kCases) {
        const std::vector<uint8_t> in = image(c.h, c.w);
        const std::vector<uint8_t> out = vis::resize_bicubic_u8(in.data(), c.h, c.w, c.oh, c.ow);
        const std::string name = std::to_string(c.h) + " x " + std::to_string(c.w) + " -> " + std::to_string(c.oh) + " x " + std::to_string(c.ow);
        const bool sized = out.size() == size_t(c.oh) * c.ow * 3;
        check(sized, name + ": the output has the asked size");
        if (!sized) continue;
        const int s0 = out[0], s1 = out[(size_t(c.oh / 2) * c.ow + size_t(c.ow / 3)) * 3 + 1],
                  s2 = out[(size_t(c.oh - 1) * c.ow + size_t(c.ow - 1)) * 3 + 2];
        check(s0 == c.sample[0] && s1 == c.sample[1] && s2 == c.sample[2],
              name + ": three sample bytes are the reference's (" + std::to_string(s0) + ", " + std::to_string(s1) + ", " + std::to_string(s2) + ")");
        check(fnv(out) == c.hash, name + ": every byte is the reference's");
    }
    {
        std::string why;
        const std::vector<uint8_t> in = image(4, 4);
        try { vis::resize_bicubic_u8(in.data(), 4, 4, 0, 8); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("cannot resize") != std::string::npos, "an empty output size is refused");
    }

    // preprocess: a 480 x 640 image is one whole view (392 x 504) and 1 x 2 slices of 560 x 392 cut from 560 x 784
    {
        const int h = 480, w = 640;
        const std::vector<uint8_t> in = image(h, w);
        const vis::ImagePlan plan = vis::plan_image(h, w);
        const std::vector<vis::Piece> pieces = vis::preprocess(in.data(), h, w, plan);
        check(pieces.size() == 3 && plan.slice_rows == 1 && plan.slice_cols == 2, "480 x 640: the whole view and two slices");
        if (pieces.size() == 3) {
            check(pieces[0].grid_h == 28 && pieces[0].grid_w == 36 && pieces[0].pixels.size() == size_t(3) * 392 * 504,
                  "the whole view comes first, 28 x 36 patches");
            check(pieces[1].grid_h == 40 && pieces[1].grid_w == 28 && pieces[2].pixels.size() == size_t(3) * 560 * 392,
                  "then the slices, 40 x 28 patches each");
            const std::vector<uint8_t> whole = vis::resize_bicubic_u8(in.data(), h, w, 392, 504);
            const std::vector<uint8_t> refine = vis::resize_bicubic_u8(in.data(), h, w, 560, 784);
            bool ok0 = true, ok1 = true, ok2 = true;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < 392; y += 7)
                    for (int x = 0; x < 504; x += 5)
                        ok0 = ok0 && pieces[0].pixels[(size_t(c) * 392 + y) * 504 + x] ==
                                         (float(whole[(size_t(y) * 504 + x) * 3 + c]) - 127.5f) / 127.5f;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < 560; y += 7)
                    for (int x = 0; x < 392; x += 5) {
                        ok1 = ok1 && pieces[1].pixels[(size_t(c) * 560 + y) * 392 + x] ==
                                         (float(refine[(size_t(y) * 784 + x) * 3 + c]) - 127.5f) / 127.5f;
                        ok2 = ok2 && pieces[2].pixels[(size_t(c) * 560 + y) * 392 + x] ==
                                         (float(refine[(size_t(y) * 784 + 392 + x) * 3 + c]) - 127.5f) / 127.5f;
                    }
            check(ok0, "the whole view is the resized image as planes, (byte - 127.5) / 127.5");
            check(ok1, "slice 0 is the left half of the refine image");
            check(ok2, "slice 1 is the right half of the refine image");
            float lo = 2, hi = -2;
            for (float v : pieces[0].pixels) { lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
            check(lo >= -1.0f && hi <= 1.0f && lo < -0.9f && hi > 0.9f, "values span [-1, 1]");
        }
    }
    // a small image: scaled up to one 448 x 448 view, no slices
    {
        const std::vector<uint8_t> in = image(100, 100);
        const std::vector<vis::Piece> pieces = vis::preprocess(in.data(), 100, 100, vis::plan_image(100, 100));
        check(pieces.size() == 1 && pieces[0].grid_h == 32 && pieces[0].grid_w == 32, "100 x 100: one 32 x 32-patch view");
    }
    // End to end on a published pack: bytes -> plan -> resize -> pieces -> tower, against a reference run of the same
    // image (the reference's plan, torch's resize, the transformers tower). ONEBIT_MINICPM_E2E holds image.txt
    // ("height width") and out_<k>.f32 for each piece.
    const char* pack = std::getenv("ONEBIT_MINICPM_PACK");
    const char* e2e = std::getenv("ONEBIT_MINICPM_E2E");
    if (pack && *pack && e2e && *e2e) {
        try {
            int h = 0, w = 0;
            { std::ifstream f(std::string(e2e) + "/image.txt"); f >> h >> w; }
            const std::vector<uint8_t> in = image(h, w);
            const vis::ImagePlan plan = vis::plan_image(h, w);
            const std::vector<vis::Piece> pieces = vis::preprocess(in.data(), h, w, plan);
            const vis::Pack p(pack);
            std::printf("  %d x %d image: %zu piece(s), %d text rows\n", h, w, pieces.size(), plan.total_rows());
            for (size_t k = 0; k < pieces.size(); ++k) {
                const std::vector<float> out = vis::tower_forward(p, pieces[k].pixels.data(), pieces[k].grid_h, pieces[k].grid_w);
                std::vector<float> want;
                {
                    std::ifstream f(std::string(e2e) + "/out_" + std::to_string(k) + ".f32", std::ios::binary | std::ios::ate);
                    if (f) {
                        want.resize(size_t(f.tellg()) / 4);
                        f.seekg(0);
                        f.read(reinterpret_cast<char*>(want.data()), std::streamsize(want.size() * 4));
                    }
                }
                double mx = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                const size_t n = out.size() < want.size() ? out.size() : want.size();
                for (size_t i = 0; i < n; ++i) {
                    mx = std::fmax(mx, std::fabs(double(out[i]) - want[i]));
                    sa += out[i]; sb += want[i]; saa += double(out[i]) * out[i]; sbb += double(want[i]) * want[i]; sab += double(out[i]) * want[i];
                }
                const double corr = n ? (sab - sa * sb / double(n)) / std::sqrt((saa - sa * sa / double(n)) * (sbb - sb * sb / double(n))) : 0;
                std::printf("  piece %zu (%d x %d patches): %zu values, max |d| %.3g, corr %.9f\n", k, pieces[k].grid_h, pieces[k].grid_w,
                            out.size(), mx, corr);
                check(out.size() == want.size() && !out.empty() && corr > 0.99999 && mx < 5e-3,
                      "published pack, piece " + std::to_string(k) + ": the rows are the reference's");
            }
        } catch (const std::exception& e) {
            check(false, std::string("the published pack runs end to end (threw: ") + e.what() + ")");
        }
    } else {
        std::printf("  (ONEBIT_MINICPM_PACK / ONEBIT_MINICPM_E2E not set: the end-to-end arm did not run)\n");
    }
    std::printf("vision image: %d checks, %d failed\n", g_arms, g_fail);
    return g_fail ? 1 : 0;
}
