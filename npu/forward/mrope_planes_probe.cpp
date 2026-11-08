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
// MRoPE plumbing probe (engine#1311/TQVL, plan P4's `rope.planes`): does step_batch_embeds()'s three-plane position
// input do what it claims?
//
//   (a) three EQUAL planes (t = h = w, i.e. a text prompt) must be BYTE-IDENTICAL to the token path -- the identity
//       proved numerically (0 angle difference over positions 0..262143, theta 5e6);
//   (b) three DIFFERENT planes (an image's (t, h, w) walk) must run and must move the logits, which is what an image
//       prompt needs and what plain RoPE cannot express;
//   (c) a pack that declares MRoPE without `mrope_section` is refused by name rather than guessed at (the parse is
//       in q4nx_semantics.c; this probe runs on a pack that does carry the sections).
//
//   mrope_planes_probe <model.q4nx> <kernels dir>            # device path (inside npu-hold)
//   NPU_INFER_HOST_GEMM=1 mrope_planes_probe <model> <dir>   # the same body on the host, no device
#include "q4nx_forward.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.q4nx> <kernels dir>\n", argv[0]); return 2; }
    Q4nxNpuForward f;
    std::unique_ptr<xrt::device> dev;
    bool ok;
    if (getenv("NPU_INFER_HOST_GEMM")) {
        ok = f.init_host(argv[1], argv[2]);
    } else {
        dev = std::make_unique<xrt::device>(0);
        ok = f.init(*dev, argv[1], argv[2]);
    }
    if (!ok) { std::fprintf(stderr, "init: %s\n", f.last_error()); return 1; }
    const std::vector<int> ids = {9707, 11, 1879, 374, 264, 1234, 16, 13};
    const int rows = int(ids.size());
    std::vector<float> embeds;
    for (int id : ids) { std::vector<float> row; f.embed_row(id, row); embeds.insert(embeds.end(), row.begin(), row.end()); }

    std::vector<float> plain, eq, diff;
    f.reset(); if (!f.step_batch(ids, 0, plain)) { std::fprintf(stderr, "step_batch: %s\n", f.last_error()); return 1; }

    std::vector<int> planes_eq(size_t(3 * rows)), planes_img(size_t(3 * rows));
    for (int r = 0; r < rows; r++) {
        planes_eq[r] = planes_eq[rows + r] = planes_eq[2 * rows + r] = r;      // t = h = w: text
        planes_img[r] = r;                                                      // t
        planes_img[rows + r] = r / 2;                                           // h (a 2-wide image grid)
        planes_img[2 * rows + r] = r % 2;                                       // w
    }
    f.reset(); if (!f.step_batch_embeds(embeds, planes_eq, 0, eq)) { std::fprintf(stderr, "eq: %s\n", f.last_error()); return 1; }
    f.reset(); if (!f.step_batch_embeds(embeds, planes_img, 0, diff)) { std::fprintf(stderr, "img: %s\n", f.last_error()); return 1; }

    int eq_diff = 0; double img_delta = 0.0;
    for (size_t i = 0; i < plain.size(); i++) {
        if (plain[i] != eq[i]) eq_diff++;
        img_delta += std::abs(double(plain[i]) - double(diff[i]));
    }
    std::printf("equal planes vs token path: %s (%d of %zu logits differ)\n", eq_diff ? "MISMATCH" : "byte-identical",
                eq_diff, plain.size());
    std::printf("image-style planes vs plain : mean |dlogit| %.4f (they must differ)\n", img_delta / double(plain.size()));
    const bool pass = eq_diff == 0 && img_delta > 0.0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
