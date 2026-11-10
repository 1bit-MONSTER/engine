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
// A MiniCPM-V 4.7 image from RGB bytes to what the vision tower takes (engine#1788): resize, slice, normalise.
//
// plan_image() (vision_plan.h) says which sizes; this does the pixels. The pieces come out in the order the reference
// processor feeds the tower: the whole view first, then the slices row by row. Each is (3, height, width) float32,
// value (byte - 127.5) / 127.5, i.e. rescale by 1/255 then mean 0.5 and std 0.5 (preprocessor_config.json).
//
// The resize is bicubic with antialiasing on BYTES, and it is torch's, bit for bit: torch.nn.functional.interpolate(
// uint8, mode="bicubic", antialias=True) at torch 2.13. That is integer arithmetic, so it can be exact, and three things
// about it are load bearing (tests/npu_vision_image_test.cpp has the reference's bytes):
//   - the kernel is the a = -0.5 cubic, stretched by the scale when shrinking, normalised per output pixel;
//   - weights become integers at the largest precision whose biggest weight still fits int16 (NOT PIL's fixed 22 bits:
//     that differs from torch by one in about 1 % of bytes);
//   - width first, then height, with the intermediate image rounded to bytes in between. A pass whose size does not
//     change is skipped.
#pragma once

#include "vision_plan.h"

#include <cstdint>
#include <vector>

namespace onebit::npu::vision {

// `rgb` is (height, width, 3) interleaved bytes; the result is (out_height, out_width, 3). Throws on empty sizes.
std::vector<uint8_t> resize_bicubic_u8(const uint8_t* rgb, int height, int width, int out_height, int out_width);

struct Piece {
    int grid_h = 0, grid_w = 0;  // patches: what tower_forward() takes as gh, gw
    std::vector<float> pixels;   // (3, grid_h * patch, grid_w * patch)
};

// Every piece of `plan` (which must be plan_image(height, width, ...) for this image), source first.
std::vector<Piece> preprocess(const uint8_t* rgb, int height, int width, const ImagePlan& plan);

}  // namespace onebit::npu::vision
