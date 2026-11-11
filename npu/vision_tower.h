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
// The MiniCPM-V 4.7 vision tower, host float32 (engine#1788): pixels in, the rows the text model receives out.
//
// A reference, not a serving path: plain loops (spread over threads, see tower_threading), one image, no device. It exists so that whatever later runs this
// tower on the NPU has numbers to be compared with, and it is itself compared with the transformers implementation
// (transformers.models.minicpmv4_7, 5.22.2) on a published pack's weights: docs/npu.md has the figures.
//
// The order of operations, each line read from that implementation and pinned by tests/npu_vision_tower_test.cpp:
//   1. patch embedding: a Conv2d whose kernel is its stride, so one matrix product per patch, patches row-major;
//   2. + a learned position row per patch, picked NEAREST on the pack's square table: row floor(y * side / gh),
//      column floor(x * side / gw). No interpolation. (The reference buckets y/gh in float32 against k/side; the integer
//      form gives the same index for every grid height and width from 1 to 140 at side 70, measured.)
//   3. blocks 0..insert_layer: x += proj(attention(LN1(x))), x += fc2(gelu_tanh(fc1(LN2(x)))). Full attention over
//      the image, no rotary term, scale 1/sqrt(head_dim), LayerNorm eps 1e-6;
//   4. the vit_merger, once, after block insert_layer: LN1, attention INSIDE each 2x2 window only, the residual, then
//      per window  fc2(gelu_tanh(fc1(pre_norm(the four tokens side by side)))) + the mean of the four. The grid halves;
//   5. the remaining blocks on the halved grid, then one LayerNorm (the pack's `merger.norm`);
//   6. the merger: 2x2 groups again, pre_norm over the four tokens side by side, fc1, GELU (the exact erf form here,
//      not the tanh approximation the blocks use), fc2 to the text model's width. The grid halves again.
// So a gh x gw patch grid gives (gh/4) * (gw/4) rows, row-major, and both extents must be multiples of 4.
#pragma once

#include "vision_pack.h"

#include <vector>

namespace onebit::npu::vision {

// `pixels` is (channels, gh*patch, gw*patch) float32, already rescaled and normalised. Returns (gh/4 * gw/4,
// out_hidden). When `tower` is given it receives step 5's output, (gh/2 * gw/2, hidden): the tower before the merger.
// Throws std::runtime_error when the grid is not a multiple of 4 in both extents, is empty, or the pack's geometry
// lacks what only config.json states (the head count).
std::vector<float> tower_forward(const Pack& pack, const float* pixels, int gh, int gw, std::vector<float>* tower = nullptr);

// How the forward spreads its loops over threads. `threads` 0 means the default (ONEBIT_VISION_THREADS, else the
// machine's count, capped at 16) and 1 means none; `min_work` is the number of multiply-adds below which a loop stays
// on the calling thread (default 4e6, so a small tower never starts one). The output does not depend on either: every
// token and every attention head is computed by exactly one thread with the single-threaded arithmetic. Not for use
// while a tower_forward() is running.
void tower_threading(int threads, double min_work = 4e6);

// Step 2's table row for patch (y, x) of a gh x gw grid on a side x side table.
inline int nearest_position(int y, int x, int gh, int gw, int side) {
    return int((int64_t(y) * side) / gh) * side + int((int64_t(x) * side) / gw);
}

}  // namespace onebit::npu::vision
