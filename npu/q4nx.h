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
// The three Q4NX chunk kinds, decoded. A Q4NX weight is a grid of 32-row x 256-column
// tiles, each one chunk; a tensor's chunk kind is the last dimension of its shape
// ([tiles, 5120] or [tile rows, tile cols, chunk bytes]), and tiles are stored row-major
// over the grid. Index g*32 + r means column group g (32 columns) of row r.
//
//   q4_1  5120 B  bf16 scale[g*32+r] @0, bf16 zero[g*32+r] @512, then 4-bit codes in 16
//                 regions of 256 B: region h*8+g (h = r/16), byte 8*j + (r%16)/2, even
//                 row in the low nibble.                w = code * scale + zero
//   Q4_K  4736 B  u8 scale[g*32+r] @0, u8 min[g*32+r] @256, 4-bit codes @512 at byte
//                 k*16 + r/2 (even row low), bf16 S[r] @4608, bf16 M[r] @4672.
//                                                       w = S[r]*scale*code + M[r]*min
//   Q8    8704 B  bf16 d[g*32+r] @0, int8 codes @512 at byte k*32 + r.   w = d * code
//
// Q8 comes in TWO code layouts under the same 8704-byte chunk, and nothing in the chunk or
// the container (both flm_version 1.0.3, no metadata) says which. Measured against GGUF
// ground truth on tile (0, 0) of lm_head.weight (docs/npu.md, "Q4NX chunks"):
//   Qwen3.5-4B-NPU2 (lm_head tied to the embedding): byte k*32 + r   (above; corr 0.9998)
//   Qwen3.6-35B-A3B-NPU2 (untied lm_head, q8 projections): byte (r/16)*4096 + g*512 +
//     (k%32)*16 + r%16 with g = k/32                         (corr 0.9998; k*32 + r: 0.011)
// dequant_tile implements the first. The 35B's q8 tensors are read by the private 35B route,
// which implements the second. Do not run this decoder on a Q8 tensor of an unverified model.
//
// Checked bit for bit against 1bit-MONSTER's verified decoders (npu-infer/tools/
// q4nx_dequant.py) on real Qwen3.5-4B and Qwen3-0.6B tiles: tests/npu_q4nx_test.cpp.
#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace onebit::npu::q4nx {

constexpr int kRows = 32, kCols = 256, kGroup = 32;

enum class Chunk : uint32_t { Q4_1 = 5120, Q4K = 4736, Q8 = 8704 };

// The chunk kind of a quantized tensor; throws if its shape names none of the three.
Chunk chunk_of(const Tensor& t);

// One tile to f32: out[r * ld + c] for r < 32, c < 256.
void dequant_tile(Chunk kind, const uint8_t* tile, float* out, int ld);

// A whole rows x cols tensor to f32 (rows % 32 == 0, cols % 256 == 0).
void dequant(const Tensor& t, int rows, int cols, float* out);

// One Q4_K tile as a q4_1 tile, so the lane and dx run it unchanged. Exact apart from
// rounding the products S[r]*scale and M[r]*min to bf16 (nearest even).
void q4k_to_q4_1(const uint8_t* q4k, uint8_t* q41);

// The tensor's tiles as q4_1: a copy for q4_1, converted for Q4_K. Throws for Q8, whose
// 8-bit codes do not fit q4_1.
std::vector<uint8_t> as_q4_1(const Tensor& t);

}  // namespace onebit::npu::q4nx
