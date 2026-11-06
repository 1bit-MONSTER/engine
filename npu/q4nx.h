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
// The Q4NX chunk kinds, decoded. A Q4NX weight is a grid of 32-row x 256-column
// tiles, each one chunk; a tensor's chunk kind is the last dimension of its shape
// ([tiles, 5120] or [tile rows, tile cols, chunk bytes]), and tiles are stored row-major
// over the grid. Index g*32 + r means column group g (32 columns) of row r.
//
//   t2    2176 B  2-bit ternary codes, 4 per byte, row-major: byte r*64 + c/4, code c%4
//                at bits 2*(c%4), value 0 -> -1, 1 -> 0, 2 -> +1, 3 reserved; then one
//                IEEE fp16 scale per (128-K block, row) at 2048 + r*4 + 2*(c/128).
//                                                       w = code * (float)scale
//               2.125 bits/weight. NOT bf16: the source formats already store this
//               scale as fp16 (PTQ1_0/TQ1_0/TQ2_0 d, and pb16i's int16 ms * 2^e_ref),
//               and a bf16 re-encode is not exact -- measured on the 431 ternary
//               Bonsai-2 images, 87.5 % of the scales are not bf16-representable
//               (worst |rel| 3.9e-3) while 100 % are fp16-exact. Same 2 bytes, so the
//               2.125 b/w is unchanged. See docs/npu.md "The t2 chunk kind".
//
//   q4_1  5120 B  bf16 scale[g*32+r] @0, bf16 zero[g*32+r] @512, then 4-bit codes in 16
//                 regions of 256 B: region h*8+g (h = r/16), byte 8*j + (r%16)/2, even
//                 row in the low nibble.                w = code * scale + zero
//   q4_1f 6144 B  fp32 scale[g*32+r] @0, fp32 zero[g*32+r] @1024, then the same 16 code
//                 regions at +2048 (region h*8+g, byte 8*j + (r%16)/2, even row low).
//                                                       w = code * scale + zero
//               The imported-container form (engine#1185): an OGA pack's zero is -zp*scale, an
//               arbitrary f32 whose bf16 rounding was measured at 2.7-3.3e-3 relative on ~57 % of
//               groups, so `1bit npu-pack import` stores the pack's own f32 scale and zero.
//               GGUF-derived containers keep the bf16 q4_1: that is where the bf16 choice came
//               from. 20 % more bytes per tile; the device still streams bf16 (as_q4_1()).
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

// The t2 chunk: 2048 B of 2-bit codes + 128 B of fp16 scales per 32x256 tile.
constexpr int kT2Tile = 2176, kT2CodeBytes = 2048, kT2Rows = 32, kT2Cols = 256, kT2Group = 128;

enum class Chunk : uint32_t {
    Q4_1 = kTileBytes, Q4_1F = kTileBytesF32, Q4K = 4736, Q8 = 8704, T2 = kT2Tile
};

// The chunk kind of a quantized tensor; throws if its shape names none of them.
Chunk chunk_of(const Tensor& t);

// Which q4_1 parameter form a container holds. It is a container property, named by the container's
// own `layout` metadata (`__metadata__.layout` in model.q4nx, `q4nx_source.layout` in config.json):
// absent or "q4nx-1" is the bf16 q4_1 form -- every container written before engine#1185, and every
// GGUF-derived one; "q4nx-2" is the fp32 q4_1f form `1bit npu-pack import` writes, so an OGA pack's
// scale/zero survive bit for bit. The decoder keys off the chunk's shape either way, so a container
// with no layout still loads; these are for the caller that has to state which fidelity it holds.
enum class Params : uint32_t { Bf16, F32 };
const char* layout_of(Params params);             // "q4nx-1" | "q4nx-2"
// Throws on a version this decoder does not know rather than assuming the old form.
Params params_of_layout(const std::string& layout);
// The q4_1 chunk kind a parameter form is written in.
Chunk chunk_of(Params params);

// One q4_1 tile's per-group scale and zero as stored, index g*32+r: the exact f32 values for Q4_1F,
// the widened bf16 ones for Q4_1. Throws for a chunk kind that has no per-group scale/zero. Exposed
// so a verifier can compare a container's parameters with its source without a float round trip
// through the weights (tests/npu_pack_import_test.cpp is the one that does).
void q4_1_params(Chunk kind, const uint8_t* tile, float* scale, float* zero);

// One tile to f32: out[r * ld + c] for r < 32, c < 256.
void dequant_tile(Chunk kind, const uint8_t* tile, float* out, int ld);

// t2 only: the ternary code and the fp16 scale of element (r, c) of a tile, exposed because the
// converters and the layout test need the mapping without a float round trip. code_of() returns
// -1, 0 or +1, and kReserved (2) for the reserved bit pattern, which no converter may emit.
// kReserved is a SENTINEL, not a value: dequant_tile() decodes it as 0, never as 2 * scale.
constexpr int kReserved = 2;
int t2_code(const uint8_t* tile, int r, int c);
float t2_scale(const uint8_t* tile, int r, int c);

// t2 only: how many of the tile's 8192 codes carry the reserved pattern (3). Zero for a
// converter's output; the loader uses it to refuse a corrupt t2 tensor instead of decoding
// the reserved value as a weight.
size_t t2_reserved(const uint8_t* tile);

// A whole rows x cols tensor to f32 (rows % 32 == 0, cols % 256 == 0).
void dequant(const Tensor& t, int rows, int cols, float* out);

// One Q4_K tile as a q4_1 tile, so the lane and dx run it unchanged. Exact apart from
// rounding the products S[r]*scale and M[r]*min to bf16 (nearest even).
void q4k_to_q4_1(const uint8_t* q4k, uint8_t* q41);

// The tensor's tiles as q4_1: a copy for q4_1, a bf16-parameter repack for Q4_1F (so a device
// path that only reads 5120-byte tiles can take an imported container), converted for Q4_K. Throws
// for Q8, whose 8-bit codes do not fit q4_1, and for t2, whose per-128 fp16 scale does not fit
// q4_1's per-32 bf16 one (a t2 tensor is expanded on the target, not repacked through q4_1).
std::vector<uint8_t> as_q4_1(const Tensor& t);

}  // namespace onebit::npu::q4nx
