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
// npu_padded_grid_test: a container may store a weight on a WIDER tile grid than the model's own
// dimensions (engine#1097: the FastFlowLM repack pads intermediate_size to 512 columns, 8960 ->
// 9216, so the weight holds 1728 tiles where the model's dims imply 1680). q4nx_dequant_tensor_packed
// reads the container's grid and writes only the model's [rows, cols], so the caller's buffer, its
// row stride and the kernel's B layout never change. This pins that contract with synthetic Q8 tiles
// -- no model, no device, no numpy -- and pins the refusals, because a wrong grid must fail rather
// than decode garbage.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "q4nx_dequant.h"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        failures++;
    }
}

// A Q8 tile whose value(r, k) is exactly `code`: d[256] bf16 = 1.0, q[8192] i8 = code at byte
// k*32 + r (q4nx_dequant.h, "Q8 chunk").
static void make_q8_tile(uint8_t* t, int8_t code) {
    std::memset(t, 0, Q4NX_CHUNK_Q8);
    for (int g = 0; g < 8; g++)
        for (int r = 0; r < Q4NX_TILE_ROWS; r++) {
            const uint16_t one_bf16 = 0x3F80;              // bf16 1.0
            std::memcpy(t + 2 * (g * Q4NX_TILE_ROWS + r), &one_bf16, 2);
        }
    for (int k = 0; k < Q4NX_TILE_COLS; k++)
        for (int r = 0; r < Q4NX_TILE_ROWS; r++)
            t[512 + k * Q4NX_TILE_ROWS + r] = (uint8_t)code;
}

// Tile (tr, tc) of a grid holds code = 1 + tr*10 + tc, so a wrong tile shows up as a wrong value.
static std::vector<uint8_t> make_grid(int ntr, int ntc) {
    std::vector<uint8_t> g((size_t)ntr * (size_t)ntc * Q4NX_CHUNK_Q8);
    for (int tr = 0; tr < ntr; tr++)
        for (int tc = 0; tc < ntc; tc++)
            make_q8_tile(&g[((size_t)tr * (size_t)ntc + (size_t)tc) * Q4NX_CHUNK_Q8],
                         (int8_t)(1 + tr * 10 + tc));
    return g;
}

// engine#1786: the two code orders of the 8704-byte Q8 chunk (q4nx_dequant.h, q4nx_dequant_tile_q8_split16). A symmetric
// Q8 group holds one code of magnitude 127 (its absmax), so each (row, 32-column group) here gets one +-127 and 31 codes
// below it, with a bf16 scale of its own. The same values are written in each order: each must be detected as its own
// order and decode to exactly the values written (bf16 scale x int8 code is exact in float).
static uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float f_of_bf16(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void test_q8_split16() {
    const int ntr = 2, ntc = 2, rows = ntr * Q4NX_TILE_ROWS, cols = ntc * Q4NX_TILE_COLS;
    std::vector<float> want((size_t)rows * cols);
    std::vector<uint8_t> old_order((size_t)ntr * ntc * Q4NX_CHUNK_Q8, 0), split((size_t)ntr * ntc * Q4NX_CHUNK_Q8, 0);
    uint32_t seed = 1786;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (int tr = 0; tr < ntr; tr++)
        for (int tc = 0; tc < ntc; tc++) {
            uint8_t* to = &old_order[((size_t)tr * ntc + tc) * Q4NX_CHUNK_Q8];
            uint8_t* ts = &split[((size_t)tr * ntc + tc) * Q4NX_CHUNK_Q8];
            for (int r = 0; r < Q4NX_TILE_ROWS; r++)
                for (int g = 0; g < 8; g++) {
                    const uint16_t d = bf16_of(0.001f * (float)(1 + rnd() % 50));
                    std::memcpy(to + 2 * (g * Q4NX_TILE_ROWS + r), &d, 2);
                    std::memcpy(ts + 2 * (g * Q4NX_TILE_ROWS + r), &d, 2);
                    const int peak = (int)(rnd() % 32);
                    for (int j = 0; j < 32; j++) {
                        const int k = g * 32 + j;
                        int code = j == peak ? ((rnd() & 1) ? 127 : -127) : (int)(rnd() % 253) - 126;
                        to[512 + k * 32 + r] = (uint8_t)(int8_t)code;
                        ts[512 + (r / 16) * 4096 + k * 16 + (r % 16)] = (uint8_t)(int8_t)code;
                        want[(size_t)(tr * 32 + r) * cols + (size_t)tc * 256 + k] = f_of_bf16(d) * (float)code;
                    }
                }
        }
    check(q4nx_q8_split16(split.data(), (size_t)ntr * ntc, Q4NX_CHUNK_Q8) == 1, "q8: split16 codes are detected as split16");
    check(q4nx_q8_split16(old_order.data(), (size_t)ntr * ntc, Q4NX_CHUNK_Q8) == 0,
          "q8: 32-row codes are detected as the 32-row order");
    std::vector<float> got((size_t)rows * cols);
    check(q4nx_dequant_tensor_chunked(split.data(), split.size(), rows, cols, Q4NX_CHUNK_Q8, got.data()) == 0 && got == want,
          "q8: a split16 tensor decodes to the values written");
    check(q4nx_dequant_tensor_chunked(old_order.data(), old_order.size(), rows, cols, Q4NX_CHUNK_Q8, got.data()) == 0 &&
              got == want,
          "q8: a 32-row tensor still decodes to the values written");
    // The per-token row path reads the same order from its tile-row's chunks.
    std::vector<float> row(cols);
    bool rows_ok = true;
    for (int r : {0, 15, 16, 31, 32, 63}) {
        rows_ok &= q4nx_dequant_tile_row(split.data(), split.size(), rows, cols, Q4NX_CHUNK_Q8, r, row.data()) == 0 &&
                   std::memcmp(row.data(), &want[(size_t)r * cols], (size_t)cols * sizeof(float)) == 0;
    }
    check(rows_ok, "q8: the row path decodes split16 rows on both 16-row halves");
    // A tile that reaches 127 in both orders (constant codes) is not evidence for split16: the 32-row reading stays.
    std::vector<uint8_t> flat(Q4NX_CHUNK_Q8);
    make_q8_tile(flat.data(), 127);
    check(q4nx_q8_split16(flat.data(), 1, Q4NX_CHUNK_Q8) == 0, "q8: a tile that fits both orders keeps the 32-row order");
}

int main() {
    const int rows = 2 * Q4NX_TILE_ROWS;    // 64 rows  -> 2 tile-rows
    const int cols = 2 * Q4NX_TILE_COLS;    // 512 cols -> 2 tile-cols
    const int rp = 3 * Q4NX_TILE_ROWS;      // container padded to 3x3 tiles
    const int cp = 3 * Q4NX_TILE_COLS;

    // Reference: the model's own 2x2 grid decoded with the shipped chunked decoder.
    const std::vector<uint8_t> tight = make_grid(2, 2);
    std::vector<float> want((size_t)rows * (size_t)cols, -1.0f);
    check(q4nx_dequant_tensor_chunked(tight.data(), tight.size(), rows, cols,
                                      Q4NX_CHUNK_Q8, want.data()) == 0,
          "the tight grid decodes with the shipped chunked decoder");

    // The padded container is a 3x3 grid whose first 2x2 tiles are the same weights.
    const std::vector<uint8_t> padded = make_grid(3, 3);
    std::vector<float> got((size_t)rows * (size_t)cols, -1.0f);
    check(q4nx_dequant_tensor_packed(padded.data(), padded.size(), rows, cols, rp, cp,
                                     Q4NX_CHUNK_Q8, got.data()) == 0,
          "a padded grid decodes into the model's [rows, cols]");
    check(got == want, "a padded grid decodes to exactly the same values as the tight grid");

    // The tile stride is the container's, not the model's: tile (1, 1) of the tight 2x2 grid is code
    // 12, and in the padded 3x3 grid it sits after three tiles in its row. Read with the model's
    // stride it would come out as code 11 (tile (1,1) of the 3-wide layout) -- pin that difference,
    // because that is the failure a tile-count-only fix would ship silently.
    check(got[(size_t)(Q4NX_TILE_ROWS) * cols + Q4NX_TILE_COLS] == 12.0f,
          "tile (1,1) is read at the container's stride (code 12, not 11)");

    // Equal grids must be exactly the shipped call.
    std::vector<float> same((size_t)rows * (size_t)cols, -1.0f);
    check(q4nx_dequant_tensor_packed(tight.data(), tight.size(), rows, cols, rows, cols,
                                     Q4NX_CHUNK_Q8, same.data()) == 0 && same == want,
          "an unpadded grid is exactly the shipped decoder");

    // Refusals: a smaller grid than the model's, a short buffer, a chunk kind that is not one.
    std::vector<float> out((size_t)rows * (size_t)cols, 0.0f);
    check(q4nx_dequant_tensor_packed(padded.data(), padded.size(), rows, cols, Q4NX_TILE_ROWS, cp,
                                     Q4NX_CHUNK_Q8, out.data()) == -1,
          "a grid smaller than the model's is refused");
    check(q4nx_dequant_tensor_packed(padded.data(), (size_t)4 * Q4NX_CHUNK_Q8, rows, cols, rp, cp,
                                     Q4NX_CHUNK_Q8, out.data()) == -1,
          "a buffer shorter than the padded grid is refused");
    check(q4nx_dequant_tensor_packed(padded.data(), padded.size(), rows, cols, rp, cp, 4096,
                                     out.data()) == -1,
          "an unknown chunk size is refused");
    check(q4nx_dequant_tensor_packed(padded.data(), padded.size(), rows + 1, cols, rp, cp,
                                     Q4NX_CHUNK_Q8, out.data()) == -1,
          "a geometry off the tile grid is refused");

    // q4nx_packed_grid(): the container's grid, recovered from data_size, and every case it refuses.
    int cb = 0, rp_g = 0, cp_g = 0;
    const size_t zr1_im_tiles = (size_t)(1536 / 32) * (size_t)(8960 / 256);   // 1680: the model's own grid
    check(q4nx_packed_grid(1728 * Q4NX_TILE_BYTES, 1536, 8960, &cb, &rp_g, &cp_g) == 1 &&
          cb == Q4NX_TILE_BYTES && rp_g == 1536 && cp_g == 9216,
          "down_proj 1536x8960 in a 1728-tile container is read as 1536x9216 (columns padded)");
    check(q4nx_packed_grid(1728 * Q4NX_TILE_BYTES, 8960, 1536, &cb, &rp_g, &cp_g) == 1 &&
          cb == Q4NX_TILE_BYTES && rp_g == 9216 && cp_g == 1536,
          "gate_proj 8960x1536 in the same container is read as 9216x1536 (rows padded)");
    check(q4nx_packed_grid(zr1_im_tiles * Q4NX_TILE_BYTES, 1536, 8960, &cb, &rp_g, &cp_g) == 0,
          "a container on the model's own grid needs no padding (0, not a grid)");
    // A tile count that admits both (ntr, ntc+1) and (ntr+1, ntc) with equal excess: a transposed
    // decode waiting to happen, so it must be refused rather than resolved by loop order.
    check(q4nx_packed_grid(2 * Q4NX_TILE_BYTES, Q4NX_TILE_ROWS, Q4NX_TILE_COLS, &cb, &rp_g, &cp_g) == 0,
          "an ambiguous padded grid (two grids, same excess) is refused");
    check(q4nx_packed_grid(100 * Q4NX_TILE_BYTES, 1536, 8960, &cb, &rp_g, &cp_g) == 0,
          "a grid smaller than the model's is refused");
    check(q4nx_packed_grid(7, 1536, 8960, &cb, &rp_g, &cp_g) == 0,
          "a size no chunk kind divides is refused");

    test_q8_split16();

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok: a padded grid decodes to the model's own values at the container's stride, an "
                "unpadded grid is the shipped decoder, and a wrong grid is refused\n");
    return 0;
}
