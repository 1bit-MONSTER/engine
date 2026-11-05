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

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok: a padded grid decodes to the model's own values at the container's stride, an "
                "unpadded grid is the shipped decoder, and a wrong grid is refused\n");
    return 0;
}
