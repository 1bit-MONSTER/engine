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
// The t2 chunk contract (npu/q4nx.h, docs/npu.md "The t2 chunk kind"): the byte layout, the
// 2.125 bits per weight, the reserved code, and the decoder against a hand-computed tile.
// Compiled at test time by tests/npu_t2_layout_test.sh, so it needs no build target of its own
// and runs in CI without a model, a device or numpy.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "q4nx.h"

using namespace onebit::npu;
using namespace onebit::npu::q4nx;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        failures++;
    }
}

// fp16 0x3C00 is 1.0 and 0xBC00 is -1.0, 0x0000 is +0.
static const uint16_t kHalf1 = 0x3C00, kHalfMinus1 = 0xBC00;

int main() {
    // 1. the arithmetic the issue and the spec both quote
    check(kT2Tile == 2176, "tile is 2176 bytes");
    check(kT2CodeBytes == 2048, "codes are 2048 bytes");
    check(kT2Tile - kT2CodeBytes == 128, "scales are 128 bytes");
    check(kT2Tile * 8 == 17408, "2176 bytes = 17408 bits = 2.125 * 8192");
    check((size_t)Chunk::T2 == (size_t)kT2Tile, "Chunk::T2 is the tile size, like every other kind");

    // 2. the layout: every element has its own 2-bit slot, and the 8192 slots are exactly the
    //    2048 code bytes (a bijection -- a scale that overlaps a code byte would still "work")
    {
        std::vector<int> seen(8192, 0);
        for (int r = 0; r < kT2Rows; r++)
            for (int c = 0; c < kT2Cols; c++) {
                const size_t byte = size_t(r) * 64 + size_t(c) / 4;
                const int slot = 2 * (c % 4);
                check(byte < size_t(kT2CodeBytes), "code byte inside the code region");
                const int elem = int(byte) * 4 + slot / 2;
                check(elem >= 0 && elem < 8192, "code element index in range");
                seen[size_t(elem)]++;
            }
        bool once = true;
        for (int v : seen) once = once && v == 1;
        check(once, "each of the 8192 slots is claimed exactly once");
    }

    // 3. a hand-computed tile: row 3 code +1 in column 200, row 3 code -1 in column 5, row 0 code
    //    +1 in column 0; scales 1.0 for block 0 and -1.0 for block 1 of those rows.
    {
        std::vector<uint8_t> tile(kT2Tile, 0);
        auto put_code = [&](int r, int c, int value) {
            const size_t byte = size_t(r) * 64 + size_t(c) / 4;
            const int shift = 2 * (c % 4);
            const int code = value + 1;  // -1 -> 0, 0 -> 1, +1 -> 2
            tile[byte] |= uint8_t((code & 3) << shift);
        };
        auto put_scale = [&](int r, int block, uint16_t bits) {
            const size_t at = size_t(kT2CodeBytes) + size_t(r) * 4 + 2 * size_t(block);
            tile[at] = uint8_t(bits & 0xFF);
            tile[at + 1] = uint8_t(bits >> 8);
        };
        for (int r = 0; r < kT2Rows; r++) {
            put_scale(r, 0, kHalf1);
            put_scale(r, 1, kHalf1);
        }
        put_code(3, 200, +1);
        put_code(3, 5, -1);
        put_code(0, 0, +1);
        put_code(7, 7, 0);   // bits 01 are the zero code; untouched bits are 00, i.e. -1
        put_scale(3, 0, kHalf1);
        put_scale(3, 1, kHalfMinus1);

        check(t2_code(tile.data(), 3, 200) == 1, "code at (3,200) is +1");
        check(t2_code(tile.data(), 3, 5) == -1, "code at (3,5) is -1");
        check(t2_code(tile.data(), 0, 0) == 1, "code at (0,0) is +1");
        // 0 = -1 in the value table, so a zero-filled tile is a tile of -1 weights, not a tile of
        // zeros. That is Q2_0's c-1 encoding and it is exactly why the reserved pattern and the
        // converter's refusal exist: "empty" and "zero" must not be the same bytes.
        check(t2_code(tile.data(), 7, 7) == 0, "bits 01 are the zero code");
        check(t2_code(tile.data(), 31, 255) == -1, "an untouched slot (bits 00) is code -1");
        check(t2_scale(tile.data(), 3, 200) == -1.0f, "column 200 reads block 1's scale");
        check(t2_scale(tile.data(), 3, 100) == 1.0f, "column 100 reads block 0's scale");

        std::vector<float> out(32 * 256, 12345.0f);
        dequant_tile(Chunk::T2, tile.data(), out.data(), 256);
        check(out[3 * 256 + 200] == -1.0f, "dequant(3,200) = +1 * -1.0");
        check(out[3 * 256 + 5] == -1.0f, "dequant(3,5) = -1 * +1.0");
        check(out[0 * 256 + 0] == 1.0f, "dequant(0,0) = +1 * +1.0");
        check(out[7 * 256 + 7] == 0.0f, "dequant of a zero code is 0");
        check(t2_reserved(tile.data()) == 0, "no reserved code in this tile");
    }

    // 4. the reserved pattern (3) is counted, and decodes as 0 rather than as a phantom weight.
    //    The scale is deliberately NON-zero here: with a zero scale every decoding looks like 0,
    //    which is how a decoder that multiplied the raw sentinel (+2) passed this check before
    //    pi-prs caught it. The tile below therefore carries 1.0 in every scale slot.
    {
        std::vector<uint8_t> tile(kT2Tile, 0);
        for (int r = 0; r < kT2Rows; r++)
            for (int b = 0; b < 2; b++) {
                const size_t at = size_t(kT2CodeBytes) + size_t(r) * 4 + 2 * size_t(b);
                tile[at] = uint8_t(kHalf1 & 0xFF);
                tile[at + 1] = uint8_t(kHalf1 >> 8);
            }
        tile[0] |= 0x03;  // row 0, column 0 -> bits 0..1 = 3
        check(t2_reserved(tile.data()) == 1, "one reserved code is counted");
        check(t2_code(tile.data(), 0, 0) == kReserved, "code 3 reports kReserved");
        std::vector<float> out(32 * 256, 12345.0f);
        dequant_tile(Chunk::T2, tile.data(), out.data(), 256);
        check(out[0] == 0.0f, "a reserved code decodes as 0, not as 2 * scale");
        check(out[1] == -1.0f, "the neighbouring untouched slot is code -1, unchanged");
    }

    // 5. subnormal fp16 scales are not silently 0: 0x0001 is 5.96e-8, and a 27B ternary model can
    //    legitimately carry a tiny per-column scale.
    {
        std::vector<uint8_t> tile(kT2Tile, 0);
        tile[kT2CodeBytes] = 0x01;  // row 0, block 0 = smallest subnormal
        tile[kT2CodeBytes + 1] = 0x00;
        tile[0] |= 0x02;            // row 0, column 0 = +1
        const float s = t2_scale(tile.data(), 0, 0);
        check(s > 0.0f && s < 1e-7f, "a subnormal fp16 scale decodes to its own value, not 0");
        check(t2_scale(tile.data(), 0, 0) == 5.96046448e-08f, "smallest subnormal decoded exactly");
    }

    // 6. chunk_of() recognises a t2 tensor by its last dimension, and as_q4_1 refuses it
    {
        Tensor t;
        t.shape = {2, 3, int64_t(kT2Tile)};
        check(chunk_of(t) == Chunk::T2, "chunk_of([.., 2176]) is T2");
        t.shape = {1, 1, 2175};
        bool threw = false;
        try {
            chunk_of(t);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "a 2175-byte last dimension is not a chunk kind");
    }

    std::printf(failures ? "FAIL t2 chunk contract (%d)\n" : "PASS t2 chunk contract (2176 B, 2.125 b/w)\n",
                failures);
    return failures ? 1 : 0;
}
