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

// engine#1786: q4nx_bf16_matrix_into (npu/forward/include/bf16_matrix.h) -- an unquantised BF16 matrix in either of the
// two orientations FastFlowLM's packs use, into the forward's [rows][cols] grid. Each value is its own (row, col)
// written as a bf16 word, so a wrong orientation or a wrong pad shows as a wrong number at a named place.
#include "bf16_matrix.h"

#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
// bf16 holds small integers exactly: encode value v = 100*r + c (r < 64, c < 100).
static uint16_t enc(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

int main() {
    // [out, in] = [3, 5] into a [3, 8] grid: copied, columns 5..7 zero.
    {
        std::vector<uint16_t> w(3 * 5);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 5; c++) w[r * 5 + c] = enc(100.0f * r + c);
        std::vector<float> o(3 * 8, -1.0f);
        check(q4nx_bf16_matrix_into(w.data(), w.size() * 2, 3, 5, 3, 8, o.data()) == 0, "[out, in] into a padded grid is read");
        bool ok = true;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 8; c++) ok &= o[r * 8 + c] == (c < 5 ? 100.0f * r + c : 0.0f);
        check(ok, "[out, in]: each value at its own place, the pad columns zero");
    }
    // [in, out] = [5, 3] asked as a [3, 5] grid: read transposed (Qwen3.6-35B-A3B's ssm_alpha [2048, 32]).
    {
        std::vector<uint16_t> w(5 * 3);
        for (int i = 0; i < 5; i++) for (int o_ = 0; o_ < 3; o_++) w[i * 3 + o_] = enc(100.0f * o_ + i);   // value of (row o_, col i)
        std::vector<float> o(3 * 5, -1.0f);
        check(q4nx_bf16_matrix_into(w.data(), w.size() * 2, 5, 3, 3, 5, o.data()) == 0, "[in, out] is read");
        bool ok = true;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 5; c++) ok &= o[r * 5 + c] == 100.0f * r + c;
        check(ok, "[in, out]: read transposed, each value at its own place");
    }
    // Square: the shape cannot say, so it is [out, in].
    {
        std::vector<uint16_t> w(4 * 4);
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) w[r * 4 + c] = enc(100.0f * r + c);
        std::vector<float> o(16);
        check(q4nx_bf16_matrix_into(w.data(), w.size() * 2, 4, 4, 4, 4, o.data()) == 0 && o[1] == 1.0f && o[4] == 100.0f,
              "a square matrix is read as [out, in]");
    }
    // Refusals: wider than the grid, and short of bytes.
    {
        std::vector<uint16_t> w(6 * 6, 0);
        std::vector<float> o(16);
        check(q4nx_bf16_matrix_into(w.data(), w.size() * 2, 6, 6, 4, 4, o.data()) == -1, "a matrix wider than the grid is refused");
        check(q4nx_bf16_matrix_into(w.data(), 10, 3, 5, 3, 8, o.data()) == -1, "a matrix short of bytes is refused");
    }
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("ok: both orientations, the padded grid and the refusals\n");
    return 0;
}
