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

// engine#1786: an unquantised BF16 matrix read into the forward's [rows][cols] grid, header-only so its arm
// (tests/bf16_matrix_test.cpp) drives it without a container.
//
// Two orientations reach the forward. FastFlowLM's Embedding-Gemma-300M-NPU2 ships every projection as HF's [out, in]
// (= [rows, cols]), possibly narrower than a grid padded to 256 columns (IM 1152 -> 1280): copied, the pad zero.
// Its Qwen3.6-35B-A3B-NPU2 stores a small matrix as [in, out] (ssm_alpha/beta_proj [2048, 32], like its [H][E]
// router): against the GGUF of the same checkpoint, read transposed, corr 0.99999. A declared shape that is exactly
// the transpose of the grid asked for is read that way; a square one cannot be told from its shape and is read as
// [out, in]. Anything else -- wider than the grid, short of bytes -- is refused (-1), never guessed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

inline float q4nx_bf16_word_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

inline int q4nx_bf16_matrix_into(const uint16_t* w, size_t nbytes, int64_t r0, int64_t c0, int rows, int cols,
                                 float* out) {
    if (!w || !out || rows <= 0 || cols <= 0 || r0 <= 0 || c0 <= 0) return -1;
    if (nbytes < (size_t)r0 * (size_t)c0 * 2) return -1;
    if (r0 == cols && c0 == rows && rows != cols) {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) out[(size_t)r * cols + c] = q4nx_bf16_word_to_f32(w[(size_t)c * (size_t)c0 + r]);
        return 0;
    }
    if (r0 > rows || c0 > cols) return -1;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[(size_t)r * cols + c] = (r < r0 && c < c0) ? q4nx_bf16_word_to_f32(w[(size_t)r * (size_t)c0 + c]) : 0.0f;
    return 0;
}
