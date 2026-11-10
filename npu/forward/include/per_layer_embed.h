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
// The per-layer embedding a multi-component decoder consumes as `per_layer_inputs` (engine#1725).
//
// gemma-4 1.9 packs ship it beside the token table: `embed_tokens_per_layer.weight` (V x layers*head_dim) and
// `per_layer_projection_norm.weight` (layers*head_dim), both f32 in the pack and BF16 in our container. A decoder run
// gathers ONE ROW for the token, then projects it through the 256-wide (per layer) norm -- that value is the layer's
// per-layer input. Pure, header-only, no container and no device, so the projection can be pinned by a test with three
// numbers in it.
#pragma once

#include <cstdint>
#include <cstddef>

namespace onebit::npu {

// bf16 bits -> float, round-to-nearest-even on the widened mantissa (the bits are already bf16, so this is exact).
inline float pl_bf16_to_f32(uint16_t v) {
    const uint32_t bits = uint32_t(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// out[0..width) = table[id][k] * norm[k] for k in [0, width). Sub-normal and zero norms are passed through as a plain
// multiply (a zero norm zeroes that entry, which is what the arithmetic says; no special case, no clamp).
inline void per_layer_gather(const uint16_t* table, const uint16_t* norm, int64_t V, int64_t width, int64_t id,
                             float* out) {
    if (table == nullptr || out == nullptr || V <= 0 || width <= 0 || id < 0 || id >= V) return;
    const uint16_t* row = table + size_t(id) * size_t(width);
    for (int64_t k = 0; k < width; ++k) {
        const float w = pl_bf16_to_f32(row[size_t(k)]);
        out[k] = norm != nullptr ? w * pl_bf16_to_f32(norm[size_t(k)]) : w;
    }
}

}  // namespace onebit::npu
