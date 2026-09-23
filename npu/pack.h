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

// Buffer contents the fast lane's kernels read (docs/npu.md). Pure host code:
// every layout here is checked byte for byte against the buffers the working
// lane fed the device (tests/npu_pack_test.cpp).
//
// Per layer the layer kernel takes five buffers:
//   act     the hidden state, bf16, shared by every layer
//   weights q, k, v, o, then gate/up interleaved in H/16-tile chunks, then down;
//           each projection's tiles reordered within groups of G = K/128
//   i5      input_layernorm then post_attention_layernorm, bf16
//   i6      [cos x64][sin x64][q_norm x128][k_norm x128], bf16; the RoPE half
//           is rewritten for every position
//   kv      this layer's KV cache
// The lm head takes logits, its reordered weights (G = H/128), act and the
// final norm.
#pragma once

#include "model.h"

#include <cstdint>

namespace onebit::npu {

uint16_t f32_to_bf16(float v);  // round to nearest even
float bf16_to_f32(uint16_t v);

// Reorder n tiles in groups of G: within a group, even positions take the first
// half and odd positions the second.
void reorder_tiles(uint8_t* dst, const uint8_t* src, int n_tiles, int G);

// Bytes of the largest layer's weight buffer.
size_t layer_weight_bytes(const Model& m);
// Pack layer L's weights; dst holds layer_weight_bytes(m). Returns tiles written.
int pack_layer_weights(const Model& m, int L, uint8_t* dst);

size_t lmhead_weight_bytes(const Model& m);
void pack_lmhead_weights(const Model& m, uint8_t* dst);

constexpr size_t kNormBytes = 768;  // the i6 bytes the kernel reads
void fill_i5(const Model& m, int L, uint8_t* dst);
void fill_i6(const Model& m, int L, uint8_t* dst);
// The RoPE half of i6 (256 bytes) for one position.
void fill_rope(uint16_t* dst, int pos, float theta);

}  // namespace onebit::npu
