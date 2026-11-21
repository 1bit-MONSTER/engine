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
// loom_kv_layout.h -- the state layout the Loom hybrid handover must write (engine#3157, L4).
//
// The Loom GPU switch makes Loom the decode side of the hybrid route, so the NPU prefill stops filling
// llama.cpp's KV cache and starts filling Loom's. Loom takes both state roots FROM THE CALLER, which is what
// makes a zero-copy handover possible at all:
//
//   model_decode.loom:14   @qwen38_gdn_layer_decode(%layer, %gdn_ordinal)        launch(%parameters, %residual, %gdn_state)
//   model_decode.loom:16   @qwen38_attention_layer_decode(%layer, %attn_ordinal) launch(%parameters, %residual, %control,
//                                                                                        %attention_cache, %attention_output)
//   epoch.loom:16          prefill body                                          launch(%parameters, %residual, %control,
//                                                                                        %gdn_state, %attention_cache, …)
//
// measured from hrx-loom-library ba7abef (MIT, (c) 2026 AMD ROCm Software), model Qwen/Qwen3.8-27B -- a hybrid like
// ours: 64 layers, 15 full-attention ordinals, 47 gated-delta-net ordinals. The arithmetic below is that file's, not an
// interpretation: attention_layer_decode.loom:73-83 and gdn_layer_decode.loom:69-76.
//
// This header is arithmetic only: no device, no runtime, no Loom. It exists so the NPU writer and the tests agree on
// one set of offsets before any of it touches memory, the same way app/mem_admit.h carries the admission arithmetic.
#pragma once

#include <cstdint>

namespace onebit::loom {

// Offset alignment Loom's `buffer.pack [align(256) …]` imposes on every state block.
inline constexpr uint64_t kStateAlign = 256;

// Attention cache: f16 key and value rows, token-major. `kv_elements` is the per-token row width of ONE of K or V
// (Loom's `%key_value_size`), so a token of key storage is kv_elements * 2 bytes and a token of the pair is twice that.
inline constexpr uint64_t kF16Bytes = 2;

// GDN state, per ordinal, in bytes: Loom's `%recurrent_state_bytes` (786432 f32) and `%conv_history_bytes`
// (30720 f32). Both are already 256-aligned, which is why the packed block is their plain sum -- asserted in the test,
// so a future change to either constant cannot silently break the alignment the pack assumes.
inline constexpr uint64_t kGdnRecurrentBytes = 3145728;
inline constexpr uint64_t kGdnConvBytes      = 122880;

inline constexpr uint64_t align_up(uint64_t value, uint64_t align) {
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

// ---- attention cache ----------------------------------------------------------------------------
// cache_capacity is a Loom SPECIALIZATION (attention_layer_decode.loom:15, range 1..262144), fixed when the command
// buffer is materialized -- not an issue-time value. Two capacities are two Loom programs.

// Bytes of one key (or one value) cache at this capacity.
inline constexpr uint64_t attention_side_bytes(uint64_t capacity, uint64_t kv_elements) {
    return align_up(capacity * kv_elements * kF16Bytes, kStateAlign);
}

// Where the value array starts inside a layer block: Loom packs [align(256) cache_bytes, align(256) cache_bytes], so
// the key array is at 0 and the value array one aligned side later.
inline constexpr uint64_t attention_value_offset(uint64_t capacity, uint64_t kv_elements) {
    return attention_side_bytes(capacity, kv_elements);
}

// One attention ordinal's block: key array then value array.
inline constexpr uint64_t attention_layer_bytes(uint64_t capacity, uint64_t kv_elements) {
    return attention_side_bytes(capacity, kv_elements) * 2;
}

// First byte of that ordinal's block inside the shared root.
inline constexpr uint64_t attention_layer_offset(uint64_t ordinal, uint64_t capacity, uint64_t kv_elements) {
    return ordinal * attention_layer_bytes(capacity, kv_elements);
}

inline constexpr uint64_t attention_total_bytes(uint64_t capacity, uint64_t kv_elements, uint64_t ordinals) {
    return ordinals * attention_layer_bytes(capacity, kv_elements);
}

// ---- gated delta net state ----------------------------------------------------------------------
inline constexpr uint64_t gdn_conv_offset() { return align_up(kGdnRecurrentBytes, kStateAlign); }

inline constexpr uint64_t gdn_layer_bytes() { return gdn_conv_offset() + align_up(kGdnConvBytes, kStateAlign); }

inline constexpr uint64_t gdn_layer_offset(uint64_t ordinal) { return ordinal * gdn_layer_bytes(); }

inline constexpr uint64_t gdn_total_bytes(uint64_t ordinals) { return ordinals * gdn_layer_bytes(); }

// ---- both roots ---------------------------------------------------------------------------------
// What the exporter has to allocate and export for one sequence: the attention cache (f16, capacity-dependent) plus the
// GDN state (f32, capacity-independent -- a recurrence's state does not grow with context).
inline constexpr uint64_t state_total_bytes(uint64_t capacity, uint64_t kv_elements, uint64_t attention_ordinals,
                                            uint64_t gdn_ordinals) {
    return attention_total_bytes(capacity, kv_elements, attention_ordinals) + gdn_total_bytes(gdn_ordinals);
}

// Token-major element index inside one side of one attention ordinal: element = position * kv_elements + j.
inline constexpr uint64_t attention_element_index(uint64_t position, uint64_t j, uint64_t kv_elements) {
    return position * kv_elements + j;
}

}  // namespace onebit::loom
