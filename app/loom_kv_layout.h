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
// measured from hrx-loom-library **v0.2.0** (`1fd6bd23`, MIT, (c) 2026 AMD ROCm Software), model Qwen/Qwen3.8-27B -- a
// hybrid like ours: 64 layers, 16 full-attention ordinals (0..15) and 48 gated-delta-net ordinals (0..47). The arithmetic
// below is that file's, not an interpretation: attention_layer_decode.loom:74-83 and gdn_layer_decode.loom:69-76.
//
// WHY v0.2.0 AND NOT HEAD. A shipped card records the exact toolchain it was built with, and the Qwen3.8-27B card's
// `llm_config.json`/`info.json` pair names hrx_loom 0.2.0 from PyPI, hrx_system v0.2.0, and its
// `loom/packaging/python/version.json` names `model-catalog: github:ROCm/hrx-loom-library@v0.2.0` — so the programs a
// card runs are the tag's, and the tag is what this file is measured against. HEAD is not a different branch: `v0.2.0`
// (`1fd6bd23`) is an ANCESTOR of `cb9ae9e`, 85 commits back (`git rev-list --count 1fd6bd23..cb9ae9e`), so the tag is a
// subset of HEAD and anything verified at HEAD holds at the tag. The anchor is the tag because a card names it, and the
// one thing that does NOT hold across the 85 commits is the packet's word-2 contract, described per revision below.
//
// CARDINALITY, because the declarations read one short if you take a range as exclusive. model_decode.loom unrolls 16
// groups of 3 GDN layers + 1 attention layer (`:26-27`), launches gdn ordinals `3*group + {0,1,2}` (`:48-50`) and the
// attention ordinal `%group` (`:54`), so group 15 uses gdn ordinal 47 and attention ordinal 15. Loom's `range(lo, hi)` is
// INCLUSIVE -- docs/src/guide/facts-and-specialization.md:88, "`range(%value, lo, hi)` is inclusive" -- so the decls'
// `range(..., 0, 47)` and `range(..., 0, 15)` admit 48 and 16 ordinals, and a root sized for 47 or 15 would let the decode
// body run off the end of both. 16 + 48 = 64 layers, which is the model's own depth.
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

// First byte of that ordinal's block inside the shared root. Ordinals are 0-based and there are 16 of them here, so the
// last block starts at attention_layer_offset(15, …) and the root needs attention_total_bytes(…, 16).
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

// 48 blocks for this model: the last one starts at gdn_layer_offset(47).

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

// ---- the sequence packet: `%control`, the third caller-owned root -----------------------------------
// `%control` looked opaque in the .loom because the files pass it straight to kernels. It is not: it is the sequence
// packet, its geometry is the function bodies in motif/sequence/qwen3_8/packet.loom, and its first word is the position
// that advances the attention cache -- read directly by every attention layer:
//
//   attention_decode.loom:104   %position_i32 = view.load %control_view[0] : view<1xi32> -> i32
//   attention_decode.loom:108   %token_cache_base = index.mul %position, %c1024
//
// So the handover WRITES this header; it does not negotiate an interface. The three words carry two ABIs over the same
// storage, and which one is live depends only on which command runs next:
//
//   decode  (decode_control.loom:17-19)          [ position,      generated_count, eos_token ]
//   prefill (decode_control.loom:49-51)          [ token_count,   position_base,   eos_token ]
//
// qwen38_prefill_finish_control converts the second into the first: position := position_base + token_count,
// generated_count := 1 (decode_control.loom:54-59, fixtures in test/decode_control_test.loom:53-71). That kernel is the
// handover's landing point, and it reads token_count with no range assumption -- so token_count == 0 is a legal
// zero-token finish that only establishes the position. The public prefill commands are stricter (range 1..512), so the
// kernel is the only zero-length path in the library.
//
// Caveat a writer must carry: word 2 means different things on the two revisions, and neither meaning is the other's.
// To the commit kernels it is `eos_token` on the decode-only control block (decode_control.loom:19,51) — that is true on
// both revisions.
//
//   v0.2.0 (the shipped cards): in the PACKET, the samplers read word 2 as the live row count
//   (kernel/sampling.loom:45,92), as does kernel/output_spans.loom:35, and NOTHING IN THE LIBRARY WRITES THE PACKET'S
//   EPOCH HEADER at this revision — so it is a caller-owned convention: whoever fills the packet must store the count.
//   (Word 2 of *other* three-word headers does have writers — kernel/dflash_proposal.loom:108,111 store a selection into
//   the draft and selection headers — so a grep for `[2]` alone will not settle this; it is the epoch bank's header that
//   the samplers bound on, and that one is the caller's.)
//   HEAD (cb9ae9e): `qwen38_packet_compact_prefix` (motif/sequence/qwen3_8/packet.loom, absent at v0.2.0) writes
//   `[token_count, span_count, output_count]` across words 0/1/2, and the samplers take a `%first` window and bound on
//   `first + row < word 2` (kernel/sampling.loom:45,93) — an ABSOLUTE output count.
//
// So on the release a handover writer MUST store the count itself; on HEAD compaction owns it and the count is absolute.
// A writer that mixes the two revisions' conventions fails silently in both directions: an eos id smaller than the row
// count drops sampled rows on v0.2.0, and a per-call count written where HEAD expects an absolute one drops rows past
// the window.
inline constexpr uint64_t kControlWordBytes  = 4;
inline constexpr uint64_t kControlHeaderBytes = 12;
inline constexpr uint64_t kControlPositionOffset  = 0;  // decode: KV position; prefill: token_count
inline constexpr uint64_t kControlGeneratedOffset = 4;  // decode: generated_count; prefill: position_base
inline constexpr uint64_t kControlEosOffset       = 8;  // eos_token, or the sampling live-row count

// Packet geometry, from packet.loom: the bank pair, then indices / policies / identities / sources / shape.
//
// WHO OWNS THE PACKET'S CAPACITY (open question 2, answered from the shipped sources rather than by preference): the
// packet's capacity is NOT a specialization. `program/text/qwen3_8/epoch.loom:91-92` computes
// `%metadata_bytes = buffer.length %metadata` and then `qwen38_packet_capacity(%metadata_bytes)`, i.e. the encode step
// INFERS capacity from the allocation the caller passed, and the card's own encoder bounds it at
// `range(%packet_capacity, 1, 524288)` (control.loom:2519,2551,3215). So the packet is caller-sized per request -- up to
// the shipped ceiling -- while the ATTENTION CACHE capacity is a specialization fixed at materialization. The two roots
// therefore do not have to move together: one attention-cache class can serve requests with different packet sizes.
//
// The motif's own bound is `range(2, 2147483646), multiple_of(2)`, so the binding limit is the shipped encoder's 524288,
// which fixes the largest packet at 58 720 296 B (~56 MiB) -- the number an exporter budgets for.
inline constexpr uint64_t kPacketCapacityCeiling = 524288;
inline constexpr uint64_t kPacketFixedBytes  = 40;  // two mixed-input capacities after the shape block
inline constexpr uint64_t kPacketStrideBytes = 112;
inline constexpr uint64_t kPolicyRecordBytes = 24;  // f32 temperature, f32 top_p, i32 top_k, i32 flags, i64 seed
inline constexpr uint64_t kPolicySeedOffset  = 16;
inline constexpr uint64_t kIdentityRecordBytes = 8;  // i32 owner, i32 position -- the per-row position to fill
inline constexpr uint64_t kIdentityPositionOffset = 4;
inline constexpr uint64_t kMixedInputCapacitiesBytes = 8;

inline constexpr uint64_t packet_bytes(uint64_t capacity) { return kPacketFixedBytes + capacity * kPacketStrideBytes; }

// One of the two identical banks: 3-word header + 4 bytes of pad, then 6 descriptor words, 1 selection word, 2 option
// words and one EOS word per token.
inline constexpr uint64_t packet_bank_bytes(uint64_t capacity) { return 16 + capacity * 36; }

inline constexpr uint64_t packet_indices_origin(uint64_t capacity)  { return 32 + capacity * 72; }
inline constexpr uint64_t packet_policy_origin(uint64_t capacity)   { return 32 + capacity * 76; }
inline constexpr uint64_t packet_identity_origin(uint64_t capacity) { return 32 + capacity * 100; }
inline constexpr uint64_t packet_sources_origin(uint64_t capacity)  { return 32 + capacity * 108; }
inline constexpr uint64_t packet_shape_origin(uint64_t capacity)    { return 32 + capacity * 112; }

// Inverse of packet_bytes, the way packet.loom:90-97 recovers capacity from an allocation's length (HEAD's copy is at
// :92-99; the file gained a header comment after the tag).
inline constexpr uint64_t packet_capacity(uint64_t bytes) { return (bytes - kPacketFixedBytes) / kPacketStrideBytes; }

}  // namespace onebit::loom
