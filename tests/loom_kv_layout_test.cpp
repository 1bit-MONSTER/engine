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
// loom_kv_layout_test.cpp -- the Loom hybrid handover's state arithmetic (app/loom_kv_layout.h, engine#3157).
//
// Every expected number here is read out of the MIT Loom sources (hrx-loom-library ba7abef, Qwen/Qwen3.8-27B), not
// produced by the header, so this is a check on the arithmetic rather than a snapshot of it:
//
//   attention_layer_decode.loom:75  %cache_token_bytes = 2048        (1024 f16 per token, per K and per V)
//   attention_layer_decode.loom:77  buffer.pack [align(256) cache_bytes, align(256) cache_bytes]
//   attention_layer_decode.loom:82  view<[cache_capacity * 1024] x f16>   (token-major)
//   gdn_layer_decode.loom:69        %recurrent_state_bytes = 3145728 (786432 f32)
//   gdn_layer_decode.loom:70        %conv_history_bytes    = 122880  (30720 f32)
//
// No device and no Loom: the whole point of keeping this as arithmetic is that the writer's contract can be tested on
// any box, including CI runners with no HRX at all.
#include "loom_kv_layout.h"

#include <cstdio>

namespace L = onebit::loom;

static int failures = 0;

static void check(bool cond, const char * what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

static void check_eq(uint64_t got, uint64_t want, const char * what) {
    if (got != want) {
        std::printf("FAIL %s: got %llu, want %llu\n", what, (unsigned long long) got, (unsigned long long) want);
        ++failures;
    } else {
        std::printf("ok   %s (%llu)\n", what, (unsigned long long) got);
    }
}

int main() {
    // The two numbers a writer must not get wrong, quoted from the sources above.
    check_eq(L::align_up(1, 256), 256, "align_up rounds up, does not truncate");
    check_eq(L::kGdnRecurrentBytes % 256, 0, "the recurrent state is 256-aligned (so the pack adds no padding)");
    check_eq(L::kGdnConvBytes % 256, 0, "the conv history is 256-aligned (same reason)");

    // Qwen3.8-27B's shape: 1024 f16 per token per side, 15 attention ordinals, 47 gdn ordinals.
    const uint64_t kv = 1024;

    check_eq(kv * L::kF16Bytes, 2048, "one token of key storage is 2048 B (Loom's %cache_token_bytes)");
    check_eq(L::attention_side_bytes(4096, kv), 4096ull * 2048ull, "a 4096-token side needs no padding");
    check_eq(L::attention_value_offset(4096, kv), 4096ull * 2048ull, "the value array starts one side after the key");
    check_eq(L::attention_layer_bytes(4096, kv), 2ull * 4096ull * 2048ull, "a layer block is key + value");
    check_eq(L::attention_layer_offset(3, 4096, kv), 3ull * 2ull * 4096ull * 2048ull, "ordinal 3's block offset");

    // A capacity whose sides are not 256-aligned must be padded, not overlapped: kv=1023 f16 = 2046 B/token.
    check_eq(L::attention_side_bytes(5, 1023), L::align_up(5 * 1023 * 2, 256), "a ragged side is aligned up");
    check_eq(L::attention_side_bytes(5, 1023) % 256, 0, "…so every value offset stays 256-aligned");
    check_eq(L::attention_value_offset(1, 1) % 256, 0, "the smallest possible side is still aligned");

    // GDN: per-ordinal block and offsets, from the two constants Loom packs.
    check_eq(L::gdn_conv_offset(), L::kGdnRecurrentBytes, "conv history follows the recurrent state with no padding");
    check_eq(L::gdn_layer_bytes(), 3145728ull + 122880ull, "one gdn ordinal is recurrent + conv = 3268608 B");
    check_eq(L::gdn_layer_offset(47 - 1), 46ull * 3268608ull, "the last gdn ordinal's block offset");

    // The whole state for that model at 4096 context: 15 attention blocks + 47 gdn blocks.
    const uint64_t attn = L::attention_total_bytes(4096, kv, 15);
    const uint64_t gdn  = L::gdn_total_bytes(47);
    check_eq(attn, 15ull * 2ull * 4096ull * 2048ull, "15 attention ordinals at 4096 tokens");
    check_eq(gdn, 47ull * 3268608ull, "47 gdn ordinals (capacity-independent)");
    check_eq(L::state_total_bytes(4096, kv, 15, 47), attn + gdn, "the two roots add up to what the exporter allocates");

    // Sizing sanity against the number this lane already measures on the other route: docs/zero-copy.md:197 records the
    // GDN sweep as 604 MB read+written = 48 layers x 3.14 MB x ~4 passes. Loom's per-ordinal state is the same 3.14 MB
    // object, so the two routes hold the same state -- which is the fact that makes a handover a re-layout, not a
    // re-computation.
    check_eq(L::gdn_layer_bytes() / 100000, 32ull, "one gdn block is ~3.27 MB (~3.14 MB of recurrent state)");
    check(L::gdn_layer_bytes() < 4ull * 1024ull * 1024ull, "…and fits comfortably under the 3.14 MB x 4 figure");

    // Token-major indexing: position p, element j inside one side.
    check_eq(L::attention_element_index(7, 0, kv), 7ull * kv, "element 0 of position 7 is at 7 * kv");
    check_eq(L::attention_element_index(7, kv - 1, kv), 7ull * kv + kv - 1, "the last element of position 7");
    check(L::attention_element_index(8, 0, kv) - L::attention_element_index(7, kv - 1, kv) == 1,
          "positions are contiguous: the row after 7's last element is 8's first");

    // Alignment invariants over a sweep, because a writer indexes with these and an unaligned block would put a value
    // array across two pages for no reason.
    bool aligned = true;
    for (uint64_t cap = 1; cap <= 8192; cap += 7) {
        for (uint64_t row = 1; row <= 2048; row += 13) {
            if (L::attention_layer_offset(11, cap, row) % 256 != 0 || L::attention_value_offset(cap, row) % 256 != 0) {
                aligned = false;
            }
        }
    }
    check(aligned, "every attention block and value offset is 256-aligned over a capacity/row sweep");
    check(L::gdn_layer_offset(1) % 256 == 0 && L::gdn_layer_offset(46) % 256 == 0, "every gdn block offset is aligned");

    // ---- the control header: the position that advances the cache -----------------------------------
    // attention_layer_decode.loom:104 loads it as view<1xi32> at offset 0, so the handover establishes the decode
    // position by writing word 0. Nothing else has to be taught anything.
    check_eq(L::kControlPositionOffset, 0, "the KV position is word 0 of the control buffer");
    check_eq(L::kControlGeneratedOffset, 4, "generated_count / position_base is word 1");
    check_eq(L::kControlEosOffset, 8, "eos / live row count is word 2");
    check_eq(L::kControlHeaderBytes, 3 * L::kControlWordBytes, "the header is three words");
    check(L::kControlHeaderBytes + 4 == 16, "3 words plus the pad are the bank's fixed 16 bytes");

    // ---- packet geometry, checked against Loom's own sampling fixture -------------------------------
    // test/sampling_test.loom fills a tensor<3594xi32> epoch and hardcodes policy_origin 9760 and identity_origin
    // 12832 with 128 identity rows and 16 policy rows. Those literals are independent of packet.loom's arithmetic,
    // so agreeing with them checks the geometry rather than restating it.
    const uint64_t cap = 128;
    check_eq(L::packet_policy_origin(cap), 9760, "policy origin at capacity 128 matches the sampling fixture");
    check_eq(L::packet_identity_origin(cap), 12832, "identity origin at capacity 128 matches the sampling fixture");
    check_eq(L::packet_bytes(cap), 3594 * 4, "capacity 128 is 3594 words -- the fixture's epoch tensor");
    check_eq(L::packet_capacity(3594 * 4), 128, "capacity recovered from the allocation length");
    check_eq(L::packet_bank_bytes(cap), 16 + cap * 36, "a bank is a 3-word header plus 9 words per token");
    check_eq(2 * L::packet_bank_bytes(cap), L::packet_indices_origin(cap), "the second bank ends where the indices begin");

    // The banks and the tail regions must tile the packet exactly: any gap is a region a writer can corrupt, and any
    // overlap is two regions fighting over the same bytes.
    bool tiles = true;
    for (uint64_t c = 2; c <= 2048; c += 2) {
        const bool exact = L::packet_policy_origin(c) + c * L::kPolicyRecordBytes == L::packet_identity_origin(c) &&
                           L::packet_identity_origin(c) + c * L::kIdentityRecordBytes == L::packet_sources_origin(c) &&
                           L::packet_sources_origin(c) + c * L::kControlWordBytes == L::packet_shape_origin(c) &&
                           L::packet_shape_origin(c) + L::kMixedInputCapacitiesBytes == L::packet_bytes(c) &&
                           L::packet_capacity(L::packet_bytes(c)) == c;
        if (!exact) { tiles = false; }
    }
    check(tiles, "banks, indices, policies, identities, sources and shape tile the packet with no gap or overlap");

    // The policy and identity records are what the sampler and the attention writer index into.
    check_eq(L::kPolicySeedOffset + 8, L::kPolicyRecordBytes, "a policy record ends with its i64 seed");
    check_eq(L::kIdentityPositionOffset + 4, L::kIdentityRecordBytes, "an identity record is (owner, position)");

    if (failures == 0) {
        std::printf("loom_kv_layout_test: pass\n");
        return 0;
    }
    std::printf("loom_kv_layout_test: %d failure(s)\n", failures);
    return 1;
}
