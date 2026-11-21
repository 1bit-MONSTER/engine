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
// loom_epoch_driver.h -- how a hybrid handover is DRIVEN, given the layout (engine#3157, L4).
//
// app/loom_kv_layout.h answers "where does the state live"; this answers "in what chunks is it filled and what does the
// control block say between chunks". Two facts from the shipped card force the answer:
//
//   1. A Loom epoch carries at most 8192 tokens. `loom-link --print-config-schema <card>/program/llm.loom` reports
//      llm.spans / llm.tokens / llm.epoch_spans / llm.epoch_tokens all `required, range 1 .. 8192`, while llm.context is
//      `range 16 .. 262144`. A 262144-token prompt is therefore 32 epochs, not one.
//   2. The position that places KV is word 0 of the caller-owned control block, absolute and not epoch-relative:
//      kernel/attention_decode.loom:104 loads it and :108 multiplies it into %token_cache_base, and
//      kernel/decode_control.loom:54-59 sets it to position_base + token_count when a prefill finishes.
//
// So a handover of N tokens at absolute base B is a sequence of epochs at bases B, B+limit, ... whose token counts sum to
// N, after which the decode form of the control block carries position B+N. That is all this header computes: no device,
// no Loom, no runtime. It is arithmetic so that a driver's chunking can be tested before it touches memory, the same
// reason app/loom_kv_layout.h is arithmetic.
#pragma once

#include <cstdint>

namespace onebit::loom {

// The per-epoch token ceiling of the `llm.*` card family, which is every text card we ship except one: 55 of the 59
// cards in the corpus declare it and **all 55 agree on 8192** (`llm.tokens`/`llm.spans` `range(1, 8192)`, checked across
// the corpus, not sampled).
//
// IT IS NOT UNIVERSAL, and a driver must read the limit from its card rather than assume this default. The exception is
// Qwen3.8-Flash-Next, which has no `llm.loom` at all: its epoch command declares `flash_next.token_count`
// (`program/epoch.loom:11`) with `range(1, 32768)` -- **four times** the `llm.*` bound, under a different config prefix.
// So the two families are 8192 and 32768, both are shipped, and passing the wrong one is a real failure mode: too small
// only costs extra epochs, too large asks the program for more tokens in one epoch than its own schema admits.
inline constexpr uint64_t kEpochTokenLimit = 8192;
inline constexpr uint64_t kFlashNextEpochTokenLimit = 32768;

// The capacity class a card may be materialized at -- ALSO PER CARD, for the same reason. 54 of the 55 `llm.*` cards bound
// `llm.context` at `range(16, 262144)`; **unsloth-Inkling-2-Mini-loom bounds it at `range(16, 1048576)`** and ships a single
// preset at that full capacity. So this ceiling is the common one, not the only one, and a driver sizing a cache must use
// the card's own bound.
//
// AND A CAPACITY IS NOT A RESIDENCY, which is why no sizing formula belongs in this header. Inkling's cache is PAGED:
// `program/llm.loom:8-32` maps `position / 16` through a per-session page map into a `pool` of physical slots
// (`plane_slot = plane * pool + physical * 16 + within`), and `llm.pool` is a separate config the card's preset does not
// set. app/loom_kv_layout.h models the CONTIGUOUS cache (the 27B's `%token_cache_base = position * 1024`), so its totals
// and its element indexing describe the contiguous-class cards and not the paged ones; for a paged card the resident size
// is 2 * pool * 2048 * layers, which needs the program, not the preset.
inline constexpr uint64_t kContextCeiling = 262144;
inline constexpr uint64_t kInklingContextCeiling = 1048576;

struct Epoch {
    uint64_t position_base;  // absolute position of this epoch's first token -- control word 0 of the prefill form
    uint64_t token_count;    // tokens this epoch fills -- control word 0 is token_count, word 1 is position_base
};

// Epochs needed to fill `tokens` at `limit` per epoch. A zero-token handover is one epoch of zero: that is the
// zero-length `qwen38_prefill_finish_control` path, which establishes a position without writing a cache byte, and a
// driver that returns 0 epochs here would skip the only thing that sets up the decode block.
inline constexpr uint64_t epoch_count(uint64_t tokens, uint64_t limit = kEpochTokenLimit) {
    if (limit == 0) {
        return 0;
    }
    if (tokens == 0) {
        return 1;
    }
    return (tokens + limit - 1) / limit;
}

// The i-th epoch. Its base advances by the FULL limit, not by tokens actually consumed, so that a short final epoch
// keeps its absolute base: position is a property of the token, never of the chunk index.
inline constexpr Epoch epoch_at(uint64_t index, uint64_t base_position, uint64_t tokens,
                                uint64_t limit = kEpochTokenLimit) {
    const uint64_t base = base_position + index * limit;
    const uint64_t consumed = index * limit;
    const uint64_t remaining = tokens > consumed ? tokens - consumed : 0;
    return Epoch{base, remaining < limit ? remaining : limit};
}

// The absolute position that places the FIRST token of the NEXT step after a handover of `tokens` at `base_position` --
// i.e. the decode form's word 0 (kernel/decode_control.loom:54-59). A driver that hands over and then decodes must use
// this and nothing else, because attention derives its context length from it as position + 1.
inline constexpr uint64_t handover_position(uint64_t base_position, uint64_t tokens) {
    return base_position + tokens;
}

// Words of the control block, in both ABIs the library uses over the same storage.
inline constexpr uint64_t prefill_control_word(uint64_t epoch_index, uint64_t base_position, uint64_t tokens,
                                               uint64_t word, uint64_t limit = kEpochTokenLimit) {
    const Epoch e = epoch_at(epoch_index, base_position, tokens, limit);
    // 0 = token_count, 1 = position_base (kernel/decode_control.loom:49-51)
    return word == 0 ? e.token_count : (word == 1 ? e.position_base : 0);
}

inline constexpr uint64_t decode_control_word(uint64_t base_position, uint64_t tokens, uint64_t word) {
    // 0 = position, 1 = generated_count (kernel/decode_control.loom:17-19); a handover leaves 1 generated token
    return word == 0 ? handover_position(base_position, tokens) : (word == 1 ? 1 : 0);
}

}  // namespace onebit::loom
