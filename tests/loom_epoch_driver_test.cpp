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
// loom_epoch_driver_test.cpp -- the handover's chunking and control words (app/loom_epoch_driver.h, engine#3157).
//
// Every expected number here is either a literal derived from the Loom sources or a property that must hold for ANY
// chunking (coverage, no gap, no overlap, absolute bases), never a snapshot of what the header happens to return:
//
//   <card>/program/llm.loom via `loom-link --print-config-schema`: llm.spans / llm.tokens / llm.epoch_spans /
//     llm.epoch_tokens are `required, range 1 .. 8192`; llm.context is `range 16 .. 262144`
//   kernel/attention_decode.loom:104,108  position is word 0 of the control block, absolute
//   kernel/decode_control.loom:49-51      prefill form  [token_count, position_base, eos]
//   kernel/decode_control.loom:17-19      decode form   [position, generated_count, eos]
//   kernel/decode_control.loom:54-59      finish sets position := position_base + token_count, generated_count := 1
//
// No device and no Loom: this is the arithmetic a driver runs before it touches memory.
#include "loom_epoch_driver.h"

#include <cstdio>

namespace L = onebit::loom;

static int failures = 0;

static void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL %s\n", what);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

static void check_eq(uint64_t got, uint64_t want, const char* what) {
    if (got != want) {
        std::printf("FAIL %s: got %llu, want %llu\n", what, (unsigned long long) got, (unsigned long long) want);
        ++failures;
    } else {
        std::printf("ok   %s (%llu)\n", what, (unsigned long long) got);
    }
}

int main() {
    // The two bounds the driver inherits from the card, quoted as literals.
    check_eq(L::kEpochTokenLimit, 8192, "the card's per-epoch token ceiling (llm.tokens range 1..8192)");
    check_eq(L::kContextCeiling, 262144, "the card's context ceiling (llm.context range 16..262144)");
    check_eq(L::kContextCeiling / L::kEpochTokenLimit, 32, "so a full-context prompt is 32 epochs, not one");

    // Epoch counting: exact multiples, remainders, one token, and the zero-token finish.
    check_eq(L::epoch_count(8192), 1, "one full epoch");
    check_eq(L::epoch_count(8193), 2, "one token over the limit needs a second epoch");
    check_eq(L::epoch_count(1), 1, "a single token is one epoch");
    check_eq(L::epoch_count(0), 1, "zero tokens is still one epoch: the zero-length finish that sets the position");
    check_eq(L::epoch_count(262144), 32, "a 262144-token prompt");
    check_eq(L::epoch_count(18432), 3, "18.8k context class -> 3 epochs");
    check_eq(L::epoch_count(0, 0), 0, "a zero limit is a caller error, not an infinite loop");

    // Epoch shapes: base advances by the full limit, the last epoch is short, positions are absolute.
    const L::Epoch e0 = L::epoch_at(0, 0, 18432);
    const L::Epoch e1 = L::epoch_at(1, 0, 18432);
    const L::Epoch e2 = L::epoch_at(2, 0, 18432);
    check_eq(e0.position_base, 0, "epoch 0 starts at the handover base");
    check_eq(e0.token_count, 8192, "epoch 0 is full");
    check_eq(e1.position_base, 8192, "epoch 1 starts at base + limit");
    check_eq(e1.token_count, 8192, "epoch 1 is full");
    check_eq(e2.position_base, 16384, "epoch 2 starts at base + 2*limit");
    check_eq(e2.token_count, 2048, "the last epoch is short: 18432 - 16384");
    check_eq(e2.position_base + e2.token_count, 18432, "and it ends exactly at base + tokens");

    // A non-zero base -- a handover into a session that already decoded -- moves every epoch by the same amount.
    const L::Epoch shifted = L::epoch_at(1, 1000, 18432);
    check_eq(shifted.position_base, 9192, "a base of 1000 shifts epoch 1 to 1000 + 8192");
    check_eq(shifted.token_count, 8192, "shifting the base does not change the chunking");

    // Properties that must hold for ANY (tokens, limit): exact coverage, no gap, no overlap, and the last epoch's end is
    // base + tokens. This is the arm that would catch a driver that rounded, dropped the tail, or overlapped chunks.
    bool covered = true, monotone = true, bounded = true;
    for (uint64_t tokens = 0; tokens <= 40000; tokens += 97) {
        for (uint64_t limit = 1; limit <= L::kEpochTokenLimit; limit *= 3) {
            const uint64_t n = L::epoch_count(tokens, limit);
            uint64_t sum = 0, expected_base = 128;  // non-zero base on purpose
            for (uint64_t i = 0; i < n; ++i) {
                const L::Epoch e = L::epoch_at(i, 128, tokens, limit);
                if (e.position_base != expected_base) { monotone = false; }
                if (e.token_count > limit) { bounded = false; }
                sum += e.token_count;
                expected_base += limit;
            }
            if (sum != tokens) { covered = false; }
        }
    }
    check(covered, "every epoch's tokens sum to the handover length over a sweep");
    check(monotone, "each epoch starts where the previous one's base plus limit lands (no gap, no overlap)");
    check(bounded, "no epoch exceeds the limit over a sweep");

    // The control words, in both ABIs, straight from the library's field order.
    check_eq(L::prefill_control_word(0, 0, 18432, 0), 8192, "prefill word 0 of epoch 0 is its token count");
    check_eq(L::prefill_control_word(1, 0, 18432, 1), 8192, "prefill word 1 of epoch 1 is its position base");
    check_eq(L::prefill_control_word(2, 0, 18432, 0), 2048, "prefill word 0 of the last epoch is the remainder");
    check_eq(L::prefill_control_word(2, 0, 18432, 2), 0, "word 2 is not computed here: eos is the caller's");

    check_eq(L::handover_position(0, 18432), 18432, "after prefilling N tokens at base 0 the decode position is N");
    check_eq(L::handover_position(1000, 10), 1010, "…and it shifts with the base");
    check_eq(L::decode_control_word(0, 18432, 0), 18432, "decode word 0 is the absolute position after the handover");
    check_eq(L::decode_control_word(0, 18432, 1), 1, "decode word 1 is generated_count = 1 after a handover");
    check_eq(L::decode_control_word(0, 18432, 2), 0, "word 2 (eos) is the caller's, not the driver's");

    // The zero-token finish: it must NOT advance the position, which is what makes it usable to establish a decode block
    // on top of a cache someone else filled.
    check_eq(L::handover_position(2000, 0), 2000, "a zero-token handover leaves the position where it was");
    check_eq(L::epoch_at(0, 2000, 0).token_count, 0, "and its single epoch carries no tokens");

    if (failures == 0) {
        std::printf("loom_epoch_driver_test: pass\n");
        return 0;
    }
    std::printf("loom_epoch_driver_test: %d failure(s)\n", failures);
    return 1;
}
