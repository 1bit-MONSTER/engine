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

// engine#2897: the token budget's three cases, as a row CI runs.
//
// The defect this pins: `n_predict: -1` is llama.cpp's "no limit" and llama-server's own default, and
// `std::clamp(max_tokens, 0, room)` turned it into 0 -- an empty completion with `finish_reason: "length"`, which
// reads as a successful request. A sentinel that means "as much as fits" is arithmetic, so it is tested rather than
// reviewed: no device, no model, no server.

#include "token_limit.h"

#include <cstdio>
#include <limits>

using onebit::npu::token_limit;

static int failures = 0;

static void check(const char* what, int got, int want) {
    if (got == want) {
        std::printf("ok   %s (%d)\n", what, got);
    } else {
        std::printf("FAIL %s: want %d, got %d\n", what, want, got);
        ++failures;
    }
}

int main() {
    // "--1" is the sentinel: the room, not a budget of zero.
    check("-1 (no limit) is the whole room", token_limit(-1, 512), 512);
    check("-5 is also no limit", token_limit(-5, 40), 40);
    check("-1 with one slot of room", token_limit(-1, 1), 1);
    check("INT_MIN is still no limit", token_limit(std::numeric_limits<int>::min(), 7), 7);

    // Zero is a real budget: the short warm-up asks for exactly this and must keep getting nothing.
    check("0 stays 0", token_limit(0, 512), 0);
    check("0 with one slot of room", token_limit(0, 1), 0);

    // A positive budget is itself, capped by the context.
    check("a budget inside the room is unchanged", token_limit(2, 512), 2);
    check("a budget past the room is the room", token_limit(4096, 12), 12);
    check("a budget equal to the room", token_limit(12, 12), 12);

    if (failures == 0) {
        std::printf("npu_token_limit: all checks ok\n");
        return 0;
    }
    std::printf("npu_token_limit: %d failed\n", failures);
    return 1;
}
