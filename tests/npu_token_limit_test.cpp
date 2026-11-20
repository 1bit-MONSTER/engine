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

// Token generation on the fast lane: one lane run per prompt token, then decode

#include "token_limit.h"

#include <cstdio>

using onebit::npu::token_limit;

static int failures = 0;

static void check(const char * what, int got, int want) {
    if (got == want) {
        std::printf("ok   %s (%d)\n", what, got);
    } else {
        std::printf("FAIL %s: got %d, want %d\n", what, got, want);
        ++failures;
    }
}

int main() {
    // -1 is llama.cpp's default for n_predict and its documented "no limit": the context is the limit.
    check("-1 (no limit) is the whole room", token_limit(-1, 512), 512);
    // Any negative is the same statement; llama.cpp has never defined -2 as anything else.
    check("-5 is also 'no limit'", token_limit(-5, 40), 40);
    // A room of one still yields one: the "no limit" path must not read as "nothing left".
    check("-1 with one slot of room", token_limit(-1, 1), 1);
    // Zero keeps its literal meaning: a caller asking for no tokens gets none.
    check("0 stays 0", token_limit(0, 512), 0);
    // A positive budget is capped by the room, unchanged from before.
    check("3 with room to spare", token_limit(3, 512), 3);
    check("999 is capped by the room", token_limit(999, 512), 512);
    // The boundary the old clamp got wrong is exactly -1 -> 0, so pin the pair side by side.
    check("the regression: -1 is not 0", token_limit(-1, 512) == 0 ? 0 : 512, 512);

    if (failures == 0) {
        std::printf("token_limit: %d arms pass\n", 7);
        return 0;
    }
    std::printf("token_limit: %d failure(s)\n", failures);
    return 1;
}
