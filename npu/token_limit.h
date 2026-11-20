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

// How many tokens a request may generate, given the context left (engine#2897).
//
// This exists as a function rather than an expression at the call site for one reason: the sentinel. `n_predict: -1`
// is llama.cpp's "no limit" and llama-server's own default, and `std::clamp(max_tokens, 0, room)` turned it into 0 --
// so a request that asked for "as much as fits" answered an empty completion with `finish_reason: "length"`, which
// reads as success. Three cases, one pure function of two ints, pinned by tests/npu_token_limit_test.cpp.

#pragma once

namespace onebit::npu {

inline int token_limit(int max_tokens, int room) {
    if (max_tokens < 0) {
        return room;  // "no limit": the context is the limit, not a budget of zero
    }
    return max_tokens < room ? max_tokens : room;
}

}  // namespace onebit::npu
