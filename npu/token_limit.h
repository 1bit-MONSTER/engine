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

#pragma once

namespace onebit::npu {

inline int token_limit(int max_tokens, int room) {
    if (max_tokens < 0) {
        return room;  // "no limit": the context is the limit, not a budget of zero
    }
    return max_tokens < room ? max_tokens : room;
}

}  // namespace onebit::npu
