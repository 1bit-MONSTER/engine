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

// engine#2111: the chat reply's `usage` object, as a pure function.
//
// Its own translation unit for one reason: tests/serve_usage_test.cpp has to link it, and linking
// app/forward_serve.cpp would drag in the NPU runtime (npu/rt.h refuses a build without ONEBIT_NPU_RT_LIB), so
// the field this issue is about could only ever be checked by running a model on the device. Here it is a decision
// table with known answers - see the header comment for the three cases that must stay distinct.

#include "forward_serve.h"

#include <nlohmann/json.hpp>

namespace onebit {

using json = nlohmann::json;

// The usage object, verbatim what the reply carries. Two fields are not OpenAI's and are here because a lane
// needed them: how many prompt tokens came from a saved prefix, and what the cache decided.
// `reasoning_tokens` is ABSENT when the route could not count it (-1), which is not the same as 0 (counted, none).
std::string usage_json_text(const UsageCounters& c) {
    json u{{"prompt_tokens", c.prompt_tokens},
           {"completion_tokens", c.completion_tokens},
           {"total_tokens", c.prompt_tokens + c.completion_tokens},
           {"prompt_tokens_details", {{"cached_tokens", c.cached_tokens}}},
           {"flm_prefix_cache", {{"slot", c.prefix_slot},
                                 {"cached_tokens", c.cached_tokens},
                                 {"reason", c.prefix_reason},
                                 {"pinned", c.prefix_pinned}}}};
    // With content "" and finish_reason "length", reasoning_tokens == completion_tokens is the sentence that says the
    // budget ran out inside a thinking block, rather than that the model said nothing.
    if (c.reasoning_tokens >= 0) u["completion_tokens_details"] = {{"reasoning_tokens", c.reasoning_tokens}};
    return u.dump();
}

}  // namespace onebit
