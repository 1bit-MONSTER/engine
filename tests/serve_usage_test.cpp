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

// engine#2111: the reply's `usage` object, checked as a decision table instead of on a device.
//
// The issue is that an empty `content` with `finish_reason: "length"` reads like "the model said nothing" when the
// budget went into a thinking block the engine strips. The field that tells them apart is
// `completion_tokens_details.reasoning_tokens`, and the three cases below must stay three different answers:
//
//   reasoning_tokens  > 0   the route counted them; the field is PRESENT with that number
//   reasoning_tokens == 0   the route counted and found none; the field is PRESENT with 0
//   reasoning_tokens == -1  the route could not count (e.g. /v1/completions); the field is ABSENT, and must never
//                           be read as zero
//
// No model, no NPU, no HTTP: app/serve_usage.cpp is the only thing this links.

#include "forward_serve.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

namespace {

int bad = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++bad;
}

nlohmann::json usage(const onebit::UsageCounters& c) {
    return nlohmann::json::parse(onebit::usage_json_text(c));
}

bool has_reasoning(const nlohmann::json& u) {
    return u.contains("completion_tokens_details") && u["completion_tokens_details"].contains("reasoning_tokens");
}

}  // namespace

int main() {
    {
        onebit::UsageCounters c;
        c.prompt_tokens = 24;
        c.completion_tokens = 8;
        c.reasoning_tokens = 8;
        auto u = usage(c);
        check(has_reasoning(u) && u["completion_tokens_details"]["reasoning_tokens"] == 8,
              "an unclosed block that took the whole budget: the field is present with 8 of 8");
        check(u["completion_tokens"] == 8 && u["total_tokens"] == 32,
              "and the counters are untouched by it (completion 8, total 24+8)");
    }
    {
        onebit::UsageCounters c;
        c.prompt_tokens = 10;
        c.completion_tokens = 5;
        c.reasoning_tokens = 0;
        auto u = usage(c);
        check(has_reasoning(u) && u["completion_tokens_details"]["reasoning_tokens"] == 0,
              "a route that counted and found none reports 0, which is not the same as absent");
    }
    {
        onebit::UsageCounters c;
        c.prompt_tokens = 10;
        c.completion_tokens = 5;
        c.reasoning_tokens = -1;
        auto u = usage(c);
        check(!has_reasoning(u), "a route that could not count it omits the field entirely (never a zero)");
    }
    {
        onebit::UsageCounters c;
        c.prompt_tokens = 7;
        c.completion_tokens = 3;
        c.cached_tokens = 4;
        c.prefix_slot = 2;
        c.prefix_reason = "hit";
        c.prefix_pinned = true;
        auto u = usage(c);
        check(u["prompt_tokens_details"]["cached_tokens"] == 4, "the prefix cache's cached_tokens is still reported");
        check(u["flm_prefix_cache"]["slot"] == 2 && u["flm_prefix_cache"]["reason"] == "hit" &&
                  u["flm_prefix_cache"]["pinned"] == true,
              "and the cache decision (slot, reason, pinned) is unchanged by the extraction");
    }
    {
        // the default-constructed struct is the shape a /v1/completions reply has: no counting was possible
        onebit::UsageCounters c;
        check(!has_reasoning(usage(c)), "a default UsageCounters omits the field, so a caller that never set it is safe");
        check(usage(c)["flm_prefix_cache"]["slot"] == -1, "and 'off' is -1, as the prefix cache already spells it");
    }

    std::printf("serve_usage: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
