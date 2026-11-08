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

// The route decision record of `1bit serve --laya-models` (app/route_log.h) and the one field that
// makes a decision log usable as data: the classified state (engine#1313, docs/laya-models.md).
//
// A decision that carries `laya_class` but not the text Laya classified is a verdict with no
// evidence: it cannot become an `--eval` case and it cannot become fine-tune data, which is what
// item 3 of the issue asks for.  The state is therefore recorded, bounded, and its full length kept
// so a truncated state is visibly truncated rather than silently "the state".
// Header only: no model, no checkpoint, no device.
#include "route_log.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}

onebit::RouteDecision classified(const std::string& state) {
    onebit::RouteDecision d;
    d.id = 7;
    d.cls = d.laya_class = "reflex";
    d.confidence = 0.91f;
    d.model = "small-npu";
    d.device = "npu";
    d.router_ms = 12.5;
    d.state = onebit::bounded_state(state, d.state_len);
    return d;
}
}  // namespace

int main() {
    using nlohmann::json;

    // a step-sized state is kept whole, and the log says so
    {
        const std::string text = "A tree is in front of you. What do you do now?";
        const onebit::RouteDecision d = classified(text);
        check("a short state is kept verbatim", d.state == text);
        check("its full length is recorded", d.state_len == static_cast<long>(text.size()));
        const json j = d.to_json();
        check("to_json carries the state", j["state"] == text);
        check("to_json carries its length", j["state_len"] == static_cast<long>(text.size()));
        check("the verdict fields are unchanged next to it",
              j["class"] == "reflex" && j["laya_class"] == "reflex" && j["model"] == "small-npu" &&
                  j["device"] == "npu");
    }

    // a long prompt is bounded, and the reader can see that what it got is a prefix
    {
        const std::string text(onebit::kRouteStateMax + 1000, 'x');
        const onebit::RouteDecision d = classified(text);
        check("a long state is bounded to kRouteStateMax", d.state.size() == onebit::kRouteStateMax);
        check("the full length is still the full length", d.state_len == static_cast<long>(text.size()));
        const json j = d.to_json();
        check("to_json marks it as truncated (state_len > state.size())",
              j["state_len"].get<long>() > static_cast<long>(j["state"].get<std::string>().size()));
    }

    // exactly at the bound is not truncated -- an off-by-one here would silently clip real steps
    {
        const std::string text(onebit::kRouteStateMax, 'y');
        const onebit::RouteDecision d = classified(text);
        check("a state exactly at the bound is kept whole",
              d.state.size() == onebit::kRouteStateMax && d.state_len == static_cast<long>(text.size()));
    }

    // an empty state IS a decision (a request with no user message still gets classified): the fields
    // are present and say zero, which is different from "no state was recorded" below
    {
        const onebit::RouteDecision d = classified("");
        const json j = d.to_json();
        check("an empty classified state is recorded as empty, not as absent",
              j.contains("state") && j["state"] == "" && j["state_len"] == 0L);
    }

    // a pinned request was never a decision: no state, and no empty field pretending otherwise
    {
        onebit::RouteDecision d;
        d.id = 8;
        d.cls = "pinned";
        d.model = "small-npu";
        d.device = "npu";
        const json j = d.to_json();
        check("a pinned decision has no state field", !j.contains("state") && !j.contains("state_len"));
    }

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
