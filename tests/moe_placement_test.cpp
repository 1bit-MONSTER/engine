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

// moe_placement_test: the reader for a tools/moe_placement.py table (app/moe_placement.h, engine#1702). One table that
// a consumer may adopt, then one arm per refusal: each is a plausible-but-wrong table that a reader without the check
// would adopt silently. Header only, no device, no model.
#include "moe_placement.h"

#include <cstdio>
#include <functional>
#include <string>

namespace {

using onebit::moe::parse_placement;
using onebit::moe::Placement;
using onebit::moe::PlacementExpect;
using nlohmann::json;

int fails = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) fails++;
}

// The tool's own shape (docs/moe-streaming.md): two routed layers (1 and 3; 0 and 2 are dense), 8 experts, 2 groups.
json good() {
    return json::parse(R"({"schema": "1bit/moe-placement/1", "groups": 2,
        "share_in_sample": 0.5113, "share_held_out": 0.2252, "held_out_steps": 14400,
        "layers": {"1": {"decode_steps": 720, "experts": 8, "experts_seen": 6, "unseen": 2,
                         "groups": [[0, 2, 4, 6], [1, 3, 5, 7]]},
                   "3": {"decode_steps": 720, "experts": 8, "experts_seen": 8, "unseen": 0,
                         "groups": [[7, 6, 5], [0, 1, 2, 3, 4]]}}})");
}
PlacementExpect want() { return PlacementExpect{{1, 3}, 8, 2}; }

// The table must be refused, and the message must name `needle` so the operator sees which check it failed.
void refused(const std::string& what, const json& t, const PlacementExpect& w, const std::string& needle) {
    try {
        parse_placement(t, w);
        check(false, what + " (adopted)");
    } catch (const std::runtime_error& e) {
        const std::string m = e.what();
        check(m.find(needle) != std::string::npos, what + " -> " + m);
    }
}
json with(const std::function<void(json&)>& edit) {
    json t = good();
    edit(t);
    return t;
}

}  // namespace

int main() {
    {
        const Placement p = parse_placement(good(), want());
        check(p.groups == 2 && p.layers.size() == 2, "a table for this model is adopted: 2 groups, layers 1 and 3");
        check(p.group_of(1, 4) == 0 && p.group_of(1, 5) == 1 && p.group_of(3, 7) == 0 && p.group_of(3, 0) == 1,
              "group_of reads the placement back by (layer, global expert id)");
        check(p.share_held_out == 0.2252 && p.share_in_sample == 0.5113 && p.held_out_steps == 14400,
              "both shares are kept, the held-out one is the gated one");
        check(p.layers.at(1).unseen == 2, "unseen is carried, so a layer placed mostly by default reads that way");
        bool threw = false;
        try { (void)p.group_of(2, 0); } catch (const std::out_of_range&) { threw = true; }
        check(threw, "a dense layer has no placement to read");
    }
    refused("another schema", with([](json& t) { t["schema"] = "something/else"; }), want(), "something/else");
    refused("no schema", with([](json& t) { t.erase("schema"); }), want(), "schema \"\"");
    refused("a table for 4 devices on a 2-device run", with([](json& t) { t["groups"] = 4; }), want(), "4 groups");
    refused("no held-out share: in-sample is not the gate", with([](json& t) { t.erase("share_held_out"); }), want(),
            "share_in_sample (51.1 %)");
    refused("held-out share under the 15 % floor", with([](json& t) { t["share_held_out"] = 0.07; }), want(),
            "7.0 % is under the gate 15.0 %");
    {
        PlacementExpect w = want();
        w.min_held_out = 0.0;
        json t = good();
        t["share_held_out"] = 0.07;
        bool ok = true;
        try { parse_placement(t, w); } catch (...) { ok = false; }
        check(ok, "the floor is the caller's: a caller that sets it to 0 adopts a 7 % table");
    }
    refused("a routed layer missing", with([](json& t) { t["layers"].erase("3"); }), want(), "layer 3 is missing");
    refused("a layer this model routes nowhere", with([](json& t) { t["layers"]["2"] = t["layers"]["1"]; }), want(),
            "layer 2 is not a routed layer");
    refused("a non-numeric layer key", with([](json& t) { t["layers"]["01"] = t["layers"]["1"]; }), want(),
            "\"01\" is not a layer number");
    refused("built for another expert count", with([](json& t) { t["layers"]["1"]["experts"] = 16; }), want(),
            "built for 16 experts");
    refused("an expert id outside the model",
            with([](json& t) { t["layers"]["3"]["groups"] = json::parse("[[7, 6, 5, 8], [0, 1, 2, 3, 4]]"); }),
            want(), "expert 8 is outside 0..7");
    refused("an expert placed twice",
            with([](json& t) { t["layers"]["3"]["groups"] = json::parse("[[7, 6, 5, 4], [0, 1, 2, 3, 4]]"); }),
            want(), "expert 4 is placed twice");
    refused("an expert placed nowhere",
            with([](json& t) { t["layers"]["3"]["groups"] = json::parse("[[7, 6], [0, 1, 2, 3, 4]]"); }), want(),
            "layer 3: expert 5 is not placed");
    refused("a layer with the wrong group count",
            with([](json& t) { t["layers"]["1"]["groups"] = json::parse("[[0,1,2,3,4,5,6,7]]"); }), want(),
            "needs 2 groups");
    refused("a fractional id", with([](json& t) { t["layers"]["1"]["groups"][0][0] = 0.5; }), want(), "non-integer id");
    {
        // The stale-table case the reader exists for: 8 experts' table on a 16-expert model with no `experts` field.
        json t = good();
        for (auto& [k, v] : t["layers"].items()) v.erase("experts");
        PlacementExpect w = want();
        w.n_experts = 16;
        refused("an 8-expert table on a 16-expert model, with no experts field to compare", t, w,
                "expert 8 is not placed");
    }
    refused("the caller names nothing", good(), PlacementExpect{}, "must name its routed layers");
    std::printf("%s\n", fails ? "moe_placement_test: FAILED" : "moe_placement_test: ok");
    return fails ? 1 : 0;
}
