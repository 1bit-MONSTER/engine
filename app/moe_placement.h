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

// Reading a `tools/moe_placement.py` table (schema "1bit/moe-placement/1", engine#1702) for a consumer such as a
// two-device expert splitter. The table is an offline artifact keyed by (layer, global expert id), so the failure that
// matters is silent: a table written for another model, another group count or another expert count still places
// "every expert exactly once" of *its* model. This reader therefore checks the table against what the caller is about to
// run, and refuses by name rather than adopting a plausible table:
//   - `schema` is exactly "1bit/moe-placement/1";
//   - `groups` is the caller's device count, and every layer has exactly that many groups;
//   - the layers are exactly the caller's routed layers (none missing, none extra), keyed by decimal layer number;
//   - in each layer every expert id 0..n_experts-1 appears exactly once (an id out of range, a repeat or a gap refuses),
//     and a recorded `experts` count must equal n_experts;
//   - `share_held_out` is present and at least the gate. The #1702 decision states the gate on the held-out share (about
//     15 % on the traces that motivated it), never on the in-sample 44-60 %, so a table with only `share_in_sample`
//     is refused rather than gated on the wrong number.
// Header-only, no device. A table that passes is a proposal the caller may adopt; nothing here moves a tensor.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::moe {

inline constexpr const char* kPlacementSchema = "1bit/moe-placement/1";
// The floor the #1702 decision put on a consumer: held-out share on the motivating traces was 18.9-26.3 %.
inline constexpr double kPlacementMinHeldOut = 0.15;

// What the caller is about to run, so the table can be checked against it.
struct PlacementExpect {
    std::vector<int> layers;   // the routed (MoE) layer numbers, in any order
    int n_experts = 0;         // experts per routed layer
    int groups = 0;            // devices the experts are split across
    double min_held_out = kPlacementMinHeldOut;
};

struct PlacementLayer {
    std::vector<std::vector<int>> groups;  // groups[g] = the expert ids placed on device g
    std::vector<int> group_of;             // group_of[e] = g
    int unseen = -1;                       // experts placed by balance alone (-1: not recorded)
};

struct Placement {
    int groups = 0;
    double share_held_out = 0.0;
    double share_in_sample = -1.0;         // -1: not recorded; reported, never gated on
    long long held_out_steps = -1;
    std::map<int, PlacementLayer> layers;

    int group_of(int layer, int expert) const {
        auto it = layers.find(layer);
        if (it == layers.end() || expert < 0 || size_t(expert) >= it->second.group_of.size())
            throw std::out_of_range("moe placement: no entry for layer " + std::to_string(layer) + " expert " +
                                    std::to_string(expert));
        return it->second.group_of[size_t(expert)];
    }
};

namespace detail {
[[noreturn]] inline void refuse(const std::string& what) { throw std::runtime_error("moe placement: " + what); }
inline bool decimal_key(const std::string& s, int& out) {
    if (s.empty() || s.size() > 6) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    if (s.size() > 1 && s[0] == '0') return false;
    out = std::stoi(s);
    return true;
}
inline std::string pct(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f %%", v * 100.0);
    return b;
}
}  // namespace detail

inline Placement parse_placement(const nlohmann::json& t, const PlacementExpect& want) {
    using detail::refuse;
    if (want.n_experts <= 0 || want.groups <= 0 || want.layers.empty())
        refuse("the caller must name its routed layers, expert count and group count");
    if (!t.is_object()) refuse("the table is not a JSON object");
    const std::string schema = t.contains("schema") && t["schema"].is_string() ? t["schema"].get<std::string>() : "";
    if (schema != kPlacementSchema)
        refuse("schema \"" + schema + "\" is not \"" + kPlacementSchema + "\"");

    Placement p;
    if (!t.contains("groups") || !t["groups"].is_number_integer()) refuse("no integer \"groups\"");
    p.groups = t["groups"].get<int>();
    if (p.groups != want.groups)
        refuse("the table splits into " + std::to_string(p.groups) + " groups, this run has " +
               std::to_string(want.groups));

    if (t.contains("share_in_sample") && t["share_in_sample"].is_number())
        p.share_in_sample = t["share_in_sample"].get<double>();
    if (!t.contains("share_held_out") || !t["share_held_out"].is_number())
        refuse("no \"share_held_out\": the gate is the held-out share, not share_in_sample" +
               std::string(p.share_in_sample >= 0 ? " (" + detail::pct(p.share_in_sample) + ")" : ""));
    p.share_held_out = t["share_held_out"].get<double>();
    if (p.share_held_out < want.min_held_out)
        refuse("share_held_out " + detail::pct(p.share_held_out) + " is under the gate " +
               detail::pct(want.min_held_out));
    if (t.contains("held_out_steps") && t["held_out_steps"].is_number_integer())
        p.held_out_steps = t["held_out_steps"].get<long long>();

    if (!t.contains("layers") || !t["layers"].is_object()) refuse("no \"layers\" object");
    std::map<int, bool> wanted;
    for (int l : want.layers) wanted[l] = false;
    for (auto it = t["layers"].begin(); it != t["layers"].end(); ++it) {
        int layer = -1;
        if (!detail::decimal_key(it.key(), layer)) refuse("layer key \"" + it.key() + "\" is not a layer number");
        auto w = wanted.find(layer);
        if (w == wanted.end()) refuse("layer " + it.key() + " is not a routed layer of this model");
        w->second = true;
        const nlohmann::json& e = it.value();
        const std::string at = "layer " + it.key() + ": ";
        if (!e.is_object()) refuse(at + "not an object");
        if (e.contains("experts") && (!e["experts"].is_number_integer() || e["experts"].get<int>() != want.n_experts))
            refuse(at + "the table was built for " + e["experts"].dump() + " experts, this model has " +
                   std::to_string(want.n_experts));
        if (!e.contains("groups") || !e["groups"].is_array() || e["groups"].size() != size_t(want.groups))
            refuse(at + "needs " + std::to_string(want.groups) + " groups");
        PlacementLayer pl;
        pl.group_of.assign(size_t(want.n_experts), -1);
        for (size_t g = 0; g < e["groups"].size(); ++g) {
            const nlohmann::json& ids = e["groups"][g];
            if (!ids.is_array()) refuse(at + "group " + std::to_string(g) + " is not a list");
            std::vector<int> grp;
            for (const auto& v : ids) {
                if (!v.is_number_integer()) refuse(at + "group " + std::to_string(g) + " holds a non-integer id");
                const long long id = v.get<long long>();
                if (id < 0 || id >= want.n_experts)
                    refuse(at + "expert " + std::to_string(id) + " is outside 0.." + std::to_string(want.n_experts - 1));
                if (pl.group_of[size_t(id)] >= 0)
                    refuse(at + "expert " + std::to_string(id) + " is placed twice");
                pl.group_of[size_t(id)] = int(g);
                grp.push_back(int(id));
            }
            pl.groups.push_back(std::move(grp));
        }
        for (int id = 0; id < want.n_experts; ++id)
            if (pl.group_of[size_t(id)] < 0) refuse(at + "expert " + std::to_string(id) + " is not placed");
        if (e.contains("unseen") && e["unseen"].is_number_integer()) pl.unseen = e["unseen"].get<int>();
        p.layers[layer] = std::move(pl);
    }
    for (const auto& [layer, seen] : wanted)
        if (!seen) refuse("layer " + std::to_string(layer) + " is missing from the table");
    return p;
}

inline Placement load_placement(const std::string& path, const PlacementExpect& want) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("moe placement: cannot open " + path);
    nlohmann::json t;
    try {
        t = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("moe placement: " + path + " is not JSON (" + e.what() + ")");
    }
    try {
        return parse_placement(t, want);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string(e.what()) + " [" + path + "]");
    }
}

}  // namespace onebit::moe
