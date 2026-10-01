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

#include "policy.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "route.h"
#include "route_policy_default.h"  // generated from config/route-policy.json

namespace onebit::laya {

const char* default_route_policy_json() { return kDefaultRoutePolicyJson; }

bool RoutePolicy::parse(const std::string& json_text, std::string& err) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (const std::exception& e) {
        err = std::string("route policy: ") + e.what();
        return false;
    }
    if (!j.is_object() || !j.contains("classes") || !j["classes"].is_object()) {
        err = "route policy: needs a \"classes\" object";
        return false;
    }
    min_confidence = j.value("min_confidence", 0.0f);
    fallback = j.value("default", std::string("hrx"));
    device.clear();
    const auto& known = request_classes();
    for (const auto& [cls, row] : j["classes"].items()) {
        if (std::find(known.begin(), known.end(), cls) == known.end()) {
            err = "route policy: unknown class \"" + cls + "\"";
            return false;
        }
        if (!row.is_object() || !row.contains("device") || !row["device"].is_string()) {
            err = "route policy: class \"" + cls + "\" needs a \"device\"";
            return false;
        }
        device[cls] = row["device"].get<std::string>();
    }
    return true;
}

std::string RoutePolicy::pick(const std::string& cls, float confidence, const std::vector<std::string>& candidates) const {
    auto have = [&](const std::string& d) { return std::find(candidates.begin(), candidates.end(), d) != candidates.end(); };
    if (confidence >= min_confidence)
        if (const auto it = device.find(cls); it != device.end() && have(it->second)) return it->second;
    if (have(fallback)) return fallback;
    return candidates.empty() ? std::string() : candidates.front();
}

RoutePolicy load_route_policy(const std::string& path) {
    std::string text = default_route_policy_json();
    if (!path.empty()) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("route policy: cannot read " + path);
        std::stringstream ss;
        ss << in.rdbuf();
        text = ss.str();
    }
    RoutePolicy p;
    std::string err;
    if (!p.parse(text, err)) throw std::runtime_error(err + (path.empty() ? " (built-in)" : " in " + path));
    return p;
}

}  // namespace onebit::laya
