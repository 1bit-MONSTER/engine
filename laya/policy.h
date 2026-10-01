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

// laya/policy.h — request class -> device (RFC #186, docs/laya.md). The policy is data:
// config/route-policy.json, compiled in as the default and replaceable with --route-policy.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace onebit::laya {

struct RoutePolicy {
    float min_confidence = 0.0f;               // below it a request takes `fallback`
    std::string fallback = "hrx";              // the policy's "default" device
    std::map<std::string, std::string> device;  // class -> device

    // Parses policy JSON; on failure returns false and leaves the reason in `err`.
    bool parse(const std::string& json_text, std::string& err);

    // The device for a request of class `cls` with Laya confidence `confidence`, among
    // `candidates` (the devices that can run the model): the class's row, else the policy
    // default, else the first candidate. Never empty when `candidates` is not.
    std::string pick(const std::string& cls, float confidence, const std::vector<std::string>& candidates) const;
};

// config/route-policy.json as built into this binary.
const char* default_route_policy_json();

// Loads `path` when non-empty, else the built-in policy.
RoutePolicy load_route_policy(const std::string& path);

}  // namespace onebit::laya
