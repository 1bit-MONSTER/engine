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

#include "route.h"

#include "scorer.h"

namespace onebit::laya {

namespace {

// One-line descriptions for the devices the engine serves, keyed by name.
const std::vector<std::pair<std::string, std::string>>& device_descriptions() {
    static const std::vector<std::pair<std::string, std::string>> d = {
        {"npu", "low-power AMD XDNA NPU, single context at a time"},
        {"hrx", "AMD HRX on the Radeon iGPU"},
        {"vulkan", "Vulkan on the Radeon iGPU"},
        {"zinc", "ZINC GPU (also reaches NVIDIA and Apple)"},
    };
    return d;
}

}  // namespace

std::vector<Question> routing_question(const std::vector<std::string>& devices) {
    Question q;
    q.type = "choice";
    q.instructions = "Which device should run this request?";
    for (const std::string& dev : devices)
        for (const auto& [name, desc] : device_descriptions())
            if (name == dev) q.criteria.emplace_back(name, desc);
    return {std::move(q)};
}

std::string route_device(Scorer& scorer, const std::string& state, const std::vector<std::string>& devices) {
    std::vector<Answer> answers;
    if (!scorer.score(state, routing_question(devices), answers)) return "";
    return answers.empty() ? "" : answers.front().choice;
}

}  // namespace onebit::laya
