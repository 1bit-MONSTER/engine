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

std::vector<Question> routing_question() {
    Question q;
    q.type = "choice";
    q.instructions = "Which device should run this request?";
    q.criteria = {
        {"npu", "low-power AMD XDNA NPU, single context at a time"},
        {"hrx", "AMD HRX on the Radeon iGPU"},
        {"vulkan", "Vulkan on the Radeon iGPU"},
        {"zinc", "ZINC GPU (also reaches NVIDIA and Apple)"},
    };
    return {std::move(q)};
}

std::string route_device(Scorer& scorer, const std::string& state) {
    std::vector<Answer> answers;
    if (!scorer.score(state, routing_question(), answers)) return "";
    return answers.empty() ? "" : answers.front().choice;
}

}  // namespace onebit::laya
