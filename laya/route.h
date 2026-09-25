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

// laya/route.h — the step-4 router: maps a request to a device via the Laya
// scorer (docs/laya.md). One forward pass answers the fixed routing question
// ("which device should run this request?") and the winning option is the
// device: npu, hrx, vulkan or zinc.
#pragma once

#include <string>
#include <vector>

namespace onebit::laya {

class Scorer;
struct Question;

// The fixed routing question. The criteria keys are the device names, so the
// argmax option IS the device.
std::vector<Question> routing_question();

// Asks the scorer which device should run `state` (free text or a serialized
// request) and returns the winning device name. Returns an empty string when
// the scorer fails; the scorer's error is left on the passed-in scorer.
std::string route_device(Scorer& scorer, const std::string& state);

}  // namespace onebit::laya
