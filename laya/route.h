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
// device. The candidate devices are passed in, so a .gguf routes among the GPU
// backends (vulkan, hrx, zinc) while an NPU model directory routes to npu.
#pragma once

#include <string>
#include <vector>

namespace onebit::laya {

class Scorer;
struct Question;

// The fixed routing question restricted to the given device names, in that
// order. The criteria keys are the device names, so the argmax option IS the
// device. Recognized names: npu, hrx, vulkan, zinc.
std::vector<Question> routing_question(const std::vector<std::string>& devices);

// Asks the scorer which of `devices` should run `state` (free text or a
// serialized request) and returns the winning device name. Returns an empty
// string when the scorer fails; the scorer's error is left on the passed-in
// scorer.
std::string route_device(Scorer& scorer, const std::string& state, const std::vector<std::string>& devices);

// ── Request classes (RFC #186) ──────────────────────────────────────────────
// Laya knows nothing about the hardware, so asking it for a device does not work (it picked
// zinc for every request). It classifies the request instead, and a measured policy
// (config/route-policy.json) maps the class to a device and settings.

// Requests of at least this many characters are long_doc without asking the model (pasted
// documents, logs, tables, files: about 256 tokens and up).
constexpr size_t kLongDocChars = 1024;

// The class keys, in the order of the class question's options.
const std::vector<std::string>& request_classes();

// The request-class question. `variant` selects a wording (0 = the one serve uses; the
// others are kept for tests/laya_classify_eval, ONEBIT_LAYA_CLASS_VARIANT).
Question request_class_question(int variant = 0);

struct RequestClass {
    std::string label;        // one of request_classes(), empty when the scorer failed
    float confidence = 0.0f;  // Laya's calibrated confidence, 1 - H(p)/log(k)
    std::vector<std::pair<std::string, float>> probabilities;
};

// Classifies `state` (the request text). The wording comes from ONEBIT_LAYA_CLASS_VARIANT
// when set, else variant 0.
RequestClass classify_request(Scorer& scorer, const std::string& state);

}  // namespace onebit::laya
