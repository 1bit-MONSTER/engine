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

#include <algorithm>
#include <cstdlib>

#include "scorer.h"

namespace onebit::laya {

namespace {

// One-line descriptions for the devices the engine serves, keyed by name.
const std::vector<std::pair<std::string, std::string>>& device_descriptions() {
    static const std::vector<std::pair<std::string, std::string>> d = {
        {"npu", "low-power AMD XDNA NPU, single context at a time"},
        {"hrx", "AMD HRX on the Radeon iGPU"},
        {"cpu", "the CPU, no GPU layers"},
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

const std::vector<std::string>& request_classes() {
    static const std::vector<std::string> k = {"code", "prose", "short", "long_doc"};
    return k;
}

Question request_class_question(int variant) {
    Question q;
    q.type = "choice";
    switch (variant) {
    case 1:
        q.instructions = "What kind of request is this?";
        q.criteria = {{"code", "writing, fixing or explaining program code"},
                      {"prose", "an explanation, essay, story, plan or advice that needs a longer written answer"},
                      {"short", "a greeting, a quick fact, a calculation or a one-line answer"},
                      {"long_doc", "work over a long pasted document, log, table, transcript or file"}};
        break;
    case 2:
        q.instructions = "Classify this chat request.";
        q.criteria = {{"code", "programming"},
                      {"prose", "long-form writing or explanation"},
                      {"short", "a quick reply: greeting, fact, calculation, translation of a phrase"},
                      {"long_doc", "work on a long document the user pasted"}};
        break;
    default:
        q.instructions = "What kind of answer does this request need?";
        q.criteria = {{"code", "source code, a script, a fix to a program, or an explanation of code"},
                      {"prose", "several paragraphs of explanation, writing, advice or a story"},
                      {"short", "one word, one number or one sentence"},
                      {"long_doc", "reading a long pasted document, log, transcript, table or file"}};
    }
    return q;
}

RequestClass classify_request(Scorer& scorer, const std::string& state) {
    int variant = 0;
    if (const char* v = std::getenv("ONEBIT_LAYA_CLASS_VARIANT")) variant = std::atoi(v);
    RequestClass out;
    // Long pasted material is a matter of size, which needs no model: Laya read long documents
    // as code or prose 30-56% of the time (tests/laya_classify_eval), and they are its slowest
    // decisions (a full 512-token window). Below the gate Laya picks among the other classes.
    if (state.size() >= kLongDocChars) {
        out.label = "long_doc";
        out.confidence = 1.0f;
        out.probabilities = {{"long_doc", 1.0f}};
        return out;
    }
    Question q = request_class_question(variant);
    q.criteria.erase(std::remove_if(q.criteria.begin(), q.criteria.end(),
                                    [](const auto& c) { return c.first == "long_doc"; }),
                     q.criteria.end());
    std::vector<Answer> answers;
    if (!scorer.score(state, {q}, answers) || answers.empty()) return out;
    out.label = answers.front().choice;
    out.confidence = answers.front().confidence;
    out.probabilities = answers.front().probabilities;
    return out;
}

}  // namespace onebit::laya
