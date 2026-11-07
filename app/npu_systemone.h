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

// The HTTP half of the d3 decision route (engine #653): a request body into the questions
// `npu/systemone.h` renders, and per-option probabilities back out as the answer object.
//
// The shapes here are copied from the d3 server vLLM Semantic Router already speaks to
// (`vllm-sr/d3-lite`'s `d3_server.py` at revision 2fc06364…), not invented, because that envelope is
// the interop contract:
//
//   POST /v1/systemone            {"questions": {id: question, ...}, "state": ...}
//     -> 200 {"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens": 0}}
//     -> 422 {"detail": "<why the body cannot be read>"}                       nothing ran
//     -> 422 {"detail": {"invalid_questions": {id: "<why>"}}}                   nothing ran
//   a question that is valid but cannot be answered (over the input limit, or the readout could not be
//   read at all) comes back INSIDE `answers` as {"type", "error", "message"} with error one of
//   `max_length_exceeded` / `invalid_model_output`, and the other questions are still answered.
//
// `questions` is an object, not a list: the ids are the caller's and are echoed back as the keys of
// `answers`. Everything here is device-free and model-free (it needs the pack's `codes` only to
// render the prompt), which is what makes it testable without a backbone -- tests/systemone_route_test.cpp.
#pragma once

#include "../npu/systemone.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu {

// The text path's input limit, per question (issue #653): 32,768 tokens, and a longer prompt is
// refused with `max_length_exceeded` rather than truncated. The endpoint uses the smaller of this and
// the backbone's own context, because either one overflowing is the same failure to the model.
inline constexpr size_t kSystemOneMaxInputTokens = 32768;

// `description` of the served model in GET /v1/systemone/models: upstream's own wording.
inline constexpr const char* kSystemOneDescription = "Typed decisions with a probability for every option";

// A refusal of the WHOLE request: the caller answers 422 {"detail": what()} and runs nothing. A
// malformed question is not this -- those are collected per id, so one reply can name every bad
// question id at once, as upstream does.
struct SystemOneRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct SystemOneQuestion {
    std::string id;      // the caller's question id, echoed back as the key of `answers`
    d3::Question question;
};

struct SystemOneParsed {
    nlohmann::ordered_json state;              // ordered on purpose: upstream renders the request's key order
    std::vector<SystemOneQuestion> questions;  // in request order, valid questions only
    std::vector<std::pair<std::string, std::string>> invalid;  // id -> why, in request order
};

// A request body into questions. Throws SystemOneRefusal for the 422 whole-request cases (not an
// object, a field the route does not serve, `questions` missing/empty/not an object, an empty id).
// `model` is accepted and ignored -- the reply names the served model, as upstream's does.
inline SystemOneParsed systemone_request_of(const nlohmann::ordered_json& body) {
    if (!body.is_object())
        throw SystemOneRefusal("expected a JSON object such as {\"questions\": {\"route\": {\"type\": \"choice\", ...}}}");
    for (const auto& [field, value] : body.items()) {
        (void)value;
        if (field == "state" || field == "questions" || field == "model") continue;
        // The vision path is a separate issue (#653 says so): refused by name rather than ignored,
        // because a caller who sent images would otherwise read a text-only answer as a visual one.
        if (field == "images" || field == "videos")
            throw SystemOneRefusal(field + ": this build serves the text path only (choice, noul and score)");
        throw SystemOneRefusal(field + " not understood; a request may carry state, questions, model");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty())
        throw SystemOneRefusal("questions is required: an object of one or more question IDs, each to a question");
    SystemOneParsed out;
    out.state = body.contains("state") ? body.at("state") : nlohmann::ordered_json(nullptr);
    for (const auto& [id, q] : body.at("questions").items()) {
        if (id.empty()) throw SystemOneRefusal("question IDs must be nonempty strings");
        try {
            out.questions.push_back(SystemOneQuestion{id, d3::question_of(q)});
        } catch (const std::exception& ex) {
            out.invalid.emplace_back(id, ex.what());
        }
    }
    return out;
}

// The prompt text of one question: the pack's chat template with `enable_thinking=False`
// (`d3_format._chat_text`), which is system turn, user turn, then the assistant turn opened with the
// EMPTY think block. Those trailing bytes are part of what the model reads, so this is not
// forward-serve's chatml(); swapping one for the other changes the model's input.
// Jinja's `|trim` on both turns is a no-op here: kSystemPrompt and user_prompt() are already trimmed.
inline std::string systemone_chat(const std::string& user) {
    return std::string("<|im_start|>system\n") + d3::kSystemPrompt + "<|im_end|>\n" + "<|im_start|>user\n" +
           user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

inline std::string systemone_prompt(const nlohmann::ordered_json& state, const d3::Question& q,
                                    const std::vector<std::string>& codes) {
    return systemone_chat(d3::user_prompt(state, q, codes));
}

// A question that could not be answered, in the shape upstream puts inside `answers`.
inline nlohmann::json systemone_question_error(const std::string& kind, const std::string& error,
                                               const std::string& message) {
    return nlohmann::json{{"type", kind.empty() ? nlohmann::json(nullptr) : nlohmann::json(kind)},
                          {"error", error},
                          {"message", message}};
}

inline std::string systemone_length_message(size_t tokens, size_t limit) {
    return "the question prompt has " + std::to_string(tokens) + " tokens, over the maximum context length of " +
           std::to_string(limit) + " tokens; nothing was truncated";
}

// {"model", "answers", "usage"} -- the reply the d3 server returns. `output_tokens` is 0 because a
// decision generates no text; `input_tokens` is the sum of every answered question's prompt length.
inline nlohmann::ordered_json systemone_reply(const std::string& model, const nlohmann::ordered_json& answers,
                                              size_t input_tokens) {
    return nlohmann::ordered_json{
        {"model", model},
        {"answers", answers},
        {"usage", nlohmann::ordered_json{{"input_tokens", input_tokens}, {"output_tokens", 0}}}};
}

// GET /v1/systemone/models: the decision models this build serves. `modalities` is ["text"] because
// box/track/assets need the vision tower, which this route does not carry.
inline nlohmann::ordered_json systemone_models(const std::string& name, size_t max_input_tokens) {
    return nlohmann::ordered_json{{"models", nlohmann::ordered_json::array({nlohmann::ordered_json{
                                       {"name", name},
                                       {"description", kSystemOneDescription},
                                       {"max_input_tokens", max_input_tokens},
                                       {"modalities", nlohmann::ordered_json::array({"text"})}}})}};
}

}  // namespace onebit::npu
