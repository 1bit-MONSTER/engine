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

// npu_prompt_ids_test: `/v1/completions`' prompt may be text or the caller's exact token ids
// (app/npu_prompt_ids.h). No model, no device, no network.
//
// The assertions are the acceptance rules of the parser, including the ones that exist because the
// embedding lookup indexes by id: a non-integer element, an id outside the vocabulary and an empty
// array are all refused with a message that names the offending position, never clamped or dropped.
#include "npu_prompt_ids.h"

#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;
using onebit::npu::prompt_ids_from_body;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

// The parser's refusal message, or "" when it did not throw.
static std::string refusal(const json& body, int vocab) {
    std::vector<int> out;
    try {
        (void)prompt_ids_from_body(body, vocab, out);
    } catch (const std::exception& ex) {
        return ex.what();
    }
    return "";
}

static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

int main() {
    const int vocab = 151936;  // Zyphra/ZR1-1.5B's size, the model this was measured on

    // A string keeps the route exactly as it was: the caller's text, tokenized by the container.
    std::vector<int> out;
    check(prompt_ids_from_body(json{{"prompt", "The capital of France is"}}, vocab, out) == false,
          "a string prompt is the caller's text, not ids");
    check(out.empty(), "a string prompt leaves the id vector untouched");

    // An id array is used literally, in order, including the ids the vocabulary only reaches as
    // special tokens (BOS 151646 here) -- that is the whole point for a parity run.
    out.clear();
    const json ids = json{{"prompt", {151646, 785, 6722, 315, 9625, 374}}};
    check(prompt_ids_from_body(ids, vocab, out) == true, "an id array is taken as ids");
    check(out == std::vector<int>({151646, 785, 6722, 315, 9625, 374}), "the ids are kept in order, unchanged");

    // One id is a legal prompt: the empty-prompt refusal belongs to the string path (ids.empty() after
    // encoding), and an array of one is not that case.
    out.clear();
    check(prompt_ids_from_body(json{{"prompt", {151646}}}, vocab, out) == true && out.size() == 1,
          "a one-id prompt is accepted");

    // Refusals: each names what is wrong, because a silently changed prompt is an unexplained answer.
    check(has(refusal(json{{"max_tokens", 8}}, vocab), "prompt: missing"), "a missing prompt is named");
    check(has(refusal(json{{"prompt", json::array()}}, vocab), "empty"), "an empty id array is refused, not run");
    check(refusal(json{{"prompt", "hi"}, {"max_tokens", 1}}, -1).empty(),
          "vocab <= 0 disables the range check (the caller has no vocabulary size to give)");
    check(has(refusal(json{{"prompt", {1, "2"}}}, vocab), "prompt[1]"), "a non-integer element names its index");
    check(has(refusal(json{{"prompt", {1, 2.5}}}, vocab), "prompt[1]"), "a fractional element is not truncated");
    check(has(refusal(json{{"prompt", {1, true}}}, vocab), "prompt[1]"), "a boolean is not an id");
    check(has(refusal(json{{"prompt", {1, -1}}}, vocab), "outside the vocabulary"), "a negative id is refused");
    check(has(refusal(json{{"prompt", {1, vocab}}}, vocab), "outside the vocabulary"), "id == vocab is refused");
    check(refusal(json{{"prompt", {1, 151935}}}, vocab).empty(), "the last id in the vocabulary is accepted");
    check(has(refusal(json{{"prompt", {{"a", 1}}}}, vocab), "string or an array"), "an object prompt is refused");

    // Out-of-range ids are a request error, not a clamp: the embedding lookup would index by them.
    check(refusal(json{{"prompt", {1500000000}}}, vocab).find("1500000000") != std::string::npos,
          "the message quotes the offending id");

    if (failures == 0) std::printf("npu_prompt_ids: ok\n");
    return failures == 0 ? 0 : 1;
}
