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
// Copyright 2026 The 1bit-MONSTER Authors.
// SPDX-License-Identifier: Apache-2.0
//
// warmup_max_tokens_test -- engine#2884: the warm-up bodies spoke llama.cpp's `n_predict` and the route's parser
// read only `max_completion_tokens`/`max_tokens`, so the short warm-up generated up to 256 tokens instead of 0
// and the spec warm-up 256 instead of 2. The 10.4 s host-copy build those requests exist to move to load time was
// paid by the first real client request instead.
//
// The arm has to run on the bodies themselves, not on a copy of them: a body whose every key is silently ignored
// is byte-for-byte indistinguishable from a correct one from the outside, which is how this survived a review of
// the route (#2855) and a check that the paths are registered. So `warmup_body`/`warmup_short_body`/
// `warmup_spec_body` live in app/warmup_bodies.h and this test reads the cap out of the same object the server
// posts, through the same function the route parses it with.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "npu_sampling.h"
#include "warmup_bodies.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) failures++;
}

}  // namespace

int main() {
    using namespace onebit;
    using json = nlohmann::json;

    // The three warm-up bodies, as the server builds them.
    check(npu::request_max_tokens(serve_warmup::warmup_short_body()) == 0,
          "the short warm-up asks for 0 tokens: its prompt is the work, and a 256-token decode is the bug (#2884)");
    check(npu::request_max_tokens(serve_warmup::warmup_spec_body()) == 2,
          "the spec warm-up asks for 2: the second token is the step that builds the one-token decode layout");
    check(npu::request_max_tokens(serve_warmup::warmup_body()) == 8,
          "the sampling chat warm-up keeps its own 8-token cap (the OpenAI field name)");

    // The precedence, on bodies a client can send. /completion is llama.cpp's endpoint, so n_predict is the field
    // its clients use; the OpenAI names still work for the chat dialect.
    check(npu::request_max_tokens(json{ { "n_predict", 5 } }) == 5, "n_predict is read");
    check(npu::request_max_tokens(json{ { "max_tokens", 7 } }) == 7, "max_tokens is still read");
    check(npu::request_max_tokens(json{ { "max_completion_tokens", 9 } }) == 9, "max_completion_tokens is still read");
    check(npu::request_max_tokens(json{ { "n_predict", 5 }, { "max_tokens", 7 } }) == 5,
          "when both are present the llama.cpp field wins: it is the one this endpoint's clients set");
    check(npu::request_max_tokens(json{ { "max_completion_tokens", 9 }, { "n_predict", 5 } }) == 9,
          "and the most specific field wins over both");
    check(npu::request_max_tokens(json{}) == 256, "no cap field keeps the route's default");
    check(npu::request_max_tokens(json{ { "n_predict", nullptr } }) == 256, "an explicit null is still absent");
    check(npu::request_max_tokens(json{ { "n_predict", 2.9 } }) == 2, "a fractional cap truncates, as it did before");

    // A wrong type must be the route's 400, not a json::type_error that escapes as a 500 (engine#1106).
    bool threw = false;
    try {
        npu::request_max_tokens(json{ { "n_predict", "2" } });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a non-numeric n_predict throws (the handler turns it into a 400)");

    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
