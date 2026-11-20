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
// The warm-up request bodies, apart from the code that sends them, so a host test can read the cap out of
// exactly the body the server posts (tests/warmup_max_tokens_test.cpp, engine#2884).
//
// Why they moved: the bodies spoke llama.cpp's `n_predict` and the route's parser read only
// `max_completion_tokens`/`max_tokens`, so the short warm-up ran up to 256 tokens instead of 0 and the spec
// warm-up 256 instead of 2 -- the 10.4 s host-copy build those requests exist to move to load time was paid
// by the first client request instead. A body whose key nothing reads is indistinguishable from a correct one
// from the outside, so the assertion has to be on the body and it has to run on the body itself.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace onebit::serve_warmup {

using json = nlohmann::json;

inline json warmup_body() {
    std::string text = "warm up the decode and prefill shapes";
    for (int i = 0; i < 60; ++i) text += " lorem ipsum dolor sit amet";
    return json{ { "messages", json::array({ { { "role", "user" }, { "content", text } } }) },
                 { "temperature", 0 }, { "max_tokens", 8 }, { "cache_prompt", false } };
}

// The second warm-up request: one 16-token prompt, " a" 16 times (one token each in the byte-level BPE
// vocabularies of the models we serve; a BOS makes it 17, still inside the range below). n_predict 0: the
// prompt is the work. (Token ids instead of text ended in a 500 on Qwen3.5-4B: id 100 is a lone UTF-8
// byte there, which llama-server's output parser rejects.)
// The first request whose batch holds 13 to 20 tokens takes HRX's small-batch matmul kernels, and they
// read the weights in a layout ggml-hrx builds on the host the first time it needs it (re-quantizing the
// affected tensors: 6.1 s of CPU, +1.0 GiB of GTT on Qwen3.8-27B UD-Q4_K_XL). A chat request splits its
// prompt at the last user message and before its last 4 tokens, so short chats land there: Lemonade's
// chat-short prompt (67 tokens) runs as 50 + 13 + 4, and its first time after a load cost 2.43 s against
// 0.53 s later, with lemond on 4 cores. This request moves that build to load time (2.1 s on 4 cores).
// On a fresh server, 13-20 tokens cost 2.03-2.07 s the first time; 1-12 cost 0.1-0.5 s and 21-512 cost
// 0.45-1.23 s (docs/serve.md, "Warm-up").
inline constexpr int kWarmupShortTokens = 16;

inline json warmup_short_body() {
    std::string prompt;
    for (int i = 0; i < kWarmupShortTokens; ++i) prompt += " a";
    return json{ { "prompt", prompt }, { "n_predict", 0 },
                 { "temperature", 0 }, { "cache_prompt", false } };
}

// The third warm-up request, sent only when the backend speculates (--mtp, an own MTP head, --dflash): the same
// 16-token prompt, n_predict 2. A speculating server decodes the LAST token of a request without a draft
// (llama-server's get_n_draft_max() is n_remaining - 1 = 0 there), so that step runs the plain one-token decode
// program, and the first time it runs, ggml-hrx builds the layout that program reads for any weight the verify pass
// reads in another one. On Qwen3.8-27B Q4_K_M that is the mixed layers' Q6_K ffn weights: 76 host copies, ~5.6 GB,
// 10.4 s, paid by whichever client request first ends on such a step. With n_predict 2 the second token is that
// step, so the build happens here instead (DECODE-BW 11-15: the duplicates appeared during this request, and the
// next perf.py pass decoded 23.55 tok/s instead of absorbing the 10 s).
inline json warmup_spec_body() {
    json body = warmup_short_body();
    body["n_predict"] = 2;
    return body;
}

}  // namespace onebit::serve_warmup
