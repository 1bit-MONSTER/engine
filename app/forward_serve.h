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

// The GGUF-on-NPU serve route (docs/npu.md, "Open"): `1bit forward-serve -m
// <model.q4nx> --kernels <dir> -p <port>` serves a repacked Q4NX model through
// the model-generic forward (npu/forward/q4nx_forward.cpp), behind the same
// OpenAI endpoints as `1bit unified`.
#pragma once

#include <string>
#include <vector>

namespace onebit {

int run_forward_serve(int argc, char** argv);

// engine#2111: the `usage` object a chat reply carries, as a pure function of the counters a call produced.
//
// It lived as a lambda inside the reply builder, which meant the one thing the issue is about - *is
// completion_tokens_details.reasoning_tokens present, and when is it absent?* - could only be observed by
// running a model on the NPU. As a function it is a decision table with known answers (tests/serve_usage_test.cpp),
// and the rule is the same one this fleet applies everywhere: a field that says "the budget went into reasoning"
// must be distinguishable from a field that says nothing, and `-1` ("the route could not count it") is neither.
struct UsageCounters {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int cached_tokens = 0;      // prefix cache: how many prompt tokens came from a saved prefix
    int prefix_slot = -1;       // -1 = "off" (no prefix slots configured)
    std::string prefix_reason;  // why the cache decided what it decided
    bool prefix_pinned = false;
    int reasoning_tokens = -1;  // -1 = the route could not count it; 0 = counted, none; >0 = that many
};

// Declared here, defined in forward_serve.cpp beside the reply it belongs to (nlohmann/json stays out of this
// header, so a test that only wants the decision table does not take the JSON dependency).
std::string usage_json_text(const UsageCounters& c);



// engine#1862: `1bit npu-run --ids` on a pack whose kernel directory holds the SEALED set (SEALED-*/full_i8_*), which is
// every pack since that route shipped. Runs the ids through ONE multi-row pass of the same forward `forward-serve`
// drives -- the pass the k-row identity check needs -- and prints each row's argmax. `host_gemm` (NPU_INFER_HOST_GEMM)
// takes no device lock, exactly as forward-serve decides it. Returns the process exit code.
int run_forward_ids(const std::string& model_dir, const std::string& kernels, const std::vector<int>& ids,
                    int max_tokens, float penalty, bool host_gemm);

}  // namespace onebit
