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

// engine#1862: `1bit npu-run --ids` on a pack whose kernel directory holds the SEALED set (SEALED-*/full_i8_*), which is
// every pack since that route shipped. Runs the ids through ONE multi-row pass of the same forward `forward-serve`
// drives -- the pass the k-row identity check needs -- and prints each row's argmax. `host_gemm` (NPU_INFER_HOST_GEMM)
// takes no device lock, exactly as forward-serve decides it. Returns the process exit code.
int run_forward_ids(const std::string& model_dir, const std::string& kernels, const std::vector<int>& ids,
                    int max_tokens, float penalty, bool host_gemm);

}  // namespace onebit
