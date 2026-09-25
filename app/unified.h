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

// `1bit unified`: one native NPU model behind OpenAI endpoints (unified.cpp).
#pragma once

#include "private_route.h"

#include <string>

namespace onebit {

// argv holds the options after "unified".
int run_unified(int argc, char** argv);

// What runs an NPU model directory (model.q4nx, config.json, tokenizer.json):
//   Lane     a Qwen2/Qwen3 dense model with npu/ holding the fast lane's kernels (docs/npu.md)
//   Private  a model_type a private route registered (npu/private_route.h) and can serve with opts
enum class NpuModel { None, Lane, Private };
NpuModel npu_model_kind(const std::string& dir, const npu::PrivateOptions& opts = {});
bool is_npu_model_dir(const std::string& dir, const npu::PrivateOptions& opts = {});
// Why dir is not an NPU model directory here, for an error message; empty if it is one.
std::string npu_model_problem(const std::string& dir, const npu::PrivateOptions& opts = {});
std::string npu_kernel_dir(const std::string& model_dir);

}  // namespace onebit
