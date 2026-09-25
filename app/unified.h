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

#include <string>

namespace onebit {

// argv holds the options after "unified".
int run_unified(int argc, char** argv);

// What runs an NPU model directory (model.q4nx, config.json, tokenizer.json):
//   Lane  a Qwen2/Qwen3 dense model with npu/ holding the fast lane's kernels (docs/npu.md)
//   Lax   Qwen3.6-35B-A3B (model_type qwen3_5_moe) on the lax kernels (docs/npu-lax.md),
//         found at lax_kernels, else <dir>/npu/lax, else $ONEBIT_NPU_LAX_KERNELS
enum class NpuModel { None, Lane, Lax };
NpuModel npu_model_kind(const std::string& dir, const std::string& lax_kernels = "");
bool is_npu_model_dir(const std::string& dir, const std::string& lax_kernels = "");
std::string npu_kernel_dir(const std::string& model_dir);
// The lax kernel directory for dir, as npu_model_kind looks it up; empty if none.
std::string lax_kernel_dir(const std::string& dir, const std::string& lax_kernels = "");

}  // namespace onebit
