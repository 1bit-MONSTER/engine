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

// `1bit npu-lax`: Qwen3.6-35B-A3B on the NPU's lax kernels (docs/npu-lax.md), greedy chat
// or the three-position parity check against the fp64 reference.
#pragma once

namespace onebit {

int run_npu_lax(int argc, char** argv);

}  // namespace onebit
