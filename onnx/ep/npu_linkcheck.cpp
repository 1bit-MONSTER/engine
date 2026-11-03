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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The root CMakeLists' `onebit_ep_npu_linkcheck` runs this: it exists so the CI `build` job compiles
// AND links the ONNX EP's NPU leg, which no other configuration in the tree builds without ONNX
// Runtime and the private runtime (engine #874 -- a missing declaration, then a missing source, both
// green for a week). The call is device-free on purpose: `has_kernel` answers from the build's shape
// list, so the check never opens /dev/accel and never takes the device lock. What it proves is the
// link, not the device.

#include "onebit_npu_gemm.h"

#include <cstdio>

int main() {
    const bool shape = onebit::npu::has_kernel(1024, 1024);
    std::printf(
        "PASS onebit_ep_npu_linkcheck: npu/ort/onebit_npu_gemm.cpp compiles and links (has_kernel("
        "1024,1024)=%d)\n",
        shape ? 1 : 0);
    return 0;
}
