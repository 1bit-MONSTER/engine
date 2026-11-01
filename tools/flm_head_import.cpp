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
//
// flm-head-import <container.q4nx> -o <out dir>: FastFlowLM's MTP-head container -> our Q4NX sidecar.
// The implementation is npu/flm_head_import.cpp; it is a separate executable rather than a `1bit` subcommand
// because the `1bit` binary does not link onebit_npu (docs/npu.md, "The MTP head").
#include "flm_head_import.h"

int main(int argc, char** argv) {
    return onebit::npu::flm_head::run(argc - 1, argv + 1);  // drop argv[0]
}
