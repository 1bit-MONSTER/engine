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

// engine#1786: Q4nxNpuForward::embed_pooled() on a container named by ONEBIT_TEST_Q4NX, host GEMM, no device.
// tests/embed_forward_ci.py writes the container (a bidirectional gemma3 encoder in FastFlowLM's
// Embedding-Gemma-300M-NPU2 layout) and holds the vector this writes to ONEBIT_TEST_EMBED_OUT to a numpy reference.
// ONEBIT_TEST_IDS is the space-separated input. Exit 0 on a vector, 3 when the forward refused (its reason on
// stdout, so the script can check a refusal it asked for), 77 when the container is not set.
#include "q4nx_forward.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    const char* ids_env = std::getenv("ONEBIT_TEST_IDS");
    const char* out_path = std::getenv("ONEBIT_TEST_EMBED_OUT");
    if (!dir || !*dir || !ids_env || !out_path) {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX, ONEBIT_TEST_IDS and ONEBIT_TEST_EMBED_OUT\n");
        return 77;
    }
    setenv("NPU_INFER_HOST_GEMM", "1", 1);
    const std::string model = std::string(dir) + "/model.q4nx";
    Q4nxNpuForward fwd;
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("FAIL: init_host: %s\n", fwd.last_error());
        return 1;
    }
    std::vector<int> ids;
    std::istringstream is(ids_env);
    for (int t; is >> t;) ids.push_back(t);
    std::vector<float> v;
    if (!fwd.embed_pooled(ids, v)) {
        std::printf("REFUSED: %s\n", fwd.last_error());
        return 3;
    }
    FILE* f = std::fopen(out_path, "wb");
    if (!f || std::fwrite(v.data(), sizeof(float), v.size(), f) != v.size()) {
        std::printf("FAIL: cannot write %s\n", out_path);
        return 1;
    }
    std::fclose(f);
    std::printf("embedded %zu ids -> %zu floats\n", ids.size(), v.size());
    return 0;
}
