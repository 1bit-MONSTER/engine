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
// llama_npu_server.cpp -- llama.cpp's server with the NPU ACCEL backend registered (ggml_npu_accel.cpp): the model runs
// on the CPU backend and its weight matmuls run on the NPU. Every argument is llama-server's own.
#include <cstdio>

#include "ggml_npu_accel.h"

int llama_server(int argc, char** argv);

int main(int argc, char** argv) {
    if (!ggml_npu_accel_register()) std::fprintf(stderr, "llama-npu-server: NPU backend not registered, CPU only\n");
    return llama_server(argc, argv);
}
