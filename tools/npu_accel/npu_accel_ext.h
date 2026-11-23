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
// npu_accel_ext.h -- the hook a private NPU add-on uses to take matmuls in the ACCEL backend (ggml_npu_accel.cpp).
//
// The public backend runs weight matmuls through the int8 per-op ELF family. An extension -- linked in only when the
// engine is built with the private add-on (-DONEBIT_NPU_PRIVATE) -- can claim matmuls first, with its own kernels and
// weight layouts, which stay in the private repository. The backend lends it the device and its budgeted weight
// allocator, so ONEBIT_NPU_ACCEL_MAX_GB bounds the extension's buffers too.
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "rt.h"

struct ggml_tensor;

namespace onebit::npu_accel {

struct Host {
    xrt::device* dev = nullptr;
    // A weight buffer counted against ONEBIT_NPU_ACCEL_MAX_GB; aborts with a message past it.
    std::unique_ptr<xrt::ext::bo> (*weight_bo)(size_t bytes, const char* what) = nullptr;
    // Give back the host pages a packed weight came from (the CPU never reads them again).
    void (*release_pages)(const void* p, size_t n) = nullptr;
};

struct Ext {
    const char* name = nullptr;
    // Read the extension's configuration; false = it takes nothing (logged by the extension).
    bool (*configure)() = nullptr;
    // Called once the device is open.
    bool (*open)(const Host& h, std::string* why) = nullptr;
    // Whether it will take this weight's MUL_MAT (src0 2-D) / MUL_MAT_ID (src0 = expert stack): for supports_op.
    bool (*takes_mul_mat)(const ggml_tensor* src0) = nullptr;
    bool (*takes_mul_mat_id)(const ggml_tensor* as) = nullptr;
    // Run the whole op (every row / every token); false = not taken, the int8 path runs it.
    bool (*mul_mat)(const Host& h, ggml_tensor* dst) = nullptr;
    bool (*mul_mat_id)(const Host& h, ggml_tensor* dst) = nullptr;
    void (*stats)() = nullptr;
};

// At most one extension; registered from a static initializer in the add-on.
void register_ext(const Ext* e);

}  // namespace onebit::npu_accel
