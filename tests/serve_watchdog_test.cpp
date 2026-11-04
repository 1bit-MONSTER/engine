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

// serve_watchdog_test: the two "the device is gone" classifiers of app/serve_watchdog.h, which decide
// that a backend is killed, or a serve process leaves and is reloaded:
//   - device_lost_line: the backend's own stderr (the HRX/llama-server path, engine TSVCHRX),
//   - npu_run_failed: the NPU runtime's failed-runlist exception (the in-process `1bit unified` path,
//     measured on the c22 board 2026-11-04: code 614 + ERT state 8, and the same process then answered
//     every later request with 614 while a fresh process answered 200).
// Device-free and model-free: the inputs are the messages those two paths actually produce.
#include "serve_watchdog.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void expect_npu(bool want, const std::string& what) {
    const bool got = onebit::npu_run_failed(what);
    if (got != want) {
        ++failures;
        std::printf("FAIL npu_run_failed(%s) -> %d, want %d\n", what.c_str(), int(got), int(want));
    }
}

static void expect_line(bool want, const std::string& line) {
    const bool got = onebit::device_lost_line(line);
    if (got != want) {
        ++failures;
        std::printf("FAIL device_lost_line(%s) -> %d, want %d\n", line.c_str(), int(got), int(want));
    }
}

int main() {
    // The c22 board's message, verbatim from the SSE error frame of the failing chat request.
    expect_npu(true, "npu runtime: invalid call (code 614): state 8, run 0 of 28");
    // The match's second arm: the ERT state named in words (should the runtime spell it that way), under the
    // same prefix.
    expect_npu(true, "npu runtime: ERT_CMD_STATE_TIMEOUT, run 0 of 28");
    // A bare state, a bare code, or the string without the runtime's prefix is not this signal: the
    // classifier must not fire on a message that merely mentions a number.
    expect_npu(false, "invalid call (code 614): state 8, run 0 of 28");
    expect_npu(false, "npu runtime: state 8");
    expect_npu(false, "npu runtime: invalid call (code 500)");
    expect_npu(false, "token id out of range");
    expect_npu(false, "the NPU is busy with another request");
    expect_npu(false, "");

    // The backend-stderr classifier, unchanged, for contrast: a device-lost line from a GPU backend.
    expect_line(true, "ggml_hrx: DATA_LOSS; AMDGPU hardware exception");
    expect_line(true, "ErrorDeviceLost");
    expect_line(true, "vk::Device::waitForFences: ErrorDeviceLost");
    expect_line(false, "npu runtime: invalid call (code 614): state 8, run 0 of 28");
    expect_line(false, "");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("serve_watchdog: ok\n");
    return 0;
}
