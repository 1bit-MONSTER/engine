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

// npu/rt.h against the own NPU runtime (ONEBIT_NPU_RT_LIB or ONEBIT_NPU_RT_DIR): every call of the
// public interface npu/npu_rt_api.h appears below (the engine's NPU code, npu/lane.cpp and
// npu/forward, and the private routes' add-ons use these), so a runtime library that lacks one fails
// to link this test. Nothing here opens the device: `uses` is linked, never called.
#include "rt.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static void uses(const std::vector<char>& image) {
    xrt::device dev{0};
    auto ctx = std::make_unique<xrt::hw_context>(dev, xrt::elf(image.data(), image.size()));
    ctx->add_config(xrt::elf(image.data(), image.size()));
    xrt::elf from_file(std::string("kernel.elf"));
    if (from_file.is_full_elf()) std::puts(from_file.get_cfg_uuid().to_string().c_str());
    xrt::ext::kernel k(*ctx, std::string("kernel"));
    auto act = std::make_unique<xrt::ext::bo>(dev, size_t(4096));
    xrt::ext::bo e{dev, size_t(4096)};
    auto plain = std::make_unique<xrt::bo>(e);
    static_cast<char*>(act->map())[0] = 0;
    act->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    act->sync(XCL_BO_SYNC_BO_FROM_DEVICE, size_t(256), size_t(0));
    std::printf("%zu\n", plain->size());
    xrt::run r(k);
    r.set_arg(0, *act);
    r.start();
    if (r.wait(std::chrono::milliseconds(5000)) != ERT_CMD_STATE_COMPLETED) std::puts("not completed");
    std::vector<xrt::run> runs;
    runs.reserve(2);
    xrt::run& r2 = runs.emplace_back(k);
    r2.set_arg(0, *plain);
    auto rl = std::make_unique<xrt::runlist>(*ctx);
    rl->add(r2);
    rl->execute();
    rl->wait();
    rl->reset();
    xrt::run r3 = k(*act, *plain, e);
    r3.wait();
    if (r3.state() != ERT_CMD_STATE_COMPLETED) std::puts("not completed");
    xrt::bo row(e, size_t(256), size_t(512));  // a sub-buffer
    row.map<char*>()[0] = 0;
    xrt::elf none;
    xrt::hw_context no_ctx;
    xrt::ext::kernel no_kernel;
    xrt::run no_run;
    if (!none && !no_run && row) no_ctx = *ctx;
    no_kernel = k;
}

int main() {
    // Taking its address keeps `uses` in the binary, so every call in it must link.
    void (*volatile keep)(const std::vector<char>&) = uses;
    (void)keep;
    std::puts("npu/rt.h: own runtime behind npu/npu_rt_api.h");
    return 0;
}
