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

// npu/rt.h against the selected runtime (-DONEBIT_NPU_RUNTIME=xrt|own): every runtime call the
// engine's NPU code makes (npu/lane.cpp, npu/forward) appears below, so a runtime that lacks one
// fails to build this test. Nothing here opens the device: `uses` is compiled, never called.
#include "rt.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

[[maybe_unused]] static void uses(const std::vector<char>& image) {
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
    xrt::run r3 = k(*act, *plain, e);
    r3.wait();
}

int main() {
#if defined(ONEBIT_NPU_RUNTIME_OWN)
    std::puts("npu/rt.h: own runtime (ONEBIT_NPU_RT_DIR)");
#else
    std::puts("npu/rt.h: XRT");
#endif
    return 0;
}
