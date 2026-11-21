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
// xdna-driver#3: a SIGKILLed client does not finish teardown before its process is gone (an in-flight job holds the
// process's mm and drops it asynchronously), so a create in that window is refused with -ENODEV
// (`aie2_alloc_resource: Allocate AIE resource failed, ret -19`, runtime code 102). The driver issue measured 140-220 ms
// and asked, as its optional item, to "measure the kill-to-free time before and after". npu_rt retries that window for
// 500 ms (npu_rt_retry.h, npu-kernels#109); this probe is how that number is measured from userspace instead of inferred.
//
//   ctx_create_probe hold <full.elf>              create a context, say when it holds it, then sleep until killed
//   ctx_create_probe wait <full.elf> [--budget-ms N] [--interval-ms M]
//                                                 create contexts until one succeeds; print the wait, the attempts and
//                                                 every errno seen; rc 0 when it succeeded inside the budget
//
// `hold` is the victim: SIGKILL it, then run `wait` immediately. The wait's number is the kill-to-free time. Both modes go
// through `xrt::hw_context`, i.e. through npu_rt's own create path, so a retry the runtime made shows up on stderr as
// `hardware context refused (…); retry N in 50 ms` *before* the wait's own success line -- which is the difference between
// "the driver freed it quickly" and "the retry absorbed it".
//
// On a box with no `temporal_only=0` second client the create itself may be refused for another reason; that refusal is
// printed with its errno and the mode exits 1 rather than pretending the wait was measured.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <string>
#include <thread>

#include "npu_rt_api.h"

namespace xrt = npu_rt_api;

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s hold|wait <full.elf> [--budget-ms N] [--interval-ms M]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1], elf_path = argv[2];
    int budget_ms = 3000, interval_ms = 2;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--budget-ms")) budget_ms = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--interval-ms")) interval_ms = std::atoi(argv[i + 1]);
    }

    xrt::device dev{0};
    xrt::elf image(elf_path);

    if (mode == "hold") {
        xrt::hw_context hc(dev, image);   // the context whose file the kill has to release
        std::printf("ctx_create_probe: holding a hardware context (pid %d); SIGKILL me\n", int(getpid()));
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(600));
        return 0;
    }
    if (mode != "wait") {
        std::fprintf(stderr, "ctx_create_probe: unknown mode '%s' (hold|wait)\n", mode.c_str());
        return 2;
    }

    const auto t0 = clk::now();
    int attempts = 0;
    std::string last;
    while (ms_since(t0) < budget_ms) {
        ++attempts;
        try {
            xrt::hw_context hc(dev, image);
            std::printf("ctx_create_probe: context created after %.1f ms in %d attempt(s)\n", ms_since(t0), attempts);
            return 0;
        } catch (const std::exception& e) {
            last = e.what();
            if (attempts == 1 || attempts % 50 == 0)
                std::fprintf(stderr, "ctx_create_probe: attempt %d refused after %.1f ms: %s\n", attempts, ms_since(t0), last.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    std::printf("ctx_create_probe: no context in %.1f ms (%d attempts); last refusal: %s\n", ms_since(t0), attempts, last.c_str());
    return 1;
}
