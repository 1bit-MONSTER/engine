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
// ops_dump_test -- the NPU_INFER_DUMP_OPS gate: off by default, off for "0" and "", on for anything else, and it
// must print nothing at all when it is off (a diagnostic that changes a run is worse than none).
#include "ops-dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

int main() {
    using onebit::npu::ops_dump_enabled;
    unsetenv("NPU_INFER_DUMP_OPS");
    check(!ops_dump_enabled(), "absorbs unset");

    setenv("NPU_INFER_DUMP_OPS", "", 1);
    check(!ops_dump_enabled(), "an empty value is off");

    setenv("NPU_INFER_DUMP_OPS", "0", 1);
    check(!ops_dump_enabled(), "\"0\" is off");

    setenv("NPU_INFER_DUMP_OPS", "1", 1);
    check(ops_dump_enabled(), "\"1\" is on");

    // The line itself: one per call, the tag and the layer in it, and the arithmetic is mean|.|/absmax, which is
    // what makes two runs comparable.
    const float x[] = {1.0f, -3.0f, 2.0f};
    std::fprintf(stderr, "expect one line: ");
    onebit::npu::ops_dump("selftest", 7, 1, x, 3);
    check(true, "ops_dump is callable");

    std::printf("%s\n", failures ? "FAILED" : "ops_dump: ok");
    return failures ? 1 : 0;
}
