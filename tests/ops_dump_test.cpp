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
// ops_dump_test -- NPU_INFER_DUMP_OPS (ops-dump.h): the on/off rule, the arithmetic of the line, its text, and
// that the variable is read once per process. No device, no container, no model.
//
//   ops_dump_test             the checks below
//   ops_dump_test --enabled   prints 1 or 0: what ops_dump_enabled() says in a process started with this environment
#include "ops-dump.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

// What a fresh process decides for one environment: the cached read cannot be re-armed in this one.
static int child_says(const char* self, const char* env_prefix) {
    const std::string cmd = std::string(env_prefix) + " '" + self + "' --enabled";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return -1;
    int c = std::fgetc(p);
    pclose(p);
    return c == '1' ? 1 : c == '0' ? 0 : -1;
}

int main(int argc, char** argv) {
    using namespace onebit::npu;
    if (argc > 1 && !std::strcmp(argv[1], "--enabled")) {
        std::printf("%d\n", ops_dump_enabled() ? 1 : 0);
        return 0;
    }

    // The rule.
    check(!ops_dump_value_on(nullptr), "unset is off");
    check(!ops_dump_value_on(""), "an empty value is off");
    check(!ops_dump_value_on("0"), "\"0\" is off");
    check(ops_dump_value_on("1"), "\"1\" is on");
    check(ops_dump_value_on("00"), "anything else is on");

    // The rule as a process sees it.
    check(child_says(argv[0], "env -u NPU_INFER_DUMP_OPS") == 0, "process: unset is off");
    check(child_says(argv[0], "env NPU_INFER_DUMP_OPS=") == 0, "process: empty is off");
    check(child_says(argv[0], "env NPU_INFER_DUMP_OPS=0") == 0, "process: 0 is off");
    check(child_says(argv[0], "env NPU_INFER_DUMP_OPS=1") == 1, "process: 1 is on");

    // Read once: whatever this process decided first, a later change of the variable does not move it.
    const bool first = ops_dump_enabled();
    setenv("NPU_INFER_DUMP_OPS", first ? "0" : "1", 1);
    check(ops_dump_enabled() == first, "the variable is read once per process");

    // The arithmetic: mean of |x| over n, and the largest |x|.
    const float x[] = {1.0f, -3.0f, 2.0f, 0.0f};
    OpsStats s = ops_stats(x, 4);
    check(s.mean == 1.5, "mean|x| of {1,-3,2,0} is 1.5");
    check(s.absmax == 3.0f, "absmax of {1,-3,2,0} is 3 (the negative one)");
    check(s.nonfinite == 0, "no non-finite value in a finite array");
    s = ops_stats(x, 2);
    check(s.mean == 2.0 && s.absmax == 3.0f, "n bounds the read: {1,-3} gives mean 2, absmax 3");
    s = ops_stats(x, 0);
    check(s.mean == 0.0 && s.absmax == 0.0f, "n = 0 is all zeros, not a division by zero");
    s = ops_stats(nullptr, 4);
    check(s.mean == 0.0 && s.absmax == 0.0f, "a null array is all zeros");

    // Non-finite values are counted and kept out of the two numbers.
    const float bad[] = {1.0f, std::numeric_limits<float>::quiet_NaN(), -5.0f, std::numeric_limits<float>::infinity()};
    s = ops_stats(bad, 4);
    check(s.nonfinite == 2, "NaN and Inf are counted");
    check(s.mean == 3.0 && s.absmax == 5.0f, "mean and absmax are over the finite values");

    // The text.
    char buf[192];
    int n = ops_dump_line(buf, sizeof buf, "attn_qkv", 7, 16, x, 4);
    check(std::string(buf) == "  [op] L07 attn_qkv   rows=16 |x| mean 1.5 absmax 3\n", std::string("line text, got: ") + buf);
    check(n == (int)std::strlen(buf), "the count is the line's length");
    ops_dump_line(buf, sizeof buf, "rmsnorm", 0, 1, bad, 4);
    check(std::string(buf) == "  [op] L00 rmsnorm    rows=1 |x| mean 3 absmax 5 nonfinite 2/4\n",
          std::string("non-finite line text, got: ") + buf);
    check(ops_dump_line(buf, sizeof buf, "t", 0, 1, nullptr, 4) == 0 && buf[0] == '\0', "a null array prints nothing");
    check(ops_dump_line(buf, sizeof buf, "t", 0, 1, x, 0) == 0 && buf[0] == '\0', "n = 0 prints nothing");
    char small[8];
    ops_dump_line(small, sizeof small, "attn_qkv", 7, 16, x, 4);
    check(std::strlen(small) == sizeof small - 1, "a short buffer is truncated and terminated");

    std::printf("%s\n", failures ? "FAILED" : "ops_dump: ok");
    return failures ? 1 : 0;
}
