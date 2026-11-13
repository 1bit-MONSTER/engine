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

// engine#1783: npu/target/xdna/launch_fit.h -- a launch larger than its context is refused, with the numbers.
#include "launch_fit.h"

#include <cstdio>
#include <string>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
static bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

int main() {
    // The shapes from the standalone run: role 0's context is O (M=1536 K=768 N=1024).
    check(xdna_launch_misfit(1536, 768, 1024, 1536, 768, 1024).empty(), "the context's own shape fits");
    check(xdna_launch_misfit(1, 768, 1024, 16, 768, 1024).empty(), "one decode row on a 16-row twin fits");
    check(xdna_launch_misfit(100, 700, 1000, 128, 768, 1024).empty(), "a smaller launch fits");
    const std::string gu = xdna_launch_misfit(1536, 768, 6144, 1536, 768, 1024);
    check(has(gu, "N=6144 > 1024") && !has(gu, "K=768 >"), "GU on O's context: N named, K not");
    const std::string d = xdna_launch_misfit(1536, 3072, 1024, 1536, 768, 1024);
    check(has(d, "K=3072 > 768"), "D on O's context: K named");
    const std::string m = xdna_launch_misfit(1536, 768, 1024, 128, 768, 1024);
    check(has(m, "M=1536 > 128"), "more rows than the context holds");
    check(has(xdna_launch_misfit(200, 900, 2000, 128, 768, 1024), "M=200 > 128, K=900 > 768, N=2000 > 1024"),
          "every dimension over is named");
    check(xdna_launch_misfit(1536, 3072, 6144, 0, 0, 0).empty(), "an unknown geometry (0) is not checked");
    check(!xdna_launch_misfit(0, 768, 1024, 128, 768, 1024).empty(), "an empty launch is refused");
    check(has(d, "wrong role index"), "the message says what to look at");

    if (failures) { std::printf("launch_fit: %d failure(s)\n", failures); return 1; }
    std::printf("launch_fit: ok\n");
    return 0;
}
