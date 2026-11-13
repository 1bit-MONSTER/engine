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

// engine#1783: npu/target/xdna/prof_acc.h -- the NPU_PROFILE accumulator keeps a weight-cache fill out of `pack`.
// Four cases of the rule, then the accumulator over a window shaped like the warm-up pass (half fills, half hits).
#include "prof_acc.h"

#include <cmath>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    check(near(prof_pack_without_fill(0.25, 0.0), 0.25), "fill == 0: pack unchanged (a cached launch)");
    check(near(prof_pack_without_fill(6.5, 6.25), 0.25), "0 < fill < pack: pack - fill");
    check(near(prof_pack_without_fill(6.5, 6.5), 0.0), "fill == pack: 0");
    check(near(prof_pack_without_fill(6.5, 7.0), 0.0), "fill > pack: 0, not negative");
    check(near(prof_pack_without_fill(0.25, -1.0), 0.25), "a negative fill is no fill");

    // 12 layers: each launches once with a fill (6.25 ms of a 6.5 ms span) and once cached (0.25 ms).
    ProfAcc a;
    for (int l = 0; l < 12; l++) {
        prof_acc_add(a, 6.5, 0.1, 1.2, 0.002, 6.25);
        prof_acc_add(a, 0.25, 0.1, 1.2, 0.002, 0.0);
    }
    check(a.n == 24, "every launch is counted");
    check(a.nfill == 12, "only the fills are counted as fills");
    check(near(a.fill, 12 * 6.25), "the fill total is the fills' own time");
    check(near(a.pack / a.n, 0.25), "pack per launch is what a cached launch pays");
    check(near(a.wait / a.n, 1.2) && near(a.quant / a.n, 0.1), "wait and quant are untouched");

    if (failures) { std::printf("prof_acc: %d failure(s)\n", failures); return 1; }
    std::printf("prof_acc: ok\n");
    return 0;
}
