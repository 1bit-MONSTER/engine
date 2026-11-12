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

// spec_greedy_test.cpp -- npu/forward/include/spec_greedy.h: the greedy round and the fallback rule (engine#1781).

#include <cstdio>

#include "spec_greedy.h"

using namespace onebit::npu;

static int g_fail = 0, g_n = 0;
static void check(bool ok, const char* what) {
    ++g_n;
    if (!ok) { std::printf("FAIL %s\n", what); ++g_fail; }
}

int main() {
    {
        const int d[3] = {5, 6, 7};
        const int all[4] = {5, 6, 7, 9};
        GreedyRound r = greedy_round(d, 3, all, 4);
        check(r.accepted == 3 && r.next == 9, "every draft kept: the 4th row gives the token after them");
        const int mid[4] = {5, 4, 7, 9};
        r = greedy_round(d, 3, mid, 4);
        check(r.accepted == 1 && r.next == 4, "a miss in the middle: one kept, the row that missed gives the next token, "
                                              "and a later match does not count");
        const int none[4] = {8, 6, 7, 9};
        r = greedy_round(d, 3, none, 4);
        check(r.accepted == 0 && r.next == 8, "the first draft missed: nothing kept, row 0's choice is next");
        r = greedy_round(nullptr, 0, all, 1);
        check(r.accepted == 0 && r.next == 5, "no drafts: a plain step through the same rule");
        check(greedy_round(d, 3, all, 3).accepted == -1, "a pass without the k + 1-th row is refused");
        check(greedy_round(d, -1, all, 4).accepted == -1 && greedy_round(d, 3, nullptr, 4).accepted == -1,
              "a negative k or no choices is refused");
    }
    {
        SpecFallback f(3.7, 3);
        check(!f.record(8) && !f.record(8), "fewer rounds than the window: never falls back");
        check(!f.record(1), "8, 8, 1: average 5.67, above 3.7");
    }
    {
        SpecFallback f(3.7, 3);
        f.record(8); f.record(8); f.record(1);
        check(f.record(2), "the window is the LAST three rounds: 8, 1, 2 averages 3.67, under 3.7");
        f.record(8); f.record(8); f.record(8);
        check(f.record(8) && f.off(), "once off, it stays off, even after rounds that would pass the window again");
    }
    {
        SpecFallback f(3.7, 3);
        bool off = false;
        for (int i = 0; i < 10; i++) off = off || f.record(7);
        check(!off && f.rounds() == 10, "a request that keeps accepting never falls back");
        SpecFallback never(0.0, 3);
        check(!never.record(1) && !never.record(1) && !never.record(1), "below <= 0 turns the rule off");
    }
    if (g_fail) { std::printf("spec_greedy: %d of %d checks FAILED\n", g_fail, g_n); return 1; }
    std::printf("spec_greedy: ok (%d checks)\n", g_n);
    return 0;
}
