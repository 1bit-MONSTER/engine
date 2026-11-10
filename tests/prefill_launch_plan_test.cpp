// Copyright 2026 bong-water-water-bong. All rights reserved.
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).
//
// The prefill launch decision (engine#1959). The numbers in the comment are the issue's measured unit costs (16-row pass
// 1073.4 ms, 128-row pass 6992.2 ms, ratio 6.51x); the arms below pin the crossover those numbers imply, so a future pack with
// different twins can move the constant *and* see which M values change side.
#include "plan/prefill_launch_plan.h"

#include <cstdio>
#include <initializer_list>

using npu::plan::plan_prefill_launches;

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++fails; }
}

int main() {
    // one launch covers up to 16 rows
    for (int M = 1; M <= 16; ++M) {
        auto p = plan_prefill_launches(M, true);
        if (p.twin_launches != 1 || p.wide_launches != 0) {
            std::printf("FAIL M=%d: %d twin / %d wide, wanted 1/0\n", M, p.twin_launches, p.wide_launches);
            ++fails;
        }
    }
    check(fails == 0, "1..16 rows: one M=16 launch, never the wide kernel");

    // the issue's case: 17 rows is 2 launches, not one 6992 ms pass
    {
        auto p = plan_prefill_launches(17, true);
        check(p.twin_launches == 2 && p.wide_launches == 0, "17 rows: two M=16 launches (2147 ms, not 6992 ms)");
    }
    // the crossover, from the measured ratio: k <= 6 is cheaper, k = 7 is not
    {
        auto a = plan_prefill_launches(96, true);
        check(a.twin_launches == 6 && a.wide_launches == 0, "96 rows: six M=16 launches (6440 ms < one 6992 ms pass)");
        auto b = plan_prefill_launches(97, true);
        check(b.wide_launches == 1 && b.twin_launches == 0, "97 rows: seven launches would cost 7514 ms, so the wide path wins");
        auto c = plan_prefill_launches(128, true);
        check(c.wide_launches == 1 && c.twin_launches == 0, "128 rows: one wide launch");
    }
    // a pack with no _m16 twin can only go wide, whatever the arithmetic says
    for (int M : {1, 16, 17, 96}) {
        auto p = plan_prefill_launches(M, false);
        if (p.wide_launches != 1 || p.twin_launches != 0) {
            std::printf("FAIL M=%d no twin: %d/%d, wanted wide only\n", M, p.twin_launches, p.wide_launches);
            ++fails;
        }
    }
    check(true, "no _m16 twin: the wide path regardless of row count");
    check(plan_prefill_launches(0, true).wide_launches == 0 && plan_prefill_launches(0, true).twin_launches == 0,
          "0 rows: no launch at all");

    // A second pack's measured pair changes the answer -- and reproduces the 4B A/B that said splitting loses at 48 rows:
    // 2416.3 ms for one wide pass against ~834 ms per M=16 launch (2502.6 ms for three), i.e. k <= 2 wins, k = 3 loses.
    {
        const npu::plan::PrefillCost fourb{834.0, 2416.3};
        check(npu::plan::plan_prefill_launches(32, true, fourb).twin_launches == 2,
              "4B costs: 32 rows -> 2 launches (1668 ms < 2416 ms)");
        check(npu::plan::plan_prefill_launches(48, true, fourb).wide_launches == 1,
              "4B costs: 48 rows -> the wide pass, because 3 x 834 = 2502 ms > 2416 ms (the measured 3.6% loss)");
    }
    // and a pack whose twin is free is never worth splitting either way -- the degenerate guards
    {
        const npu::plan::PrefillCost zero{0.0, 6992.2};
        check(npu::plan::plan_prefill_launches(48, true, zero).wide_launches == 1,
              "a zero-cost twin is treated as unusable, not as infinitely fast");
    }

    if (fails == 0) std::printf("prefill launch plan: 11 arms pass (16 boundary, 17, 96/97 crossover, no-twin, 0, 4B costs x2, zero-cost guard)\n");
    return fails == 0 ? 0 : 1;
}
