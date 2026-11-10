// Copyright 2026 bong-water-water-bong. All rights reserved.
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).

// How a prefill chunk of `rows` rows should be launched (engine#1959).
//
// Measured on the served route (Qwen2.5-7B b8 pack, `NPU_PHASE_PROF=1`, one npu-hold ticket): a 16-row pass costs
// **1073.4 ms** wall and a 128-row pass **6992.2 ms** -- 6.51x -- so `k = ceil(rows/16)` 16-row launches are cheaper than
// one wide pass while `k <= 6`, i.e. rows <= 96. A 17-row chunk is the worst case: 6992 ms instead of 2147 ms, 3.26x.
//
// THESE CONSTANTS ARE THAT PACK'S, NOT A RULE (engine#1959, measured on the 4B b8 pack 2026-11-10 19:44Z, one npu-hold
// ticket, 48-row prompt): the wide pass took **2416.3 ms** and three M=16 launches **2502.6 ms** -- the split is **3.6%
// SLOWER** there, i.e. 50 vs 52 ms/row. So the 6.5x cliff belongs to the 7B kernel set, and whether splitting pays has to
// be measured per pack before it is enabled for it: this function encodes one measurement, not a law. Make the two unit
// costs parameters when a second pack's numbers land.
//
// Header-only and dependency-free on purpose: the decision is the part that can be wrong by arithmetic alone, so it is
// pinned by tests/prefill_launch_plan_test.cpp without a device and without linking the XDNA runtime.
//
// NOTE: this is the decision, not the launch path. `XdnaTarget::npu_gemm_rows()` still sends 17..96 rows wide, because
// splitting means the GEMM entry point has to loop, write each 16-row block into the same output buffer and keep the prof
// accounting. Until that lands, this header is what the fix will call and what `ctx_for()`'s threshold refers to.
#ifndef ONEBIT_NPU_PLAN_PREFILL_LAUNCH_PLAN_H
#define ONEBIT_NPU_PLAN_PREFILL_LAUNCH_PLAN_H

namespace npu::plan {

struct PrefillLaunchPlan {
    int twin_launches = 0;   // M=16 launches
    int wide_launches = 0;   // M=128 launches
};

constexpr int kPrefillM16Rows = 16;   // rows one M=16 launch carries
constexpr int kPrefillMaxTwinLaunches = 6;   // 6 x 1073.4 = 6440 ms < one 6992 ms wide pass; 7 is 7514 ms, so it loses

// `has_twin` false means the pack carries no _m16 kernel for this projection: the wide path is then the only option, whatever
// the arithmetic says (the existing `if (!m16_) return prefill;` guard must not be bypassed by a split).
inline PrefillLaunchPlan plan_prefill_launches(int rows, bool has_twin) {
    PrefillLaunchPlan p;
    if (rows <= 0) return p;
    if (!has_twin) { p.wide_launches = 1; return p; }
    const int k = (rows + kPrefillM16Rows - 1) / kPrefillM16Rows;
    if (k <= kPrefillMaxTwinLaunches) { p.twin_launches = k; return p; }
    p.wide_launches = 1;
    return p;
}

}  // namespace npu::plan

#endif  // ONEBIT_NPU_PLAN_PREFILL_LAUNCH_PLAN_H
