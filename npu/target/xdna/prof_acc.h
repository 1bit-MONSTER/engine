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
// npu/target/xdna/prof_acc.h -- the NPU_PROFILE per-role accumulator and its one rule, header-only so a test with no
// device and no private runtime can arm it (xdna_target.cpp includes the private runtime header, so no CI job compiles
// it).
//
// engine#1783: a launch that missed the weight cache spent the fill inside the span the `pack` column times. The fill
// is a one-time cost per (layer, weight), so it is counted apart and taken out of `pack`, clamped at zero so a clock
// that reads the fill a little longer than the span can never print a negative cost.
#ifndef ONEBIT_NPU_TARGET_XDNA_PROF_ACC_H
#define ONEBIT_NPU_TARGET_XDNA_PROF_ACC_H

struct ProfAcc {
    double pack = 0, quant = 0, wait = 0, rb = 0;
    long   n = 0;
    double fill = 0;    // weight-cache fills: one-time, never in `pack`
    long   nfill = 0;
};

// What a launch paid for `pack` once its weight-cache fill (0 on a cached launch) is taken out.
inline double prof_pack_without_fill(double pack, double fill) {
    if (!(fill > 0.0)) return pack;
    return pack > fill ? pack - fill : 0.0;
}

inline void prof_acc_add(ProfAcc& a, double pack, double quant, double wait, double rb, double fill) {
    a.pack += prof_pack_without_fill(pack, fill);
    a.quant += quant; a.wait += wait; a.rb += rb;
    a.n++;
    if (fill > 0.0) { a.fill += fill; a.nfill++; }
}

#endif
