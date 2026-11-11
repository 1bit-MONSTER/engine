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
// The sentence a serve owes an operator when it is about to run the slow route (engine#2102).
//
// The fact already exists as a machine line — `[forward] decode family: 0 role(s) at M=16, 4 on M=128` — and that line is
// grepped byte-for-byte by the gates, so it must not be reworded. What was missing is a sentence at the ready line's level:
// a serve with no 16-row decode kernels runs every decode step as prefill-width launches, measured at 1.9 tok/s where the
// fused route on the same box does 88-101, and nothing said so. This header is the whole decision, as a pure function, so it
// can be armed without a device and so the "say nothing" case is as testable as the "say something" one.
//
// It deliberately returns "" for the cases it cannot judge (a host-GEMM arm has no family at all; a count of 0 of 0 is a
// different fault, reported elsewhere) — a warning that fires on healthy serves is a warning nobody reads.

#pragma once

#include <cstdio>
#include <string>

namespace onebit::npu {

// decode_roles: roles that got an M=16 decode kernel. design_roles: roles the model has. kernels_dir: what the operator
// pointed at, echoed so the message can be acted on without reading the rest of the log.
inline std::string decode_family_note(int decode_roles, int design_roles, const std::string& kernels_dir) {
    if (design_roles <= 0) return {};          // no roles: not this issue's fault, and no number to quote
    if (decode_roles >= design_roles) return {};   // the fast route: say nothing
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "npu: %d of %d role(s) have a 16-row decode kernel in %s -- decode runs prefill-width launches and will be "
                  "far slower than the fused route (engine#2102). Ship the full_i8_*_m16 designs, or expect the many-call path.",
                  decode_roles, design_roles, kernels_dir.empty() ? "(no kernel directory)" : kernels_dir.c_str());
    return std::string(buf);
}

}  // namespace onebit::npu
