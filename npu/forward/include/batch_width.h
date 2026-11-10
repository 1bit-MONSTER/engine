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
// batch_width.h -- how many rows one launch may carry (engine#1900).
//
// The forward chunks a batched pass at this width, and the activation BO holds exactly this many rows: an over-wide
// launch reads and poisons past the BO, which is memory corruption rather than a slow path (engine#1372). So the
// aggregate over a design set must be the MINIMUM of the widths actually loaded, never the first one that happens to
// be populated -- a 16/16/128 set returning 128 corrupts memory on the two narrow roles, while returning 16 is only
// narrower than it needs to be.
//
// Pure and header-only on purpose: the aggregation is the part that can be wrong in the dangerous direction, so it is
// pinned by a host test with no device, no ELF and no runtime.
#pragma once

namespace onebit {
namespace npu {

// The smallest positive width in `widths`, or `fallback` when none is positive.
//
// `widths` may contain 0 for "no design at this role", which is the ordinary state of a design array -- the old code
// returned the FIRST populated entry, so a zero-width slot simply moved the answer to whichever role came next.
inline int batch_width_min(const int* widths, int n, int fallback) {
    int w = 0;
    if (widths)
        for (int i = 0; i < n; ++i) {
            const int v = widths[i];
            if (v > 0 && (w == 0 || v < w)) w = v;
        }
    return w > 0 ? w : (fallback > 0 ? fallback : 1);
}

}  // namespace npu
}  // namespace onebit
