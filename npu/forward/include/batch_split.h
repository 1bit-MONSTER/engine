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
// batch_split.h -- whether a prefill chunk runs as one wide launch or as several narrow ones.
//
// A pack that carries both kernel families has a narrow design (16 rows) and a wide one (128 rows) for the same
// projection. The wide design walks all of its baked rows whatever the chunk holds, so a 17-row chunk pays for 128:
// measured on an imported 7B pack, one 16-row pass is 1073 ms and one 17-row pass is 6992 ms. Two narrow launches
// would have cost a third of that. Which side wins depends on the family (on another family the wide launch is only
// twice a narrow one), so the choice is made from the two MEASURED launch costs of this process and not from a
// constant.
//
// Pure and header-only, like batch_width.h: the decision is the part that can be wrong, so it is pinned by a host
// test with no device. Row results do not depend on it -- every row of a batched GEMM is computed independently of
// the others, which is the property the blocked walk for chunks wider than the design already relies on.
#pragma once

namespace onebit {
namespace npu {

// The block width to walk an M-row chunk at, or 0 to keep it as one launch of the wide design.
//
//   narrow, wide          rows the two designs take (16 and 128); no split unless 2 <= narrow < wide
//   narrow_ms, wide_ms    measured wall of one launch of each for this projection; 0 or less means "not measured
//                         yet", and an unmeasured side keeps the one wide launch (which is how it gets measured)
//
// The split must save at least a tenth of the wide launch, so two costs that are level do not make the route depend
// on which one was measured in a noisier moment.
inline int batch_split_rows(int M, int narrow, int wide, double narrow_ms, double wide_ms) {
    if (narrow < 2 || wide <= narrow) return 0;        // one family only
    if (M <= narrow || M > wide) return 0;             // already a narrow launch, or the caller blocks it first
    if (!(narrow_ms > 0.0) || !(wide_ms > 0.0)) return 0;
    const int launches = (M + narrow - 1) / narrow;
    return double(launches) * narrow_ms < 0.9 * wide_ms ? narrow : 0;
}

}  // namespace npu
}  // namespace onebit
