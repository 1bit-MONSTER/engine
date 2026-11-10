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
// ops-dump.h -- NPU_INFER_DUMP_OPS=1 prints mean|.| and absmax after every projection and every norm, tagged
// with the call site's own name.
//
// Why it exists: NPU_INFER_DUMP_DIR/residual-dump.h writes the residual stream at LAYER boundaries, which tells
// you that a model diverges and not which op did it -- phi-4's residual was already 1e4 too large at layer 0 while
// its container matched its pack tensor for tensor (engine#1503). The two insertion points are `rmsnorm()` and
// `npu_gemm_rows()`; the tag each one already carries is the label, so there is no second name table and no
// per-op plumbing through the layer loop.
//
// Off unless the variable is set to something other than "" or "0", and it must never be able to fail a run: a
// diagnostic that changes behaviour is worse than no diagnostic (the same rule as the residual dump).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace onebit::npu {

inline bool ops_dump_enabled() {
    const char* v = std::getenv("NPU_INFER_DUMP_OPS");
    return v && *v && std::strcmp(v, "0") != 0;
}

// Row 0 only: one line per call site, which is what localises a divergence; the full tensors would flood a log.
inline void ops_dump(const char* tag, int layer, int rows, const float* x, int n) {
    if (!x || n <= 0) return;
    double sum = 0.0;
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        sum += a;
        if (a > mx) mx = a;
    }
    std::fprintf(stderr, "  [op] L%02d %-10s rows=%d |x| mean %.6g absmax %.6g\n", layer, tag, rows, sum / n, mx);
}

}  // namespace onebit::npu
