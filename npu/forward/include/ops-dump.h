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
// ops-dump.h -- NPU_INFER_DUMP_OPS=1 prints mean|x| and absmax after every projection and every weighted RMSNorm,
// tagged with the call site's own name:
//
//   [op] L00 rmsnorm    rows=1 |x| mean 0.0105 absmax 0.0598
//   [op] L00 attn_qkv   rows=1 |x| mean 1.3877 absmax 31.8909
//
// Why it exists: NPU_INFER_DUMP_DIR (residual-dump.h) writes the residual stream at LAYER boundaries, which says
// that a model diverges and not which op did it (engine#1503: a residual already 1e4 too large at layer 0 while
// the container matched its pack tensor for tensor). The insertion points are `rmsnorm()`, `npu_gemm()` and
// `npu_gemm_rows()`; the tag each projection already carries is the label, so there is no second name table.
// The line is printed on the host arms and on the device arm alike, so the same run under NPU_INFER_HOME_GEMM=1
// and without it can be compared line for line.
//
// Row 0 only (the first N values of a batched result): one line per call, which is what localises a divergence.
//
// Off unless the variable is set to something other than "" or "0". The variable is read ONCE per process
// (ops_dump_enabled() is called per projection per token, and off has to cost a load and a branch, not a
// getenv). It can never fail a run: no allocation, no file, stderr only.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace onebit::npu {

// The rule itself, on a value: what the tests pin (the cached read below cannot be re-armed inside one process).
inline bool ops_dump_value_on(const char* v) { return v && *v && std::strcmp(v, "0") != 0; }

inline bool ops_dump_enabled() {
    static const bool on = ops_dump_value_on(std::getenv("NPU_INFER_DUMP_OPS"));
    return on;
}

struct OpsStats {
    double mean = 0.0;   // mean of |x[i]|
    float absmax = 0.0f; // max of |x[i]|
    int nonfinite = 0;   // NaN/Inf count: they are counted, and left out of mean and absmax
};

// The whole content of the diagnostic. A NaN would otherwise vanish from absmax (`a > mx` is false) and turn the
// mean into "nan" with no count, so non-finite values are reported as a number of their own.
inline OpsStats ops_stats(const float* x, int n) {
    OpsStats s;
    if (!x || n <= 0) return s;
    double sum = 0.0;
    int finite = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) { ++s.nonfinite; continue; }
        const float a = std::fabs(x[i]);
        sum += a;
        if (a > s.absmax) s.absmax = a;
        ++finite;
    }
    if (finite > 0) s.mean = sum / finite;
    return s;
}

// Formats the line into `buf` (always NUL-terminated when cap > 0) and returns snprintf's count; 0 when there is
// nothing to say. Separate from the print so a test can match the text.
inline int ops_dump_line(char* buf, size_t cap, const char* tag, int layer, int rows, const float* x, int n) {
    if (!buf || cap == 0) return 0;
    buf[0] = '\0';
    if (!x || n <= 0) return 0;
    const OpsStats s = ops_stats(x, n);
    if (s.nonfinite)
        return std::snprintf(buf, cap, "  [op] L%02d %-10s rows=%d |x| mean %.6g absmax %.6g nonfinite %d/%d\n", layer,
                             tag ? tag : "?", rows, s.mean, (double)s.absmax, s.nonfinite, n);
    return std::snprintf(buf, cap, "  [op] L%02d %-10s rows=%d |x| mean %.6g absmax %.6g\n", layer, tag ? tag : "?",
                         rows, s.mean, (double)s.absmax);
}

inline void ops_dump(const char* tag, int layer, int rows, const float* x, int n) {
    char buf[192];
    if (ops_dump_line(buf, sizeof buf, tag, layer, rows, x, n) > 0) std::fputs(buf, stderr);
}

}  // namespace onebit::npu
