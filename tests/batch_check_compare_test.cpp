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
// batch_check_compare_test.cpp -- pins the comparator behind NPU_BATCH_CHECK (engine#1781), so the oracle's answer is
// arithmetic that a host test can check rather than a device run that has to be trusted. No model, no device, no
// runtime: it is two float arrays and a count.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "batch_check_compare.h"

using onebit::npu::compare_rows_bitwise;
using onebit::npu::RowMismatch;

static int failures = 0;

static void check(bool ok, const char* what, const char* detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, ok ? "" : " -- ", ok ? "" : detail);
    if (!ok) ++failures;
}

static std::vector<float> rows(int nrows, int cols, float base) {
    std::vector<float> v((size_t)nrows * (size_t)cols);
    for (size_t i = 0; i < v.size(); ++i) v[i] = base + (float)(i % 97) * 0.011f;
    return v;
}

int main() {
    const int R = 7, C = 64;                       // the shape the gate asks about: k = 7 rows of logits

    {   // 1. identical buffers match, row for row
        auto a = rows(R, C, 0.25f);
        auto b = a;
        RowMismatch m;
        check(compare_rows_bitwise(a.data(), b.data(), R, C, &m) == 0, "identical rows report no divergence");
        check(m.row == -1 && m.index == -1, "and leave the mismatch untouched");
    }
    {   // 2. a difference in the LAST row is found -- the old oracle's only case, so it must still work
        auto a = rows(R, C, 0.25f);
        auto b = a;
        b[(size_t)(R - 1) * C + 5] += 1e-7f;
        RowMismatch m;
        int n = compare_rows_bitwise(a.data(), b.data(), R, C, &m);
        char d[160];
        std::snprintf(d, sizeof d, "n=%d row=%d index=%d", n, m.row, m.index);
        check(n == 1 && m.row == R - 1 && m.index == 5, "a difference in the last row is found", d);
    }
    {   // 3. ...and a difference in an EARLY row is found, which the old oracle could not see at all
        auto a = rows(R, C, 0.25f);
        auto b = a;
        b[(size_t)2 * C + 11] += 1e-7f;            // row 2 of 7: a drafted position, not the last one
        RowMismatch m;
        int n = compare_rows_bitwise(a.data(), b.data(), R, C, &m);
        char d[160];
        std::snprintf(d, sizeof d, "n=%d row=%d index=%d", n, m.row, m.index);
        check(n == 1 && m.row == 2 && m.index == 11, "a difference in an early row is found too", d);
    }
    {   // 4. bit-exactness is the standard: one ulp apart is a divergence, not a rounding
        auto a = rows(R, C, 1.0f);
        auto b = a;
        b[3] = std::nextafter(b[3], 2.0f);
        RowMismatch m;
        int n = compare_rows_bitwise(a.data(), b.data(), R, C, &m);
        check(n == 1 && m.row == 0 && m.index == 3, "a one-ulp difference is reported (the bar is bit-exact)", "no divergence reported");
    }
    {   // 5. every differing row is counted, so a run reports the size of the divergence and not just its presence
        auto a = rows(R, C, 0.25f);
        auto b = a;
        for (int r = 0; r < R; ++r) b[(size_t)r * C + 1] += 1e-6f;
        RowMismatch m;
        int n = compare_rows_bitwise(a.data(), b.data(), R, C, &m);
        char d[80];
        std::snprintf(d, sizeof d, "n=%d (wanted %d)", n, R);
        check(n == R, "all seven differing rows are counted", d);
        check(m.row == 0, "and the first one is the first row");
    }
    {   // 6. NaN on one side only is a divergence; NaN on both is not (the same non-answer)
        const float nan = std::numeric_limits<float>::quiet_NaN();
        auto a = rows(2, 4, 0.5f);
        auto b = a;
        b[6] = nan;                                 // row 1, col 2: batched has NaN, sequential does not
        RowMismatch m;
        check(compare_rows_bitwise(a.data(), b.data(), 2, 4, &m) == 1 && m.row == 1 && m.index == 2 && m.nan_mismatch,
              "a one-sided NaN is reported as a mismatch");
        auto c = a;
        c[6] = nan;
        auto d2 = a;
        d2[6] = nan;
        check(compare_rows_bitwise(c.data(), d2.data(), 2, 4, nullptr) == 0, "and two NaNs in the same place agree");
    }
    {   // 7. degenerate shapes are refusals, not silent success
        auto a = rows(2, 4, 0.5f);
        check(compare_rows_bitwise(nullptr, a.data(), 2, 4, nullptr) == 2, "a null side counts every row as differing");
        check(compare_rows_bitwise(a.data(), a.data(), 0, 4, nullptr) == 0, "zero rows compares as equal (nothing to check)");
    }

    std::printf("batch_check_compare: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
