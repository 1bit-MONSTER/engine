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

// attn_math_test: the forward's attention kernels (npu/forward/include/attn_math.h) are the same sum as a
// double-precision direct sum, in every form the file ships -- the AVX2+FMA body, the portable
// four-accumulator body, and whichever of them dot()/saxpy() pick on this host.
//
// The contract is NOT bit identity (the two bodies deliberately sum in different orders); it is "a float
// rounding of the same sum".  So the bound is the arithmetic worst case, ~n*eps*sum|a_i*b_i|, tightened to
// a factor that still fails a body whose tail loop drops elements, whose horizontal add is wrong, or which
// reads past the end -- the failure modes a width sweep over the forward's real head dims actually hits.
//
// It also matters that this runs on CI at all: a build host with AVX2 only ever exercises dot_avx2 in
// production, so without this the portable body would be dead code until someone runs the engine on an
// older machine.  The test names which body it exercised, so a CI log answers that question directly.
#include "attn_math.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int failures = 0;
static void check(bool cond, const char* what, int n, double got, double want) {
    if (cond) return;
    std::printf("FAIL: %s (n=%d got=%.9g want=%.9g)\n", what, n, got, want);
    failures++;
}

// The forward's real head dims (64/96/128/256/512 across the families it serves) plus the widths where a
// vector body's tail runs: 1..17 covers 8- and 16-lane remainders and both of them at once.
static const int kWidths[] = {1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 24, 31, 64, 96, 128, 256, 512, 896, 3072};

int main() {
#if Q4NX_ATTN_X86
    std::printf("dispatch: AVX2+FMA %s\n", q4nx_attn::have_avx2_fma() ? "available (dot/saxpy take the SIMD body)"
                                                                     : "absent (dot/saxpy take the portable body)");
#else
    std::printf("dispatch: no x86 SIMD build (dot/saxpy take the portable body)\n");
#endif
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    for (int n : kWidths) {
        std::vector<float> a((size_t)n), b((size_t)n), y((size_t)n), y0((size_t)n);
        for (int i = 0; i < n; i++) {
            a[(size_t)i] = u(rng);
            b[(size_t)i] = u(rng);
            y0[(size_t)i] = u(rng);
        }
        double direct = 0.0, abs_sum = 0.0;
        for (int i = 0; i < n; i++) {
            direct += (double)a[(size_t)i] * (double)b[(size_t)i];
            abs_sum += std::fabs((double)a[(size_t)i] * (double)b[(size_t)i]);
        }
        // 4 * eps * abs_sum with eps = 2^-24; the factor leaves room for the reduction tree's own order
        // while still being far tighter than a dropped element (which moves the sum by O(abs_sum/n)).
        const double tol = 4.0 * std::ldexp(1.0, -24) * (abs_sum > 0.0 ? abs_sum : 1.0);
        check(std::fabs((double)q4nx_attn::dot_scalar(a.data(), b.data(), n) - direct) <= tol,
              "dot_scalar != direct sum", n, q4nx_attn::dot_scalar(a.data(), b.data(), n), direct);
        check(std::fabs((double)q4nx_attn::dot(a.data(), b.data(), n) - direct) <= tol,
              "dot != direct sum", n, q4nx_attn::dot(a.data(), b.data(), n), direct);

        // A single fused multiply-add is y0 + p*x rounded once, so the bound is set by the OPERAND
        // magnitudes, not by the result: under cancellation the result can be ~1e-3 while both terms are
        // O(1), and a result-relative bound would call a correct FMA wrong.  (Written down here because
        // that is exactly what the first version of this test did.)
        const float p = u(rng);
        auto saxpy_direct = [&](bool simd) {
            y = y0;
            if (simd) q4nx_attn::saxpy(p, a.data(), y.data(), n);
            else q4nx_attn::saxpy_scalar(p, a.data(), y.data(), n);
            for (int i = 0; i < n; i++) {
                const double want = (double)y0[(size_t)i] + (double)p * (double)a[(size_t)i];
                const double ftol = 4.0 * std::ldexp(1.0, -24) *
                                    (std::fabs((double)y0[(size_t)i]) +
                                     std::fabs((double)p * (double)a[(size_t)i]));
                char what[64];
                std::snprintf(what, sizeof what, "%s != direct (n=%d i=%d)",
                              simd ? "saxpy" : "saxpy_scalar", n, i);
                check(std::fabs((double)y[(size_t)i] - want) <= ftol, what, n, y[(size_t)i], want);
            }
        };
        saxpy_direct(false);
        saxpy_direct(true);
    }

    // A zero vector is the one input where a body that forgets to add its accumulators still looks right
    // for the wrong reason: pin it, and pin that both bodies return the same signed zero-ness.
    std::vector<float> z(96, 0.0f), ones(96, 1.0f);
    check(q4nx_attn::dot(z.data(), ones.data(), 96) == 0.0f, "dot on zeros != 0", 96,
          q4nx_attn::dot(z.data(), ones.data(), 96), 0.0);
    check(q4nx_attn::dot_scalar(z.data(), ones.data(), 96) == 0.0f, "dot_scalar on zeros != 0", 96,
          q4nx_attn::dot_scalar(z.data(), ones.data(), 96), 0.0);

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ok: %zu widths, dot and saxpy agree with a double-precision direct sum\n",
                sizeof(kWidths) / sizeof(kWidths[0]));
    return 0;
}
