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

// hadamard_rotate_test: the SIMD activation rotation (npu/forward/include/hadamard.h) must be *bit-identical*
// to the scalar loop it replaces, on every shape a container ships, and the scalar loop must still be the
// Hadamard transform (checked against a direct, independently written sum over the sign pattern).
//
// Bit identity is the whole contract: a folded container's weights were baked in this basis, so a rotation
// that differs in the last mantissa bit is a different activation and the model's text can move.  The gate
// that matters downstream is the device text gate, but that needs an NPU; this test is what runs on every
// build, on any host, with no device at all.
#include "hadamard.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
static void check(bool cond, const char* what) {
    if (cond) return;
    std::printf("FAIL: %s\n", what);
    failures++;
}

// The transform, written the slow and stupid way: out[o] = sum_i (-1)^popcount(i & o) a[i] * s[i], in double,
// scaled.  Independent of the butterfly's loop nest, so it catches a butterfly that is self-consistently wrong.
static void direct(const float* signs, const float* a, int block, std::vector<double>& out) {
    const double sc = 1.0 / std::sqrt((double)block);
    out.assign((size_t)block, 0.0);
    for (int o = 0; o < block; o++)
        for (int i = 0; i < block; i++) {
            const double v = (double)a[i] * (double)signs[i];
            out[(size_t)o] += (__builtin_popcount((unsigned)(i & o)) & 1) ? -v : v;
        }
    for (int o = 0; o < block; o++) out[(size_t)o] *= sc;
}

static void one_shape(int n, int block, unsigned seed, bool verbose) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> signs((size_t)n), src((size_t)n);
    for (int i = 0; i < n; i++) {
        signs[(size_t)i] = (i % 7 == 0) ? -1.0f : 1.0f;      // the converter writes +-1 signs
        src[(size_t)i] = dist(rng);
    }
    std::vector<float> scalar = src, simd = src;
    q4nx_hadamard::rotate_scalar(signs.data(), scalar.data(), n, block);
    q4nx_hadamard::rotate(signs.data(), simd.data(), n, block);

    char what[160];
    std::snprintf(what, sizeof what, "n=%d block=%d: SIMD rotation differs from the scalar loop", n, block);
    check(std::memcmp(scalar.data(), simd.data(), sizeof(float) * (size_t)n) == 0, what);

    // ... and the scalar loop is the transform (one block at a time; leftovers past the last full block only
    // get the sign multiply, exactly as the shipped loop does).
    std::vector<double> ref;
    double worst = 0.0;
    for (int b = 0; b + block <= n; b += block) {
        std::vector<float> a(src.begin() + b, src.begin() + b + block);
        direct(signs.data() + b, a.data(), block, ref);
        for (int i = 0; i < block; i++) {
            const double got = (double)scalar[(size_t)(b + i)];
            const double err = std::fabs(got - ref[(size_t)i]) / (1.0 + std::fabs(ref[(size_t)i]));
            if (err > worst) worst = err;
        }
    }
    std::snprintf(what, sizeof what, "n=%d block=%d: scalar loop is not the Hadamard transform (rel %.3g)",
                  n, block, worst);
    check(worst < 1e-5, what);
    if (verbose)
        std::printf("  n=%-6d block=%-5d bit-identical, transform rel err %.2e\n", n, block, worst);
}

int main() {
    std::printf("hadamard_rotate: SIMD path %s\n",
                q4nx_hadamard::simd_shape(1024) ? "selected (AVX2 host)" : "not selected on this host");
    // Every block a container can carry, and the real activation widths of this fleet's models: Bonsai-2's
    // 5120 hidden and 17408 intermediate, Qwen3.5-4B's 2560, plus a tail that is not a whole number of blocks.
    const int blocks[] = {1, 2, 4, 8, 16, 32, 1024, 2048};
    const int widths[] = {5120, 17408, 2560, 1024, 6153};
    for (int block : blocks)
        for (int n : widths) {
            const bool simd = q4nx_hadamard::simd_shape(block);
            one_shape(n, block, 1234u + (unsigned)(n * 31 + block), simd);
        }
    // The shipped container's own shape, on a few seeds, since a permutation bug can hide on one seed.
    for (unsigned seed = 1; seed <= 5; seed++) one_shape(5120, 1024, seed, false);
    // A block larger than the vector does nothing but the sign multiply: both paths must agree there too.
    one_shape(5120, 8192, 99, false);
    // NaN/inf must not be invented by either path.
    {
        std::vector<float> signs(16, 1.0f), a(16, 1e30f), s2, d2;
        s2 = a; d2 = a;
        q4nx_hadamard::rotate_scalar(signs.data(), s2.data(), 16, 16);
        q4nx_hadamard::rotate(signs.data(), d2.data(), 16, 16);
        for (int i = 0; i < 16; i++) {
            if (std::isfinite(s2[(size_t)i])) continue;
            check(std::isfinite(d2[(size_t)i]), "SIMD rotation invented a non-finite value the scalar loop did not");
            break;
        }
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("hadamard_rotate: all checks green\n");
    return 0;
}
