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

// quant_i8_row_test: the SIMD activation quantiser (npu/forward/include/quant_i8.h) must produce the *same
// bytes* as the shipped scalar loop on every input a layer can hand it, and the scalar loop must still be
// roundf (away from zero on .5) with the same clamps.  Header only, no device, no model: runs in CI, on any
// host -- the AVX2 arm is compared only where the host has AVX2, and the scalar arm always runs.
//
// The cases below are the ones that separate a correct fast path from a plausible one: exact .5 boundaries
// (where round-half-to-even is a different answer), inputs whose product overflows to +-inf (the scalar does
// NOT zero those, it casts them and clamps to -127), values just either side of 2^31, NaN, +-inf, denormals,
// -0.0, and lengths that leave a tail after the last 8-lane block.
#include "quant_i8.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int failures = 0;

static void expect_close(const char* what, int got, int want) {
    if (got != want) {
        printf("FAIL %s: got %d, want %d\n", what, got, want);
        failures++;
    }
}

// The shipped loop is the definition: this is it, spelled out again, so a silent edit to the header's
// reference copy cannot move both sides together.
static void reference(const float* a, int n, float ais, int8_t* out) {
    for (int k = 0; k < n; k++) {
        float v = a[k];
        if (!std::isfinite(v)) v = 0;
        int q = (int)roundf(v * ais);
        if (q > 127) q = 127;
        else if (q < -127) q = -127;
        out[k] = (int8_t)q;
    }
}

static void check_arm(const char* arm, const std::vector<float>& A, float scale) {
    const float ais = 1.0f / scale;
    std::vector<int8_t> want(A.size(), 0), got(A.size(), 0);
    reference(A.data(), (int)A.size(), ais, want.data());
    if (std::strcmp(arm, "avx2") == 0) {
#if Q4NX_QUANT_I8_X86
        q4nx_quant_i8::row_avx2(A.data(), (int)A.size(), ais, got.data());
#else
        return;   // no AVX2 on this target: nothing to compare
#endif
    } else {
        q4nx_quant_i8::row_scalar(A.data(), (int)A.size(), ais, got.data());
    }
    for (size_t i = 0; i < A.size(); i++) {
        if (want[i] != got[i]) {
            printf("FAIL %s scale=%.9g at %zu: in=%.9g (x scale=%.9g) got %d want %d\n", arm,
                   scale, i, A[i], A[i] * ais, got[i], want[i]);
            failures++;
            return;
        }
    }
}

int main() {
    const float inf = std::numeric_limits<float>::infinity();
    const float fmax = std::numeric_limits<float>::max();
    const float fmin = std::numeric_limits<float>::denorm_min();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // 1. the arithmetic itself, before any timing: round half away from zero, and the clamps
    {
        const float ais = 1.0f;
        struct { float in; int want; } cases[] = {
            {0.0f, 0}, {-0.0f, 0}, {0.5f, 1}, {-0.5f, -1}, {1.5f, 2}, {-1.5f, -2},
            {2.5f, 3}, {-2.5f, -3}, {126.4f, 126}, {126.5f, 127}, {127.0f, 127}, {127.4f, 127},
            {127.6f, 127}, {-127.4f, -127}, {-127.6f, -127}, {1e9f, 127}, {-1e9f, -127},
            {nan, 0}, {inf, 0}, {-inf, 0},                 // isfinite() -> 0
            {fmax, -127}, {-fmax, -127},                   // finite in, huge product: cast saturates
            {2147483648.0f, -127}, {-2147483648.0f, -127}, // exactly 2^31 and beyond
            {2147483520.0f, 127}, {-2147483520.0f, -127},  // just inside 2^31
        };
        for (auto& c : cases) {
            int8_t out = 0;
            q4nx_quant_i8::row_scalar(&c.in, 1, ais, &out);
            expect_close("round", out, c.want);
        }
        // the same table through the dispatcher, which is what the engine calls
        for (auto& c : cases) {
            int8_t out = 0;
            q4nx_quant_i8::row(&c.in, 1, ais, &out);
            expect_close("dispatch", out, c.want);
        }
    }

    // 2. byte identity on bulk data, at scales a real layer uses (amax/127 and friends)
    std::mt19937 rng(20261103);
    for (float scale : {0.00390625f, 0.007874015748031496f, 1.0f, 0.0197f, 3.0f, 0.0001f}) {
        const float ais = 1.0f / scale;
        std::vector<float> A(4096 + 3);   // deliberately not a multiple of 8: exercises the tail
        std::uniform_real_distribution<float> u(-127.0f * scale, 127.0f * scale);
        for (auto& x : A) x = u(rng);
        check_arm("scalar", A, scale);
        check_arm("avx2", A, scale);

        // half-integer grid across the whole representable range of |v*ais| < 2^31
        std::uniform_int_distribution<int> iu(-2000000000, 2000000000);
        for (size_t i = 0; i < A.size(); i++) {
            A[i] = ((float)iu(rng) + ((rng() & 1) ? 0.5f : 0.0f)) * scale;
            if (i % 37 == 0) A[i] = ((float)((int)(rng() % 255) - 127) + 0.5f) * scale;  // exact .5
        }
        check_arm("avx2", A, scale);
        (void)ais;
    }

    // 3. the special values, in bulk and in every lane position
    for (float scale : {0.00390625f, 0.007874015748031496f}) {
        std::vector<float> A(2048);
        const float specials[] = {nan, inf, -inf, fmax, -fmax, fmin, -fmin, 0.0f, -0.0f,
                                  0.5f * scale, -0.5f * scale, 1.5f * scale, -1.5f * scale,
                                  2147483648.0f * scale, -2147483648.0f * scale};
        for (size_t i = 0; i < A.size(); i++) A[i] = specials[i % (sizeof(specials) / sizeof(float))];
        check_arm("scalar", A, scale);
        check_arm("avx2", A, scale);
        // rotate so each special lands in a different lane of the 8-wide block
        for (int off = 0; off < 8; off++) {
            std::vector<float> B(A.size());
            for (size_t i = 0; i < A.size(); i++) B[i] = A[(i + off) % A.size()];
            check_arm("avx2", B, scale);
        }
    }

    // 4. random *bit patterns*, not random floats: NaN payloads, both zeros, denormals and the infinities then
    // turn up at their real density instead of only where a case list remembers to put them. Every length 1..64
    // (so the tail is exercised at every offset), plus shapes that straddle the 8-lane blocks a real layer uses.
    // Rounds are env-tunable so CI stays fast; the default is ~10 M elements, about 0.15 s.
    {
        const char* env = std::getenv("QUANT_I8_FUZZ_ROUNDS");
        const long rounds = env ? std::atol(env) : 200000;
        std::mt19937_64 rng(0x20261103ULL);
        const float scales[] = {0.00390625f, 0.007874015748031496f, 1.0f, 0.0197f, 3.0f, 0.0001f, 1e-9f, 1e9f};
        std::vector<float> A(72), B;
        long elements = 0;
        for (long r = 0; r < rounds; r++) {
            const float scale = scales[r % (sizeof(scales) / sizeof(float))];
            const int n = (int)(r % 64) + 1;
            for (int i = 0; i < n; i++) {
                uint32_t bits = (uint32_t)rng();
                std::memcpy(&A[i], &bits, 4);
                // every fourth element: something a layer could plausibly produce, at full magnitude
                if ((rng() & 3) == 0) A[i] = (float)((int64_t)(rng() % 4000000) - 2000000) * scale;
            }
            check_arm("avx2", std::vector<float>(A.begin(), A.begin() + n), scale);
            elements += n;
        }
        for (int n : {1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 4093, 4096, 4099, 5120, 17408}) {
            B.resize((size_t)n);
            for (int i = 0; i < n; i++) {
                uint32_t bits = (uint32_t)rng();
                std::memcpy(&B[i], &bits, 4);
            }
            check_arm("avx2", B, 0.0197f);
            elements += n;
        }
        printf("random bit patterns: %ld rounds, %ld elements, %d failure(s)\n", rounds, elements, failures);
    }

    if (failures) {
        printf("quant_i8_row: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("quant_i8_row: PASS%s\n",
#if Q4NX_QUANT_I8_X86
           q4nx_quant_i8::have_avx2() ? " (scalar + avx2 arms)" : " (scalar arm; host has no AVX2)"
#else
           " (scalar arm; target has no AVX2)"
#endif
    );
    return 0;
}
