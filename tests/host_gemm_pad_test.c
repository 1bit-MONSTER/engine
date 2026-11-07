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

// host_gemm_pad_test -- the host fallback GEMM reads a padded design's activations the way the kernel
// does: at stride `a_k`, inside the caller's buffer, with the tail zero-filled.
//
// The case that motivated it (engine#808 review): the NpuTarget split moved the engine#845 pad into
// XdnaTarget and left the host branch of npu_gemm()/npu_gemm_rows() on the raw A. At a padded design
// (0 < real_k < K, e.g. hidden 896 stored at K=1024) that branch then read A[real_k .. K) past the end
// of the model's activation vector -- UB, and in the rows form at stride K from a buffer laid out at
// stride real_k, so every row after the first was the wrong row as well as overrun. The device rows and
// ctest all passed, because no container on the box has a non-256-aligned hidden size.
//
// What makes this a regression test rather than a tautology is the guard: the activation buffer is
// allocated with a NaN-filled tail and the test asserts the NaN tail is untouched afterwards, so a
// future version that drops the pad fails here even if its numbers happen to look plausible.
//
// Host-only: no model, no device.
#include "q4nx_pack.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check_int(int got, int want, const char *what) {
    if (got != want) {
        printf("FAIL %s: got %d want %d\n", what, got, want);
        failures++;
    }
}

static void check_close(float got, float want, float tol, const char *what) {
    if (!(fabsf(got - want) <= tol)) {
        printf("FAIL %s: got %.9g want %.9g\n", what, (double)got, (double)want);
        failures++;
    }
}

#define GUARD 64  // floats of NaN behind every activation buffer

// The reference the helper has to match: pad explicitly, then the plain triple loop over K.
static void reference(const float *A, int a_k, int M, int K, int N, const float *B, float *out) {
    float *padded = (float *)malloc(sizeof(float) * (size_t)M * (size_t)K);
    q4nx_pad_a_f32(A, M, a_k, K, padded);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)padded[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            out[(size_t)m * N + n] = (float)acc;
        }
    }
    free(padded);
}

// A padded design's activation buffer: M rows of `a_k` real values, followed by GUARD NaNs that must
// survive the call untouched.
static float *activation_with_guard(int M, int a_k) {
    float *buf = (float *)malloc(sizeof(float) * (size_t)(M * a_k + GUARD));
    for (int i = 0; i < M * a_k; i++) buf[i] = (float)(((i * 37) % 101) - 50) / 16.0f;
    for (int i = 0; i < GUARD; i++) buf[(size_t)M * a_k + i] = NAN;
    return buf;
}

static int guard_intact(const float *buf, int M, int a_k) {
    for (int i = 0; i < GUARD; i++)
        if (!isnan(buf[(size_t)M * a_k + i])) return 0;
    return 1;
}

static void case_padded(int M, int a_k, int K, int N, const char *what) {
    float *A = activation_with_guard(M, a_k);
    float *B = (float *)malloc(sizeof(float) * (size_t)K * (size_t)N);
    for (int i = 0; i < K * N; i++) B[i] = (float)(((i * 13) % 59) - 29) / 32.0f;
    float *got = (float *)malloc(sizeof(float) * (size_t)M * (size_t)N);
    float *want = (float *)malloc(sizeof(float) * (size_t)M * (size_t)N);
    memset(got, 0x7f, sizeof(float) * (size_t)M * (size_t)N);

    q4nx_host_gemm_f32_padded(A, a_k, M, K, N, B, got);
    reference(A, a_k, M, K, N, B, want);

    // The guard first: an implementation that reads past the buffer is the defect, whatever it computes.
    check_int(guard_intact(A, M, a_k), 1, what);

    // Then the numbers, against the explicit-pad reference. The helper accumulates in float and the
    // reference in double, so this is a tolerance rather than a bit comparison; the point of the case is
    // that the rows are read from the right place, not the summation order.
    float worst = 0.0f;
    for (int i = 0; i < M * N; i++) {
        const float d = fabsf(got[i] - want[i]);
        if (d > worst) worst = d;
    }
    char label[128];
    snprintf(label, sizeof(label), "%s: worst |d| %.3g", what, (double)worst);
    check_int(worst <= 1e-4f * (1.0f + fabsf(want[0])), 1, label);

    free(A); free(B); free(got); free(want);
}

// a_k == K (and the MLA form, a_k <= 0) must behave exactly like the old plain loop: the pad is a no-op.
static void case_unpadded(void) {
    const int M = 3, K = 8, N = 5;
    float A[3 * 8], B[8 * 5], got[3 * 5], want[3 * 5];
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) - 3.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5) - 2.0f;
    q4nx_host_gemm_f32_padded(A, K, M, K, N, B, got);
    reference(A, K, M, K, N, B, want);
    for (int i = 0; i < M * N; i++) check_close(got[i], want[i], 1e-5f, "a_k == K matches the reference");
    q4nx_host_gemm_f32_padded(A, 0, M, K, N, B, got);  // real_k <= 0: MLA designs
    for (int i = 0; i < M * N; i++) check_close(got[i], want[i], 1e-5f, "a_k = 0 (MLA) matches the reference");
}

int main(void) {
    // The shape the review named: hidden 896 stored at K = 1024, plus a one-row (decode) case and a
    // shape where a_k is not a multiple of anything.
    case_padded(1, 896, 1024, 64, "decode row, real_k 896 of K 1024");
    case_padded(4, 896, 1024, 64, "4 rows, real_k 896 of K 1024");
    case_padded(3, 5, 8, 7, "small odd shape, real_k 5 of K 8");
    case_unpadded();
    printf("host_gemm_pad: %s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
