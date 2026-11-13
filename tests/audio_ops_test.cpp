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
// audio_ops_test.cpp -- known-answer checks for npu/forward/include/audio_ops.h. No container, no device, no pack: every
// expected value was computed in double in Python from the same formulas the Whisper encoder reference uses
// (tools/whisper_encoder_ref.py), so this binary runs wherever CI builds, which is the point.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "audio_ops.h"

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

void expect_eq(int got, int want, const std::string& what) {
    if (got != want) {
        std::printf("FAIL %s: got %d want %d\n", what.c_str(), got, want);
        ++g_failures;
    }
}

void expect_close(float got, double want, double tol, const std::string& what) {
    const double d = std::fabs(static_cast<double>(got) - want);
    if (!(d <= tol)) {
        std::printf("FAIL %s: got %.10g want %.10g (|d| %.3g > %.3g)\n", what.c_str(), static_cast<double>(got), want, d,
                    tol);
        ++g_failures;
    }
}

// 1. The front-end conv, at the shapes the encoder actually uses: two strided 3-tap convs, pad 1 on both ends.
void conv1d_arms() {
    // cin 2, T 5, cout 3, k 3, stride 2, pad 1 -> tout 3
    const std::vector<float> x = {0.5f, -1.0f, 0.25f, 2.0f, -0.75f,   // channel 0
                                  1.5f, 0.0f, -0.5f, 1.0f, 0.25f};   // channel 1
    std::vector<float> w(3 * 2 * 3);
    for (int i = 0; i < 18; ++i) w[i] = static_cast<float>((i + 1) * 0.0625);
    const std::vector<float> b = {0.125f, -0.25f, 0.5f};

    expect_eq(onebit::audio::conv1d_out_len(5, 3, 2, 1), 3, "conv1d_out_len T5 k3 s2 p1");
    expect_eq(onebit::audio::conv1d_out_len(3000, 3, 2, 1), 1500, "conv1d_out_len 3000 -> 1500");

    std::vector<float> out(3 * 3);
    onebit::audio::conv1d(x.data(), 2, 5, w.data(), b.data(), 3, 3, 2, 1, out.data());

    // double, same formulas: [0.46875, 0.6875, 0.484375, 0.46875, 0.96875, 1.046875, 1.59375, 2.375, 2.734375]
    const double want[9] = {0.46875, 0.6875, 0.484375, 0.46875, 0.96875, 1.046875, 1.59375, 2.375, 2.734375};
    for (int i = 0; i < 9; ++i) expect_close(out[i], want[i], 1e-6, "conv1d value " + std::to_string(i));

    // edge case: stride 1, pad 0, k 3 -- the second conv's arithmetic without the padding, and a 1-tap full-window
    // where tout * stride + k - pad == T exactly (no tail row may be emitted past the input)
    std::vector<float> out2(2);
    onebit::audio::conv1d(x.data(), 1, 5, w.data(), nullptr, 1, 3, 1, 0, out2.data());
    expect_close(out2[0], 0.03076171875 * 0.0 + 0.0625 * 0.5 + 0.125 * -1.0 + 0.1875 * 0.25, 1e-6, "conv1d no bias");
    expect_eq(onebit::audio::conv1d_out_len(5, 3, 1, 0), 3, "conv1d_out_len T5 k3 s1 p0");
}

// engine#1783: the Conv as a GEMM (what the device runs) must be the Conv. Same strided, padded case as above, so a
// wrong column order (j*cin + ci instead of ci*k + j) or a missed pad tap shows up as a value, not just a shape.
void conv1d_gemm_arm() {
    const std::vector<float> x = {0.5f, -1.0f, 0.25f, 2.0f, -0.75f, 1.5f, 0.0f, -0.5f, 1.0f, 0.25f};
    std::vector<float> w(3 * 2 * 3);
    for (int i = 0; i < 18; ++i) w[i] = static_cast<float>((i + 1) * 0.0625);
    const std::vector<float> b = {0.125f, -0.25f, 0.5f};
    const int cin = 2, T = 5, cout = 3, k = 3, stride = 2, pad = 1;
    const int tout = onebit::audio::conv1d_out_len(T, k, stride, pad);
    std::vector<float> ref(size_t(cout) * tout);
    onebit::audio::conv1d(x.data(), cin, T, w.data(), b.data(), cout, k, stride, pad, ref.data());

    std::vector<float> A(size_t(tout) * cin * k), B(size_t(cin) * k * cout);
    onebit::audio::conv1d_im2col(x.data(), cin, T, k, stride, pad, A.data());
    onebit::audio::conv1d_weight_as_kn(w.data(), cout, cin, k, B.data());
    expect_close(A[0], 0.0, 0.0, "im2col t0 tap 0 is the left pad");
    expect_close(A[1], 0.5, 0.0, "im2col t0 tap 1 is x[0][0]");
    expect_close(A[3], 0.0, 0.0, "im2col t0 channel 1 tap 0 is the left pad");
    expect_close(B[1 * cout + 2], w[2 * 6 + 1], 0.0, "B[ci0,j1][co2] = w[co2][ci0][j1]");
    for (int co = 0; co < cout; ++co)
        for (int t = 0; t < tout; ++t) {
            double acc = b[size_t(co)];
            for (int r = 0; r < cin * k; ++r) acc += double(A[size_t(t) * cin * k + r]) * B[size_t(r) * cout + co];
            expect_close(float(acc), ref[size_t(co) * tout + t], 1e-6,
                         "im2col GEMM = conv1d at co " + std::to_string(co) + " t " + std::to_string(t));
        }
}

// 2. Layer norm: biased variance, eps added under the root, gamma/beta applied after the normalise.
void layernorm_arm() {
    const std::vector<float> v = {0.25f, -1.5f, 2.0f, 0.75f};
    const std::vector<float> g = {1.0f, 0.5f, -1.0f, 2.0f};
    const std::vector<float> be = {0.0f, 1.0f, 0.25f, -0.5f};
    std::vector<float> out(4);
    onebit::audio::layer_norm(v.data(), g.data(), be.data(), 4, 1e-5f, out.data());
    // double: [-0.0995034038, 0.2537244718, -1.0435442489, 0.0970204226]
    const double want[4] = {-0.0995034038, 0.2537244718, -1.0435442489, 0.0970204226};
    for (int i = 0; i < 4; ++i) expect_close(out[i], want[i], 1e-6, "layer_norm value " + std::to_string(i));

    // A constant row normalises to exactly beta whatever eps is: the variance is 0, so the numerator is 0 too.
    const std::vector<float> flat = {2.0f, 2.0f, 2.0f, 2.0f};
    onebit::audio::layer_norm(flat.data(), nullptr, nullptr, 4, 1e-5f, out.data());
    for (int i = 0; i < 4; ++i) expect_close(out[i], 0.0, 1e-7, "layer_norm flat " + std::to_string(i));
    // ... and with no gamma/beta the default is identity scale, zero shift.
    std::vector<float> out2(4);
    const std::vector<float> v2 = {1.0f, 2.0f, 3.0f, 4.0f};
    onebit::audio::layer_norm(v2.data(), nullptr, nullptr, 4, 1e-5f, out2.data());
    expect_close(out2[0], -1.34163542, 1e-6, "layer_norm null gamma/beta");
}

// 3. erf-GELU -- the export is approximate=none, so tanh must not pass this.
void gelu_arm() {
    // double: -0.0455002639, 0.3457312306, 2.9959503059
    expect_close(onebit::audio::gelu_erf(-2.0f), -0.0455002639, 1e-6, "gelu(-2)");
    expect_close(onebit::audio::gelu_erf(0.5f), 0.3457312306, 1e-6, "gelu(0.5)");
    expect_close(onebit::audio::gelu_erf(3.0f), 2.9959503059, 1e-6, "gelu(3)");

    // The two properties that separate erf-GELU from tanh-GELU: GELU(x) - GELU(-x) == x exactly, and the tanh form is
    // 1.7e-4 away at x = 2 -- far outside the 1e-6 this test runs at, so a tanh substitute cannot pass silently.
    expect_close(onebit::audio::gelu_erf(2.0f) - onebit::audio::gelu_erf(-2.0f), 2.0, 1e-6, "gelu odd part");
    const float tanh_form = 0.5f * 2.0f * (1.0f + std::tanh(0.7978845608f * (2.0f + 0.044715f * 8.0f)));
    check(std::fabs(tanh_form - onebit::audio::gelu_erf(2.0f)) > 1e-5, "tanh GELU is distinguishable from erf GELU");
}

// 4. Softmax: normalised, row-max stable, and order-preserving.
void softmax_arm() {
    std::vector<float> s = {1.0f, -2.0f, 0.5f, 0.25f};
    onebit::audio::softmax_row(s.data(), 4);
    // double: [0.4697737513, 0.0233886579, 0.2849321833, 0.2219054075]
    const double want[4] = {0.4697737513, 0.0233886579, 0.2849321833, 0.2219054075};
    for (int i = 0; i < 4; ++i) expect_close(s[i], want[i], 1e-6, "softmax value " + std::to_string(i));
    double sum = 0.0;
    for (float v : s) sum += v;
    expect_close(static_cast<float>(sum), 1.0, 1e-6, "softmax sums to 1");

    // +1000 on every logit must not change the result: the encoder's attention logits reach this range after the
    // 1/sqrt(d) scale is dropped on a long window.
    std::vector<float> big = {1001.0f, 998.0f, 1000.5f, 1000.25f};
    onebit::audio::softmax_row(big.data(), 4);
    for (int i = 0; i < 4; ++i) expect_close(big[i], want[i], 1e-6, "softmax shift-invariance " + std::to_string(i));
}

}  // namespace

int main() {
    conv1d_arms();
    conv1d_gemm_arm();
    layernorm_arm();
    gelu_arm();
    softmax_arm();
    if (g_failures != 0) {
        std::printf("audio ops: %d failures\n", g_failures);
        return 1;
    }
    std::printf("audio ops: conv1d (strided + padded + edge), conv1d as im2col GEMM, layer_norm (biased var, null affine), gelu (erf), softmax "
                "(stable, shift-invariant) -- all pass\n");
    return 0;
}
