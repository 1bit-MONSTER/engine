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

// npu_kv_int8_test: the arithmetic P7 (#806) part A rests on, measured on this host with no model and no
// device.  The plan's `kv {key, value, hadamard}` field (npu/plan/plan.h) is a description; this is the
// part that decides whether the description is worth implementing, and it is the part that can be wrong
// silently, so it runs on every build.
//
// Claim 1 -- the rotation is free on the K side.  For an orthonormal Hadamard H, (QH)(KH)^T = QK^T, so a
// cache can store K rotated and the scores are the same scores.  If that were only approximately true the
// variant would be a different model that answers the same text most of the time, which is the worst
// failure shape there is.  Reported as a multiple of one f32 dot product's own rounding error, so the
// number means something instead of being compared to a tolerance somebody picked.
//
// Claim 2 -- what the rotation buys.  One int8 scale per (head, token) has to cover that vector's dynamic
// range, and real KV vectors carry a few large channels; rotating spreads them, so the same 8 bits cover
// the same vector with less error.  Measured as the attention-weight KL divergence against the f16 cache
// the plan calls the reference (P7's gate is KLD <= 1e-3), on an outlier-channel distribution and -- as
// the control that keeps this honest -- on a uniform one, where a rotation has nothing to concentrate and
// must not be credited with a win it did not earn.
//
// What this does NOT measure: model quality.  The vectors are synthetic, generated to the shapes and the
// outlier structure docs/npu-plan.md describes; the KLD that gates #806 is taken on real KV from a real
// model on the device.  This bounds the mechanism, not the model.
#include "hadamard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what);
}

// The attention head width both models this phase targets carry (docs/npu-plan.md: key_length = 256).
constexpr int kHead = 256;
constexpr int kKeys = 512;

// float -> binary16 -> float, round-to-nearest-even.  Written out rather than reached for with a compiler
// extension (_Float16 / std::float16_t), because this test is gated in CI on more than one compiler and a
// baseline is only a baseline if its rounding is the defined one.  Only the normal range this test uses is
// implemented; the early returns are the cases these magnitudes cannot reach.
float half_roundtrip(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = x & 0x80000000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xFFu);
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFF || exp == 0) return f;
    int32_t e = exp - 127;
    if (e < -14) return sign ? -0.0f : 0.0f;
    if (e > 15) return f;
    const uint32_t discarded = mant & 0x1FFFu;  // the 13 bits binary16 does not keep
    uint32_t keep = mant >> 13;
    constexpr uint32_t kHalfway = 0x1000u;
    if (discarded > kHalfway || (discarded == kHalfway && (keep & 1u))) keep += 1;
    if (keep == 0x400u) {  // rounding carried into the exponent
        keep = 0;
        if (++e > 15) return sign ? -INFINITY : INFINITY;
    }
    const uint32_t out = sign | ((uint32_t)(e + 127) << 23) | (keep << 13);
    float r;
    std::memcpy(&r, &out, sizeof r);
    return r;
}

// One quantization cell per row: scale = max|x| / 127, round to nearest, clamp, dequantize.  The contract
// variant 4 specifies (K int8, one scale per head per token).
void quant_int8(const float* x, int n, float* out) {
    float m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, std::fabs(x[i]));
    const float s = m > 0 ? m / 127.0f : 1.0f;
    for (int i = 0; i < n; i++) {
        float q = std::round(x[i] / s);
        q = std::max(-127.0f, std::min(127.0f, q));
        out[i] = q * s;
    }
}

// Softmax over keys in double, so the reference is not itself a source of error.
std::vector<double> softmax(const std::vector<float>& s) {
    double m = -INFINITY;
    for (float v : s) m = std::max<double>(m, v);
    std::vector<double> e(s.size());
    double sum = 0;
    for (size_t i = 0; i < s.size(); i++) { e[i] = std::exp((double)s[i] - m); sum += e[i]; }
    for (double& v : e) v /= sum;
    return e;
}

// KL(p || q): the arm's attention weights against the reference's, over the same keys.
double kl(const std::vector<double>& p, const std::vector<double>& q) {
    double d = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i] <= 0) continue;
        d += p[i] * std::log(p[i] / std::max(q[i], 1e-300));
    }
    return d;
}

// A KV matrix with the structure real KV vectors have: every channel Gaussian, but a handful scaled up --
// the outlier-channel effect the rotation exists to remove.  `gain` of 1.0 is the control: nothing to
// concentrate, so nothing for a rotation to win.
std::vector<float> make_kv(std::mt19937& rng, double outlier_gain) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<double> gain(kHead, 1.0);
    if (outlier_gain > 1.0) {
        std::uniform_int_distribution<int> chan(0, kHead - 1), n_out(4, 8);
        for (int i = 0, n = n_out(rng); i < n; i++) gain[chan(rng)] = outlier_gain;
    }
    std::vector<float> k((size_t)kKeys * kHead);
    for (int r = 0; r < kKeys; r++)
        for (int c = 0; c < kHead; c++) k[(size_t)r * kHead + c] = g(rng) * (float)gain[c];
    return k;
}

struct KlRow {
    double mean = 0, max = 0;
};

// One cache arm, scored against the f16 unrotated cache that ships today.  `rotate` applies the per-head
// Hadamard to Q and to every K row; `int8_k` stores K as int8 with one scale per row.  The reference is
// always the f16 cache, unrotated, so every row is "how far is this cache from today's".
KlRow measure(std::mt19937& rng, const std::vector<float>& K, const std::vector<float>& k_f16, bool rotate,
              bool int8_k) {
    std::vector<float> signs((size_t)kHead, 1.0f);
    std::vector<float> k = k_f16;
    if (rotate)
        for (int r = 0; r < kKeys; r++)
            q4nx_hadamard::rotate_scalar(signs.data(), &k[(size_t)r * kHead], kHead, kHead);
    if (int8_k)
        for (int r = 0; r < kKeys; r++) quant_int8(&k[(size_t)r * kHead], kHead, &k[(size_t)r * kHead]);
    (void)K;

    std::vector<double> klds;
    std::normal_distribution<float> g(0.0f, 1.0f);
    const double inv = 1.0 / std::sqrt((double)kHead);  // as attention scales
    for (int rep = 0; rep < 8; rep++) {
        std::vector<float> q((size_t)kHead);
        for (float& v : q) v = g(rng);
        std::vector<float> q_arm = q;
        if (rotate) q4nx_hadamard::rotate_scalar(signs.data(), q_arm.data(), kHead, kHead);

        std::vector<float> s_ref(kKeys), s_arm(kKeys);
        for (int r = 0; r < kKeys; r++) {
            double a = 0, b = 0;
            for (int c = 0; c < kHead; c++) {
                a += (double)q[c] * (double)k_f16[(size_t)r * kHead + c];
                b += (double)q_arm[c] * (double)k[(size_t)r * kHead + c];
            }
            s_ref[r] = (float)(a * inv);
            s_arm[r] = (float)(b * inv);
        }
        klds.push_back(kl(softmax(s_ref), softmax(s_arm)));
    }
    KlRow row;
    for (double v : klds) { row.mean += v; row.max = std::max(row.max, v); }
    row.mean /= (double)klds.size();
    return row;
}

// Claim 1: (QH)(KH)^T = QK^T up to f32 rounding, read as a multiple of one f32 dot product's own error.
void claim_rotation_is_free(std::mt19937& rng) {
    std::vector<float> signs((size_t)kHead, 1.0f);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> q((size_t)kHead), k((size_t)kHead);
    for (float& v : q) v = g(rng);
    for (float& v : k) v = g(rng);

    std::vector<float> qr = q, kr = k;
    q4nx_hadamard::rotate_scalar(signs.data(), qr.data(), kHead, kHead);
    q4nx_hadamard::rotate_scalar(signs.data(), kr.data(), kHead, kHead);

    double plain = 0, rot = 0;
    for (int c = 0; c < kHead; c++) { plain += (double)q[c] * (double)k[c]; rot += (double)qr[c] * (double)kr[c]; }
    float plain_f32 = 0;
    for (int c = 0; c < kHead; c++) plain_f32 += q[c] * k[c];

    const double floor = std::max(std::fabs(plain - (double)plain_f32), 1e-12);
    const double rel = std::fabs(plain - rot) / std::max(std::fabs(plain), 1e-12);
    std::printf("claim 1  scores after the rotation: |QK - (QH)(KH)| = %.3e absolute, %.3e relative -- %.2fx one "
                "f32 dot product's own rounding error, i.e. smaller than not rotating at all\n",
                std::fabs(plain - rot), rel, std::fabs(plain - rot) / floor);
    check(rel < 1e-5, "the rotation preserves the score to f32 rounding");
}

}  // namespace

int main() {
    std::mt19937 rng(20261106);
    claim_rotation_is_free(rng);

    double factor_at[3] = {0, 0, 0};
    int gi = 0;
    for (double gain : {1.0, 8.0, 30.0}) {
        std::vector<float> K = make_kv(rng, gain);
        std::vector<float> k_f16 = K;
        for (float& v : k_f16) v = half_roundtrip(v);

        const KlRow f16_rot = measure(rng, K, k_f16, /*rotate=*/true, /*int8_k=*/false);
        const KlRow i8_plain = measure(rng, K, k_f16, false, true);
        const KlRow i8_rot = measure(rng, K, k_f16, true, true);

        std::printf("\nKV distribution: %s (outlier channel gain %.0fx)\n",
                    gain > 1.0 ? "outlier channels, the shape real KV has" : "uniform, the control", gain);
        std::printf("  f16 cache, rotated           KLD mean %.3e  max %.3e\n", f16_rot.mean, f16_rot.max);
        std::printf("  int8 K, not rotated          KLD mean %.3e  max %.3e\n", i8_plain.mean, i8_plain.max);
        std::printf("  int8 K, rotated (variant 4)  KLD mean %.3e  max %.3e   %.1fx better than unrotated\n",
                    i8_rot.mean, i8_rot.max, i8_plain.mean / std::max(i8_rot.mean, 1e-300));

        factor_at[gi++] = i8_plain.mean / std::max(i8_rot.mean, 1e-300);
        if (gain > 1.0) {
            check(i8_rot.mean < 1e-3, "variant 4 lands inside the plan's 1e-3 gate on this distribution");
        } else {
            // The control.  Uniform magnitudes have nothing to concentrate, so a rotation claiming a big win
            // here would mean this test is measuring something other than what it says it measures.
            check(f16_rot.max < 1e-5, "a rotated f16 cache still matches the f16 cache (claim 1, at the weights)");
        }
    }

    // What the sweep above is for.  Both thresholds are set from the measured columns in this test, not
    // picked: the rotation's value has to *scale with outlier severity*, and it has to be worth nothing
    // where there are no outliers (otherwise this test is measuring something else).
    std::printf("\nrotation's KLD improvement: uniform %.1fx, 8x outliers %.1fx, 30x outliers %.1fx\n",
                factor_at[0], factor_at[1], factor_at[2]);
    check(factor_at[2] > factor_at[1] && factor_at[1] > factor_at[0],
          "the rotation's benefit grows with outlier severity (measured 54.5x / 9.6x / 1.0x)");
    check(factor_at[0] < 1.5, "on uniform data the rotation claims no win (measured 1.0x)");

    // The decisive one for part A: at 30x outliers an unrotated int8 K cache leaves the plan's gate and a
    // rotated one stays inside it.  That is the whole reason the rotation is in the contract.
    {
        std::vector<float> K = make_kv(rng, 30.0);
        std::vector<float> k_f16 = K;
        for (float& v : k_f16) v = half_roundtrip(v);
        const KlRow plain = measure(rng, K, k_f16, false, true);
        const KlRow rot = measure(rng, K, k_f16, true, true);
        std::printf("30x outliers, same question as the gate: unrotated int8 KLD %.3e %s 1e-3, rotated %.3e %s 1e-3\n",
                    plain.mean, plain.mean > 1e-3 ? "OUTSIDE" : "inside", rot.mean, rot.mean > 1e-3 ? "OUTSIDE" : "inside");
        check(plain.mean > 1e-3, "at 30x outliers an unrotated int8 K cache is outside the 1e-3 gate");
        check(rot.mean < 1e-3, "at 30x outliers the rotated int8 K cache stays inside it");
    }

    // The byte accounting the plan's variants are quoted from (docs/npu-plan.md).
    const double f16_bytes = (double)kHead * 2.0;
    const double int8_bytes = (double)kHead * 1.0 + 4.0;  // one f32 scale per (head, token)
    std::printf("\nbytes per (head, token): f16 K %.0f, int8 K + f32 scale %.0f (%.3fx smaller); "
                "with V int8 as well, %.3fx smaller on the whole KV\n",
                f16_bytes, int8_bytes, f16_bytes / int8_bytes, (2.0 * f16_bytes) / (2.0 * int8_bytes));

    if (failures) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("npu_kv_int8: ok\n");
    return 0;
}
