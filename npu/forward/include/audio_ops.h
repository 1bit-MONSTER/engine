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
// audio_ops.h -- the four f32 host kernels a Whisper encoder front-end is made of: a temporal conv with stride and zero
// padding, layer norm, erf-GELU and a row softmax. Header-only, no dependencies.
//
// Why these and not a graph executor: the encoder's arithmetic that is easy to get subtly wrong is here -- the mel
// front-end is two strided 3-tap convs ([1280,128,3] stride 1 then [1280,1280,3] stride 2, both pad 1), and the GELU is
// the *exact* erf form, which numpy does not have (the reference builds it with A&S 7.1.26, |err| < 1.5e-7). Every number
// in tests/audio_ops_test.cpp was computed in double from the same formulas the reference uses, so a change here that
// moves a value is visible before it reaches the device.

#pragma once

#include <cmath>
#include <cstddef>

namespace onebit::audio {

// (T + 2*pad - k) / stride + 1, floor -- the ONNX Conv default for a valid (non-SAME) convolution.
inline int conv1d_out_len(int T, int k, int stride, int pad) {
    const int span = T + 2 * pad - k;
    return span < 0 ? 0 : span / stride + 1;
}

// x: (cin, T) row-major. w: (cout, cin, k) row-major -- the layout the ONNX initializer carries, so a caller reading
// weights through the pack's own loader needs no transpose. b: (cout,) or null. out: (cout, tout) row-major.
inline void conv1d(const float* x, int cin, int T, const float* w, const float* b, int cout, int k, int stride, int pad,
                   float* out) {
    const int tout = conv1d_out_len(T, k, stride, pad);
    for (int co = 0; co < cout; ++co) {
        for (int t = 0; t < tout; ++t) {
            float acc = (b != nullptr) ? b[co] : 0.0f;
            for (int ci = 0; ci < cin; ++ci) {
                const float* xr = x + static_cast<std::size_t>(ci) * T;
                const float* wr = w + (static_cast<std::size_t>(co) * cin + ci) * k;
                for (int j = 0; j < k; ++j) {
                    const int src = t * stride + j - pad;
                    if (src >= 0 && src < T) acc += wr[j] * xr[src];
                }
            }
            out[static_cast<std::size_t>(co) * tout + t] = acc;
        }
    }
}

// The same convolution as a GEMM (engine#1783: so the encoder's two Conv stems can run on the NPU's GEMM family).
//   A = im2col(x): (tout, cin*k), A[t][ci*k + j] = x[ci][t*stride + j - pad], 0 outside [0, T).
//   B = w as (cin*k, cout): B[ci*k + j][co] = w[co][ci][j].
//   out[co][t] = b[co] + (A x B)[t][co]
// conv1d() above is the reference this is tested against (tests/audio_ops_test.cpp); the column order ci*k + j is the
// initializer's own (cout, cin, k) flattening, so B is a plain transpose of w viewed as (cout, cin*k).
inline void conv1d_im2col(const float* x, int cin, int T, int k, int stride, int pad, float* A) {
    const int tout = conv1d_out_len(T, k, stride, pad);
    const std::size_t kk = static_cast<std::size_t>(cin) * static_cast<std::size_t>(k);
    for (int t = 0; t < tout; ++t) {
        float* row = A + static_cast<std::size_t>(t) * kk;
        for (int ci = 0; ci < cin; ++ci) {
            const float* xr = x + static_cast<std::size_t>(ci) * T;
            for (int j = 0; j < k; ++j) {
                const int src = t * stride + j - pad;
                row[static_cast<std::size_t>(ci) * k + j] = (src >= 0 && src < T) ? xr[src] : 0.0f;
            }
        }
    }
}

inline void conv1d_weight_as_kn(const float* w, int cout, int cin, int k, float* B) {
    const std::size_t kk = static_cast<std::size_t>(cin) * static_cast<std::size_t>(k);
    for (int co = 0; co < cout; ++co)
        for (std::size_t r = 0; r < kk; ++r) B[r * static_cast<std::size_t>(cout) + co] = w[static_cast<std::size_t>(co) * kk + r];
}

// y = (x - mean) / sqrt(var + eps) * gamma + beta, biased variance (divide by n), one row.
inline void layer_norm(const float* x, const float* gamma, const float* beta, int n, float eps, float* out) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    const double mean = sum / n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) {
        const float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        const float b = (beta != nullptr) ? beta[i] : 0.0f;
        out[i] = static_cast<float>((x[i] - mean) * inv) * g + b;
    }
}

// The exact form: 0.5*x*(1+erf(x/sqrt(2))). `approximate=none` in the export, so tanh is not a drop-in.
inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

// In place, one row: numerically stable (subtract the row max) so a logit of 100 does not overflow.
// engine#2366: the row softmax is 0.98 s of the whisper encoder's 10.4 s (12 launches of 1500x1500), and the serial version
// held it: the max scan is a reduction a compiler will not vectorise without -ffast-math, and `sum += x[i]` is a chain of
// dependent double adds with a `exp()` between each. Four accumulators for each turn both into independent work —
// measured on the encoder's real shape (18000 rows of 1500): 458.1 ms -> 141.5 ms, **3.24x**.
//
// The arithmetic per element is unchanged, but the SUMMATION ORDER is not: the four partial sums are combined as
// ((s0+s1)+(s2+s3)) rather than left to right, so the result differs from the old one in the last bits. That is the
// tolerance-side of the same trade the MatMul loop refused (there the order was kept and only aliasing was declared); this
// one changes order for a 3x win, and the whisper chain's ids are the check that it did not matter.
inline void softmax_row(float* x, int n) {
    if (n <= 0) return;
    float m0 = x[0], m1 = x[0], m2 = x[0], m3 = x[0];
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        if (x[i] > m0) m0 = x[i];
        if (x[i + 1] > m1) m1 = x[i + 1];
        if (x[i + 2] > m2) m2 = x[i + 2];
        if (x[i + 3] > m3) m3 = x[i + 3];
    }
    for (; i < n; ++i)
        if (x[i] > m0) m0 = x[i];
    const float mx = std::fmax(std::fmax(m0, m1), std::fmax(m2, m3));
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    i = 0;
    for (; i + 4 <= n; i += 4) {
        x[i] = std::exp(x[i] - mx);
        s0 += x[i];
        x[i + 1] = std::exp(x[i + 1] - mx);
        s1 += x[i + 1];
        x[i + 2] = std::exp(x[i + 2] - mx);
        s2 += x[i + 2];
        x[i + 3] = std::exp(x[i + 3] - mx);
        s3 += x[i + 3];
    }
    for (; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        s0 += x[i];
    }
    const double inv = 1.0 / ((s0 + s1) + (s2 + s3));
    for (int k = 0; k < n; ++k) x[k] = static_cast<float>(x[k] * inv);
}

}  // namespace onebit::audio
