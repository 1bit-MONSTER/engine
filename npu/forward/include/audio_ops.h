// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
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
inline void softmax_row(float* x, int n) {
    if (n <= 0) return;
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const double inv = 1.0 / sum;
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(x[i] * inv);
}

}  // namespace onebit::audio
