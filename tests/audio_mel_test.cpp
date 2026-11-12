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
// audio_mel_test.cpp -- the log-mel front-end against the reference extractor's numbers.
//
// Input: 30 s at 16 kHz of three summed sinusoids (440, 1250, 3100 Hz), built from a formula so both sides use the same
// signal. Expected values are transformers' WhisperFeatureExtractor (feature_size=80) on that signal.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "audio_mel.h"

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

void expect_close(float got, double want, double tol, const std::string& what) {
    const double d = std::fabs(double(got) - want);
    if (!(d <= tol)) {
        std::printf("FAIL %s: got %.9g want %.9g (|d| %.3g)\n", what.c_str(), double(got), want, d);
        ++g_failures;
    }
}

}  // namespace

int main() {
    onebit::audio::MelConfig cfg;
    const int sr = cfg.sample_rate;
    const size_t n = size_t(sr) * 30;
    std::vector<float> sig(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = double(i) / sr;
        sig[i] = float(0.3 * std::sin(2 * M_PI * 440.0 * t) + 0.2 * std::sin(2 * M_PI * 1250.0 * t) +
                       0.1 * std::sin(2 * M_PI * 3100.0 * t));
    }

    std::vector<float> mel;
    if (!onebit::audio::log_mel_spectrogram(sig.data(), sig.size(), cfg, &mel)) {
        std::printf("FAIL front-end refused a 30 s window\n");
        return 1;
    }
    check(mel.size() == size_t(cfg.bins) * size_t(cfg.frames), "mel is (80, 3000)");
    if (mel.size() != size_t(cfg.bins) * size_t(cfg.frames)) return 1;

    double mean = 0.0, mn = 1e30, mx = -1e30;
    for (float v : mel) {
        mean += v;
        mn = std::min<double>(mn, v);
        mx = std::max<double>(mx, v);
    }
    mean /= double(mel.size());
    auto at = [&](int b, int t) { return mel[size_t(b) * size_t(cfg.frames) + size_t(t)]; };

    // reference: mean -0.33795589, min -0.672718, max 1.327282
    expect_close(float(mean), -0.33795589, 2e-4, "mel mean");
    expect_close(float(mn), -0.672718, 2e-4, "mel min");
    expect_close(float(mx), 1.327282, 2e-4, "mel max");

    // bin 0, first five frames -- the onset and the dynamic-range clamp (three equal -0.6727176 values)
    const double p0[5] = {0.92388016, 0.41213650, -0.67271757, -0.67271757, -0.67271757};
    for (int i = 0; i < 5; ++i) expect_close(at(0, i), p0[i], 2e-3, "bin 0 frame " + std::to_string(i));
    // mid band, the end of the window
    const double p1[5] = {-0.65874779, -0.66997766, -0.65873981, -0.66998839, -0.65872025};
    for (int i = 0; i < 5; ++i) expect_close(at(40, 1499 + i), p1[i], 2e-3, "bin 40 frame " + std::to_string(1499 + i));
    // top bin
    const double p2[3] = {0.18210298, -0.32799470, -0.67271757};
    for (int i = 0; i < 3; ++i) expect_close(at(79, i), p2[i], 2e-3, "bin 79 frame " + std::to_string(i));

    // A silent window still produces a full, finite spectrogram: the log floor and the max-8 clamp own that case, and a
    // NaN there would poison every frame the encoder sees.
    std::vector<float> silent(n, 0.0f), smel;
    check(onebit::audio::log_mel_spectrogram(silent.data(), silent.size(), cfg, &smel), "silence is accepted");
    bool finite = true;
    for (float v : smel) finite = finite && std::isfinite(v);
    check(finite, "a silent window produces finite numbers");
    // Silence is the floor of the whole chain: power 0 -> log10(1e-10) = -10, the max is that too, so the max-8 clamp never
    // bites and the scale gives (-10+4)/4 = -1.5. (My first expectation here said 1.0 -- the test caught its author again.)
    expect_close(smel[0], -1.5f, 1e-6, "silence lands on the log floor: (-10+4)/4 == -1.5");

    if (g_failures != 0) {
        std::printf("audio mel: %d failures\n", g_failures);
        return 1;
    }
    std::printf("audio mel: front-end matches the reference extractor on tones, silences on the clamp\n");
    return 0;
}
