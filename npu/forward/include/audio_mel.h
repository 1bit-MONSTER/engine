// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// audio_mel.h -- the log-mel front-end: hann-windowed STFT (n_fft 400, hop 160, reflect padding), power spectrum, a
// slaney-normalised mel filterbank, then log10 and the (max - 8, +4)/4 scaling a Whisper encoder's input expects.
//
// This is the one part of the pipeline that is *not* in the pack: the pack's graph starts at the mel window, so the numbers
// here have to match the reference extractor's, and tests/audio_mel_test.cpp checks them against it.
//
// The filterbank is built, not shipped: librosa's slaney mel scale (f_sp = 200/3 Hz, linear below 1 kHz, log above,
// logstep = ln(6.4)/27) with slaney normalisation, which is what Whisper's feature extractor uses.

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace onebit::audio {

inline double hz_to_mel(double hz) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    if (hz < min_log_hz) return hz / f_sp;
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

inline double mel_to_hz(double mel) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    if (mel < min_log_mel) return mel * f_sp;
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

// (bins, 1 + n_fft/2): triangular filters, slaney-normalised, spanning 0..nyquist.
inline std::vector<float> mel_filterbank(int bins, int n_fft, int sample_rate) {
    const int nfreqs = n_fft / 2 + 1;
    const double nyq = sample_rate / 2.0;
    const double mlo = hz_to_mel(0.0), mhi = hz_to_mel(nyq);
    std::vector<double> f(bins + 2);
    for (int i = 0; i < bins + 2; ++i) f[i] = mel_to_hz(mlo + (mhi - mlo) * double(i) / double(bins + 1));
    // FFT bin centres in Hz, and the per-filter bandwidth normalisation.
    std::vector<double> hz(nfreqs);
    for (int k = 0; k < nfreqs; ++k) hz[k] = double(k) * nyq / double(nfreqs - 1);
    std::vector<double> enorm(bins);
    for (int i = 0; i < bins; ++i) {
        const double w = f[i + 2] - f[i];
        enorm[i] = w > 0 ? 2.0 / w : 0.0;
    }
    std::vector<float> m(size_t(bins) * size_t(nfreqs), 0.0f);
    for (int i = 0; i < bins; ++i) {
        const double lo = f[i], ctr = f[i + 1], hi = f[i + 2];
        for (int k = 0; k < nfreqs; ++k) {
            const double x = hz[k];
            double v = 0.0;
            if (x >= lo && x <= ctr && ctr > lo) v = (x - lo) / (ctr - lo);
            else if (x > ctr && x <= hi && hi > ctr) v = (hi - x) / (hi - ctr);
            m[size_t(i) * nfreqs + k] = float(v * enorm[i]);
        }
    }
    return m;
}

struct MelConfig {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop = 160;
    int bins = 80;
    int frames = 3000;   // the encoder's window: 30 s
};

// audio: `samples` mono f32 in [-1,1]. out: (bins, frames) row-major. Returns false only on a size the config cannot fill.
inline bool log_mel_spectrogram(const float* audio, size_t samples, const MelConfig& cfg, std::vector<float>* out) {
    const int pad = cfg.n_fft / 2;
    const int nfreqs = cfg.n_fft / 2 + 1;
    if (samples + 2 * size_t(pad) < size_t(cfg.n_fft)) return false;

    // reflect padding, both sides (torch's center=True default for stft)
    std::vector<double> x(samples + 2 * size_t(pad));
    for (int i = 0; i < pad; ++i) x[size_t(i)] = audio[pad - i];
    for (size_t i = 0; i < samples; ++i) x[size_t(pad) + i] = audio[i];
    for (int i = 0; i < pad; ++i) x[size_t(pad) + samples + size_t(i)] = audio[samples - 2 - size_t(i)];

    std::vector<double> win(cfg.n_fft);
    for (int i = 0; i < cfg.n_fft; ++i)   // periodic hann, as torch.hann_window(w) is by default
        win[size_t(i)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * double(i) / double(cfg.n_fft));

    const std::vector<float> fb = mel_filterbank(cfg.bins, cfg.n_fft, cfg.sample_rate);
    out->assign(size_t(cfg.bins) * size_t(cfg.frames), 0.0f);
    std::vector<double> re(nfreqs), im(nfreqs), power(nfreqs);
    for (int t = 0; t < cfg.frames; ++t) {
        const size_t base = size_t(t) * size_t(cfg.hop);
        for (int k = 0; k < nfreqs; ++k) {
            double sr = 0.0, si = 0.0;
            for (int n = 0; n < cfg.n_fft; ++n) {
                const double v = x[base + size_t(n)] * win[size_t(n)];
                const double ang = -2.0 * M_PI * double(k) * double(n) / double(cfg.n_fft);
                sr += v * std::cos(ang);
                si += v * std::sin(ang);
            }
            re[size_t(k)] = sr;
            im[size_t(k)] = si;
            power[size_t(k)] = sr * sr + si * si;
        }
        for (int b = 0; b < cfg.bins; ++b) {
            const float* row = fb.data() + size_t(b) * nfreqs;
            double acc = 0.0;
            for (int k = 0; k < nfreqs; ++k) acc += double(row[k]) * power[size_t(k)];
            out->at(size_t(b) * size_t(cfg.frames) + size_t(t)) = float(acc);
        }
    }
    // log10 with a floor, then Whisper's dynamic-range clamp and scale.
    double mx = -1e30;
    for (float& v : *out) {
        v = float(std::log10(std::max(double(v), 1e-10)));
        if (v > mx) mx = v;
    }
    const double floor_db = mx - 8.0;
    for (float& v : *out) {
        const double c = std::max(double(v), floor_db);
        v = float((c + 4.0) / 4.0);
    }
    return true;
}

}  // namespace onebit::audio
