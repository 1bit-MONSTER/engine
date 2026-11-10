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
// whisper_transcribe.cpp -- the whole host pipeline: wav -> log-mel -> encoder -> greedy decode -> token ids.
//
//     ./whisper_transcribe <encoder.onnx> <decoder.onnx> <file.wav> [expected-ids-csv ...]
//
// A tool rather than a test because one forward pass of each graph is ~26 s at these shapes and a sentence is 25 of them
// (see PR #1931); `ctest` would time out for a reason that has nothing to do with correctness. The numbers it prints are
// the ones a test would assert, so `expected-ids-csv` turns it into a check when run by hand.
//
// The pack on disk carries no tokenizer, so this stops at ids: `400 370 452 ...` is the sentence, and comparing ids is a
// stricter check than comparing text (two ids can share a detokenisation, and neither can share an id).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "audio_exec.h"
#include "audio_mel.h"
#include "onnx_graph.h"

namespace {

std::vector<float> read_wav_16k_mono(const char* path, std::string* why) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        *why = std::string("cannot open ") + path;
        return {};
    }
    std::vector<uint8_t> raw;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) raw.insert(raw.end(), buf, buf + n);
    std::fclose(f);
    if (raw.size() < 44 || std::memcmp(raw.data(), "RIFF", 4) != 0 || std::memcmp(raw.data() + 8, "WAVE", 4) != 0) {
        *why = "not a RIFF/WAVE file";
        return {};
    }
    size_t pos = 12;
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* pcm = nullptr;
    size_t pcm_len = 0;
    while (pos + 8 <= raw.size()) {
        const uint32_t len = uint32_t(raw[pos + 4]) | (uint32_t(raw[pos + 5]) << 8) | (uint32_t(raw[pos + 6]) << 16) |
                             (uint32_t(raw[pos + 7]) << 24);
        const size_t body = pos + 8;
        if (std::memcmp(raw.data() + pos, "fmt ", 4) == 0 && body + 16 <= raw.size()) {
            channels = uint16_t(raw[body + 2] | (raw[body + 3] << 8));
            rate = uint32_t(raw[body + 4]) | (uint32_t(raw[body + 5]) << 8) | (uint32_t(raw[body + 6]) << 16) |
                   (uint32_t(raw[body + 7]) << 24);
            bits = uint16_t(raw[body + 14] | (raw[body + 15] << 8));
        } else if (std::memcmp(raw.data() + pos, "data", 4) == 0) {
            pcm = raw.data() + body;
            pcm_len = std::min<size_t>(len, raw.size() - body);
        }
        pos = body + len + (len & 1);
    }
    if (pcm == nullptr || channels == 0) {
        *why = "no data chunk";
        return {};
    }
    if (rate != 16000) {
        *why = "expected 16 kHz, got " + std::to_string(rate);
        return {};
    }
    if (bits != 16) {
        *why = "expected 16-bit PCM, got " + std::to_string(bits) + "-bit";
        return {};
    }
    const size_t frames = pcm_len / (2 * size_t(channels));
    std::vector<float> out(frames);
    for (size_t i = 0; i < frames; ++i) {
        int32_t acc = 0;
        for (uint16_t c = 0; c < channels; ++c) {
            const uint8_t* s = pcm + (i * channels + c) * 2;
            acc += int16_t(uint16_t(s[0] | (s[1] << 8)));
        }
        out[i] = float(double(acc) / double(channels) / 32768.0);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <encoder.onnx> <decoder.onnx> <file.wav> [expected-id ...]\n", argv[0]);
        return 2;
    }
    using namespace onebit::npu::pack_import;

    std::string why;
    // The spans the last line reports, each named for what it contains and nothing else: wav (reading the file), mel
    // (the log-mel front-end), load (parsing both graphs), encoder, decode. `wall` is their sum, from before the file
    // is opened.
    const auto t_begin = std::chrono::steady_clock::now();
    std::vector<float> audio = read_wav_16k_mono(argv[3], &why);
    if (audio.empty()) {
        std::fprintf(stderr, "wav: %s\n", why.c_str());
        return 1;
    }
    // Whisper's window is a fixed 30 s: pad with silence, do not scale.
    onebit::audio::MelConfig cfg;
    const size_t want = size_t(cfg.sample_rate) * size_t(cfg.frames) * size_t(cfg.hop) / size_t(cfg.frames);
    audio.resize(std::max(audio.size(), want), 0.0f);

    const auto t_start = std::chrono::steady_clock::now();
    std::vector<float> mel;
    if (!onebit::audio::log_mel_spectrogram(audio.data(), audio.size(), cfg, &mel)) {
        std::fprintf(stderr, "mel: refused\n");
        return 1;
    }
    const auto t_mel = std::chrono::steady_clock::now();
    OnnxGraphInfo enc, dec;
    try {
        enc = parse_onnx_graph(argv[1]);
        dec = parse_onnx_graph(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "graph: %s\n", e.what());
        return 1;
    }
    const auto t_load = std::chrono::steady_clock::now();
    AudioRunResult xa;
    if (!run_audio_graph(enc, mel.data(), cfg.bins, cfg.frames, "", &xa, &why)) {
        std::fprintf(stderr, "encoder: %s\n", why.c_str());
        return 1;
    }
    const auto t_enc = std::chrono::steady_clock::now();
    std::printf("encoder: [");
    for (int64_t d : xa.dims) std::printf(" %lld", (long long)d);
    std::printf(" ]\n");

    // <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|> -- then greedy, reading the logits of the last real
    // prompt position (HF pads the 448/128-token window with <|endoftext|>, so the prompt is a prefix, not a suffix).
    const int64_t kSot = 50258, kEn = 50259, kTranscribe = 50359, kNoTimestamps = 50363, kEot = 50257;
    const int window = 128;
    std::vector<int64_t> seq = {kSot, kEn, kTranscribe, kNoTimestamps};
    std::vector<int64_t> ids;
    for (int step = 0; step < 60; ++step) {
        std::vector<int64_t> toks(size_t(window), kEot);
        for (size_t i = 0; i < seq.size() && i < size_t(window); ++i) toks[i] = seq[i];
        AudioRunResult y;
        if (!run_decoder_graph(dec, toks.data(), window, xa.data.data(), xa.dims, "", &y, &why)) {
            std::fprintf(stderr, "decoder at step %d: %s\n", step, why.c_str());
            return 1;
        }
        const int64_t vocab = y.dims.empty() ? 0 : y.dims.back();
        const size_t row = (seq.size() - 1) * size_t(vocab);
        int64_t best = 0;
        for (int64_t v = 1; v < vocab; ++v)
            if (y.data[row + size_t(v)] > y.data[row + size_t(best)]) best = v;
        if (best == kEot || seq.size() >= size_t(window)) break;
        ids.push_back(best);
        seq.push_back(best);
    }
    const auto t_dec = std::chrono::steady_clock::now();
    const auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    std::printf("time: wav %.3f s, mel %.3f s, load %.3f s, encoder %.3f s, decode %.3f s for %zu token(s) -> %.3f s/token, "
                "wall %.3f s\n",
                secs(t_begin, t_start), secs(t_start, t_mel), secs(t_mel, t_load), secs(t_load, t_enc), secs(t_enc, t_dec),
                ids.size(), ids.empty() ? 0.0 : secs(t_enc, t_dec) / double(ids.size()), secs(t_begin, t_dec));
    std::printf("ids:");
    for (int64_t v : ids) std::printf(" %lld", (long long)v);
    std::printf("\n");

    if (argc > 4) {
        int bad = 0;
        const int n = argc - 4;
        if (int(ids.size()) != n) {
            std::printf("FAIL length: got %zu want %d\n", ids.size(), n);
            ++bad;
        } else {
            for (int i = 0; i < n; ++i)
                if (ids[size_t(i)] != std::atoll(argv[4 + i])) {
                    std::printf("FAIL id[%d]: got %lld want %s\n", i, (long long)ids[size_t(i)], argv[4 + i]);
                    ++bad;
                }
        }
        if (bad == 0) std::printf("transcript ids match the reference decode\n");
        return bad == 0 ? 0 : 1;
    }
    return 0;
}
