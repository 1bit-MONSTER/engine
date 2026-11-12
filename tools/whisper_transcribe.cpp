// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// whisper_transcribe.cpp -- the whole host pipeline: wav -> log-mel -> encoder -> greedy decode -> token ids.
//
//     ./whisper_transcribe <pack dir> <file.wav> [expected-ids-csv ...]
//     ./whisper_transcribe <encoder.onnx> <decoder.onnx> <file.wav> [expected-ids-csv ...]
//
// The pack form reads both graphs, and the mel-bin count, from the directory (npu/audio_pack.h): 80 bins for whisper-small and 128
// for large-v3, and passing the wrong one produces a graph that runs and answers wrongly, so the count comes from the encoder.
//
// A tool rather than a test because one forward pass of each graph is ~26 s at these shapes and a sentence is 25 of them
// (see PR #1931); `ctest` would time out for a reason that has nothing to do with correctness. The numbers it prints are
// the ones a test would assert, so `expected-ids-csv` turns it into a check when run by hand.
//
// The pack on disk carries no tokenizer, so this stops at ids: `400 370 452 ...` is the sentence, and comparing ids is a
// stricter check than comparing text (two ids can share a detokenisation, and neither can share an id).

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "audio_exec.h"
#include "audio_pack.h"
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


// The expected ids as a file (tests/data/whisper-jfk-ids.json), so a lane does not have to retype 25 numbers from a PR body. The
// parser is deliberately narrow: it finds "greedy_ids" and reads the integers that follow, and refuses a file with none.
static bool read_ids_file(const char* path, std::vector<std::string>* out, std::string* why) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        *why = std::string("cannot open ") + path;
        return false;
    }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    const size_t key = text.find("\"greedy_ids\"");
    if (key == std::string::npos) {
        *why = std::string(path) + " has no \"greedy_ids\" list";
        return false;
    }
    for (size_t i = key; i < text.size();) {
        if (std::isdigit(static_cast<unsigned char>(text[i])) || (text[i] == '-' && i + 1 < text.size())) {
            const size_t start = i;
            ++i;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
            out->emplace_back(text, start, i - start);
        } else {
            ++i;
        }
    }
    if (out->empty()) {
        *why = std::string(path) + ": the greedy_ids list is empty";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // ctest form: no arguments means "run the reference sentence from the environment", so the chain can be checked by CI or by
    // a lane without remembering the 25 ids. Missing inputs skip with 77, the same convention as every other gated test here.
    if (argc == 1) {
        static std::vector<std::string> from_env;
        const char* pack = std::getenv("ONEBIT_TEST_WHISPER_PACK");
        const char* wav = std::getenv("ONEBIT_TEST_WHISPER_WAV");
        const char* ids = std::getenv("ONEBIT_TEST_WHISPER_IDS");
        const char* ids_file = std::getenv("ONEBIT_TEST_WHISPER_IDS_FILE");
        if (pack == nullptr || wav == nullptr || (ids == nullptr && ids_file == nullptr)) {
            std::printf("SKIP: set ONEBIT_TEST_WHISPER_PACK and _WAV, and either _IDS or _IDS_FILE "
                        "(tests/data/whisper-jfk-ids.json)\n");
            return 77;
        }
        from_env.push_back(argv[0]);
        from_env.push_back(pack);
        from_env.push_back(wav);
        if (ids == nullptr) {
            std::string ids_why;
            if (!read_ids_file(ids_file, &from_env, &ids_why)) {
                std::fprintf(stderr, "ids: %s\n", ids_why.c_str());
                return 1;
            }
            ids = "";
        }
        for (size_t i = 0; ids != nullptr && i < std::strlen(ids);) {   // split on whitespace
            while (i < std::strlen(ids) && std::isspace(static_cast<unsigned char>(ids[i]))) ++i;
            const size_t start = i;
            while (i < std::strlen(ids) && !std::isspace(static_cast<unsigned char>(ids[i]))) ++i;
            if (i > start) from_env.emplace_back(ids + start, i - start);
        }
        std::vector<char*> argv_env;
        for (auto& s : from_env) argv_env.push_back(const_cast<char*>(s.c_str()));
        return main(static_cast<int>(argv_env.size()), argv_env.data());
    }
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <pack dir> <file.wav> [expected-id ...]\n"
                     "       %s <encoder.onnx> <decoder.onnx> <file.wav> [expected-id ...]\n"
                     "  The pack form reads the graphs, and the mel-bin count, from the directory (engine#1783).\n",
                     argv[0], argv[0]);
        return 2;
    }
    using namespace onebit::npu::pack_import;
    std::string enc_path = (argc > 1) ? argv[1] : "";
    std::string dec_path = (argc > 2) ? argv[2] : "";

    std::string why;
    // `1` means the two-graph form is in use and the wav is argv[3]; the pack form sets this to 2 after it discovers the graphs.
    int audio_arg_index = 1;
    onebit::audio::MelConfig cfg;
    // The pack form is decided by the first argument: a directory (or anything without `.onnx`) is a pack, and it also carries the
    // mel-bin count, which has to be known before the wav is turned into a spectrogram.
    if (argc >= 3 && std::string(argv[1]).find(".onnx") == std::string::npos) audio_arg_index = 2;
    if (audio_arg_index == 2) {
        const std::string pack_dir = argv[1];
        onebit::audio::AudioPack pack;
        if (!onebit::audio::discover_audio_pack(pack_dir, &pack, &why)) {
            std::fprintf(stderr, "pack: %s\n", why.c_str());
            return 1;
        }
        if (!pack.concern.empty()) std::fprintf(stderr, "pack note: %s\n", pack.concern.c_str());
        std::fprintf(stderr, "pack: %s (mel bins %d from the %s)\n", pack_dir.c_str(), pack.mel_bins,
                     pack.mel_bins_source.c_str());
        cfg.bins = pack.mel_bins;
        cfg.sample_rate = pack.sample_rate;
        cfg.n_fft = pack.n_fft;
        cfg.hop = pack.hop;
        cfg.frames = pack.frames;
        enc_path = pack.encoder;
        dec_path = pack.decoder;
    }
    const size_t want = size_t(cfg.sample_rate) * 30;
    std::vector<float> audio = read_wav_16k_mono(argv[audio_arg_index], &why);
    if (audio.empty()) {
        std::fprintf(stderr, "wav: %s\n", why.c_str());
        return 1;
    }
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
        enc = parse_onnx_graph(enc_path);
        dec = parse_onnx_graph(dec_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "graph: %s\n", e.what());
        return 1;
    }
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
    std::printf("time: mel+load %.3f s, encoder %.3f s, decode %.3f s for %zu token(s) -> %.3f s/token, wall %.3f s\n",
                secs(t_start, t_mel), secs(t_mel, t_enc), secs(t_enc, t_dec), ids.size(),
                ids.empty() ? 0.0 : secs(t_enc, t_dec) / double(ids.size()), secs(t_start, t_dec));
    std::printf("ids:");
    for (int64_t v : ids) std::printf(" %lld", (long long)v);
    std::printf("\n");

    if (argc > audio_arg_index + 1) {
        int bad = 0;
        const int n = argc - (audio_arg_index + 1);
        if (int(ids.size()) != n) {
            std::printf("FAIL length: got %zu want %d\n", ids.size(), n);
            ++bad;
        } else {
            for (int i = 0; i < n; ++i)
                if (ids[size_t(i)] != std::atoll(argv[audio_arg_index + 1 + i])) {
                    std::printf("FAIL id[%d]: got %lld want %s\n", i, (long long)ids[size_t(i)], argv[audio_arg_index + 1 + i]);
                    ++bad;
                }
        }
        if (bad == 0) std::printf("transcript ids match the reference decode\n");
        return bad == 0 ? 0 : 1;
    }
    return 0;
}
