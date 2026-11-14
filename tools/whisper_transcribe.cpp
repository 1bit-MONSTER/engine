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
// whisper_transcribe_npu is this file built with ONEBIT_WHISPER_NPU (engine#1783, the encoder route). With
// ONEBIT_WHISPER_NPU_KERNELS=<dir> in the environment, the encoder's projections and Conv stems run on the NPU through
// NpuTarget::gemv, and the tool prints `device_gemm=<X> s + host_remainder=<Y> s` for the encoder pass. Without the variable
// the binary is the host tool. The kernel ELFs are generated at one M; ONEBIT_WHISPER_NPU_ROWS names it (default 1536).
//
// The pack on disk carries no tokenizer, so this stops at ids: `400 370 452 ...` is the sentence, and comparing ids is a
// stricter check than comparing text (two ids can share a detokenisation, and neither can share an id).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "whisper_args.h"
#include "audio_exec.h"
#ifdef ONEBIT_WHISPER_NPU
#include "whisper_npu_device.h"
#endif
#include "audio_pack.h"
#include "whisper_ids.h"
#include "audio_parallel.h"
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
    // The scan stops at the array's closing bracket. Running to the END of the file (as this did) reads every digit after
    // the key, including numbers in later keys and any mention of the key in a prose note.
    size_t end = text.find('[', key);
    if (end == std::string::npos) {
        *why = std::string(path) + ": \"greedy_ids\" is not followed by a list";
        return false;
    }
    int depth = 0;
    for (size_t i = end; i < text.size(); ++i) {
        if (text[i] == '[') ++depth;
        else if (text[i] == ']' && --depth == 0) { end = i; break; }
    }
    for (size_t i = key; i < end;) {
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
    onebit::audio::MelConfig cfg;
    // Where the wav is: argv[2] for the pack form, argv[3] for the two-graph form. Decided once, in tools/whisper_args.h, so
    // that a test can hold it -- the old sentinel `1` was never converted and made the two-graph form read the encoder graph.
    const int audio_arg_index = audio_arg_index_for(argv[1]);
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
    // engine#1783: a recording longer than one window is cut into consecutive 30 s windows, each padded with silence to 30 s
    // and turned into its OWN log-mel. That is what Whisper's feature extractor does to one window, and it matters: the
    // log-mel is normalised against its own maximum, so a mel computed over the whole recording and cut afterwards is a
    // different input (measured on fx-5min's first window: max 8.5e-3 from WhisperFeatureExtractor that way, 1e-5 per window).
    // Each window is decoded with the same prompt and no carried text, which is the plain chunked form; there are no
    // timestamps, so nothing is stitched across a boundary.
    const size_t want = size_t(cfg.sample_rate) * 30;
    std::vector<float> audio = read_wav_16k_mono(argv[audio_arg_index], &why);
    if (audio.empty()) {
        std::fprintf(stderr, "wav: %s\n", why.c_str());
        return 1;
    }
    const size_t n_windows = std::max<size_t>(1, (audio.size() + want - 1) / want);
    const double audio_s = double(audio.size()) / double(cfg.sample_rate);

    const auto t_start = std::chrono::steady_clock::now();
    const auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    OnnxGraphInfo enc, dec;
    try {
        enc = parse_onnx_graph(enc_path);
        dec = parse_onnx_graph(dec_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "graph: %s\n", e.what());
        return 1;
    }
    AudioGemm* dev_hook = nullptr;
#ifdef ONEBIT_WHISPER_NPU
    whisper_npu::Device npu_dev;
    if (const char* kdir = std::getenv("ONEBIT_WHISPER_NPU_KERNELS")) {
        const char* rows_env = std::getenv("ONEBIT_WHISPER_NPU_ROWS");
        const int rows = rows_env != nullptr ? std::atoi(rows_env) : 1536;
        if (!whisper_npu::open(enc, kdir, rows, &npu_dev, &why)) {
            std::fprintf(stderr, "npu: %s\n", why.c_str());
            return 1;
        }
        for (const auto& m : npu_dev.missing) std::fprintf(stderr, "npu: no ELF for staged %s -- its products run on the host\n", m.c_str());
        dev_hook = &npu_dev.hook;
    }
#endif
    const auto t_load = std::chrono::steady_clock::now();

    // <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|> -- then greedy, reading the logits of the last real
    // prompt position (HF pads the 448/128-token window with <|endoftext|>, so the prompt is a prefix, not a suffix).
    // The token window is the DECODER's declared input length, not a constant: whisper-small exports 128, large-v3 448,
    // and a 128-wide window against a 448-wide graph builds a mask the scores cannot broadcast against (measured).
    int window = 128;
    if (!dec.input_dims.empty() && dec.input_dims[0].size() == 2 && dec.input_dims[0][1] > 0) {
        window = int(dec.input_dims[0][1]);
        std::printf("decoder window: %d positions (from the graph's input shape)\n", window);
    }

    // Whisper's special ids are a property of the VOCABULARY, and the two packs here do not share one: whisper-small exports
    // 51865 logits, large-v3 exports 51866 (v3 added a token), which shifts <|translate|>/<|transcribe|>/<|nospeech|>/
    // <|notimestamps|> by one. Hard-coding either set is what produced the first version of this tool: 50359/50363 decode
    // whisper-small correctly *because English audio makes <|translate|> and <|transcribe|> agree*, and produce <|endoftext|>
    // at the first step on large-v3 (measured: argmax 50257 with 50359/50363, 400 with 50360/50364, same pack and input).
    // Neither pack ships a tokenizer or generation_config, so the vocabulary width is the only fact available -- and it is the
    // right one, because the ids are offsets into it.
    // The decoder's declared output shape is [1, window, vocab]; the vocabulary is its last dimension. (A scan over the MatMul
    // weights cannot find it: the export applies the head through a Transpose, so the widest weight is the MLP's 4h.)
    int64_t vocab = 0;
    if (!dec.output_dims.empty() && !dec.output_dims[0].empty()) vocab = dec.output_dims[0].back();

    // The ids come from the pack's own vocabulary width (npu/forward/include/whisper_ids.h), for the reason recorded there.
    onebit::audio::WhisperIds slot;
    if (!onebit::audio::whisper_ids_for_vocab(vocab, &slot)) {
        std::fprintf(stderr, "special ids: no measured layout for a vocabulary of %lld\n", (long long)vocab);
        return 1;
    }
    // Say what the split is: a rate quoted without it is unreadable, because the default follows the CPUs this process is
    // ALLOWED (a thermal-run scope allows far fewer than the machine has) and the executor is only faster than serial
    // when the threads actually fit (engine#1783).
    std::printf("threads: %zu\n", onebit::npu::pack_import::audio_thread_count());
    std::printf("special ids: vocab %lld -> transcribe %lld, notimestamps %lld\n", (long long)vocab,
                (long long)slot.transcribe, (long long)slot.no_timestamps);
    const int64_t kEot = slot.eot, kSot = slot.sot, kEn = slot.lang_en;
    const int64_t kTranscribe = slot.transcribe, kNoTimestamps = slot.no_timestamps;
    const std::vector<int64_t> prompt = {kSot, kEn, kTranscribe, kNoTimestamps};
    std::printf("audio: %.2f s -> %zu window(s) of 30 s\n", audio_s, n_windows);

    std::vector<int64_t> ids;
    double mel_s = 0.0, enc_s = 0.0, dec_s = 0.0, ttft_s = 0.0, ttft_mel_s = 0.0, ttft_enc_s = 0.0;
    for (size_t w = 0; w < n_windows; ++w) {
        const auto tw0 = std::chrono::steady_clock::now();
        const size_t from = w * want;
        const size_t to = std::min(audio.size(), from + want);
        std::vector<float> chunk(audio.begin() + std::ptrdiff_t(from), audio.begin() + std::ptrdiff_t(to));
        chunk.resize(want, 0.0f);
        std::vector<float> mel;
        if (!onebit::audio::log_mel_spectrogram(chunk.data(), chunk.size(), cfg, &mel)) {
            std::fprintf(stderr, "mel: refused (window %zu)\n", w);
            return 1;
        }
        const auto tw_mel = std::chrono::steady_clock::now();
        AudioRunResult xa;
#ifdef ONEBIT_WHISPER_NPU
        const double dev_ms0 = dev_hook != nullptr ? npu_dev.gemm->device_ms() : 0.0;
#endif
        // engine#2399: ONEBIT_AUDIO_NODE_TIMES=1 prints per-node lines for the whole run, which is encoder PLUS a
        // 25-token decode -- and the decode's GEMMs are M=1, so their MatMul rows swamp the encoder's. This marker
        // is what makes the per-node output attributable: everything between here and the decoder marker is the
        // encoder, which is the pass this issue is about.
        const bool node_times = std::getenv("ONEBIT_AUDIO_NODE_TIMES") != nullptr;
        if (node_times) {
            std::fprintf(stdout, "[audio-graph] encoder\n");
            std::fflush(stdout);   // the node lines go to stderr; without this they interleave after the marker
        }
        if (!run_audio_graph(enc, mel.data(), cfg.bins, cfg.frames, "", &xa, &why, dev_hook)) {
            std::fprintf(stderr, "encoder (window %zu): %s\n", w, why.c_str());
            return 1;
        }
        const auto tw_enc = std::chrono::steady_clock::now();
#ifdef ONEBIT_WHISPER_NPU
        if (dev_hook != nullptr) {
            // The split the issue is scored on, from this run: the wall inside gemv against everything else in this window's
            // encoder pass. device_ms() accumulates over the process, so a window's share is the difference.
            const double enc_ms = std::chrono::duration<double, std::milli>(tw_enc - tw_mel).count();
            const double dev_ms = npu_dev.gemm->device_ms() - dev_ms0;
            std::printf("encoder split: device_gemm=%.3f s + host_remainder=%.3f s per window (window %zu; %d product(s) on the "
                        "device in %d launch(es) so far, %d on the host)\n",
                        dev_ms / 1000.0, (enc_ms - dev_ms) / 1000.0, w, dev_hook->device, npu_dev.gemm->launches(),
                        dev_hook->host);
            if (w == 0)
                for (const auto& kv : dev_hook->host_why) std::printf("  host: %d x %s\n", kv.second, kv.first.c_str());
        }
#endif
        if (w == 0) {
            std::printf("encoder: [");
            for (int64_t d : xa.dims) std::printf(" %lld", (long long)d);
            std::printf(" ]\n");
        }
        // The loop runs until <|endoftext|> or until the token window is full. It used to stop after 60 steps, which a
        // 30 s window of dense speech outruns: fx-5min's first window needs 66 tokens, and the tail was dropped with no
        // message (engine#1783).
        std::vector<int64_t> seq = prompt;
        size_t got = 0;
        if (node_times) {
            // engine#2399: closes the encoder region and names the rest. Everything after this is the greedy decode,
            // whose GEMMs are M=1 and must not be counted against the encoder's MatMul share.
            std::fprintf(stdout, "[audio-graph] decoder\n");
            std::fflush(stdout);
        }
        bool full = false;
        while (true) {
            if (seq.size() >= size_t(window)) { full = true; break; }
            std::vector<int64_t> toks(size_t(window), kEot);
            for (size_t i = 0; i < seq.size(); ++i) toks[i] = seq[i];
            AudioRunResult y;
            if (!run_decoder_graph(dec, toks.data(), window, xa.data.data(), xa.dims, "", &y, &why)) {
                std::fprintf(stderr, "decoder at window %zu step %zu: %s\n", w, got, why.c_str());
                return 1;
            }
            const int64_t out_vocab = y.dims.empty() ? 0 : y.dims.back();
            const size_t row = (seq.size() - 1) * size_t(out_vocab);
            int64_t best = 0;
            for (int64_t v = 1; v < out_vocab; ++v)
                if (y.data[row + size_t(v)] > y.data[row + size_t(best)]) best = v;
            if (best == kEot) break;
            ids.push_back(best);
            seq.push_back(best);
            ++got;
        }
        const auto tw_dec = std::chrono::steady_clock::now();
        mel_s += secs(tw0, tw_mel);
        enc_s += secs(tw_mel, tw_enc);
        dec_s += secs(tw_enc, tw_dec);
        if (w == 0) {
            ttft_mel_s = secs(t_start, tw_mel);
            ttft_enc_s = secs(tw_mel, tw_enc);
            ttft_s = secs(t_start, tw_enc);
        }
        if (n_windows > 1 || full)
            std::printf("window %zu: mel %.3f s, encoder %.3f s, decode %.3f s for %zu token(s)%s\n", w, secs(tw0, tw_mel),
                        secs(tw_mel, tw_enc), secs(tw_enc, tw_dec), got,
                        full ? " -- the token window filled before <|endoftext|>" : "");
    }
    const auto t_dec = std::chrono::steady_clock::now();
    const double load_s = secs(t_start, t_load);
    std::printf("time: mel+load %.3f s, encoder %.3f s, decode %.3f s for %zu token(s) -> %.3f s/token, wall %.3f s\n",
                mel_s + load_s, enc_s, dec_s, ids.size(), ids.empty() ? 0.0 : dec_s / double(ids.size()),
                secs(t_start, t_dec));
    if (n_windows > 1)
        std::printf("per window: encoder %.3f s, wall %.3f s; recording %.2f s in %.3f s (RTF %.3f)\n",
                    enc_s / double(n_windows), secs(t_load, t_dec) / double(n_windows), audio_s, secs(t_start, t_dec),
                    secs(t_start, t_dec) / audio_s);
    // The row's completion asks for "s/window AND time to first token", and TTFT for this chain is everything before the first
    // decoded token: the graph load, the first window's mel front-end and its encoder pass. Named rather than left to be derived
    // from the line above, because "which of those three is the TTFT" is exactly the question a reader gets wrong at 2 a.m.
    std::printf("ttft: %.3f s (load+mel %.3f + encoder %.3f; the first token needs no extra forward)\n", ttft_s, ttft_mel_s,
                ttft_enc_s);
    std::printf("ids:");
    for (int64_t v : ids) std::printf(" %lld", (long long)v);
    std::printf("\n");

    // The expectation is either the ids themselves (one per argument) or, when a single .json path is given, that file's
    // `greedy_ids`. A path used to be compared as a literal id, which reported "length: got 25 want 1" -- true, and useless:
    // it said nothing about whether the file was readable or held the list.
    std::vector<std::string> want_ids;
    if (argc > audio_arg_index + 1) {
        const std::string first = argv[audio_arg_index + 1];
        if (argc == audio_arg_index + 2 && first.size() > 5 && first.compare(first.size() - 5, 5, ".json") == 0) {
            std::string ids_why;
            if (!read_ids_file(first.c_str(), &want_ids, &ids_why)) {
                std::fprintf(stderr, "ids: %s\n", ids_why.c_str());
                return 1;
            }
        } else {
            for (int i = audio_arg_index + 1; i < argc; ++i) want_ids.emplace_back(argv[i]);
        }
    }
    if (!want_ids.empty()) {
        int bad = 0;
        if (ids.size() != want_ids.size()) {
            std::printf("FAIL length: got %zu want %zu\n", ids.size(), want_ids.size());
            ++bad;
        } else {
            for (size_t i = 0; i < want_ids.size(); ++i)
                if (ids[i] != std::atoll(want_ids[i].c_str())) {
                    std::printf("FAIL id[%zu]: got %lld want %s\n", i, (long long)ids[i], want_ids[i].c_str());
                    ++bad;
                }
        }
        if (bad == 0) std::printf("transcript ids match the reference decode\n");
        return bad == 0 ? 0 : 1;
    }
    return 0;
}
