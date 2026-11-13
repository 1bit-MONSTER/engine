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
// tests/audio_pack_test.cpp -- audio pack discovery (npu/audio_pack.h, engine#1783). Fixture directories only: no model, no
// device, no network. The graph-derived arms need a real pack and skip (77) without ONEBIT_TEST_WHISPER.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "audio_pack.h"

namespace fs = std::filesystem;
namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) {
        std::printf("ok   %s\n", what.c_str());
    } else {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

void write(const fs::path& p, const std::string& body) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << body;
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "audio-pack-test";

    // 7. a pack whose graphs are named after the model (large_v3_encoder.onnx): found by scan, and the choice is REPORTED --
    //    and a `decoder_with_past` sibling is never picked, because it is a different graph with a different input list
    {
        const fs::path d = root / "model-named";
        write(d / "large_v3_encoder.onnx", "x");
        write(d / "large_v3_decoder_with_past.onnx", "x");
        write(d / "large_v3_decoder.onnx", "x");
        onebit::audio::AudioPack p;
        std::string why;
        // no bins anywhere: the graphs are not real, so discovery refuses on the bin count -- but the NAME choice must already
        // have been made, which is what the two checks below read out of the pack struct.
        onebit::audio::discover_audio_pack(d.string(), &p, &why);
        check(p.decoder.find("large_v3_decoder.onnx") != std::string::npos,
              "model-named: the plain decoder is chosen, not the with_past one");
        std::printf("   [debug] concern=<%s>\n", p.concern.c_str());
        check(p.concern.find("taken from its name") != std::string::npos,
              "model-named: the name-based choice is reported, not silent");
    }

    fs::remove_all(root);

    // 1. a complete pack, whisper-* spelling, with a config.json that states the bins
    {
        const fs::path d = root / "complete";
        write(d / "encoder_model.onnx", "not really a graph");
        write(d / "decoder_model.onnx", "not really a graph");
        write(d / "config.json", R"({"num_mel_bins": 80, "sampling_rate": 16000, "n_fft": 400, "hop_length": 160})");
        onebit::audio::AudioPack p;
        std::string why;
        const bool ok = onebit::audio::discover_audio_pack(d.string(), &p, &why);
        check(ok, "complete pack: discovered");
        check(p.encoder.find("encoder_model.onnx") != std::string::npos &&
                  p.decoder.find("decoder_model.onnx") != std::string::npos,
              "complete pack: both graphs found by their whisper names");
        check(p.mel_bins == 80 && p.mel_bins_source == "config.json",
              "complete pack: the bins come from config.json when the graph cannot be read");
        check(!p.concern.empty(), "complete pack: the unreadable graph is reported as a concern, not a refusal");
    }

    // 2. our own spelling (encoder.onnx / decoder.onnx) is found too
    {
        const fs::path d = root / "own-names";
        write(d / "encoder.onnx", "x");
        write(d / "decoder.onnx", "x");
        write(d / "config.json", R"({"num_mel_bins": 128})");
        onebit::audio::AudioPack p;
        std::string why;
        check(onebit::audio::discover_audio_pack(d.string(), &p, &why), "own names: discovered");
        check(p.mel_bins == 128, "own names: 128 bins read from config.json");
    }

    // 3. a missing graph is refused, by name, and says which one
    {
        const fs::path d = root / "no-decoder";
        write(d / "encoder_model.onnx", "x");
        write(d / "config.json", R"({"num_mel_bins": 80})");
        onebit::audio::AudioPack p;
        std::string why;
        check(!onebit::audio::discover_audio_pack(d.string(), &p, &why), "no decoder: refused");
        check(p.missing.size() == 1 && p.missing[0].find("decoder") != std::string::npos,
              "no decoder: the missing piece is named");
        check(why.find("missing") != std::string::npos, "no decoder: the reason says missing");
    }

    // 4. no mel-bin count anywhere: refused rather than guessed -- this is the run-that-answers-wrongly case
    {
        const fs::path d = root / "no-bins";
        write(d / "encoder_model.onnx", "x");
        write(d / "decoder_model.onnx", "x");
        onebit::audio::AudioPack p;
        std::string why;
        check(!onebit::audio::discover_audio_pack(d.string(), &p, &why), "no bin count: refused");
        check(why.find("guessing") != std::string::npos, "no bin count: the refusal says why guessing is not done");
    }

    // 5. not a directory at all
    {
        onebit::audio::AudioPack p;
        std::string why;
        check(!onebit::audio::discover_audio_pack((root / "nope").string(), &p, &why), "absent directory: refused");
        check(why.find("not a directory") != std::string::npos, "absent directory: said plainly");
    }

    // 6. a real pack, when one is named: the bins come from the encoder's own Conv
    {
        const char* pack = std::getenv("ONEBIT_TEST_WHISPER_PACK");
        if (pack == nullptr || *pack == '\0') {
            std::printf("skip a real pack: set ONEBIT_TEST_WHISPER_PACK to a pack directory\n");
        } else {
            onebit::audio::AudioPack p;
            std::string why;
            check(onebit::audio::discover_audio_pack(pack, &p, &why), std::string("real pack: discovered (") + why + ")");
            check(p.mel_bins > 0 && p.mel_bins_source == "encoder graph",
                  "real pack: the mel bins come from the encoder graph, not from a config field");
        }
    }

    fs::remove_all(root);
    if (failures != 0) {
        std::printf("audio pack: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("audio pack: all cases ok\n");
    return 0;
}
