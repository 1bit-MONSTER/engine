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
// audio_pack.cpp -- see audio_pack.h. The discovery is deliberately boring: known names, in a fixed order, and a refusal that
// names what is missing. The one interesting step is the mel-bin count, which comes from the encoder graph rather than a config.

#include "audio_pack.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "onnx_graph.h"

namespace fs = std::filesystem;

namespace onebit::audio {
namespace {

bool is_file(const std::string& p) {
    std::error_code ec;
    return fs::is_regular_file(fs::path(p), ec);
}

// The names exporters actually use, most specific first. A pack that carries both spellings is a pack someone assembled by
// hand; the first match wins and the other is ignored, which is why `dir` is recorded.
const char* kEncoderNames[] = {"encoder_model.onnx", "encoder.onnx"};
const char* kDecoderNames[] = {"decoder_model.onnx", "decoder.onnx"};
const char* kTokenizerNames[] = {"tokenizer.json", "tokenizer_config.json"};

int mel_bins_from_encoder(const std::string& path, std::string* why) {
    using namespace onebit::npu::pack_import;
    OnnxGraphInfo g;
    try {
        g = parse_onnx_graph(path);
    } catch (const std::exception& e) {
        if (why != nullptr) *why = e.what();
        return 0;
    }
    // The mel front-end is the graph's first Conv; its input-channel count is the bin count the encoder was exported for.
    for (const auto& n : g.nodes) {
        if (n.op != "Conv" || n.ins.size() < 2) continue;
        for (const auto& ti : g.init_tensors) {
            if (ti.name != n.ins[1] || ti.dims.size() != 3) continue;
            return int(ti.dims[1]);
        }
    }
    if (why != nullptr) *why = "no Conv with a weight tensor found, so the mel-bin count cannot be derived";
    return 0;
}

}  // namespace

bool discover_audio_pack(const std::string& dir, AudioPack* out, std::string* why) {
    if (out == nullptr) return false;
    *out = AudioPack{};
    out->dir = dir;
    std::error_code ec;
    if (!fs::is_directory(fs::path(dir), ec)) {
        if (why != nullptr) *why = dir + ": not a directory";
        return false;
    }
    for (const char* name : kEncoderNames) {
        const std::string p = dir + "/" + name;
        if (is_file(p)) { out->encoder = p; break; }
    }
    for (const char* name : kDecoderNames) {
        const std::string p = dir + "/" + name;
        if (is_file(p)) { out->decoder = p; break; }
    }
    for (const char* name : kTokenizerNames) {
        const std::string p = dir + "/" + name;
        if (is_file(p)) { out->tokenizer_json = p; break; }
    }
    // A pack an operator actually has may name its graphs after the model (large_v3_encoder.onnx), so after the exact names a
    // scan is allowed -- but it never picks a `with_past` decoder, which is a different graph with a different input list, and
    // whatever it picks is reported as a concern so the choice is visible rather than silent.
    if (out->encoder.empty() || out->decoder.empty()) {
        std::vector<std::string> candidates;
        for (const auto& entry : fs::directory_iterator(fs::path(dir), ec)) {
            if (!entry.is_regular_file(ec)) continue;
            const std::string name = entry.path().filename().string();
            if (name.size() < 6 || name.compare(name.size() - 5, 5, ".onnx") != 0) continue;
            candidates.push_back(name);
        }
        std::sort(candidates.begin(), candidates.end());
        const auto pick = [&](const char* needle, bool exclude_with_past) -> std::string {
            for (const auto& name : candidates) {
                if (name.find(needle) == std::string::npos) continue;
                if (exclude_with_past && name.find("with_past") != std::string::npos) continue;
                return dir + "/" + name;
            }
            return std::string();
        };
        if (out->encoder.empty()) {
            out->encoder = pick("encoder", false);
            if (!out->encoder.empty()) {
                out->concern += "encoder graph taken from its name: " + out->encoder;
            }
        }
        if (out->decoder.empty()) {
            out->decoder = pick("decoder", true);
            if (!out->decoder.empty()) {
                out->concern += (out->concern.empty() ? "" : "; ") + std::string("decoder graph taken from its name: ") + out->decoder;
            }
        }
    }
    if (out->encoder.empty()) out->missing.push_back("an encoder graph (encoder_model.onnx, encoder.onnx, or a *encoder*.onnx)");
    if (out->decoder.empty()) out->missing.push_back("a decoder graph (decoder_model.onnx, decoder.onnx, or a *decoder*.onnx without with_past)");
    if (!out->missing.empty()) {
        if (why != nullptr) *why = dir + ": missing " + out->missing.front();
        return false;
    }

    std::string graph_why;
    const int from_graph = mel_bins_from_encoder(out->encoder, &graph_why);
    if (from_graph > 0) {
        out->mel_bins = from_graph;
        out->mel_bins_source = "encoder graph";
    } else {
        out->concern += (out->concern.empty() ? std::string() : std::string("; ")) +
                        std::string("the encoder graph could not be read for its mel-bin count (") + graph_why + ")";
    }

    // config.json is a cross-check, never the authority: a disagreement is reported, and the graph wins because it is what the
    // weights were exported with.
    const std::string cfg_path = dir + "/config.json";
    if (is_file(cfg_path)) {
        const std::string text = [&] {
            std::string body;
            FILE* f = std::fopen(cfg_path.c_str(), "rb");
            if (f == nullptr) return body;
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) body.append(buf, n);
            std::fclose(f);
            return body;
        }();
        const auto number_after = [&](const char* key) -> int {
            const std::string k = std::string("\"") + key + "\"";
            const size_t at = text.find(k);
            if (at == std::string::npos) return 0;
            const size_t colon = text.find(':', at);
            if (colon == std::string::npos) return 0;
            return std::atoi(text.c_str() + colon + 1);
        };
        const int cfg_bins = number_after("num_mel_bins");
        if (cfg_bins > 0) {
            if (out->mel_bins == 0) {
                out->mel_bins = cfg_bins;
                out->mel_bins_source = "config.json";
            } else if (cfg_bins != out->mel_bins) {
                out->concern += (out->concern.empty() ? "" : "; ") + std::string("config.json says ") +
                                std::to_string(cfg_bins) + " mel bins but the encoder graph's Conv takes " +
                                std::to_string(out->mel_bins) + " (the graph is what the weights were exported with)";
            }
        }
        const int sr = number_after("sampling_rate");
        if (sr > 0) out->sample_rate = sr;
        const int fft = number_after("n_fft");
        if (fft > 0) out->n_fft = fft;
        const int hop = number_after("hop_length");
        if (hop > 0) out->hop = hop;
        const int frames = number_after("nb_max_frames");
        if (frames > 0) out->frames = frames;
    }
    if (out->mel_bins == 0) {
        if (why != nullptr) {
            *why = dir + ": the mel-bin count is in neither the encoder graph nor config.json, and guessing it is how a 128-bin "
                         "window reaches an 80-bin encoder";
        }
        return false;
    }
    return true;
}

}  // namespace onebit::audio
