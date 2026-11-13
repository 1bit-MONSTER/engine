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
// audio_pack.h -- what an audio pack is, discovered rather than configured (engine#1783).
//
// Until now the host chain was driven by two file paths on a command line, which is enough for a bench and wrong for a route:
// a pack directory is what an operator has, and the two graphs inside it are named by whoever exported it. This finds them,
// derives the front-end's mel-bin count from the encoder graph's own first Conv instead of trusting a config field, and
// refuses by name when a pack is not one.
//
// The mel-bin derivation matters: whisper-small exports 80 bins and large-v3 exports 128, and feeding a 128-bin window to an
// 80-bin encoder produces a graph that runs and answers wrongly -- the failure mode this whole tree has spent the night
// removing. The Conv's input-channel count is the pack's own statement; a `config.json` is only consulted when the graph
// cannot be read, and a disagreement between the two is reported rather than resolved silently.

#pragma once

#include <string>
#include <vector>

namespace onebit::audio {

struct AudioPack {
    std::string dir;
    std::string encoder;          // absolute paths, empty when absent
    std::string decoder;
    std::string tokenizer_json;
    int mel_bins = 0;             // from the encoder's first Conv; 0 when it could not be derived
    std::string mel_bins_source;  // "encoder graph" | "config.json" | "" -- so a caller can say where the number came from
    int sample_rate = 16000;      // the front-end's defaults, overridable by config.json
    int n_fft = 400;
    int hop = 160;
    int frames = 3000;            // the fixed 30 s window the encoder was exported for
    std::vector<std::string> missing;   // what a complete pack needs and this one lacks, named
    std::string concern;          // a disagreement or an oddity worth printing, never a refusal on its own
};

// True when the directory yields both graphs. `why` carries the first concrete reason when it does not.
bool discover_audio_pack(const std::string& dir, AudioPack* out, std::string* why);

}  // namespace onebit::audio
