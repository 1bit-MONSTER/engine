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
// whisper_ids.h -- the special-token ids a Whisper decoder's prompt needs, selected by VOCABULARY WIDTH (engine#1783).
//
// Why by width and not by a constant: whisper-small exports 51865 logits and large-v3 exports 51866, and the extra token shifts
// <|translate|>/<|transcribe|>/<|nospeech|>/<|notimestamps|> by one. A hard-coded 50359/50363 decodes whisper-small correctly --
// because English audio makes <|translate|> and <|transcribe|> agree -- and makes large-v3 answer <|endoftext|> at the first step.
// Measured on both packs against onnxruntime: argmax 50257 with 50359/50363 on large-v3, 400 with 50360/50364.
//
// Neither pack ships a tokenizer or generation_config, so the width (read from the decoder's declared output shape) is the only
// fact available. A `.en` model (51864) is NOT covered: its prompt differs and nobody has measured it here, so it refuses rather
// than guessing one of the two known layouts.

#pragma once

#include <cstdint>

namespace onebit::audio {

struct WhisperIds {
    int64_t sot = 50258;            // <|startoftranscript|>
    int64_t lang_en = 50259;        // <|en|>
    int64_t transcribe = 0;         // <|transcribe|>
    int64_t no_timestamps = 0;      // <|notimestamps|>
    int64_t eot = 50257;            // <|endoftext|>
};

// Returns false, leaving `out` untouched, for a vocabulary this header has no measured layout for.
inline bool whisper_ids_for_vocab(int64_t vocab, WhisperIds* out) {
    if (out == nullptr) return false;
    WhisperIds ids;
    if (vocab >= 51866) {          // large-v3 and its derivatives
        ids.transcribe = 50360;
        ids.no_timestamps = 50364;
    } else if (vocab == 51865) {   // v1/v2/small/medium, multilingual
        ids.transcribe = 50359;
        ids.no_timestamps = 50363;
    } else {
        return false;              // a `.en` vocabulary or an unknown family: refuse rather than guess
    }
    *out = ids;
    return true;
}

}  // namespace onebit::audio
