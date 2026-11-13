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
// tests/whisper_ids_test.cpp -- the special-token layout selection (npu/forward/include/whisper_ids.h, engine#1783).
//
// These are the two verdicts the tool got wrong before it read a pack: a hard-coded 50359/50363 decodes whisper-small and makes
// large-v3 answer <|endoftext|> at the first step. Both values here were measured against onnxruntime on the real packs.

#include <cstdio>
#include <string>

#include "whisper_ids.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main() {
    using onebit::audio::whisper_ids_for_vocab;
    using onebit::audio::WhisperIds;

    WhisperIds ids;
    check(whisper_ids_for_vocab(51865, &ids), "51865 (v1/v2/small/medium): a layout exists");
    check(ids.sot == 50258 && ids.lang_en == 50259 && ids.eot == 50257, "51865: sot/en/eot are unchanged across families");
    check(ids.transcribe == 50359 && ids.no_timestamps == 50363, "51865: transcribe 50359, notimestamps 50363");

    check(whisper_ids_for_vocab(51866, &ids), "51866 (large-v3): a layout exists");
    check(ids.transcribe == 50360 && ids.no_timestamps == 50364, "51866: both shift by one");

    // A larger derivative of the v3 line keeps v3's layout; the whole point is that the ids are offsets into the vocabulary.
    check(whisper_ids_for_vocab(51890, &ids) && ids.transcribe == 50360, "a wider v3 derivative keeps v3's layout");

    // A `.en` model (51864) is refused: its prompt differs and nobody measured it here. Guessing one of the two known layouts is
    // how a wrong prompt looks right on English audio.
    WhisperIds untouched;
    untouched.transcribe = -1;
    check(!whisper_ids_for_vocab(51864, &untouched), "51864 (.en): refused rather than guessed");
    check(untouched.transcribe == -1, "... and the caller's struct is left untouched");
    check(!whisper_ids_for_vocab(0, &untouched), "vocabulary 0: refused");
    check(!whisper_ids_for_vocab(51865, nullptr), "a null output is refused, not dereferenced");

    // The two families must not agree on transcribe: that difference is the defect this header exists to prevent.
    WhisperIds small, v3;
    whisper_ids_for_vocab(51865, &small);
    whisper_ids_for_vocab(51866, &v3);
    check(small.transcribe != v3.transcribe && small.no_timestamps != v3.no_timestamps,
          "the two families differ in both ids (a shared constant is the bug)");

    if (failures != 0) {
        std::printf("whisper ids: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("whisper ids: all cases ok\n");
    return 0;
}
