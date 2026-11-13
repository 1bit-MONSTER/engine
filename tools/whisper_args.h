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
// whisper_args.h -- where the wav sits on whisper_transcribe's command line (engine#1783, engine#2183).
//
// Two forms are documented in the tool's own usage block:
//
//   whisper_transcribe <pack dir> <file.wav> ...                        the wav is argv[2]
//   whisper_transcribe <encoder.onnx> <decoder.onnx> <file.wav> ...     the wav is argv[3]
//
// The tool used to carry `int audio_arg_index = 1;` as a "sentinel" meaning "two-graph form", and nothing ever converted it,
// so the two-graph form read argv[1] -- the ENCODER GRAPH -- as the wav and answered "wav: not a RIFF/WAVE file" for the
// invocation its own --help prints. This function is that decision, in one place, so a test can hold it and a rebase cannot
// quietly lose it: the defect survived twelve heads of a PR that kept being regenerated from a base without the fix.
#pragma once

#include <string>

// The index of the wav among argv. `first` is argv[1]: anything carrying `.onnx` is a graph, which means the two-graph form.
inline int audio_arg_index_for(const std::string& first) {
    return first.find(".onnx") == std::string::npos ? 2 : 3;
}
