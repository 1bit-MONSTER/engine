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

// laya/encoder.h — another place for the scorer's encoder (its 28 ModernBERT layers): the scorer
// hands each sequence's embedded rows to a registered encoder and runs the layers on the CPU only
// when the encoder declines. The engine registers none; a private NPU add-on (docs/npu.md,
// "Private routes") registers "npu" from register_private_addon(), and `1bit serve --laya` with
// ONEBIT_LAYA_DEVICE=npu uses it (docs/laya.md).
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace onebit::laya {

// Runs the encoder layers on h [rows][1024] in place: n real tokens, the rows after them padding
// (masked as keys). Leaves h as the last layer's output, before the final norm. false: declined
// (h unchanged; the scorer runs the layers itself).
using EncoderFn = std::function<bool(float* h, int rows, int n)>;

// model_dir (the checkpoint Scorer::load read) -> an encoder; on failure an empty function and
// the reason in *err.
using EncoderFactory = std::function<EncoderFn(const std::string& model_dir, std::string* err)>;

// A later registration under the same name replaces the earlier one.
void register_encoder(const std::string& name, EncoderFactory factory);
// nullptr when nothing is registered under name.
const EncoderFactory* find_encoder(const std::string& name);
std::vector<std::string> encoder_names();

}  // namespace onebit::laya
