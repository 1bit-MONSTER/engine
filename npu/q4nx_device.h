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
// The device form of an imported Q4NX container (engine#1185).
//
// `1bit npu-pack import` writes the fp32 q4_1f chunk kind (6,144 B per 32x256 tile,
// npu/q4nx.h) so the pack's own scale/zero survive bit for bit. The device kernels stream the
// 5,120-byte bf16 q4_1 tile, so the forward's loader is handed a device-form directory instead:
// every q4_1f tensor repacked through q4nx::as_q4_1(), every other tensor and every other file in
// the model directory copied unchanged. The container on disk is never modified: the conversion
// lives in the Q4NX cache ($ONEBIT_Q4NX_CACHE, else ~/.cache/1bit/q4nx), keyed by the source
// directory and its model.q4nx size/mtime, and is reused by a later serve.
#pragma once

#include <cstdint>
#include <string>

namespace onebit::npu::q4nx {

// The directory the Q4NX cache lives in: $ONEBIT_Q4NX_CACHE, else ~/.cache/1bit/q4nx. The home
// directory comes from the user database when HOME is unset; with neither, this throws rather
// than falling back to a world-writable /tmp path a cached directory could be planted in.
std::string cache_root();

struct DeviceForm {
    std::string dir;         // the model directory holding the device-form model.q4nx
    std::string model;       // dir + "/model.q4nx"
    bool converted = false;  // false: the input already is in the device's form (dir is the input)
    bool reused = false;     // true: a previous conversion in the cache was reused
    long long tensors = 0;   // q4_1f tensors repacked (0 when nothing needed converting)
    long long tiles = 0;     // 5,120-byte device tiles they became
};

// The device form of the container file `model_path` (…/model.q4nx): every q4_1f I8 tensor
// repacked through as_q4_1() to the 5,120-byte device tile; only the converted model.q4nx, a
// config.json copy and a stamp are written (the rest of the model directory is still read from the
// source). `cache_root` may be empty to use cache_root(), which is resolved only when a conversion
// is actually needed; a test passes a temporary one. A container with no q4_1f tensor is returned
// as it is (converted = false) and nothing is written. Throws on I/O; logged once to stderr.
DeviceForm device_form(const std::string& model_path, const std::string& cache_root);

}  // namespace onebit::npu::q4nx
