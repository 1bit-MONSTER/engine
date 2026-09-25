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

// laya/safetensors.h — minimal .safetensors weight reader (F32/F16/BF16 -> f32).
//
// The container is the same one npu::Model reads (8-byte little-endian header
// length, a JSON tensor table, then the raw data section), but here the reader
// is weight-oriented: tensors are fetched by Hugging Face name and dequantized
// to f32 in one call. Laya's model.safetensors is 205 F16 tensors + one F32
// (temperature), so those are the paths that matter.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::laya {

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;
};

class Safetensors {
public:
    Safetensors() = default;
    ~Safetensors();
    Safetensors(const Safetensors&) = delete;
    Safetensors& operator=(const Safetensors&) = delete;

    // Opens one .safetensors file and parses its tensor table.
    bool open(const std::string& path);

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    // Decode one tensor into f32. Returns false if absent or dtype unsupported.
    bool get_f32(const std::string& name, std::vector<float>& out) const;

    const std::string& error() const { return err_; }

private:
    const Tensor* find(const std::string& name) const;

    uint8_t* map_ = nullptr;
    size_t size_ = 0;
    std::map<std::string, Tensor> tensors_;
    std::string err_;
};

}  // namespace onebit::laya
