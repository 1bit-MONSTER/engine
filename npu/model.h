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

// A Q4NX model directory: model.q4nx (8-byte header length, JSON tensor table,
// tensor data, as in safetensors) and the Hugging Face config.json next to it.
// Projection weights are I8 tiles of 5120 bytes: 512 B bf16 scales, 512 B bf16
// zeros, 4096 B packed int4, covering 32 x 256 weights. A container imported from
// an OGA pack uses the fp32 form of the same tile instead (6144 B, npu/q4nx.h
// Q4_1F, engine#1185); the container's layout metadata says which form it holds.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::npu {

constexpr size_t kTileBytes = 5120;
// The imported-container q4_1 chunk (npu/q4nx.h Q4_1F): the same 32x256 tile with fp32 scale and
// zero, so a pack import keeps the pack's own parameter values (engine#1185).
constexpr size_t kTileBytesF32 = 6144;

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;
    uint64_t avail = 0;  // bytes from data to the end of the file: the outer bound a reader may not cross
    // engine#1881: the most a reader may touch. `avail` alone is only the FILE bound, so a bounds check written
    // against it passes for a tensor whose own bytes are short and lets the read run on into the next tensor's
    // data (same file, wrong tensor). Both bounds matter: a malformed container can also declare `bytes` past
    // the end of the file, so the tight bound is the smaller of the two.
    uint64_t usable() const { return bytes < avail ? bytes : avail; }

    // Tiles in a projection: shape[0] rows of shape[1] bytes, in 5120-byte tiles.
    int tiles() const;
};

struct Dims {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
    float rope_theta = 0;
    std::string model_type;  // config.json model_type, e.g. "qwen3"
    std::vector<int> eos;  // eos_token_id from config.json
};

class Model {
public:
    // dir holds model.q4nx and config.json.
    explicit Model(const std::string& dir);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Dims& dims() const { return dims_; }
    // Throws if the tensor is absent.
    const Tensor& tensor(const std::string& name) const;
    const Tensor* find(const std::string& name) const;
    // model.layers.<layer>.<suffix>
    const Tensor& layer(int layer, const std::string& suffix) const;
    const Tensor* find_layer(int layer, const std::string& suffix) const;
    // The container's __metadata__ member `key` as JSON text, "" when it is absent. The header
    // is kept as text so this header carries no JSON dependency; nlohmann parses it in the one
    // place that needs a member (npu/moe.cpp reads "moe").
    std::string metadata(const std::string& key) const;

private:
    uint8_t* map_ = nullptr;
    size_t size_ = 0;
    std::string header_;
    std::map<std::string, Tensor> tensors_;
    Dims dims_;
};

}  // namespace onebit::npu
