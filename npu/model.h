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
// zeros, 4096 B packed int4, covering 32 x 256 weights.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::npu {

constexpr size_t kTileBytes = 5120;

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;

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

private:
    uint8_t* map_ = nullptr;
    size_t size_ = 0;
    std::map<std::string, Tensor> tensors_;
    Dims dims_;
};

}  // namespace onebit::npu
