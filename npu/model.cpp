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

#include "model.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace onebit::npu {

int Tensor::tiles() const {
    // [tile rows, tile cols, chunk bytes] (Q4NX with Q4_K or Q8 chunks, npu/q4nx.h)
    if (shape.size() == 3 && shape[0] > 0 && shape[1] > 0) return int(shape[0] * shape[1]);
    if (shape.size() != 2 || shape[0] <= 0) return 0;
    // A row is one tile only when it is 5120 bytes wide; narrower rows pack
    // several to a tile.
    if (shape[1] > 0 && size_t(shape[1]) != kTileBytes)
        return int((shape[0] * shape[1] + int64_t(kTileBytes) - 1) / int64_t(kTileBytes));
    return int(shape[0]);
}

Model::Model(const std::string& dir) {
    const std::string path = dir + "/model.q4nx";
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    ::fstat(fd, &st);
    size_ = size_t(st.st_size);
    void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("cannot map " + path);
    map_ = static_cast<uint8_t*>(m);

    uint64_t header = 0;
    if (size_ < 8) throw std::runtime_error(path + ": truncated");
    std::memcpy(&header, map_, 8);
    if (8 + header > size_) throw std::runtime_error(path + ": header past the end of the file");
    const uint64_t base = 8 + header;
    const auto table = nlohmann::json::parse(map_ + 8, map_ + 8 + header);
    for (const auto& [name, t] : table.items()) {
        if (!t.is_object() || !t.contains("data_offsets")) continue;  // __metadata__
        Tensor x;
        x.dtype = t.value("dtype", "");
        x.shape = t.at("shape").get<std::vector<int64_t>>();
        const auto off = t.at("data_offsets").get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[1] < off[0] || base + off[1] > size_)
            throw std::runtime_error(path + ": bad data_offsets for " + name);
        x.data = map_ + base + off[0];
        x.bytes = off[1] - off[0];
        tensors_.emplace(name, std::move(x));
    }

    std::ifstream cf(dir + "/config.json");
    if (!cf) throw std::runtime_error("cannot read " + dir + "/config.json");
    const auto c = nlohmann::json::parse(cf);
    dims_.hidden = c.at("hidden_size");
    dims_.layers = c.at("num_hidden_layers");
    dims_.heads = c.at("num_attention_heads");
    dims_.kv_heads = c.value("num_key_value_heads", dims_.heads);
    dims_.head_dim = c.value("head_dim", dims_.hidden / dims_.heads);
    dims_.intermediate = c.at("intermediate_size");
    dims_.vocab = int(tensor("model.embed_tokens.weight").shape.at(0));
    dims_.rope_theta = c.value("rope_theta", 10000.0f);
    dims_.model_type = c.value("model_type", "");
    if (c.contains("eos_token_id")) {
        const auto& e = c["eos_token_id"];
        if (e.is_array()) dims_.eos = e.get<std::vector<int>>();
        else if (e.is_number_integer()) dims_.eos = {e.get<int>()};
    }
}

Model::~Model() {
    if (map_) ::munmap(map_, size_);
}

const Tensor* Model::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& Model::tensor(const std::string& name) const {
    if (const Tensor* t = find(name)) return *t;
    throw std::runtime_error("model has no tensor " + name);
}

const Tensor* Model::find_layer(int layer, const std::string& suffix) const {
    return find("model.layers." + std::to_string(layer) + "." + suffix);
}

const Tensor& Model::layer(int layer, const std::string& suffix) const {
    return tensor("model.layers." + std::to_string(layer) + "." + suffix);
}

}  // namespace onebit::npu
