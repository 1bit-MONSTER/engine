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

#include "file_map.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace onebit::npu {

int Tensor::tiles() const {
    // [tile rows, tile cols, chunk bytes] (Q4NX with Q4_K or Q8 chunks, npu/q4nx.h)
    if (shape.size() == 3 && shape[0] > 0 && shape[1] > 0) return int(shape[0] * shape[1]);
    if (shape.size() != 2 || shape[0] <= 0) return 0;
    // A row is one tile when it is one whole chunk wide: 5120 (q4_1) or 6144 (the imported
    // container's fp32 q4_1f, engine#1185). Narrower rows pack several to a tile.
    if (shape[1] > 0 && size_t(shape[1]) != kTileBytes && size_t(shape[1]) != kTileBytesF32)
        return int((shape[0] * shape[1] + int64_t(kTileBytes) - 1) / int64_t(kTileBytes));
    return int(shape[0]);
}

Model::Model(const std::string& dir) {
    const std::string path = dir + "/model.q4nx";
    std::string why;
    const uint8_t* m = map_file(path, size_, &why);
    if (!m) throw std::runtime_error(why);
    map_ = const_cast<uint8_t*>(m);

    uint64_t header = 0;
    if (size_ < 8) throw std::runtime_error(path + ": truncated");
    std::memcpy(&header, map_, 8);
    if (header > size_ - 8) throw std::runtime_error(path + ": header past the end of the file");
    const uint64_t base = 8 + header;  // <= size_
    header_.assign(reinterpret_cast<const char*>(map_ + 8), header);
    const auto table = nlohmann::json::parse(map_ + 8, map_ + 8 + header);
    for (const auto& [name, t] : table.items()) {
        if (!t.is_object() || !t.contains("data_offsets")) continue;  // __metadata__
        Tensor x;
        x.dtype = t.value("dtype", "");
        x.shape = t.at("shape").get<std::vector<int64_t>>();
        const auto off = t.at("data_offsets").get<std::vector<uint64_t>>();
        // compared against what is left after base, never base + offset (which can wrap)
        if (off.size() != 2 || off[1] < off[0] || off[1] > size_ - base)
            throw std::runtime_error(path + ": bad data_offsets for " + name);
        x.data = map_ + base + off[0];
        x.bytes = off[1] - off[0];
        x.avail = size_ - base - off[0];
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
    // engine#1782: the vocabulary comes from config.json first. Two things went wrong when it came from the
    // tensor: (1) a pack that declares `tie_word_embeddings` may ship the table ONLY as `lm_head.weight` (the
    // 1.9/1.0.12 Decision packs do), so `tensor("model.embed_tokens.weight")` threw "model has no tensor …";
    // (2) a tiled entry's shape is [tile_rows, tile_cols, tile_bytes], so `shape[0]` is a TILE count -- a 248320-row
    // table is [7760, 4, 8704] and was reported as vocab 7760, hidden 4.
    if (c.contains("vocab_size")) {
        dims_.vocab = c.at("vocab_size");
    } else {
        const Tensor* emb = find("model.embed_tokens.weight");
        if (!emb && c.value("tie_word_embeddings", false)) emb = find("lm_head.weight");
        if (!emb)
            throw std::runtime_error(dir + "/config.json states no vocab_size and the container has no embedding "
                                             "table (`model.embed_tokens.weight`, or `lm_head.weight` for a model "
                                             "that declares tie_word_embeddings)");
        // 3-D is one tile grid (npu/q4nx.h: 32 rows x 256 columns per tile); 2-D is the plain row shape.
        static constexpr int kTileRows = 32;
        dims_.vocab = emb->shape.size() >= 3 ? int(emb->shape[0] * kTileRows) : int(emb->shape.at(0));
    }
    dims_.rope_theta = c.value("rope_theta", 10000.0f);
    dims_.model_type = c.value("model_type", "");
    if (c.contains("eos_token_id")) {
        const auto& e = c["eos_token_id"];
        if (e.is_array()) dims_.eos = e.get<std::vector<int>>();
        else if (e.is_number_integer()) dims_.eos = {e.get<int>()};
    }
}

Model::~Model() {
    unmap_file(map_, size_);
}

const Tensor* Model::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& Model::tensor(const std::string& name) const {
    if (const Tensor* t = find(name)) return *t;
    throw std::runtime_error("model has no tensor " + name);
}

std::string Model::metadata(const std::string& key) const {
    if (header_.empty()) return {};
    const auto table = nlohmann::json::parse(header_);
    const auto it = table.find("__metadata__");
    if (it == table.end() || !it->is_object()) return {};
    const auto member = it->find(key);
    return member == it->end() ? std::string() : member->dump();
}

const Tensor* Model::find_layer(int layer, const std::string& suffix) const {
    return find("model.layers." + std::to_string(layer) + "." + suffix);
}

const Tensor& Model::layer(int layer, const std::string& suffix) const {
    return tensor("model.layers." + std::to_string(layer) + "." + suffix);
}

}  // namespace onebit::npu
