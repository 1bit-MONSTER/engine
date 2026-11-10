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

// The tile geometry and chunk sizes the forward reads an embedding table with -- included, not restated, so the vocab
// check below cannot drift from the reader it stands in for (engine#1878 review).
#include "forward/include/q4nx_dequant.h"

namespace onebit::npu {

uint64_t Tensor::expected_bytes(uint64_t elements) const {
    // Element widths of the dtypes this reader decodes. Callers pass a row-major element count, so a tiled/chunked entry
    // (whose shape product is already a byte count: its last dimension is the chunk's BYTES) must not be sent here --
    // npu/lane.cpp checks `shape.size() == 2` for that reason. 0 means "not understood", and the caller then keeps the
    // file-end bound rather than refusing on a guess.
    uint64_t per = 0;
    if (dtype == "BF16" || dtype == "F16") per = 2;
    else if (dtype == "F32") per = 4;
    else if (dtype == "I32") per = 4;
    else if (dtype == "I8" || dtype == "U8") per = 1;
    if (per == 0 || elements == 0) return 0;
    return per * elements;
}

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

namespace {

// engine#1843: does `t` hold exactly `vocab` rows of `hidden`, under ANY layout the forward reads an embedding table in
// (q4nx_forward.cpp's init: BF16/F16/F32 rows at stride `hidden`; I8 either plain [vocab, hidden] -- bytes == V*H -- or
// chunked tiles of 32 rows x 256 columns at one of the three chunk sizes, as a 3-D [row_tiles, col_tiles, chunk] grid or
// a 2-D [tiles, chunk] list)? A tiled table may carry vocab padded up to a whole 32-row tile. An unrecognised dtype
// returns true: it is not evidence either way, and the forward has its own refusal for it. `seen` gets what the table
// does hold, for the message.
bool table_holds_vocab(const Tensor& t, int64_t vocab, int64_t hidden, std::string* seen) {
    if (vocab <= 0 || hidden <= 0) return true;
    const int64_t pad_v = (vocab + Q4NX_TILE_ROWS - 1) / Q4NX_TILE_ROWS * Q4NX_TILE_ROWS;
    const int64_t col_tiles = (hidden + Q4NX_TILE_COLS - 1) / Q4NX_TILE_COLS;
    const auto rows_ok = [&](int64_t r) { return r == vocab || r == pad_v; };
    int esz = 0;
    if (t.dtype == "BF16" || t.dtype == "F16") esz = 2;
    else if (t.dtype == "F32") esz = 4;
    if (esz) {
        const uint64_t row = uint64_t(hidden) * uint64_t(esz);
        if (t.bytes % row != 0) { *seen = "is " + std::to_string(t.bytes) + " B, not a whole number of rows"; return false; }
        const int64_t r = int64_t(t.bytes / row);
        *seen = "has " + std::to_string(r) + " rows of " + std::to_string(hidden);
        // A plain row table has no tile grid, so nothing pads it: every BF16 table on strixhalo (78 measured, 11-10)
        // has exactly vocab_size rows, and the forward reads it at row stride `hidden` with no pad rows.
        return r == vocab;
    }
    if (t.dtype != "I8") return true;
    if (t.bytes == uint64_t(vocab) * uint64_t(hidden)) return true;   // plain int8 + per-32-column scale (Gemma4)
    const auto is_chunk = [](int64_t b) { return b == Q4NX_TILE_BYTES || b == Q4NX_CHUNK_Q4K || b == Q4NX_CHUNK_Q8; };
    int64_t r = -1;
    if (t.shape.size() == 3 && is_chunk(t.shape[2]) && t.shape[1] == col_tiles) r = t.shape[0] * Q4NX_TILE_ROWS;
    else if (t.shape.size() == 2 && is_chunk(t.shape[1]) && t.shape[0] % col_tiles == 0)
        r = t.shape[0] / col_tiles * Q4NX_TILE_ROWS;
    if (r < 0) {
        *seen = "is neither " + std::to_string(vocab) + " x " + std::to_string(hidden) +
                " plain int8 nor a tile grid " + std::to_string(col_tiles) + " tiles wide";
        return false;
    }
    *seen = "holds " + std::to_string(r) + " rows (" + std::to_string(r / Q4NX_TILE_ROWS) + " row tiles)";
    return rows_ok(r);
}

}  // namespace

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
        // engine#1843: when the container also ships the table, it is evidence, and a table that cannot hold
        // vocab_size rows is refused by name. A config stating MORE rows than the table makes the embedding read run
        // into whatever tensor follows (Tensor::avail is measured to the end of the FILE, so lane.cpp's bound does not
        // catch it); FEWER leaves rows nothing can reach. Neither number can be trusted then.
        const Tensor* emb = find("model.embed_tokens.weight");
        if (!emb && c.value("tie_word_embeddings", false)) emb = find("lm_head.weight");
        if (emb) {
            std::string seen;
            if (!table_holds_vocab(*emb, dims_.vocab, dims_.hidden, &seen))
                throw std::runtime_error(dir + "/config.json states vocab_size " + std::to_string(dims_.vocab) +
                                         " but the embedding table (" + emb->dtype + ") " + seen +
                                         "; refusing rather than reading past it or leaving rows unreachable");
        }
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
