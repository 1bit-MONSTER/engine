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
#include "vision_pack.h"

#include "file_map.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace onebit::npu::vision {

namespace {

using nlohmann::json;
using Shape = std::vector<int64_t>;

const char* const kPrefix = "model.visual.";

std::string shape_text(const Shape& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

// The product of a shape, or false when a dimension is not positive or the product does not fit: a shape is file
// content, and its product sizes a read.
bool elements_of(const Shape& s, uint64_t* out) {
    uint64_t n = 1;
    if (s.empty()) return false;
    for (int64_t d : s) {
        if (d <= 0) return false;
        if (n > (uint64_t(1) << 40) / uint64_t(d)) return false;
        n *= uint64_t(d);
    }
    *out = n;
    return true;
}

int as_int(int64_t v) { return v > 0 && v <= (int64_t(1) << 30) ? int(v) : 0; }

}  // namespace

Pack::Pack(const std::string& pack_dir) {
    std::ifstream cf(pack_dir + "/config.json");
    if (!cf) throw std::runtime_error("cannot read " + pack_dir + "/config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    const std::string text = ss.str();
    std::string name = "vision_weight.q4nx";
    const json c = json::parse(text, nullptr, false);
    if (c.is_discarded() || !c.is_object()) throw std::runtime_error(pack_dir + "/config.json is not a JSON object");
    if (c.contains("vision_model_weight") && c["vision_model_weight"].is_string())
        name = c["vision_model_weight"].get<std::string>();
    // The name is file content and is joined to a directory: a separator in it would leave the pack.
    if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos || name == "..")
        throw std::runtime_error(pack_dir + "/config.json: vision_model_weight \"" + name + "\" is not a file name");
    open(pack_dir + "/" + name, text);
}

Pack::Pack(const std::string& container_path, const std::string& config_json) { open(container_path, config_json); }

Pack::~Pack() {
    if (map_) unmap_file(map_, size_);
}

void Pack::open(const std::string& container_path, const std::string& config_json) {
    path_ = container_path;
    const auto fail = [&](const std::string& what) -> void { throw std::runtime_error(path_ + ": " + what); };

    std::string why;
    map_ = map_file(path_, size_, &why);
    if (!map_) throw std::runtime_error(why);
    if (size_ < 8) fail("truncated (no header length)");
    uint64_t header = 0;
    std::memcpy(&header, map_, 8);
    if (header > size_ - 8) fail("the header runs past the end of the file");
    const json table = json::parse(map_ + 8, map_ + 8 + header, nullptr, false);
    if (table.is_discarded() || !table.is_object()) fail("the header is not a JSON object");
    payload_ = size_ - 8 - header;
    const uint8_t* base = map_ + 8 + header;

    // ── every tensor: BF16, inside the payload, exactly as long as its shape says ──
    for (const auto& [name, t] : table.items()) {
        if (name == "__metadata__") {
            if (t.is_object())
                for (const auto& [k, v] : t.items()) meta_[k] = v.is_string() ? v.get<std::string>() : v.dump();
            continue;
        }
        if (!t.is_object() || !t.contains("data_offsets") || !t.contains("shape") || !t.contains("dtype"))
            fail("entry " + name + " is not a tensor record (dtype, shape, data_offsets)");
        if (!t["dtype"].is_string() || t["dtype"].get<std::string>() != "BF16")
            fail(name + " is " + (t["dtype"].is_string() ? t["dtype"].get<std::string>() : std::string("of no named dtype")) +
                 "; every tensor of a vision component is BF16 and this reader converts nothing else");
        Tensor x;
        std::vector<uint64_t> off;
        try {
            x.shape = t["shape"].get<Shape>();
            off = t["data_offsets"].get<std::vector<uint64_t>>();
        } catch (const json::exception&) {
            fail(name + " has a shape or data_offsets that is not a list of integers");
        }
        if (off.size() != 2 || off[1] < off[0] || off[1] > payload_)
            fail(name + " has data_offsets outside the payload (" + std::to_string(payload_) + " B)");
        uint64_t n = 0;
        if (!elements_of(x.shape, &n)) fail(name + " has shape " + shape_text(x.shape) + ", which is not a size");
        x.bytes = off[1] - off[0];
        if (x.bytes != n * 2)
            fail(name + " spans " + std::to_string(x.bytes) + " B but its shape " + shape_text(x.shape) + " is " +
                 std::to_string(n * 2) + " B of BF16");
        x.offset = off[0];
        x.data = base + off[0];
        tensors_.emplace(name, std::move(x));
    }
    if (tensors_.empty()) fail("holds no tensors");

    // ── no two tensors share a byte ──
    {
        std::vector<std::pair<uint64_t, const std::string*>> order;
        for (const auto& [name, t] : tensors_) order.push_back({t.offset, &name});
        std::sort(order.begin(), order.end());
        uint64_t end = 0, covered = 0;
        const std::string* prev = nullptr;
        for (const auto& [off, name] : order) {
            if (prev && off < end) fail(*name + " overlaps " + *prev + " in the payload");
            end = off + tensors_.at(*name).bytes;
            covered += tensors_.at(*name).bytes;
            prev = name;
        }
        unaccounted_ = payload_ - covered;
    }

    // ── the geometry, from the shapes ──
    const auto need = [&](const std::string& suffix) -> const Tensor& {
        const auto it = tensors_.find(kPrefix + suffix);
        if (it == tensors_.end()) fail("has no tensor " + std::string(kPrefix) + suffix);
        return it->second;
    };
    const Shape& pe = need("patch_embed.proj.weight").shape;
    if (pe.size() != 4 || pe[2] != pe[3])
        fail("model.visual.patch_embed.proj.weight is " + shape_text(pe) + ", not [hidden, channels, patch, patch]");
    geom_.hidden = as_int(pe[0]);
    geom_.channels = as_int(pe[1]);
    geom_.patch = as_int(pe[2]);
    const int64_t H = geom_.hidden;

    int max_block = -1;
    std::vector<char> seen;
    const std::string bp = std::string(kPrefix) + "blocks.";
    for (const auto& [name, t] : tensors_) {
        if (name.compare(0, bp.size(), bp) != 0) continue;
        const size_t dot = name.find('.', bp.size());
        const std::string idx = name.substr(bp.size(), dot == std::string::npos ? std::string::npos : dot - bp.size());
        if (dot == std::string::npos || idx.empty() || idx.size() > 4 || idx.find_first_not_of("0123456789") != std::string::npos)
            fail(name + " does not name a block index");
        const int i = std::stoi(idx);
        if (i >= int(seen.size())) seen.resize(size_t(i) + 1, 0);
        seen[size_t(i)] = 1;
        max_block = std::max(max_block, i);
    }
    if (max_block < 0) fail("has no model.visual.blocks.* tensors");
    for (int i = 0; i <= max_block; ++i)
        if (!seen[size_t(i)]) fail("has block " + std::to_string(max_block) + " but no block " + std::to_string(i));
    geom_.layers = max_block + 1;

    const Shape& fc1 = need("blocks.0.mlp.linear_fc1.weight").shape;
    if (fc1.size() != 2) fail("model.visual.blocks.0.mlp.linear_fc1.weight is " + shape_text(fc1) + ", not a matrix");
    geom_.intermediate = as_int(fc1[0]);
    const int64_t I = geom_.intermediate;

    const Shape& pos = need("pos_embed.weight").shape;
    if (pos.size() != 2) fail("model.visual.pos_embed.weight is " + shape_text(pos) + ", not [positions, hidden]");
    geom_.pos_rows = as_int(pos[0]);
    for (int g = 1; int64_t(g) * g <= geom_.pos_rows; ++g)
        if (int64_t(g) * g == geom_.pos_rows) geom_.grid = g;
    if (geom_.grid == 0)
        fail("model.visual.pos_embed.weight has " + std::to_string(pos[0]) + " rows, which is not a square patch grid");

    geom_.merge_hidden = int(4 * H);
    const int64_t M = geom_.merge_hidden;
    const Shape& vfc1 = need("vit_merger.mlp.linear_fc1.weight").shape;
    if (vfc1.size() != 2) fail("model.visual.vit_merger.mlp.linear_fc1.weight is " + shape_text(vfc1) + ", not a matrix");
    geom_.vit_merger_intermediate = as_int(vfc1[0]);
    const int64_t VI = geom_.vit_merger_intermediate;
    const Shape& mfc2 = need("merger.linear_fc2.weight").shape;
    if (mfc2.size() != 2) fail("model.visual.merger.linear_fc2.weight is " + shape_text(mfc2) + ", not a matrix");
    geom_.out_hidden = as_int(mfc2[0]);
    const int64_t O = geom_.out_hidden;
    if (!geom_.hidden || !geom_.channels || !geom_.patch || !geom_.intermediate || !geom_.vit_merger_intermediate || !geom_.out_hidden)
        fail("a dimension of the tower is zero or too large to be one");

    // ── every tensor the tower has, with the shape that geometry gives it; and nothing else in the file ──
    std::map<std::string, Shape> want;
    const auto add = [&](const std::string& suffix, Shape s) { want[kPrefix + suffix] = std::move(s); };
    const auto linear = [&](const std::string& stem, int64_t rows, int64_t cols) {
        add(stem + ".weight", {rows, cols});
        add(stem + ".bias", {rows});
    };
    const auto norm = [&](const std::string& stem, int64_t n) {
        add(stem + ".weight", {n});
        add(stem + ".bias", {n});
    };
    add("patch_embed.proj.weight", {H, geom_.channels, geom_.patch, geom_.patch});
    add("patch_embed.proj.bias", {H});
    add("pos_embed.weight", {geom_.pos_rows, H});
    for (int i = 0; i < geom_.layers; ++i) {
        const std::string b = "blocks." + std::to_string(i) + ".";
        norm(b + "norm1", H);
        linear(b + "attn.qkv", 3 * H, H);
        linear(b + "attn.proj", H, H);
        norm(b + "norm2", H);
        linear(b + "mlp.linear_fc1", I, H);
        linear(b + "mlp.linear_fc2", H, I);
    }
    norm("vit_merger.norm1", H);
    linear("vit_merger.attn.qkv", 3 * H, H);
    linear("vit_merger.attn.proj", H, H);
    norm("vit_merger.pre_norm", M);
    linear("vit_merger.mlp.linear_fc1", VI, M);
    linear("vit_merger.mlp.linear_fc2", H, VI);
    norm("merger.norm", H);
    norm("merger.pre_norm", M);
    linear("merger.linear_fc1", M, M);
    linear("merger.linear_fc2", O, M);

    for (const auto& [name, shape] : want) {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) fail("has no tensor " + name);
        if (it->second.shape != shape)
            fail(name + " is " + shape_text(it->second.shape) + " where the tower's other tensors make it " + shape_text(shape));
    }
    for (const auto& [name, t] : tensors_)
        if (!want.count(name)) fail("holds " + name + " " + shape_text(t.shape) + ", a tensor this reader has no place for");

    // ── config.json: what the tensors cannot say, and what ties the tower to something outside it ──
    geom_.model_intermediate = geom_.intermediate;
    geom_.model_vit_merger_intermediate = geom_.vit_merger_intermediate;
    if (config_json.empty()) return;
    const json c = json::parse(config_json, nullptr, false);
    if (c.is_discarded() || !c.is_object()) fail("the config beside it is not a JSON object");
    const json vc = c.contains("vision_config") && c["vision_config"].is_object() ? c["vision_config"] : json::object();
    const auto field = [&](const char* key, int* out) {
        const std::string k = std::string("MINICPM_V_") + key;
        if (!vc.contains(k) || !vc[k].is_number_integer()) return false;
        *out = as_int(vc[k].get<int64_t>());
        return true;
    };
    const auto refuse = [&](const char* key, int stated, const std::string& held) {
        fail("config.json states MINICPM_V_" + std::string(key) + " " + std::to_string(stated) + " but " + held);
    };
    int v = 0;
    if (field("VISION_EMBED_DIM", &v) && v != geom_.hidden)
        refuse("VISION_EMBED_DIM", v, "the patch embedding is " + std::to_string(geom_.hidden) + " wide");
    if (field("VISION_NUM_LAYERS", &v) && v != geom_.layers)
        refuse("VISION_NUM_LAYERS", v, "the container holds " + std::to_string(geom_.layers) + " blocks");
    if (field("PATCH_SIZE", &v) && v != geom_.patch)
        refuse("PATCH_SIZE", v, "the patch embedding's kernel is " + std::to_string(geom_.patch));
    if (field("VISION_NUM_POSITION_EMBEDDINGS", &v) && v != geom_.pos_rows)
        refuse("VISION_NUM_POSITION_EMBEDDINGS", v, "the position table has " + std::to_string(geom_.pos_rows) + " rows");
    if (field("VISION_OUT_HIDDEN_SIZE", &v) && v != geom_.out_hidden)
        refuse("VISION_OUT_HIDDEN_SIZE", v, "the merger projects to " + std::to_string(geom_.out_hidden));
    if (field("MERGER_HIDDEN_SIZE", &v) && v != geom_.merge_hidden)
        refuse("MERGER_HIDDEN_SIZE", v, "four tokens of this tower are " + std::to_string(geom_.merge_hidden) + " wide");
    if (field("IMAGE_SIZE", &v)) {
        geom_.image_size = v;
        if (v != geom_.grid * geom_.patch)
            refuse("IMAGE_SIZE", v, "the position table is a " + std::to_string(geom_.grid) + " x " + std::to_string(geom_.grid) +
                                        " grid of " + std::to_string(geom_.patch) + "-pixel patches (" +
                                        std::to_string(geom_.grid * geom_.patch) + " pixels)");
    }
    int heads = 0, head_dim = 0;
    const bool have_heads = field("VISION_NUM_HEADS", &heads), have_dim = field("VISION_HEAD_DIM", &head_dim);
    if (have_heads && !have_dim && heads > 0 && geom_.hidden % heads == 0) head_dim = geom_.hidden / heads;
    if (have_heads) {
        if (heads <= 0 || int64_t(heads) * head_dim != H)
            refuse("VISION_NUM_HEADS", heads, "head_dim " + std::to_string(head_dim) + " times that is not the tower's width " +
                                                  std::to_string(geom_.hidden));
        geom_.heads = heads;
        geom_.head_dim = head_dim;
    }
    if (c.contains("hidden_size") && c["hidden_size"].is_number_integer() && c["hidden_size"].get<int64_t>() != O)
        fail("config.json states the text model's hidden_size " + std::to_string(c["hidden_size"].get<int64_t>()) +
             " but the merger projects to " + std::to_string(geom_.out_hidden) + ": this tower is not this text model's");

    {
        // top level first (the model's own config key), then the vision_config spelling; both present must agree
        int top = -1, inner = -1;
        const bool have_top = c.contains("insert_layer_id") && c["insert_layer_id"].is_number_integer();
        if (have_top) top = int(c["insert_layer_id"].get<int64_t>());
        const bool have_inner = vc.contains("MINICPM_V_INSERT_LAYER_ID") && vc["MINICPM_V_INSERT_LAYER_ID"].is_number_integer();
        if (have_inner) inner = int(vc["MINICPM_V_INSERT_LAYER_ID"].get<int64_t>());
        if (have_top && have_inner && top != inner)
            fail("config.json states insert_layer_id " + std::to_string(top) + " and MINICPM_V_INSERT_LAYER_ID " +
                 std::to_string(inner) + ": one tower, two merge points");
        if (have_top || have_inner) geom_.insert_layer = have_top ? top : inner;
        if ((have_top || have_inner) && (geom_.insert_layer < 0 || geom_.insert_layer >= geom_.layers))
            fail("config.json states insert_layer_id " + std::to_string(geom_.insert_layer) + " but the container holds " +
                 std::to_string(geom_.layers) + " blocks");
    }

    // An MLP's stated width against its tensors. `stems` are the MLPs that share the width (every block, or the
    // vit_merger alone): fc1 is [width, in] with a [width] bias, fc2 is [out, width].
    const auto zero_from = [&](const std::string& name, int64_t rows, int64_t cols, int64_t from, bool by_row) {
        const Tensor& t = tensors_.at(kPrefix + name);
        const auto is_zero = [&](int64_t i) {
            uint16_t h;
            std::memcpy(&h, t.data + 2 * i, 2);
            return (h & 0x7fff) == 0;
        };
        if (by_row) {
            for (int64_t i = from * cols; i < rows * cols; ++i)
                if (!is_zero(i)) return false;
        } else {
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = from; c < cols; ++c)
                    if (!is_zero(r * cols + c)) return false;
        }
        return true;
    };
    const auto mlp_width = [&](const char* key, int held, const std::vector<std::string>& stems, int64_t in, int64_t out,
                               const char* tensor, int* model) {
        int stated = 0;
        *model = held;
        if (!field(key, &stated) || stated == held) return;
        if (stated > held)
            refuse(key, stated, std::string(tensor) + " has only " + std::to_string(held) + " rows");
        bool pad = true;
        for (const std::string& st : stems)
            pad = pad && zero_from(st + ".linear_fc1.weight", held, in, stated, true) &&
                  zero_from(st + ".linear_fc1.bias", held, 1, stated, true) &&
                  zero_from(st + ".linear_fc2.weight", out, held, stated, false);
        const std::string head = "config.json states MINICPM_V_" + std::string(key) + " " + std::to_string(stated) + "; " + tensor +
                                 " has " + std::to_string(held) + " rows";
        if (pad) {
            *model = stated;
            notes_.push_back(head + ", the last " + std::to_string(held - stated) +
                             " all zero: the container pads the model's width, and sizes come from the tensor");
        } else {
            notes_.push_back(head + " and the rows past the stated width are not zero: this is not padding, the tensor's "
                                    "width is the model's");
        }
    };
    std::vector<std::string> block_mlps;
    for (int i = 0; i < geom_.layers; ++i) block_mlps.push_back("blocks." + std::to_string(i) + ".mlp");
    mlp_width("VISION_MLP_INTERMEDIATE_SIZE", geom_.intermediate, block_mlps, H, H,
              "model.visual.blocks.*.mlp.linear_fc1.weight", &geom_.model_intermediate);
    mlp_width("VIT_MERGER_INTERMEDIATE_SIZE", geom_.vit_merger_intermediate, {"vit_merger.mlp"}, M, H,
              "model.visual.vit_merger.mlp.linear_fc1.weight", &geom_.model_vit_merger_intermediate);
}

const Tensor& Pack::tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error(path_ + ": has no tensor " + name);
    return it->second;
}

std::vector<float> Pack::f32(const std::string& name) const {
    const Tensor& t = tensor(name);
    std::vector<float> out(static_cast<size_t>(t.elements()));
    for (size_t i = 0; i < out.size(); ++i) {
        uint16_t h = 0;
        std::memcpy(&h, t.data + 2 * i, 2);
        const uint32_t bits = uint32_t(h) << 16;
        std::memcpy(&out[i], &bits, 4);
    }
    return out;
}

std::string Pack::metadata(const std::string& key) const {
    const auto it = meta_.find(key);
    return it == meta_.end() ? std::string() : it->second;
}

int print_info(const std::string& pack_dir) {
    const Pack p(pack_dir);
    const Geometry& g = p.geometry();
    std::printf("vision component  %s\n", p.path().c_str());
    std::printf("tensors           %zu, all BF16, %llu B of payload, %llu B unaccounted\n", p.tensors().size(),
                (unsigned long long)p.payload_bytes(), (unsigned long long)p.unaccounted_bytes());
    std::printf("patch embed       %d x %d pixels x %d channels -> %d\n", g.patch, g.patch, g.channels, g.hidden);
    std::printf("positions         %d (a %d x %d grid; image %d)\n", g.pos_rows, g.grid, g.grid, g.grid * g.patch);
    std::printf("blocks            %d at width %d, %d heads x %d, MLP %d (%d stored)\n", g.layers, g.hidden, g.heads, g.head_dim,
                g.model_intermediate, g.intermediate);
    std::printf("vit_merger        after block %d; attention at %d, MLP %d -> %d (%d stored) -> %d\n", g.insert_layer, g.hidden, g.merge_hidden,
                g.model_vit_merger_intermediate, g.vit_merger_intermediate, g.hidden);
    std::printf("merger            %d -> %d -> %d (the text model's hidden size)\n", g.merge_hidden, g.merge_hidden, g.out_hidden);
    for (const char* k : {"source_file", "source_sha256", "source_dtype", "converter"})
        if (!p.metadata(k).empty()) std::printf("%-17s %s\n", k, p.metadata(k).c_str());
    for (const std::string& n : p.notes()) std::printf("note              %s\n", n.c_str());
    return 0;
}

}  // namespace onebit::npu::vision
