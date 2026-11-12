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
// dflash_draft.cpp -- see dflash_draft.h.

#include "dflash_draft.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "file_map.h"
#include "forward/include/q4nx_dequant.h"
#include "pack_import.h"

namespace onebit::npu::dflash {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("dflash draft: " + what); }

float bf16(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int need_int(const json& c, const char* key, const std::string& path) {
    if (!c.contains(key)) fail(path + ": no '" + key + "'");
    if (!c[key].is_number_integer()) fail(path + ": '" + key + "' is not an integer");
    const int64_t v = c[key].get<int64_t>();
    if (v <= 0 || v > (int64_t(1) << 30)) fail(path + ": '" + key + "' is " + std::to_string(v));
    return int(v);
}

float need_float(const json& c, const char* key, const std::string& path) {
    if (!c.contains(key)) fail(path + ": no '" + key + "'");
    if (!c[key].is_number()) fail(path + ": '" + key + "' is not a number");
    const double v = c[key].get<double>();
    if (!std::isfinite(v) || v <= 0.0) fail(path + ": '" + key + "' is " + std::to_string(v));
    return float(v);
}

bool close_to(double a, double b) { return std::fabs(a - b) <= 1e-5 * std::max(1.0, std::fabs(b)); }

void read_target_layers(const json& from, const std::string& path, DraftConfig& o) {
    if (!from.contains("target_layer_ids") || !from["target_layer_ids"].is_array() || from["target_layer_ids"].empty())
        fail(path + ": no 'target_layer_ids'");
    int prev = -1;
    for (const auto& v : from["target_layer_ids"]) {
        if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > 4096)
            fail(path + ": 'target_layer_ids' holds something that is not a layer index");
        const int id = int(v.get<int64_t>());
        if (id <= prev) fail(path + ": 'target_layer_ids' is not strictly increasing at " + std::to_string(id));
        o.target_layer_ids.push_back(id);
        prev = id;
    }
}

// DFlash2DraftModel: the geometry is at the top level as in any Qwen3 config, the draft's interface is under
// "dflash_config", the rotation's base under "rope_parameters", and the weight file has the family's fixed name.
DraftConfig read_generation2(const json& c, const std::string& path) {
    DraftConfig o;
    o.generation = 2;
    o.hidden = need_int(c, "hidden_size", path);
    o.intermediate = need_int(c, "intermediate_size", path);
    o.layers = need_int(c, "num_hidden_layers", path);
    o.heads = need_int(c, "num_attention_heads", path);
    o.kv_heads = need_int(c, "num_key_value_heads", path);
    o.head_dim = need_int(c, "head_dim", path);
    o.vocab = need_int(c, "vocab_size", path);
    o.sliding_window = need_int(c, "sliding_window", path);
    o.rms_eps = need_float(c, "rms_norm_eps", path);
    if (!c.contains("rope_parameters") || !c["rope_parameters"].is_object())
        fail(path + ": no 'rope_parameters'");
    o.rope_theta = need_float(c["rope_parameters"], "rope_theta", path + " rope_parameters");
    o.embedding_scale = 1.0f;
    if (!c.contains("dflash_config") || !c["dflash_config"].is_object()) fail(path + ": no 'dflash_config'");
    const json& d = c["dflash_config"];
    const std::string dp = path + " dflash_config";
    o.block_size = need_int(d, "block_size", dp);
    o.conv_kernel = need_int(d, "conv_kernel_size", dp);
    o.conv_group = need_int(d, "conv_group_size", dp);
    o.selector_rank = need_int(d, "selector_rank", dp);
    o.selector_top_k = need_int(d, "selector_top_k", dp);
    if (!d.contains("mask_token_id") || !d["mask_token_id"].is_number_integer() || d["mask_token_id"].get<int64_t>() < 0)
        fail(dp + ": no usable 'mask_token_id'");
    o.mask_token_id = d["mask_token_id"].get<int64_t>();
    read_target_layers(d, dp, o);
    if (c.contains("is_causal")) {
        if (!c["is_causal"].is_boolean()) fail(path + ": 'is_causal' is not a boolean");
        o.causal = c["is_causal"].get<bool>();
    }
    o.draft_weight = "dflash_weight.q4nx";
    if (c.contains("draft_weight")) {
        if (!c["draft_weight"].is_string() || c["draft_weight"].get<std::string>().empty()) fail(path + ": 'draft_weight' is not a file name");
        o.draft_weight = c["draft_weight"].get<std::string>();
        if (o.draft_weight.find('/') != std::string::npos || o.draft_weight.find("..") != std::string::npos)
            fail(path + ": 'draft_weight' must be a file name inside the package, got '" + o.draft_weight + "'");
    }
    if (o.kv_heads > o.heads || o.heads % o.kv_heads != 0)
        fail(path + ": " + std::to_string(o.heads) + " heads cannot share " + std::to_string(o.kv_heads) + " key/value heads");
    if (o.block_size < 2) fail(path + ": block_size " + std::to_string(o.block_size) + " drafts nothing");
    if (o.conv_kernel < 1 || o.conv_group < 1 || o.hidden % o.conv_group != 0)
        fail(dp + ": a convolution of kernel " + std::to_string(o.conv_kernel) + " over groups of " +
             std::to_string(o.conv_group) + " does not fit hidden_size " + std::to_string(o.hidden));
    if (o.selector_rank < 1 || o.selector_top_k < 1 || o.selector_top_k > o.vocab)
        fail(dp + ": selector_rank or selector_top_k is not usable");
    if (o.causal) fail(path + ": a causal draft is not read by this loader");
    return o;
}

}  // namespace

DraftConfig read_draft_config(const std::string& package_dir) {
    const std::string path = package_dir + "/config.json";
    std::ifstream in(path);
    if (!in) fail("cannot open " + path);
    json c;
    try {
        in >> c;
    } catch (const std::exception& e) {
        fail(path + ": not JSON (" + e.what() + ")");
    }
    if (!c.is_object()) fail(path + ": not a JSON object");
    const std::string type = c.value("model_type", json("")).is_string() ? c.value("model_type", std::string()) : "";
    // The architecture decides which tensors exist, so it is read first. Two are known; a package naming another is a
    // different model, and reading it with these rules would drop what it adds silently.
    std::string arch;
    if (c.contains("architectures")) {
        if (!c["architectures"].is_array() || c["architectures"].empty() || !c["architectures"][0].is_string())
            fail(path + ": 'architectures' is not a list of names");
        arch = c["architectures"][0].get<std::string>();
    }
    if (arch == "DFlash2DraftModel") return read_generation2(c, path);
    if (type != "dflash") fail(path + ": model_type is '" + type + "', not 'dflash'");
    if (!arch.empty() && arch != "DFlashDraftModel")
        fail(path + ": architecture '" + arch + "' is not read by this loader (DFlashDraftModel and DFlash2DraftModel are)");
    DraftConfig o;
    o.hidden = need_int(c, "hidden_size", path);
    o.intermediate = need_int(c, "intermediate_size", path);
    o.layers = need_int(c, "num_hidden_layers", path);
    o.heads = need_int(c, "num_attention_heads", path);
    o.kv_heads = need_int(c, "num_key_value_heads", path);
    o.head_dim = need_int(c, "head_dim", path);
    o.block_size = need_int(c, "block_size", path);
    o.sliding_window = need_int(c, "sliding_window", path);
    o.rms_eps = need_float(c, "rms_norm_eps", path);
    o.rope_theta = need_float(c, "rope_theta", path);
    o.embedding_scale = need_float(c, "embedding_scale", path);
    if (!c.contains("mask_token_id") || !c["mask_token_id"].is_number_integer() || c["mask_token_id"].get<int64_t>() < 0)
        fail(path + ": no usable 'mask_token_id'");
    o.mask_token_id = c["mask_token_id"].get<int64_t>();
    read_target_layers(c, path, o);
    if (!c.contains("draft_weight") || !c["draft_weight"].is_string() || c["draft_weight"].get<std::string>().empty())
        fail(path + ": no 'draft_weight'");
    o.draft_weight = c["draft_weight"].get<std::string>();
    if (o.draft_weight.find('/') != std::string::npos || o.draft_weight.find("..") != std::string::npos)
        fail(path + ": 'draft_weight' must be a file name inside the package, got '" + o.draft_weight + "'");
    if (c.contains("causal")) {
        if (!c["causal"].is_boolean()) fail(path + ": 'causal' is not a boolean");
        o.causal = c["causal"].get<bool>();
    }
    if (c.contains("base_model") && c["base_model"].is_string()) o.base_model = c["base_model"].get<std::string>();
    if (o.kv_heads > o.heads || o.heads % o.kv_heads != 0)
        fail(path + ": " + std::to_string(o.heads) + " heads cannot share " + std::to_string(o.kv_heads) + " key/value heads");
    if (o.block_size < 2) fail(path + ": block_size " + std::to_string(o.block_size) + " drafts nothing");
    return o;
}

DraftModel load_draft(const std::string& package_dir) {
    DraftModel m;
    m.cfg = read_draft_config(package_dir);
    const DraftConfig& c = m.cfg;
    const std::string path = package_dir + "/" + c.draft_weight;

    const pack_import::Q4nxContainer table = pack_import::read_q4nx_container(path);
    if (!table.why.empty()) fail(table.why);
    size_t size = 0;
    std::string why;
    const uint8_t* base = map_file(path, size, &why);
    if (base == nullptr) fail(why);
    m.keep = std::shared_ptr<const void>(base, [size](const void* p) { unmap_file(static_cast<const uint8_t*>(p), size); });
    m.container_bytes = size;
    m.tensors = int(table.tensors.size());
    const size_t payload = 8 + size_t(table.header_bytes);

    std::map<std::string, const pack_import::Q4nxTensor*> by_name;
    for (const auto& t : table.tensors) by_name[t.name] = &t;
    std::set<std::string> used;

    // Every tensor is taken through here, so "inside the file" and "accounted for" are checked once, for all of them.
    const auto take = [&](const std::string& name, const char* dtype, size_t want_bytes, int rows, int cols) {
        const auto it = by_name.find(name);
        if (it == by_name.end()) fail(path + ": no tensor '" + name + "'");
        const pack_import::Q4nxTensor& t = *it->second;
        if (t.dtype != dtype) fail(path + ": '" + name + "' is declared " + t.dtype + ", expected " + dtype);
        if (t.begin < 0 || t.end < t.begin || payload + size_t(t.end) > size)
            fail(path + ": '" + name + "' lies outside the file");
        const size_t have = size_t(t.end - t.begin);
        if (have != want_bytes)
            fail(path + ": '" + name + "' holds " + std::to_string(have) + " B, and " + std::to_string(rows) + " x " +
                 std::to_string(cols) + " needs " + std::to_string(want_bytes));
        used.insert(name);
        TensorView v;
        v.name = name;
        v.data = base + payload + size_t(t.begin);
        v.bytes = have;
        v.rows = rows;
        v.cols = cols;
        return v;
    };
    // A generation-2 package stores its projections as Q4_K chunks (4736 B per 32 x 256), not q4_1 tiles (5120 B).
    const int chunk = c.generation == 2 ? Q4NX_CHUNK_Q4K : Q4NX_TILE_BYTES;
    const auto projection = [&](const std::string& name, int n, int k) {
        if (n % 32 != 0 || k % 256 != 0)
            fail(path + ": '" + name + "' would be " + std::to_string(n) + " x " + std::to_string(k) +
                 ", which is not a whole 32 x 256 tile grid");
        TensorView v = take(name, "I8", size_t(n / 32) * size_t(k / 256) * size_t(chunk), n, k);
        v.chunk = chunk;
        return v;
    };
    const auto gain = [&](const std::string& name, int width) { return take(name, "BF16", size_t(width) * 2, 1, width); };
    const auto plain = [&](const std::string& name, int rows, int cols) {
        return take(name, "BF16", size_t(rows) * size_t(cols) * 2, rows, cols);
    };

    // The container repeats the interface scalars; a package whose two halves disagree is not one model.
    const auto scalar = [&](const std::string& name, const char* dtype, size_t count) {
        return take(name, dtype, count * 4, 1, int(count)).data;
    };
    const auto i32_at = [](const uint8_t* p, size_t i) { int32_t v; std::memcpy(&v, p + i * 4, 4); return v; };
    const auto f32_at = [](const uint8_t* p) { float v; std::memcpy(&v, p, 4); return v; };
    const auto agree = [&](bool same, const std::string& name, const std::string& got, const std::string& want) {
        if (!same) fail(path + ": '" + name + "' is " + got + " and config.json says " + want);
    };
    if (c.generation == 1) {
        const uint8_t* p = scalar("dflash.block_size", "I32", 1);
        agree(i32_at(p, 0) == c.block_size, "dflash.block_size", std::to_string(i32_at(p, 0)), std::to_string(c.block_size));
        p = scalar("dflash.sliding_window", "I32", 1);
        agree(i32_at(p, 0) == c.sliding_window, "dflash.sliding_window", std::to_string(i32_at(p, 0)), std::to_string(c.sliding_window));
        p = scalar("dflash.target_layer_ids", "I32", size_t(c.taps()));
        for (int i = 0; i < c.taps(); ++i)
            agree(i32_at(p, size_t(i)) == c.target_layer_ids[size_t(i)], "dflash.target_layer_ids[" + std::to_string(i) + "]",
                  std::to_string(i32_at(p, size_t(i))), std::to_string(c.target_layer_ids[size_t(i)]));
        p = scalar("dflash.rope_theta", "F32", 1);
        agree(close_to(f32_at(p), c.rope_theta), "dflash.rope_theta", std::to_string(f32_at(p)), std::to_string(c.rope_theta));
        p = scalar("dflash.rms_eps", "F32", 1);
        agree(close_to(f32_at(p) * 1e6, double(c.rms_eps) * 1e6), "dflash.rms_eps", std::to_string(f32_at(p)), std::to_string(c.rms_eps));
        p = scalar("dflash.embedding_scale", "F32", 1);
        agree(close_to(f32_at(p), c.embedding_scale), "dflash.embedding_scale", std::to_string(f32_at(p)), std::to_string(c.embedding_scale));
    }

    // The fusion matrix: hidden x (taps * hidden), stored in 32-row blocks (see decode_fusion), so hidden must fill them
    // and the declared shape must be that blocking -- a plain [hidden, taps*hidden] label would mean another element order.
    if (c.hidden % 32 != 0) fail(path + ": hidden_size " + std::to_string(c.hidden) + " does not fill 32-row blocks");
    const int fused = c.taps() * c.hidden;
    if (c.generation == 2) {
        m.fc = projection("model.fc.weight", c.hidden, fused);
        const int groups = c.hidden / c.conv_group;
        m.selector_hproj = plain("model.candidate_selector.hidden_projection.weight", c.selector_rank, c.hidden);
        m.selector_pred = plain("model.candidate_selector.predecessor_codebook", c.vocab, c.selector_rank);
        m.selector_succ = plain("model.candidate_selector.successor_codebook", c.vocab, c.selector_rank);
        (void)groups;
    } else {
        m.fc = take("model.fc.weight", "BF16", size_t(c.hidden) * size_t(fused) * 2, c.hidden, fused);
        const std::vector<int64_t> want = {c.hidden / 32, fused, 32};
        if (by_name["model.fc.weight"]->shape != want)
            fail(path + ": 'model.fc.weight' is not declared [hidden/32, taps*hidden, 32], so its element order is not the one this loader undoes");
    }
    m.hidden_norm = gain("model.hidden_norm.weight", c.hidden);
    m.norm = gain("model.norm.weight", c.hidden);
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        DraftLayer L;
        L.q = projection(p + "self_attn.q_proj.weight", c.q_dim(), c.hidden);
        L.k = projection(p + "self_attn.k_proj.weight", c.kv_dim(), c.hidden);
        L.v = projection(p + "self_attn.v_proj.weight", c.kv_dim(), c.hidden);
        L.o = projection(p + "self_attn.o_proj.weight", c.hidden, c.q_dim());
        L.gate = projection(p + "mlp.gate_proj.weight", c.intermediate, c.hidden);
        L.up = projection(p + "mlp.up_proj.weight", c.intermediate, c.hidden);
        L.down = projection(p + "mlp.down_proj.weight", c.hidden, c.intermediate);
        L.input_norm = gain(p + "input_layernorm.weight", c.hidden);
        L.post_norm = gain(p + "post_attention_layernorm.weight", c.hidden);
        L.q_norm = gain(p + "self_attn.q_norm.weight", c.head_dim);
        L.k_norm = gain(p + "self_attn.k_norm.weight", c.head_dim);
        if (c.generation == 2) {
            const int kernels = 2 * c.conv_kernel, groups = c.hidden / c.conv_group;
            L.attn_conv_base = plain(p + "attention_conv.base_kernel", kernels, c.hidden);
            L.attn_conv_proj = plain(p + "attention_conv.kernel_projection.weight", kernels * groups, c.hidden);
            L.mlp_conv_base = plain(p + "mlp_conv.base_kernel", kernels, c.hidden);
            L.mlp_conv_proj = plain(p + "mlp_conv.kernel_projection.weight", kernels * groups, c.hidden);
        }
        m.layers.push_back(L);
    }
    for (const auto& t : table.tensors)
        if (used.count(t.name) == 0)
            fail(path + ": tensor '" + t.name + "' is not part of this draft's architecture (" + std::to_string(table.tensors.size() - used.size()) +
                 " unexplained): this is another architecture");
    return m;
}

std::vector<float> decode_projection(const TensorView& t) {
    if (!t.ok()) fail("decode_projection of an empty view");
    std::vector<float> out(size_t(t.rows) * size_t(t.cols));
    // Not the size-dispatching q4nx_dequant_tensor(): a projection here is a tile grid by contract, and a grid whose
    // byte count happened to equal rows*cols*2 must not be read as plain 2-byte floats.
    const int chunk = t.chunk > 0 ? t.chunk : Q4NX_TILE_BYTES;
    if (t.rows % 32 != 0 || t.cols % 256 != 0 || t.bytes != size_t(t.rows / 32) * size_t(t.cols / 256) * size_t(chunk) ||
        q4nx_dequant_tensor_chunked(t.data, t.bytes, t.rows, t.cols, chunk, out.data()) != 0)
        fail("'" + t.name + "' does not decode as a " + std::to_string(t.rows) + " x " + std::to_string(t.cols) + " grid of " +
             std::to_string(chunk) + "-byte tiles");
    return out;
}

std::vector<float> decode_gain(const TensorView& t) {
    if (!t.ok() || t.bytes != size_t(t.cols) * 2) fail("decode_gain of '" + t.name + "': not 2 bytes per element");
    std::vector<float> out(size_t(t.cols));
    for (int i = 0; i < t.cols; ++i) out[size_t(i)] = bf16(t.data + size_t(i) * 2);
    return out;
}

std::vector<float> decode_plain(const TensorView& t) {
    if (!t.ok() || t.bytes != size_t(t.rows) * size_t(t.cols) * 2) fail("decode_plain of '" + t.name + "': not 2 bytes per element");
    std::vector<float> out(size_t(t.rows) * size_t(t.cols));
    for (size_t i = 0; i < out.size(); ++i) out[i] = bf16(t.data + i * 2);
    return out;
}

void decode_row(const TensorView& t, int row, float* out) {
    if (!t.ok() || t.bytes != size_t(t.rows) * size_t(t.cols) * 2) fail("decode_row of '" + t.name + "': not 2 bytes per element");
    if (row < 0 || row >= t.rows || out == nullptr) fail("decode_row of '" + t.name + "': row " + std::to_string(row) + " is outside the tensor");
    const uint8_t* p = t.data + size_t(row) * size_t(t.cols) * 2;
    for (int i = 0; i < t.cols; ++i) out[i] = bf16(p + size_t(i) * 2);
}

std::vector<float> decode_fusion(const TensorView& t) {
    if (!t.ok() || t.rows % 32 != 0 || t.bytes != size_t(t.rows) * size_t(t.cols) * 2)
        fail("decode_fusion of '" + t.name + "': not hidden x fused at 2 bytes per element");
    const size_t cols = size_t(t.cols);
    std::vector<float> out(size_t(t.rows) * cols);
    for (size_t a = 0; a < size_t(t.rows) / 32; ++a)
        for (size_t k = 0; k < cols; ++k) {
            const uint8_t* p = t.data + ((a * cols + k) * 32) * 2;
            for (size_t r = 0; r < 32; ++r) out[(a * 32 + r) * cols + k] = bf16(p + r * 2);
        }
    return out;
}

}  // namespace onebit::npu::dflash
