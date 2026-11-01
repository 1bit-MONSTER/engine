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

#include "pack_import.h"

#include "file_map.h"
#include "pack.h"
#include "q4nx.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

namespace onebit::npu::pack_import {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

// ── protobuf wire format, as much of it as ONNX needs ────────────────────────
struct Field {
    uint32_t num = 0, wire = 0;
    uint64_t v = 0;                 // wire 0
    const uint8_t* p = nullptr;     // wire 2
    size_t n = 0;
};

uint64_t rd_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (p >= end) throw std::runtime_error("protobuf: truncated varint");
        const uint8_t c = *p++;
        v |= uint64_t(c & 0x7F) << shift;
        if (!(c & 0x80)) return v;
    }
    throw std::runtime_error("protobuf: varint overflow");
}

std::vector<Field> pb(const uint8_t* p, size_t n) {
    const uint8_t* const end = p + n;
    std::vector<Field> out;
    while (p < end) {
        const uint64_t key = rd_varint(p, end);
        Field f;
        f.num = uint32_t(key >> 3);
        f.wire = uint32_t(key & 7);
        if (f.num == 0) throw std::runtime_error("protobuf: field 0");
        switch (f.wire) {
            case 0: f.v = rd_varint(p, end); break;
            case 1:
                if (size_t(end - p) < 8) throw std::runtime_error("protobuf: truncated fixed64");
                p += 8;
                break;
            case 2: {
                const uint64_t len = rd_varint(p, end);
                if (len > uint64_t(end - p)) throw std::runtime_error("protobuf: length past the end");
                f.p = p;
                f.n = size_t(len);
                p += len;
                break;
            }
            case 5:
                if (size_t(end - p) < 4) throw std::runtime_error("protobuf: truncated fixed32");
                p += 4;
                break;
            default: throw std::runtime_error("protobuf: unsupported wire type " + std::to_string(f.wire));
        }
        out.push_back(f);
    }
    return out;
}

std::string pb_str(const Field& f) { return std::string(reinterpret_cast<const char*>(f.p), f.n); }

const Field* field1(const std::vector<Field>& fs, uint32_t num) {
    for (const auto& f : fs)
        if (f.num == num) return &f;
    return nullptr;
}

std::vector<const Field*> fields(const std::vector<Field>& fs, uint32_t num) {
    std::vector<const Field*> out;
    for (const auto& f : fs)
        if (f.num == num) out.push_back(&f);
    return out;
}

// ── ONNX: only what a weight importer reads ─────────────────────────────────
// ModelProto.graph=7; GraphProto{node=1, initializer=5}; NodeProto{input=1,
// output=2, op_type=4, attribute=5}; AttributeProto{name=1, graph=6, graphs=11};
// TensorProto{dims=1, data_type=2, name=8, raw_data=9, external_data=13,
// data_location=14}; StringStringEntryProto{key=1, value=2}.
//
// An OGA pack keeps every weight inside the `If` branches: the top-level graph's
// initializer list is EMPTY and its four nodes are Shape/Constant/Equal/If. So the
// walk descends into subgraphs; that is also how both branches' copies of
// model.embed_tokens.weight are recognised as one tensor.
enum OnnxType { kF32 = 1, kU8 = 2, kI8 = 3, kI32 = 6, kI64 = 7, kF16 = 10, kBF16 = 16 };

struct OnnxTensor {
    std::string name;
    int type = 0;
    std::vector<int64_t> dims;
    std::string location;
    uint64_t offset = 0, length = 0;
    const uint8_t* raw = nullptr;
    size_t raw_len = 0;
};

struct OnnxMatMul {
    std::string data, weight, scales, zeros;
    int64_t K = 0, N = 0;
    int bits = 0, block = 0;
};

struct OnnxModel {
    std::vector<OnnxTensor> tensors;
    std::vector<OnnxMatMul> matmuls;
    std::map<std::string, size_t> index;
};

void walk_graph(const uint8_t* p, size_t n, OnnxModel& m) {
    const auto g = pb(p, n);
    for (const auto* f : fields(g, 5)) {
        const auto t = pb(f->p, f->n);
        OnnxTensor x;
        if (const auto* d = field1(t, 8)) x.name = pb_str(*d);
        if (const auto* d = field1(t, 2)) x.type = int(d->v);
        for (const auto* d : fields(t, 1)) x.dims.push_back(int64_t(d->v));
        if (const auto* d = field1(t, 9)) {
            x.raw = d->p;
            x.raw_len = d->n;
        }
        for (const auto* d : fields(t, 13)) {
            const auto kv = pb(d->p, d->n);
            const auto* k = field1(kv, 1);
            const auto* v = field1(kv, 2);
            if (!k || !v) continue;
            const auto key = pb_str(*k), val = pb_str(*v);
            if (key == "location") x.location = val;
            else if (key == "offset") x.offset = std::stoull(val);
            else if (key == "length") x.length = std::stoull(val);
        }
        if (x.name.empty() || m.index.count(x.name)) continue;
        m.index.emplace(x.name, m.tensors.size());
        m.tensors.push_back(std::move(x));
    }
    for (const auto* f : fields(g, 1)) {
        const auto node = pb(f->p, f->n);
        const auto* opf = field1(node, 4);
        const auto op = opf ? pb_str(*opf) : std::string();
        std::vector<std::string> ins;
        for (const auto* i : fields(node, 1)) ins.push_back(pb_str(*i));
        std::map<std::string, int64_t> attrs;
        for (const auto* a : fields(node, 5)) {
            const auto at = pb(a->p, a->n);
            const auto* nm = field1(at, 1);
            if (!nm) continue;
            if (const auto* iv = field1(at, 3)) attrs[pb_str(*nm)] = int64_t(iv->v);
            for (const auto* sg : fields(at, 6)) walk_graph(sg->p, sg->n, m);
            for (const auto* sg : fields(at, 11)) walk_graph(sg->p, sg->n, m);
        }
        if (op == "MatMulNBits") {
            OnnxMatMul mm;
            if (ins.size() >= 4) {
                mm.data = ins[0];
                mm.weight = ins[1];
                mm.scales = ins[2];
                mm.zeros = ins[3];
            }
            if (attrs.count("K")) mm.K = attrs["K"];
            if (attrs.count("N")) mm.N = attrs["N"];
            if (attrs.count("bits")) mm.bits = int(attrs["bits"]);
            if (attrs.count("block_size")) mm.block = int(attrs["block_size"]);
            if (!mm.weight.empty()) m.matmuls.push_back(std::move(mm));
        }
    }
}

OnnxModel parse_onnx(const std::string& path) {
    size_t size = 0;
    std::string why;
    const uint8_t* map = map_file(path, size, &why);
    if (!map) throw std::runtime_error(why);
    OnnxModel m;
    try {
        const auto model = pb(map, size);
        const auto* g = field1(model, 7);
        if (!g) throw std::runtime_error(path + ": no graph");
        walk_graph(g->p, g->n, m);
    } catch (...) {
        unmap_file(map, size);
        throw;
    }
    unmap_file(map, size);
    return m;
}

// ── the .fconst files: layer block geometry (observed, byte-exact) ───────────
// amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §2.1. Every layer block opens
// with bf16 1e-6 (the RMSNorm epsilon, little-endian 86 35), which is what locates a
// block; the two norms are the only layer tensors whose position is established, and
// their values were verified byte-exact against the hybrid pack's copies.
constexpr int64_t kTokStride = 8708288;
constexpr int64_t kNormOff = 64;         // input_layernorm.weight, bf16 [hidden]
constexpr int64_t kPostNormOff = 1116032;
constexpr int64_t kPostEpsOff = 1115968;  // the block's second epsilon
constexpr int64_t kTailNorm = 1856;       // [1792 norm][2 eps][62 pad] at the file's end
constexpr int64_t kFconstHidden = 896;    // the geometry both fconsts describe

struct Fconst {
    std::string path;
    const uint8_t* map = nullptr;
    size_t size = 0;
};

Fconst open_fconst(const std::string& path) {
    if (path.empty()) throw std::runtime_error("no dd_metastate_Llm_Token_Token_*.fconst in the pack");
    Fconst f;
    f.path = path;
    std::string why;
    f.map = map_file(path, f.size, &why);
    if (!f.map) throw std::runtime_error(why);
    return f;
}

bool eps_at(const Fconst& f, int64_t off) {
    return off >= 0 && size_t(off) + 4 <= f.size && f.map[off] == 0x86 && f.map[off + 1] == 0x35;
}

const uint8_t* at(const Fconst& f, int64_t off, int64_t len) {
    if (off < 0 || len < 0 || size_t(off + len) > f.size)
        throw std::runtime_error(f.path + ": offset past the end of the file");
    return f.map + off;
}

std::string find_one(const std::string& dir, const std::string& prefix, const std::string& suffix) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() >= prefix.size() + suffix.size() && n.compare(0, prefix.size(), prefix) == 0 &&
            n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0)
            return e.path().string();
    }
    return {};
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0x1F) bits = sign | 0x7F800000u | (man << 13);
    else if (exp == 0) bits = sign;
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

constexpr const char* kRefuseFconst =
    "weights are only in dd_metastate_*.fconst, and the 4-bit packing of the transposed tile is not "
    "established (amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §3)";

Planned make(const std::string& name, const std::string& source, const std::string& dtype,
             std::vector<int64_t> shape) {
    Planned p;
    p.name = name;
    p.source = source;
    p.dtype = dtype;
    p.shape = std::move(shape);
    return p;
}

Dims read_dims(const json& c) {
    Dims d;
    d.hidden = c.at("hidden_size").get<int>();
    d.layers = c.at("num_hidden_layers").get<int>();
    d.heads = c.at("num_attention_heads").get<int>();
    d.kv_heads = c.value("num_key_value_heads", d.heads);
    d.head_dim = c.value("head_dim", d.hidden / d.heads);
    d.intermediate = c.at("intermediate_size").get<int>();
    d.vocab = c.value("vocab_size", 0);
    d.rope_theta = c.value("rope_theta", 10000.0f);
    d.model_type = c.value("model_type", "");
    if (c.contains("eos_token_id")) {
        const auto& e = c["eos_token_id"];
        if (e.is_array()) d.eos = e.get<std::vector<int>>();
        else if (e.is_number_integer()) d.eos = {e.get<int>()};
    }
    return d;
}

}  // namespace

// ── MatMulNBits → (codes, scale, zero) per 32-column group ───────────────────
Q4 decode_matmulnbits(const uint8_t* qweight, const float* scales, const uint8_t* zeros,
                      int N, int K, int block, bool swap_nibbles, bool drop_zero) {
    if (N <= 0 || K <= 0 || block <= 0 || K % block || K % 32)
        throw std::runtime_error("MatMulNBits: bad N/K/block_size");
    Q4 q;
    q.N = N;
    q.K = K;
    q.block = block;
    q.codes.assign(size_t(N) * K, 0);
    q.scale.assign(size_t(N) * (K / 32), 0.0f);
    q.zero.assign(size_t(N) * (K / 32), 0.0f);
    const int gpb = block / 32, bpb = block / 2;
    for (int n = 0; n < N; ++n) {
        const uint8_t* qw = qweight + size_t(n) * (K / block) * bpb;
        for (int b = 0; b < K / block; ++b) {
            const float s = scales[size_t(n) * (K / block) + b];
            int zp = 0;
            if (!drop_zero && zeros) {
                const uint8_t z = zeros[size_t(n) * ((K / block + 1) / 2) + b / 2];
                zp = (b & 1) ? (z >> 4) : (z & 0x0F);
            }
            for (int g = 0; g < gpb; ++g) {
                const size_t idx = size_t(n) * (K / 32) + size_t(b) * gpb + g;
                q.scale[idx] = s;
                q.zero[idx] = -float(zp) * s;  // (code - zp) * s  ==  code*s + (-zp*s)
            }
            for (int i = 0; i < bpb; ++i) {
                const uint8_t byte = qw[size_t(b) * bpb + i];
                const int k = b * block + 2 * i;
                q.codes[size_t(n) * K + k] = swap_nibbles ? (byte >> 4) : (byte & 0x0F);
                q.codes[size_t(n) * K + k + 1] = swap_nibbles ? (byte & 0x0F) : (byte >> 4);
            }
        }
    }
    return q;
}

// ── (codes, scale, zero) → q4_1 chunks (docs/npu.md, "Q4NX chunks") ──────────
void tile_q4_1(const Q4& q, std::vector<uint8_t>& out, long long* zero_inexact) {
    if (q.N % q4nx::kRows) throw std::runtime_error("q4_1 tiles need N % 32 == 0");
    // The grid is 32 x 256, so a tensor whose K is not a whole number of 256-column
    // tiles (Qwen2.5-0.5B: hidden 896 = 3.5 tiles) is stored with the last column tile
    // zero-padded: codes 0, scale 0, zero 0, i.e. w = 0 in the padding. The real K is
    // in the manifest and the caller's config; nothing reads the padding as a weight.
    const int KP = ((q.K + q4nx::kCols - 1) / q4nx::kCols) * q4nx::kCols;
    const int RT = q.N / q4nx::kRows, KT = KP / q4nx::kCols;
    out.assign(size_t(RT) * KT * kTileBytes, 0);
    long long inexact = 0;
    for (int rt = 0; rt < RT; ++rt) {
        for (int kt = 0; kt < KT; ++kt) {
            uint8_t* tile = out.data() + (size_t(rt) * KT + kt) * kTileBytes;
            auto* sc16 = reinterpret_cast<uint16_t*>(tile);
            auto* zr16 = reinterpret_cast<uint16_t*>(tile + 512);
            for (int r = 0; r < q4nx::kRows; ++r) {
                const int row = rt * q4nx::kRows + r;
                for (int g = 0; g < 8; ++g) {
                    const int group = (kt * 8 + g) * 32;
                    if (group >= q.K) {  // zero padding
                        sc16[g * 32 + r] = 0;
                        zr16[g * 32 + r] = 0;
                        continue;
                    }
                    const size_t idx = size_t(row) * (q.K / 32) + size_t(kt) * 8 + g;
                    sc16[g * 32 + r] = f32_to_bf16(q.scale[idx]);
                    zr16[g * 32 + r] = f32_to_bf16(q.zero[idx]);
                    if (zero_inexact && bf16_to_f32(zr16[g * 32 + r]) != q.zero[idx]) ++inexact;
                }
            }
            // codes: region h*8+g, byte 8*j + (r%16)/2, even row in the low nibble
            for (int h = 0; h < 2; ++h)
                for (int g = 0; g < 8; ++g)
                    for (int j = 0; j < 32; ++j) {
                        const int k = kt * q4nx::kCols + g * 32 + j;
                        for (int r2 = 0; r2 < 8; ++r2) {
                            const int r_even = rt * q4nx::kRows + h * 16 + r2 * 2;
                            if (k >= q.K) {  // zero padding column
                                tile[1024 + h * 2048 + g * 256 + j * 8 + r2] = 0;
                                continue;
                            }
                            const uint8_t even = q.codes[size_t(r_even) * q.K + k] & 0x0F;
                            const uint8_t odd = q.codes[size_t(r_even + 1) * q.K + k] & 0x0F;
                            tile[1024 + h * 2048 + g * 256 + j * 8 + r2] = uint8_t(even | (odd << 4));
                        }
                    }
        }
    }
    if (zero_inexact) *zero_inexact = inexact;
}

// ── plan ────────────────────────────────────────────────────────────────────
std::vector<std::string> Plan::refused() const {
    std::vector<std::string> out;
    for (const auto& t : tensors)
        if (!t.refusal.empty()) out.push_back(t.name + ": " + t.refusal);
    return out;
}

std::string Plan::manifest() const {
    json j;
    j["format"] = "q4nx";
    j["layout"] = "q4nx-1";  // container layout version (npu/q4nx.h chunk kinds)
    j["source"] = {{"kind", "oga-npu-pack"}, {"pack", pack}, {"importer", "1bit npu-pack import"}};
    j["dims"] = {{"hidden_size", dims.hidden},
                 {"intermediate_size", dims.intermediate},
                 {"num_hidden_layers", dims.layers},
                 {"num_attention_heads", dims.heads},
                 {"num_key_value_heads", dims.kv_heads},
                 {"head_dim", dims.head_dim},
                 {"vocab_size", dims.vocab},
                 {"rope_theta", dims.rope_theta},
                 {"model_type", dims.model_type}};
    json ts = json::array();
    for (const auto& t : tensors)
        ts.push_back({{"name", t.name}, {"source", t.source}, {"dtype", t.dtype}, {"shape", t.shape},
                      {"refusal", t.refusal}});
    j["tensors"] = ts;
    j["incomplete"] = refused();
    return j.dump(2);
}

Plan plan(const std::string& pack_dir) {
    Plan p;
    p.dir = pack_dir;
    p.pack = fs::path(pack_dir).lexically_normal().filename().string();

    std::ifstream cf(pack_dir + "/config.json");
    if (!cf) throw std::runtime_error("cannot read " + pack_dir + "/config.json");
    p.dims = read_dims(json::parse(cf));
    const Dims& d = p.dims;

    const OnnxModel onnx = parse_onnx(pack_dir + "/model.onnx");
    const OnnxTensor* embed = nullptr;
    for (const auto& t : onnx.tensors)
        if (t.name == "model.embed_tokens.weight") embed = &t;
    if (!embed) throw std::runtime_error("the pack has no model.embed_tokens.weight");
    if (d.vocab == 0) p.dims.vocab = int(embed->dims.at(0));
    const int64_t H = embed->dims.at(1);
    p.tensors.push_back(make("model.embed_tokens.weight", embed->location, "BF16",
                             {embed->dims.at(0), H}));

    for (const auto& mm : onnx.matmuls) {
        const auto it = onnx.index.find(mm.weight);
        if (it == onnx.index.end()) continue;
        auto x = make("lm_head.weight", onnx.tensors[it->second].location, "q4_1", {mm.N, mm.K});
        if (mm.bits != 4) x.refusal = "MatMulNBits is " + std::to_string(mm.bits) + "-bit, not 4-bit";
        p.tensors.push_back(std::move(x));
        break;
    }
    if (onnx.matmuls.empty()) {
        auto x = make("lm_head.weight", "fconst:Llm_Token_Token", "q4_1", {d.vocab, d.hidden});
        x.refusal = "the pack has no host-side MatMulNBits head; lm_head is only in the .fconst tail";
        p.tensors.push_back(std::move(x));
    }

    // norms: fixed offsets in the .fconst (§2.1), the only layer tensors with a known position
    const std::string norm_source = "fconst:Llm_Token_Token";
    const std::string norm_refusal =
        H == kFconstHidden ? std::string() : "the .fconst offsets are established for hidden 896 only";
    for (int L = 0; L < d.layers; ++L) {
        auto a = make("model.layers." + std::to_string(L) + ".input_layernorm.weight", norm_source, "BF16", {H});
        auto b = make("model.layers." + std::to_string(L) + ".post_attention_layernorm.weight", norm_source,
                      "BF16", {H});
        a.refusal = norm_refusal;
        b.refusal = norm_refusal;
        p.tensors.push_back(std::move(a));
        p.tensors.push_back(std::move(b));
    }
    // Measured 2026-10-31 on the shipped 0.5B pack: the 1,856-byte tail of EITHER .fconst
    // is not an RMSNorm weight. Token_Token's tail opens with a byte-pattern run
    // (33 33 33 33 55 55 55 55 77 77 ...) and ends in 64 constant bf16; Llm_Prefill's opens
    // with the epsilon at +0, then 62 pad, then 448 f32 in 7.6..16.8 -- the tail prologue
    // FACTS §2.2 already flagged as "not a plain fp32 scale array". So FACTS §2's
    // "final-norm/eps bytes appear at the tail end of both files" holds only for the
    // epsilon; the final norm's position is not established, and it is refused rather
    // than guessed.
    auto fin = make("model.norm.weight", norm_source, "BF16", {H});
    fin.refusal = "the .fconst tail envelope is [eps][pad][448 f32], not an RMSNorm weight; the final "
                  "norm's position is not established (measured 2026-10-31)";
    p.tensors.push_back(std::move(fin));

    // projections: the one thing a pack states and this importer cannot yet read
    const int q_out = d.heads * d.head_dim, kv_out = d.kv_heads * d.head_dim;
    struct Proj { const char* suffix; int rows, cols; };
    const Proj projs[] = {{"self_attn.q_proj", q_out, d.hidden},
                          {"self_attn.k_proj", kv_out, d.hidden},
                          {"self_attn.v_proj", kv_out, d.hidden},
                          {"self_attn.o_proj", d.hidden, q_out},
                          {"mlp.gate_proj", d.intermediate, d.hidden},
                          {"mlp.up_proj", d.intermediate, d.hidden},
                          {"mlp.down_proj", d.hidden, d.intermediate}};
    for (int L = 0; L < d.layers; ++L)
        for (const auto& pr : projs) {
            auto x = make("model.layers." + std::to_string(L) + "." + pr.suffix + ".weight",
                          "fconst:Llm_Token_Token", "q4_1", {pr.rows, pr.cols});
            x.refusal = kRefuseFconst;
            p.tensors.push_back(std::move(x));
        }
    return p;
}

// ── import ──────────────────────────────────────────────────────────────────
namespace {

struct Writer {
    std::string dir;
    json table = json::object();
    std::vector<std::pair<std::string, std::vector<uint8_t>>> blobs;
    uint64_t offset = 0;

    void add(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape,
             std::vector<uint8_t> bytes) {
        table[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + bytes.size()}}};
        offset += bytes.size();
        blobs.emplace_back(name, std::move(bytes));
    }
    void finish(const json& metadata) {
        json t = json::object();
        t["__metadata__"] = metadata;
        for (const auto& [k, v] : table.items()) t[k] = v;
        std::string header = t.dump();
        header.append((64 - header.size() % 64) % 64, ' ');
        std::ofstream f(dir + "/model.q4nx", std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + dir + "/model.q4nx");
        const uint64_t len = header.size();
        f.write(reinterpret_cast<const char*>(&len), 8);
        f.write(header.data(), std::streamsize(header.size()));
        for (const auto& [name, bytes] : blobs)
            f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        if (!f) throw std::runtime_error("short write into " + dir + "/model.q4nx");
    }
};

std::vector<uint8_t> as_bf16(const uint8_t* src, int type, size_t count) {
    std::vector<uint8_t> out(count * 2);
    auto* d = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < count; ++i) {
        if (type == kBF16) std::memcpy(&d[i], src + i * 2, 2);
        else if (type == kF16) {
            uint16_t h;
            std::memcpy(&h, src + i * 2, 2);
            d[i] = f32_to_bf16(f16_to_f32(h));
        } else if (type == kF32) {
            float v;
            std::memcpy(&v, src + i * 4, 4);
            d[i] = f32_to_bf16(v);
        } else {
            throw std::runtime_error("dtype " + std::to_string(type) + " is not convertible to bf16");
        }
    }
    return out;
}

double corr(const Q4& q, const uint8_t* embed, int H, int rows) {
    double sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
    const int n = std::min(rows, q.N);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < q.K; ++k) {
            const size_t g = size_t(r) * (q.K / 32) + size_t(k) / 32;
            const double x = double(q.codes[size_t(r) * q.K + k]) * q.scale[g] + q.zero[g];
            uint16_t h;
            std::memcpy(&h, embed + (size_t(r) * H + k) * 2, 2);
            const double y = f16_to_f32(h);
            sxy += x * y;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
        }
    const double m = double(n) * q.K;
    const double den = (m * sxx - sx * sx) * (m * syy - sy * sy);
    return den > 0 ? (m * sxy - sx * sy) / std::sqrt(den) : 0.0;
}

}  // namespace

Report import_pack(const std::string& pack_dir, const std::string& out_dir, const Options& opt) {
    const Plan p = plan(pack_dir);
    Report r;
    r.out_dir = out_dir;
    r.refused = p.refused();
    if (!r.refused.empty() && !opt.allow_incomplete) return r;  // refuse by name, write nothing

    bool tied = false;
    {
        std::ifstream cf(pack_dir + "/config.json");
        if (cf) {
            const auto c = json::parse(cf);
            tied = c.value("tie_word_embeddings", false);
        }
    }
    const OnnxModel onnx = parse_onnx(pack_dir + "/model.onnx");
    std::map<std::string, const uint8_t*> ext;
    std::map<std::string, size_t> ext_size;
    for (const auto& t : onnx.tensors) {
        if (t.location.empty() || ext.count(t.location)) continue;
        size_t size = 0;
        std::string why;
        const uint8_t* m = map_file(pack_dir + "/" + t.location, size, &why);
        if (!m) throw std::runtime_error(why);
        ext[t.location] = m;
        ext_size[t.location] = size;
    }
    auto bytes_of = [&](const OnnxTensor& t) -> const uint8_t* {
        if (t.raw) return t.raw;
        const auto it = ext.find(t.location);
        if (it == ext.end()) throw std::runtime_error(t.name + ": no data");
        if (t.offset + t.length > ext_size[t.location])
            throw std::runtime_error(t.name + ": external data past the end of " + t.location);
        return it->second + t.offset;
    };

    fs::create_directories(out_dir);
    Writer w;
    w.dir = out_dir;

    const OnnxTensor* embed = nullptr;
    for (const auto& t : onnx.tensors)
        if (t.name == "model.embed_tokens.weight") embed = &t;
    if (!embed) throw std::runtime_error("no model.embed_tokens.weight");
    const int64_t V = embed->dims.at(0), H = embed->dims.at(1);
    const uint8_t* embed_bytes = bytes_of(*embed);
    w.add("model.embed_tokens.weight", "BF16", {V, H}, as_bf16(embed_bytes, embed->type, size_t(V) * H));
    ++r.tensors;

    for (const auto& mm : onnx.matmuls) {
        const auto wi = onnx.index.find(mm.weight), si = onnx.index.find(mm.scales);
        if (wi == onnx.index.end() || si == onnx.index.end()) continue;
        if (mm.bits != 4) throw std::runtime_error("lm_head: MatMulNBits is not 4-bit");
        const auto& wt = onnx.tensors[wi->second];
        const auto& st = onnx.tensors[si->second];
        const auto zi = onnx.index.find(mm.zeros);
        const bool have_zp = zi != onnx.index.end();

        std::vector<float> scales(size_t(mm.N) * (size_t(mm.K) / mm.block));
        if (st.type == kF32) std::memcpy(scales.data(), bytes_of(st), scales.size() * 4);
        else {
            const auto b = as_bf16(bytes_of(st), st.type, scales.size());
            const auto* d = reinterpret_cast<const uint16_t*>(b.data());
            for (size_t i = 0; i < scales.size(); ++i) scales[i] = bf16_to_f32(d[i]);
        }
        const uint8_t* zp = have_zp ? bytes_of(onnx.tensors[zi->second]) : nullptr;
        const Q4 q = decode_matmulnbits(bytes_of(wt), scales.data(), zp, int(mm.N), int(mm.K), mm.block);

        std::vector<uint8_t> tiled;
        long long zinexact = 0;
        tile_q4_1(q, tiled, &zinexact);
        r.zero_exact_pct = 100.0 * double(q.scale.size() - zinexact) / double(q.scale.size());
        r.codes_total = q.codes.size();

        // self-check 1: tie_word_embeddings makes the pack's own fp16 embedding an
        // independent reference for the decode, with two wrong decoders as controls.
        r.selfcheck_reference = tied;
        if (tied) {
            double sum_abs = 0, sum_ref = 0;
            for (int n = 0; n < q.N; ++n)
                for (int k = 0; k < q.K; ++k) {
                    const size_t g = size_t(n) * (q.K / 32) + size_t(k) / 32;
                    const double x = double(q.codes[size_t(n) * q.K + k]) * q.scale[g] + q.zero[g];
                    double e;
                    if (embed->type == kF16) {
                        uint16_t h;
                        std::memcpy(&h, embed_bytes + (size_t(n) * H + k) * 2, 2);
                        e = f16_to_f32(h);
                    } else {
                        e = bf16_to_f32(reinterpret_cast<const uint16_t*>(embed_bytes)[size_t(n) * H + k]);
                    }
                    sum_abs += std::fabs(x - e);
                    sum_ref += std::fabs(e);
                }
            r.lmhead_rel = sum_abs / sum_ref;
            r.lmhead_corr = corr(q, embed_bytes, int(H), q.N);
            const Q4 swap = decode_matmulnbits(bytes_of(wt), scales.data(), zp, int(mm.N), int(mm.K),
                                               mm.block, /*swap_nibbles*/ true, false);
            const Q4 nozp = decode_matmulnbits(bytes_of(wt), scales.data(), nullptr, int(mm.N), int(mm.K),
                                               mm.block, false, /*drop_zero*/ true);
            r.ctrl_corr_nibble = corr(swap, embed_bytes, int(H), 2048);
            r.ctrl_corr_nozp = corr(nozp, embed_bytes, int(H), 2048);
        }

        // self-check 2: the tiles as written, read back through the engine's own decoder.
        double scale_sum = 0;
        const int KT = (q.K + q4nx::kCols - 1) / q4nx::kCols;  // the padded grid
        std::vector<float> w32(q4nx::kRows * q4nx::kCols);
        for (int i = 0; i < int(tiled.size() / kTileBytes); i += 97) {
            const int rt = i / KT, kt = i % KT;
            q4nx::dequant_tile(q4nx::Chunk::Q4_1, tiled.data() + size_t(i) * kTileBytes, w32.data(),
                               q4nx::kCols);
            for (int rr = 0; rr < q4nx::kRows; ++rr) {
                const int row = rt * q4nx::kRows + rr;
                for (int cc = 0; cc < q4nx::kCols; ++cc) {
                    const int k = kt * q4nx::kCols + cc;
                    if (k >= q.K) continue;  // padding column: not a weight
                    const size_t g = size_t(row) * (q.K / 32) + size_t(k) / 32;
                    const double ref = double(q.codes[size_t(row) * q.K + k]) * q.scale[g] + q.zero[g];
                    const double err = std::fabs(w32[size_t(rr) * q4nx::kCols + cc] - ref);
                    r.q4nx_max_abs = std::max(r.q4nx_max_abs, err);
                    r.q4nx_rel += err;
                    scale_sum += std::fabs(ref);
                }
            }
        }
        r.q4nx_rel = scale_sum > 0 ? r.q4nx_rel / scale_sum : 0;

        // the count has to be taken before tiled is moved (argument order is unspecified)
        const int64_t head_tiles = int64_t(tiled.size() / kTileBytes);
        r.tiles += head_tiles;
        w.add("lm_head.weight", "I8", {head_tiles, int64_t(kTileBytes)}, std::move(tiled));
        ++r.tensors;
        break;
    }

    if (H == kFconstHidden) {
        Fconst fc = open_fconst(find_one(pack_dir, "dd_metastate_Llm_Token_Token_", ".fconst"));
        auto add_norm = [&](int64_t off, const std::string& name) {
            w.add(name, "BF16", {H}, as_bf16(at(fc, off, H * 2), kBF16, size_t(H)));
            ++r.tensors;
        };
        for (int L = 0; L < p.dims.layers; ++L) {
            const int64_t base = int64_t(L) * kTokStride;
            r.norm_anchors_total += 2;
            r.norm_anchors_ok += int(eps_at(fc, base)) + int(eps_at(fc, base + kPostEpsOff));
            add_norm(base + kNormOff, "model.layers." + std::to_string(L) + ".input_layernorm.weight");
            add_norm(base + kPostNormOff,
                     "model.layers." + std::to_string(L) + ".post_attention_layernorm.weight");
        }
        // the final norm is refused in the plan: its position in the .fconst is not
        // established (see plan()), so nothing is written for it
        unmap_file(fc.map, fc.size);
    }

    // the manifest goes in the container, so a loader (or a person) can see what this
    // pack did and did not contribute without re-reading the pack
    w.finish(json::parse(p.manifest()));

    {
        std::ifstream cf(pack_dir + "/config.json");
        json c = json::parse(cf);
        c["vocab_size"] = V;
        c["q4nx_source"] = {{"converter", "1bit npu-pack import"},
                            {"pack", p.pack},
                            {"layout", "q4nx-1"},
                            {"incomplete", r.refused.size()}};
        std::ofstream out(out_dir + "/config.json");
        out << c.dump(2) << "\n";
    }
    for (const char* f : {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                          "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
                          "generation_config.json"}) {
        const std::string src = pack_dir + "/" + f;
        if (fs::exists(src)) fs::copy_file(src, out_dir + "/" + f, fs::copy_options::overwrite_existing);
    }
    for (const auto& [name, m] : ext) unmap_file(m, ext_size[name]);
    return r;
}

// ── the subcommand ──────────────────────────────────────────────────────────
int run(int argc, char** argv) {
    const std::string usage =
        "usage: 1bit npu-pack info   <pack dir>\n"
        "       1bit npu-pack import <pack dir> -o <out dir> [--allow-incomplete]\n"
        "  <pack dir>  a Ryzen AI OGA NPU model pack (*_rai_*_npu_*: model.onnx,\n"
        "              model.onnx.data, dd_metastate_*.fconst, config.json, tokenizer)\n"
        "  info        print what the pack states as JSON; writes nothing\n"
        "  import      write <out dir>/model.q4nx + config.json (+ tokenizer files)\n"
        "  --allow-incomplete  write what is representable and list the rest in the\n"
        "              container's manifest instead of refusing\n";
    if (argc < 1) {
        std::fprintf(stderr, "%s", usage.c_str());
        return 2;
    }
    const std::string cmd = argv[0];
    if (cmd == "--help" || cmd == "-h") {
        std::printf("%s", usage.c_str());
        return 0;
    }
    std::string dir, out;
    bool allow = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-o" || a == "--out") {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            out = argv[++i];
        } else if (a == "--allow-incomplete") allow = true;
        else if (a == "--help" || a == "-h") {
            std::printf("%s", usage.c_str());
            return 0;
        } else if (a.size() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else if (dir.empty()) dir = a;
        else throw std::runtime_error("unexpected argument " + a);
    }
    if (dir.empty()) {
        std::fprintf(stderr, "%s", usage.c_str());
        return 2;
    }
    if (cmd == "info") {
        std::printf("%s\n", plan(dir).manifest().c_str());
        return 0;
    }
    if (cmd != "import") throw std::runtime_error("unknown subcommand " + cmd);
    if (out.empty()) throw std::runtime_error("import needs -o <out dir>");

    const Report r = import_pack(dir, out, Options{allow});
    if (!r.refused.empty() && !allow) {
        std::fprintf(stderr,
                     "1bit npu-pack: refusing to import %s: %zu of the pack's tensors cannot be "
                     "represented yet, so nothing was written.\n",
                     dir.c_str(), r.refused.size());
        for (const auto& s : r.refused) std::fprintf(stderr, "  %s\n", s.c_str());
        std::fprintf(stderr, "  (--allow-incomplete writes the rest and records these in the manifest)\n");
        return 3;
    }
    std::printf("imported %d tensors, %lld q4_1 tiles into %s\n", r.tensors, r.tiles, out.c_str());
    if (r.selfcheck_reference) {
        std::printf("  self-check: lm_head vs the pack's fp16 embedding  corr %.6f  rel_err %.6f\n",
                    r.lmhead_corr, r.lmhead_rel);
        std::printf("  controls:   nibble-swapped corr %.4f, no zero-point corr %.4f (both must fail)\n",
                    r.ctrl_corr_nibble, r.ctrl_corr_nozp);
    } else {
        std::printf("  self-check: skipped -- the head is not tied to the embedding, so the pack\n"
                    "              carries no independent reference to test the decode against\n");
    }
    std::printf("  container:  read back through q4nx::dequant_tile  rel_err %.3e  max |diff| %.3e\n",
                r.q4nx_rel, r.q4nx_max_abs);
    std::printf("  parameters: bf16 scale/zero exact for %.2f%% of 32-column groups (format limit)\n",
                r.zero_exact_pct);
    if (r.norm_anchors_total)
        std::printf("  fconst:     %d/%d RMSNorm anchors found at their documented offsets\n",
                    r.norm_anchors_ok, r.norm_anchors_total);
    if (!r.refused.empty()) std::printf("  incomplete: %zu tensors, listed in %s/model.q4nx\n",
                                        r.refused.size(), out.c_str());
    return 0;
}

}  // namespace onebit::npu::pack_import
