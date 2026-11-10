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
constexpr int64_t kTailNorm = 1856;       // the .fconst tail envelope: [2 eps][62 pad][1792 norm]
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

bool eps_at(const Fconst& f, int64_t off, unsigned short eps_bits) {
    return off >= 0 && size_t(off) + 4 <= f.size && f.map[off] == uint8_t(eps_bits & 0xFF) &&
           f.map[off + 1] == uint8_t(eps_bits >> 8);
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


// ── locating the layer blocks in a .fconst, for ANY geometry (engine#734) ────
// The offsets measured on Qwen2.5-0.5B (stride 8,708,288; second header at 1,115,968) are functions of
// hidden size and of each projection's N and K, so constants only work for one pack. What does generalise
// is the structure: every layer block opens with the bf16 epsilon (1e-6 = 86 35) plus zero padding, and
// carries a second such header right after the attention payload. So the header positions form a
// sequence with one period per block, and the stride is positions[i+P] - positions[i] for that P.
// Measured this way on the npu_4K packs: 0.5B -> 8,708,288 (P=2), 1.7B -> 27,931,456 (P=4), 4B -> 55,308,096
// (P=4). The 0.5B hybrid pack shares the 0.5B hidden size and has stride 8,786,048, so this is not a table.
bool is_eps_header(const uint8_t* p, const uint8_t* end, unsigned short eps_bits) {
    // The pack's own bf16 epsilon little-endian, then zero padding: ten zero bytes is far beyond
    // chance for a 2-byte prefix but short enough that a genuine 64-byte header always qualifies.
    if (end - p < 12) return false;
    if (p[0] != uint8_t(eps_bits & 0xFF) || p[1] != uint8_t(eps_bits >> 8)) return false;
    for (int i = 2; i < 12; ++i)
        if (p[i] != 0) return false;
    return true;
}

std::vector<int64_t> eps_header_positions(const std::string& path, size_t max_bytes,
                                          unsigned short eps_bits) {
    std::ifstream f(path, std::ios::binary);
    std::vector<int64_t> out;
    if (!f) return out;
    const size_t kChunk = 8u << 20;
    std::vector<uint8_t> buf(kChunk + 64);
    size_t base = 0;
    while (base < max_bytes && out.size() < 96) {
        f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(kChunk));
        const size_t got = size_t(f.gcount());
        if (got == 0) break;
        const uint8_t* p = buf.data();
        const uint8_t* end = p + got;
        for (; p + 2 < end; ++p) {
            if (*p == uint8_t(eps_bits & 0xFF) && p[1] == uint8_t(eps_bits >> 8) &&
                is_eps_header(p, end, eps_bits))
                out.push_back(int64_t(base) + (p - buf.data()));
        }
        base += got;
    }
    return out;
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



Planned make(const std::string& name, const std::string& source, const std::string& dtype,
             std::vector<int64_t> shape) {
    Planned p;
    p.name = name;
    p.source = source;
    p.dtype = dtype;
    p.shape = std::move(shape);
    return p;
}


// The pack's cache/*_meta.json (the JSON twin of the .meta protobuf) declares, per op, its attrs
// N/K and every const tensor with dtype, shape and byte size. Reading it lets the plan check itself
// against the vendor's own numbers instead of only against our arithmetic (engine#734).
struct Declared {
    int64_t n = 0, k = 0;      // from op attrs, per projection
    int64_t bytes = 0;         // the preformat const tensor's size
};
using DeclaredMap = std::map<std::string, Declared>;

// `/model/layers.7/attn/q_proj/MatMulNBits` -> `model.layers.7.self_attn.q_proj`
std::string op_to_projection(const std::string& op) {
    std::string s = op;
    for (auto& c : s)
        if (c == '/') c = '.';
    while (!s.empty() && s.front() == '.') s.erase(s.begin());
    if (s.size() > 12 && s.compare(s.size() - 12, 12, ".MatMulNBits") == 0) s.erase(s.size() - 12);
    auto sub = [&](const char* from, const char* to) {
        const auto p = s.find(from);
        if (p != std::string::npos) s.replace(p, std::strlen(from), to);
    };
    sub(".attn.q_proj", ".self_attn.q_proj");
    sub(".attn.k_proj", ".self_attn.k_proj");
    sub(".attn.v_proj", ".self_attn.v_proj");
    sub(".attn.o_proj", ".self_attn.o_proj");
    sub(".mlp.", ".mlp.");
    return s;
}

DeclaredMap read_declared(const std::string& pack_dir) {
    DeclaredMap out;
    // `cache/` holds the compiler's per-op metadata (the JSON twin of the .meta protobuf) and is what makes
    // the declared-N/K cross-checks possible.  It is absent in the *hybrid* flavour of these packs -- the
    // ones that ship model.bin instead of dd_metastate_*.fconst -- and iterating a missing directory threw
    // std::filesystem_error straight out of the importer, so an unsupported pack surfaced as
    // "FAIL exception: filesystem error: directory iterator cannot open directory" instead of a diagnosis.
    // No cache means no declarations, which is exactly what an empty map says.
    const std::string cache_dir = pack_dir + "/cache";
    std::error_code ec;
    if (!fs::is_directory(cache_dir, ec)) {
        std::fprintf(stderr,
                     "note: %s has no cache/ (the compiler's per-op metadata), so the declared-N/K "
                     "cross-checks are skipped for this pack\n", pack_dir.c_str());
        return out;
    }
    for (const auto& e : fs::directory_iterator(cache_dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.size() < 10 || n.compare(n.size() - 10, 10, "_meta.json") != 0) continue;
        std::ifstream f(e.path());
        if (!f) continue;
        json d;
        try {
            d = json::parse(f);
        } catch (const std::exception&) {
            continue;
        }
        if (!d.contains("op_list")) continue;
        for (const auto& op : d["op_list"]) {
            if (!op.contains("type") || op["type"] != "MladfMatMul") continue;
            if (!op.contains("name")) continue;
            Declared dec;
            if (op.contains("attrs")) {
                const auto& a = op["attrs"];
                auto val = [&](const char* key) -> int64_t {
                    if (!a.contains(key)) return 0;
                    const auto& v = a[key];
                    if (v.contains("value") && v["value"].is_array() && !v["value"].empty())
                        return v["value"][0].is_string() ? std::atoll(v["value"][0].get<std::string>().c_str())
                                                         : v["value"][0].get<int64_t>();
                    return 0;
                };
                dec.n = val("N");
                dec.k = val("K");
            }
            const std::string proj = op_to_projection(op["name"].get<std::string>());
            if (op.contains("const_args") && d.contains("tensor_map")) {
                for (const auto& ca : op["const_args"]) {
                    const auto key = ca.get<std::string>();
                    if (key.find("qweight.preformat") == std::string::npos) continue;
                    if (!d["tensor_map"].contains(key)) continue;
                    const auto& t = d["tensor_map"][key];
                    if (t.contains("size_in_bytes")) dec.bytes = t["size_in_bytes"].get<int64_t>();
                }
            }
            if (dec.n || dec.k) out[proj] = dec;
        }
    }
    return out;
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
    d.rms_norm_eps = c.value("rms_norm_eps", 1e-6f);
    d.eps_bits = bf16_bits_of(d.rms_norm_eps);
    if (c.contains("eos_token_id")) {
        const auto& e = c["eos_token_id"];
        if (e.is_array()) d.eos = e.get<std::vector<int>>();
        else if (e.is_number_integer()) d.eos = {e.get<int>()};
    }
    return d;
}

}  // namespace

// engine#734: exported so the epsilon anchor itself can be tested against a crafted .fconst, without a pack.
// The two bytes a bf16 epsilon is stored as (little-endian): round the f32 to nearest-even bf16.
unsigned short bf16_bits_of(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

FconstGeom locate_fconst(const std::string& path, unsigned short eps_bits) {
    FconstGeom g;
    const auto pos = eps_header_positions(path, size_t(1) << 30, eps_bits);
    if (pos.size() < 6) {
        g.why = "fewer than 6 RMSNorm epsilon headers in the first GiB";
        return g;
    }
    // The header positions repeat with one period per layer block. Take the most common
    // pos[i+P] - pos[i] for each small P and require it to dominate: a header missed at a read-chunk
    // boundary must not defeat the measurement (it did on the 4B pack, where the strict test failed).
    for (int P = 1; P <= 8; ++P) {
        if (int(pos.size()) <= P + 2) break;
        std::map<int64_t, int> votes;
        int pairs = 0;
        for (size_t i = 0; i + P < pos.size(); ++i) {
            const int64_t d = pos[i + P] - pos[i];
            if (d < 1000000) continue;
            ++votes[d];
            ++pairs;
        }
        for (const auto& [d, n] : votes) {
            if (pairs < 4 || n * 10 < pairs * 8) continue;  // needs >= 80 % of the pairs
            if (!g.stride || d > g.stride) {                 // prefer the largest consistent period
                g.stride = d;
                g.period = P;
            }
        }
        if (g.stride) break;
    }
    if (!g.stride) {
        g.why = "the epsilon headers do not repeat with a single period";
        return g;
    }
    g.h1 = pos[1] - pos[0];
    if (g.h1 <= 0 || g.h1 >= g.stride) {
        g.why = "the second header is not inside the first block";
        g.stride = 0;
        return g;
    }
    // The rest of the first block's headers, which are its section boundaries. A block whose detected
    // period is 4 has three of them and plan() reads a different layout from it than from a 2-header
    // block, so they are recorded here rather than re-derived there.
    for (int i = 1; i < g.period; ++i) g.hdr.push_back(pos[i] - pos[0]);
    g.blocks = int(pos.size());
    return g;
}


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

// ── the .fconst attention payload: 64-row tiles of q/k/v/o ───────────────────
// amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §7.13 established this layout by known-answer write
// and §7.15 confirmed it against shipped tiles; the field positions below are the ones that survive the
// round trip below, which is run on every tile of every layer at import time.
//
// A tile is 2*(K/128) + 2 subvolumes of 2,176 B: one pair per (column half, K block of 128) plus the pair
// the writer puts its bias input in. Inside a weight subvolume:
//     +0             2,048 B of quants: byte (cp%4) + kk*4 + (cp/4)*512 holds rows 2*cp (low nibble)
//                    and 2*cp+1 (high nibble) at column kk of the block
//     +2,048         16 B of zero points, one nibble per (row/2) -- rows 0..15 low, 16..31 high
//     +2,112         32 bf16 scales, one per row of the half
// Codes and zero points are int4 in two's complement. An asymmetric 4-bit quantiser needs (code - zp), and
// the +8 offset every ONNX MatMulNBits applies cancels between the two fields, so the stored nibbles can be
// used as they are: w = (code_nibble - zp_nibble) * scale.
// (the attention payload's offset is not a constant: it is `kNormOff + 2 x hidden`, i.e. just past the
// block's input_layernorm weight -- see its two use sites)
constexpr int64_t kSubvBytes = 2176;  // 64 columns of K=128: 2,048 quants + 64 zeros + 32 bf16 scales
constexpr int64_t kZerosOff = 2048, kScalesOff = 2112;
constexpr int kTileRows = 64, kSubvRows = 32, kKBlock = 128;
// The MLP section writes gate and up as ONE tensor (rows interleaved in 32-column chunks) through the
// 4,288-byte instance of the same subvolume: 128 columns of K=128 = 4,096 quants + 64 zeros + 64 bf16
// scales. down_proj stays at the 2,176-byte width. See amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md
// §7.16.6 and §7.17.
constexpr int64_t kMlpSubvBytes = 4288;
constexpr int kMlpRows = 128;

// One subvolume's size at a given group width: 64*(rows/2) quants + 64 zero-point bytes + rows bf16
// scales. rows/2 is the subvolume's column count (32 -> 2,176 B, 64 -> 4,288 B).
int64_t subv_bytes(int rows) {
    if (rows <= 0 || rows % 64) return 0;
    return 64 * int64_t(rows / 2) + 64 + rows;
}

// One group of that layout: 2*(k/128) + 2 subvolumes. 0 when the shape cannot be groups at all.
int64_t group_bytes(int64_t k, int rows) {
    if (k <= 0 || k % kKBlock) return 0;
    const int64_t sv = subv_bytes(rows);
    if (sv == 0) return 0;
    return (2 * (k / kKBlock) + 2) * sv;
}

int64_t attn_tile_bytes(int64_t k) { return group_bytes(k, kTileRows); }
int64_t attn_payload_off(int64_t hidden) { return kNormOff + hidden * 2; }

TileW decode_tile(const uint8_t* tile, int k, int rows) {
    if (k <= 0 || k % kKBlock) throw std::runtime_error("tile: K is not a positive multiple of 128");
    const int64_t sv = subv_bytes(rows);
    if (sv == 0) throw std::runtime_error("tile: rows is not a positive multiple of 64");
    const int nsubv = rows / 2;  // columns per subvolume
    const int64_t zeros_off = 64 * int64_t(nsubv), scales_off = zeros_off + 64;
    const int kbs = k / kKBlock;
    TileW t;
    t.k = k;
    t.rows = rows;
    t.code.assign(size_t(rows) * size_t(k), 0);
    t.scale.assign(size_t(rows) * size_t(kbs), 0.0f);
    t.zp.assign(size_t(rows) * size_t(kbs), 0);
    t.bias.assign(rows, 0);
    for (int h = 0; h < 2; ++h) {
        const int r0 = h * nsubv;
        for (int i = 0; i < nsubv; ++i)  // the bias subvolumes are 0 and 1
            std::memcpy(&t.bias[size_t(r0 + i)], tile + int64_t(h) * sv + 2 * i, 2);
        for (int kb = 0; kb < kbs; ++kb) {
            const uint8_t* s = tile + (2 + 2 * kb + h) * sv;
            for (int cp = 0; cp < nsubv / 2; ++cp) {
                const int n0 = r0 + 2 * cp;
                const int64_t base = (cp % 4) + (cp / 4) * 512;
                for (int kk = 0; kk < kKBlock; ++kk) {
                    const uint8_t b = s[base + int64_t(kk) * 4];
                    t.code[size_t(n0) * size_t(k) + size_t(kb * kKBlock + kk)] = b & 0x0F;
                    t.code[size_t(n0 + 1) * size_t(k) + size_t(kb * kKBlock + kk)] = b >> 4;
                }
            }
            // The zero-point nibbles pair with rows exactly as the codes do (even/odd), not in halves:
            // a correct zero point sits where its 128 codes cluster, and on the shipped pack's own tiles
            // only even/odd satisfies that (corr(zp, mean code) +0.998 vs -0.023 for halves). FACTS §7.17.
            // nsubv/2 bytes cover nsubv rows per half at either subvolume width: 32 rows at 2,176 B
            // (attention, down_proj) and 64 at 4,288 B (the fused gate|up unit) -- the MLP payload is
            // checked with the same statistic in tests/npu_pack_import_test.cpp.
            for (int i = 0; i < nsubv / 2; ++i) {
                const uint8_t z = s[zeros_off + i];
                t.zp[size_t(r0 + 2 * i) * size_t(kbs) + size_t(kb)] = z & 0x0F;
                t.zp[size_t(r0 + 2 * i + 1) * size_t(kbs) + size_t(kb)] = z >> 4;
            }
            for (int i = 0; i < nsubv; ++i) {
                uint16_t h16;
                std::memcpy(&h16, s + scales_off + 2 * i, 2);
                t.scale[size_t(r0 + i) * size_t(kbs) + size_t(kb)] = bf16_to_f32(h16);
            }
        }
    }
    return t;
}

std::vector<uint8_t> encode_tile(const TileW& t) {
    if (t.k <= 0 || t.k % kKBlock) throw std::runtime_error("tile: K is not a positive multiple of 128");
    const int64_t sv = subv_bytes(t.rows);
    if (sv == 0) throw std::runtime_error("tile: rows is not a positive multiple of 64");
    const int nsubv = t.rows / 2, kbs = t.k / kKBlock;
    const int64_t zeros_off = 64 * int64_t(nsubv), scales_off = zeros_off + 64;
    std::vector<uint8_t> out(size_t(2 * kbs + 2) * size_t(sv), 0);
    for (int h = 0; h < 2; ++h) {
        const int r0 = h * nsubv;
        for (int i = 0; i < nsubv; ++i)
            std::memcpy(out.data() + int64_t(h) * sv + 2 * i, &t.bias[size_t(r0 + i)], 2);
        for (int kb = 0; kb < kbs; ++kb) {
            uint8_t* s = out.data() + (2 + 2 * kb + h) * sv;
            for (int cp = 0; cp < nsubv / 2; ++cp) {
                const int n0 = r0 + 2 * cp;
                const int64_t base = (cp % 4) + (cp / 4) * 512;
                for (int kk = 0; kk < kKBlock; ++kk) {
                    const size_t idx = size_t(n0) * size_t(t.k) + size_t(kb * kKBlock + kk);
                    s[base + int64_t(kk) * 4] = uint8_t((t.code[idx] & 0x0F) |
                                                        ((t.code[idx + size_t(t.k)] & 0x0F) << 4));
                }
            }
            for (int i = 0; i < nsubv / 2; ++i)  // rows 2i, 2i+1, the pairing decode_tile() reads
                s[zeros_off + i] = uint8_t((t.zp[size_t(r0 + 2 * i) * size_t(kbs) + size_t(kb)] & 0x0F) |
                                           ((t.zp[size_t(r0 + 2 * i + 1) * size_t(kbs) + size_t(kb)] & 0x0F) << 4));
            for (int i = 0; i < nsubv; ++i) {                const uint16_t h16 = f32_to_bf16(t.scale[size_t(r0 + i) * size_t(kbs) + size_t(kb)]);
                std::memcpy(s + scales_off + 2 * i, &h16, 2);
            }
        }
    }
    return out;
}

Q4 q_from_tiles(const std::vector<TileW>& tiles) {
    if (tiles.empty()) throw std::runtime_error("no tiles");
    const int k = tiles.at(0).k, kbs = k / kKBlock, rows = tiles.at(0).rows;
    for (const auto& t : tiles)
        if (t.k != k || t.rows != rows) throw std::runtime_error("tiles of different shape");
    Q4 q;
    q.K = k;
    q.block = kKBlock;
    q.N = int(tiles.size()) * rows;
    q.codes.assign(size_t(q.N) * size_t(k), 0);
    q.scale.assign(size_t(q.N) * size_t(k / 32), 0.0f);
    q.zero.assign(size_t(q.N) * size_t(k / 32), 0.0f);
    const int gpb = kKBlock / 32;  // a block's scale and zero point apply to its four 32-column groups
    for (size_t ti = 0; ti < tiles.size(); ++ti) {
        const TileW& t = tiles[ti];
        for (int r = 0; r < rows; ++r) {
            const int row = int(ti) * rows + r;
            for (int i = 0; i < k; ++i) q.codes[size_t(row) * size_t(k) + size_t(i)] = t.code[size_t(r) * size_t(k) + size_t(i)];
            for (int kb = 0; kb < kbs; ++kb) {
                const float s = t.scale[size_t(r) * size_t(kbs) + size_t(kb)];
                const float z = -float(t.zp[size_t(r) * size_t(kbs) + size_t(kb)]) * s;  // (code - zp) * s
                for (int g = 0; g < gpb; ++g) {
                    const size_t idx = size_t(row) * size_t(k / 32) + size_t(kb * gpb + g);
                    q.scale[idx] = s;
                    q.zero[idx] = z;
                }
            }
        }
    }
    return q;
}

TileW gu_half(const TileW& group, bool up) {
    if (group.rows != kMlpRows) throw std::runtime_error("gu_half: the group is not 128 columns wide");
    const int k = group.k, kbs = k / kKBlock, n = 32;
    TileW t;
    t.k = k;
    t.rows = 2 * n;
    t.code.resize(size_t(t.rows) * size_t(k));
    t.scale.resize(size_t(t.rows) * size_t(kbs));
    t.zp.resize(size_t(t.rows) * size_t(kbs));
    t.bias.resize(t.rows);
    for (int h = 0; h < 2; ++h) {
        const int s0 = h * 64 + (up ? n : 0), d0 = h * n;
        std::memcpy(&t.code[size_t(d0) * k], &group.code[size_t(s0) * k], size_t(n) * size_t(k));
        std::memcpy(&t.scale[size_t(d0) * kbs], &group.scale[size_t(s0) * kbs],
                    size_t(n) * size_t(kbs) * sizeof(float));
        std::memcpy(&t.zp[size_t(d0) * kbs], &group.zp[size_t(s0) * kbs], size_t(n) * size_t(kbs));
        std::memcpy(&t.bias[d0], &group.bias[s0], size_t(n) * sizeof(uint16_t));
    }
    return t;
}

// ── the .fconst MLP section ────────────────────────────────────────────────
// The section fills the rest of the layer block after the second norm, in three parts:
//   gate|up   ONE fused tensor of hidden x 2*intermediate columns, its rows interleaved in 32-column
//             chunks (gate 32, up 32, gate 32, ...), in 128-column groups of 4,288-byte subvolumes
//   ...then   a single 64-byte parameter subvolume (`01 00 ..`, the same bytes in all 24 layer blocks)
//   down      intermediate x hidden, in 64-column groups of 2,176-byte subvolumes
// The offsets are not constants: the three parts have to fill the block exactly, and that identity is
// what turns the geometry from an assumption about one pack into a measurement (amd-oss/npu-re
// docs/FACTS-oga-pack-weight-layout.md §7.17).
struct MlpGeom {
    int64_t off = 0, gu = 0, gap = 64, down = 0, total = 0;
    int64_t gu_units = 0, down_groups = 0;
    std::string why;  // empty when it adds up
    bool ok() const { return why.empty(); }
};

MlpGeom mlp_geom(int64_t hidden, int64_t inter, int64_t mlp_off, int64_t stride) {
    MlpGeom m;
    m.off = mlp_off;
    const int64_t gu_unit = group_bytes(hidden, kMlpRows);     // 128 columns of K=hidden
    const int64_t down_group = group_bytes(inter, kTileRows);  // 64 columns of K=intermediate
    if (inter <= 0) {
        m.why = "config.json gives no intermediate_size";
        return m;
    }
    if (gu_unit == 0) {
        m.why = "hidden " + std::to_string(hidden) + " is not a positive multiple of 128, so the fused "
                "gate/up section is not 128-column groups of this shape";
        return m;
    }
    if (down_group == 0) {
        m.why = "intermediate " + std::to_string(inter) + " is not a positive multiple of 128, so "
                "down_proj is not 64-column groups of this shape";
        return m;
    }
    if (inter % kTileRows || hidden % kTileRows) {
        m.why = "intermediate " + std::to_string(inter) + " / hidden " + std::to_string(hidden) +
                " are not multiples of 64, which the fused gate/up split needs";
        return m;
    }
    m.gu_units = inter / kTileRows;
    m.down_groups = hidden / kTileRows;
    m.gu = m.gu_units * gu_unit;
    m.down = m.down_groups * down_group;
    m.total = m.off + m.gu + m.gap + m.down;
    if (m.total != stride)
        m.why = "the MLP section ends at " + std::to_string(m.total) + " B in a " + std::to_string(stride) +
                "-byte layer block, so its geometry is not the one this decoder was established on";
    return m;
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
    for (const auto& t : tensors) {
        json e = {{"name", t.name}, {"source", t.source}, {"dtype", t.dtype}, {"shape", t.shape},
                  {"refusal", t.refusal}};
        if (t.declared_n) e["declared_n"] = t.declared_n;
        if (t.declared_k) e["declared_k"] = t.declared_k;
        if (!t.declared_check.empty()) e["declared_check"] = t.declared_check;
        ts.push_back(std::move(e));
    }
    j["tensors"] = ts;
    j["attention_payload"] = {{"tiles_measured", attn_tiles},
                             {"o_proj_k", o_k},
                             {"qk_norm", knorm_off != 0},
                             {"note", attn_note}};
    j["mlp_section"] = {{"gate_up_units_measured", mlp_gu_units},
                        {"down_groups_measured", mlp_down_groups},
                        {"note", mlp_note}};
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

    // norms: the layer blocks are located by their epsilon anchors at import time, so the offsets work
    // for any hidden size (engine#734). plan() cannot prove that without reading the file, so the
    // refusal here is empty and import_pack reports what the anchors actually found.
    const std::string norm_source = "fconst:Llm_Token_Token";
    const std::string norm_refusal;
    for (int L = 0; L < d.layers; ++L) {
        auto a = make("model.layers." + std::to_string(L) + ".input_layernorm.weight", norm_source, "BF16", {H});
        auto b = make("model.layers." + std::to_string(L) + ".post_attention_layernorm.weight", norm_source,
                      "BF16", {H});
        a.refusal = norm_refusal;
        b.refusal = norm_refusal;
        p.tensors.push_back(std::move(a));
        p.tensors.push_back(std::move(b));
    }
    // The .fconst tail: the file's layer blocks stop at `layers x stride`, and what follows is a 1,856-byte
    // envelope `[eps bf16 1e-6][62 pad][hidden x bf16]` -- the final norm -- and, in the Token_Token file,
    // the lm_head tile stream (§7.15). Measured on all three packs here: the epsilon is at `layers x stride`
    // and the norm at +64, and where both .fconst files exist the two copies are byte-identical. The
    // position is confirmed at import time by the epsilon marker plus that cross-check.
    auto fin = make("model.norm.weight", norm_source, "BF16", {H});
    p.tensors.push_back(std::move(fin));

    // projections: the attention payload (q/k/v/o) decodes tile by tile and the MLP section group by
    // group -- both layouts are established and the import re-encodes every group it reads and requires
    // the same bytes back. A pack whose sections do not add up to the block stride is refused by name.
    //
    // The tile count is MEASURED here, not derived: the attention payload is whatever sits between the
    // block's first epsilon header and its second one, and a pack whose payload is not a whole number of
    // 64-row tiles (or whose q/k/v/o row split does not add up to them) is refused rather than guessed at.
    const int q_out = d.heads * d.head_dim, kv_out = d.kv_heads * d.head_dim;
    {
        const std::string tok = find_one(pack_dir, "dd_metastate_Llm_Token_Token_", ".fconst");
        const int64_t tile_bytes = attn_tile_bytes(H);
        if (tok.empty()) {
            p.attn_note = "no dd_metastate_Llm_Token_Token_*.fconst in the pack";
            p.mlp_note = p.attn_note;
        } else if (tile_bytes == 0) {
            p.attn_note = "hidden " + std::to_string(H) + " is not a multiple of 128, so the attention "
                                                                  "payload is not 64-row tiles of this shape";
            p.mlp_note = p.attn_note;
        } else {
            const FconstGeom g = locate_fconst(tok, p.dims.eps_bits);
            if (g.stride == 0) {
                p.attn_note = "layer blocks not located: " + g.why;
                p.mlp_note = p.attn_note;
            } else {
                // The payload starts just past the block's input_layernorm weight, so its offset is
                // `attn_payload_off(hidden)` (engine#979). How the rest of the section is divided is what
                // the block's own header count says: q/k/v/o in one uniform tile run (Qwen2.5, Llama), or
                // q/k/v, then a per-head QK norm each, then o_proj at K = q_proj's width (Qwen3). Every
                // section is checked against that arithmetic to the byte; a pack that does not match is
                // refused with the numbers rather than decoded on a guess.
                const int64_t attn_off = attn_payload_off(H);
                const int64_t per_head = int64_t(d.head_dim) * 2;  // one QK norm gain, bf16
                const int64_t tile_o = attn_tile_bytes(q_out);
                const int64_t n_kv = kv_out / 64, n_q = q_out / 64;
                auto refuse = [&](const std::string& s) {
                    p.attn_note = s;
                    p.knorm_off = p.qnorm_off = p.o_off = 0;  // an unread section is not a half-read one
                    p.o_k = 0;
                };
                if (g.hdr.size() == 1) {
                    // One header after the block's first: q, k, v and o share one uniform payload, so
                    // o_proj's K is hidden and q_proj has to be hidden wide for the run to be uniform.
                    const int64_t payload = g.hdr[0] - attn_off;
                    const int64_t want = n_kv * 2 + n_q * 2;  // k, q, v, o = 2/14/2/14 on 0.5B
                    if (payload <= 0 || payload % tile_bytes)
                        refuse("attention payload is " + std::to_string(payload) + " B, not a whole number of " +
                               std::to_string(tile_bytes) + "-byte tiles");
                    else if (q_out != H)
                        refuse("the attention payload holds q/k/v/o as one uniform tile run, which needs "
                               "q_proj's width to be hidden; this pack's q_proj is " + std::to_string(q_out) +
                               " rows against hidden " + std::to_string(H) +
                               ", so o_proj's K would not be the tile's K");
                    else if (payload / tile_bytes != want)
                        refuse("attention payload holds " + std::to_string(payload / tile_bytes) +
                               " tiles; q/k/v/o at 64 rows each need " + std::to_string(want));
                    else {
                        p.attn_tiles = payload / tile_bytes;
                        p.o_off = attn_off + (n_kv + n_q + n_kv) * tile_bytes;
                        p.o_k = int(H);
                    }
                } else if (g.hdr.size() == 3) {
                    // engine#979: the Qwen3 families split the attention section in four. Measured on
                    // both shipped packs: [k,q,v tiles][header + per-head k_norm][header + per-head q_norm
                    // + o_proj tiles], with o_proj's tiles at K = q_proj's output width.
                    const int64_t kv_tiles = n_kv + n_q + n_kv;
                    const int64_t s0 = g.hdr[0] - attn_off;
                    const int64_t s1 = g.hdr[1] - g.hdr[0];
                    const int64_t s2 = g.hdr[2] - g.hdr[1];
                    if (s0 != kv_tiles * tile_bytes)
                        refuse("the attention payload before the first per-head norm is " + std::to_string(s0) +
                               " B; k/q/v at 64 rows each are " + std::to_string(kv_tiles) + " tiles = " +
                               std::to_string(kv_tiles * tile_bytes) + " B");
                    else if (s1 != kNormOff + per_head)
                        refuse("the section between the two per-head QK norms is " + std::to_string(s1) +
                               " B; a " + std::to_string(kNormOff) + "-byte header plus a bf16 [head_dim " +
                               std::to_string(d.head_dim) + "] gain is " + std::to_string(kNormOff + per_head));
                    else if (s2 != kNormOff + per_head + (H / 64) * tile_o)
                        refuse("the section holding o_proj is " + std::to_string(s2) + " B; a header, a gain and " +
                               std::to_string(H / 64) + " o_proj tiles (hidden " + std::to_string(H) +
                               " rows) at K = q_proj's " + std::to_string(q_out) + " are " +
                               std::to_string(kNormOff + per_head + (H / 64) * tile_o));
                    else {
                        p.attn_tiles = kv_tiles;
                        p.knorm_off = g.hdr[0] + kNormOff;
                        p.qnorm_off = g.hdr[1] + kNormOff;
                        p.o_off = g.hdr[1] + kNormOff + per_head;
                        p.o_k = q_out;
                    }
                } else {
                    refuse("the layer block has " + std::to_string(g.hdr.size() + 1) +
                           " epsilon headers, and this importer reads blocks of 2 (q/k/v/o in one payload) "
                           "or 4 (q/k/v, two per-head QK norms, o_proj) -- recorded 2026-11-04 on Qwen3");
                }
                // the MLP section starts right after the block's post-attention norm
                const int64_t mlp_off = g.post_attn_hdr() + kNormOff + H * 2;
                const MlpGeom mg = mlp_geom(H, d.intermediate, mlp_off, g.stride);
                p.mlp_note = mg.why;
                if (mg.ok()) {
                    p.mlp_gu_units = mg.gu_units;
                    p.mlp_down_groups = mg.down_groups;
                    p.mlp_off = mlp_off;
                }
            }
        }
    }
    struct Proj { const char* suffix; int rows, cols; };
    const Proj projs[] = {{"self_attn.q_proj", q_out, d.hidden},
                          {"self_attn.k_proj", kv_out, d.hidden},
                          {"self_attn.v_proj", kv_out, d.hidden},
                          {"self_attn.o_proj", d.hidden, q_out},
                          {"mlp.gate_proj", d.intermediate, d.hidden},
                          {"mlp.up_proj", d.intermediate, d.hidden},
                          {"mlp.down_proj", d.hidden, d.intermediate}};
    const DeclaredMap declared = read_declared(pack_dir);
    for (int L = 0; L < d.layers; ++L)
        for (const auto& pr : projs) {
            const std::string name = "model.layers." + std::to_string(L) + "." + pr.suffix + ".weight";
            auto x = make(name, "fconst:Llm_Token_Token", "q4_1", {pr.rows, pr.cols});
            const bool attention = std::string(pr.suffix).compare(0, 9, "self_attn") == 0;
            if (!attention) x.refusal = p.mlp_note;
            else if (!p.attn_note.empty()) x.refusal = p.attn_note;
            const std::string key = "model.layers." + std::to_string(L) + "." + pr.suffix;
            if (const auto it = declared.find(key); it != declared.end()) {
                x.declared_n = it->second.n;
                x.declared_k = it->second.k;
                if (it->second.n && it->second.n != pr.rows)
                    x.declared_check = "N: plan " + std::to_string(pr.rows) + " vs pack " +
                                       std::to_string(it->second.n);
                else if (it->second.k && it->second.k != pr.cols)
                    x.declared_check = "K: plan " + std::to_string(pr.cols) + " vs pack " +
                                       std::to_string(it->second.k);
            }
            p.tensors.push_back(std::move(x));
        }
    // the attention biases are per-row data inside the tiles' bias subvolumes, byte-identical to the pack's
    // own values; they are planned only when the attention payload itself is readable
    for (int L = 0; L < d.layers; ++L)
        for (const auto& [suffix, rows] : {std::pair<const char*, int>{"q_proj", q_out},
                                           std::pair<const char*, int>{"k_proj", kv_out},
                                           std::pair<const char*, int>{"v_proj", kv_out}}) {
            auto b = make("model.layers." + std::to_string(L) + ".self_attn." + suffix + ".bias",
                          "fconst:Llm_Token_Token", "BF16", {rows});
            if (!p.attn_note.empty()) b.refusal = p.attn_note;
            b.declared_check = "written only when the tile's bias subvolumes are not all zero";
            p.tensors.push_back(std::move(b));
        }
    // per-head QK RMSNorm gains (Qwen3 and later): bf16 [head_dim], in the two sections that sit between
    // q/k/v and o_proj. Planned only when the block measured those two sections (engine#979).
    if (p.knorm_off)
        for (int L = 0; L < d.layers; ++L)
            for (const char* nm : {"self_attn.q_norm.weight", "self_attn.k_norm.weight"})
                p.tensors.push_back(make("model.layers." + std::to_string(L) + "." + nm, norm_source,
                                         "BF16", {d.head_dim}));
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

// one dim of a protobuf shape as a usable count: positive and not absurd
uint64_t dim_count(const std::string& what, int64_t v) {
    if (v <= 0) throw std::runtime_error(what + ": dimension " + std::to_string(v) + " is not positive");
    return uint64_t(v);
}

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
    // Every read below takes its *length* from the file's own declared metadata (protobuf dims, the node's
    // N/K/block), so a length is only honoured when it is proved to lie inside the tensor. bytes_of(t, need)
    // is the single gate: `need` is computed through mul_ok so the multiply cannot wrap, and it must fit in
    // what the tensor actually carries (raw_len for an embedded tensor, its data_offsets span for external
    // data). A blanket product(dims) == length check is not the rule: 4-bit packed weights deliberately do
    // not satisfy it, so each call site states what it reads -- and `need` has no default, so a reader added
    // later cannot opt out of the check by omitting it.
    auto bytes_of = [&](const OnnxTensor& t, uint64_t need) -> const uint8_t* {
        if (t.raw) {
            if (need > t.raw_len)
                throw std::runtime_error(t.name + ": needs " + std::to_string(need) + " B, carries " +
                                         std::to_string(t.raw_len));
            return t.raw;
        }
        const auto it = ext.find(t.location);
        if (it == ext.end()) throw std::runtime_error(t.name + ": no data");
        if (!ext_range_ok(ext_size[t.location], t.offset, t.length))
            throw std::runtime_error(t.name + ": external data past the end of " + t.location);
        if (need > t.length)
            throw std::runtime_error(t.name + ": needs " + std::to_string(need) + " B, declares " +
                                     std::to_string(t.length) + " in " + t.location);
        return it->second + t.offset;
    };

    fs::create_directories(out_dir);
    Writer w;
    w.dir = out_dir;

    const OnnxTensor* embed = nullptr;
    for (const auto& t : onnx.tensors)
        if (t.name == "model.embed_tokens.weight") embed = &t;
    if (!embed) throw std::runtime_error("no model.embed_tokens.weight");
    if (embed->dims.size() != 2) throw std::runtime_error("model.embed_tokens.weight is not 2-D");
    const int64_t V = embed->dims.at(0), H = embed->dims.at(1);
    const uint64_t Vc = dim_count("model.embed_tokens.weight", V), Hc = dim_count("model.embed_tokens.weight", H);
    uint64_t elems = 0;
    if (!mul_ok(Vc, Hc, elems)) throw std::runtime_error("model.embed_tokens.weight: V*H overflows");
    const uint64_t src_width = embed->type == kF32 ? 4 : 2;
    uint64_t embed_need = 0;
    if (!mul_ok(elems, src_width, embed_need))
        throw std::runtime_error("model.embed_tokens.weight: V*H*width overflows");
    const uint8_t* embed_bytes = bytes_of(*embed, embed_need);
    w.add("model.embed_tokens.weight", "BF16", {V, H}, as_bf16(embed_bytes, embed->type, size_t(elems)));
    ++r.tensors;

    for (const auto& mm : onnx.matmuls) {
        const auto wi = onnx.index.find(mm.weight), si = onnx.index.find(mm.scales);
        if (wi == onnx.index.end() || si == onnx.index.end()) continue;
        if (mm.bits != 4) throw std::runtime_error("lm_head: MatMulNBits is not 4-bit");
        const auto& wt = onnx.tensors[wi->second];
        const auto& st = onnx.tensors[si->second];
        const auto zi = onnx.index.find(mm.zeros);
        const bool have_zp = zi != onnx.index.end();

        if (mm.N <= 0 || mm.K <= 0 || mm.block <= 0 || mm.K % mm.block)
            throw std::runtime_error("lm_head: bad MatMulNBits geometry N=" + std::to_string(mm.N) +
                                     " K=" + std::to_string(mm.K) + " block=" + std::to_string(mm.block));
        const uint64_t groups = uint64_t(mm.K) / uint64_t(mm.block);
        const uint64_t nrows = uint64_t(mm.N);
        uint64_t groups_n = 0;   // N * K/block
        if (!mul_ok(nrows, groups, groups_n)) throw std::runtime_error("lm_head: N*K/block overflows");
        std::vector<float> scales(static_cast<std::vector<float>::size_type>(groups_n));
        const uint64_t scale_width = st.type == kF32 ? 4 : 2;
        uint64_t scales_need = 0;
        if (!mul_ok(groups_n, scale_width, scales_need))
            throw std::runtime_error("lm_head: scales length overflows");
        if (st.type == kF32) std::memcpy(scales.data(), bytes_of(st, scales_need), scales.size() * 4);
        else {
            const auto b = as_bf16(bytes_of(st, scales_need), st.type, scales.size());
            const auto* d = reinterpret_cast<const uint16_t*>(b.data());
            for (size_t i = 0; i < scales.size(); ++i) scales[i] = bf16_to_f32(d[i]);
        }
        uint64_t zp_need = 0;   // one nibble per (row, group)
        if (!mul_ok(nrows, (groups + 1) / 2, zp_need)) throw std::runtime_error("lm_head: qzeros length overflows");
        const uint8_t* zp = have_zp ? bytes_of(onnx.tensors[zi->second], zp_need) : nullptr;
        uint64_t qw_need = 0;   // N * K/block * block/2
        if (!mul_ok(groups_n, uint64_t(mm.block) / 2, qw_need)) throw std::runtime_error("lm_head: qweight length overflows");
        const Q4 q = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), zp, int(mm.N), int(mm.K), mm.block);

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
            const Q4 swap = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), zp, int(mm.N), int(mm.K),
                                               mm.block, /*swap_nibbles*/ true, false);
            const Q4 nozp = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), nullptr, int(mm.N), int(mm.K),
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

    {
        const std::string path = find_one(pack_dir, "dd_metastate_Llm_Token_Token_", ".fconst");
        const FconstGeom g = locate_fconst(path, p.dims.eps_bits);
        // refusals decided while reading the payload (a tile that does not survive its own round trip,
        // or an o_proj bias the container has nowhere to put); they land before anything is written
        std::vector<std::string> late;
        long long zero_inexact_all = 0, zero_groups_all = 0;
        long long mlp_zero_inexact = 0, mlp_zero_groups = 0;
        if (g.stride == 0) {
            r.fconst_note = "layer blocks not located: " + g.why;
        } else {
            r.stride = g.stride;
            r.second_header = g.h1;
            Fconst fc = open_fconst(path);
            auto add_norm = [&](int64_t off, const std::string& name) {
                w.add(name, "BF16", {H}, as_bf16(at(fc, off, H * 2), kBF16, size_t(H)));
                ++r.tensors;
            };
            for (int L = 0; L < p.dims.layers; ++L) {
                const int64_t base = int64_t(L) * g.stride;
                r.norm_anchors_total += 2;
                r.norm_anchors_ok += int(eps_at(fc, base, p.dims.eps_bits)) +
                                 int(eps_at(fc, base + g.post_attn_hdr(), p.dims.eps_bits));
                add_norm(base + kNormOff, "model.layers." + std::to_string(L) + ".input_layernorm.weight");
                add_norm(base + g.post_attn_hdr() + kNormOff,
                         "model.layers." + std::to_string(L) + ".post_attention_layernorm.weight");
            }
            // the final norm: the tail envelope opens at `layers x stride`, the norm sits at +64, and the
            // prefill file (when the pack ships one) holds a second copy that has to match byte for byte
            {
                const int64_t tail = int64_t(p.dims.layers) * g.stride;
                if (!eps_at(fc, tail, p.dims.eps_bits)) {
                    late.push_back("model.norm.weight: no RMSNorm epsilon at " + std::to_string(tail) +
                                   " B, where the .fconst tail envelope opens, so its position is not "
                                   "established for this pack");
                } else {
                    const uint8_t* v = at(fc, tail + kNormOff, H * 2);
                    w.add("model.norm.weight", "BF16", {H}, as_bf16(v, kBF16, size_t(H)));
                    ++r.tensors;
                    r.final_norm_imported = true;
                    const std::string pre = find_one(pack_dir, "dd_metastate_Llm_Prefill_", ".fconst");
                    if (!pre.empty()) {
                        // the prefill program has its own stride, so its envelope sits at its own
                        // `layers x stride` -- measured the same way, from its own epsilon anchors
                        const FconstGeom pg = locate_fconst(pre, p.dims.eps_bits);
                        const int64_t ptail = int64_t(p.dims.layers) * pg.stride;
                        // The duplicate is optional evidence, so a prefill file that cannot even be mapped
                        // (empty, truncated, unreadable) must leave the cross-check unavailable rather than
                        // abort an import whose Token_Token copy is fine.
                        bool mapped = false;
                        Fconst pf;
                        try {
                            pf = open_fconst(pre);
                            mapped = true;
                        } catch (const std::exception& e) {
                            std::fprintf(stderr,
                                         "1bit npu-pack: note: the prefill .fconst could not be read (%s), so the "
                                         "final norm has no cross-check (the import uses the Token_Token copy)\n",
                                         e.what());
                        }
                        const bool there = mapped && pg.stride > 0 && eps_at(pf, ptail, p.dims.eps_bits) &&
                                           size_t(ptail + kNormOff + H * 2) <= pf.size;
                        r.prefill_tail_seen = there;
                        r.final_norm_crosscheck =
                            there && std::memcmp(at(pf, ptail + kNormOff, H * 2), v, size_t(H * 2)) == 0;
                        // The cross-check compares two copies the VENDOR ships; the one that matters is the
                        // Token_Token tail, because that is the program this importer reads the norm for and
                        // the one the device runs. So a mismatch is REPORTED, not fatal: refusing the import
                        // over a duplicated copy would make an otherwise complete pack unimportable for a
                        // vendor-side inconsistency we cannot repair. (Measured on this box: the 0.5B pack's
                        // prefill .fconst, rewritten 2026-11-03 15:10, differs from its Token copy in exactly
                        // one bf16 of model.norm.weight -- index 895, 9.375 vs -0.1465 -- and every other one
                        // of the 896 matches; the container is unaffected because the norm comes from Token.)
                        if (!there)
                            std::fprintf(stderr,
                                         "1bit npu-pack: note: the prefill .fconst has no readable tail envelope at "
                                         "%lld B, so the final norm has no cross-check (the import uses the "
                                         "Token_Token copy)\n",
                                         (long long)ptail);
                        else if (!r.final_norm_crosscheck)
                            std::fprintf(stderr,
                                         "1bit npu-pack: WARNING: the Token_Token and Llm_Prefill tails disagree, "
                                         "so the pack's two copies of model.norm.weight are not the same bytes; "
                                         "the container uses the Token_Token copy\n");
                        if (mapped) unmap_file(pf.map, pf.size);
                    }
                }
            }
            // the final norm is refused in the plan only when the block geometry itself is unknown

            // ── the attention payload: q/k/v/o ─────────────────────────────────────────────────────
            // The projection order is the one measured on the shipped pack: each tile's bias subvolumes
            // matched the pack's own q/k/v bias bytes exactly, and that is what identified the runs
            // (k 2 tiles, q 14, v 2, o 14 at hidden 896). Every tile read here is decoded, re-encoded and
            // compared with the bytes it came from before any of it is written: a tile that does not
            // survive that is a tile whose fields we do not understand, and it is refused by name.
            if (p.attn_tiles) {
                const int q_out = p.dims.heads * p.dims.head_dim;
                const int kv_out = p.dims.kv_heads * p.dims.head_dim;
                const int n_kv = kv_out / kTileRows, n_q = q_out / kTileRows;
                // k, q, v share the block's first payload at K = hidden; o_proj has a base and a K of its
                // own (engine#979: on the Qwen3 families the two per-head QK norms sit between them and
                // o_proj's K is q_proj's output width).
                struct Slot { const char* proj; int rows; int64_t base; int k; int first; };
                const int64_t attn_off = attn_payload_off(H);
                const Slot slots[] = {{"k_proj", kv_out, attn_off, int(H), 0},
                                      {"q_proj", q_out, attn_off, int(H), n_kv},
                                      {"v_proj", kv_out, attn_off, int(H), n_kv + n_q},
                                      {"o_proj", int(H), p.o_off, p.o_k, 0}};
                for (int L = 0; L < p.dims.layers; ++L) {
                    const int64_t block = int64_t(L) * g.stride;
                    for (const auto& sl : slots) {
                        const int nt = sl.rows / kTileRows;
                        const int tile_bytes = int(attn_tile_bytes(sl.k));
                        std::vector<TileW> tiles;
                        tiles.reserve(size_t(nt));
                        bool exact = true;
                        for (int i = 0; i < nt; ++i) {
                            const uint8_t* src = at(fc, block + sl.base + int64_t(sl.first + i) * tile_bytes,
                                                    tile_bytes);
                            tiles.push_back(decode_tile(src, sl.k));
                            ++r.attn_tiles_total;
                            const std::vector<uint8_t> back = encode_tile(tiles.back());
                            if (std::memcmp(back.data(), src, size_t(tile_bytes)) == 0) ++r.attn_tiles_ok;
                            else exact = false;
                        }
                        const std::string name = "model.layers." + std::to_string(L) + ".self_attn." + sl.proj;
                        if (!exact) {
                            late.push_back(name + ".weight: the .fconst tile does not survive decode->encode, "
                                                   "so its fields are not understood");
                            continue;
                        }
                        bool any_bias = false;
                        for (const auto& t : tiles)
                            for (uint16_t v : t.bias) any_bias |= (v != 0);
                        if (std::string(sl.proj) == "o_proj" && any_bias) {
                            late.push_back(name + ".weight: o_proj's bias subvolumes are not empty and the "
                                                   "container has no o_proj bias");
                            continue;
                        }
                        Q4 q = q_from_tiles(tiles);
                        std::vector<uint8_t> tiled;
                        long long zinexact = 0;
                        tile_q4_1(q, tiled, &zinexact);
                        zero_inexact_all += zinexact;
                        zero_groups_all += (long long)q.scale.size();
                        const int64_t nt_tiles = int64_t(tiled.size() / kTileBytes);
                        r.tiles += nt_tiles;
                        w.add(name + ".weight", "I8", {nt_tiles, int64_t(kTileBytes)}, std::move(tiled));
                        ++r.tensors;
                        ++r.attn_projections;
                        if (std::string(sl.proj) != "o_proj" && any_bias) {
                            std::vector<uint8_t> bytes;
                            bytes.reserve(size_t(sl.rows) * 2);
                            for (const auto& t : tiles)
                                for (uint16_t v : t.bias) {
                                    bytes.push_back(uint8_t(v & 0xFF));
                                    bytes.push_back(uint8_t(v >> 8));
                                }
                            w.add(name + ".bias", "BF16", {sl.rows}, std::move(bytes));
                            ++r.tensors;
                            ++r.attn_bias_tensors;
                        }
                    }
                    // per-head QK norms (Qwen3 and later): plain bf16 [head_dim] gains that the
                    // container already carries for families that have them.
                    if (p.knorm_off)
                        for (const auto& [off, nm] : {std::pair<int64_t, const char*>{p.knorm_off, "k_norm"},
                                                      std::pair<int64_t, const char*>{p.qnorm_off, "q_norm"}}) {
                            const int64_t n = p.dims.head_dim;
                            w.add("model.layers." + std::to_string(L) + ".self_attn." + nm + ".weight",
                                  "BF16", {n}, as_bf16(at(fc, block + off, n * 2), kBF16, size_t(n)));
                            ++r.tensors;
                        }
                }
                r.attn_zero_inexact_pct =
                    zero_groups_all ? 100.0 * double(zero_inexact_all) / double(zero_groups_all) : 0.0;
            }

            // ── the MLP section: gate|up fused, then down_proj ───────────────────────────────────
            // Same contract as the attention payload: every group read here is decoded, re-encoded and
            // compared with the bytes it came from before any of it is written, and a group that does
            // not survive that is refused by name. The geometry is re-derived from this block rather
            // than taken from plan(), so the stride identity is checked against the file being read.
            if (p.mlp_gu_units) {
                const MlpGeom mg = mlp_geom(H, p.dims.intermediate, p.mlp_off, g.stride);
                if (!mg.ok()) {
                    late.push_back("the MLP section: " + mg.why);
                } else {
                    const int inter = p.dims.intermediate;
                    const int64_t gu_unit = group_bytes(H, kMlpRows);
                    const int64_t down_group = group_bytes(inter, kTileRows);
                    for (int L = 0; L < p.dims.layers; ++L) {
                        const int64_t base_gu = int64_t(L) * g.stride + mg.off;
                        const int64_t base_down = base_gu + mg.gu + mg.gap;
                        std::vector<TileW> gate, up, down;
                        bool gate_ok = true, up_ok = true, down_ok = true;
                        bool gate_bias = false, up_bias = false, down_bias = false;
                        for (int u = 0; u < int(mg.gu_units); ++u) {
                            const uint8_t* src = at(fc, base_gu + int64_t(u) * gu_unit, gu_unit);
                            const TileW grp = decode_tile(src, int(H), kMlpRows);
                            ++r.mlp_groups_total;
                            const std::vector<uint8_t> back = encode_tile(grp);
                            if (back.size() == size_t(gu_unit) &&
                                std::memcmp(back.data(), src, size_t(gu_unit)) == 0)
                                ++r.mlp_groups_ok;
                            else
                                gate_ok = up_ok = false;
                            gate.push_back(gu_half(grp, false));
                            up.push_back(gu_half(grp, true));
                        }
                        for (int gi = 0; gi < int(mg.down_groups); ++gi) {
                            const uint8_t* src = at(fc, base_down + int64_t(gi) * down_group, down_group);
                            const TileW grp = decode_tile(src, inter, kTileRows);
                            ++r.mlp_groups_total;
                            const std::vector<uint8_t> back = encode_tile(grp);
                            if (back.size() == size_t(down_group) &&
                                std::memcmp(back.data(), src, size_t(down_group)) == 0)
                                ++r.mlp_groups_ok;
                            else
                                down_ok = false;
                            down.push_back(std::move(grp));
                        }
                        auto emit = [&](const std::vector<TileW>& tiles, const char* suffix, bool exact,
                                        bool& bias_flag) {
                            const std::string name = "model.layers." + std::to_string(L) + "." + suffix;
                            for (const auto& t : tiles)
                                for (uint16_t v : t.bias) bias_flag |= (v != 0);
                            if (!exact) {
                                late.push_back(name + ".weight: the .fconst group does not survive "
                                                            "decode->encode, so its fields are not understood");
                                return;
                            }
                            if (bias_flag) {
                                late.push_back(name + ".weight: the MLP bias subvolumes are not empty and the "
                                                       "container has no MLP bias");
                                return;
                            }
                            Q4 q = q_from_tiles(tiles);
                            std::vector<uint8_t> tiled;
                            long long zinexact = 0;
                            tile_q4_1(q, tiled, &zinexact);
                            mlp_zero_inexact += zinexact;
                            mlp_zero_groups += (long long)q.scale.size();
                            const int64_t nt = int64_t(tiled.size() / kTileBytes);
                            r.tiles += nt;
                            w.add(name + ".weight", "I8", {nt, int64_t(kTileBytes)}, std::move(tiled));
                            ++r.tensors;
                            ++r.mlp_projections;
                        };
                        emit(gate, "mlp.gate_proj", gate_ok, gate_bias);
                        emit(up, "mlp.up_proj", up_ok, up_bias);
                        emit(down, "mlp.down_proj", down_ok, down_bias);
                    }
                    r.mlp_zero_inexact_pct =
                        mlp_zero_groups ? 100.0 * double(mlp_zero_inexact) / double(mlp_zero_groups) : 0.0;
                }
            }
            unmap_file(fc.map, fc.size);
        }
        if (!late.empty()) {
            for (auto& s : late) r.refused.push_back(std::move(s));
            if (!opt.allow_incomplete) return r;  // nothing has been written yet
        }
    }

    // The manifest goes in the container, so a loader (or a person) can see what this pack did and did
    // not contribute without re-reading the pack. The plan's manifest cannot know what only the import
    // discovers -- a group that does not survive decode->encode, a nonempty bias subvolume, a missing
    // tail envelope -- so the runtime refusals are merged in here: each is attributed to the tensor it
    // names, and all of them land in `incomplete` too. Without this a container written with
    // --allow-incomplete reported `incomplete: []` while silently missing those weights (CodeRabbit/Copilot
    // review of #842 and #844).
    std::string manifest = p.manifest();
    if (!r.refused.empty()) {
        json m = json::parse(manifest);
        auto& inc = m["incomplete"];
        for (const auto& s : r.refused) {
            const size_t c = s.find(": ");
            const std::string name = c == std::string::npos ? std::string() : s.substr(0, c);
            const std::string why = c == std::string::npos ? s : s.substr(c + 2);
            if (!name.empty())
                for (auto& t : m["tensors"])
                    if (t.value("name", std::string()) == name) {
                        t["refusal"] = why;
                        t["imported"] = false;
                    }
            bool seen = false;
            for (const auto& e : inc)
                if (e.is_string() && e.get<std::string>() == s) seen = true;
            if (!seen) inc.push_back(s);
        }
        manifest = m.dump(2);
    }
    w.finish(json::parse(manifest));

    {
        std::ifstream cf(pack_dir + "/config.json");
        json c = json::parse(cf);
        c["vocab_size"] = V;
        c["attention_bias"] = r.attn_bias_tensors > 0;  // what the container actually carries
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
    if (r.attn_tiles_total)
        std::printf("  attention:  %lld/%lld .fconst tiles survive decode->encode byte for byte; "
                    "%d projections + %d bias tensors imported, bf16 zero exact for %.2f%% of groups\n",
                    r.attn_tiles_ok, r.attn_tiles_total, r.attn_projections, r.attn_bias_tensors,
                    100.0 - r.attn_zero_inexact_pct);
    if (r.mlp_groups_total)
        std::printf("  MLP:        %lld/%lld .fconst groups survive decode->encode byte for byte; "
                    "%d projections imported, bf16 zero exact for %.2f%% of groups\n",
                    r.mlp_groups_ok, r.mlp_groups_total, r.mlp_projections,
                    100.0 - r.mlp_zero_inexact_pct);
    if (r.stride)
        std::printf("  fconst:     stride %lld, second header %lld (measured from the eps anchors); "
                    "%d/%d RMSNorm anchors found, final norm %s\n",
                    (long long)r.stride, (long long)r.second_header, r.norm_anchors_ok,
                    r.norm_anchors_total,
                    !r.final_norm_imported
                        ? "REFUSED (no readable tail envelope)"
                        : (r.prefill_tail_seen ? (r.final_norm_crosscheck
                                                      ? "identical in both .fconst tails"
                                                      : "taken from the Token_Token tail -- the two copies DISAGREE")
                                               : "taken from the tail envelope (the pack ships no prefill copy)"));
    if (!r.fconst_note.empty()) std::printf("  fconst:     %s\n", r.fconst_note.c_str());
    if (!r.refused.empty()) std::printf("  incomplete: %zu tensors, listed in %s/model.q4nx\n",
                                        r.refused.size(), out.c_str());
    return 0;
}

}  // namespace onebit::npu::pack_import
