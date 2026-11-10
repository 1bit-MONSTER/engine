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
#include <set>
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
    // `com.ryzenai:MatMulNBitsBf` is the same quantised matmul as `com.microsoft:MatMulNBits`, and on the 1.9
    // blob/external-data packs it is the ONLY one: phi-4's 201 weights are all `MatMulNBitsBf`, so a walker that
    // matches the Microsoft name alone reports a weightless graph for a 9.7 GB pack (engine#1503).
    int64_t ryzenai_matmuls = 0;
    // Node inputs that name a `*.qweight.empty` placeholder whose `*.qweight.packed` sibling carries the bytes
    // (resolved below). Counted so `info` can say "the weights are here, behind a placeholder" instead of "none".
    int64_t packed_siblings = 0;
    // Matmuls whose `*.empty` placeholder AND its `*.packed` sibling both carry no bytes (engine#1671). Whether that is
    // a fault depends on the pack: a 1.9 fconst pack spells every graph initializer this way and its weights are in the
    // .fconst, so plan() drops them; with no .fconst there is no other source and plan() refuses by name (#1687).
    int64_t weightless_matmuls = 0;
    std::string weightless_example;   // the first one, for the refusal's text
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
        // The graph is the authority on how THIS reader should spell a quantised matmul: the same pack also states the
        // op in `cache/*_meta.json` (which the block census reads) and in the `.onnx` alias table, and if those ever
        // disagree the graph wins here, because the graph is what the nodes — and therefore the weights — actually are.
        // A disagreement is a pack fault to report, not something to reconcile silently: `tools/npu_plan.py` walks the
        // metadata sidecar for the fconst tile census, a different question, and the two must not be averaged.
        if (op == "MatMulNBits" || op == "MatMulNBitsBf") {
            if (op == "MatMulNBitsBf") ++m.ryzenai_matmuls;
            OnnxMatMul mm;
            // THREE inputs is a real spelling, not a truncation: `amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu` names its
            // projections `(activations, ...MatMul.weight_Q4, ...MatMul.weight_scales)` with `bits 4, block_size 32` and
            // no qzeros tensor at all, and a `>= 4` test therefore recorded NO matmuls for it -- `info` printed
            // matmulnbits_ops 0 and the importer wrote only the embedding (engine#1503). A missing qzeros means the
            // implicit zero point, which `implicit_zp_zero()` below applies.
            if (ins.size() >= 3) {
                mm.data = ins[0];
                mm.weight = ins[1];
                mm.scales = ins[2];
                if (ins.size() >= 4) mm.zeros = ins[3];
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
    // A 1.9 external-data pack names its weights through placeholders: the node's input is
    // `model.layers.0.attn.o_proj.MatMulNBits.qweight.empty`, an initializer with NO dims, while the bytes sit in
    // `...qweight.packed` -- an initializer the node never references (measured on phi-4: 201 of 201 matmuls, and
    // the packed lengths match N x (K/128+1) x 68 exactly, the same 68-byte per-row-per-128-block stride the
    // .fconst attention tile carries). Following node inputs only makes a 9.7 GB pack look weightless, so the
    // reader resolves the sibling here and every consumer downstream sees a real weight (engine#1503).
    // Same tensor, two spellings: a node names `/model/layers.0/attn/...` while the initializer is
    // `model.layers.0.attn...` (measured on phi-4, 201 of 201). Resolving by exact name alone finds the node's input
    // in no pack that spells them this way.
    const auto find_init = [&m](const std::string& name) -> const OnnxTensor* {
        auto it = m.index.find(name);
        if (it != m.index.end()) return &m.tensors[it->second];
        if (!name.empty() && name[0] == '/') {
            std::string dots = name.substr(1);
            for (auto& c : dots)
                if (c == '/') c = '.';
            const auto it2 = m.index.find(dots);
            if (it2 != m.index.end()) return &m.tensors[it2->second];
        }
        return nullptr;
    };
    // A placeholder states its length as zero rather than stating nothing: the real pack's
    // `...qweight.empty` is dims [0] (a 1-D tensor of no elements), not a dimensionless scalar. Both spellings mean
    // "no bytes here", and a reader that only checks `dims.empty()` calls that a real weight of length 0.
    const auto no_bytes = [](const OnnxTensor& t) {
        if (t.dims.empty()) return true;
        for (auto d : t.dims)
            if (d != 0) return false;
        return true;
    };
    // A matmul whose placeholder and whose `*.packed` sibling BOTH carry no bytes has no weight in this graph at all,
    // and it is DROPPED rather than left pointing at the placeholder. Leaving it was engine#1671: a 1.9 fconst pack
    // declares 592 `[0]`-dim initializers (its 363 `.empty` AND their 363 `.packed` siblings are placeholders; the
    // weights are in the Token .fconst), so the node kept naming `...qweight.empty`, the reader called bytes_of() on it
    // and threw `model.layers.0.attn.qkv_proj.MatMulNBits.qweight.empty: no data` -- which broke `npu-pack import` for
    // every 1.9 pack. Dropping them is also what tells the rest of the importer the truth about this pack: with no
    // readable graph head, `matmuls` is empty and the head is read from the .fconst tail, which is where it is.
    std::vector<OnnxMatMul> kept;
    kept.reserve(m.matmuls.size());
    for (auto& mm : m.matmuls) {
        if (mm.weight.empty()) { kept.push_back(mm); continue; }
        const OnnxTensor* w = find_init(mm.weight);
        if (w && !no_bytes(*w)) {
            mm.weight = w->name;   // the initializer's own spelling, so every later lookup is exact
            kept.push_back(mm);
            continue;
        }
        if (!w) { kept.push_back(mm); continue; }   // names nothing this graph declares: not this genre's business
        std::string packed = w->name;
        if (packed.size() >= 6 && packed.compare(packed.size() - 6, 6, ".empty") == 0)
            packed.replace(packed.size() - 6, 6, ".packed");
        else { kept.push_back(mm); continue; }
        const OnnxTensor* pw = find_init(packed);
        // No sibling at all: not this genre's pack. Keep the node exactly as it was -- it still names its `*.empty`
        // initializer and the reader below decides what that means. Dropping it here would be a decision made by an
        // absence: `onnx.matmuls.empty()` is what tells plan() "this is a 1.9 fconst pack, take the head from the tail"
        // (:1685), so a truncated pack with real graph matmuls and no `.packed` siblings would silently get a head
        // from the tail instead of the refusal #1687 asks for (engine#1690 review).
        if (!pw) { kept.push_back(mm); continue; }
        if (no_bytes(*pw)) {
            // The sibling exists and is ITSELF a zero-length placeholder. Whether that is a fault depends on something
            // this walker cannot see: on a Ryzen AI 1.9 fconst pack EVERY graph initializer is spelled this way (363
            // `.empty` and 363 `.packed`, all dims [0]) because the weights are in the Token .fconst, so dropping the
            // node is right and the import succeeds from the fconst. On a pack with no .fconst there is nothing to fall
            // back to, and refusing by name (#1687's rule) is the honest answer. So the walker records it and plan(),
            // which knows whether an .fconst is there, decides. Deferring it silently produced
            // `...qweight.empty: no data` from a lookup much later, naming the placeholder rather than the problem.
            ++m.weightless_matmuls;
            m.weightless_example = w->name + " (its " + packed + " sibling carries no data either)";
            continue;
        }
        mm.weight = pw->name;
        ++m.packed_siblings;
        kept.push_back(mm);
    }
    m.matmuls = std::move(kept);
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

// The container this importer writes: fp32 q4_1f parameters (npu/q4nx.h), named by the container's
// layout "q4nx-2". The bf16 q4_1 form stays the GGUF-derived containers' -- that is where the bf16
// choice came from -- and an absent layout means it, so every container already on disk loads.
constexpr q4nx::Params kContainerParams = q4nx::Params::F32;
constexpr int64_t kImportTileBytes = int64_t(q4nx::Chunk::Q4_1F);

struct Fconst {
    std::string path;
    const uint8_t* map = nullptr;
    size_t size = 0;
};

Fconst open_fconst(const std::string& path) {
    if (path.empty())
        throw std::runtime_error("no dd_metastate_Llm_Token_*.fconst in the pack (1.8 spelled it Llm_Token_Token_, "
                                 "Ryzen AI 1.9 tags it with the aie2 program name)");
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

// ── Ryzen AI 1.9 Token .fconst: a measured shape, not a derivable one ────────
// A 1.9 pack of the same model differs from its 1.8 twin in exactly two ways: a 4*NH-byte header is
// inserted after the q/k/v bundle (the four bytes 00 00 7f ff, once per head), and that bundle is stored
// q,k,v where 1.8 stores k,q,v. Everything else in the block is verbatim, and the stream after the header
// reads at -4*NH. The tile shape cannot be derived from the pack itself -- it is not a subset sum of the
// pack's own const sizes -- so each rule below was measured against the 1.8 pack of the SAME model by
// whole-file diff. A 1.9 pack whose geometry is not in this table is refused rather than decoded on a guess.
// The correlation above which an imported head is taken to BE the embedding (tied). Named because the container's
// field, the readout and the note all have to agree on it: three copies of 0.99 is how they drifted apart before.
constexpr double kTieCorr = 0.99;

// Every entry was measured by diffing the 1.9 pack against the 1.8 pack of the SAME model: the bundle's tile shape and
// its rotation are not derivable from the pack, and neither is `prefix`. It was tempting to read the extra bytes as
// Qwen3's per-head QK norm gains, but Llama-3.2-1B -- which has no QK norms at all -- carries the same +4,352 at hidden
// 2,048, so the extra is a property of the geometry, not of the architecture. Measured anyway, because a plausible
// explanation that survives one model is how this table would get a wrong row.
constexpr Fconst19Rule kFconst19Rules[] = {
    // Qwen2.5-0.5B-Instruct: normalises to its 1.8 twin byte for byte, whole file, all 24 blocks.
    {896, 14, 24, 1856, 16, 2, 34816},
    // Qwen3-1.7B: same rule at hidden 2048 / 16 heads / 28 layers; 535 code bytes of the two builds differ, all in
    // layer 24, which is content rather than layout (every block still lines up).
    {2048, 16, 28, 8512, 3, 1, 1183744},
    // SmolLM2-135M-Instruct: hidden 576 / 9 heads / 30 layers, headers of 36 B (4 x 9) at a 2,360,292 B stride.
    // Measured by diffing this pack against its 1.8 twin: prefix 3,776 B, then FOUR tiles of 76,800 rotated by 1, then
    // verbatim to the header. 29 of the 30 blocks normalise byte-identically; the 112 differing bytes are all in layer 3
    // (content: this pair is a fine-tune on both sides). Note what this row does NOT buy: the pack's weights are
    // quantised at block 32 (151 ops), which is a tile rhythm this importer cannot walk -- so the import reaches the
    // block gate and refuses there by number (engine#1503). The row is what makes that refusal the FIRST thing wrong
    // with the pack rather than the 1.9 header layout.
    {576, 9, 30, 3776, 4, 1, 76800},
    // Qwen2.5-Coder-1.5B-Instruct: hidden 1,536 / 12 heads / 28 layers, 748 of 733,564,160 layer-block bytes differ
    // (1.0e-6). Its TAIL is 262,147,136 B against the 1.8 pack's 134,314,560 -- this pack quantises its head at block 32
    // while its 140 layer ops are block 128 (engine#1503), so the tail is where the difference lives and the layers are
    // what this row describes.
    // Qwen-2.5_1.5B_Instruct measured to the SAME bundle and rotation as the Coder pack -- byte for byte over all 28
    // layer blocks -- so the two packs share one row now that the key is geometry, not the pack name.
    {1536, 12, 28, 3136, 7, 1, 226304},
    // Qwen3-4B: 2,933 of 2,208,061,248 bytes differ (1.3e-6), again content not layout. The prefix corroborates the
    // Qwen3 offset seen at 1.7B -- 64 + 2*hidden + 4,352, so 8,512 at hidden 2,048 and 9,536 at 2,560 -- and the tile
    // count differs from 1.7B's 3, which is why this is measured per pack and never derived.
    {2560, 32, 36, 9536, 5, 1, 1462272},
    // Llama-3.2-1B-Instruct: hidden 2,048 with 32/16 heads and no QK norms -- the first non-Qwen family here. Same
    // prefix as Qwen3-1.7B at this hidden size (8,512), a different bundle (5 tiles of 591,872, not 3 of 1,183,744),
    // and 1,857 of 686,518,336 bytes differ (2.7e-6). This row is the one that disproved the QK-norm explanation above.
    {2048, 32, 16, 8512, 5, 1, 591872},
    // Phi-3.5-mini-instruct at hidden 3,072 / 32 heads / 32 layers (and Phi-3-mini-128k, the same phi3 geometry; the
    // table is keyed by shape, and the Llama row above is what established that the prefix is a property of the shape).
    // Measured against Phi-3.5-mini's own 1.8 build (`_rai_1.8.0_npu_4K`) by whole-file normalisation: 28 of the 32
    // blocks come out byte-identical and the four that differ hold all 26,145 differing bytes of 2,036,067,648
    // (1.28e-5), i.e. content in four layers, not layout. The separation that makes this a rule rather than a fit is
    // the wrong-rotation end: any rotation other than this one differs in 307,097,372 bytes (15 %) across all 32
    // blocks, a factor of 10,000. The two halves of the bundle are swapped relative to 1.8 -- the same q/k/v order
    // change the transform above notes -- so n=2/rot=1 is the minimal spelling of it (n=4/rot=2 and n=8/rot=4 are the
    // identical permutation, which is why the measurement tool prints the same residual for each).
    {3072, 32, 32, 10560, 2, 1, 5222400},
    // hidden 3,584 / 28 heads / 28 layers -- the Qwen2.5-7B family. Measured TWICE, against two different models' 1.8 twins, and
    // both pairs give this row: Qwen2.5-7B-Instruct (2,219 of 3,830,638,912 bytes differ, 5.79e-05 %) and Qwen2.5-Coder-7B-Instruct
    // (28,271 differ, 7.38e-04 %), each against its own `_rai_1.8.0_npu_16K` build. Two independent models of one geometry agreeing
    // on `rot` is the corroboration a single pair cannot give, and `--probe-wrong` separates the rotation by four orders of
    // magnitude on the Coder pair: rot+1 differs in 208,982,210 bytes (5.46 %), n+1 in 52,522,601 (1.37 %), against this row's
    // 28,271.
    // The prefix is the ONE field this pair does not separate: `prefix+64` differs in 33,291 bytes, which is under the tool's
    // 38,306-byte tolerance -- so `prefix = 7,232` here rests on the two models agreeing and on the plain Qwen2.5 arithmetic
    // (64 + 2*hidden, with no QK-norm term, unlike the Qwen3 line's +4,352), not on the pair alone. Said plainly because a row
    // whose comment implies all seven fields were equally pinned would be the same kind of overstatement this table exists to
    // avoid.
    // The block's arithmetic, from the measurement rather than from a formula: prefix 7,232 + 9 x tile 1,009,664 = 9,094,208,
    // which is exactly the `header_off` the tool reports for this pair -- so the payload after the prefix is NINE tile slots,
    // of which the rule rotates eight and the ninth is the gap the post-attention header sits behind. (An earlier version of
    // this comment subtracted the prefix from a per-block figure and produced a number three orders of magnitude off; the
    // review of #1758 caught it. Written out here so the next reader can check it with a calculator.)
    {3584, 28, 28, 7232, 8, 1, 1009664},
    // Qwen2.5_3B_Instruct at hidden 2,048 / 16 heads / 36 layers: measured with tools/fconst19_measure.py against
    // amd/Qwen2.5_3B_Instruct_rai_1.8.0_npu_4K, 1,917 of 1,706,629,440 bytes differ (1.12e-06 of the file, which the tool
    // prints as 1.12e-04 %). Same hidden size as the Qwen3-1.7B row and NOT the same row: the prefix is 4,160 against
    // 1.7B's 8,512, and the bundle is nine tiles of 295,936 (2,663,424 B) against three of 1,183,744 -- so at this hidden
    // size both the prefix and the division move with the layer count. Measured, never derived: 64 + 2*hidden + 4,352
    // would predict 8,512 here and be wrong by 4,352.
    {2048, 16, 36, 4160, 9, 1, 295936},
    // Mistral-7B-Instruct-v0.2 at hidden 4,096 / 32 heads / 32 layers: measured with tools/fconst19_measure.py against
    // amd/Mistral-7B-Instruct-v0.2_rai_1.8.0_npu_4K, 1,954 of 3,843,419,200 bytes differ (5.08e-05) -- content, not
    // layout. The prefix follows the Qwen3 line (64 + 2*hidden + 4,352 = 12,608), the bundle is 5 tiles of 2,297,856
    // (11,489,280 B, which is NOT header_off minus the prefix: the header sits after the prefix and the bundle follows
    // it, so this row is one the arithmetic would have got wrong), and the header stride 117,862,720 is uniform over
    // all 32 blocks. This is the largest family on #1503 (31 packs). Measured by another lane on main, whose
    // table is keyed on geometry; merged here in the same form.
    {4096, 32, 32, 12608, 5, 1, 2297856},
    // Llama-3.2-3B (and -Instruct, same geometry) at hidden 3,072 / 24 heads / 28 layers: measured with
    // tools/fconst19_measure.py against amd/Llama-3.2-3B_rai_1.8.0_npu_4K, 1,090 of 1,756,930,368 bytes differ
    // (6.20e-07 of the file, which the tool prints as 6.20e-05 %) -- content, not layout. The prefix is the Qwen3 line's 64 + 2*hidden + 4,352 = 10,560 at this hidden
    // size, but the bundle is FOUR tiles of 1,740,800 (6,963,200 B) where the Phi/h3072 row has two of 5,222,400 --
    // which is the whole reason a row is measured per geometry and never derived from the hidden size.
    {3072, 24, 28, 10560, 4, 1, 1740800},
};

const Fconst19Rule* select_rule_in(const Fconst19Rule* rules, int count, int hidden, int heads, int layers,
                                   std::string* why) {
    const Fconst19Rule* found = nullptr;
    std::string ambiguity;
    for (int i = 0; i < count; ++i) {
        const Fconst19Rule& r = rules[i];
        if (r.hidden != hidden || r.heads != heads || r.layers != layers) continue;
        if (!found) { found = &r; continue; }
        if (r.prefix != found->prefix || r.n_tiles != found->n_tiles || r.rot != found->rot || r.tile != found->tile)
            ambiguity += (ambiguity.empty() ? "" : " and ") + std::string("prefix ") + std::to_string(found->prefix) +
                         "/" + std::to_string(r.prefix) + " B, " + std::to_string(found->n_tiles) + "/" +
                         std::to_string(r.n_tiles) + " tiles of " + std::to_string(found->tile) + "/" +
                         std::to_string(r.tile) + " B, rot " + std::to_string(found->rot) + "/" + std::to_string(r.rot);
    }
    if (!ambiguity.empty()) {
        if (why)
            *why = "hidden " + std::to_string(hidden) + ", " + std::to_string(heads) + " heads, " +
                   std::to_string(layers) + " layers has been measured with two DIFFERENT block shapes (" + ambiguity +
                   "). Which one a pack needs cannot be decided from its geometry, and choosing either would rotate its"
                   " weights into the wrong projections and answer quietly wrong text, so this refuses: measure this"
                   " pack against its own 1.8 twin with tools/fconst19_measure.py and keep one row per geometry";
        return nullptr;
    }
    if (!found && why)
        *why = "its block shape has not been measured for hidden " + std::to_string(hidden) + ", " +
               std::to_string(heads) + " heads, " + std::to_string(layers) +
               " layers. The tile shape cannot be derived from the pack, so decoding it would be a guess: the 1.8 pack"
               " of the same model is what establishes it.";
    return found;
}

// The header a 1.9 block carries: 00 00 7f ff once per head. Its length being the head count is what makes
// it a test rather than a magic number. 0x00007fff is the big-endian reading of those four bytes; the bytes
// are what is compared, because the little-endian reading of the same bytes is 0xff7f0000.
std::vector<uint8_t> fconst19_header(int heads) {
    std::vector<uint8_t> h(size_t(heads) * 4, 0);
    for (int i = 0; i < heads; ++i) {
        h[size_t(i) * 4 + 2] = 0x7f;
        h[size_t(i) * 4 + 3] = 0xff;
    }
    return h;
}

// True when `fc` is a 1.9 Token .fconst of this geometry: exactly `layers` headers, evenly spaced.
// Fills the first header's offset and the block stride. False means the 1.8 form (or a shape this
// importer does not know, which the caller decides about).
bool fconst19_detect(const Fconst& fc, int heads, int layers, int64_t* hdr_off, int64_t* stride) {
    if (heads <= 0 || layers < 2) return false;
    const std::vector<uint8_t> pat = fconst19_header(heads);
    std::vector<int64_t> runs;
    for (size_t i = 0; i + pat.size() <= fc.size;) {
        if (fc.map[i] == pat[0] && fc.map[i + 2] == 0x7f && fc.map[i + 3] == 0xff &&
            std::memcmp(fc.map + i, pat.data(), pat.size()) == 0) {
            runs.push_back(int64_t(i));
            i += pat.size();
        } else {
            ++i;
        }
    }
    if (int(runs.size()) != layers) return false;
    const int64_t s = runs[1] - runs[0];
    if (s <= int64_t(pat.size())) return false;
    for (size_t k = 0; k + 1 < runs.size(); ++k)
        if (runs[k + 1] - runs[k] != s) return false;
    *hdr_off = runs[0];
    *stride = s;
    return true;
}

// The 1.8-equivalent form of a Token .fconst. A pack already in the 1.8 form comes back unchanged (the same
// path, no copy); a 1.9 pack comes back as a file of canonical bytes, so every offset below the two forms is
// one set of numbers and the whole importer above needs no second arithmetic. `work` is where the canonical
// copy is written -- the caller removes it.
//
// The q/k/v bundle is normalised here, which is what makes the tiles line up: 1.9 stores q first and k last.
std::string canonical_token_fconst(const std::string& path, int hidden, int heads, int layers, const std::string& work) {
    if (path.empty()) return path;
    Fconst fc = open_fconst(path);
    int64_t hdr_off = 0, stride9 = 0;
    if (!fconst19_detect(fc, heads, layers, &hdr_off, &stride9)) {
        unmap_file(fc.map, fc.size);
        return path;
    }
    std::string rule_why;
    const Fconst19Rule* rule = select_fconst19_rule(hidden, heads, layers, &rule_why);
    if (!rule) {
        unmap_file(fc.map, fc.size);
        throw std::runtime_error(path + ": this is a Ryzen AI 1.9 Token .fconst (it carries " +
                                 std::to_string(layers) + " headers of " + std::to_string(heads * 4) +
                                 " B, one per layer, 00 00 7f ff per head), and " + rule_why);
    }
    const int64_t hdr_len = int64_t(heads) * 4;
    const int64_t stride8 = stride9 - hdr_len;
    const int64_t prefix = rule->prefix;
    const int64_t bundle = int64_t(rule->n_tiles) * rule->tile;
    if (rule->n_tiles < 2 || rule->tile <= 0 || prefix + bundle > hdr_off ||
        hdr_off + hdr_len > stride9 || stride8 <= 0) {
        unmap_file(fc.map, fc.size);
        throw std::runtime_error(path + ": the measured 1.9 block shape does not fit this file");
    }
    const int64_t tail_off = int64_t(layers) * stride9;
    if (tail_off > int64_t(fc.size)) {
        unmap_file(fc.map, fc.size);
        throw std::runtime_error(path + ": fewer bytes than " + std::to_string(layers) + " blocks of " +
                                 std::to_string(stride9) + " B");
    }
    // The canonical body is a function of this file alone, so its length is known before it is built: reuse the copy
    // when it already has that length, instead of writing ~291 MB (1.5B: ~1 GB) again on every call (engine#1662). The
    // name already carries the source's size and mtime, so a reached-here path cannot be a stale copy of another pack --
    // and a TRUNCATED copy from an interrupted run is caught by the length test rather than trusted.
    const int64_t expected = int64_t(fc.size) - int64_t(layers) * hdr_len;
    if (canonical_copy_reusable(work, expected)) {
        // engine#1797 review: the eviction age is `last_write_time`, and a reuse used to READ the file without writing
        // it -- so a copy in continuous use aged past `max_age_seconds` and was swept, and the next run rewrote a
        // multi-hundred-MB file. Touching on reuse makes the timestamp mean what the rule assumes: last USE. Errors
        // are ignored: failing to bump an mtime must not fail an import that already has a good copy.
        std::error_code tec;
        fs::last_write_time(fs::path(work), fs::file_time_type::clock::now(), tec);
        unmap_file(fc.map, fc.size);
        return work;
    }
    std::vector<uint8_t> out;
    out.reserve(size_t(fc.size - int64_t(layers) * hdr_len));
    for (int L = 0; L < layers; ++L) {
        const uint8_t* b = at(fc, int64_t(L) * stride9, stride9);
        out.insert(out.end(), b, b + prefix);
        for (int i = 0; i < rule->n_tiles; ++i) {   // canonical i is stored at (i - rot) mod n
            const int64_t j = ((int64_t(i) - rule->rot) % rule->n_tiles + rule->n_tiles) % rule->n_tiles;
            out.insert(out.end(), b + prefix + j * rule->tile, b + prefix + (j + 1) * rule->tile);
        }
        out.insert(out.end(), b + prefix + bundle, b + hdr_off);
        out.insert(out.end(), b + hdr_off + hdr_len, b + stride9);
    }
    out.insert(out.end(), fc.map + tail_off, fc.map + fc.size);
    unmap_file(fc.map, fc.size);
    {
        std::ofstream os(work, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error(work + ": cannot write the canonical copy of the 1.9 .fconst");
        os.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        if (!os) throw std::runtime_error(work + ": short write of the canonical copy of the 1.9 .fconst");
    }
    return work;
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

// The pack's **Token program** .fconst — the file that carries the layer blocks and the head tiles.
//
// Ryzen AI 1.8 names it `dd_metastate_Llm_Token_Token_<pipeline>.fconst`. Ryzen AI 1.9 names the same artifact
// after the aie2 compiler's own per-node tag: `dd_metastate_Llm_Token_aie2_token_modellayers.0input_layernorm
// MLADFRMSNORM.fconst`. A fixed `Llm_Token_Token_` prefix therefore matches **no** 1.9 pack, which is exactly how
// `amd/Qwen3-8B_rai_1.9.0_npu` failed: `npu-pack info` read the dims and the tokenizer, then reported all 360 layer
// tensors as "no dd_metastate_Llm_Token_Token_*.fconst in the pack".
//
// The stem that is stable across both releases is **Llm_Token_**. The prefill program is a different stem
// (`Llm_Prefill_`, found separately), so it cannot be picked up here.
//
// Resolution order, and it never guesses: the 1.8 spelling first, so nothing that imports today changes; then a
// *single* `Llm_Token_*` file. Two or more unmatched candidates is a pack shape we have not seen, and choosing the
// alphabetically first would be a coin flip over the model's weights, so it is refused by name with the list.
// defined with the hybrid line's graph resolution below; the fconst lookup uses it to find a component pack's decoder.
std::string graph_filename(const std::string& pack_dir);

std::string find_token_fconst(const std::string& dir, std::string* why = nullptr) {
    const std::string exact = find_one(dir, "dd_metastate_Llm_Token_Token_", ".fconst");
    if (!exact.empty()) return exact;
    // A multi-component pack (`type: compositemodel`) keeps its LLM graph AND its Token .fconst in the component
    // `genai_config.json` names (`decoder/`), while the config files that state the geometry stay at the pack root.
    // #1644 taught `graph_filename()` to follow that relative path; the fconst lookup was still scanning only the root,
    // so pointing npu-pack at the pack root reported "the pack carries no Token .fconst either" for a pack that has one
    // two directories down. Same pack, same rule: look where the pack says its decoder is.
    {
        const std::string graph = graph_filename(dir);
        const size_t slash = graph.rfind('/');
        if (slash != std::string::npos) {
            const std::string component = dir + "/" + graph.substr(0, slash);
            std::error_code cec;
            if (fs::is_directory(component, cec)) {
                const std::string in_component = find_one(component, "dd_metastate_Llm_Token_Token_", ".fconst");
                if (!in_component.empty()) return in_component;
                const std::string pre2 = "dd_metastate_Llm_Token_", suf2 = ".fconst";
                std::vector<std::string> c2;
                for (const auto& e : fs::directory_iterator(component)) {
                    const std::string n = e.path().filename().string();
                    if (n.size() <= pre2.size() + suf2.size()) continue;
                    if (n.compare(0, pre2.size(), pre2) != 0) continue;
                    if (n.compare(n.size() - suf2.size(), suf2.size(), suf2) != 0) continue;
                    c2.push_back(e.path().string());
                }
                if (c2.size() == 1) return c2.front();
            }
        }
    }
    const std::string pre = "dd_metastate_Llm_Token_", suf = ".fconst";
    std::vector<std::string> cands;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() <= pre.size() + suf.size()) continue;
        if (n.compare(0, pre.size(), pre) != 0) continue;
        if (n.compare(n.size() - suf.size(), suf.size(), suf) != 0) continue;
        cands.push_back(e.path().string());
    }
    std::sort(cands.begin(), cands.end());
    if (cands.size() == 1) return cands[0];
    if (why) {
        if (cands.empty()) {
            *why = "no dd_metastate_Llm_Token_*.fconst in the pack (neither the 1.8 spelling "
                   "Llm_Token_Token_ nor a single Llm_Token_<program> file)";
        } else {
            *why = "the pack carries " + std::to_string(cands.size()) +
                   " dd_metastate_Llm_Token_*.fconst files and none is named Llm_Token_Token_, so which one holds "
                   "the layer weights cannot be decided here:";
            for (const auto& c : cands) *why += " " + fs::path(c).filename().string();
        }
    }
    return {};
}

// The Token .fconst as the rest of this file reads it. A Ryzen AI 1.9 pack comes back normalised to its 1.8 form (see
// canonical_token_fconst); a 1.8 pack comes back as itself, at no cost.
//
// THIS WRITES. For a 1.9 pack it materialises the canonical copy -- 291 MB on Qwen2.5-0.5B, about 1 GB on a 1.5B -- into the
// temporary directory, and it does NOT cache, so every call rewrites it. It is named `materialise_` for that reason: it was
// `resolve_token_fconst` until a review of engine#1655 caught a call site using it as a cheap "does this pack have an
// .fconst?" predicate, which cost a second full rewrite per `npu-pack` command (measured: two O_TRUNC opens of the same
// 291 MB file for one `npu-pack info`). Call it ONCE per plan/import and pass the path around. "Is there an .fconst?" is
// find_token_fconst(), which touches nothing.
std::string materialise_token_fconst(const std::string& pack_dir, const Dims& dims) {
    const std::string found = find_token_fconst(pack_dir);
    if (found.empty()) return found;
    // Named by pack and geometry, so a rerun overwrites its own copy instead of filling the temp dir.
    const std::string tag =
        std::to_string(std::hash<std::string>{}(fs::path(pack_dir).lexically_normal().string()) % 100000000);
    // The name carries the SOURCE file's identity as well as the pack path and geometry (engine#1662): the copy is
    // reused when it already exists at the right size, and a repacked pack at the same path with the same geometry would
    // otherwise be served a stale copy of the same length. Size alone is not identity; the mtime completes it cheaply.
    std::error_code src_ec;
    const std::uintmax_t src_size = fs::file_size(found, src_ec);
    const auto src_stamp = [&] {
        std::error_code e;
        const auto t = fs::last_write_time(found, e);
        return e ? int64_t(0) : int64_t(t.time_since_epoch().count());
    }();
    // The scratch copy is only reusable while the RULES that produced it are the ones in this binary: a change to the
    // canonical form (a new kFconst19Rules entry, a different tile order) leaves the source file untouched, so source
    // identity alone would keep serving a copy of the old form. Bump this when the canonical form changes.
    constexpr int kCanonicalFormVersion = 1;
    // Every generation of the scratch cache starts with this (engine#1770). Named because the sweep below has to match the
    // generations that have no version suffix.
    constexpr const char* kScratchStem = "1bit-fconst19";
    // A scratch copy older than this is a pack no run is importing: no import holds one for hours, so an mtime this far back
    // means the pack was abandoned, not that somebody is using it. Kept well above the longest import measured (~minutes).
    constexpr int64_t kScratchMaxAgeSeconds = 6 * 3600;
    // Everything but the source identity: the group this run evicts its stale siblings from (engine#1770).
    const std::string ident_prefix = "1bit-fconst19v" + std::to_string(kCanonicalFormVersion) + "-" + tag + "-" +
                                     std::to_string(dims.hidden) + "-" + std::to_string(dims.heads) + "-" +
                                     std::to_string(dims.layers) + "-";
    const std::string name = ident_prefix + std::to_string(src_ec ? std::uintmax_t(0) : src_size) + "-" +
                             std::to_string(static_cast<std::uint64_t>(src_stamp)) + ".fconst";
    // `temp_directory_path()` THROWS when TMPDIR names a directory that no longer exists, which is the ordinary state of
    // a shell after a reboot has cleared /tmp -- and it threw before any work was attempted, so `npu-pack info` on a
    // 1.9 pack failed with `filesystem error: temp_directory_path: No such file or directory [/tmp/x]` and no hint that
    // the pack was fine. Fall back rather than fail: /tmp, then the current directory. The copy is scratch, and a note
    // says where it went so nobody wonders what that 291 MB file is.
    // Where the canonical copy lives. This is a CACHE (reused on size+mtime+rules version), not a leak, but its size is the
    // pack's fconst -- 3.6 GB for an h4096 pack -- and on a box whose /tmp is a RAM disk that is RAM, per pack, held until
    // reboot: three of them from one afternoon's imports were 11.4 GB of tmpfs (measured 2026-11-10). So the order is:
    // an explicit override, then the on-disk cache directory, and only then the temp directory this used to prefer.
    std::vector<fs::path> dirs;
    for (const auto& d : fconst_scratch_dirs()) dirs.emplace_back(d);
    dirs.push_back(fs::current_path());
    std::string work;
    for (const auto& d : dirs) {
        if (!fconst_scratch_ready(d.string())) continue;
        const fs::path cand = d / name;
        // Writable in fact, not just in theory: the copy is written immediately after this.
        std::ofstream probe(cand, std::ios::binary | std::ios::app);
        if (!probe) continue;
        probe.close();
        work = cand.string();
        // This is the copy that will be written, so the copies of the same pack and geometry at other sizes/mtimes are
        // the ones nothing will reuse (engine#1770).
        // The pre-version generation of the same pack and geometry: `1bit-fconst19-<tag>-<h>-<heads>-<layers>.fconst`,
        // with no source identity in the name, so the prefix above cannot match it (engine#1770: three such files, 10.67
        // GiB of tmpfs, had been resident all day).
        const std::string legacy_name = std::string(kScratchStem) + "-" + tag + "-" + std::to_string(dims.hidden) + "-" +
                                        std::to_string(dims.heads) + "-" + std::to_string(dims.layers) + ".fconst";
        evict_stale_canonical_copies(d.string(), name, ident_prefix, legacy_name, kScratchMaxAgeSeconds);
        // Say where it went whenever that is not the ordinary place: the copy is a few hundred MB and it is scratch.
        std::error_code ec2;
        const fs::path tmpdir = [&] { try { return fs::temp_directory_path(); } catch (...) { return fs::path(); } }();
        if (tmpdir.empty() || fs::weakly_canonical(cand.parent_path(), ec2) != fs::weakly_canonical(tmpdir, ec2))
            std::fprintf(stderr, "1bit npu-pack: note: the 1.9 pack's canonical copy is %s (remove it when done)\n",
                         work.c_str());
        break;
    }
    if (work.empty())
        throw std::runtime_error(pack_dir + ": this is a Ryzen AI 1.9 pack, whose Token .fconst has to be rewritten into "
                                            "its 1.8 form before it can be read, and no writable directory is available "
                                            "for that copy (tried the temporary directory, /tmp and the current "
                                            "directory)");
    return canonical_token_fconst(found, dims.hidden, dims.heads, dims.layers, work);
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
    // MatMulNBits' block size along K, as the compiler declares it per op. Every rai_1.8 pack says 128 and so does
    // Qwen3-8B rai_1.9, but **SmolLM2-135M rai_1.9 declares 32 on all 151 ops** (measured from cache/*_meta.json,
    // 2026-11-08) -- and the tile geometry here is written for 128: a group is 2*(K/block)+2 subvolumes, so at block
    // 32 every offset moves. Reading the declaration is what turns that from a wrong parse into a refusal.
    int64_t block = 0;
    int64_t bits = 0;
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
                dec.block = val("block_size");
                dec.bits = val("bits");
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


}  // namespace: read_embedding_component is declared in pack_import.h so a test can drive it

// ── the embedding component ───────────────────────────────────────────────────
// A hybrid pack keeps its embedding table in its OWN graph: `embedding.onnx` (+ `embedding.onnx.data`) holds
// `language_model.embed_tokens.weight_Q4` [V, H/2] UINT8 and `..._weight_scales` [V, groups] FLOAT, gathered by
// `GatherBlockQuantized`. The main graph then has no `model.embed_tokens.weight` at all, which is what the importer
// used to refuse on (measured on amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu: 151936 x 1280 UINT8 + 151936 x 20 FLOAT,
// i.e. 4-bit, group 128, hidden 2560; engine #1503).
//
// Returns [V, H] f32 (row-major), empty when the pack has no such component. Everything it needs already existed:
// walk_graph() reads the graph with its external-data declarations, decode_matmulnbits() is block-parameterised, and
// the scale-per-32-column-group layout is the container's own.
std::vector<float> read_embedding_component(const std::string& pack_dir, std::string* why) {
    // A multi-component pack (`type: compositemodel`) names its embedding component in genai_config.json
    // (`model.embedding.filename = "embedding/model.onnx"`), exactly as it names its decoder; a single-graph pack keeps
    // the flat `embedding.onnx` this reader was written for. Follow the pack's own declaration first — the same rule the
    // graph and the Token .fconst now follow (#1644, #1657) — and fall back to the flat name.
    std::string gpath = pack_dir + "/embedding.onnx";
    {
        std::ifstream gf(pack_dir + "/genai_config.json");
        if (gf) {
            json g;
            try { g = json::parse(gf); } catch (...) { g = json::object(); }
            if (g.is_object() && g.contains("model") && g["model"].is_object() && g["model"].contains("embedding") &&
                g["model"]["embedding"].is_object()) {
                const json& emb = g["model"]["embedding"];
                if (emb.contains("filename") && emb["filename"].is_string()) {
                    const std::string rel = emb["filename"].get<std::string>();
                    if (!rel.empty() && rel.front() != '/' && rel.find("..") == std::string::npos) {
                        std::error_code nec;
                        if (fs::is_regular_file(pack_dir + "/" + rel, nec)) gpath = pack_dir + "/" + rel;
                    }
                }
            }
        }
    }
    std::error_code ec;
    if (!fs::is_regular_file(gpath, ec)) { if (why) *why = "no embedding.onnx"; return {}; }
    // parse_onnx() unwraps the ModelProto's graph field and walks it; walk_graph() alone expects the GRAPH protobuf,
    // which is why handing it the whole file yields zero tensors.
    OnnxModel em;
    try {
        em = parse_onnx(gpath);
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return {};
    }
    const OnnxTensor* table = nullptr;
    const OnnxTensor* scales = nullptr;
    for (const auto& t : em.tensors) {
        const bool emb = t.name.find("embed_tokens") != std::string::npos;
        if (!emb) continue;
        if (t.name.find("scale") != std::string::npos) scales = &t;
        else if (t.name.size() > 3 && t.name.compare(t.name.size() - 3, 3, "_Q4") == 0) table = &t;
        else if (!table && t.dims.size() == 2) table = &t;   // an unquantised [V, H] table
    }
    // engine#1725: the gemma-4 components are f32 and UNQUANTISED -- an `embed_tokens` table with no `*scales*` beside
    // it -- while the shape above is the 4-bit packed one another family ships. Reading the f32 table is worth doing
    // (the container wants BF16 and the conversion is exact); claiming the pack is not, because its decoder also carries
    // `per_layer_inputs`, fed by `embed_tokens_per_layer.weight` (262144x8960 on the e2b pack, i.e. layers x head_dim)
    // and a 256-wide projection norm, and no reader here implements that mechanism yet. So: read the token table, refuse
    // by NAME when the per-layer table is present, instead of reporting the shape mismatch as a missing table.
    if (table && !scales) {
        const bool per_layer = [&] {
            for (const auto& t2 : em.tensors)
                if (t2.name.find("embed_tokens_per_layer") != std::string::npos) return true;
            return false;
        }();
        if (per_layer) {
            if (why)
                *why = "the component carries embedding.embed_tokens_per_layer.weight, i.e. a PER-LAYER embedding table "
                       "(262144x8960 on gemma-4-e2b = layers x head_dim) that feeds the decoder's per_layer_inputs, and "
                       "the per-layer embedding mechanism is not implemented here yet (engine#1725): refusing rather "
                       "than importing a decoder without that input";
            return {};
        }
        if (table->type != kF32) {
            if (why) *why = "embed_tokens is unquantised with dtype " + std::to_string(table->type) +
                            ", and only f32 is read here (engine#1725)";
            return {};
        }
        if (table->dims.size() != 2 || table->dims.at(0) <= 0 || table->dims.at(1) <= 0) {
            if (why) *why = "an f32 embed_tokens table must be 2-D and positive";
            return {};
        }
        const int64_t Vf = table->dims.at(0), Hf = table->dims.at(1);
        std::map<std::string, const uint8_t*> fext;
        const uint8_t* raw = nullptr;
        if (!table->location.empty()) {
            size_t sz = 0;
            std::string w3;
            const uint8_t* m = map_file(fs::path(gpath).parent_path().string() + "/" + table->location, sz, &w3);
            if (!m) { if (why) *why = w3; return {}; }
            fext[table->location] = m;
            const uint64_t need = uint64_t(Vf) * uint64_t(Hf) * 4ull;
            if (table->offset + need > sz) {
                if (why) *why = "embed_tokens (f32) declares more bytes than its external data carries";
                return {};
            }
            raw = m + table->offset;
        } else if (table->raw != nullptr) {
            raw = table->raw;
        }
        if (raw == nullptr) { if (why) *why = "an f32 embed_tokens table with no data"; return {}; }
        std::vector<float> out(size_t(Vf) * size_t(Hf));
        std::memcpy(out.data(), raw, out.size() * sizeof(float));
        return out;
    }
    if (!table || !scales) { if (why) *why = "embedding.onnx has no embed_tokens table + scales"; return {}; }
    if (table->dims.size() != 2 || scales->dims.size() != 2) { if (why) *why = "embed_tokens dims are not 2-D"; return {}; }
    const int64_t V = table->dims.at(0), packed = table->dims.at(1), groups = scales->dims.at(1);
    if (V <= 0 || packed <= 0 || groups <= 0) { if (why) *why = "embed_tokens dims are not positive"; return {}; }
    const int64_t H = packed * 2;                       // 4-bit: two codes per byte
    if (H % groups || (H / groups) % 32) { if (why) *why = "embed_tokens group " + std::to_string(H / groups) + " is not a multiple of 32"; return {}; }
    // the component's own external data, bounds-checked the same way the main graph's is
    std::map<std::string, const uint8_t*> ext;
    std::map<std::string, size_t> ext_size;
    for (const OnnxTensor* t : {table, scales}) {
        if (t->location.empty() || ext.count(t->location)) continue;
        size_t sz = 0;
        std::string w2;
        // external data is named relative to the GRAPH, which for a component pack is a subdirectory: the real pack's
        // embedding graph declares `embedding.onnx.data`, which lives beside it in `embedding/`, not at the pack root.
        const uint8_t* m = map_file(fs::path(gpath).parent_path().string() + "/" + t->location, sz, &w2);
        if (!m) { if (why) *why = w2; return {}; }
        ext[t->location] = m;
        ext_size[t->location] = sz;
    }
    auto get = [&](const OnnxTensor& t, uint64_t need) -> const uint8_t* {
        const auto it = ext.find(t.location);
        if (it == ext.end()) throw std::runtime_error(t.name + ": no data");
        if (!ext_range_ok(ext_size[t.location], t.offset, t.length))
            throw std::runtime_error(t.name + ": external data past the end of " + t.location);
        if (need > t.length)
            throw std::runtime_error(t.name + ": needs " + std::to_string(need) + " B, declares " +
                                     std::to_string(t.length));
        return it->second + t.offset;
    };
    const uint64_t n_scales = uint64_t(V) * uint64_t(groups);
    if (scales->type != kF32) { if (why) *why = "embed_tokens scales are not f32"; return {}; }
    const uint8_t* sc = get(*scales, n_scales * 4);
    const uint8_t* qw = get(*table, uint64_t(V) * uint64_t(packed));
    const int block = int(H / groups);
    // Two conventions, both settled against ONNX Runtime running this pack's own embedding.onnx rather than assumed:
    // the first element of a pair is the LOW nibble (`swap_nibbles = false`), and the codes are **asymmetric int4 with
    // zero point 8**, i.e. value = (code - 8) * scale. Measured on token 1 of
    // amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu: low-first + zp 8 reproduces ORT's row with cosine 1.000000 and max|d| 0,
    // while low-first + zp 0 gives cosine 0.317 and either high-first variant gives ~-0.05.
    const int zp = 8;                     // 2^(bits-1) for GatherBlockQuantized's unsigned 4-bit codes
    const Q4 q = decode_matmulnbits(qw, reinterpret_cast<const float*>(sc), nullptr, int(V), int(H), block, false, true);
    std::vector<float> out(size_t(V) * size_t(H));
    for (int64_t v = 0; v < V; ++v)
        for (int64_t k = 0; k < H; ++k) {
            const size_t gi = size_t(v) * size_t(H / 32) + size_t(k) / 32;
            out[size_t(v) * size_t(H) + size_t(k)] =
                float(int(q.codes[size_t(v) * size_t(H) + size_t(k)]) - zp) * q.scale[gi];
        }
    return out;
}

namespace {

// ── the pack's geometry, from wherever this generation put it ────────────────
// Ryzen AI 1.6 ships `config.json = {}` and 1.7/1.7.1 ship **no config.json at all**; all of them carry
// genai_config.json with the decoder block. So the geometry comes from config.json when it is usable and from
// genai_config.json otherwise. Every refusal names the key it needed and the file it looked in, because a
// nlohmann exception out of a loader ("key hidden_size not found", "type must be string, but is null") is a bug
// report, not a diagnosis -- pi-inferzo's item 3.
// ── the multi-component packs: geometry under model_config.json ─────────────────────────────────
// The gemma-4 1.9 packs ship `config.json = {}` and a genai_config decoder block that stops at
// num_key_value_heads/head_size, so the third value read_dims() needs -- intermediate_size -- is stated nowhere
// those two files are read from, and the importer refused with "the pack states no intermediate_size". It is one
// file over: model_config.json keeps the text tower under config.model_attributes.text_config.
//   amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu: 1536 hidden, 35L, 8 heads, 1 kv, head_dim 256, IM 6144, vocab 262144
//   amd/gemma-4-12B-it_rai_1.9.0_npu:            3840 hidden, 48L, 16 heads, 8 kv, head_dim 256, IM 15360, vocab 262144
// (measured 2026-11-09 on both packs' own JSON). `type: compositemodel` says what the file is; the text tower is
// the decoder this importer builds, so its keys fill gaps and never override a value already resolved -- the same
// rule the hybrid text_config lift uses. Returns how many keys it added, so the caller can name the source.
static int fill_from_model_config_impl(const std::string& pack_dir, json& cfg) {
    std::ifstream f(pack_dir + "/model_config.json");
    if (!f) return 0;
    json mc;
    try { mc = json::parse(f); } catch (...) { return 0; }
    if (!mc.is_object()) return 0;
    const json* attrs = nullptr;
    if (mc.contains("config") && mc["config"].is_object() && mc["config"].contains("model_attributes") &&
        mc["config"]["model_attributes"].is_object()) {
        attrs = &mc["config"]["model_attributes"];
    } else if (mc.contains("model_attributes") && mc["model_attributes"].is_object()) {
        attrs = &mc["model_attributes"];
    }
    if (!attrs) return 0;
    const json* tower = attrs;
    if (attrs->contains("text_config") && (*attrs)["text_config"].is_object()) tower = &(*attrs)["text_config"];
    int added = 0;
    for (auto it = tower->begin(); it != tower->end(); ++it) {
        if (cfg.contains(it.key()) || it.value().is_null()) continue;
        cfg[it.key()] = it.value();
        ++added;
    }
    return added;
}

json load_geometry(const std::string& pack_dir, std::string* source) {
    auto readable_object = [](const std::string& path, json& out) {
        std::ifstream f(path);
        if (!f) return false;
        try { out = json::parse(f); } catch (...) { return false; }
        return out.is_object() && !out.empty();
    };
    json cfg;
    if (readable_object(pack_dir + "/config.json", cfg)) {
        if (source) *source = "config.json";
        // genai_config.json still carries rope values some 1.6 packs omit from config.json
        json g;
        if (readable_object(pack_dir + "/genai_config.json", g)) {
            for (const char* k : {"rope_theta", "rms_norm_eps", "intermediate_size"})
                if (!cfg.contains(k) && g.contains(k)) cfg[k] = g[k];
        }
        // A hybrid pack can keep its TEXT tower under `text_config` INSIDE config.json, with no separate model_config.json
        // to read: amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu states hidden_size/num_hidden_layers/num_attention_heads/
        // intermediate_size (2560/36/32/9728) there and none at the top level, and its model_config.json has
        // model_attributes: null, so fill_from_model_config_impl() adds nothing. The importer then refused with "the pack
        // states no intermediate_size". The text tower IS the decoder this importer builds, so its keys fill gaps -- the
        // same rule fill_from_model_config_impl() and tools/npu_plan.py already use for VL packs, never overriding.
        int from_text_config = 0;
        if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
            for (auto it = cfg["text_config"].begin(); it != cfg["text_config"].end(); ++it) {
                if (cfg.contains(it.key()) || it.value().is_null()) continue;
                cfg[it.key()] = it.value();
                ++from_text_config;
            }
        }
        // `rope_theta` (and the rotary block's other keys) can be ONE LEVEL DEEPER still: Qwen3-VL-4B declares
        // `text_config.rope_parameters = {mrope_interleaved: true, mrope_section: [24,20,20], rope_theta: 5000000,
        // rope_type: default}` with no `rope_theta` anywhere else, so this reader fell back to its 10000 default and the
        // importer warned that the pack's cos/sin caches "no plain theta explains (best 0.5918 at 3.2e+06, declared 10000)".
        // Measured against the pack's own cos_cache (262144 x 64, external): **theta 5e6 reproduces row 1 to max|d|
        // 0.000000**, 10000 is off by 0.088 and 3.2e6 by 0.005 -- so the caches are not unexplained at all, the theta was
        // simply not read (engine#1503). Gaps only, never overriding, same rule as the lift above.
        int from_rope_params = 0;
        if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
            for (auto it = cfg["rope_parameters"].begin(); it != cfg["rope_parameters"].end(); ++it) {
                if (cfg.contains(it.key()) || it.value().is_null()) continue;
                cfg[it.key()] = it.value();
                ++from_rope_params;
            }
        }
        if (from_rope_params > 0 && source) *source += " + config.json rope_parameters";
        if (from_text_config > 0 && source) *source += " + config.json text_config";
        if (fill_from_model_config_impl(pack_dir, cfg) > 0 && source) *source += " + model_config.json (text_config)";
        return cfg;
    }
    json g;
    if (readable_object(pack_dir + "/genai_config.json", g) && g.contains("model") &&
        g["model"].is_object() && g["model"].contains("decoder") && g["model"]["decoder"].is_object()) {
        const json& m = g["model"];
        const json& dec = m["decoder"];
        cfg = json::object();
        if (dec.contains("hidden_size")) cfg["hidden_size"] = dec["hidden_size"];
        if (dec.contains("num_hidden_layers")) cfg["num_hidden_layers"] = dec["num_hidden_layers"];
        if (dec.contains("num_attention_heads")) cfg["num_attention_heads"] = dec["num_attention_heads"];
        if (dec.contains("num_key_value_heads")) cfg["num_key_value_heads"] = dec["num_key_value_heads"];
        if (dec.contains("head_size")) cfg["head_dim"] = dec["head_size"];
        if (m.contains("vocab_size")) cfg["vocab_size"] = m["vocab_size"];
        if (m.contains("context_length")) cfg["max_position_embeddings"] = m["context_length"];
        if (m.contains("type") && m["type"].is_string()) cfg["model_type"] = m["type"];
        // The stop/start tokens live HERE on this line -- these packs ship no generation_config.json at all, which I
        // had assumed they did. A container that does not state its eos cannot stop on it.
        for (const char* k : {"eos_token_id", "bos_token_id", "pad_token_id"})
            if (m.contains(k) && !m[k].is_null()) cfg[k] = m[k];
        if (m.contains("type") && m["type"].is_string()) {
            // genai_config states the family in lower case ("qwen2", "llama"); the Hugging Face spelling that other
            // routes match is capitalised, so it is normalised rather than concatenated as-is.
            std::string fam = m["type"].get<std::string>();
            if (!fam.empty()) fam[0] = char(std::toupper((unsigned char)fam[0]));
            cfg["architectures"] = std::vector<std::string>{fam + "ForCausalLM"};
        }
        if (source) *source = "genai_config.json (decoder block)";
        if (fill_from_model_config_impl(pack_dir, cfg) > 0 && source) *source += " + model_config.json (text_config)";
        return cfg;
    }
    if (source) *source = "neither config.json nor genai_config.json";
    return json::object();
}

// The graph's own name: 1.9 and the hybrid lines ship `model_jit.onnx` / `<Model>_jit.onnx` / `<Model>.onnx` and say
// which in `genai_config.json`'s `model.decoder.filename` (pi-inferzo's item 2). A pack that names no file keeps the
// 1.8 default.
std::string graph_filename(const std::string& pack_dir) {
    std::ifstream f(pack_dir + "/genai_config.json");
    if (f) {
        try {
            const auto g = json::parse(f);
            if (g.contains("model") && g["model"].is_object() && g["model"].contains("decoder") &&
                g["model"]["decoder"].is_object()) {
                const json& dec = g["model"]["decoder"];
                if (dec.contains("filename") && dec["filename"].is_string()) {
                    const std::string n = dec["filename"].get<std::string>();
                    // A multi-component pack (gemma-4 1.9, `type: compositemodel`) keeps its LLM graph in a
                    // subdirectory -- genai_config.json says `decoder/model.onnx` -- so a relative path is accepted.
                    // Absolute paths and `..` are still refused: the name comes from the pack, and a pack must not be
                    // able to aim the reader outside itself.
                    if (!n.empty() && n.front() != '/' && n.find("..") == std::string::npos) return n;
                }
            }
        } catch (...) {
            // an unreadable genai_config is reported by load_geometry; here it only costs the default name
        }
    }
    return "model.onnx";
}

// One key of the geometry, refused by name (and by source) instead of as a json exception.
int dims_required(const json& c, const char* key, const std::string& source) {
    if (!c.contains(key) || c[key].is_null())
        throw std::runtime_error("the pack's geometry has no " + std::string(key) + " (read from " + source +
                                 "): the pack must carry it in config.json or genai_config.json");
    if (!c[key].is_number_integer())
        throw std::runtime_error(std::string("the pack's ") + key + " is not an integer (read from " + source + ")");
    return c[key].get<int>();
}

// An optional field, read the way a pack means it. nlohmann's value() throws when the key is
// present AND null (`type must be number, but is null`), which is not absence: AMD's 1.9
// Mistral-7B-Instruct-v0.2 pack writes `"head_dim": null` (and `"sliding_window": null`) to mean
// "unset, use the default", and the throw turned an 8 GB pack into "info failed" with no geometry
// at all (engine#1650). A wrong TYPE still throws inside get<T>(): that is a real disagreement,
// not a default.
template <typename T>
T opt_field(const json& c, const char* key, T fallback) {
    const auto it = c.find(key);
    return (it == c.end() || it->is_null()) ? fallback : it->get<T>();
}

Dims read_dims(const json& c, const std::string& source = "config.json") {
    Dims d;
    d.hidden = dims_required(c, "hidden_size", source);
    d.layers = dims_required(c, "num_hidden_layers", source);
    d.heads = dims_required(c, "num_attention_heads", source);
    d.kv_heads = opt_field<int>(c, "num_key_value_heads", d.heads);
    d.head_dim = opt_field<int>(c, "head_dim", d.hidden / d.heads);
    d.intermediate = dims_required(c, "intermediate_size", source);
    d.vocab = opt_field<int>(c, "vocab_size", 0);
    d.rope_theta = opt_field<float>(c, "rope_theta", 10000.0f);
    d.model_type = opt_field<std::string>(c, "model_type", "");
    d.rms_norm_eps = opt_field<float>(c, "rms_norm_eps", 1e-6f);
    // engine#1889: the pack's own statement about per-head q/k gains. `opt_field<bool>` because the key is absent on
    // every pack that has no such norm, and absent must mean false rather than a parse error.
    d.use_qk_norm = opt_field<bool>(c, "use_qk_norm", false);
    d.eps_bits = bf16_bits_of(d.rms_norm_eps);
    if (c.contains("eos_token_id")) {
        const auto& e = c["eos_token_id"];
        if (e.is_array()) d.eos = e.get<std::vector<int>>();
        else if (e.is_number_integer()) d.eos = {e.get<int>()};
    }
    return d;
}

}  // namespace

// engine#1650: exported so the config reader can be tested on a pack that declares an optional
// field as null, without a pack and without the vendor stack.
Dims read_dims_exported(const json& c, const std::string& source) { return read_dims(c, source); }
int parse_onnx_matmul_census(const std::string& path, int* matmuls, int* without_zeros, int* K, int* N, int* block) {
    const OnnxModel m = parse_onnx(path);
    if (matmuls) *matmuls = int(m.matmuls.size());
    if (without_zeros) {
        int n = 0;
        for (const auto& mm : m.matmuls) if (mm.zeros.empty()) ++n;
        *without_zeros = n;
    }
    if (!m.matmuls.empty()) {
        if (K) *K = int(m.matmuls[0].K);
        if (N) *N = int(m.matmuls[0].N);
        if (block) *block = m.matmuls[0].block;
    }
    return int(m.matmuls.size());
}
// The 1.9 external-data genre states no FFN width anywhere a file shows it: the weights are MatMulNBitsBf nodes
// whose initializers carry dims `[0]` (bytes live in a *.qweight.packed sibling), so the initializer-dimension
// heuristic in plan() sees nothing. The NODE carries K and N as attributes, and walk_graph already descends into
// the `If` branches every OGA pack keeps its decoder in. The FFN is then a pair that agrees with itself: the down
// projection's K is the gate/up projection's N, and the gate/up projection's K is the hidden size.
//   amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu (hidden 2560): 42 activations each feed two (N=10240,K=2560)
//   gate/up projections, and 42 down projections are (N=2560,K=10240) -> 10240. Its per-token "mobius" paths
//   (N=512/2048, K=2560) have no matching down projection, so they cannot be mistaken for the FFN. Measured
//   2026-11-09 on the pack itself (onnx 1.23.2, node attributes only; its initializers' dims are all [0]).
// Returns 0 when there is no such pair, or when two widths both pair up: a derivation that can be wrong in two
// ways is refused rather than picked from.
int intermediate_size_from_graph(const std::string& model_path, int64_t hidden) {
    if (hidden <= 0) return 0;
    const OnnxModel m = parse_onnx(model_path);
    // Attention forms the same (N == others' K, K == hidden) shape as the FFN: q_proj is (N = heads*head_dim, K = hidden)
    // and o_proj is (N = hidden, K = heads*head_dim), so a bare "pair that agrees" returns the ATTENTION width as
    // intermediate_size on any model whose attention is wider than its hidden size (review of #1657: fused gate_up +
    // q_dim 4096 against a true 9728 gave 4096 -- a silently wrong number from a function whose contract is to refuse).
    // Two things keep it out, and neither is a guess:
    //   1. the role names this genre really carries (`...attn.qkv_proj.MatMulNBits.qweight`, `...mlp.down_proj...`), and
    //   2. a structural witness the FFN has and attention does not: gate AND up read the SAME activation tensor, so two
    //      nodes share one input and one N. A lone q_proj never does, and o_proj has no partner.
    // A fused gate_up (N = 2*w over the hidden width) is paired explicitly with the down projection's K = 2*w.
    const auto excluded = [](const std::string& weight) {
        for (const char* role : {"attn", "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "lm_head"})
            if (weight.find(role) != std::string::npos) return true;
        return false;
    };
    std::map<std::string, std::map<int64_t, int>> shared;  // activation tensor -> N -> how many nodes read it
    std::set<int64_t> down_k, gate_n;                     // down: N == hidden, K > hidden;  gate/up: K == hidden, N > hidden
    for (const auto& mm : m.matmuls) {
        if (mm.K <= 0 || mm.N <= 0 || excluded(mm.weight)) continue;
        if (mm.K == hidden && mm.N > hidden) {
            shared[mm.data][mm.N]++;
            gate_n.insert(mm.N);
        }
        if (mm.N == hidden && mm.K > hidden) down_k.insert(mm.K);
    }
    // separate gate/up first: it is the witness attention cannot produce
    std::set<int64_t> separate;
    for (const int64_t dk : down_k)
        for (const auto& [activation, by_n] : shared) {
            const auto it = by_n.find(dk);
            if (it != by_n.end() && it->second >= 2) separate.insert(dk);
        }
    if (separate.size() == 1) return int(*separate.begin());
    if (separate.size() > 1) return 0;
    // else a fused gate_up: ONE node covers gate and up, so its N is 2*w -- but the down projection reads the GATED
    // product, which is w wide, so its K is w, not 2*w. So the width is a down K whose double is a gate N (engine#1756).
    // The previous rule paired dk with gate_n.count(dk) and halved it: on a real fused MLP (gate_up N = 2w, down K = w)
    // that never matched and returned 0, and with opaque tensor names it matched ATTENTION instead (o_proj's K and
    // q_proj's N are both q_dim) and returned q_dim / 2 -- a wrong intermediate_size rather than a refusal.
    //
    // Attention CAN satisfy this rule, so the `size() != 1` guard below is load-bearing, not tidiness: a fused qkv_proj
    // has N = q_dim + 2*kv_dim, which is exactly 2*q_dim at a 2:1 GQA ratio, and o_proj's K is q_dim, so with opaque
    // names (no role filter) the pair matches and offers q_dim as a candidate. Beside any FFN that is a SECOND candidate
    // and the guard refuses (0) instead of guessing; a separate gate/up returns above before reaching here. So a false
    // match can only cost a derivation, never produce a wrong width -- keep the guard (engine#1756 review of #1757).
    std::set<int64_t> fused;
    for (const int64_t dk : down_k)
        if (gate_n.count(2 * dk)) fused.insert(dk);
    if (fused.size() != 1) return 0;
    return int(*fused.begin());
}
// engine#1740: a 1.9 pack's stride is KNOWN before the eps vote -- canonical_token_fconst() derives it from the 112-byte
// per-head header to build its copy -- but `locate_fconst`'s epsilon-anchor vote can still lose when the payload is dense in
// eps-SHAPED bytes (bf16 1e-6 followed by ten zeros is not rare in 4-bit weights). Measured on Qwen2-7B 1.9: the right stride
// won only 52.6 % of lag-2 pairs, under the 66 % bar, so a 3.8 GB pack structurally identical to two measured ones was refused
// with "the epsilon headers do not repeat with a single period" while its own header reported the stride exactly.
// So ask the header first on the SOURCE file (the canonical copy has the 1.9 header removed by construction), and keep the eps
// vote as the fallback for packs with no 1.9 header -- the 1.8 / R1-Distill line, where it works today.
FconstGeom locate_fconst_with_header(const std::string& tok, const std::string& pack_dir, const Dims& dims) {
    FconstGeom g = locate_fconst(tok, dims.eps_bits);
    const std::string src = find_token_fconst(pack_dir);
    if (src.empty()) return g;
    Fconst fc = open_fconst(src);
    int64_t hdr_off = 0, stride9 = 0;
    const bool is19 = fconst19_detect(fc, dims.heads, dims.layers, &hdr_off, &stride9);
    unmap_file(fc.map, fc.size);
    if (!is19) return g;                                  // no 1.9 header: the eps vote is the only witness, keep it
    const int64_t stride8 = stride9 - int64_t(dims.heads) * 4;
    if (stride8 <= 0) return g;
    if (g.stride != 0 && g.stride != stride8) {
        // Two witnesses that disagree: say so rather than pick. They cannot both be right, and a wrong stride reads weights
        // from the wrong place instead of failing cleanly.
        g.why = "the 1.9 header gives stride " + std::to_string(stride8) + " and the epsilon anchors give " +
                std::to_string(g.stride) + "; refusing rather than choosing";
        g.stride = 0;
        return g;
    }
    g.stride = stride8;
    g.why.clear();
    // The second header offset (`h1`) is what tells the payload reader whether a block holds 2 eps headers or 4, so the eps
    // vote failing must not leave it at 0: with the stride known, the anchors' offsets *within* a block repeat, and the mode
    // of `pos mod stride` (excluding 0, the block start) is the second header. This is the same pool of anchors the vote
    // could not win on, used for a question that does not need a dominance test -- 28 blocks each contribute their residue.
    // engine#1831: only when the vote did NOT record the block's headers. When it did, its list is the measurement
    // (`plan()` reads 2-header vs 4-header from `hdr.size()`, and `post_attn_hdr()` is `hdr.back()`), and rewriting it
    // to one entry turned every Qwen3 4-header 1.9 block into a 2-header one whose "post-attention" header was k_norm's.
    if (!g.hdr.empty()) return g;
    const std::vector<int64_t> pos = eps_header_positions(tok, size_t(1) << 30, dims.eps_bits);
    // The scan window is the first GiB (about 8 blocks at h3584), so a real section boundary repeats ~8 times against
    // 1-2 for a payload coincidence; 3 separates them (requiring `layers/2` was too strict and dropped the real one).
    const std::vector<int64_t> inner = fconst_block_headers(pos, stride8, 3);
    if (!fconst_set_block_headers(g, inner)) {
        g.why = "the 1.9 header gives stride " + std::to_string(stride8) + ", but the epsilon anchors repeat at " +
                std::to_string(inner.size()) + " in-block offset(s), and a layer block has 1 (2-header) or 3 (4-header); "
                "refusing rather than guessing the layout";
        g.stride = 0;
    }
    return g;
}

// engine#1831: every in-block header offset the anchors support, sorted. A header at offset `r` of a block of `stride`
// bytes lands at `r` modulo `stride` in every block, so the offsets that recur `min_repeat` or more times are the block's
// section boundaries; offset 0 is the block start, not an inner header. ALL of them are kept rather than the most common
// one: a Qwen3 block has three equally frequent inner headers, and keeping only the first maximum picked k_norm's.
std::vector<int64_t> fconst_block_headers(const std::vector<int64_t>& pos, int64_t stride, int min_repeat) {
    std::vector<int64_t> out;
    if (stride <= 0) return out;
    std::map<int64_t, int> count;
    for (const int64_t q : pos) {
        const int64_t r = q % stride;
        if (r > 0) ++count[r];
    }
    for (const auto& [r, n] : count)
        if (n >= min_repeat) out.push_back(r);   // std::map iterates in order, so `out` is sorted
    return out;
}

// engine#1831: the fields `locate_fconst()` sets from the same list -- `hdr` (whose SIZE is the block layout), `h1` (its
// first entry) and `period` (headers per block, counting the block's own). One inner header is a 2-header block
// (Qwen2.5/Llama), three is a 4-header block (Qwen3); anything else is not a layout this reader knows, so nothing is set.
bool fconst_set_block_headers(FconstGeom& g, const std::vector<int64_t>& inner) {
    if (inner.size() != 1 && inner.size() != 3) return false;
    g.hdr = inner;
    g.h1 = inner.front();
    g.period = int(inner.size()) + 1;
    return true;
}

std::string find_token_fconst_exported(const std::string& pack_dir) { return find_token_fconst(pack_dir); }

// engine#1638: the geometry READ (not just its dims), so a test can assert where each number came from -- the
// text_config lift and the top-level-wins rule are both properties of load_geometry(), not of read_dims().
json load_geometry_exported(const std::string& pack_dir, std::string* source) { return load_geometry(pack_dir, source); }
int fill_from_model_config(const std::string& pack_dir, nlohmann::json& cfg) {
    return fill_from_model_config_impl(pack_dir, cfg);
}

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
    // Two passes over the same vote. The first keeps the original 80 % dominance rule, so every pack that
    // locates today locates identically. The second (2/3) exists for Ryzen AI 1.9: its 8B Token .fconst has a
    // constant 104,350,656-byte layer block -- 36 blocks plus an 8,256-byte tail, measured -- but the four
    // epsilon headers inside a block sit at offsets that VARY per layer (14 distinct residues across 36
    // layers), so the stride is the right answer in only 117 of 151 lag-4 pairs (77.5 %): correct, and just
    // under the old bar. A weaker bar risks a wrong stride, which is why the geometry it produces is still
    // checked the way it always was -- locate_fconst only proposes; the import re-encodes every tile it reads
    // and refuses when the bytes do not come back identical.
    for (int pass = 0; pass < 2 && !g.stride; ++pass) {
        const int need_num = pass == 0 ? 8 : 2, need_den = pass == 0 ? 10 : 3;
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
                if (pairs < 4 || n * need_den < pairs * need_num) continue;
                if (!g.stride || d > g.stride) {                 // prefer the largest consistent period
                    g.stride = d;
                    g.period = P;
                }
            }
            if (g.stride) break;
        }
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


// ── The implicit zero point ───────────────────────────────────────────────────
// A quantised matmul with no `qzeros` input takes an implicit zero point of 2^(bits-1) -- ONNX Runtime's own
// MatMulNBits does exactly that when the tensor is absent, and it is corroborated in this collection: the *embedding*
// component of the same family (GatherBlockQuantized, also no zeros) decodes to ONNX Runtime's output with cosine
// 1.00000000 and max|d| 0 at zero point 8, against 0.317 at zero point 0 (measured on
// amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu, engine#1725/#1661). `decode_matmulnbits` folds the zero into `q.zero`, so
// this subtracts the implicit one the same way: zero term = -(2^(bits-1)) * scale.
void implicit_zp_zero(Q4& q, int bits) {
    if (bits <= 0 || q.scale.empty() || q.scale.size() != q.zero.size()) return;
    const float zp = float(1 << (bits - 1));
    for (size_t i = 0; i < q.zero.size(); ++i) q.zero[i] -= zp * q.scale[i];
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
// One tiler, two parameter forms. q4_1 stores bf16 scale/zero: that is the GGUF-derived containers'
// form, and where the bf16 choice came from. q4_1f stores them as f32 (npu/q4nx.h), which is what
// `1bit npu-pack import` writes: an OGA pack's zero is -zp*scale, an arbitrary f32 that bf16 rounds
// (measured 2.7-3.3e-3 relative on ~57 % of groups, engine#1185). `param_inexact` counts the groups
// whose stored parameter is not the source's -- the bf16 loss for q4_1, and 0 for q4_1f, where the
// values are copied bit for bit.
namespace {

void tile_q4_impl(const Q4& q, std::vector<uint8_t>& out, bool f32, long long* param_inexact) {
    if (q.N % q4nx::kRows) throw std::runtime_error("q4_1 tiles need N % 32 == 0");
    // The grid is 32 x 256, so a tensor whose K is not a whole number of 256-column
    // tiles (Qwen2.5-0.5B: hidden 896 = 3.5 tiles) is stored with the last column tile
    // zero-padded: codes 0, scale 0, zero 0, i.e. w = 0 in the padding. The real K is
    // in the manifest and the caller's config; nothing reads the padding as a weight.
    const size_t tile_bytes = f32 ? size_t(q4nx::Chunk::Q4_1F) : size_t(q4nx::Chunk::Q4_1);
    const size_t code_off = f32 ? 2048 : 1024;
    const int KP = ((q.K + q4nx::kCols - 1) / q4nx::kCols) * q4nx::kCols;
    const int RT = q.N / q4nx::kRows, KT = KP / q4nx::kCols;
    out.assign(size_t(RT) * KT * tile_bytes, 0);
    long long inexact = 0;
    for (int rt = 0; rt < RT; ++rt) {
        for (int kt = 0; kt < KT; ++kt) {
            uint8_t* tile = out.data() + (size_t(rt) * KT + kt) * tile_bytes;
            auto* sc32 = reinterpret_cast<float*>(tile);
            auto* zr32 = reinterpret_cast<float*>(tile + 1024);
            auto* sc16 = reinterpret_cast<uint16_t*>(tile);
            auto* zr16 = reinterpret_cast<uint16_t*>(tile + 512);
            for (int r = 0; r < q4nx::kRows; ++r) {
                const int row = rt * q4nx::kRows + r;
                for (int g = 0; g < 8; ++g) {
                    const size_t i = size_t(g * 32 + r);
                    if ((kt * 8 + g) * 32 >= q.K) {  // zero padding
                        if (f32) {
                            sc32[i] = 0.0f;
                            zr32[i] = 0.0f;
                        } else {
                            sc16[i] = 0;
                            zr16[i] = 0;
                        }
                        continue;
                    }
                    const size_t idx = size_t(row) * (q.K / 32) + size_t(kt) * 8 + g;
                    if (!f32) {
                        sc16[i] = f32_to_bf16(q.scale[idx]);
                        zr16[i] = f32_to_bf16(q.zero[idx]);
                        if (param_inexact && bf16_to_f32(zr16[i]) != q.zero[idx]) ++inexact;
                    } else {
                        sc32[i] = q.scale[idx];
                        zr32[i] = q.zero[idx];
                        // read the written bytes back: a q4_1f parameter is the source float, bit for bit
                        if (param_inexact && (std::memcmp(&sc32[i], &q.scale[idx], 4) != 0 ||
                                              std::memcmp(&zr32[i], &q.zero[idx], 4) != 0))
                            ++inexact;
                    }
                }
            }
            // codes: region h*8+g, byte 8*j + (r%16)/2, even row in the low nibble
            for (int h = 0; h < 2; ++h)
                for (int g = 0; g < 8; ++g)
                    for (int j = 0; j < 32; ++j) {
                        const int k = kt * q4nx::kCols + g * 32 + j;
                        for (int r2 = 0; r2 < 8; ++r2) {
                            const int r_even = rt * q4nx::kRows + h * 16 + r2 * 2;
                            uint8_t& byte = tile[code_off + size_t(h) * 2048 + size_t(g) * 256 +
                                                size_t(j) * 8 + size_t(r2)];
                            if (k >= q.K) {  // zero padding column
                                byte = 0;
                                continue;
                            }
                            const uint8_t even = q.codes[size_t(r_even) * q.K + k] & 0x0F;
                            const uint8_t odd = q.codes[size_t(r_even + 1) * q.K + k] & 0x0F;
                            byte = uint8_t(even | (odd << 4));
                        }
                    }
        }
    }
    if (param_inexact) *param_inexact = inexact;
}

}  // namespace

void tile_q4_1(const Q4& q, std::vector<uint8_t>& out, long long* zero_inexact) {
    tile_q4_impl(q, out, false, zero_inexact);
}

void tile_q4_1_f32(const Q4& q, std::vector<uint8_t>& out, long long* param_inexact) {
    tile_q4_impl(q, out, true, param_inexact);
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

// ── the 1.9 external-data tile (engine#1503) ─────────────────────────────────
// The `.fconst` tile's subvolume, written densely per tensor instead of inside the fconst envelope. Measured field by
// field on `amd/phi-4_rai_1.9.0_npu`: the blob length is exactly `N x (K/128 + 1) x 68` on all 201 matmuls, the
// all-zero subvolume pair sits at `index % (2*(K/128+1)) == 0,1` (the writer's bias pair, zero here), the scales read
// as bf16 in 5.7e-3 .. 8.5e-3 while reading as ~1.0 the same bytes as f16, and the zero points are int4 nibbles
// (`0x5555`, `0x6666`, `0xAAAA` appear as such).
//
// What the host side could NOT settle is which way the 128 code bytes of one row lie inside the 2,048-byte quants
// region: the subvolume's own doc formula (`(cp%4) + kk*4 + (cp/4)*512`, known-answer verified for the .fconst tile)
// against a plain row-major reading. Three statistics were tried and they disagreed with each other, which discredits
// all three (they all assumed a row's codes centre on its zero-point, and the measured zero points span 0..11). So the
// default is the verified prior and the alternative is one constant, to be settled by the known-answer run.
const char* external_role_of(const std::string& tensor_name) {
    struct Role {
        const char* needle;
        const char* role;
    };
    // Longest-needle-first where one name could contain another: `qkv_proj` before nothing else matters, and
    // `gate_proj`/`up_proj`/`down_proj` do not overlap, but keeping the table explicit is cheaper than a rule.
    static const Role kRoles[] = {
        {"qkv_proj", "QKV"}, {"o_proj", "O"}, {"gate_proj", "G"},
        {"up_proj", "U"},    {"down_proj", "D"}, {"lm_head", "HEAD"},
    };
    // A COMPONENT match, not a substring one: the needle must sit at a name boundary and be followed by a `.` or
    // the end. `model.layers.0.attn.qkv_proj.MatMulNBits.qweight.packed` matches; `...qkv_proj_matmul` does not,
    // and would otherwise be mapped to a role it is not -- the "guess by resemblance" failure this table exists to
    // avoid, caught by the test below rather than by a container full of mislabelled weights.
    for (const auto& r : kRoles) {
        const std::string needle = r.needle;
        for (size_t at = tensor_name.find(needle); at != std::string::npos;
             at = tensor_name.find(needle, at + 1)) {
            const bool left_ok = at == 0 || tensor_name[at - 1] == '.' || tensor_name[at - 1] == '/';
            const size_t end = at + needle.size();
            const bool right_ok = end == tensor_name.size() || tensor_name[end] == '.';
            if (left_ok && right_ok) return r.role;
        }
    }
    return nullptr;
}

bool external_container_tensors(const std::string& role, int nh, int nkv, int hd, std::vector<ContainerTensor>& out,
                                std::string& err) {
    out.clear();
    if (role == "QKV") {
        if (nh <= 0 || nkv <= 0 || hd <= 0) {
            err = "QKV split: the geometry states no head count (nh " + std::to_string(nh) + ", nkv " +
                  std::to_string(nkv) + ", hd " + std::to_string(hd) + ")";
            return false;
        }
        const int q = nh * hd, kv = nkv * hd;
        out.push_back({"self_attn.q_proj.weight", 0, q});
        out.push_back({"self_attn.k_proj.weight", q, q + kv});
        out.push_back({"self_attn.v_proj.weight", q + kv, q + 2 * kv});
        return true;
    }
    if (role == "O") { out.push_back({"self_attn.o_proj.weight", 0, 0}); return true; }
    if (role == "G") { out.push_back({"mlp.gate_proj.weight", 0, 0}); return true; }
    if (role == "U") { out.push_back({"mlp.up_proj.weight", 0, 0}); return true; }
    if (role == "D") { out.push_back({"mlp.down_proj.weight", 0, 0}); return true; }
    if (role == "HEAD") { out.push_back({"lm_head.weight", 0, 0}); return true; }
    err = "external role \"" + role + "\" has no container tensor in this forward (a weight with nowhere to go is a "
          "refusal, not a silent drop)";
    return false;
}

Q4 container_from_external_q4(const Q4& pack, long long* m_inexact) {
    Q4 q = pack;   // same codes, same geometry
    long long inexact = 0;
    for (size_t i = 0; i < q.zero.size(); ++i) {
        // pack: w = (code - zp) * scale  ->  container: w = code * scale + m, so m = -zp * scale.
        const float m = -q.zero[i] * q.scale[i];
        if (bf16_to_f32(bf16_bits_of(m)) != m) ++inexact;
        q.zero[i] = m;
    }
    if (m_inexact) *m_inexact = inexact;
    return q;
}

Q4 decode_external_tile(const uint8_t* data, size_t bytes, int N, int K, std::string& err, bool row_major) {
    Q4 q;
    if (N <= 0 || K <= 0 || N % kExtTileRows != 0 || K % 128 != 0) {
        err = "external tile: N must be a multiple of 64 and K a multiple of 128 (got " + std::to_string(N) + " x " +
              std::to_string(K) + ")";
        return q;
    }
    const int64_t blocks = K / 128, tiles = N / kExtTileRows, per_tile = 2 * (blocks + 1);
    const int64_t want = tiles * per_tile * kExtSubvBytes;
    if (int64_t(bytes) != want) {
        err = "external tile: " + std::to_string(bytes) + " B is not " + std::to_string(want) + " B = " +
              std::to_string(tiles) + " tiles x " + std::to_string(per_tile) + " subvolumes of " +
              std::to_string(kExtSubvBytes) + " (N " + std::to_string(N) + " x K " + std::to_string(K) + ")";
        return q;
    }
    q.N = N;
    q.K = K;
    q.block = 32;
    q.codes.assign(size_t(N) * K, 0);
    q.scale.assign(size_t(N) * (K / 32), 0.0f);
    q.zero.assign(size_t(N) * (K / 32), 0.0f);
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int half = 0; half < 2; ++half) {
            for (int64_t kk = 1; kk <= blocks; ++kk) {   // kk == 0 is the tile's bias pair, all zero here
                const uint8_t* subv = data + (tile * per_tile + kk * 2 + half) * kExtSubvBytes;
                const uint8_t* quants = subv;
                const uint8_t* zeros = subv + 2048;
                const uint8_t* scales = subv + 2112;
                for (int r = 0; r < kExtSubvRows; ++r) {
                    const int64_t row = tile * kExtTileRows + half * kExtSubvRows + r;
                    const float s = bf16_to_f32(uint16_t(scales[r * 2] | (scales[r * 2 + 1] << 8)));
                    // four int4 zero points per row, one per 32-column group of the 128-column block
                    const uint8_t z0 = zeros[r * 2], z1 = zeros[r * 2 + 1];
                    const uint8_t zp[4] = {uint8_t(z0 & 0xF), uint8_t(z0 >> 4), uint8_t(z1 & 0xF), uint8_t(z1 >> 4)};
                    for (int g = 0; g < 4; ++g) {
                        q.scale[size_t(row) * (K / 32) + (kk - 1) * 4 + g] = s;
                        q.zero[size_t(row) * (K / 32) + (kk - 1) * 4 + g] = float(zp[g]);
                    }
                    for (int c = 0; c < 128; ++c) {
                        // Row-major: a row's 128 codes are 64 contiguous bytes, even column in the low nibble.
                        // The .fconst formula: one byte per COLUMN holds the row PAIR 2cp (low) / 2cp+1 (high), so a
                        // row's codes are strided by 4 through a 512-byte block -- no column pairing at all.
                        uint8_t nib;
                        if (row_major) {
                            const uint8_t b = quants[r * 64 + c / 2];
                            nib = (c % 2 == 0) ? uint8_t(b & 0xF) : uint8_t(b >> 4);
                        } else {
                            const int cp = r / 2;
                            const uint8_t b = quants[(cp % 4) + c * 4 + (cp / 4) * 512];
                            nib = (r % 2 == 0) ? uint8_t(b & 0xF) : uint8_t(b >> 4);
                        }
                        q.codes[size_t(row) * K + (kk - 1) * 128 + c] = nib;
                    }
                }
            }
        }
    }
    return q;
}

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

// Exported for the unit test (engine#1690). The head's self-check compares a quantised head against the embedding
// buffer the importer actually holds -- which is the container's bf16 copy, not the pack's original F16 bytes. Reading
// that buffer at the wrong width produced a plausible-looking number (0.885220 on a tied pack whose container scores
// 0.999999) for every pack from #1661 until this existed, so the width is pinned directly rather than only end to end.
namespace {
static double corr(const Q4& q, const EmbedTable& embed, int rows);   // defined below, internal to this file
}
double embed_at(const EmbedTable& t, int row, int k) {
    const uint16_t v = reinterpret_cast<const uint16_t*>(t.p)[size_t(row) * size_t(t.H) + size_t(k)];
    return t.fmt == EmbedTable::Fmt::BF16 ? bf16_to_f32(v) : f16_to_f32(v);
}

double corr_bf16(const Q4& q, const uint8_t* embed_bf16, int H, int rows) {
    return corr(q, EmbedTable{embed_bf16, EmbedTable::Fmt::BF16, H}, rows);
}
namespace {
const Fconst19Rule* select_rule_in(const Fconst19Rule* rules, int count, int hidden, int heads, int layers,
                                   std::string* why);   // defined with the table, internal to this file
}
bool fconst_scratch_ready(const std::string& dir) {
    // A candidate scratch directory is usable when it is (or can be made into) a directory. The cache directory is OURS to create:
    // the version of engine#1801 that shipped first did not, so on a fresh box the loop walked straight past the new first choice
    // and every pack fell through to tmpfs -- the bug this function exists to make testable.
    std::error_code ec;
    if (fs::is_directory(dir, ec) && !ec) return true;
    std::error_code made;
    fs::create_directories(dir, made);
    return fs::is_directory(dir, ec) && !ec;
}

std::vector<std::string> fconst_scratch_dirs() {
    // Where the canonical .fconst copy may live, in preference order. A function because the ORDER is the fix from engine#1801 and
    // the reviewer's ask was a test that pins it: on strixhalo /tmp is tmpfs, so the first choice has to be disk, and a version of
    // this that forgot to create the cache directory silently fell through to /tmp for every pack.
    std::vector<std::string> dirs;
    if (const char* scratch = std::getenv("ONEBIT_FCONST_SCRATCH"); scratch && *scratch)
        dirs.push_back(scratch);
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        dirs.push_back(std::string(xdg) + "/1bit/fconst19");
    else if (const char* home = std::getenv("HOME"); home && *home)
        dirs.push_back(std::string(home) + "/.cache/1bit/fconst19");
    // engine#1818 review: `$TMPDIR` belongs BETWEEN the cache and the literal /tmp. It was dropped in this extraction, and on a
    // lane that points TMPDIR at disk (which is what people do here precisely because /tmp is RAM) a pack whose cache directory
    // is unusable then landed in tmpfs -- the outcome engine#1801 exists to prevent. temp_directory_path() throws when TMPDIR
    // names a missing directory, hence the try and the note; it also returns /tmp when TMPDIR is unset, so the literal below is
    // still the final disc-resort and not a duplicate.
    try {
        const fs::path tmp = fs::temp_directory_path();
        if (!tmp.empty() && tmp != fs::path("/tmp")) dirs.push_back(tmp.string());
    } catch (...) {
        // The temporary directory is unusable (TMPDIR names something that is not there): go elsewhere.
    }
    // Last resorts only: on this box /tmp is tmpfs, so each of these costs RAM equal to the pack's .fconst.
    dirs.push_back("/tmp");
    dirs.push_back(".");
    return dirs;
}

const Fconst19Rule* fconst19_rules(int* count) {
    if (count) *count = int(sizeof(kFconst19Rules) / sizeof(kFconst19Rules[0]));
    return kFconst19Rules;
}

const Fconst19Rule* select_fconst19_rule(int hidden, int heads, int layers, std::string* why) {
    return select_rule_in(kFconst19Rules, int(sizeof(kFconst19Rules) / sizeof(kFconst19Rules[0])), hidden, heads, layers,
                          why);
}

const Fconst19Rule* select_fconst19_rule_from(const Fconst19Rule* rules, int count, int hidden, int heads, int layers,
                                              std::string* why) {
    return select_rule_in(rules, count, hidden, heads, layers, why);
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
    /* engine#1918: the nibble packer below applies `& 0x0F`, so a code outside 0..15 WRAPS instead of saturating -- a
     * weight a full code-range away that still looks plausible (`container == expected mod 4.0` over 32/32 columns of a
     * group, measured on #1863's arm). Whether the caller is supposed to guarantee the range or the writer should clamp
     * is not documented anywhere, and a wrapped weight on a model that then answers fluently is the worst outcome
     * available; so the writer refuses instead, naming the position and the value. A caller that means 15 says 15. */
    for (size_t i = 0; i < t.code.size(); ++i)
        if (t.code[i] > 0x0F)
            throw std::runtime_error("tile: code " + std::to_string(int(t.code[i])) + " at k=" + std::to_string(t.k) +
                                     " rows=" + std::to_string(t.rows) + ", element " + std::to_string(i) +
                                     ": a 4-bit code is 0..15, and packing it would wrap a full code range "
                                     "(engine#1918) -- clamp or re-quantise before encoding");
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


constexpr int kLocateSlack = 8;  // tiles of search before the zero-run position

// The placement arithmetic on its own: no file, no decode, no correlation. Split out because it is the half that can be
// wrong without anything throwing -- an off-by-one tile here reads a plausible region and only the correlation notices
// -- so it is pinned by a host test that needs no pack, no device and no fixture.
// TailHeadPlacement is declared in the header: the placement arithmetic is pinned by a host test.
// bytes = the whole .fconst's length; zeros = trailing all-zero run measured by the caller.
TailHeadPlacement plan_tail_head(int64_t bytes, int64_t tail_off, int64_t env, int64_t rows, int64_t tile,
                                        int64_t zeros) {
    TailHeadPlacement p;
    p.rows = rows;
    p.zeros = zeros;
    p.have = bytes - tail_off;
    p.want = env + rows * tile;
    if (p.have < p.want) {
        p.why = "the .fconst tail is " + std::to_string(p.have) + " B, shorter than the " + std::to_string(p.want) +
                " B its envelope and head tiles would take";
        return p;
    }
    p.exact = (p.have == p.want);
    const int64_t base = tail_off + env;
    p.last = base + ((bytes - zeros - base) / tile - rows) * tile;
    p.first = p.last - kLocateSlack * tile;
    if (p.first < base) p.first = base;
    if (p.last < base) p.last = base;
    return p;
}

// engine#1740: where the 1.9 Token .fconst keeps the lm head.
//
// The tail is [envelope][rows x tile][padding], and only the middle term is the head. The importer used to require
// `have == env + rows*tile` exactly, which is true for a pack that ends with the head and false for one that ends with
// padding -- and on the hidden-1536 family the padding is 110,208,000 B, 11 % of the file. Refusing there is a named
// refusal, but it refuses the head for a reason that is about placement, not about the fields, and 12+ of the 42 1.9
// packs keep their head only here.
//
// So the placement is measured rather than assumed, in this order:
//   1. the run of all-zero bytes at EOF (`zeros`), whatever its alignment -- a pack can also end in junk that is not a
//      whole number of tiles (Hunyuan-0.5B's tail is 28 B off), and that has to be visible rather than rounded away;
//   2. `rows` consecutive tile-aligned candidates, each starting at most kSlack tiles before the position the zero run
//      implies, because "the data ends where the zeros begin" is arithmetic and the head may not sit there;
//   3. the round-trip (decode -> encode) proves the FIELD ORDER and nothing else: an all-zero tile decodes to zeros and
//      re-encodes to zeros exactly, so padding passes it. Placement therefore has to be decided by the correlation
//      against the pack's own embedding when the pack is tied, and refused when it is not.
//
// The outcome is a value, not a std::string, so the caller cannot accidentally proceed on a refusal.
struct TailHeadLocate {
    int64_t off = -1;        // byte offset of head tile 0
    int64_t rows = 0;        // tiles of the head
    int64_t zeros = 0;       // trailing all-zero bytes at EOF
    int64_t slack = 0;       // tiles the accepted placement sits before the zero-run position
    double  corr = 0.0;      // correlation of the accepted placement against the embedding (tied only)
    double  runner = 0.0;    // best rejected placement's correlation, for the ambiguity message
    bool    verified = false;// corr above kTieCorr: this is the head, not merely a decodable region
    bool    exact = false;   // the tail length matched with no padding at all (the pre-#1740 path)
    std::string why;         // set on refusal: what was measured, in numbers
};

static int64_t trailing_zero_run(const Fconst& fc, int64_t from) {
    const uint8_t* m = fc.map;
    int64_t i = fc.size;
    while (i > from && m[i - 1] == 0) --i;
    return int64_t(fc.size) - i;
}

// `want` is env + rows*tile at this geometry; `tail_off` is where the tail begins (layers x stride).
static TailHeadLocate locate_tail_head(const Fconst& fc, int64_t tail_off, int64_t env, int64_t rows, int64_t tile,
                                      int64_t k, const EmbedTable& embed, bool tied) {
    TailHeadLocate r;
    r.rows = rows;
    r.zeros = trailing_zero_run(fc, tail_off);
    const TailHeadPlacement plan = plan_tail_head(int64_t(fc.size), tail_off, env, rows, tile, r.zeros);
    r.exact = plan.exact;
    if (!plan.why.empty()) {
        r.why = plan.why;
        return r;
    }
    // Candidate 0 is where the zero run says the head ends; earlier candidates cover a head that does not end at the
    // last non-zero byte. Every candidate must be tile-aligned to the same phase as the envelope, so a jumbled offset
    // cannot be mistaken for a placement.
    const int64_t base = tail_off + env;
    // The planner gives the earliest and latest tile-aligned candidate; the zero run decides the latest.
    const int64_t k0 = (plan.last - base) / tile;
    int64_t best_slack = -1, second_slack = -1;
    double best = -2.0, second = -2.0;
    int64_t tried = 0;
    // engine#1773 review: the search window is anchored on the zero run, so when the file ends in NON-zero junk after
    // the head the window slides forward and can EXCLUDE the true start (`base`, the envelope's end) -- arm (c2) builds
    // exactly that shape and blessed the span without noticing the one placement that matters was outside it. `base` is
    // known independently of the zero run, so it is always a candidate; the slack window is kept for the rest.
    std::vector<int64_t> cands;
    cands.push_back(0);                                  // the head begins right after the envelope
    for (int64_t cand = (plan.first - base) / tile; cand <= k0; ++cand)
        if (cand != 0) cands.push_back(cand);
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        const int64_t cand = cands[ci];
        const int64_t s = base + cand * tile;
        if (s < base || s + rows * tile > int64_t(fc.size)) continue;
        std::vector<TileW> tiles;
        tiles.reserve(size_t(rows));
        bool exact_rt = true;
        for (int64_t i = 0; i < rows; ++i) {
            const uint8_t* src = fc.map + s + i * tile;
            tiles.push_back(decode_tile(src, int(k)));
            const std::vector<uint8_t> back = encode_tile(tiles.back());
            if (std::memcmp(back.data(), src, size_t(tile)) != 0) exact_rt = false;
        }
        ++tried;
        if (!exact_rt) continue;
        double c = 0.0;
        if (tied) {
            const Q4 q = q_from_tiles(tiles);
            if (q.N < int(embed.H) || q.K != int(embed.H)) continue;
            c = corr(q, embed, int(embed.H));
        }
        if (c > best) {
            second = best; second_slack = best_slack;
            best = c; best_slack = cand;
        } else if (c > second) {
            second = c; second_slack = cand;
        }
    }
    if (tried == 0) {
        r.why = "no tile-aligned placement in the last " + std::to_string(rows + kLocateSlack) + " tiles decodes and "
                "re-encodes exactly, so the head's fields are not understood at this geometry";
        return r;
    }
    if (best_slack < 0) {
        r.why = "the tail holds " + std::to_string(tried) + " tile-aligned placement(s) and none of them survives "
                "decode->encode, so the head's offsets are not established";
        return r;
    }
    // `off` stays -1 until a placement is ACCEPTED: the caller decides on `off < 0`, so a refusal that left an offset
    // behind would be read by the caller -- which is exactly what the self-test's untied arm caught, because every
    // refusal path here sets `why` but the offset had already been assigned.
    const int64_t chosen = base + best_slack * tile;
    if (!tied) {
        // No reference inside the pack: only the exact placement is taken, and it is reported as unverified.
        if (!r.exact || best_slack != 0) {
            r.why = "this pack is untied, so the head cannot be verified against its embedding, and its tail carries " +
                    std::to_string(r.zeros) + " B of trailing zeros -- the placement would be a guess";
            return r;
        }
        r.off = base;
        r.slack = best_slack;
        r.corr = best;
        r.runner = second;
        return r;
    }
    if (best < kTieCorr) {
        r.off = -1;
        r.why = "the best placement correlates " + std::to_string(best) + " with the pack's own embedding (the tie "
                "threshold is " + std::to_string(kTieCorr) + "), so no placement in the tail is the head";
        return r;
    }
    if (second > kTieCorr) {
        r.off = -1;
        r.why = "two placements both correlate with the embedding (" + std::to_string(best) + " at offset " +
                std::to_string(base + best_slack * tile) + ", " + std::to_string(second) + " at offset " +
                std::to_string(base + (second_slack < 0 ? 0 : second_slack) * tile) +
                "), so the head's offset is ambiguous";
        return r;
    }
    r.off = chosen;
    r.slack = best_slack;
    r.corr = best;
    r.runner = second;
    r.verified = true;
    return r;
}

// engine#1740: a self-test for the SEARCH, not the arithmetic. `plan_tail_head` is pinned by the host test; this pins
// the part that can go silently wrong -- the candidate loop, best/runner-up tracking, the kTieCorr cut, the ambiguity
// refusal and the untied-with-padding refusal -- using a synthetic head built through encode_tile, so it needs no pack,
// no device and no fixture. Called from the importer's test, and usable while W1b works on the block-32 placement.
bool tail_head_selftest(std::string* why) {
    auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
    const int H = kKBlock, rows_per_tile = kTileRows, ntiles = 4;
    const int64_t tile = attn_tile_bytes(H);
    const int64_t env = attn_payload_off(H);
    // A deterministic embedding, and the bf16 copy the correlation reads. Pearson is shift- and scale-invariant, so a
    // min-max quantisation of this table correlates with it at ~1 without having to reproduce it byte for byte.
    std::vector<float> emb(size_t(ntiles) * rows_per_tile * H);
    std::vector<uint16_t> emb_bf16(emb.size());
    for (size_t n = 0; n < size_t(ntiles) * rows_per_tile; ++n)
        for (int k = 0; k < H; ++k) {
            const float v = float(std::sin(double(n) * 0.37 + double(k) * 0.11) * 0.5 + 0.25 * double(n % 7));
            emb[n * H + k] = v;
            uint32_t b; std::memcpy(&b, &v, 4);
            emb_bf16[n * H + k] = uint16_t((b + 0x8000) >> 16);
        }
    const EmbedTable embed{reinterpret_cast<const uint8_t*>(emb_bf16.data()), EmbedTable::Fmt::BF16, H};
    // One tile per `rows_per_tile` rows, quantised the way the importer's synthesis path does it (per 32 columns).
    auto make_tile = [&](size_t tile_index) {
        TileW t;
        t.k = H; t.rows = rows_per_tile;
        t.code.assign(size_t(rows_per_tile) * H, 0);
        t.scale.assign(size_t(rows_per_tile) * (H / kKBlock), 0.0f);
        t.zp.assign(size_t(rows_per_tile) * (H / kKBlock), 0);
        t.bias.assign(rows_per_tile, 0);
        for (int r = 0; r < rows_per_tile; ++r) {
            const size_t n = tile_index * rows_per_tile + r;
            for (int kb = 0; kb < H / kKBlock; ++kb) {
                // CENTRED quantisation about a zero point of 7, i.e. exactly what the decoder reconstructs:
                // x = (code - zp) * scale. A min-max scale with the offset dropped (`zp == 0`) reconstructs
                // code * scale, which is per-block shifted -- the first version of this self-test scored corr 0.576
                // for that reason, which is the synthetic head teaching the same lesson the real one does.
                float amax = 0.0f;
                for (int k = 0; k < kKBlock; ++k)
                    amax = std::max(amax, std::fabs(emb[n * H + kb * kKBlock + k]));
                const float sc = amax > 0 ? amax / 7.0f : 1.0f;
                t.scale[size_t(r) * (H / kKBlock) + kb] = sc;
                t.zp[size_t(r) * (H / kKBlock) + kb] = 7;
                for (int k = 0; k < kKBlock; ++k) {
                    const float v = emb[n * H + kb * kKBlock + k];
                    const int code = int(std::lround(v / sc)) + 7;
                    t.code[size_t(r) * H + kb * kKBlock + k] = uint8_t(std::min(15, std::max(0, code)));
                }
            }
        }
        return encode_tile(t);
    };
    auto build = [&](int junk_before, int zero_tiles_after, Q4* want_head) {
        std::vector<uint8_t> buf(size_t(env), 0);
        auto push = [&](const std::vector<uint8_t>& b) { buf.insert(buf.end(), b.begin(), b.end()); };
        if (junk_before > 0) push(std::vector<uint8_t>(size_t(tile) * size_t(junk_before), 0x11));
        std::vector<TileW> tiles;
        for (int i = 0; i < ntiles; ++i) {
            const std::vector<uint8_t> t = make_tile(size_t(i));
            tiles.push_back(decode_tile(t.data(), H, rows_per_tile));
            push(t);
        }
        if (want_head) *want_head = q_from_tiles(tiles);
        if (zero_tiles_after > 0) push(std::vector<uint8_t>(size_t(tile) * size_t(zero_tiles_after), 0));
        return buf;
    };
    const int64_t base = env;   // tail_off == 0 for this synthetic file
    // (a) the head ends the file
    {
        Q4 head;
        std::vector<uint8_t> buf = build(0, 0, &head);
        const Fconst fc{"synthetic", buf.data(), buf.size()};
        const TailHeadLocate loc = locate_tail_head(fc, 0, env, ntiles, tile, H, embed, true);
        if (loc.off < 0) return fail("a synthetic head that ends the file was not located: " + loc.why);
        if (loc.slack != 0) return fail("the head that ends the file was located at slack " + std::to_string(loc.slack));
        if (!loc.verified) return fail("and it was not verified (corr " + std::to_string(loc.corr) + ")");
        if (loc.corr < 0.99) return fail("corr " + std::to_string(loc.corr) + " is below the tie threshold");
    }
    // (b) one junk tile before the head and three zero tiles after it: the head is a tile later, and the junk tile's
    // placement must stay below the threshold -- this is the search doing its job rather than the arithmetic
    {
        std::vector<uint8_t> buf = build(1, 3, nullptr);
        const Fconst fc{"synthetic", buf.data(), buf.size()};
        const TailHeadLocate loc = locate_tail_head(fc, 0, env, ntiles, tile, H, embed, true);
        if (loc.off < 0) return fail("a head behind one junk tile and three zero tiles was not located: " + loc.why);
        if (loc.slack != 1) return fail("it was located at slack " + std::to_string(loc.slack) + ", not 1");
        if (!loc.verified) return fail("and it was not verified");
        if (loc.runner >= kTieCorr) return fail("the runner-up also passed the threshold (" + std::to_string(loc.runner) + ")");
    }
    // (e) engine#1773 review: junk AFTER the head, then zeros. The zero run then sits 18 tiles past the envelope's end,
    // so `plan_tail_head`'s window -- anchored on the zero run and walking backwards -- starts at `base + 10*tile` and
    // the head's true placement at `base` is OUTSIDE it. Arm (b) cannot see this: one junk tile before the head leaves
    // the window clamped to `base`. Only a candidate list that always contains the envelope's end finds the head here.
    {
        std::vector<uint8_t> buf = build(0, 0, nullptr);
        buf.insert(buf.end(), size_t(tile) * 18, 0x11);   // junk after the head ...
        buf.insert(buf.end(), size_t(tile) * 3, 0);       // ... then zeros: last = base + 18 tiles, first = base + 10
        const Fconst fc{"synthetic", buf.data(), buf.size()};
        const TailHeadLocate loc = locate_tail_head(fc, 0, env, ntiles, tile, H, embed, true);
        if (loc.off < 0)
            return fail("a head followed by junk and then zeros was not located: " + loc.why);
        if (loc.off != base)
            return fail("a head followed by junk and then zeros was located at " + std::to_string(loc.off) +
                        " B, not the envelope's end " + std::to_string(base));
    }
    // (c) untied + padding: refused rather than taken
    {
        std::vector<uint8_t> buf = build(0, 3, nullptr);
        const Fconst fc{"synthetic", buf.data(), buf.size()};
        const TailHeadLocate loc = locate_tail_head(fc, 0, env, ntiles, tile, H, embed, false);
        if (loc.off >= 0) return fail("an untied padded tail was taken instead of refused");
        if (loc.why.empty()) return fail("and the refusal says nothing");
    }
    // (d) untied + exact: taken and marked unverified
    {
        std::vector<uint8_t> buf = build(0, 0, nullptr);
        const Fconst fc{"synthetic", buf.data(), buf.size()};
        const TailHeadLocate loc = locate_tail_head(fc, 0, env, ntiles, tile, H, embed, false);
        if (loc.off < 0) return fail("an untied exact tail was refused: " + loc.why);
        if (loc.off != base) return fail("an untied exact tail was located at the wrong offset");
        if (loc.verified) return fail("an untied tail reported itself as verified");
    }
    if (why) why->clear();
    return true;
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
    // Container layout version (npu/q4nx.h): q4nx-2 is the fp32 q4_1f parameter form this importer
    // writes; absent or q4nx-1 (every container written before engine#1185, and every GGUF-derived
    // one) is the bf16 q4_1 form. The string and the chunk bytes come from one constant each.
    j["layout"] = q4nx::layout_of(kContainerParams);
    j["parameters"] = "fp32 scale/zero per 32-column group, bit-equal to the pack's";
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
    // What the *graph* carries, in the graph's own terms. A pack's genre is the first question every sweep row asks,
    // and until now it could only be answered by an external tool: the rai_1.3/1.6/1.7.1 line ships its 4-bit weights
    // as ordinary ONNX initializers (pi-inferzo: 25/25 real qweights, UINT8 [1152,7,64], location model_jit.onnx.data)
    // while 1.8 and R1-Distill ship placeholders and put the weights in model.bin. Reported here so `npu-pack info`
    // answers it with no download beyond the metadata and no third-party parser.
    if (graph_matmuls >= 0) {
        json g;
        g["matmulnbits_ops"] = graph_matmuls;
        g["real_weight_initializers"] = graph_real_weights;
        g["external_tensors"] = graph_external;
        if (graph_blocks.size() == 1) g["block_size"] = graph_blocks.begin()->first;
        else if (!graph_blocks.empty()) g["block_size_per_op"] = graph_blocks;
        if (!graph_weight_dims.empty()) g["weight_dims_sample"] = graph_weight_dims;
        g["external_data_present"] = graph_external_bytes;  // bytes of the named .data file found on disk
        // Which quantised-matmul op the graph speaks, and whether the weights are behind placeholders. A pack whose
        // node inputs are `*.qweight.empty` and whose bytes are in a `*.qweight.packed` sibling is NOT weightless:
        // following the node inputs alone reports zero real weights for a 9.7 GB pack (measured: phi-4, engine#1503).
        g["ryzenai_matmuls_ops"] = graph_ryzenai_matmuls;
        g["packed_weight_siblings"] = graph_packed_siblings;
        // Matmuls dropped because neither their `*.empty` placeholder nor its `*.packed` sibling carried any bytes
        // (engine#1671). Reported so a test can pin the drop without needing a 1.9 pack, and so `info` says how many
        // graph weights this pack does not keep in its graph.
        g["weightless_matmuls"] = graph_weightless_matmuls;
        if (graph_stride_of > 0)
            g["external_row_stride"] = {{"matched", graph_stride_ok}, {"of", graph_stride_of},
                                        {"rule", "length == N * (K/128 + 1) * 68"}};
        j["graph_weights"] = g;
    }
    j["mlp_section"] = {{"gate_up_units_measured", mlp_gu_units},
                        {"down_groups_measured", mlp_down_groups},
                        {"note", mlp_note}};
    j["incomplete"] = refused();
    const TextArtifacts text = scan_text_artifacts(dir);
    j["text_artifacts"] = {{"present", text.present},
                           {"missing", text.missing},
                           // a container is servable only with tokenizer.json (docs/npu.md)
                           {"servable", text.tokenizer}};
    return j.dump(2);
}

std::string qk_norm_refusal(bool declared, bool found) {
    if (declared && !found)
        return "this pack declares use_qk_norm and its per-head q/k gains are not where the block layout says "
               "(engine#1889): importing without them would normalise no q/k, which answers plausibly-wrong text "
               "instead of refusing. The section location for this family is not measured yet";
    return std::string();
}

Plan plan(const std::string& pack_dir) {
    Plan p;
    p.dir = pack_dir;
    p.pack = fs::path(pack_dir).lexically_normal().filename().string();

    std::string geom_src;
    json geom = load_geometry(pack_dir, &geom_src);
    // `intermediate_size` is the one key no 1.6/1.7 pack states: genai_config.json has no such field, and on the
    // hybrid line the FFN projections are not in the main graph either (measured: 25 of 121 qweights present). So
    // it is taken from the graph when the graph carries an MLP projection, and otherwise refused with that reason
    // -- never guessed from the other dimensions.
    const std::string graph = graph_filename(pack_dir);
    auto im_from_graph = [&](const std::string& gname) -> int {
        std::error_code ec;
        if (!fs::exists(pack_dir + "/" + gname, ec)) return 0;
        const OnnxModel gm = parse_onnx(pack_dir + "/" + gname);
        int best = 0;
        for (const auto& t : gm.tensors) {
            const bool weight = t.name.find("qweight") != std::string::npos ||
                               t.name.find("MatMulNBits") != std::string::npos;
            if (!weight || t.dims.size() != 3) continue;          // [N, K/128, 64]
            const std::string& n = t.name;
            if (n.find("lm_head") != std::string::npos) continue;
            if (n.find("mlp") == std::string::npos && n.find("gate") == std::string::npos &&
                n.find("up_proj") == std::string::npos && n.find("down_proj") == std::string::npos)
                continue;
            const int64_t w = t.dims.at(0);                        // N of the projection
            if (n.find("gate_up") != std::string::npos) { if (w / 2 > best) best = int(w / 2); }
            else if (w > best) best = int(w);
        }
        // The external-data genre states its weight as a 1-D byte blob, so the shape is not in the tensor at all --
        // it is in the NODE's N/K attributes, which are the pack's own declaration (phi-4: gate_proj N 17920,
        // down_proj K 17920). Reading them is not guessing from other dimensions: it is reading what the graph says
        // about this very projection (engine#1503).
        if (best == 0) {
            for (const auto& mm : gm.matmuls) {
                if (mm.N <= 0 || mm.K <= 0) continue;
                const std::string& n = mm.weight;
                if (n.find("lm_head") != std::string::npos) continue;
                const bool mlp = n.find("mlp") != std::string::npos || n.find("gate") != std::string::npos ||
                                 n.find("up_proj") != std::string::npos || n.find("down_proj") != std::string::npos;
                if (!mlp) continue;
                int64_t w = 0;
                if (n.find("down_proj") != std::string::npos) w = mm.K;      // down: K is the FFN width
                else w = mm.N;                                                // gate/up: N is the FFN width
                if (n.find("gate_up") != std::string::npos) w /= 2;
                if (w > best) best = int(w);
            }
        }
        if (best == 0 && geom.contains("hidden_size") && geom["hidden_size"].is_number())
            best = intermediate_size_from_graph(pack_dir + "/" + gname, geom["hidden_size"].get<int64_t>());
        return best;
    };
    if (!geom.contains("intermediate_size") || geom["intermediate_size"].is_null()) {
        const int im = im_from_graph(graph);
        if (im > 0) {
            geom["intermediate_size"] = im;
            geom_src += ", intermediate_size from the graph's MLP projection width";
        } else {
            throw std::runtime_error(
                "the pack states no intermediate_size: " + geom_src + " has none, and neither the initializers nor "
                "the MatMulNBits/MatMulNBitsBf node attributes of " + graph + " (including its If branches) hold a "
                "down/gate_up pair whose K and N agree -- on the hybrid line only the attention/head qweights are in "
                "the main graph, and the 1.9 external-data genre keeps the FFN in the branch with dims [0] on every "
                "initializer -- refusing rather than guessing it from the other dimensions");
        }
    }
    p.dims = read_dims(geom, geom_src);
    const Dims& d = p.dims;
    // (graph census is filled below, once the graph is parsed)

    std::error_code ec_graph;
    if (!fs::exists(pack_dir + "/" + graph, ec_graph))
        throw std::runtime_error("the pack's graph file " + graph + " is not in " + pack_dir +
                                 " (named by genai_config.json's model.decoder.filename)");
    const OnnxModel onnx = parse_onnx(pack_dir + "/" + graph);
    // Census the graph's own weight carrier (see plan_import.h's note): this is what tells a 1.3/1.6/1.7.1 pack
    // (real ONNX initializers) from a 1.8/R1-Distill pack (placeholders, weights in model.bin) without parsing
    // anything outside the pack.
    {
        p.graph_matmuls = int64_t(onnx.matmuls.size());
        std::set<std::vector<int64_t>> dims_seen;
        std::string ext_file;
        for (const auto& mm : onnx.matmuls) {
            if (mm.block) p.graph_blocks[mm.block]++;
            if (mm.weight.empty()) continue;
            const auto wi = onnx.index.find(mm.weight);
            if (wi == onnx.index.end()) continue;
            const OnnxTensor& w = onnx.tensors[wi->second];
            if (!w.dims.empty()) {
                p.graph_real_weights++;
                if (dims_seen.size() < 4 && dims_seen.insert(w.dims).second)
                    p.graph_weight_dims.push_back(w.dims);
            }
            if (!w.location.empty() && ext_file.empty()) ext_file = w.location;
            // The external-data packs state one row per output row and one extra 128-K block at the end of each:
            // length == N x (K/128 + 1) x 68, where 64 B is the 4-bit codes and the 4 B trailer is the block's
            // scale pair. Verified on all 201 of phi-4's matmuls; counted here so `info` reports a measurement
            // rather than an assumption, and a pack that does not fit it says so with numbers.
            if (!w.location.empty() && mm.block == 128 && mm.bits == 4 && mm.K > 0 && mm.N > 0) {
                p.graph_stride_of++;
                if (int64_t(w.length) == mm.N * (mm.K / 128 + 1) * 68) p.graph_stride_ok++;
            }
        }
        p.graph_ryzenai_matmuls = onnx.ryzenai_matmuls;
        p.graph_packed_siblings = onnx.packed_siblings;
        p.graph_weightless_matmuls = onnx.weightless_matmuls;
        for (const auto& t : onnx.tensors)
            if (!t.location.empty()) p.graph_external++;
        if (!ext_file.empty()) {
            std::error_code ec;
            const auto sz = fs::file_size(pack_dir + "/" + ext_file, ec);
            if (!ec) p.graph_external_bytes = int64_t(sz);
        }
    }
    // The Token .fconst, found once (see find_token_fconst) and used by the attention block below. Empty means this
    // pack keeps no weights there -- which for the rai_1.3/1.6/1.7.1 line is the normal case, not a fault.
    std::string tok_why;
    std::string tok_path = materialise_token_fconst(pack_dir, p.dims);
    // engine#1671 / #1687: a pack whose graph names `*.qweight.empty` placeholders whose `*.packed` siblings carry no
    // data has no weight for those ops IN THE GRAPH. With a Token .fconst that is the normal shape of a 1.9 pack -- its
    // graph is all placeholders and the weights are in the .fconst -- so the walker drops them and the import proceeds.
    // With no .fconst there is nothing to fall back to, and the honest answer is #1687's refusal, by name.
    if (tok_path.empty() && onnx.weightless_matmuls > 0)
        throw std::runtime_error(
            pack_dir + ": " + onnx.weightless_example +
            ", and the pack carries no Token .fconst either (no dd_metastate_Llm_Token_*.fconst): these weights are "
            "genuinely absent from the pack's external data, so this pack cannot be imported -- not a tile-layout or "
            "block-size problem");
    const bool no_fconst = tok_path.empty();
    // The 1.9 external-data genre (phi-4: 201 `MatMulNBitsBf` ops whose bytes are `*.qweight.packed` siblings in
    // `eager-mm.onnx.data`) must NOT ride the graph-weights path below, which was written for the 1.3/1.6/1.7.1 line's
    // ordinary initializers (`UINT8 [N, K/128, 64]`). There the weight is 3-D and the tile is the initializer; here it
    // is a 1-D byte blob whose row stride is N x (K/128 + 1) x 68, and repacking it into this importer's per-role tile
    // form is a writer this tree does not have yet (engine#1503). Recognising the genre in the census and then letting
    // it fall through here would import a half-readable pack -- so it is refused by name, with the layout stated,
    // until the writer lands.
    const bool graph_weights = p.graph_real_weights > 0 && no_fconst;
    // The projections the graph carries as real initializers, by their own names (`.MatMulNBits.` marks one).
    std::set<std::string> graph_projs;
    if (graph_weights) {
        for (const auto& t : onnx.tensors) {
            if (t.dims.empty()) continue;                              // 0-dim placeholders sit beside them
            // Two spellings of the same thing: `<proj>.MatMulNBits.qweight` (the 1.8/phi-4 form) and
            // `<proj>.MatMul.weight_Q4` (Qwen3-VL-4B's, three-input nodes, engine#1503).
            size_t pos = t.name.find(".MatMulNBits.");
            if (pos != std::string::npos) { graph_projs.insert(t.name.substr(0, pos)); continue; }
            const std::string q4 = ".MatMul.weight_Q4";
            pos = t.name.find(q4);
            if (pos != std::string::npos) graph_projs.insert(t.name.substr(0, pos));
        }
    }
    const OnnxTensor* embed = nullptr;
    for (const auto& t : onnx.tensors)
        if (t.name == "model.embed_tokens.weight") embed = &t;
    if (!embed) {
        // A hybrid pack keeps the table in its own graph; if that component is there, this pack HAS an embedding and the
        // refusal below would be wrong about it.
        std::string emb_why;
        const std::vector<float> comp = read_embedding_component(pack_dir, &emb_why);
        if (comp.empty())
            throw std::runtime_error("the pack has no model.embed_tokens.weight, and " + emb_why);
        const int64_t Vc2 = d.vocab > 0 ? d.vocab : 0, Hc2 = d.hidden > 0 ? d.hidden : 0;
        if (d.vocab == 0 || d.hidden == 0)
            throw std::runtime_error("the pack's embedding comes from embedding.onnx but the geometry does not give V/H");
        if (comp.size() != size_t(Vc2) * size_t(Hc2))
            throw std::runtime_error("embedding.onnx carries " + std::to_string(comp.size()) + " values, not V*H = " +
                                     std::to_string(int64_t(Vc2) * Hc2));
        p.tensors.push_back(make("model.embed_tokens.weight", "embedding.onnx", "BF16", {Vc2, Hc2}));
    }
    if (embed) {
        if (d.vocab == 0) p.dims.vocab = int(embed->dims.at(0));
        p.tensors.push_back(make("model.embed_tokens.weight", embed->location, "BF16",
                                 {embed->dims.at(0), embed->dims.at(1)}));
    }

    // Only the node whose geometry IS the head (N = V, K = H) is the head. This loop used to take the FIRST MatMulNBits and
    // `break`, so on a pack whose first matmul is a projection the plan reported that projection's shape as
    // `lm_head.weight` -- [1152, 896], a fused qkv, instead of [151936, 896] on Qwen2.5-0.5B-Instruct 1.9 (engine#1671).
    // import_pack() already gates this way; the plan's copy of the loop did not.
    bool head_from_graph = false;
    for (const auto& mm : onnx.matmuls) {
        if (!(mm.N == d.vocab && mm.K == d.hidden)) continue;
        const auto it = onnx.index.find(mm.weight);
        if (it == onnx.index.end()) continue;
        auto x = make("lm_head.weight", onnx.tensors[it->second].location, "q4_1", {mm.N, mm.K});
        if (mm.bits != 4) x.refusal = "MatMulNBits is " + std::to_string(mm.bits) + "-bit, not 4-bit";
        p.tensors.push_back(std::move(x));
        head_from_graph = true;
        break;
    }
    if (!head_from_graph) {
        // The head is in the .fconst tail (the Ryzen AI 1.9 lines). The importer reads it, and refuses by name
        // there if the tail's arithmetic does not come out, so the plan does not refuse it here. Keyed on "did a graph
        // head get written", not "is the graph free of matmuls": a graph of projections only is not a pack without a head
        // (engine#1671).
        p.tensors.push_back(make("lm_head.weight", "fconst:tail", "q4_1", {d.vocab, d.hidden}));
    }

    // norms: the layer blocks are located by their epsilon anchors at import time, so the offsets work
    // for any hidden size (engine#734). plan() cannot prove that without reading the file, so the
    // refusal here is empty and import_pack reports what the anchors actually found.
    const std::string norm_source = "fconst:Llm_Token_Token";
    const std::string norm_refusal;
    const int64_t H = d.hidden;   // the geometry's hidden; the embedding, when the main graph has none, must match it
    // Genre A: the norms are ordinary F16 initializers named after themselves, with real offsets -- not the
    // .fconst tail envelope -- so the refusal names the initializer and where it sits.
    auto graph_norm_refusal = [&](const std::string& name) -> std::string {
        if (!graph_weights) return norm_refusal;
        const auto ti = onnx.index.find(name);
        if (ti == onnx.index.end()) return "the graph carries no initializer named " + name;
        const OnnxTensor& t = onnx.tensors[ti->second];
        if (t.dims.empty()) return "the graph carries " + name + " as a 0-dim placeholder, not a weight";
        return "weights are in " + (t.location.empty() ? std::string("the graph") : t.location) + " + " +
               std::to_string((unsigned long long)t.offset) + " B -- the graph-initializer reader is not "
               "implemented yet (this pack is not a .fconst pack)";
    };
    for (int L = 0; L < d.layers; ++L) {
        auto a = make("model.layers." + std::to_string(L) + ".input_layernorm.weight", norm_source, "BF16", {H});
        auto b = make("model.layers." + std::to_string(L) + ".post_attention_layernorm.weight", norm_source,
                      "BF16", {H});
        const std::string ia = "model.layers." + std::to_string(L) + ".input_layernorm.weight";
        const std::string ib = "model.layers." + std::to_string(L) + ".post_attention_layernorm.weight";
        if (graph_weights && onnx.index.count(ia) && onnx.index.count(ib) &&
            !onnx.tensors[onnx.index.at(ia)].dims.empty() && !onnx.tensors[onnx.index.at(ib)].dims.empty()) {
            // genre A: ordinary F16 initializers with real offsets -> readable now, no refusal
            a.source = onnx.tensors[onnx.index.at(ia)].location;
            b.source = onnx.tensors[onnx.index.at(ib)].location;
            a.refusal.clear();
            b.refusal.clear();
        } else {
            a.refusal = graph_norm_refusal(ia);
            b.refusal = graph_norm_refusal(ib);
        }
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
        const std::string& tok = tok_path;
        const int64_t tile_bytes = attn_tile_bytes(H);
        if (tok.empty()) {
            p.attn_note = tok_why;
            p.mlp_note = p.attn_note;
        } else if (tile_bytes == 0) {
            p.attn_note = "hidden " + std::to_string(H) + " is not a multiple of 128, so the attention "
                                                                  "payload is not 64-row tiles of this shape";
            p.mlp_note = p.attn_note;
        } else {
            const FconstGeom g = locate_fconst_with_header(tok, pack_dir, p.dims);
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
                    // A pack whose attention section is [q,k,v at K=hidden][per-head QK norms][o_proj at K=q_out]
                    // still has ONE header, so it lands here -- and the uniform test rejects it by a handful of
                    // bytes. Measured on Hunyuan-0.5B-Instruct (h1024, 16 heads x head_dim 128, so q_out = 2,048):
                    //   q,k,v           64 tiles x 39,168                    = 2,506,752
                    //   per-head norms  2 x 128 x 2 B + 128 B pad             =       640
                    //   o_proj          16 tiles x 73,984 (K = q_out)         = 1,183,744
                    //                                                          3,691,136 = the payload to the byte
                    // The gap is checked to be a norms-sized region rather than accepted loosely: if the remainder
                    // were tiles' worth of bytes it would be a layout this importer has not measured, and reading it
                    // would put weights in the wrong place.
                    // Same composite, but for a pack whose q_out EQUALS hidden: every section is at K=hidden, so the
                    // only trace of the norms is a sub-tile remainder. Measured class: Hy-MT2-1.8B (h2048, 16 heads x
                    // head_dim 128) has payload 5,919,360 = 80 tiles x 73,984 + 640, exactly the 64 q/k/v tiles, the
                    // 16 o_proj tiles and the same 640 B of per-head norms.
                    bool handled = false;
                    {
                        const int64_t tiles_all = payload / tile_bytes;
                        const int64_t rem = payload - tiles_all * tile_bytes;
                        if (payload > 0 && tiles_all == int64_t(n_kv * 2 + n_q + H / kTileRows) && rem > 0 &&
                            rem <= int64_t(2) * int64_t(d.head_dim) * 2 + 256) {
                            p.attn_tiles = n_kv * 2 + n_q;
                            p.o_off = attn_off + p.attn_tiles * tile_bytes + rem;
                            p.o_k = int(H);
                            static bool said2 = false;
                            if (!said2) {
                                said2 = true;
                                std::fprintf(stderr,
                                             "note: attention section is %lld tiles at K=%lld plus %lld B of per-head "
                                             "QK norms before o_proj\n", (long long)p.attn_tiles, (long long)H,
                                             (long long)rem);
                            }
                            handled = true;
                        }
                    }
                    const int64_t o_tiles_h = H / kTileRows;
                    const int64_t qkv_bytes_c = int64_t(n_kv * 2 + n_q) * tile_bytes;
                    const int64_t o_bytes_c = o_tiles_h * tile_o;
                    const int64_t gap_c = payload - qkv_bytes_c - o_bytes_c;
                    const int64_t norm_ceiling = int64_t(2) * int64_t(d.head_dim) * 2 + 256;
                    if (!handled && payload > 0 && q_out != H && o_tiles_h > 0 && gap_c >= 0 && gap_c <= norm_ceiling) {
                        handled = true;
                        p.attn_tiles = n_kv * 2 + n_q;
                        p.o_off = attn_off + qkv_bytes_c + gap_c;
                        p.o_k = int(q_out);
                        // NOT p.attn_note: that field is the refusal channel -- a non-empty value there refuses the
                        // attention biases by name (it means "why this section could not be read"). This is the
                        // opposite: the section WAS read this way. Reported on stderr once instead.
                        static bool said = false;
                        if (!said) {
                            said = true;
                            std::fprintf(stderr,
                                         "note: attention section is q/k/v at K=%lld (%lld tiles), %lld B of per-head "
                                         "QK norms, then o_proj at K=%lld (%lld tiles)\n",
                                         (long long)H, (long long)p.attn_tiles, (long long)gap_c,
                                         (long long)q_out, (long long)o_tiles_h);
                        }
                    } else if (handled) { /* composite section above already assigned it */ }
                    else if (payload <= 0 || payload % tile_bytes)
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
    // Block size gate (engine#1410 §3 of FACTS-fconst-layout): the compiler states `block_size` per MatMulNBits, and
    // this importer's tile geometry is the block-128 one -- a group is 2*(K/block)+2 subvolumes, so a block-32 pack
    // has every offset in a different place. Refusing by number beats parsing it at the 1.8 geometry, which is what
    // the 1.9 8B case showed (a confident parse of wrong offsets). Bits is checked the same way: every pack measured
    // is 4-bit, and a wider code changes the byte packing.
    {
        std::map<int64_t, int> blocks, bits;
        for (const auto& [proj, dec] : declared) {
            if (dec.block) blocks[dec.block]++;
            if (dec.bits) bits[dec.bits]++;
        }
        // Where the non-128 blocks are decides whether the pack is readable at all. The .fconst tile geometry is the
        // block-128 one, so a LAYER with another block makes the payload unwalkable and the pack is refused. The LM
        // HEAD is the one exception that can be survived: it lives in the tail, after every layer, so the layers can
        // still be read -- and on a tied pack the head need not be read at all, because it IS the embedding
        // (engine#1503: the three Qwen-1.5B 1.9 packs are 140 ops at 128 and exactly one at 32, and that one is
        // `/lm_head/MatMul_Q4`). Counting the census per projection is what makes that distinction possible.
        std::vector<std::string> off_block;
        for (const auto& [proj, dec] : declared)
            if (dec.block && dec.block != kKBlock) off_block.push_back(proj + " @ " + std::to_string(dec.block));
        bool only_the_head = !off_block.empty();
        for (const auto& o : off_block) {
            const auto it = declared.find(o.substr(0, o.find(" @ ")));
            const bool head = it != declared.end() && it->first.rfind("lm_head", 0) == 0;
            if (!head) only_the_head = false;
        }
        // The gate applies only where the weights really come from the .fconst (engine#1655): a pack with none keeps
        // them as the graph's own initializers, where the block is a property of the quantised table and
        // decode_matmulnbits() already parameterises it. Measured on amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu: 253
        // MatMulNBits ops, all block_size 32, no fconst at all -- refused by this gate for a reason ("every subvolume
        // offset in the .fconst moves") that cannot apply to a pack with no .fconst. engine#1660 renamed the
        // materialising function to `materialise_token_fconst` and updated every call site except this one, which left
        // `main` not compiling; the right replacement here is NOT the materialising one: this is a predicate ("does the
        // pack carry its weights in a .fconst?"), and the predicate is `find_token_fconst()`, which touches nothing --
        // calling the writer here cost a second 278 MiB rewrite per command.
        const bool fconst_weights = !find_token_fconst(pack_dir).empty();
        if (fconst_weights && !blocks.empty() && !(blocks.size() == 1 && blocks.begin()->first == kKBlock)) {
            if (only_the_head) {
                p.head_block = declared.at(off_block[0].substr(0, off_block[0].find(" @ "))).block;
            } else {
                // Name a few and count the rest: SmolLM2-135M has 151 off-block ops, and listing every one of them made
                // the refusal a wall of text that hid the sentence explaining it.
                std::string list;
                for (size_t i = 0; i < off_block.size() && i < 4; ++i) list += (list.empty() ? "" : ", ") + off_block[i];
                if (off_block.size() > 4) list += " and " + std::to_string(off_block.size() - 4) + " more";
                throw std::runtime_error(
                    "this pack's MatMulNBits block size along K is " + [&] {
                        std::string t;
                        for (const auto& [b, n] : blocks)
                            t += (t.empty() ? "" : ", ") + std::to_string(b) + " (" + std::to_string(n) + " ops)";
                        return t;
                    }() + ", and this importer decodes the block-" + std::to_string(kKBlock) +
                    " tile layout only: at a different block every subvolume offset in the .fconst moves, so the "
                    "weights would be read from the wrong place, not merely mis-scaled. The off-block tensor(s) are " +
                    list + "; only the LM head could be tolerated, because the layers come before it in the tail.");
            }
        }
        if (!bits.empty() && !(bits.size() == 1 && bits.begin()->first == 4))
            throw std::runtime_error("this pack's MatMulNBits bits is not 4 (declared: " +
                                     std::to_string(bits.begin()->first) +
                                     "), and the 4-bit two-codes-per-byte packing is what this importer decodes");
    }
    // Genre A: those projections are not in a .fconst at all -- they are the graph's own initializers, so the
    // refusal says that and names the one it found rather than repeating a .fconst message that is untrue for
    // this pack. The reader that consumes them is the next item; nothing here claims they are importable.
    auto graph_weight_refusal = [&](const std::string& proj) -> std::string {
        if (!graph_weights) return std::string();
        const auto it = graph_projs.find(proj);
        if (it == graph_projs.end())
            return "the graph carries no " + proj + ".MatMulNBits.qweight";
        const std::string qw = proj + ".MatMulNBits.qweight";
        const auto ti = onnx.index.find(qw);
        std::string where = "the graph's own initializer";
        if (ti != onnx.index.end()) {
            const OnnxTensor& t = onnx.tensors[ti->second];
            if (!t.location.empty())
                where = t.location + " + " + std::to_string((unsigned long long)t.offset) + " B";
            if (t.dims.size() == 3) {   // [N, K/block, block/2 bytes]: two four-bit codes per byte
                const int64_t blk = t.dims[2] * 2;
                where += " [N=" + std::to_string(t.dims[0]) + " K=" + std::to_string(t.dims[1] * blk) + " block=" +
                         std::to_string(blk) + "]";
            }
        }
        return "weights are in " + where + " -- the graph-initializer reader is not implemented yet (this pack is "
               "not a .fconst pack)";
    };
    for (int L = 0; L < d.layers; ++L)
        for (const auto& pr : projs) {
            const std::string name = "model.layers." + std::to_string(L) + "." + pr.suffix + ".weight";
            auto x = make(name, "fconst:Llm_Token_Token", "q4_1", {pr.rows, pr.cols});
            const bool attention = std::string(pr.suffix).compare(0, 9, "self_attn") == 0;
            // The pack names the attention projections `layers.N.attn.*` (not `self_attn`), o_proj carries its own
            // initializer, and q/k/v have none because they are the FUSED qkv_proj MatMulNBits -- so a q/k/v
            // refusal has to name that node (and the split by head geometry that is still to come) rather than
            // hunt for an initializer that was never written.
            std::string ga_proj;
            if (attention) {
                const std::string leaf = std::string(pr.suffix).substr(10);   // q_proj / k_proj / v_proj / o_proj
                // The fused node is what most packs carry, but not all: 11 genre-A packs (Qwen3 x7, gemma-2 x3,
                // Qwen2.5-14B) ship q/k/v as three separate MatMulNBits joined by Concat, so the split spelling is
                // tried when the fused one is not in the graph. The container stores q/k/v separately either way.
                const std::string fused = "model.layers." + std::to_string(L) + ".attn.qkv_proj";
                const std::string split = "model.layers." + std::to_string(L) + ".attn." + leaf;
                ga_proj = (leaf == "o_proj") ? split
                                             : (graph_projs.count(fused) ? fused : split);
            } else {
                ga_proj = "model.layers." + std::to_string(L) + "." + std::string(pr.suffix);
            }
            const std::string ga = graph_weight_refusal(ga_proj);
            if (graph_weights) {
                // the import reads these from the graph's initializers; the refusal only stands when the
                // initializer is genuinely not there
                if (graph_projs.count(ga_proj)) {
                    x.source = "graph:" + ga_proj;
                    x.refusal.clear();
                } else {
                    x.refusal = ga;
                }
            }
            else if (!attention) x.refusal = p.mlp_note;
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
            // genre A carries the qkv bias as ONE fused [q|k|v] F16 initializer beside the fused qweight, so it is
            // readable (and split by the same row ranges as the weight) rather than a tile subvolume.
            const std::string fused_bias =
                "model.layers." + std::to_string(L) + ".attn.qkv_proj.MatMulNBits.bias";
            // A split pack carries one bias per projection instead of the fused [q|k|v] one.
            const std::string own_bias =
                "model.layers." + std::to_string(L) + ".attn." + suffix + ".MatMulNBits.bias";
            const std::string& bias_name = onnx.index.count(fused_bias) ? fused_bias : own_bias;
            if (graph_weights && onnx.index.count(bias_name) &&
                !onnx.tensors[onnx.index.at(bias_name)].dims.empty()) {
                b.source = "graph:" + bias_name;
                b.refusal.clear();
            } else if (!p.attn_note.empty()) b.refusal = p.attn_note;
            b.declared_check = "written only when the tile's bias subvolumes are not all zero";
            p.tensors.push_back(std::move(b));
        }
    // per-head QK RMSNorm gains (Qwen3 and later): bf16 [head_dim], in the two sections that sit between
    // q/k/v and o_proj. Planned only when the block measured those two sections (engine#979).
    // Genre A states them as `model.layers.N.attn.q_norm.layernorm.weight` (F16 [head_dim]), which is a different
    // source from the .fconst's two per-head sections but the same tensor in the container.
    const bool graph_qk_norm = graph_weights && !onnx.index.empty() &&
                               onnx.index.count("model.layers.0.attn.q_norm.layernorm.weight");
    /* engine#1889: a pack that DECLARES per-head q/k gains and whose gains this importer did not locate must not
     * import -- the forward would run without normalising q/k, which answers deterministic gibberish instead of
     * refusing (Hunyuan-7B: `Trease/ft//////////` three of three, at a geometry four other packs answer correctly on).
     * The section location for that family is not measured yet, so the honest outcome is a named refusal; the
     * layout arithmetic and the graph check stay the two ways the gains ARE found. */
    if (const std::string qk_why = qk_norm_refusal(d.use_qk_norm, p.knorm_off != 0 || graph_qk_norm); !qk_why.empty())
        throw std::runtime_error(pack_dir + ": " + qk_why);
    if (p.knorm_off || graph_qk_norm)
        for (int L = 0; L < d.layers; ++L)
            for (const char* nm : {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}) {
                auto x = make("model.layers." + std::to_string(L) + "." + nm, norm_source, "BF16", {d.head_dim});
                if (graph_qk_norm) {
                    // Derive the leaf from the name rather than by index arithmetic: "self_attn." is TEN characters,
                    // and substr(11, 1) here produced '_', so the source named `…attn._norm.layernorm.weight`, which
                    // exists in no pack. Nothing failed, because the import reads the norms by its own correct name --
                    // the manifest simply recorded a source that is not there, which is the "compiles, never runs"
                    // shape and exactly the kind of label a later reader would trust. A substring test cannot be off
                    // by one.
                    const bool is_q = std::string(nm).find("q_norm") != std::string::npos;
                    x.source = "graph:model.layers." + std::to_string(L) + ".attn." +
                               (is_q ? "q_norm" : "k_norm") + ".layernorm.weight";
                    x.refusal.clear();
                }
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

// `embed` is BF16 rows: the importer converts the pack's table once (`embed_bf16`) and every consumer reads that
// buffer. It used to be the pack's own F16 bytes, and this decoded them as F16 -- which engine#1661 changed underneath
// without changing this, so from then on the self-check compared the head against mis-decoded values: Qwen2.5-0.5B 1.9
// reported corr 0.885220 (and "untied", for a pack declaring tie_word_embeddings) where the same container's weights
// score 0.999999. The container was always right; the number about it was not.
static double corr(const Q4& q, const EmbedTable& embed, int rows) {
    double sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
    const int n = std::min(rows, q.N);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < q.K; ++k) {
            const size_t g = size_t(r) * (q.K / 32) + size_t(k) / 32;
            const double x = double(q.codes[size_t(r) * q.K + k]) * q.scale[g] + q.zero[g];
            const double y = embed_at(embed, r, k);
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

// The pack's text artifacts, in one place: the importer copies exactly these, and `info` reports
// which are there before anything is written.
static const char* const kTextFiles[] = {"tokenizer.json",
                                         "tokenizer_config.json",
                                         "vocab.json",
                                         "merges.txt",
                                         "added_tokens.json",
                                         "special_tokens_map.json",
                                         "chat_template.jinja",
                                         "generation_config.json"};

// The same list every path copies (see kTextFiles): a container without tokenizer.json is not an NPU model directory.
// engine#1751 follow-up: the ONE copier. Both import paths call it (the .fconst path used to keep its own loop, which
// threw a bare filesystem_error, while this one counted successes and swallowed failures), and a file the pack HAS that
// fails to copy is refused by name -- a counter nobody compared let an import "succeed" without tokenizer.json, and serve
// then refused the directory for a missing tokenizer, the exact misleading report #1751 existed to remove. A file the pack
// does not ship stays a skip: a pack without a chat template is legitimate.
static int copy_text_artifacts(const std::string& pack_dir, const std::string& out_dir) {
    int copied = 0;
    for (const char* f : kTextFiles) {
        const std::string src = pack_dir + "/" + f;
        std::error_code cec;
        if (!fs::exists(src, cec)) continue;
        fs::copy_file(src, out_dir + "/" + f, fs::copy_options::overwrite_existing, cec);
        if (cec)
            throw std::runtime_error("cannot copy the pack's " + std::string(f) + " into " + out_dir + " (" + cec.message() +
                                     "); without it the output is not a servable model directory");
        ++copied;
    }
    return copied;
}
int copy_text_artifacts_exported(const std::string& pack_dir, const std::string& out_dir) {
    return copy_text_artifacts(pack_dir, out_dir);
}

TextArtifacts scan_text_artifacts(const std::string& pack_dir) {
    TextArtifacts t;
    for (const char* f : kTextFiles) {
        if (fs::exists(pack_dir + "/" + f)) t.present.push_back(f);
        else t.missing.push_back(f);
    }
    t.tokenizer = fs::exists(pack_dir + "/tokenizer.json");
    return t;
}

// Why a layout name cannot be declared by `--weight-layout`, or "" when it can.
std::string weight_layout_refusal(const std::string& want) {
    if (want.empty() || want == "rm" || want == "b8") return "";
    if (want == "t2s")
        return "t2s also needs q4nx_weight_layout_version (5 or 6) and a directory of per-layer "
               "t2s images; import cannot invent either";
    if (want == "pb16i")
        return "pb16i also needs version 4 and a directory of per-layer pb16i images; import cannot "
               "invent either";
    return "unknown weight layout \"" + want + "\" (the containers the forward reads declare "
           "\"b8\" or nothing at all, which is the row-major default)";
}

std::string kernel_set_refusal(const std::vector<std::string>& file_names, const std::string& layout) {
    bool any_elf = false, any_b8 = false, any_plain = false;
    for (const std::string& n : file_names) {
        if (n.rfind("full_i8_", 0) != 0 || n.size() < 9 || n.compare(n.size() - 4, 4, ".elf") != 0)
            continue;
        any_elf = true;
        if (n.find("_b8") != std::string::npos) any_b8 = true;
        else any_plain = true;
    }
    if (!any_elf)
        return "no full_i8_*.elf in that directory: a kernel set is the forward's per-shape ELFs, named "
               "full_i8_<role>_K<k>_N<n>[_b8][_m16].elf, and nothing was found to link in";
    if (layout == "b8" && !any_b8)
        return "the container declares q4nx_weight_layout \"b8\" but this set has no _b8 ELF, so the "
               "forward would look for names that are not there";
    if (layout != "b8" && !any_plain)
        return "this set is the _b8 (blocked) family but the container declares no weight layout, so it "
               "needs `--weight-layout b8` to say so";
    return "";
}

// The 1.9 external-data genre's importer (engine#1503): the pack stores each projection as one 4-bit blob and the
// container stores per-role tensors, so this walks the graph's matmuls, decodes each blob, splits the fused qkv by
// the geometry the pack states, and writes the same I8 tile chunks the .fconst route writes.
//
// Deliberately its own function rather than a fourth branch inside import_pack: that one is built around the
// .fconst envelope (attention payload, MLP sections, tail) and the graph-weights path, and threading a third shape
// through it is how the two existing paths got their special cases. What it shares is the parts that are genuinely
// shared -- the plan, the geometry, the tile writer, the container metadata.
// Copies the tokenizer/chat files the container needs beside model.q4nx; defined past kTextFiles' declaration.
static int copy_text_artifacts(const std::string& pack_dir, const std::string& out_dir);

Report import_external(const std::string& pack_dir, const std::string& out_dir, const Options& opt) {
    const Plan p = plan(pack_dir);
    const std::string graph = graph_filename(pack_dir);
    const OnnxModel onnx = parse_onnx(pack_dir + "/" + graph);
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) throw std::runtime_error("cannot create " + out_dir + ": " + ec.message());
    Writer w;
    w.dir = out_dir;
    Report r;
    r.out_dir = out_dir;
    // The genre's bytes live in the external file, so map it once and slice per tensor (the same check import_pack
    // makes: `need` must fit what the tensor declares, because a 4-bit blob deliberately does not satisfy
    // product(dims) == length).
    std::map<std::string, std::pair<const uint8_t*, size_t>> ext;
    auto bytes_of = [&](const OnnxTensor& t, uint64_t need) -> const uint8_t* {
        if (t.raw) {
            if (need > t.raw_len)
                throw std::runtime_error(t.name + ": needs " + std::to_string(need) + " B, carries " +
                                         std::to_string(t.raw_len));
            return t.raw;
        }
        auto it = ext.find(t.location);
        if (it == ext.end()) {
            size_t size = 0;
            std::string why;
            const uint8_t* m = map_file(pack_dir + "/" + t.location, size, &why);
            if (!m) throw std::runtime_error(t.name + ": " + why);
            it = ext.emplace(t.location, std::make_pair(m, size)).first;
        }
        if (need > t.length) throw std::runtime_error(t.name + ": needs " + std::to_string(need) + " B, declares " +
                                                      std::to_string(t.length) + " in " + t.location);
        if (uint64_t(t.offset) + need > it->second.second)
            throw std::runtime_error(t.name + ": external data past the end of " + t.location);
        return it->second.first + t.offset;
    };
    const int64_t L = p.dims.layers;
    const int GH = int(p.dims.heads), GKV = int(p.dims.kv_heads), GHD = int(p.dims.head_dim);
    for (const auto& mm : onnx.matmuls) {
        const char* role = external_role_of(mm.weight);
        if (!role) {
            r.refused.push_back(mm.weight + ": no container role for this projection (the graph names something this "
                                             "importer has not measured)");
            continue;
        }
        const auto wi = onnx.index.find(mm.weight);
        if (wi == onnx.index.end() || mm.N <= 0 || mm.K <= 0) {
            r.refused.push_back(mm.weight + ": the graph names it but carries no bytes for it");
            continue;
        }
        const OnnxTensor& wt = onnx.tensors[wi->second];
        uint64_t need = 0;
        if (!mul_ok(uint64_t(wt.length ? wt.length : wt.raw_len), 1, need) || need == 0)
            throw std::runtime_error(mm.weight + ": no external length recorded");
        std::string derr;
        const Q4 pack = decode_external_tile(bytes_of(wt, need), size_t(need), int(mm.N), int(mm.K), derr);
        if (!derr.empty()) throw std::runtime_error(mm.weight + ": " + derr);
        const Q4 conv = container_from_external_q4(pack);
        std::vector<ContainerTensor> outs;
        std::string nerr;
        if (!external_container_tensors(role, GH, GKV, GHD, outs, nerr)) {
            r.refused.push_back(mm.weight + ": " + nerr);
            continue;
        }
        // `model.layers.N.` from the graph's own name, so the container is keyed the way the forward looks it up.
        std::string prefix;
        if (const size_t at = mm.weight.find("layers."); at != std::string::npos) {
            const size_t end = mm.weight.find('.', at + 7);
            prefix = "model.layers." + mm.weight.substr(at + 7, end - at - 7) + ".";
        }
        for (const auto& one : outs) {
            const int lo = one.row_lo, hi = one.row_hi > 0 ? one.row_hi : conv.N;
            if (lo < 0 || hi > conv.N || hi <= lo)
                throw std::runtime_error(mm.weight + ": container range [" + std::to_string(lo) + ", " +
                                         std::to_string(hi) + ") does not fit N " + std::to_string(conv.N));
            Q4 part;
            part.N = hi - lo;
            part.K = conv.K;
            part.block = 32;
            const int kg = conv.K / 32;
            part.codes.assign(size_t(part.N) * part.K, 0);
            part.scale.assign(size_t(part.N) * kg, 0.0f);
            part.zero.assign(size_t(part.N) * kg, 0.0f);
            for (int row = 0; row < part.N; ++row) {
                std::memcpy(&part.codes[size_t(row) * part.K], &conv.codes[size_t(lo + row) * part.K], part.K);
                std::memcpy(&part.scale[size_t(row) * kg], &conv.scale[size_t(lo + row) * kg], kg * sizeof(float));
                std::memcpy(&part.zero[size_t(row) * kg], &conv.zero[size_t(lo + row) * kg], kg * sizeof(float));
            }
            std::vector<uint8_t> tiled;
            long long pinexact = 0;
            tile_q4_1_f32(part, tiled, &pinexact);
            const int64_t nt = int64_t(tiled.size() / kImportTileBytes);
            r.tiles += nt;
            r.tensors++;
            w.add(prefix + one.suffix, "I8", {nt, kImportTileBytes}, std::move(tiled));
        }
    }
    if (!r.refused.empty()) {
        if (!opt.allow_incomplete) throw std::runtime_error("external-data import refused: " + r.refused.front());
        return r;
    }
    // The exact tensors the container also needs: the embedding and the norms, copied as BF16 (the container's form).
    int64_t aux = 0;
    for (const auto& t : onnx.tensors) {
        const bool norm = t.name.find("layernorm.weight") != std::string::npos ||
                          t.name == "model.norm.weight";
        if (t.name != "model.embed_tokens.weight" && !norm) continue;
        if (t.dims.empty()) continue;
        uint64_t n = 1;
        for (auto d : t.dims) n *= uint64_t(d);
        if (n == 0) continue;
        std::vector<int64_t> shape(t.dims.begin(), t.dims.end());
        w.add(t.name, "BF16", shape, as_bf16(bytes_of(t, n * (t.type == kF32 ? 4 : 2)), t.type, size_t(n)));
        ++r.tensors;
        ++aux;
    }
    // The device weight layout, when the caller declared one (`--weight-layout b8`): without this key the container
    // asks the forward for the UNTAGGED family while the caller built `_b8` ELFs, and the failure reads as "missing
    // full ELF" for a set that exists. Found by importing phi-4 with the flag, seeing the report print "declares
    // q4nx_weight_layout b8", and then finding the key absent from config.json (engine#1503).
    if (const std::string why = weight_layout_refusal(opt.weight_layout); !why.empty())
        throw std::runtime_error("--weight-layout: " + why);
    json c;
    // engine#1794 review: the container's OWN config is what the forward reads (`q4nx_semantics.c:121` value_for_top,
    // depth 1 only), and this writer started from an empty json -- so a mRoPE declaration that lives only at
    // `text_config.rope_parameters` was dropped, and the pack would be served as plain RoPE with the right theta and the
    // wrong axis layout (engine#1311: exact for a text-only prompt, ~0.29 rad off once image tokens are in it). The
    // import_pack() writer starts from load_geometry() and already carries it; this one has to copy it.
    {
        std::string geom_src;
        const json geom = load_geometry(pack_dir, &geom_src);
        if (geom.value("mrope_interleaved", false)) c["mrope_interleaved"] = true;
        if (geom.contains("mrope_section")) c["mrope_section"] = geom["mrope_section"];
    }
    c["model_type"] = p.dims.model_type;
    c["hidden_size"] = p.dims.hidden;
    c["intermediate_size"] = p.dims.intermediate;
    c["num_hidden_layers"] = p.dims.layers;
    c["num_attention_heads"] = p.dims.heads;
    c["num_key_value_heads"] = p.dims.kv_heads;
    c["head_dim"] = p.dims.head_dim;
    c["vocab_size"] = p.dims.vocab;
    c["rope_theta"] = p.dims.rope_theta;
    c["q4nx_parameters"] = "fp32 scale/zero per 32-column group";
    if (!opt.weight_layout.empty() && opt.weight_layout != "rm") c["q4nx_weight_layout"] = opt.weight_layout;
    c["q4nx_source"] = {{"kind", "oga-npu-pack external data"}, {"pack", p.pack},
                        {"importer", "1bit npu-pack import"}, {"graph", graph},
                        {"note", "weights decoded from *.qweight.packed via decode_external_tile; the code ordering is "
                                 "kExtQuantsRowMajor's compile-time choice (engine#1503)"},
                        {"weight_layout", opt.weight_layout.empty() ? "rm" : opt.weight_layout}};
    json manifest;
    manifest["format"] = "q4nx";
    manifest["layout"] = "q4nx-2";
    manifest["dims"] = {{"hidden_size", p.dims.hidden}, {"num_hidden_layers", p.dims.layers}};
    w.finish(manifest);
    std::ofstream cf(out_dir + "/config.json", std::ios::binary | std::ios::trunc);
    if (!cf) throw std::runtime_error("cannot write " + out_dir + "/config.json");
    cf << c.dump(2) << "\n";
    cf.close();
    // The text artifacts, the same list the other paths copy: without tokenizer.json the directory is not an NPU model
    // directory at all, and the first probe of this genre failed exactly there -- the container was complete and
    // `serve` refused it for the missing tokenizer, which reads like a weights problem and is not one.
    const int copied = copy_text_artifacts(pack_dir, out_dir);
    std::fprintf(stderr, "npu-pack: external-data genre: %d text artifact(s) copied\n", copied);
    std::fprintf(stderr, "npu-pack: external-data genre: %d tensors (%lld tiles, %lld exact)\n", r.tensors,
                 (long long)r.tiles, (long long)aux);
    return r;
}

Report import_pack(const std::string& pack_dir, const std::string& out_dir, const Options& opt) {
    const Plan p = plan(pack_dir);
    // Census yes, writer no (see the note where graph_weights is decided): this genre's weights are 1-D external-data
    // blocks, and until the repacker exists an import would write a container nothing can read. Refused here rather
    // than in plan(), because `info` must keep reporting the genre it can now see.
    if (p.graph_packed_siblings > 0) {
        // The genre has a writer now (engine#1503): decode -> convert -> split by role -> the same I8 tile chunks,
        // with its own function because import_pack is built around the .fconst envelope.
        return import_external(pack_dir, out_dir, opt);
    }
    if (false)
        throw std::runtime_error(
            "this pack keeps its 4-bit weights as external-data blocks (" + std::to_string(p.graph_packed_siblings) +
            " `*.qweight.packed` tensors behind `*.qweight.empty` placeholders, " + std::to_string(p.graph_stride_ok) +
            " of " + std::to_string(p.graph_stride_of) +
            " fitting length == N x (K/128 + 1) x 68) and this importer has no writer for that layout yet; the .fconst "
            "tile decoder does not apply (engine#1503)");
    Report r;
    r.out_dir = out_dir;
    r.refused = p.refused();
    if (!r.refused.empty() && !opt.allow_incomplete) return r;  // refuse by name, write nothing

    bool tied = false;
    bool tie_declared = false;   // whether the pack states it at all (a genre-A pack ships no config.json)
    {
        std::ifstream cf(pack_dir + "/config.json");
        if (cf) {
            const auto c = json::parse(cf);
            tie_declared = c.contains("tie_word_embeddings");
            tied = opt_field<bool>(c, "tie_word_embeddings", false);
        }
    }
    bool head_written = false;
    // The graph's own name, the same resolution plan() uses: on the 1.9/hybrid lines it is `model_jit.onnx` and
    // `genai_config.json` says so. Opening "model.onnx" here made an import of exactly the packs plan() had just
    // learned to read fail at the first byte, with "cannot open <pack>/model.onnx".
    const bool tok_present = !find_token_fconst(pack_dir).empty();
    const OnnxModel onnx = parse_onnx(pack_dir + "/" + graph_filename(pack_dir));
    // A graph whose quantised matmuls name a `*.qweight.empty` placeholder *and* whose packed siblings carry no bytes
    // has no weights in the graph at all. On this line that means model.bin: 1.8/R1-Distill ship their 4-bit weights
    // there and keep `[0]`-dims placeholders in the graph, and this importer has no model.bin reader. Refusing here
    // names the file and the count; without it the walk reaches bytes_of()'s `ext.find("")` and reports
    // "<tensor>.qweight.empty: no data", which says the bytes are missing rather than that they are somewhere this
    // importer does not read (engine#1671). Narrow on purpose: it can only fire when NO matmul resolved to bytes, so a
    // pack that imports today cannot start failing because of it.
    {
        size_t unresolved = 0;
        for (const auto& mm : onnx.matmuls)
            if (!mm.weight.empty() && mm.weight.find(".empty") != std::string::npos) ++unresolved;
        std::error_code bin_ec;
        const auto bin_size = fs::file_size(fs::path(pack_dir) / "model.bin", bin_ec);
        if (unresolved > 0 && unresolved == onnx.matmuls.size() && !bin_ec)
            throw std::runtime_error(
                std::to_string(unresolved) + " MatMulNBits weight(s) in this graph are `*.qweight.empty` placeholders and "
                "none of their `*.qweight.packed` siblings carries bytes; the pack carries model.bin (" +
                std::to_string(static_cast<unsigned long long>(bin_size)) + " B), which is the 1.8/hybrid genre: the "
                "4-bit weights live there and this importer has no model.bin reader, it reads ONNX initializers only. "
                "Refusing rather than reading zero-length placeholders as weights (engine#1671)");
    }
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
    if (embed && embed->dims.size() != 2) throw std::runtime_error("model.embed_tokens.weight is not 2-D");
    // V/H are read further down for the tied-head and vocab checks, so they exist whether the table came from the main
    // graph or from the embedding component (where the geometry states them instead).
    const int64_t V = embed ? embed->dims.at(0) : int64_t(p.dims.vocab);
    const int64_t H = embed ? embed->dims.at(1) : int64_t(p.dims.hidden);
    // The container takes BF16, and the tied-head self-check reads these same bytes, so convert once into a buffer that
    // outlives both. For the component path the source is f32 that the reader already dequantised.
    std::vector<uint8_t> embed_bf16;
    EmbedTable embed_tbl;   // engine#1721: the table and its width travel together
    {
        const uint64_t Vc = dim_count("model.embed_tokens.weight", V), Hc = dim_count("model.embed_tokens.weight", H);
        uint64_t elems = 0;
        if (!mul_ok(Vc, Hc, elems)) throw std::runtime_error("model.embed_tokens.weight: V*H overflows");
        if (embed) {
            const uint64_t src_width = embed->type == kF32 ? 4 : 2;
            uint64_t embed_need = 0;
            if (!mul_ok(elems, src_width, embed_need))
                throw std::runtime_error("model.embed_tokens.weight: V*H*width overflows");
            embed_bf16 = as_bf16(bytes_of(*embed, embed_need), embed->type, size_t(elems));
        } else {
            std::string emb_why;
            const std::vector<float> comp = read_embedding_component(pack_dir, &emb_why);
            if (comp.empty()) throw std::runtime_error("no model.embed_tokens.weight, and " + emb_why);
            if (comp.size() != size_t(elems))
                throw std::runtime_error("embedding.onnx carries " + std::to_string(comp.size()) + " values, not V*H = " +
                                         std::to_string(elems));
            embed_bf16 = as_bf16(reinterpret_cast<const uint8_t*>(comp.data()), kF32, size_t(elems));
        }
        embed_tbl = EmbedTable{embed_bf16.data(), EmbedTable::Fmt::BF16, int(H)};
        w.add("model.embed_tokens.weight", "BF16", {V, H}, embed_bf16);
        ++r.tensors;
    }

    // genre A = the graph carries real weights and there is no .fconst to read (the same test plan() uses)
    auto has_real_graph_weight = [&]() {
        for (const auto& mm : onnx.matmuls) {
            const auto wi = onnx.index.find(mm.weight);
            if (wi != onnx.index.end() && !onnx.tensors[wi->second].dims.empty()) return true;
        }
        return false;
    };
    const bool graph_weights_a = !tok_present && has_real_graph_weight();
    // genre A's norms: ordinary F16 [hidden] initializers named after themselves, plus the final norm. The 1.8
    // route reads them from the .fconst tail envelope; this pack has no tail, so they come from the graph.
    if (graph_weights_a) {
        auto add_norm_init = [&](const std::string& name, const std::string& out_name) {
            const auto it = onnx.index.find(name);
            if (it == onnx.index.end()) return false;
            const OnnxTensor& t = onnx.tensors[it->second];
            if (t.dims.empty()) return false;
            const size_t n = size_t(t.dims.at(0));
            const uint64_t need = uint64_t(n) * (t.type == kF16 || t.type == kBF16 ? 2u : 4u);
            w.add(out_name, "BF16", {int64_t(n)}, as_bf16(bytes_of(t, need), t.type, n));
            ++r.tensors;
            return true;
        };
        for (int L = 0; L < p.dims.layers; ++L) {
            const std::string base = "model.layers." + std::to_string(L);
            add_norm_init(base + ".input_layernorm.weight", base + ".input_layernorm.weight");
            add_norm_init(base + ".post_attention_layernorm.weight", base + ".post_attention_layernorm.weight");
            // Per-head QK RMSNorm gains (Qwen3 and later). They are graph initializers on this line
            // (`attn.q_norm.layernorm.weight`, F16 [head_dim]) rather than the .fconst's two sections. The forward
            // turns QK norm on from model_type alone, so a pack that ships these gains and an import that dropped them
            // served fluent garbage: the Qwen3-1.7B hybrid pack has them, and without them q and k come out of an
            // uninitialised gain. (Found after this block was first written inside the .fconst payload section, which
            // a genre-A pack never enters, so it silently did nothing.)
            for (const char* leaf : {"q", "k"}) {
                const auto gi = onnx.index.find(base + ".attn." + leaf + "_norm.layernorm.weight");
                if (gi == onnx.index.end()) continue;
                const OnnxTensor& gt = onnx.tensors[gi->second];
                if (gt.dims.empty() || int(gt.dims.at(0)) != p.dims.head_dim) continue;
                const size_t gn = size_t(p.dims.head_dim);
                const uint64_t need = uint64_t(gn) * (gt.type == kF16 || gt.type == kBF16 ? 2u : 4u);
                w.add(base + ".self_attn." + leaf + "_norm.weight", "BF16", {int64_t(gn)},
                      as_bf16(bytes_of(gt, need), gt.type, gn));
                ++r.tensors;
            }
        }
        // The final norm is NOT `model.norm.weight` on this line: measured, both packs name it after a virtual
        // (layers)-th block — `model.layers.30.final_norm_layernorm.weight` on SmolLM2-135M (30 layers) and
        // `model.layers.24.final_norm_layernorm.weight` on Qwen2.5-0.5B (24 layers). Missing it left the container
        // without a final norm, and every logit came back 0.0 -- which is what the head then reported.
        const std::string suffix_norm = std::to_string(p.dims.layers) + ".final_norm_layernorm.weight";
        bool got_final = add_norm_init("model.norm.weight", "model.norm.weight");
        if (!got_final) got_final = add_norm_init("model.layers." + suffix_norm, "model.norm.weight");
        if (!got_final) got_final = add_norm_init("model.final_layernorm.weight", "model.norm.weight");
        if (!got_final)
            std::fprintf(stderr,
                         "WARNING: no final RMSNorm found (tried model.norm.weight, model.layers.%s, "
                         "model.final_layernorm.weight); the container will have none and the head will read "
                         "garbage\n", suffix_norm.c_str());
        // the qkv bias: one fused [q|k|v] initializer on most packs, one per projection on a split pack
        for (int L = 0; L < p.dims.layers; ++L) {
            const std::string base = "model.layers." + std::to_string(L);
            const std::string bn = base + ".attn.qkv_proj.MatMulNBits.bias";
            // A split pack (11 genre-A packs: Qwen3 / gemma-2 / Qwen2.5-14B) carries q/k/v biases separately. Each is
            // the length of its own projection, which is what the fused test below cannot accept.
            if (!onnx.index.count(bn)) {
                const auto rkq = size_t(p.dims.heads) * size_t(p.dims.head_dim);
                const auto rkv = size_t(p.dims.kv_heads) * size_t(p.dims.head_dim);
                for (const auto& [leaf, want] : {std::pair<const char*, size_t>{"q_proj", rkq},
                                                 std::pair<const char*, size_t>{"k_proj", rkv},
                                                 std::pair<const char*, size_t>{"v_proj", rkv}}) {
                    const auto bi = onnx.index.find(base + ".attn." + leaf + ".MatMulNBits.bias");
                    if (bi == onnx.index.end()) continue;
                    const OnnxTensor& bt = onnx.tensors[bi->second];
                    if (bt.dims.empty() || size_t(bt.dims.at(0)) != want) continue;
                    const auto rawb = bytes_of(bt, uint64_t(want) * 2);
                    w.add(base + ".self_attn." + leaf + ".bias", "BF16", {int64_t(want)},
                          as_bf16(rawb, bt.type, want));
                    ++r.tensors;
                    ++r.attn_bias_tensors;
                }
                continue;
            }
            const auto it = onnx.index.find(bn);
            if (it == onnx.index.end()) continue;
            const OnnxTensor& t = onnx.tensors[it->second];
            if (t.dims.empty()) continue;
            const size_t n = size_t(t.dims.at(0));
            const uint64_t need = uint64_t(n) * 2;
            const auto raw = bytes_of(t, need);
            const auto bf = as_bf16(raw, t.type, n);
            const size_t rkq = size_t(p.dims.heads) * size_t(p.dims.head_dim);
            const size_t rkv = size_t(p.dims.kv_heads) * size_t(p.dims.head_dim);
            if (rkq + 2 * rkv != n) continue;                     // an unexpected split: leave the refusal standing
            auto slice = [&](size_t off, size_t cnt, const std::string& out) {
                std::vector<uint8_t> bytes(bf.begin() + off * 2, bf.begin() + (off + cnt) * 2);
                w.add(out, "BF16", {int64_t(cnt)}, std::move(bytes));
                ++r.tensors;
                // the container's `attention_bias` is derived from this counter, and the forward only ADDS the q/k/v
                // biases when the flag is set: without this the container carried bias tensors that the forward
                // silently ignored, which is exactly the difference between this container and the 1.8 one for the
                // same model (True vs False, byte-identical weights).
                ++r.attn_bias_tensors;
            };
            slice(0, rkq, base + ".self_attn.q_proj.bias");
            slice(rkq, rkv, base + ".self_attn.k_proj.bias");
            slice(rkq + rkv, rkv, base + ".self_attn.v_proj.bias");
        }
    }

    // ── genre A: the layer projections are graph initializers, not .fconst tiles (engine#1410 / TAMDCOL) ──
    // The rai_1.3/1.6/1.7.1 line carries each projection as `<proj>.MatMulNBits.{qweight,scales,qzeros,bias}` with a
    // location+offset, and plan() has already located them by those names. The decode is the SAME code the 1.8
    // lm_head below uses -- Q4 codes + scales + zeros -> q4_1 tiles -- applied per projection instead of once, so
    // this is a loop, not a second reader. The one projection that needs arithmetic is the fused qkv: one
    // [q|k|v] tensor that has to be split by head geometry, since only it is fused.
    if (graph_weights_a) {
        std::vector<std::string> proj_gaps;
        auto init_of = [&](const std::string& n) -> const OnnxTensor* {
            const auto it = onnx.index.find(n);
            return it == onnx.index.end() ? nullptr : &onnx.tensors[it->second];
        };
        // Decode rows [row0, row0+rows) of a projection's MatMulNBits weight into q4_1 tiles.
        auto decode_rows = [&](const std::string& proj, int row0, int rows, const std::string& out_name,
                               int k_override) -> bool {
            // The two spellings of a quantised projection (engine#1503): the 1.8/phi-4 form carries a qweight and a
            // scales sibling; Qwen3-VL-4B names them `<proj>.MatMul.weight_Q4` / `<proj>.MatMul.weight_scales` on a
            // three-input node with no qzeros (implicit zero point, applied below).
            // `proj` arrives from a plan source string (`graph:model.layers.N.attn.q_proj`, no trailing dot) as well as
            // from a role name (`...q_proj.`, with one), so normalise before appending a suffix -- otherwise the
            // three-input spelling is looked up as `...q_projMatMul.weight_Q4` and never found, which is how a pack could
            // plan all 252 projections, write only its norms, and fail on the device with `MLP dequant : data_size 0`
            // (engine#1503).
            std::string pre = proj;
            if (!pre.empty() && pre.back() != '.') pre += '.';
            // Every way out of here that writes nothing is a named gap (engine#1841), never a quiet skip.
            auto gap = [&](const std::string& why) {
                proj_gaps.push_back(out_name + ": `" + pre + "` " + why);
                return false;
            };
            const OnnxTensor* qw = init_of(pre + "MatMulNBits.qweight");
            const OnnxTensor* sc = init_of(pre + "MatMulNBits.scales");
            if (!qw || !sc) {
                qw = init_of(pre + "MatMul.weight_Q4");
                sc = init_of(pre + "MatMul.weight_scales");
            }
            // plan() counts a projection present on its codes alone, so a pack missing only the scales passed it; returning
            // quietly here then let the container report itself complete without the projection.
            if (!qw || !sc)
                return gap("has no quantised projection the decoder can pair (tried MatMulNBits.qweight + .scales and "
                           "MatMul.weight_Q4 + _scales)");
            if (qw->dims.size() != 3 || sc->dims.empty())
                return gap("qweight is not [N, K/block, block/2], or its scales have no shape");
            const int N = int(qw->dims[0]);
            const int block = int(qw->dims[2]) * 2;            // 64 B holds 128 four-bit codes
            const int K = k_override > 0 ? k_override : int(qw->dims[1]) * block;
            // NB: no block==128 test here. That constant governs the .fconst *tile* layout (2*(K/block)+2
            // subvolumes); after decode_matmulnbits every weight is in 32-column groups regardless of the
            // block the pack used, which is why a block-32 SmolLM-class pack tiles identically (pi-inferzo
            // verified the recipe on amd/SmolLM2-135M-Instruct_rai_1.7.1_hybrid, corr +0.9957 against the
            // pack's own tied embedding).
            if (block <= 0 || N <= 0 || K <= 0 || K % block || row0 < 0 || row0 + rows > N)
                return gap("has geometry N=" + std::to_string(N) + " K=" + std::to_string(K) + " block=" +
                           std::to_string(block) + " that does not hold rows [" + std::to_string(row0) + ", " +
                           std::to_string(row0 + rows) + ")");
            const uint64_t groups = uint64_t(K) / uint64_t(block), nrows = uint64_t(N);
            uint64_t groups_n = 0;
            if (!mul_ok(nrows, groups, groups_n)) return gap("declares a size that overflows");
            std::vector<float> scales(static_cast<std::vector<float>::size_type>(groups_n));
            const uint64_t sw = sc->type == kF32 ? 4 : 2;
            uint64_t s_need = 0;
            if (!mul_ok(groups_n, sw, s_need)) return gap("declares a size that overflows");
            if (sc->type == kF32) std::memcpy(scales.data(), bytes_of(*sc, s_need), scales.size() * 4);
            else {
                const auto b = as_bf16(bytes_of(*sc, s_need), sc->type, scales.size());
                const auto* d = reinterpret_cast<const uint16_t*>(b.data());
                for (size_t i = 0; i < scales.size(); ++i) scales[i] = bf16_to_f32(d[i]);
            }
            const OnnxTensor* zt = init_of(pre + "MatMulNBits.qzeros");
            uint64_t z_need = 0;
            if (!mul_ok(nrows, (groups + 1) / 2, z_need)) return gap("declares a size that overflows");
            const uint8_t* zp = zt ? bytes_of(*zt, z_need) : nullptr;
            uint64_t qw_need = 0;
            if (!mul_ok(groups_n, uint64_t(block) / 2, qw_need)) return gap("declares a size that overflows");
            Q4 q = decode_matmulnbits(bytes_of(*qw, qw_need), scales.data(), zp, N, K, block);
            // No qzeros tensor means the implicit zero point 2^(bits-1) -- the same rule the head path applies, and the
            // one ONNX Runtime's MatMulNBits uses. Missing it here decoded every PROJECTION with zero point 0 while the
            // head was decoded with 8, which is why the pack imported completely, served on the device, and answered
            // gibberish on both the device and the host-GEMM path (engine#1503: "West什么呢ưởng中国联通nio…").
            if (!zt) implicit_zp_zero(q, 4);
            // the row slice, re-tiled exactly as the whole tensor would be
            Q4 part;
            part.N = rows; part.K = K;
            const size_t rk = size_t(K), kg = size_t(K / 32);
            part.codes.resize(size_t(rows) * rk);
            part.scale.resize(size_t(rows) * kg);
            part.zero.resize(size_t(rows) * kg);
            for (int r = 0; r < rows; ++r) {
                std::memcpy(&part.codes[size_t(r) * rk], &q.codes[size_t(row0 + r) * rk], rk);
                std::memcpy(&part.scale[size_t(r) * kg], &q.scale[size_t(row0 + r) * kg], kg * sizeof(float));
                std::memcpy(&part.zero[size_t(r) * kg], &q.zero[size_t(row0 + r) * kg], kg * sizeof(float));
            }
            std::vector<uint8_t> tiled;
            long long pinexact = 0;
            tile_q4_1_f32(part, tiled, &pinexact);
            // Stored in the SAME form the .fconst route writes (raw tile chunks, dtype I8, shape [ntiles,
            // chunk_bytes]) -- not the semantic ("q4_1", [rows, K]) form. Both carry identical bytes, but only
            // this one tells the reader how to walk them: the semantic form made the forward recompute the grid
            // from shape and reject a 6,144-byte chunk ("chunk 6144 B is not a kind"), while the identical
            // geometry from a 1.8 pack served fine.
            const int64_t nt = int64_t(tiled.size() / kImportTileBytes);
            r.tiles += nt;
            w.add(out_name, "I8", {nt, kImportTileBytes}, std::move(tiled));
            ++r.tensors;
            return true;
        };
        const std::string GN = graph_filename(pack_dir);
        const int q_rows = int(p.dims.heads) * int(p.dims.head_dim);
        const int kv_rows = int(p.dims.kv_heads) * int(p.dims.head_dim);
        for (int L = 0; L < p.dims.layers; ++L) {
            const std::string base = "model.layers." + std::to_string(L);
            const std::string a = base + ".attn.", m = base + ".mlp.";
            decode_rows(a + "o_proj", 0, p.dims.hidden, base + ".self_attn.o_proj.weight", 0);
            decode_rows(m + "gate_proj", 0, p.dims.intermediate, base + ".mlp.gate_proj.weight", 0);
            decode_rows(m + "up_proj", 0, p.dims.intermediate, base + ".mlp.up_proj.weight", 0);
            decode_rows(m + "down_proj", 0, p.dims.hidden, base + ".mlp.down_proj.weight", 0);
            // Fused qkv: rows are [q (heads*head_dim) | k (kv*head_dim) | v (kv*head_dim)], in that order. A split
            // pack (Qwen3 / gemma-2 / Qwen2.5-14B genre-A: 11 packs) carries the three as separate nodes joined by
            // Concat, so each is read whole; the container stores q/k/v separately either way, so nothing downstream
            // can tell the two apart.
            if (init_of(a + "qkv_proj.MatMulNBits.qweight")) {
                decode_rows(a + "qkv_proj", 0, q_rows, base + ".self_attn.q_proj.weight", 0);
                decode_rows(a + "qkv_proj", q_rows, kv_rows, base + ".self_attn.k_proj.weight", 0);
                decode_rows(a + "qkv_proj", q_rows + kv_rows, kv_rows, base + ".self_attn.v_proj.weight", 0);
            } else {
                decode_rows(a + "q_proj", 0, q_rows, base + ".self_attn.q_proj.weight", 0);
                decode_rows(a + "k_proj", 0, kv_rows, base + ".self_attn.k_proj.weight", 0);
                decode_rows(a + "v_proj", 0, kv_rows, base + ".self_attn.v_proj.weight", 0);
            }
        }
        if (!proj_gaps.empty()) {
            for (auto& s : proj_gaps) r.refused.push_back(std::move(s));
            if (!opt.allow_incomplete) return r;   // refuse by name; the container is not written
        }
    }

    for (const auto& mm : onnx.matmuls) {
        const auto wi = onnx.index.find(mm.weight), si = onnx.index.find(mm.scales);
        if (getenv("NPU_PACK_DEBUG"))
            std::fprintf(stderr, "[pack-dbg2] weight=%s found=%d scales=%s found=%d (tensors=%zu index=%zu)\n",
                         mm.weight.c_str(), (int)(wi != onnx.index.end()), mm.scales.c_str(),
                         (int)(si != onnx.index.end()), onnx.tensors.size(), onnx.index.size());
        if (wi == onnx.index.end() || si == onnx.index.end()) continue;
        // Every node's declared geometry is checked against the bytes its qweight actually carries, BEFORE the head
        // filter below: a node that demands more qweight than the tensor holds is a broken pack and must be refused
        // by tensor name, not skipped for the accident of not being the head's geometry (the crafted-pack test in
        // tests/npu_pack_import_test.cpp is exactly that shape).
        {
            const auto& wt = onnx.tensors[wi->second];
            // A 0-dim tensor is either a genre-A *placeholder* (nothing beside it: no location, no length -- the
            // packs ship one next to every real weight) or an OVERSTATED qweight that declares bytes it does not
            // carry. Only the second is a refusal; the first is simply not this node's weight.
            if (wt.dims.empty() && wt.location.empty() && wt.length == 0 && wt.raw == nullptr)
                continue;   // a genre-A placeholder: nothing beside it, so it is not this node's weight
            // else: fall through -- a 0-dim tensor that DOES declare data is legitimate when the byte count
            // matches the geometry (the crafted tests carry both shapes), and bytes_of() below is what decides.
            if (mm.bits != 4)
                throw std::runtime_error(mm.weight + ": MatMulNBits is " + std::to_string(mm.bits) + "-bit, not 4-bit");
            if (mm.N <= 0 || mm.K <= 0 || mm.block <= 0 || mm.K % mm.block)
                throw std::runtime_error(mm.weight + ": bad MatMulNBits geometry N=" + std::to_string(mm.N) +
                                         " K=" + std::to_string(mm.K) + " block=" + std::to_string(mm.block));
            uint64_t groups_n = 0, qw_need = 0;
            if (!mul_ok(uint64_t(mm.N), uint64_t(mm.K) / uint64_t(mm.block), groups_n) ||
                !mul_ok(groups_n, uint64_t(mm.block) / 2, qw_need) ||
                !mul_ok(uint64_t(mm.N), (uint64_t(mm.K) / uint64_t(mm.block) + 1) / 2, groups_n))
                throw std::runtime_error(mm.weight + ": declared geometry overflows");
            bytes_of(wt, qw_need);   // throws, naming the tensor, when the bytes are not there
        }
        // This loop visits EVERY MatMulNBits node -- on the genre-A packs that is 24-31 fused qkv nodes plus the head
        // -- and it ends in a `break` (it was written when a pack had exactly one). So the first node was taken as
        // the head and the loop exited: a *qkv* matrix went into lm_head.weight and the real head was never read,
        // which is why the imported container answered "!!!!!!!!". Only the node whose geometry IS the head is
        // taken here; the projections are imported from the graph's initializers by the block below.
        if (!(mm.N == int(V) && mm.K == int(H))) continue;
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
        Q4 q = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), zp, int(mm.N), int(mm.K), mm.block);
        // no qzeros tensor -> the implicit zero point (engine#1503: Qwen3-VL-4B's spelling, three-input nodes)
        if (!have_zp) implicit_zp_zero(q, mm.bits);

        std::vector<uint8_t> tiled;
        long long pinexact = 0;
        tile_q4_1_f32(q, tiled, &pinexact);
        r.zero_exact_pct = 100.0 * double(q.scale.size() - pinexact) / double(q.scale.size());
        r.codes_total = q.codes.size();

        // self-check 1: tie_word_embeddings makes the pack's own fp16 embedding an
        // independent reference for the decode, with two wrong decoders as controls.
        // A genre-A pack states `tie_word_embeddings` in a config.json it does not ship, so `tied` is false and the
        // one independent reference these packs carry goes unused. Run the same check whenever the head's geometry
        // allows it and REPORT the correlation instead of gating on a flag: for a tied pack the pack's own F16
        // embedding must reproduce its 4-bit head (pi-inferzo measured +0.9957 on SmolLM2-135M rai_1.7.1_hybrid),
        // and for an untied one the number simply comes out low -- either way it is evidence, not a silent skip.
        // NB: this loop visits every MatMulNBits node (the fused qkv at 24-30 of them AND the head), so the flag is
        // only ever SET here -- assigning it per iteration left the last qkv node clearing the result of the head.
        const bool head_geometry = (q.N == int(V) && q.K == int(H));
        if (getenv("NPU_PACK_DEBUG"))
            std::fprintf(stderr, "[pack-dbg] matmul %s N=%d K=%d block=%d | V=%d H=%d -> head=%d\n",
                         mm.weight.c_str(), q.N, q.K, mm.block, int(V), int(H), (int)head_geometry);
        if (tied || head_geometry) r.selfcheck_reference = true;
        if (tied || head_geometry) {
            double sum_abs = 0, sum_ref = 0;
            for (int n = 0; n < q.N; ++n)
                for (int k = 0; k < q.K; ++k) {
                    const size_t g = size_t(n) * (q.K / 32) + size_t(k) / 32;
                    const double x = double(q.codes[size_t(n) * q.K + k]) * q.scale[g] + q.zero[g];
                    double e;
                    e = embed_at(embed_tbl, int(n), k);
                    sum_abs += std::fabs(x - e);
                    sum_ref += std::fabs(e);
                }
            r.lmhead_rel = sum_abs / sum_ref;
            r.lmhead_corr = corr(q, embed_tbl, q.N);
            // The diagnostic follows the NUMBER, not the pack's declaration: a genre-A pack ships no config.json, so
            // `tied` is false there while the head plainly reproduces its embedding, and the readout used to call those
            // packs untied while the container it wrote said tied. Same threshold the container's own decision uses.
            r.selfcheck_tied = r.lmhead_corr > kTieCorr;
            const Q4 swap = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), zp, int(mm.N), int(mm.K),
                                               mm.block, /*swap_nibbles*/ true, false);
            const Q4 nozp = decode_matmulnbits(bytes_of(wt, qw_need), scales.data(), nullptr, int(mm.N), int(mm.K),
                                               mm.block, false, /*drop_zero*/ true);
            r.ctrl_corr_nibble = corr(swap, embed_tbl, 2048);
            r.ctrl_corr_nozp = corr(nozp, embed_tbl, 2048);
        }

        // self-check 2: the tiles as written, read back through the engine's own decoder.
        double scale_sum = 0;
        const int KT = (q.K + q4nx::kCols - 1) / q4nx::kCols;  // the padded grid
        std::vector<float> w32(q4nx::kRows * q4nx::kCols);
        for (int i = 0; i < int(tiled.size() / kImportTileBytes); i += 97) {
            const int rt = i / KT, kt = i % KT;
            q4nx::dequant_tile(q4nx::chunk_of(kContainerParams), tiled.data() + size_t(i) * kImportTileBytes,
                               w32.data(), q4nx::kCols);
            for (int rr = 0; rr < q4nx::kRows; ++rr) {
                const int row = rt * q4nx::kRows + rr;
                for (int cc = 0; cc < q4nx::kCols; ++cc) {
                    const int k = kt * q4nx::kCols + cc;
                    if (k >= q.K) continue;  // padding column: not a weight
                    const size_t g = size_t(row) * (q.K / 32) + size_t(k) / 32;
                    // in f32, exactly as dequant_tile computes it: the stored parameters are the
                    // source's floats, so the reconstruction must agree to the last bit
                    const float ref =
                        float(q.codes[size_t(row) * q.K + k]) * q.scale[g] + q.zero[g];
                    const double err = std::fabs(double(w32[size_t(rr) * q4nx::kCols + cc]) - double(ref));
                    r.q4nx_max_abs = std::max(r.q4nx_max_abs, err);
                    r.q4nx_rel += err;
                    scale_sum += std::fabs(ref);
                }
            }
        }
        r.q4nx_rel = scale_sum > 0 ? r.q4nx_rel / scale_sum : 0;

        // the count has to be taken before tiled is moved (argument order is unspecified)
        const int64_t head_tiles = int64_t(tiled.size() / kImportTileBytes);
        r.tiles += head_tiles;
        w.add("lm_head.weight", "I8", {head_tiles, kImportTileBytes}, std::move(tiled));
        ++r.tensors;
        head_written = true;
        break;
    }

    {
        std::string path_why;
        const std::string path = materialise_token_fconst(pack_dir, p.dims);
        const FconstGeom g = locate_fconst_with_header(path, pack_dir, p.dims);
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
            // A Ryzen AI 1.9 line carries no host-side MatMulNBits head at all: lm_head is the .fconst tail. Its
            // layout, measured to the byte on Qwen2.5-0.5B: the same 1,856-byte envelope the final norm opens
            // with, then the head's own 64-row tiles at K = hidden -- so `1,856 + (vocab/64) x tile(hidden)` has
            // to equal the tail length exactly. When that arithmetic does not hold the head is refused by name,
            // because reading the wrong offsets with confidence is worse than not reading them.
            if (onnx.matmuls.empty() && p.head_block != 0 && p.head_block != kKBlock) {
                // engine#1503: the pack quantises its HEAD at a block this importer cannot walk, but every LAYER is at
                // kKBlock (the plan proved that before allowing this path), so the tail's head tiles are the only
                // thing out of reach. On a pack that DECLARES tied embeddings the head IS the embedding, so it is
                // synthesised here -- 4-bit, one scale and zero-point per 32 columns, the same form tile_q4_1_f32
                // writes -- and the self-check then measures the synthesis against the embedding rather than being
                // skipped. An untied pack is refused by name: its head is a matrix nothing else in the pack can
                // stand in for.
                if (!(tie_declared && tied)) {
                    late.push_back("lm_head.weight: this pack quantises its head at block " +
                                   std::to_string(p.head_block) + " along K, which this importer's tile geometry cannot "
                                   "walk, and the pack does not declare tied embeddings, so the embedding cannot stand "
                                   "in for it");
                } else {
                    const size_t rows_v = size_t(V), kk = size_t(H);
                    Q4 q;
                    q.N = int(V); q.K = int(H); q.block = 32;
                    q.codes.assign(rows_v * kk, 0);
                    q.scale.assign(rows_v * (kk / 32), 0.0f);
                    q.zero.assign(rows_v * (kk / 32), 0.0f);
                    for (size_t v = 0; v < rows_v; ++v) {
                        for (size_t gk = 0; gk < kk / 32; ++gk) {
                            float lo = 1e30f, hi = -1e30f;
                            for (size_t k = 0; k < 32; ++k) {
                                const size_t idx = v * kk + gk * 32 + k;
                                const float x = float(embed_at(embed_tbl, int(v), int(gk * 32 + k)));
                                lo = std::min(lo, x); hi = std::max(hi, x);
                            }
                            const float sc = (hi - lo) / 15.0f;
                            q.scale[v * (kk / 32) + gk] = sc > 0 ? sc : 1.0f;
                            q.zero[v * (kk / 32) + gk] = lo;
                            for (size_t k = 0; k < 32; ++k) {
                                const size_t idx = v * kk + gk * 32 + k;
                                const float x = float(embed_at(embed_tbl, int(v), int(gk * 32 + k)));
                                const int code = sc > 0 ? int(std::lround((x - lo) / sc)) : 0;
                                q.codes[idx] = uint8_t(std::min(15, std::max(0, code)));
                            }
                        }
                    }
                    // The self-check still means something here, and it is worth saying what: this head was DERIVED
                    // from the embedding, so the correlation measures the 4-bit quantisation (and the row-shifted
                    // control still shows what a misaligned read would score). It does not vouch for the pack -- the
                    // pack's own head tiles were never read, which q4nx_head_note states.
                    {
                        r.lmhead_corr = corr(q, embed_tbl, q.N);
                        r.selfcheck_reference = true;
                        r.selfcheck_tied = r.lmhead_corr > kTieCorr;
                        double sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
                        const int nsh = std::min(2048, q.N - 1);
                        for (int rr = 0; rr < nsh; ++rr)
                            for (int k = 0; k < q.K; ++k) {
                                const size_t g = size_t(rr) * (q.K / 32) + size_t(k) / 32;
                                const double x = double(q.codes[size_t(rr) * q.K + k]) * q.scale[g] + q.zero[g];
                                const double y = embed_at(embed_tbl, rr + 1, k);
                                sxy += x * y; sx += x; sy += y; sxx += x * x; syy += y * y;
                            }
                        const double m = double(nsh) * q.K;
                        const double cx = sxx - sx * sx / m, cy = syy - sy * sy / m;
                        if (m > 0 && cx > 1e-12 && cy > 1e-12)
                            r.ctrl_corr_rowshift = (sxy - sx * sy / m) / std::sqrt(cx * cy);
                    }
                    std::vector<uint8_t> tiled;
                    long long pinexact = 0;
                    tile_q4_1_f32(q, tiled, &pinexact);
                    const int64_t head_tiles = int64_t(tiled.size() / kImportTileBytes);
                    r.tiles += head_tiles;
                    w.add("lm_head.weight", "I8", {head_tiles, kImportTileBytes}, std::move(tiled));
                    ++r.tensors;
                    head_written = true;
                    r.head_synthesized_note =
                        "the pack quantises its head at block " + std::to_string(p.head_block) +
                        " along K, so the tail's head tiles are not read; the pack declares tied embeddings, so "
                        "lm_head is the 4-bit quantisation of model.embed_tokens";
                    std::fprintf(stderr,
                                 "note: this pack's head is block %lld along K, which this importer cannot walk; the "
                                 "pack is tied, so the head was taken from the embedding (engine#1503)\n",
                                 (long long)p.head_block);
                }
            } else if (onnx.matmuls.empty()) {
                const int64_t tail_off = int64_t(p.dims.layers) * g.stride;
                const int64_t tile = attn_tile_bytes(H);
                // ceil, not floor: a vocab that is not a multiple of 64 is padded up to whole tiles in the pack
                // (Hunyuan-0.5B: 120,818 rows = 1,887.8 tiles, so the tail holds 1,888 and the last one is partly
                // padding). The rows past V are dropped before the tensor is written, so the container's head has
                // exactly V rows.
                const int64_t rows = (V + kTileRows - 1) / kTileRows;
                // The envelope is the same `64 + 2*hidden` the layer blocks open with (eps, padding, the norm), not a
                // constant: 1,856 is only its qwen2.5-0.5B value. Qwen3-1.7B's tail is 4,160 + 2,374 x 73,984 to the
                // byte, which the constant would have missed by 2,304.
                const int64_t env = attn_payload_off(H);
                const int64_t want = env + rows * tile;
                // engine#1740: the head is LOCATED, not assumed to end the file. `want == have` was an equality
                // check on the tail length, so a pack whose .fconst ends in padding (110,208,000 B on the
                // hidden-1536 family) was refused for a reason about placement rather than about its fields.
                const TailHeadLocate loc = locate_tail_head(fc, tail_off, env, rows, tile, H, embed_tbl, tie_declared && tied);
                if (loc.off < 0) {
                    late.push_back("lm_head.weight: " + loc.why);
                } else {
                    // engine#1740: say WHICH placement was taken. "imported" alone cannot distinguish a tail that ends
                    // with the head from one that ends in padding, and that difference is the whole point of the
                    // search -- so the number a reviewer needs is here rather than inferred from the file size.
                    std::fprintf(stderr,
                                 "  lm head: tile 0 at %lld B, %lld tile(s) after a %lld B envelope; %lld B of trailing "
                                 "zeros, %lld tile(s) of slack%s%s\n",
                                 (long long)loc.off, (long long)rows, (long long)env, (long long)loc.zeros,
                                 (long long)loc.slack,
                                 loc.exact ? ", the tail ends with the head" : "",
                                 loc.verified ? " (tied: located by correlation with the pack's embedding)"
                                              : " (untied: placement taken, not verified)");
                    std::vector<TileW> tiles;
                    tiles.reserve(size_t(rows));
                    bool exact = true;
                    for (int64_t i = 0; i < rows; ++i) {
                        const uint8_t* src = at(fc, loc.off + i * tile, tile);
                        tiles.push_back(decode_tile(src, int(H)));
                        const std::vector<uint8_t> back = encode_tile(tiles.back());
                        if (std::memcmp(back.data(), src, size_t(tile)) != 0) exact = false;
                    }
                    if (!exact) {
                        late.push_back("lm_head.weight: the .fconst tail's tiles do not survive decode->encode, "
                                       "so their fields are not understood");
                    } else {
                        Q4 q = q_from_tiles(tiles);
                        // The pack's own padding stays: q4_1 tiles need N % 32 == 0, and the padded row count
                        // (ceil(V/64) x 64) satisfies it where V itself may not -- Hunyuan-0.5B has 120,818 rows, so
                        // the head carries 120,832 and the 14 extra are padding the vocabulary never addresses (the
                        // container still states vocab_size = 120,818).
                        if (q.N != int(V))
                            std::fprintf(stderr,
                                         "note: the head is stored as %d rows for a vocabulary of %lld (the pack pads "
                                         "to whole 64-row tiles); vocab_size stays %lld in the container\n",
                                         q.N, (long long)V, (long long)V);
                        std::vector<uint8_t> tiled;
                        long long pinexact = 0;
                        tile_q4_1_f32(q, tiled, &pinexact);
                        r.codes_total = q.codes.size();
                        r.zero_exact_pct = 100.0 * double(q.scale.size() - pinexact) / double(q.scale.size());
                        // The pack's own fp16 embedding is an independent reference for this decode: on a tied
                        // model it must reproduce the 4-bit head, and on an untied one the number simply comes
                        // out low. Report it rather than skipping: it is evidence either way (same rule the host
                        // head path uses).
                        // q.N may exceed V by the pack's tile padding (Hunyuan-0.5B: 120,832 rows for 120,818),
                        // so the reference is compared over the vocabulary's own rows.
                        if (q.N >= int(V) && q.K == int(H)) {
                            double sum_abs = 0, sum_ref = 0;
                            for (int n = 0; n < int(V); ++n)
                                for (int k = 0; k < q.K; ++k) {
                                    const size_t g = size_t(n) * (q.K / 32) + size_t(k) / 32;
                                    const double x = double(q.codes[size_t(n) * q.K + k]) * q.scale[g] + q.zero[g];
                                    const double e =
                                        embed_at(embed_tbl, int(n), k);
                                    sum_abs += std::fabs(x - e);
                                    sum_ref += std::fabs(e);
                                }
                            r.lmhead_rel = sum_abs / sum_ref;
                            r.lmhead_corr = corr(q, embed_tbl, q.N);
                            r.selfcheck_reference = true;
                            r.selfcheck_tied = r.lmhead_corr > kTieCorr;
                            // The control this path can actually run: the same tiles and the same decoder, with each
                            // head row matched against the NEXT embedding row. If a misaligned head still correlated,
                            // the number above would be measuring the vocabulary's mean brightness rather than the
                            // head. (The host path's controls mis-decode its MatMulNBits bytes instead; there are no
                            // such bytes here, and printing its numbers from struct defaults -- which is what this
                            // path did until review -- is worse than printing none.)
                            {
                                double sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
                                const int n = std::min(2048, q.N - 1);
                                for (int rr = 0; rr < n; ++rr)
                                    for (int k = 0; k < q.K; ++k) {
                                        const size_t g = size_t(rr) * (q.K / 32) + size_t(k) / 32;
                                        const double x = double(q.codes[size_t(rr) * q.K + k]) * q.scale[g] + q.zero[g];
                                        const double y =
                                            embed_at(embed_tbl, int(rr) + 1, k);
                                        sxy += x * y; sx += x; sy += y; sxx += x * x; syy += y * y;
                                    }
                                const double m = double(n) * q.K;
                                const double cx = sxx - sx * sx / m, cy = syy - sy * sy / m;
                                if (m > 0 && cx > 1e-12 && cy > 1e-12)
                                    r.ctrl_corr_rowshift = (sxy - sx * sy / m) / std::sqrt(cx * cy);
                            }
                        }
                        const int64_t head_tiles = int64_t(tiled.size() / kImportTileBytes);
                        r.tiles += head_tiles;
                        w.add("lm_head.weight", "I8", {head_tiles, kImportTileBytes}, std::move(tiled));
                        ++r.tensors;
                        head_written = true;
                    }
                }
            }
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
                        // The cross-check compares the two copies a pack can ship; the one that matters is the
                        // Token_Token tail, because that is the program this importer reads the norm for and
                        // the one the device runs. So a mismatch is REPORTED, not fatal: refusing the import
                        // over a duplicated copy would make an otherwise complete pack unimportable for a
                        // divergence we cannot repair, and the container is unaffected because the norm comes
                        // from Token. (The shipped packs agree byte for byte here -- Qwen2.5-0.5B, Qwen3-1.7B,
                        // Qwen3-4B and Llama-3.2-1B all report "identical in both .fconst tails" -- so this
                        // guards against a divergent duplicate rather than describing the packs.)
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
                        long long pinexact = 0;
                        tile_q4_1_f32(q, tiled, &pinexact);
                        zero_inexact_all += pinexact;
                        zero_groups_all += (long long)q.scale.size();
                        const int64_t nt_tiles = int64_t(tiled.size() / kImportTileBytes);
                        r.tiles += nt_tiles;
                        w.add(name + ".weight", "I8", {nt_tiles, kImportTileBytes}, std::move(tiled));
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
                            long long pinexact = 0;
                            tile_q4_1_f32(q, tiled, &pinexact);
                            mlp_zero_inexact += pinexact;
                            mlp_zero_groups += (long long)q.scale.size();
                            const int64_t nt = int64_t(tiled.size() / kImportTileBytes);
                            r.tiles += nt;
                            w.add(name + ".weight", "I8", {nt, kImportTileBytes}, std::move(tiled));
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
        // The container's config.json is built from the pack's geometry -- which on 1.6 (`config.json = {}`) and
        // 1.7/1.7.1 (no config.json at all) comes from genai_config.json. Parsing `<pack>/config.json` directly
        // here threw "parse error ... empty input" on exactly those packs, after every tensor had been written.
        std::string geom_src;
        json c = load_geometry(pack_dir, &geom_src);
        // The container states the geometry the *import* resolved, not just what one metadata file happened to
        // carry: plan() derives intermediate_size from the graph on packs whose metadata has none (1.6/1.7/1.7.1),
        // and without these keys the forward re-derives them from whatever tensors it finds -- on
        // SmolLM2-135M rai_1.7.1_hybrid that came out as IM=3072 against the pack's 1536.
        c["hidden_size"] = p.dims.hidden;
        c["intermediate_size"] = p.dims.intermediate;
        c["num_hidden_layers"] = p.dims.layers;
        c["num_attention_heads"] = p.dims.heads;
        c["num_key_value_heads"] = p.dims.kv_heads;
        c["head_dim"] = p.dims.head_dim;
        c["vocab_size"] = p.dims.vocab;
        c["model_type"] = p.dims.model_type;
        c["rope_theta"] = p.dims.rope_theta;
        // A pack that ships its own precomputed RoPE (cos_cache/sin_cache) is not necessarily built for the theta
        // its genai_config.json states -- pi-inferzo measured 10000 declared against a model that needs 1e6 on
        // Qwen2.5 rai_1.7.1_hybrid, and our forward computes RoPE from theta rather than reading a cache. That is a
        // silent-wrong-answer risk, so the container says so rather than serving garbage quietly.
        {
            const bool has_cache = onnx.index.count("cos_cache") && onnx.index.count("sin_cache");
            if (has_cache) {
                // The note stands until the cache check below proves or refutes the declared theta: printing the
                // warning here made the log contradict itself on packs that then verified (Qwen3-1.7B hybrid prints
                // "unverified" and "adopted 1e6" two lines apart).
                c["q4nx_rope_note"] =
                    "the pack ships precomputed cos_cache/sin_cache; rope_theta here comes from the pack's metadata "
                    "and has NOT been verified against those caches, and this forward computes RoPE from theta";
            }
        }
        c["rms_norm_eps"] = p.dims.rms_norm_eps;
        // The kernel family this container was imported for, so a serve that derives a different one refuses instead of
        // answering garbage (the forward compares this exact string). Shape per role, in the order the importer plans
        // them: QKV is the fused q|k|v width, GU the fused gate|up width, both on the padded K grid.
        {
            // The forward's own grids (npu/forward/include/q4nx_pack.h): K pads to the 256-column grid, N to the
            // 512-column one. Spelled out here because the importer is a different build target from that helper;
            // the forward compares this exact string, so a drift between the two shows up as a refusal (loud)
            // rather than as wrong text (silent). Verified against both packs' `derived 4 designs` lines:
            // SmolLM2 QKV 768x1024 / O 768x1024 / GU 768x3072 / D 1536x1024; Qwen2.5-0.5B 1024x1536 / 1024x1024 /
            // 1024x9728 / 4864x1024.
            auto pad_k = [](int k) { return (k + 255) / 256 * 256; };
            auto pad_n = [](int n) { return (n + 511) / 512 * 512; };
            const int kpad = pad_k(p.dims.hidden);
            const int q_out = p.dims.heads * p.dims.head_dim, kv_out = p.dims.kv_heads * p.dims.head_dim;
            // o_proj's K is q_proj's OUTPUT width, which is hidden only when heads*head_dim == hidden. On
            // Qwen3-4B it is 32 x 128 = 4096 against a hidden of 2560, and declaring pad_k(hidden) there made the
            // forward refuse this container ("declares O:2560x2560, this model derives O:4096x2560") -- the family
            // guard doing exactly its job, since the mismatch would otherwise have been wrong text.
            const int kpad_o = pad_k(int64_t(p.dims.heads) * p.dims.head_dim);
            c["q4nx_designs"] = "QKV:" + std::to_string(kpad) + "x" + std::to_string(pad_n(q_out + 2 * kv_out)) +
                                ",O:" + std::to_string(kpad_o) + "x" + std::to_string(pad_n(p.dims.hidden)) +
                                ",GU:" + std::to_string(kpad) + "x" + std::to_string(pad_n(2 * p.dims.intermediate)) +
                                ",D:" + std::to_string(pad_k(p.dims.intermediate)) + "x" +
                                std::to_string(pad_n(p.dims.hidden));
        }
        // The stop tokens: `load_geometry` already copied eos/bos/pad and `architectures` out of the pack's
        // genai_config.json (these packs ship NO generation_config.json -- I had assumed they did), and read_dims
        // parsed eos into p.dims. This line is the belt: whatever the plan resolved wins over the raw metadata.
        if (!p.dims.eos.empty()) c["eos_token_id"] = p.dims.eos;
        // ── the RoPE base, verified against the pack's own cache ──
        // These packs ship precomputed cos_cache/sin_cache, and their genai_config.json is not a trustworthy source
        // for theta (pi-inferzo measured 10000 declared where the model needs 1e6). Our forward COMPUTES RoPE from
        // theta instead of reading a cache, so a wrong base is a silent wrong answer at long context. The cache is
        // the pack's own statement of its base, so the loader tests candidates against it and adopts the winner:
        // correlation against cos(p * inv_freq_i) over p, which is invariant to the per-pack scaling of the cache
        // (the 1.8 packs store pre-scaled cosines -- pos-0 reads 1.875, not 1.0).
        if (onnx.index.count("cos_cache")) {
            const OnnxTensor& cc = onnx.tensors[onnx.index.at("cos_cache")];
            if (!cc.dims.empty() && cc.dims.size() == 2 && cc.dims.at(2 % cc.dims.size()) > 0) {
                const int64_t rows = cc.dims.at(0), half = cc.dims.at(1);
                const size_t n = size_t(rows) * size_t(half);
                uint64_t need = 0;
                if (mul_ok(n, cc.type == kF32 ? 4u : 2u, need)) {
                    std::vector<float> cache(n);
                    if (cc.type == kF32) std::memcpy(cache.data(), bytes_of(cc, need), n * 4);
                    else {
                        const auto b = as_bf16(bytes_of(cc, need), cc.type, n);
                        const auto* d = reinterpret_cast<const uint16_t*>(b.data());
                        for (size_t i = 0; i < n; ++i) cache[i] = bf16_to_f32(d[i]);
                    }
                    // candidates: what the metadata says first, then the values these families actually use
                    std::vector<double> cands{p.dims.rope_theta, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, 1.6e6, 3.2e6};
                    double best_corr = -2.0, best_theta = p.dims.rope_theta;
                    const int HD2 = int(half) * 2;          // head_dim implied by the cache
                    const int64_t pmax = std::min<int64_t>(rows, 4096);
                    std::vector<double> ang(pmax), cs(size_t(pmax));
                    for (double th : cands) {
                        if (!(th > 0)) continue;
                        // ONE Pearson r over the whole cache matrix, not a mean of per-column r's. For a large
                        // head_dim the last columns barely vary -- cos(p * inv) is nearly flat over the positions the
                        // cache holds -- so their per-column r is noise, and averaging it in drags a CORRECT theta
                        // below the 0.99 gate. Measured on Qwen3-1.7B-awq-quant-onnx-hybrid (head_dim 128): the mean
                        // of the 57 usable columns gives 0.9826 at the right theta 1e6, while the flat r gives
                        // 1.000000 and the declared 1e4 sits at 0.17 either way.
                        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                        int64_t used = 0;
                        for (int64_t i = 0; i < half; ++i) {
                            const double inv = 1.0 / std::pow(th, 2.0 * double(i) / double(HD2));
                            for (int64_t pp = 0; pp < pmax; ++pp) {
                                const double y = cache[size_t(pp) * size_t(half) + size_t(i)];
                                const double x = std::cos(double(pp) * inv);
                                sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                                ++used;
                            }
                        }
                        const double np = double(used);
                        const double cx = sxx - sx * sx / np, cy = syy - sy * sy / np;
                        if (used > 0 && cx > 1e-12 && cy > 1e-12) {
                            const double corr = (sxy - sx * sy / np) / std::sqrt(cx * cy);
                            if (corr == corr && corr > best_corr) { best_corr = corr; best_theta = th; }
                        }
                    }
                    if (best_corr > 0.99 && std::fabs(best_theta - p.dims.rope_theta) > 1e-9) {
                        std::fprintf(stderr,
                                     "note: rope_theta %.6g declared, %.6g adopted (correlation %.4f against the "
                                     "pack's own cos_cache)\n", (double)p.dims.rope_theta, best_theta, best_corr);
                        c["rope_theta"] = best_theta;
                        c.erase("q4nx_rope_note");
                    } else if (best_corr > 0.99) {
                        std::fprintf(stderr, "note: rope_theta %.6g verified against the pack's cos_cache (corr %.4f)\n",
                                     (double)p.dims.rope_theta, best_corr);
                        c.erase("q4nx_rope_note");
                    } else {
                        std::fprintf(stderr,
                                     "WARNING: this pack ships precomputed RoPE caches that no plain theta explains "
                                     "(best %.4f at %.6g, declared %.6g); this forward computes RoPE from theta, so "
                                     "attention may be wrong from the first token\n",
                                     best_corr, best_theta, (double)p.dims.rope_theta);
                    }
                }
            }
        }
        const TextArtifacts ta = scan_text_artifacts(pack_dir);
        c["vocab_size"] = V;
        c["attention_bias"] = r.attn_bias_tensors > 0;  // what the container actually carries
        if (!r.head_synthesized_note.empty()) c["q4nx_head_note"] = r.head_synthesized_note;
        // The forward picks between "logits from the token embedding" and "logits from a separate head tensor" on this
        // key, and defaults to TIED when it is absent (npu/forward/src/q4nx_semantics.c). An untied pack that ships no
        // config.json -- every genre-A hybrid, e.g. Phi-3-mini -- therefore used to look tied, and the forward decoded
        // its logits from the embedding matrix: fluent-looking garbage ('slower slower slower'), with the head tensor
        // sitting unused in the container. The pack's own declaration wins where it has one; otherwise the head we
        // actually imported decides, by the same correlation the self-check reports.
        if (tie_declared) {
            c["tie_word_embeddings"] = tied;
        } else if (head_written) {
            c["tie_word_embeddings"] = r.lmhead_corr > kTieCorr;
            c["q4nx_tie_note"] = "this pack states no tie_word_embeddings, so it is taken from the data: the " +
                                 std::string(r.lmhead_corr > kTieCorr ? "imported head reproduces the embedding"
                                                                      : "imported head is a separate matrix") +
                                 " (correlation " + std::to_string(r.lmhead_corr) + ")";
        }
        c["q4nx_source"] = {{"converter", "1bit npu-pack import"},
                            {"pack", p.pack},
                            {"layout", q4nx::layout_of(kContainerParams)},
                            {"incomplete", r.refused.size()},
                            // so a container read months later states whether it ever had text
                            {"text_artifacts", ta.present.empty() ? "none in the pack"
                                                                  : "copied from the pack"}};
        // The device weight layout, when the caller declared one (`--weight-layout b8`): the container
        // then needs the `_b8` / `_b8_m16` kernel family and packB writes the blocked image host-side.
        // The key stays absent for the default layout, exactly as every earlier container has it.
        if (!opt.weight_layout.empty() && opt.weight_layout != "rm")
            c["q4nx_weight_layout"] = opt.weight_layout;
        std::ofstream out(out_dir + "/config.json");
        out << c.dump(2) << "\n";
    }
    // engine#1751 follow-up: the same copier the external-data path uses, so the two genres cannot drift apart again.
    copy_text_artifacts(pack_dir, out_dir);
    for (const auto& [name, m] : ext) unmap_file(m, ext_size[name]);
    return r;
}

// ── the subcommand ──────────────────────────────────────────────────────────
int run(int argc, char** argv) {
    const std::string usage =
        "usage: 1bit npu-pack info   <pack dir>\n"
        "       1bit npu-pack import <pack dir> -o <out dir> [--allow-incomplete]\n"
        "                           [--weight-layout b8]\n"
        "  <pack dir>  a Ryzen AI OGA NPU model pack (*_rai_*_npu_*: model.onnx,\n"
        "              model.onnx.data, dd_metastate_*.fconst, config.json, tokenizer)\n"
        "  info        print what the pack states as JSON; writes nothing\n"
        "  import      write <out dir>/model.q4nx + config.json (+ the pack's text files, if any)\n"
        "  --allow-incomplete  write what is representable and list the rest in the\n"
        "              container's manifest instead of refusing\n"
        "  --weight-layout b8  declare the device weight layout the runtime will use, so the\n"
        "              container needs the _b8 / _b8_m16 kernel family (leaving the key out is\n"
        "              the row-major default, which is what every earlier container has)\n"
        "  --kernels <dir>     link a kernel set into the container as <out dir>/npu, which is where\n"
        "              `1bit serve -m <out dir> --device npu` looks for it, instead of doing that by\n"
        "              hand; the set's family is checked against --weight-layout before anything is\n"
        "              written\n";
    if (argc < 1) {
        std::fprintf(stderr, "%s", usage.c_str());
        return 2;
    }
    const std::string cmd = argv[0];
    if (cmd == "--help" || cmd == "-h") {
        std::printf("%s", usage.c_str());
        return 0;
    }
    std::string dir, out, layout, kernels;
    bool allow = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-o" || a == "--out") {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            out = argv[++i];
        } else if (a == "--allow-incomplete") allow = true;
        else if (a == "--kernels") {
            if (i + 1 >= argc) throw std::runtime_error("--kernels needs a value");
            kernels = argv[++i];
        }
        else if (a == "--weight-layout") {
            if (i + 1 >= argc) throw std::runtime_error("--weight-layout needs a value");
            layout = argv[++i];
            const std::string why = weight_layout_refusal(layout);
            if (!why.empty()) throw std::runtime_error("--weight-layout " + layout + ": " + why);
        }
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

    // Checked before the import runs: a set of the wrong family (or a directory that is not a set at
    // all) must refuse with nothing written, not leave a container behind that cannot be served.
    if (!kernels.empty()) {
        std::vector<std::string> names;
        std::error_code ec;
        for (fs::directory_iterator it(kernels, ec), end; !ec && it != end; it.increment(ec))
            names.push_back(it->path().filename().string());
        if (ec)
            throw std::runtime_error("--kernels " + kernels + ": cannot read that directory (" +
                                     ec.message() + ")");
        const std::string why = kernel_set_refusal(names, layout == "rm" ? "" : layout);
        if (!why.empty()) throw std::runtime_error("--kernels " + kernels + ": " + why);
    }

    const Report r = import_pack(dir, out, Options{allow, layout});
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
    if (r.selfcheck_reference && r.selfcheck_tied) {
        std::printf("  self-check: the head reproduces the pack's own embedding (as imported, bf16)  corr %.6f  rel_err %.6f\n"
                    "              (tied: this correlation is the check, and it passed)\n",
                    r.lmhead_corr, r.lmhead_rel);
        if (r.ctrl_corr_rowshift >= 0)
            std::printf("  controls:   row-shifted corr %.4f (must fail)\n", r.ctrl_corr_rowshift);
        else
            std::printf("  controls:   nibble-swapped corr %.4f, no zero-point corr %.4f (both must fail)\n",
                        r.ctrl_corr_nibble, r.ctrl_corr_nozp);
    } else if (r.selfcheck_reference) {
        // An untied pack: lm_head and model.embed_tokens are separate matrices, so correlating them is not a test of
        // anything. Say so, with the number, so the line cannot be read as a decoding failure.
        std::printf("  self-check: not a check here -- this pack is untied (lm_head and the embedding are separate\n"
                    "              matrices), so their correlation %.6f and rel_err %.6f are what an untied pack\n"
                    "              looks like, not a decode failure; the controls still run\n",
                    r.lmhead_corr, r.lmhead_rel);
        if (r.ctrl_corr_rowshift >= 0)
            std::printf("  controls:   row-shifted corr %.4f (must fail)\n", r.ctrl_corr_rowshift);
        else
            std::printf("  controls:   nibble-swapped corr %.4f, no zero-point corr %.4f (both must fail)\n",
                        r.ctrl_corr_nibble, r.ctrl_corr_nozp);
    } else {
        std::printf("  self-check: skipped -- the head is not tied to the embedding, so the pack\n"
                    "              carries no independent reference to test the decode against\n");
    }
    std::printf("  container:  read back through q4nx::dequant_tile  rel_err %.3e  max |diff| %.3e\n",
                r.q4nx_rel, r.q4nx_max_abs);
    if (!layout.empty() && layout != "rm")
        std::printf("  layout:     declares q4nx_weight_layout \"%s\" -- this container needs the "
                    "_%s kernel family, and packB writes that image host-side\n",
                    layout.c_str(), layout.c_str());
    std::printf("  parameters: fp32 scale/zero exact for %.2f%% of 32-column groups "
                "(container layout %s, the pack's own values; no bf16 round trip)\n",
                r.zero_exact_pct, q4nx::layout_of(kContainerParams));
    if (r.attn_tiles_total)
        std::printf("  attention:  %lld/%lld .fconst tiles survive decode->encode byte for byte; "
                    "%d projections + %d bias tensors imported, fp32 scale/zero exact for %.2f%% of groups\n",
                    r.attn_tiles_ok, r.attn_tiles_total, r.attn_projections, r.attn_bias_tensors,
                    100.0 - r.attn_zero_inexact_pct);
    if (r.mlp_groups_total)
        std::printf("  MLP:        %lld/%lld .fconst groups survive decode->encode byte for byte; "
                    "%d projections imported, fp32 scale/zero exact for %.2f%% of groups\n",
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
    if (!kernels.empty()) {
        // A model directory carries its own kernels in `npu/`; serve resolves the link, so the
        // container stays usable from wherever it was written.  An existing `npu/` that is a real
        // directory is left alone (and named), because deleting someone's kernels to make room for a
        // link is not an importer's decision.
        const std::string link = out + "/npu";
        std::error_code ec;
        const fs::file_status st = fs::symlink_status(link, ec);
        size_t n = 0, n_m16 = 0, n_b8 = 0;
        for (fs::directory_iterator it(kernels, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string f = it->path().filename().string();
            if (f.rfind("full_i8_", 0) != 0 || f.size() < 9 || f.compare(f.size() - 4, 4, ".elf") != 0)
                continue;
            n++;
            if (f.find("_m16") != std::string::npos) n_m16++;
            if (f.find("_b8") != std::string::npos) n_b8++;
        }
        if (fs::is_directory(st) && !fs::is_symlink(st)) {
            std::printf("  kernels:    %s already holds its own kernels; --kernels %s not linked\n",
                        link.c_str(), kernels.c_str());
            std::fprintf(stderr,
                         "WARNING: %s is a real directory, so the imported container keeps it. Remove\n"
                         "  it and re-run the import if the set at %s is the one it should serve.\n",
                         link.c_str(), kernels.c_str());
        } else {
            fs::remove(link, ec);  // a stale link (or a stray file) is replaced, not merged
            fs::create_directory_symlink(fs::absolute(kernels), link, ec);
            if (ec) throw std::runtime_error("cannot link " + link + " -> " + kernels + ": " + ec.message());
            std::printf("  kernels:    %s -> %s  (%zu full_i8 ELFs: %zu M=16 twins, %zu %s)\n", link.c_str(),
                        kernels.c_str(), n, n_m16, n_b8,
                        layout == "b8" ? "of the _b8 family" : "of the row-major family");
        }
    }
    const TextArtifacts ta = scan_text_artifacts(dir);
    if (ta.tokenizer) {
        std::printf("  text:       %zu of %zu tokenizer/chat files copied", ta.present.size(),
                    sizeof(kTextFiles) / sizeof(*kTextFiles));
        if (!ta.missing.empty()) {
            std::printf(", absent from the pack:");
            for (const auto& f : ta.missing) std::printf(" %s", f.c_str());
        }
        std::printf("\n");
    } else {
        // Not every pack has them, and the run that fails without them is two commands later.
        std::printf("  text:       none -- this pack ships no tokenizer.json\n");
        std::fprintf(stderr,
                     "WARNING: %s carries no tokenizer.json, so %s has weights and config.json\n"
                     "  only and `1bit serve` will refuse it as not an NPU model directory. Put the\n"
                     "  model's own text artifacts there (tokenizer.json, tokenizer_config.json,\n"
                     "  vocab.json, merges.txt, chat_template.jinja) -- e.g. from the upstream\n"
                     "  checkpoint the pack was built from.\n",
                     dir.c_str(), out.c_str());
    }
    return 0;
}


// Public (engine#1662): see the declaration in pack_import.h. Lives outside the anonymous namespace because the test
// calls it directly with crafted files, so the decision that guards a ~291 MB rewrite is pinned without a 1.9 pack.
bool canonical_copy_reusable(const std::string& work, int64_t expected_len) {
    if (expected_len <= 0) return false;
    std::error_code ec;
    // file_size() fails on a missing file and on a directory, which is exactly the set that must be rewritten.
    const std::uintmax_t have = fs::file_size(work, ec);
    return !ec && have == static_cast<std::uintmax_t>(expected_len);
}

// engine#1770: reuse (engine#1662) writes a NEW copy when the source changes instead of overwriting, so copies only ever
// accumulate -- 699 files and 8.7 GiB of tmpfs (i.e. RAM) on this box, most of them truncated 0-byte tails. The identity
// in the name is `1bit-fconst19v<n>-<pack tag>-<hidden>-<heads>-<layers>-<source size>-<source mtime>.fconst`, so keeping
// one copy per (pack, geometry) means removing the entries that share the prefix but not the whole name. That cannot take
// a copy a reader is about to use: reuse requires the entire name to match, and the one name that can match is the one
// being kept. A reader holding a removed file open is unaffected (the inode outlives the unlink); at worst a concurrent
// lane rewrites a copy it would have reused, which costs a file write and no correctness.
void evict_stale_canonical_copies(const std::string& dir, const std::string& keep_name, const std::string& prefix,
                                  const std::string& legacy_name, int64_t max_age_seconds) {
    if (dir.empty() || keep_name.empty() || prefix.empty() || keep_name.size() <= prefix.size()) return;
    const std::string stem = "1bit-fconst19";
    const auto        now  = fs::file_time_type::clock::now();
    std::error_code   ec;
    for (fs::directory_iterator it(fs::path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string base = it->path().filename().string();
        if (base == keep_name) continue;
        // same pack and geometry, different source identity: nothing reuses it
        if (base.size() > prefix.size() && base.compare(0, prefix.size(), prefix) == 0) {
            std::error_code rm;
            fs::remove(it->path(), rm);
            continue;
        }
        // the generation before the source identity was part of the name: at most one per pack+geometry, matched exactly
        if (!legacy_name.empty() && base == legacy_name) {
            std::error_code rm;
            fs::remove(it->path(), rm);
            continue;
        }
        // and any scratch copy at all that nobody has touched for max_age_seconds is an abandoned pack: a write-triggered
        // sweep is enough for the geometry being written, but a pack that is never imported again would otherwise stay in
        // tmpfs (i.e. in RAM) for the life of the box (engine#1770)
        if (max_age_seconds > 0 && base.size() > stem.size() && base.compare(0, stem.size(), stem) == 0) {
            std::error_code tec;
            const auto      mtime = fs::last_write_time(it->path(), tec);
            if (!tec && now - mtime > std::chrono::seconds(max_age_seconds)) {
                std::error_code rm;
                fs::remove(it->path(), rm);
            }
        }
    }
}

}  // namespace onebit::npu::pack_import
