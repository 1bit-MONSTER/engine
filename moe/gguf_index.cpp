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
#include "gguf_index.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace onebit::moe {

namespace {

// ggml type id -> (elements per block, bytes per block), for the types GGUF files carry.
bool block_size(uint32_t type, uint64_t& blk, uint64_t& bytes) {
    switch (type) {
        case 0: blk = 1; bytes = 4; return true;      // F32
        case 1: blk = 1; bytes = 2; return true;      // F16
        case 2: blk = 32; bytes = 18; return true;    // Q4_0
        case 3: blk = 32; bytes = 20; return true;    // Q4_1
        case 6: blk = 32; bytes = 22; return true;    // Q5_0
        case 7: blk = 32; bytes = 24; return true;    // Q5_1
        case 8: blk = 32; bytes = 34; return true;    // Q8_0
        case 10: blk = 256; bytes = 84; return true;  // Q2_K
        case 11: blk = 256; bytes = 110; return true; // Q3_K
        case 12: blk = 256; bytes = 144; return true; // Q4_K
        case 13: blk = 256; bytes = 176; return true; // Q5_K
        case 14: blk = 256; bytes = 210; return true; // Q6_K
        case 15: blk = 256; bytes = 292; return true; // Q8_K
        case 16: blk = 256; bytes = 66; return true;  // IQ2_XXS
        case 17: blk = 256; bytes = 74; return true;  // IQ2_XS
        case 18: blk = 256; bytes = 98; return true;  // IQ3_XXS
        case 19: blk = 256; bytes = 50; return true;  // IQ1_S
        case 20: blk = 32; bytes = 18; return true;   // IQ4_NL
        case 21: blk = 256; bytes = 110; return true; // IQ3_S
        case 22: blk = 256; bytes = 82; return true;  // IQ2_S
        case 23: blk = 256; bytes = 136; return true; // IQ4_XS
        case 24: blk = 1; bytes = 1; return true;     // I8
        case 25: blk = 1; bytes = 2; return true;     // I16
        case 26: blk = 1; bytes = 4; return true;     // I32
        case 27: blk = 1; bytes = 8; return true;     // I64
        case 28: blk = 1; bytes = 8; return true;     // F64
        case 29: blk = 256; bytes = 56; return true;  // IQ1_M
        case 30: blk = 1; bytes = 2; return true;     // BF16
        case 34: blk = 256; bytes = 54; return true;  // TQ1_0
        case 35: blk = 256; bytes = 66; return true;  // TQ2_0
        case 39: blk = 32; bytes = 17; return true;   // MXFP4
        default: return false;
    }
}

// Every length and count read from the file is checked against the bytes left in it, and the
// reader only ever moves forward, so a crafted header ends in an error, not a hang or a huge
// allocation.
struct Reader {
    FILE* f;
    uint64_t size = 0;  // the file's size
    uint64_t left() const {
        const long at = std::ftell(f);
        return at < 0 || uint64_t(at) > size ? 0 : size - uint64_t(at);
    }
    void raw(void* p, size_t n) {
        if (std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated GGUF header");
    }
    void skip(uint64_t n) {
        if (n > left() || std::fseek(f, long(n), SEEK_CUR) != 0) throw std::runtime_error("truncated GGUF header");
    }
    template <class T> T get() { T v; raw(&v, sizeof v); return v; }
    uint64_t str_len() {
        const uint64_t n = get<uint64_t>();
        if (n > left()) throw std::runtime_error("bad GGUF string length");
        return n;
    }
    // Keys and tensor names are read; other strings (a tokenizer can be megabytes) are skipped.
    std::string str() {
        const uint64_t n = str_len();
        if (n > (uint64_t(1) << 20)) throw std::runtime_error("GGUF key or tensor name longer than 1 MiB");
        std::string s(n, '\0');
        raw(s.data(), s.size());
        return s;
    }
    void skip_value(uint32_t type) {
        static const size_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type == 8) { skip(str_len()); return; }
        if (type == 9) {
            const uint32_t et = get<uint32_t>();
            const uint64_t n = get<uint64_t>();
            if (et == 8) {
                if (n > left() / 8) throw std::runtime_error("bad GGUF array length");
                for (uint64_t i = 0; i < n; ++i) skip(str_len());
                return;
            }
            if (et >= 13 || sizes[et] == 0) throw std::runtime_error("bad GGUF array type");
            if (n > left() / sizes[et]) throw std::runtime_error("bad GGUF array length");
            skip(sizes[et] * n);
            return;
        }
        if (type >= 13 || sizes[type] == 0) throw std::runtime_error("bad GGUF value type");
        skip(sizes[type]);
    }
};

std::vector<std::string> shard_names(const std::string& path) {
    const auto pos = path.find("-00001-of-");
    if (pos == std::string::npos) return {path};
    const int n = std::atoi(path.c_str() + pos + 10);
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char b[32];
        std::snprintf(b, sizeof b, "-%05d-of-", i);
        out.push_back(path.substr(0, pos) + b + path.substr(pos + 10));
    }
    return out;
}

}  // namespace

GgufIndex GgufIndex::open(const std::string& path) {
    GgufIndex idx;
    idx.files = shard_names(path);
    for (int fi = 0; fi < (int) idx.files.size(); ++fi) {
        FILE* f = std::fopen(idx.files[fi].c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + idx.files[fi]);
        Reader r{f};
        try {
            if (std::fseek(f, 0, SEEK_END) != 0 || std::ftell(f) < 0) throw std::runtime_error("cannot read " + idx.files[fi]);
            r.size = uint64_t(std::ftell(f));
            std::rewind(f);
            if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error(idx.files[fi] + " is not a GGUF file");
            if (r.get<uint32_t>() < 2) throw std::runtime_error("GGUF v1 is not supported");
            const uint64_t n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
            // a KV entry takes at least 12 bytes and a tensor entry at least 24
            if (n_kv > r.left() / 12 || n_tensors > r.left() / 24) throw std::runtime_error("bad GGUF header counts");
            uint64_t alignment = 32;
            for (uint64_t i = 0; i < n_kv; ++i) {
                const std::string key = r.str();
                const uint32_t type = r.get<uint32_t>();
                if (key == "general.alignment" && type == 4) {
                    alignment = r.get<uint32_t>();
                    if (alignment == 0 || (alignment & (alignment - 1)))
                        throw std::runtime_error("bad GGUF general.alignment " + std::to_string(alignment));
                } else {
                    r.skip_value(type);
                }
            }
            const size_t first = idx.tensors.size();
            for (uint64_t i = 0; i < n_tensors; ++i) {
                GgufTensor t;
                t.name = r.str();
                const uint32_t nd = r.get<uint32_t>();
                if (nd > 4) throw std::runtime_error("bad GGUF dimension count in " + t.name);
                for (uint32_t d = 0; d < nd; ++d) t.ne.push_back(r.get<uint64_t>());
                t.type = r.get<uint32_t>();
                t.offset = r.get<uint64_t>();
                t.file = fi;
                uint64_t blk, bb;
                if (!block_size(t.type, blk, bb)) throw std::runtime_error("unknown ggml type " + std::to_string(t.type) + " in " + t.name);
                uint64_t n = 1;
                for (uint64_t v : t.ne) {
                    if (v != 0 && n > UINT64_MAX / v) throw std::runtime_error("bad GGUF tensor shape for " + t.name);
                    n *= v;
                }
                t.bytes = n / blk * bb;
                idx.tensors.push_back(std::move(t));
            }
            const uint64_t data = (uint64_t(std::ftell(f)) + alignment - 1) / alignment * alignment;
            for (size_t i = first; i < idx.tensors.size(); ++i) idx.tensors[i].offset += data;
        } catch (...) {
            std::fclose(f);
            throw;
        }
        std::fclose(f);
    }
    // routed experts: blk.N.ffn_{gate,up,down,gate_up}_exps.weight, 3-D with the expert last
    for (const auto& t : idx.tensors) {
        if (t.name.rfind("blk.", 0) != 0 || t.name.find("_exps.weight") == std::string::npos || t.ne.size() != 3) continue;
        if (t.ne[2] == 0 || t.ne[2] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("bad expert count in " + t.name);
        const int layer = std::atoi(t.name.c_str() + 4);
        ExpertPart p;
        p.tensor = t.name;
        p.file = t.file;
        p.base = t.offset;
        p.stride = t.bytes / t.ne[2];
        p.bytes = p.stride;
        idx.experts[layer].push_back(p);
        idx.n_expert = std::max<int>(idx.n_expert, (int) t.ne[2]);
    }
    for (auto& [l, parts] : idx.experts)
        std::sort(parts.begin(), parts.end(), [](const ExpertPart& a, const ExpertPart& b) { return a.tensor < b.tensor; });
    return idx;
}

uint64_t GgufIndex::expert_bytes(int layer) const {
    auto it = experts.find(layer);
    if (it == experts.end()) return 0;
    uint64_t n = 0;
    for (const auto& p : it->second) n += p.bytes;
    return n;
}

}  // namespace onebit::moe
