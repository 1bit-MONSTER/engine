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

// Memory admission for `1bit serve` (docs/serve.md, "Will it fit?"): before a llama-server starts,
// estimate what it will hold and refuse at once, with the numbers, when the box does not have it.
//
// ggml-hrx reports free memory == total, so llama.cpp's own --fit sees the whole GTT pool free
// whatever is loaded, and a load that cannot fit runs into an allocation failure or the box's OOM
// minutes later. The estimate follows tools/mem-budget/mem_budget.py (checked against HRX0 server
// logs) from the GGUF header alone, no weights read:
//   weights   the GGUF file sizes, every shard
//   KV        per attention layer: n_head_kv x (key + value length) x context x cache type bytes;
//             MLA (kv_lora_rank): kv_lora_rank + rope dims per token, no separate V; a hybrid
//             gated delta-net (qwen35*, qwen3next): one layer in full_attention_interval; per-layer
//             head_count_kv arrays: the layers with heads; sliding-window layers: the window
//   state     gated delta-net recurrent state, per sequence
//   compute   512 MiB (HRX measured 20-505 MiB)
//   extras    mmproj and draft / MTP files, and the KV of a draft model
// Refused only when that need is larger than what is free: min(GTT free, MemAvailable) on HRX
// (the GTT pool is the same DRAM), MemAvailable on the CPU. The HRX graph-program cache (cap 64,
// 2.2-8.7 GiB measured) and engine #315's 95 GiB zone only warn: they grow after load.
#pragma once

#include "sysinfo.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace onebit::admit {

constexpr uint64_t kMiB = 1ull << 20, kGiB = 1ull << 30;

// The metadata a size estimate needs: integer scalars, integer arrays and general.architecture.
struct GgufHeader {
    bool ok = false;
    std::string arch;
    std::map<std::string, long long> ints;
    std::map<std::string, std::vector<long long>> arrays;

    long long get(const std::string& key, long long dflt = -1) const {
        if (auto it = ints.find(arch + "." + key); it != ints.end()) return it->second;
        return dflt;
    }
    const std::vector<long long>* array(const std::string& key) const {
        auto it = arrays.find(arch + "." + key);
        return it == arrays.end() ? nullptr : &it->second;
    }
};

inline GgufHeader read_gguf_header(const std::string& path) {
    GgufHeader h;
    std::ifstream f(path, std::ios::binary);
    auto rd = [&](void* p, size_t n) { return static_cast<bool>(f.read(static_cast<char*>(p), std::streamsize(n))); };
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(&magic, 4) || magic != 0x46554747u || !rd(&version, 4) || version < 2 || !rd(&n_tensors, 8) || !rd(&n_kv, 8))
        return h;
    auto str = [&](std::string* out) {
        uint64_t n = 0;
        if (!rd(&n, 8) || n > (1u << 20)) return false;
        if (!out) return static_cast<bool>(f.ignore(static_cast<std::streamsize>(n)));
        out->resize(n);
        return rd(out->data(), n);
    };
    static const int size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    auto scalar = [](uint32_t t) { return t < 13 ? size[t] : -1; };
    auto is_int = [](uint32_t t) { return t <= 5 || t == 10 || t == 11; };
    auto as_int = [](uint32_t t, const unsigned char* b) {
        long long v = 0;
        const int n = size[t];
        std::memcpy(&v, b, size_t(n));
        const bool is_signed = t == 1 || t == 3 || t == 5 || t == 11;
        if (is_signed && n < 8 && (b[n - 1] & 0x80)) v -= 1LL << (8 * n);
        return v;
    };
    for (uint64_t i = 0; i < n_kv && i < 65536; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!str(&key) || !rd(&type, 4)) return h;
        if (type == 8) {
            std::string v;
            if (!str(key == "general.architecture" ? &v : nullptr)) return h;
            if (key == "general.architecture") h.arch = v;
        } else if (type == 9) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) return h;
            if (et == 8) {
                for (uint64_t j = 0; j < n; ++j)
                    if (!str(nullptr)) return h;
            } else if (scalar(et) > 0) {
                if (is_int(et) && n <= 4096) {
                    std::vector<long long> v(n);
                    unsigned char b[8];
                    for (uint64_t j = 0; j < n; ++j) {
                        if (!rd(b, size_t(scalar(et)))) return h;
                        v[j] = as_int(et, b);
                    }
                    h.arrays[key] = std::move(v);
                } else {
                    f.ignore(static_cast<std::streamsize>(n * uint64_t(scalar(et))));
                }
            } else {
                return h;
            }
        } else if (scalar(type) > 0) {
            unsigned char b[8] = {};
            if (!rd(b, size_t(scalar(type)))) return h;
            if (is_int(type)) h.ints[key] = as_int(type, b);
        } else {
            return h;
        }
        if (!f) return h;
    }
    h.ok = true;
    return h;
}

// every shard of a split GGUF (name-00001-of-00003.gguf), or the file itself
inline std::vector<std::string> gguf_shards(const std::string& path) {
    static const std::regex split(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, split)) return {path};
    std::vector<std::string> out;
    const int n = std::stoi(m[3]);
    for (int i = 1; i <= n; i++) {
        char part[32];
        std::snprintf(part, sizeof part, "-%05d-of-%05d.gguf", i, n);
        out.push_back(m[1].str() + part);
    }
    return out;
}

inline uint64_t file_bytes(const std::string& path) {
    uint64_t total = 0;
    for (const auto& p : gguf_shards(path)) {
        std::error_code ec;
        const auto n = std::filesystem::file_size(p, ec);
        if (!ec) total += n;
    }
    return total;
}

// bytes per element of a llama.cpp KV cache type
inline double cache_type_bytes(const std::string& t) {
    if (t == "f32") return 4;
    if (t == "q8_0") return 34.0 / 32;
    if (t == "q4_0" || t == "iq4_nl") return 18.0 / 32;
    if (t == "q4_1") return 20.0 / 32;
    if (t == "q5_0") return 22.0 / 32;
    if (t == "q5_1") return 24.0 / 32;
    return 2;  // f16, bf16
}

struct ModelNeed {
    std::string arch;
    long long n_ctx = 0;       // the context the estimate is for
    bool ctx_from_model = false;
    uint64_t weights = 0, kv = 0, state = 0;
};

// One model's weights, KV and recurrent state for a context of n_ctx tokens (0: the model's own,
// as llama-server takes it) and n_seq sequences.
inline ModelNeed model_need(const std::string& path, long long n_ctx, int n_seq, double k_bytes = 2, double v_bytes = 2) {
    ModelNeed m;
    m.weights = file_bytes(path);
    const GgufHeader h = read_gguf_header(path);
    m.arch = h.arch;
    if (!h.ok || h.arch.empty()) return m;
    if (n_ctx <= 0) {
        n_ctx = std::max(0LL, h.get("context_length", 4096));
        m.ctx_from_model = true;
    }
    m.n_ctx = n_ctx;
    n_seq = std::max(1, n_seq);
    const long long n_layer_all = std::max(0LL, h.get("block_count", 0));
    const long long n_layer = std::max(0LL, n_layer_all - std::max(0LL, h.get("nextn_predict_layers", 0)));
    const long long n_head = std::max(1LL, h.get("attention.head_count", 1));
    const long long n_embd = std::max(0LL, h.get("embedding_length", 0));
    const long long k_len = h.get("attention.key_length", n_embd / n_head);
    const long long v_len = h.get("attention.value_length", k_len);
    const long long kv_lora = h.get("attention.kv_lora_rank", 0);
    const long long swa = h.get("attention.sliding_window", 0);
    const long long tokens = swa > 0 ? std::min(n_ctx, swa + 512) : n_ctx;  // SWA layers keep a window

    // heads with a KV cache, summed over the layers that have one
    long long head_layers = 0;
    if (const auto* per_layer = h.array("attention.head_count_kv")) {
        for (long long i = 0; i < std::min<long long>(n_layer, (long long) per_layer->size()); i++) head_layers += std::max(0LL, (*per_layer)[size_t(i)]);
    } else {
        long long attn_layers = n_layer;
        const bool delta_net = h.arch == "qwen35" || h.arch == "qwen35moe" || h.arch == "qwen3next";
        if (delta_net) {
            const long long every = std::max(1LL, h.get("full_attention_interval", 4));
            attn_layers = n_layer / every;
        }
        head_layers = attn_layers * std::max(0LL, h.get("attention.head_count_kv", n_head));
        if (delta_net) {
            // gated delta-net state per recurrent layer and sequence (f32): the S matrix of each
            // value head plus the convolution window
            const long long d_state = std::max(0LL, h.get("ssm.state_size", 0));
            const long long d_inner = std::max(0LL, h.get("ssm.inner_size", 0));
            const long long groups = std::max(0LL, h.get("ssm.group_count", 0));
            const long long conv = std::max(1LL, h.get("ssm.conv_kernel", 4));
            const long long rec_layers = n_layer - attn_layers;
            const uint64_t per_layer = uint64_t(d_state * d_inner + (conv - 1) * (d_inner + 2 * groups * d_state)) * 4;
            m.state = per_layer * uint64_t(std::max(0LL, rec_layers)) * uint64_t(n_seq);
        }
    }
    if (kv_lora > 0) {
        // MLA caches the compressed latent and the rope part once per token, V is a view of it
        const long long rope = std::max(0LL, h.get("rope.dimension_count", 0));
        m.kv = uint64_t(double(head_layers) * double(kv_lora + rope) * double(tokens) * k_bytes);
    } else {
        m.kv = uint64_t(double(head_layers) * (double(k_len) * k_bytes + double(v_len) * v_bytes) * double(tokens));
    }
    return m;
}

struct Need {
    std::vector<std::string> parts;   // "weights 4.7", for the message
    uint64_t total = 0;
    bool gpu = false;                  // a backend on HRX: the GTT pool counts too
    long long n_ctx = 0;
    bool ctx_from_model = false;
};

// What one llama-server command line will hold, from its own arguments (after serve's flags and
// the recipes): -m, -c, -np, -ctk/-ctv, -md (a draft model), --mmproj, --device HRX0.
inline void add_llama_server(Need& need, const std::vector<std::string>& argv, uint64_t compute = 512 * kMiB) {
    auto arg = [&](std::initializer_list<const char*> names) -> std::string {
        for (size_t i = 0; i + 1 < argv.size(); i++)
            for (const char* n : names)
                if (argv[i] == n) return argv[i + 1];
        return "";
    };
    const std::string model = arg({"-m", "--model"});
    if (model.empty()) return;
    const std::string c = arg({"-c", "--ctx-size"}), np = arg({"-np", "--parallel"});
    const long long n_ctx = c.empty() ? 0 : std::atoll(c.c_str());
    const int n_seq = np.empty() ? 4 : std::max(1, std::atoi(np.c_str()));   // llama-server's own default: 4 slots
    const double kb = cache_type_bytes(arg({"-ctk", "--cache-type-k"})), vb = cache_type_bytes(arg({"-ctv", "--cache-type-v"}));
    const bool gpu = arg({"--device", "-dev"}).rfind("HRX", 0) == 0;
    need.gpu = need.gpu || gpu;
    // --fit (llama-server's default) shrinks an unset context to what the device reports free; the
    // CPU reports what is free, so an unset context is judged at llama.cpp's floor (4096) there.
    // HRX0 reports its whole pool free, so it allocates the model's own context.
    long long ctx = n_ctx;
    if (ctx <= 0 && !gpu) ctx = 4096;
    const ModelNeed m = model_need(model, ctx, n_seq, kb, vb);
    if (need.n_ctx == 0) { need.n_ctx = m.n_ctx; need.ctx_from_model = m.ctx_from_model && n_ctx <= 0; }
    auto gib = [](uint64_t b) { char s[32]; std::snprintf(s, sizeof s, "%.1f", double(b) / double(kGiB)); return std::string(s); };
    const std::string name = std::filesystem::path(model).filename().string();
    need.parts.push_back(name + ": weights " + gib(m.weights) + " + KV/state " + gib(m.kv + m.state) + " + compute " + gib(compute));
    need.total += m.weights + m.kv + m.state + compute;
    const std::string draft = arg({"-md", "--model-draft"});
    if (!draft.empty()) {
        const ModelNeed d = model_need(draft, m.n_ctx, n_seq, kb, vb);
        need.parts.push_back("draft " + gib(d.weights + d.kv + d.state));
        need.total += d.weights + d.kv + d.state;
    }
    const std::string mmproj = arg({"--mmproj", "-mm"});
    if (!mmproj.empty()) {
        const uint64_t b = file_bytes(mmproj);
        need.parts.push_back("mmproj " + gib(b));
        need.total += b;
    }
}

struct Verdict {
    bool checked = false;   // the box said what it has (Linux)
    bool fits = true;
    uint64_t free = 0;
    std::string box;        // "GTT 20.1 of 120.0 GiB free, MemAvailable 40.1 GiB"
    std::vector<std::string> warnings;
};

// reserve: kept free for everything else (default 1 GiB)
inline Verdict judge(const Need& need, const sysinfo::Memory& mem, uint64_t reserve = kGiB) {
    Verdict v;
    if (!mem.has_meminfo) return v;
    v.checked = true;
    auto gib = [](uint64_t b) { char s[32]; std::snprintf(s, sizeof s, "%.1f", double(b) / double(kGiB)); return std::string(s); };
    v.free = mem.mem_available;
    v.box = "MemAvailable " + gib(mem.mem_available) + " GiB";
    if (need.gpu && mem.has_gtt) {
        const uint64_t gtt_free = mem.gtt_total > mem.gtt_used ? mem.gtt_total - mem.gtt_used : 0;
        v.free = std::min(v.free, gtt_free);
        v.box = "GTT " + gib(gtt_free) + " of " + gib(mem.gtt_total) + " GiB free, " + v.box;
    }
    v.fits = need.total + reserve <= v.free;
    if (v.fits && need.gpu && need.total + reserve + 4 * kGiB > v.free)
        v.warnings.push_back("little room left for the HRX graph-program cache (it grows to 2-9 GiB as prompt shapes vary)");
    if (v.fits && need.gpu && mem.has_gtt && mem.gtt_used + need.total >= 95 * kGiB)
        v.warnings.push_back("GTT use after this load (~" + gib(mem.gtt_used + need.total) +
                             " GiB) is in the zone where engine #315's faults were seen (>= 95 GiB)");
    return v;
}

}  // namespace onebit::admit
