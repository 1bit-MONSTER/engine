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
// The FastFlowLM MTP-head container -> our Q4NX sidecar. See flm_head_import.h for the format facts and where
// they were measured.
#include "flm_head_import.h"

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

namespace onebit::npu::flm_head {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr uint64_t kCell = 4736;       // the head's quantised cell == our Q4_K chunk
constexpr int kRows = 32, kCols = 256; // one cell, in weights

struct Tensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    uint64_t lo = 0, hi = 0;
    uint64_t span() const { return hi - lo; }
    bool cells() const { return dtype == "I8" && shape.size() == 3 && shape[2] == int64_t(kCell); }
    bool bf16() const { return dtype == "BF16" && shape.size() == 1; }
};

struct Container {
    json metadata;
    uint64_t prologue = 0;
    std::vector<Tensor> tensors;
    uint64_t file_size = 0;
};

Container read_container(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    Container c;
    c.file_size = uint64_t(f.tellg());
    f.seekg(0);
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n == 0 || n > c.file_size) throw std::runtime_error(path + ": not a safetensors-shaped container");
    std::string header(n, '\0');
    f.read(header.data(), std::streamsize(n));
    c.prologue = 8 + n;
    json h = json::parse(header, nullptr, false);
    if (h.is_discarded() || !h.is_object()) throw std::runtime_error(path + ": header is not JSON");
    for (auto it = h.begin(); it != h.end(); ++it) {
        if (it.key() == "__metadata__") {
            c.metadata = it.value();
            continue;
        }
        const auto& t = it.value();
        if (!t.is_object() || !t.contains("shape") || !t.contains("data_offsets")) continue;
        Tensor x;
        x.name = it.key();
        x.dtype = t.value("dtype", "");
        for (const auto& d : t.at("shape")) x.shape.push_back(d.get<int64_t>());
        const auto off = t.at("data_offsets");
        x.lo = off.at(0).get<uint64_t>();
        x.hi = off.at(1).get<uint64_t>();
        c.tensors.push_back(std::move(x));
    }
    return c;
}

std::vector<uint8_t> read_at(const std::string& path, uint64_t prologue, uint64_t lo, uint64_t len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(std::streamoff(prologue + lo));
    std::vector<uint8_t> b(static_cast<std::vector<uint8_t>::size_type>(len));
    f.read(reinterpret_cast<char*>(b.data()), std::streamsize(len));
    if (!f) throw std::runtime_error("short read at " + std::to_string(lo));
    return b;
}

}  // namespace

Report import_head(const std::string& container, const std::string& out_dir, Form form) {
    Report r;
    r.out_dir = out_dir;
    const Container c = read_container(container);
    const std::string source = c.metadata.value("source", std::string("FastFlowLM container"));

    // ── validate ─────────────────────────────────────────────────────────────
    uint64_t max_end = 0;
    for (const auto& t : c.tensors) {
        max_end = std::max(max_end, t.hi);
        if (t.cells()) {
            const uint64_t nrb = uint64_t(t.shape[0]), ncb = uint64_t(t.shape[1]);
            if (t.span() != nrb * ncb * kCell) {
                r.refused.push_back(t.name + ": " + std::to_string(t.span()) + " B is not " +
                                    std::to_string(nrb) + "x" + std::to_string(ncb) + " cells of 4736");
                continue;
            }
            r.dims.matrices++;
            r.dims.tiles += int64_t(nrb * ncb);
            r.dims.tile_bytes += int64_t(t.span());
            r.manifest.push_back(t.name + " cells " + std::to_string(nrb) + "x" + std::to_string(ncb) +
                                 " weights " + std::to_string(nrb * kRows) + "x" + std::to_string(ncb * kCols));
        } else if (t.bf16()) {
            const uint64_t n = uint64_t(t.shape[0]);
            if (t.span() != n * 2) {
                r.refused.push_back(t.name + ": BF16 span " + std::to_string(t.span()) + " is not " +
                                    std::to_string(n) + " x 2");
                continue;
            }
            r.dims.norms++;
            r.dims.hidden = std::max(r.dims.hidden, int(n));   // the widest norm vector is the hidden width
            r.manifest.push_back(t.name + " bf16 " + std::to_string(n));
        } else {
            r.refused.push_back(t.name + ": dtype " + t.dtype + " with shape " + json(t.shape).dump() +
                                " is not a 4736-byte cell or a bf16 vector");
        }
    }
    if (c.prologue + max_end != c.file_size) {
        r.refused.push_back("header describes " + std::to_string(max_end) + " data bytes after a " +
                            std::to_string(c.prologue) + "-byte prologue, file is " +
                            std::to_string(c.file_size));
    }
    if (!r.refused.empty()) return r;   // nothing written: same contract as the OGA importer

    // ── write ────────────────────────────────────────────────────────────────
    // Our container keeps the same tensor names, dtype strings and shapes: the payload is unchanged, which is
    // the whole point (FLM's cell is our Q4_K chunk). The tensors are re-laid out contiguously in header order.
    fs::create_directories(out_dir);
    json table = json::object();
    json man = json::array();
    std::vector<std::pair<std::string, std::vector<uint8_t>>> blobs;
    uint64_t off = 0;
    const uint64_t out_cell = form == Form::Q4_1 ? uint64_t(q4nx::Chunk::Q4_1) : kCell;
    for (const auto& t : c.tensors) {
        auto bytes = read_at(container, c.prologue, t.lo, t.span());
        if (form == Form::Q4_1 && t.cells()) {
            // our mode-1 chunk: 256 bf16 S, 256 bf16 M, 4096 code bytes. q4k_to_q4_1 rounds the two products
            // (S*scale, M*min) to bf16 and nothing else; the report measures exactly that.
            const size_t cells = size_t(t.span() / kCell);
            std::vector<uint8_t> conv(cells * size_t(out_cell));
            for (size_t i = 0; i < cells; ++i)
                q4nx::q4k_to_q4_1(bytes.data() + i * size_t(kCell), conv.data() + i * size_t(out_cell));
            // the factorisation error against this container's own decode, per value
            std::vector<float> a(size_t(kRows) * kCols), b(a.size());
            const size_t sample = std::min<size_t>(cells, 16);
            for (size_t i = 0; i < sample; ++i) {
                q4nx::dequant_tile(q4nx::Chunk::Q4K, bytes.data() + i * size_t(kCell), a.data(), kCols);
                q4nx::dequant_tile(q4nx::Chunk::Q4_1, conv.data() + i * size_t(out_cell), b.data(), kCols);
                for (size_t v = 0; v < a.size(); ++v) {
                    r.values++;
                    if (a[v] == b[v]) r.exactly_equal++;
                    r.max_abs = std::max(r.max_abs, double(std::fabs(a[v] - b[v])));
                    r.rel_l1 += double(std::fabs(a[v] - b[v]));
                }
            }
            bytes = std::move(conv);
        }
        json shape = t.shape;
        if (form == Form::Q4_1 && t.cells()) shape[2] = int64_t(out_cell);
        table[t.name] = {{"dtype", t.dtype}, {"shape", shape}, {"data_offsets", {off, off + bytes.size()}}};
        man.push_back({{"name", t.name}, {"dtype", t.dtype}, {"shape", shape}, {"bytes", bytes.size()}});
        off += bytes.size();
        blobs.emplace_back(t.name, std::move(bytes));
    }
    if (r.values) r.rel_l1 /= double(r.values);   /* becomes the mean |delta| after the sum above */
    json header = json::object();
    header["__metadata__"] = {{"format", "q4nx"},
                              {"layout_version", 1},
                              {"source", source},
                              {"converter", "1bit npu-pack flm-head"},
                              {"cell_kind", form == Form::Q4_1 ? "q4_1" : "q4_k"},
                              {"cell_bytes", out_cell},
                              {"note", "tiles copied unchanged from the FastFlowLM head container"}};
    for (auto it = table.begin(); it != table.end(); ++it) header[it.key()] = it.value();
    std::string text = header.dump();
    text.append((64 - text.size() % 64) % 64, ' ');
    const std::string out_path = (fs::path(out_dir) / "model.q4nx").string();
    {
        std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + out_path);
        const uint64_t len = text.size();
        f.write(reinterpret_cast<const char*>(&len), 8);
        f.write(text.data(), std::streamsize(len));
        for (const auto& [name, b] : blobs) f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    }

    // ── self-check: decode the first tile of every matrix from the source and from the copy, bit for bit ──
    const Container back = read_container(out_path);
    for (const auto& t : back.tensors) {
        if (!(t.dtype == "I8" && t.shape.size() == 3 && t.shape[2] == int64_t(out_cell))) continue;
        const size_t cell = size_t(out_cell);
        const auto src = read_at(container, c.prologue, 0 + [&] {
            for (const auto& s : c.tensors)
                if (s.name == t.name) return s.lo;
            return uint64_t(0);
        }(), cell);
        const auto copy = read_at(out_path, back.prologue, t.lo, cell);
        std::vector<float> a(size_t(kRows) * kCols), b(a.size());
        // Q4K: the copy must reproduce the source bit for bit. Q4_1: it must reproduce it within the bf16
        // factorisation the report quantifies, so "differing" there means "not bit-identical", by design.
        const auto kind = form == Form::Q4_1 ? q4nx::Chunk::Q4_1 : q4nx::Chunk::Q4K;
        q4nx::dequant_tile(kind, src.data(), a.data(), kCols);
        q4nx::dequant_tile(kind, copy.data(), b.data(), kCols);
        r.checked_tiles++;
        if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) r.differing_tiles++;
    }

    // ── config.json + manifest.json: the facts the head's consumer must not guess ──
    {
        std::ofstream f(fs::path(out_dir) / "config.json", std::ios::trunc);
        json cfg = {{"model_type", "qwen3_5_mtp_head"},
                    {"source", source},
                    {"cell_kind", form == Form::Q4_1 ? "q4_1" : "q4_k"},
                    {"cell_bytes", out_cell},
                    {"form", form == Form::Q4_1 ? "mode-1 (q4_1 chunk)" : "q4_k copied from the source"},
                    {"self_check", form == Form::Q4_1
                                       ? "against this container's own Q4_K decode; the loss is the bf16 rounding "
                                         "of S*scale and M*min"
                                       : "byte-exact against the source container"},
                    {"tiles", {{"rows_per_tile", kRows}, {"cols_per_tile", kCols}}},
                    {"dequant", "w = q4 * (scale * S) + min * M   (additive min; measured against FLM's decoder)"},
                    {"tensors", man}};
        f << cfg.dump(1) << "\n";
    }
    return r;
}

int run(int argc, char** argv) {
    std::string container, out_dir, form_arg;
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-o") out_dir = i + 1 < argc ? argv[++i] : "";
        else if (a == "--form") form_arg = i + 1 < argc ? argv[++i] : "";
        else if (container.empty()) container = a;
        else {
            std::fprintf(stderr, "usage: flm-head-import <container.q4nx> -o <out dir> [--form q4_k|q4_1]\n");
            return 2;
        }
    }
    if (container.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: flm-head-import <container.q4nx> -o <out dir> [--form q4_k|q4_1]\n");
        return 2;
    }
    try {
        const Form form = form_arg == "q4_1" ? Form::Q4_1 : Form::Q4K;
        if (!form_arg.empty() && form_arg != "q4_1" && form_arg != "q4_k") {
            std::fprintf(stderr, "--form takes q4_k (default) or q4_1\n");
            return 2;
        }
        const Report r = import_head(container, out_dir, form);
        if (!r.refused.empty()) {
            std::fprintf(stderr, "refused: %zu tensor(s)/problem(s), nothing written\n", r.refused.size());
            for (const auto& why : r.refused) std::fprintf(stderr, "  %s\n", why.c_str());
            return 3;
        }
        std::printf("imported %lld matrices (%lld tiles, %.1f MiB) and %lld bf16 norms into %s\n",
                    r.dims.matrices, r.dims.tiles, double(r.dims.tile_bytes) / (1024 * 1024), r.dims.norms,
                    r.out_dir.c_str());
        std::printf("  form: %s\n", form_arg == "q4_1" ? "mode-1 q4_1 chunks (5,120 B)" :
                                                  "q4_k cells copied unchanged (4,736 B)");
        if (r.values)
            std::printf("  bf16 S/M factorisation: %lld of %lld values unchanged, max |delta| %.3e, "
                        "mean |delta| %.3e\n", r.exactly_equal, r.values, r.max_abs, r.rel_l1);
        std::printf("  self-check: %lld tiles decoded from the source and from the copy, %lld differing\n",
                    r.checked_tiles, r.differing_tiles);
        return r.differing_tiles == 0 ? 0 : 4;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flm-head: %s\n", e.what());
        return 1;
    }
}

}  // namespace onebit::npu::flm_head
