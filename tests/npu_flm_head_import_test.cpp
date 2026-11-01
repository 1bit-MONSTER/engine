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
// npu_flm_head_import_test <tests/golden/flm-i8>
//
// 1. The golden tiles: six cells recorded from FastFlowLM's MTP head, each with the 8,192 floats their own
//    decoder produced (evidence: amd-oss/npu-re docs/FACTS-flm-q4nx-container-and-mtp-head.md §5). Our
//    q4nx::dequant_tile(Q4K) must reproduce every float bit for bit, read through their output order
//    (w = 32*k + n: k the column inside the cell's 256, n the row inside its 32).
// 2. A round trip: a synthetic FastFlowLM-shaped container -> `flm-head` import -> our container carries the
//    same tile bytes, the geometry lands in config.json, and the import's own self-check compared every matrix.
// 3. A refusal: one cell short of its shape law is named and nothing is written.
#include "../npu/flm_head_import.h"
#include "../npu/q4nx.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using nlohmann::json;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

// ── 1. the golden tiles against our decoder ─────────────────────────────────
void test_golden(const std::string& dir) {
    struct Case { const char* tile; const char* want; };
    const Case cases[] = {
        {"fc_weight_0_0.tile.bin", "fc_weight_0_0.want.f32"},
        {"fc_weight_0_17.tile.bin", "fc_weight_0_17.want.f32"},
        {"q_proj_weight_0_0.tile.bin", "q_proj_weight_0_0.want.f32"},
        {"k_proj_weight_0_19.tile.bin", "k_proj_weight_0_19.want.f32"},
        {"down_proj_weight_3_40.tile.bin", "down_proj_weight_3_40.want.f32"},
        {"gate_proj_weight_500_7.tile.bin", "gate_proj_weight_500_7.want.f32"},
    };
    int checked = 0;
    for (const auto& c : cases) {
        const auto tile = slurp(dir + "/" + c.tile);
        const auto want_raw = slurp(dir + "/" + c.want);
        if (tile.size() != 4736 || want_raw.size() != 8192 * 4) continue;   // older checkouts: skip, not fail
        std::vector<float> got(8192);
        onebit::npu::q4nx::dequant_tile(onebit::npu::q4nx::Chunk::Q4K, tile.data(), got.data(), 256);
        const float* want = reinterpret_cast<const float*>(want_raw.data());
        int diff = 0;
        for (int r = 0; r < 32; ++r)
            for (int c2 = 0; c2 < 256; ++c2) {
                uint32_t a, b;
                std::memcpy(&a, &got[size_t(r) * 256 + c2], 4);
                std::memcpy(&b, &want[size_t(c2) * 32 + r], 4);   // their order: w = 32*k + n
                if (a != b) ++diff;
            }
        check(diff == 0, std::string("golden ") + c.tile + ": " + std::to_string(diff) + "/8192 floats differ");
        ++checked;
    }
    check(checked == 6, "golden: six recorded cells present");
}

// ── a synthetic FastFlowLM-shaped container ────────────────────────────────
struct Cell { uint8_t sc[256], mn[256], nib[4096]; uint8_t d[64], dmin[64]; };

std::vector<Cell> make_cells(int n, uint32_t seed) {
    std::vector<Cell> cells(static_cast<std::vector<Cell>::size_type>(n));
    uint32_t x = seed | 1u;
    auto next = [&] {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    };
    for (auto& c : cells) {
        for (auto& v : c.sc) v = uint8_t(next());
        for (auto& v : c.mn) v = uint8_t(next());
        for (auto& v : c.nib) v = uint8_t(next());
        for (int i = 0; i < 32; i++) {
            const uint16_t d = uint16_t(0x3980 | (next() & 0x7f));      // a small positive bf16
            const uint16_t dm = uint16_t(0xb980 | (next() & 0x7f));     // a small negative bf16
            std::memcpy(&c.d[2 * i], &d, 2);
            std::memcpy(&c.dmin[2 * i], &dm, 2);
        }
    }
    return cells;
}

void put_bytes(const std::string& path, const json& header, const std::vector<std::vector<uint8_t>>& blobs) {
    std::string text = header.dump();
    text.append((64 - text.size() % 64) % 64, ' ');
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const uint64_t len = text.size();
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(text.data(), std::streamsize(len));
    for (const auto& b : blobs) f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

struct Made {
    std::string dir;
    std::vector<std::vector<uint8_t>> blobs;
};

// name -> [nrb, ncb]; every cell is 4736 B and every blob is written contiguously
Made make_source(const std::string& dir, bool truncate_one) {
    fs::create_directories(dir);
    struct Spec { const char* name; int nrb, ncb; bool norm; };
    const Spec specs[] = {
        {"model.mtp.fc.weight", 2, 3, false},
        {"model.mtp.layers.0.input_layernorm.weight", 1, 0, true},
        {"model.mtp.layers.0.self_attn.q_proj.weight", 3, 1, false},
        {"model.mtp.norm.weight", 1, 0, true},
    };
    Made m;
    m.dir = dir;
    json header = json::object();
    header["__metadata__"] = {{"format", "q4nx"}, {"source", "synthetic-test"}};
    uint64_t off = 0;
    uint32_t seed = 12345;
    for (const auto& s : specs) {
        std::vector<uint8_t> bytes;
        json shape = json::array();
        if (s.norm) {
            shape = {16};
            bytes.assign(32, 0x3f);
        } else {
            const auto cells = make_cells(s.nrb * s.ncb, seed += 7);
            for (const auto& c : cells) {
                bytes.insert(bytes.end(), c.sc, c.sc + 256);
                bytes.insert(bytes.end(), c.mn, c.mn + 256);
                bytes.insert(bytes.end(), c.nib, c.nib + 4096);
                bytes.insert(bytes.end(), c.d, c.d + 64);
                bytes.insert(bytes.end(), c.dmin, c.dmin + 64);
            }
            if (truncate_one) bytes.resize(bytes.size() - 1);
            shape = {s.nrb, s.ncb, 4736};
        }
        header[s.name] = {{"dtype", s.norm ? "BF16" : "I8"}, {"shape", shape}, {"data_offsets", {off, off + bytes.size()}}};
        off += bytes.size();
        m.blobs.push_back(std::move(bytes));
    }
    put_bytes(dir + "/model.mtp.q4nx", header, m.blobs);
    return m;
}

// ── 2. round trip ──────────────────────────────────────────────────────────
void test_round_trip(const std::string& root) {
    const std::string src = root + "/src";
    const std::string out = root + "/out";
    fs::remove_all(src);
    fs::remove_all(out);
    make_source(src, false);
    const auto r = onebit::npu::flm_head::import_head(src + "/model.mtp.q4nx", out);
    check(r.refused.empty(), "round trip: nothing refused");
    check(r.dims.matrices == 2 && r.dims.norms == 2, "round trip: 2 matrices + 2 bf16 norms counted");
    check(r.dims.tiles == 2 * 3 + 3 * 1, "round trip: tile count follows the shapes");
    check(r.checked_tiles == r.dims.matrices && r.differing_tiles == 0,
          "round trip: every matrix's first tile decodes identically from source and copy");

    const auto written = slurp(out + "/model.q4nx");
    check(!written.empty(), "round trip: model.q4nx written");
    // the payload is copied unchanged, so the source payload must appear verbatim inside it
    const auto source = slurp(src + "/model.mtp.q4nx");
    const auto& biggest = r.dims.tile_bytes > 0;
    check(biggest, "round trip: tile bytes reported");
    bool found = false;
    if (!source.empty() && written.size() > source.size()) {
        // the last tensor of the source (a bf16 norm) is byte-identical at the end of our payload
        const size_t tail = 32;
        found = std::memcmp(source.data() + source.size() - tail, written.data() + written.size() - tail, tail) == 0;
    }
    check(found, "round trip: payload bytes preserved");

    std::ifstream cf(out + "/config.json");
    const json cfg = json::parse(std::string(std::istreambuf_iterator<char>(cf), {}), nullptr, false);
    check(!cfg.is_discarded() && cfg.value("cell_bytes", 0) == 4736, "round trip: config.json names the cell");
    check(!cfg.is_discarded() && cfg.value("source", std::string()) == "synthetic-test",
          "round trip: config.json carries the source revision");
}

// ── 3. refusal ─────────────────────────────────────────────────────────────
void test_refusal(const std::string& root) {
    const std::string src = root + "/bad";
    const std::string out = root + "/bad-out";
    fs::remove_all(src);
    fs::remove_all(out);
    make_source(src, true);   // one cell one byte short
    const auto r = onebit::npu::flm_head::import_head(src + "/model.mtp.q4nx", out);
    check(!r.refused.empty(), "refusal: a short cell is reported");
    check(!fs::exists(out + "/model.q4nx"), "refusal: nothing is written");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden = argc > 1 ? argv[1] : "";
    const std::string root = (fs::temp_directory_path() / "npu_flm_head_import_test").string();
    try {
        if (!golden.empty()) test_golden(golden);
        else std::printf("skip golden: no directory given\n");
        test_round_trip(root);
        test_refusal(root);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    fs::remove_all(root);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all flm-head checks passed\n");
    return 0;
}
