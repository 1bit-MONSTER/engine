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

// npu_moe_test
//
// The host side of a MoE container (npu/moe.cpp), against crafted directories written at run
// time so CI needs no 18 GB model:
//
//   * __metadata__.moe is read and checked against the tensors it names;
//   * every expert's chunk pointer lands exactly where the metadata says, one expert after the
//     other with no gap;
//   * a plausible-looking but WRONG list is refused — an offset past the tensor, a gap between
//     experts, an expert count that disagrees with the tensor, a byte count that is not whole
//     5120-byte chunks — because a kernel must never be handed a list that is only nearly right;
//   * the router's top-k is the model's own: descending, ties to the lower id, renormalised to
//     sum 1, and it reproduces a hand-computed softmax.
#include "moe.h"

#include "model.h"
#include "pack.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

using namespace onebit::npu;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

void check_near(float got, float want, float tol, const std::string& what) {
    const bool ok = std::fabs(got - want) <= tol;
    std::printf("%s %s (got %.7f, want %.7f)\n", ok ? "ok  " : "FAIL", what.c_str(), got, want);
    if (!ok) ++failures;
}

// --- a crafted MoE model directory -----------------------------------------------------------

struct Crafted {
    int experts = 4, rows = 768, cols = 2048, tiles = 192;  // 24 tile rows x 8 tile cols
    int hidden = 2048;
    // router logits we want for a given h, by construction below
    bool break_offset = false, break_gap = false, break_bytes = false, break_experts = false;
    bool no_metadata = false;
};

fs::path write_dir(const Crafted& m, const std::string& tag) {
    fs::path dir = fs::temp_directory_path() / ("npu_moe_test_" + std::to_string(::getpid()) + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    const uint64_t chunk = uint64_t(m.tiles) * kTileBytes;
    const uint64_t stack_bytes = chunk * uint64_t(m.experts);
    const uint64_t router_bytes = uint64_t(m.experts) * uint64_t(m.hidden) * 2;

    json table;
    json tensors;
    std::vector<std::pair<std::string, uint64_t>> blobs;
    uint64_t off = 0;
    auto add = [&](const std::string& name, std::vector<int64_t> shape, const std::string& dtype,
                   uint64_t bytes) {
        table[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {off, off + bytes}}};
        blobs.emplace_back(name, bytes);
        off += bytes;
    };
    for (int layer = 0; layer < 1; ++layer) {
        for (const char* proj : {kExpertGate, kExpertUp, kExpertDown}) {
            std::vector<uint64_t> offs;
            for (int e = 0; e < m.experts; ++e) {
                uint64_t o = uint64_t(e) * chunk;
                if (m.break_offset && e == m.experts - 1) o += kTileBytes;      // past the end
                if (m.break_gap && e == m.experts - 2) o += 64;  // a gap inside the tensor
                offs.push_back(o);
            }
            const std::string name = "model.layers.0." + std::string(proj);
            // the tile-row count a stack declares: experts x rows/32, cast before the multiply
            const int64_t tile_rows = int64_t(m.experts) * int64_t(m.rows / 32);
            add(name, {tile_rows, m.cols / 256, 5120}, "I8", stack_bytes);
            tensors[name] = {{"experts", m.break_experts ? m.experts + 1 : m.experts},
                             {"rows", m.rows},
                             {"cols", m.cols},
                             {"tile_rows", m.rows / 32},
                             {"tile_cols", m.cols / 256},
                             {"tiles", m.tiles},
                             {"bytes", m.break_bytes ? chunk + 1 : chunk},
                             {"offsets", offs}};
        }
        add("model.layers.0.mlp.gate.weight", {m.experts, m.hidden}, "BF16", router_bytes);
    }
    add("model.embed_tokens.weight", {512, m.hidden}, "BF16", uint64_t(512) * m.hidden * 2);
    if (!m.no_metadata) {
        table["__metadata__"] = {{"format", "q4nx"},
                                 {"converter", "tests/npu_moe_test.cpp (crafted)"},
                                 {"moe", {{"arch", "qwen3moe"},
                                          {"experts", m.experts},
                                          {"used", 2},
                                          {"norm_topk_prob", true},
                                          {"tensors", tensors}}}};
    } else {
        table["__metadata__"] = {{"format", "q4nx"}};
    }
    const std::string header = table.dump();
    std::ofstream f(dir / "model.q4nx", std::ios::binary);
    const uint64_t hlen = header.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(header.data(), std::streamsize(header.size()));
    // Each blob gets its own recognisable byte so a wrong pointer is visible, not plausible.
    uint8_t fill = 0;
    for (auto& [name, bytes] : blobs) {
        std::vector<char> buf(size_t(bytes), char(fill++));
        f.write(buf.data(), std::streamsize(bytes));
    }
    f.close();
    std::ofstream c(dir / "config.json");
    c << json{{"hidden_size", m.hidden},        {"num_hidden_layers", 1},
              {"num_attention_heads", 2},       {"num_key_value_heads", 1},
              {"head_dim", 128},                {"intermediate_size", 768},
              {"model_type", "qwen3moe"},       {"rope_theta", 1e7},
              {"rms_norm_eps", 1e-6},           {"num_experts", m.experts},
              {"num_experts_per_tok", 2},       {"norm_topk_prob", true}}
                 .dump();
    return dir;
}

// The router weights for the crafted model: expert e's logit for h = ones is set by making the
// router rows constant, so the softmax is computable by hand.
json router_rows(int experts, int hidden, const std::vector<float>& logits) {
    std::vector<uint16_t> w(size_t(experts) * size_t(hidden));
    for (int e = 0; e < experts; ++e)
        for (int k = 0; k < hidden; ++k)
            w[size_t(e) * hidden + k] = f32_to_bf16(logits[size_t(e)] / float(hidden));
    json out = json::array();
    for (size_t i = 0; i < w.size(); ++i) out.push_back(w[i]);
    return out;
}

std::string thrown(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// Against a REAL container when one is given: npu_moe_test <model dir> <gate.json>
// gate.json comes from tools/moe_expert_chunks.py gate (A_f32 + the router's own answer), so this
// checks the C++ router against the reference on real weights instead of only on arithmetic.
int real_model(const char* dir, const char* gate) {
    std::ifstream f(gate);
    if (!f) {
        std::printf("FAIL cannot read %s\n", gate);
        return 1;
    }
    json g;
    f >> g;
    const Model model(dir);
    const Moe moe(model);
    std::printf("ok   %s: %d experts, top-%d, norm_topk=%d\n", dir, moe.dims().experts,
                moe.dims().used, int(moe.dims().norm_topk));
    std::vector<float> h = g["A_f32"].get<std::vector<float>>();
    const auto want_ids = g["router_top"].get<std::vector<int>>();
    const auto want_w = g["router_weights"].get<std::vector<float>>();
    std::vector<int> ids(want_ids.size());
    std::vector<float> w(want_ids.size());
    moe.router_topk(h.data(), g.value("layer", 0), int(ids.size()), ids.data(), w.data());
    bool same_ids = true;
    float worst = 0.0f;
    std::printf("     c++  ids:");
    for (size_t i = 0; i < ids.size(); ++i) std::printf(" %d", ids[i]);
    std::printf("\n     py   ids:");
    for (int id : want_ids) std::printf(" %d", id);
    std::printf("\n");
    for (size_t i = 0; i < ids.size(); ++i) {
        same_ids = same_ids && ids[i] == want_ids[i];
        worst = std::max(worst, std::fabs(w[i] - want_w[i]));
    }
    if (!same_ids) ++failures;
    if (worst > 1e-5f) ++failures;
    std::printf("%s the C++ router picks the reference's experts (max |dw| %.2e)\n",
                same_ids && worst <= 1e-5f ? "ok  " : "FAIL", worst);
    const ExpertChunks c = moe.expert(g.value("layer", 0), kExpertGate, want_ids[0]);
    const Tensor& t = model.layer(g.value("layer", 0), kExpertGate);
    const bool inside = c.data >= t.data && c.data + c.bytes <= t.data + t.bytes;
    if (!inside) ++failures;
    std::printf("%s expert %d's %llu B chunk lies inside %s (%llu B)\n", inside ? "ok  " : "FAIL",
                want_ids[0], (unsigned long long)c.bytes, kExpertGate,
                (unsigned long long)t.bytes);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 3) return real_model(argv[1], argv[2]);
    const Crafted base;
    // 1. a well-formed directory: the metadata checks out and every expert pointer is exact.
    {
        const fs::path dir = write_dir(base, "_good");
        const Model model(dir.string());
        const Moe moe(model);
        check(moe.dims().experts == base.experts && moe.dims().used == 2 && moe.dims().norm_topk,
              "metadata: 4 experts, top-2, norm_topk");
        const ExpertChunks e0 = moe.expert(0, kExpertGate, 0);
        const ExpertChunks e3 = moe.expert(0, kExpertGate, 3);
        check(e0.bytes == uint64_t(base.tiles) * kTileBytes && e0.tiles == base.tiles,
              "expert 0: 192 chunks of 5120 B");
        check(e3.data - e0.data == 3 * int64_t(e0.bytes), "expert 3 is 3 chunks along from expert 0");
        const ExpertChunks d0 = moe.expert(0, kExpertDown, 0);
        check(d0.rows == base.rows && d0.cols == base.cols, "down: same geometry as gate here");
        check(thrown([&] { moe.expert(0, kExpertGate, 4); }).size() > 0,
              "expert id past the end is refused");
        check(thrown([&] { moe.expert(0, "mlp.experts.nope.weight", 0); }).size() > 0,
              "an unknown projection is refused");
        std::remove((dir / "model.q4nx").c_str());
        std::remove((dir / "config.json").c_str());
    }
    // 2. the router: descending, ties to the lower id, renormalised, and the softmax is right.
    {
        const fs::path dir = write_dir(base, "_router");
        // rewrite the router rows with known logits: 1.0, 3.0, 3.0, 0.0 over h = ones
        std::fstream in(dir / "model.q4nx", std::ios::binary | std::ios::in | std::ios::out);
        uint64_t hlen = 0;
        in.read(reinterpret_cast<char*>(&hlen), 8);
        std::string header(size_t(hlen), '\0');
        in.read(header.data(), std::streamsize(hlen));
        json table = json::parse(header);
        const auto off = table["model.layers.0.mlp.gate.weight"]["data_offsets"];
        const auto rows = router_rows(base.experts, base.hidden, {1.0f, 3.0f, 3.0f, 0.0f});
        std::vector<uint16_t> bits;
        bits.reserve(rows.size());
        for (const auto& v : rows) bits.push_back(v.get<uint16_t>());
        in.seekg(std::streamoff(8 + hlen + off[0].get<uint64_t>()));
        in.write(reinterpret_cast<const char*>(bits.data()), std::streamsize(bits.size() * 2));
        in.close();

        const Model model(dir.string());
        const Moe moe(model);
        std::vector<float> h(size_t(base.hidden), 1.0f);
        std::vector<float> p(size_t(base.experts)), logits(size_t(base.experts));
        moe.router_probs(h.data(), 0, p.data(), logits.data());
        float sum = 0;
        for (float v : p) sum += v;
        check_near(sum, 1.0f, 1e-5f, "router softmax sums to 1");
        // softmax({1,3,3,0}) = e^-2, e^0, e^0, e^-3 normalised
        const double e2 = std::exp(-2.0), e0 = 1.0, e3 = std::exp(-3.0);
        const double z = e2 + e0 + e0 + e3;
        check_near(p[0], float(e2 / z), 1e-5f, "router expert 0 probability");
        check_near(p[1], float(e0 / z), 1e-5f, "router expert 1 probability");

        int ids[2] = {-1, -1};
        float w[2] = {0, 0};
        moe.router_topk(h.data(), 0, 2, ids, w);
        check(ids[0] == 1 && ids[1] == 2, "top-2 is experts 1 and 2, ties to the lower id first");
        check_near(w[0] + w[1], 1.0f, 1e-6f, "the top-2 weights are renormalised to sum 1");
        check_near(w[0], 0.5f, 1e-6f, "equal logits give equal weights");
        check(thrown([&] { moe.router_topk(h.data(), 0, 3, ids, w); }).size() > 0,
              "k above the model's own top-k is refused");
        std::remove((dir / "model.q4nx").c_str());
        std::remove((dir / "config.json").c_str());
    }
    // 3. plausible-looking wrong lists are refused, before any kernel sees them.
    {
        Crafted bad;
        bad.break_offset = true;
        const fs::path d1 = write_dir(bad, "_off");
        check(thrown([&] { const Moe m{Model(d1.string())}; }).find("ends past the tensor") !=
                  std::string::npos,
              "an expert offset past the tensor is refused");
        Crafted gap;
        gap.break_gap = true;
        const fs::path d2 = write_dir(gap, "_gap");
        check(thrown([&] { const Moe m{Model(d2.string())}; }).find("not contiguous") !=
                  std::string::npos,
              "a gap between experts is refused");
        Crafted bytes;
        bytes.break_bytes = true;
        const fs::path d3 = write_dir(bytes, "_bytes");
        check(thrown([&] { const Moe m{Model(d3.string())}; }).find("whole number of chunks") !=
                  std::string::npos,
              "a byte count that is not whole chunks is refused");
        Crafted exps;
        exps.break_experts = true;
        const fs::path d4 = write_dir(exps, "_exps");
        check(thrown([&] { const Moe m{Model(d4.string())}; }).length() > 0,
              "an expert count that disagrees with the tensor is refused");
        Crafted none;
        none.no_metadata = true;
        const fs::path d5 = write_dir(none, "_none");
        check(thrown([&] { const Moe m{Model(d5.string())}; }).find("__metadata__.moe") !=
                  std::string::npos,
              "a directory with no MoE metadata is refused");
        for (const fs::path& d : {d1, d2, d3, d4, d5}) {
            std::remove((d / "model.q4nx").c_str());
            std::remove((d / "config.json").c_str());
            fs::remove_all(d);
        }
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
