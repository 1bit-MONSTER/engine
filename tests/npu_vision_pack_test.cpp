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
// npu_vision_pack_test
//
// The vision component reader (npu/vision_pack.h, engine#1788). The container is written here, small, with the same
// tensor names and the same shape relations as FastFlowLM's MiniCPM-V 4.7 `vision_weight.q4nx`, so each arm is one
// edit to a container that otherwise loads:
//   - it loads, and the geometry is the one the tensors hold;
//   - a config that is stale about an MLP width does not move a single size (the published packs are stale there:
//     4304 stated, 4608 held) and is reported;
//   - every way the file can disagree with itself, or with the config in a way no forward survives, is refused with
//     the tensor or field named.
// When ONEBIT_MINICPM_PACK names a published pack directory, its own numbers are checked too.
#include "../npu/vision_pack.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

using nlohmann::json;
namespace vis = onebit::npu::vision;

int g_fail = 0, g_arms = 0;
void check(bool ok, const std::string& what) {
    ++g_arms;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// The small tower: width 8 (2 heads x 4), 3 channels, 2-pixel patches, a 4 x 4 grid (image 8), 2 blocks with MLP 32,
// vit_merger MLP 64, projecting to a text model 6 wide.
struct Small {
    int H = 8, C = 3, P = 2, pos = 16, L = 2, I = 32, VI = 64, O = 6;
};

uint16_t bf16(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return uint16_t(b >> 16);
}

// name -> shape, in the order the payload is laid out
std::vector<std::pair<std::string, std::vector<int64_t>>> layout(const Small& s) {
    std::vector<std::pair<std::string, std::vector<int64_t>>> v;
    const std::string p = "model.visual.";
    const auto lin = [&](const std::string& n, int64_t r, int64_t c) {
        v.push_back({p + n + ".bias", {r}});
        v.push_back({p + n + ".weight", {r, c}});
    };
    const auto nrm = [&](const std::string& n, int64_t k) {
        v.push_back({p + n + ".bias", {k}});
        v.push_back({p + n + ".weight", {k}});
    };
    const int64_t H = s.H, M = 4 * s.H;
    for (int i = 0; i < s.L; ++i) {
        const std::string b = "blocks." + std::to_string(i) + ".";
        lin(b + "attn.proj", H, H);
        lin(b + "attn.qkv", 3 * H, H);
        lin(b + "mlp.linear_fc1", s.I, H);
        lin(b + "mlp.linear_fc2", H, s.I);
        nrm(b + "norm1", H);
        nrm(b + "norm2", H);
    }
    lin("merger.linear_fc1", M, M);
    lin("merger.linear_fc2", s.O, M);
    nrm("merger.norm", H);
    nrm("merger.pre_norm", M);
    v.push_back({p + "patch_embed.proj.bias", {H}});
    v.push_back({p + "patch_embed.proj.weight", {H, s.C, s.P, s.P}});
    v.push_back({p + "pos_embed.weight", {s.pos, H}});
    lin("vit_merger.attn.proj", H, H);
    lin("vit_merger.attn.qkv", 3 * H, H);
    lin("vit_merger.mlp.linear_fc1", s.VI, M);
    lin("vit_merger.mlp.linear_fc2", H, s.VI);
    nrm("vit_merger.norm1", H);
    nrm("vit_merger.pre_norm", M);
    return v;
}

json config_of(const Small& s) {
    json vc;
    vc["MINICPM_V_PATCH_SIZE"] = s.P;
    vc["MINICPM_V_IMAGE_SIZE"] = 8;
    vc["MINICPM_V_VISION_EMBED_DIM"] = s.H;
    vc["MINICPM_V_VISION_NUM_HEADS"] = 2;
    vc["MINICPM_V_VISION_HEAD_DIM"] = 4;
    vc["MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE"] = s.I;
    vc["MINICPM_V_VISION_NUM_POSITION_EMBEDDINGS"] = s.pos;
    vc["MINICPM_V_VISION_NUM_LAYERS"] = s.L;
    vc["MINICPM_V_VISION_OUT_HIDDEN_SIZE"] = s.O;
    vc["MINICPM_V_MERGER_HIDDEN_SIZE"] = 4 * s.H;
    vc["MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE"] = s.VI;
    json c;
    c["hidden_size"] = s.O;
    c["vision_model_weight"] = "vision_weight.q4nx";
    c["vision_config"] = vc;
    return c;
}

// Writes the container. `edit` sees the table (and may change the payload size) before it is serialised. Element e of
// the tensor at payload offset `off` holds bf16(0.25 * ((off/2 + e) % 64) - 4): readable back by position.
void write(const std::string& path, const Small& s, const std::function<void(json&, std::vector<uint8_t>&)>& edit = {}) {
    json table = json::object();
    std::vector<uint8_t> payload;
    for (const auto& [name, shape] : layout(s)) {
        uint64_t n = 1;
        for (int64_t d : shape) n *= uint64_t(d);
        const uint64_t off = payload.size();
        for (uint64_t e = 0; e < n; ++e) {
            const uint16_t h = bf16(0.25f * float((off / 2 + e) % 64) - 4.0f);
            payload.push_back(uint8_t(h & 0xff));
            payload.push_back(uint8_t(h >> 8));
        }
        table[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {off, off + n * 2}}};
    }
    table["__metadata__"] = {{"source_file", "crafted.gguf"}, {"converter", "npu_vision_pack_test"}};
    if (edit) edit(table, payload);
    const std::string head = table.dump();
    const uint64_t len = head.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(head.data(), std::streamsize(head.size()));
    f.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size()));
}

const char* const kPath = "npu_vision_pack_test.q4nx";

// Loads the crafted file; "" when it loads, else what it threw.
std::string load(const std::string& config) {
    try {
        const vis::Pack p(kPath, config);
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

void refused(const std::string& arm, const std::string& config, const std::vector<std::string>& must_name) {
    const std::string why = load(config);
    check(!why.empty(), arm + ": refused");
    for (const std::string& m : must_name)
        check(why.find(m) != std::string::npos, arm + ": the refusal names \"" + m + "\" (it says: " + why + ")");
}

void test_loads() {
    const Small s;
    write(kPath, s);
    const std::string cfg = config_of(s).dump();
    try {
        const vis::Pack p(kPath, cfg);
        const vis::Geometry& g = p.geometry();
        check(g.hidden == 8 && g.channels == 3 && g.patch == 2, "geometry: the patch embedding's own shape");
        check(g.layers == 2 && g.intermediate == 32, "geometry: two blocks, MLP 32");
        check(g.pos_rows == 16 && g.grid == 4, "geometry: 16 positions are a 4 x 4 grid");
        check(g.merge_hidden == 32 && g.vit_merger_intermediate == 64 && g.out_hidden == 6, "geometry: the two mergers");
        check(g.heads == 2 && g.head_dim == 4 && g.image_size == 8, "geometry: heads and image size come from the config");
        check(p.tensors().size() == size_t(2 * 12 + 23), "every tensor is placed: 12 per block and 23 outside them");
        check(p.unaccounted_bytes() == 0, "no payload byte is outside a tensor");
        check(p.notes().empty(), "a config that agrees leaves no note");
        check(p.metadata("source_file") == "crafted.gguf" && p.metadata("absent").empty(), "metadata is read by key");
        // exact values back, by position: the first tensor starts at payload offset 0
        const std::vector<float> b = p.f32("model.visual.blocks.0.attn.proj.bias");
        bool ok = b.size() == 8;
        for (size_t e = 0; ok && e < b.size(); ++e) ok = b[e] == 0.25f * float(e % 64) - 4.0f;
        check(ok, "f32(): BF16 values come back exactly");
        const vis::Tensor& t = p.tensor("model.visual.pos_embed.weight");
        const std::vector<float> pos = p.f32("model.visual.pos_embed.weight");
        check(pos.size() == 16 * 8 && pos[3] == 0.25f * float((t.offset / 2 + 3) % 64) - 4.0f,
              "f32(): a tensor in the middle of the payload reads from its own offset");
    } catch (const std::exception& e) {
        check(false, std::string("the crafted container loads (threw: ") + e.what() + ")");
    }
    // with no config nothing is cross-checked and the config-only fields stay 0
    try {
        const vis::Pack p(kPath, "");
        check(p.geometry().hidden == 8 && p.geometry().heads == 0 && p.geometry().image_size == 0,
              "no config: the tensor geometry stands, heads and image size are unknown");
    } catch (const std::exception& e) {
        check(false, std::string("loads without a config (threw: ") + e.what() + ")");
    }
}

// The published packs' situation: the config states smaller MLP widths than the tensors hold.
void test_stale_config_widths() {
    const Small s;
    write(kPath, s);
    json c = config_of(s);
    c["vision_config"]["MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE"] = 30;
    c["vision_config"]["MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE"] = 60;
    try {
        const vis::Pack p(kPath, c.dump());
        check(p.geometry().intermediate == 32 && p.geometry().vit_merger_intermediate == 64,
              "stale config: both widths are the tensors', not the config's");
        check(p.notes().size() == 2, "stale config: one note per overruled field");
        const std::string all = p.notes().size() == 2 ? p.notes()[0] + p.notes()[1] : "";
        check(all.find("VISION_MLP_INTERMEDIATE_SIZE 30") != std::string::npos && all.find("32 rows") != std::string::npos,
              "stale config: the note states the config's value and the tensor's");
        check(all.find("VIT_MERGER_INTERMEDIATE_SIZE 60") != std::string::npos && all.find("64 rows") != std::string::npos,
              "stale config: and the same for the vit_merger");
    } catch (const std::exception& e) {
        check(false, std::string("a config stale about an MLP width still loads (threw: ") + e.what() + ")");
    }
}

void test_refusals() {
    const Small s;
    const std::string cfg = config_of(s).dump();
    const std::string qkv0 = "model.visual.blocks.0.attn.qkv.weight";

    write(kPath, s, [&](json& t, std::vector<uint8_t>&) { t[qkv0]["dtype"] = "F32"; });
    refused("a tensor that is not BF16", cfg, {qkv0, "F32"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>&) { t[qkv0]["shape"] = {24, 9}; });
    refused("a shape that is not the tensor's byte length", cfg, {qkv0, "[24, 9]"});

    // the shape the tower expects, over one element fewer bytes: a forward would read the last element out of the
    // NEXT tensor. Only the extent-against-shape check can see this one.
    write(kPath, s, [&](json& t, std::vector<uint8_t>&) {
        const uint64_t a = t[qkv0]["data_offsets"][0], b = t[qkv0]["data_offsets"][1];
        t[qkv0]["data_offsets"] = {a, b - 2};
    });
    refused("an extent shorter than the shape", cfg, {qkv0, "spans 382 B", "384 B of BF16"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>& p) {
        const uint64_t a = t[qkv0]["data_offsets"][0];
        t[qkv0]["data_offsets"] = {a + p.size(), a + p.size() + 384};
    });
    refused("offsets past the payload", cfg, {qkv0, "outside the payload"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>&) {
        // norm2.bias moved onto norm1.bias: same length, so only the overlap check can see it
        t["model.visual.blocks.0.norm2.bias"]["data_offsets"] = t["model.visual.blocks.0.norm1.bias"]["data_offsets"];
    });
    refused("two tensors on the same bytes", cfg, {"overlaps", "model.visual.blocks.0.norm"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>&) { t.erase("model.visual.blocks.1.norm2.bias"); });
    refused("a missing tensor", cfg, {"has no tensor model.visual.blocks.1.norm2.bias"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>&) {
        json moved = json::object();
        for (auto it = t.begin(); it != t.end(); ++it) {
            std::string k = it.key();
            const std::string from = "model.visual.blocks.1.";
            if (k.compare(0, from.size(), from) == 0) k = "model.visual.blocks.2." + k.substr(from.size());
            moved[k] = it.value();
        }
        t = moved;
    });
    refused("a hole in the block indices", cfg, {"block 2", "no block 1"});

    write(kPath, s, [&](json& t, std::vector<uint8_t>& p) {
        const uint64_t off = p.size();
        p.resize(p.size() + 16, 0);
        t["model.visual.blocks.0.attn.gate.weight"] = {{"dtype", "BF16"}, {"shape", {8}}, {"data_offsets", {off, off + 16}}};
    });
    refused("a tensor the tower has no place for", cfg, {"model.visual.blocks.0.attn.gate.weight", "no place"});

    // one block with a different MLP width: its own extents are consistent, so only the cross-block shape check sees it
    write(kPath, s, [&](json& t, std::vector<uint8_t>& p) {
        const auto grow = [&](const std::string& n, std::vector<int64_t> shape) {
            uint64_t k = 2;
            for (int64_t d : shape) k *= uint64_t(d);
            const uint64_t off = p.size();
            p.resize(p.size() + k, 0);
            t[n] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {off, off + k}}};
        };
        grow("model.visual.blocks.1.mlp.linear_fc1.weight", {40, 8});
        grow("model.visual.blocks.1.mlp.linear_fc1.bias", {40});
        grow("model.visual.blocks.1.mlp.linear_fc2.weight", {8, 40});
    });
    refused("a block whose MLP is another width", "", {"model.visual.blocks.1.mlp.linear_fc1", "[32"});

    {
        Small odd;
        odd.pos = 15;
        write(kPath, odd);
        refused("a position table that is not a square grid", "", {"pos_embed", "15 rows"});
    }
    write(kPath, s, [&](json&, std::vector<uint8_t>&) {});
    {
        std::ofstream f(kPath, std::ios::binary | std::ios::trunc);
        const uint64_t len = 1 << 20;
        f.write(reinterpret_cast<const char*>(&len), 8);
        f.write("{}", 2);
    }
    refused("a header length past the end of the file", cfg, {"header runs past the end"});

    // ── the config against the tensors: refused where no forward survives it ──
    write(kPath, s);
    const auto with = [&](const char* key, int v) {
        json c = config_of(s);
        c["vision_config"][std::string("MINICPM_V_") + key] = v;
        return c.dump();
    };
    refused("config: heads that do not make the width", with("VISION_NUM_HEADS", 3), {"VISION_NUM_HEADS 3", "width 8"});
    refused("config: another width", with("VISION_EMBED_DIM", 16), {"VISION_EMBED_DIM 16", "8 wide"});
    refused("config: more blocks than the container holds", with("VISION_NUM_LAYERS", 3), {"VISION_NUM_LAYERS 3", "2 blocks"});
    refused("config: another patch size", with("PATCH_SIZE", 4), {"PATCH_SIZE 4"});
    refused("config: an image the position table does not cover", with("IMAGE_SIZE", 12), {"IMAGE_SIZE 12", "4 x 4"});
    refused("config: another position count", with("VISION_NUM_POSITION_EMBEDDINGS", 25), {"NUM_POSITION_EMBEDDINGS 25", "16 rows"});
    {
        json c = config_of(s);
        c["hidden_size"] = 12;
        refused("config: a text model of another width", c.dump(), {"hidden_size 12", "projects to 6"});
    }
}

// Pack(dir): the container name comes from config.json and must stay inside the pack.
void test_pack_dir_name() {
    const Small s;
    const std::string dir = "npu_vision_pack_test.dir";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    write(dir + "/tower.q4nx", s);
    json c = config_of(s);
    c["vision_model_weight"] = "tower.q4nx";
    { std::ofstream f(dir + "/config.json"); f << c.dump(); }
    try {
        const vis::Pack p(dir);
        check(p.geometry().layers == 2 && p.path() == dir + "/tower.q4nx", "Pack(dir): opens the file config.json names");
    } catch (const std::exception& e) {
        check(false, std::string("Pack(dir) opens the named container (threw: ") + e.what() + ")");
    }
    c["vision_model_weight"] = "../npu_vision_pack_test.q4nx";
    { std::ofstream f(dir + "/config.json"); f << c.dump(); }
    std::string why;
    try { const vis::Pack p(dir); } catch (const std::exception& e) { why = e.what(); }
    check(why.find("is not a file name") != std::string::npos, "Pack(dir): a name with a separator is refused, not followed");
    std::remove((dir + "/tower.q4nx").c_str());
    std::remove((dir + "/config.json").c_str());
    std::filesystem::remove(dir, ec);
}

// The published pack, when one is on the box: the numbers engine#1788's thread read by hand.
void test_published_pack(const std::string& dir) {
    try {
        const vis::Pack p(dir);
        const vis::Geometry& g = p.geometry();
        check(p.tensors().size() == 347, "published pack: 347 tensors, all placed");
        check(p.unaccounted_bytes() == 0, "published pack: no payload byte outside a tensor");
        check(g.hidden == 1152 && g.heads == 16 && g.head_dim == 72 && g.layers == 27, "published pack: 27 blocks at 1152, 16 x 72");
        check(g.patch == 14 && g.channels == 3 && g.pos_rows == 4900 && g.grid == 70 && g.image_size == 980,
              "published pack: 14-pixel patches on a 70 x 70 grid (980)");
        check(g.intermediate == 4608 && g.vit_merger_intermediate == 17280, "published pack: MLP 4608 and 17280, from the tensors");
        check(g.merge_hidden == 4608, "published pack: mergers at 4 x 1152");
        check(p.notes().size() == 2, "published pack: the two stale config widths are noted, not obeyed");
        // The 1B pack's container names its source; the 35B-A3B's carries no __metadata__ at all. Both are accepted,
        // and a source hash that is present has to look like one.
        const std::string sha = p.metadata("source_sha256");
        check(sha.empty() || (sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string::npos),
              "published pack: a source hash, when the container carries one, is 64 hex digits");
        std::printf("  provenance: %s\n", sha.empty() ? "none in the container" : (p.metadata("source_file") + " sha256 " + sha).c_str());
        std::printf("  published pack %s: %zu tensors, out_hidden %d, %zu note(s)\n", dir.c_str(), p.tensors().size(), g.out_hidden,
                    p.notes().size());
        for (const std::string& n : p.notes()) std::printf("    note: %s\n", n.c_str());
    } catch (const std::exception& e) {
        check(false, std::string("the published pack loads (threw: ") + e.what() + ")");
    }
}

}  // namespace

int main() {
    test_loads();
    test_stale_config_widths();
    test_refusals();
    test_pack_dir_name();
    std::remove(kPath);
    if (const char* dir = std::getenv("ONEBIT_MINICPM_PACK"); dir && *dir) test_published_pack(dir);
    else std::printf("  (ONEBIT_MINICPM_PACK not set: the published-pack arm did not run)\n");
    std::printf("vision pack: %d checks, %d failed\n", g_arms, g_fail);
    return g_fail ? 1 : 0;
}
