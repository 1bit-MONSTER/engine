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
// npu_vision_tower_test
//
// The host float32 MiniCPM-V 4.7 vision tower (npu/vision_tower.h, engine#1788) against the transformers
// implementation (transformers.models.minicpmv4_7, 5.22.2, torch 2.13 CPU, float32).
//
// The fixture is a small tower written here: width 16 (2 heads x 8), 2-pixel patches, a 4 x 4 position table, 8 blocks
// with the vit_merger after block 6, MLPs 12 wide stored 16 wide (zero padded, as the published packs store theirs),
// projecting to 12. Its weights come from a 32-bit LCG, so the same file can be rebuilt anywhere; the goldens below are
// what the transformers classes return for that file and the input pixel(c, y, x) = sin(0.37c + 0.11y + 0.05x +
// 0.003xy) on a 4 x 8 patch grid. Nothing here was produced by the code under test.
//
// ONEBIT_VISION_KEEP_FIXTURE=1 leaves the container and its config.json in the working directory (that is how the
// goldens were produced). ONEBIT_MINICPM_PACK=<pack dir> with ONEBIT_MINICPM_ORACLE=<dir holding pixels.f32, tower.f32,
// out.f32 and grid.txt> compares a published pack's output with a reference run of the same input.
#include "../npu/vision_plan.h"    // engine#2324: text-side rows vs the tower's own row count
#include "../npu/vision_tower.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

const int H = 16, C = 3, P = 2, SIDE = 4, L = 8, I = 12, I_STORED = 16, VI = 48, VI_STORED = 64, O = 12;
const int GH = 4, GW = 8;

uint32_t g_lcg = 20261110u;
float lcg_unit() {  // in [-1, 1)
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return float(g_lcg >> 8) / 8388608.0f - 1.0f;
}
uint16_t bf16(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return uint16_t((b + 0x8000u) >> 16);
}

struct Writer {
    json table = json::object();
    std::vector<uint8_t> payload;
    // kind 0: a matrix or bias, 0.3 * u (bias 0.1 * u); kind 1: a norm weight, 1 + 0.2 * u.
    // Elements at or past `live` along `axis` are zero: the container's padding.
    void add(const std::string& name, std::vector<int64_t> shape, int kind, int axis = -1, int64_t live = 0) {
        uint64_t n = 1;
        for (int64_t d : shape) n *= uint64_t(d);
        const uint64_t off = payload.size();
        const int64_t cols = shape.size() > 1 ? shape[1] : 1;
        for (uint64_t e = 0; e < n; ++e) {
            const bool is_bias = shape.size() == 1 && kind == 0;
            float v = kind == 1 ? 1.0f + 0.2f * lcg_unit() : (is_bias ? 0.1f : 0.3f) * lcg_unit();
            if (axis == 0 && int64_t(e) / cols >= live) v = 0.0f;
            if (axis == 1 && int64_t(e) % cols >= live) v = 0.0f;
            const uint16_t h = bf16(v);
            payload.push_back(uint8_t(h & 0xff));
            payload.push_back(uint8_t(h >> 8));
        }
        table["model.visual." + name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {off, off + n * 2}}};
    }
    void linear(const std::string& stem, int64_t rows, int64_t cols) {
        add(stem + ".weight", {rows, cols}, 0);
        add(stem + ".bias", {rows}, 0);
    }
    void norm(const std::string& stem, int64_t n) {
        add(stem + ".weight", {n}, 1);
        add(stem + ".bias", {n}, 0);
    }
    void padded_mlp(const std::string& stem, int64_t in, int64_t live, int64_t stored, int64_t out) {
        add(stem + ".linear_fc1.weight", {stored, in}, 0, 0, live);
        add(stem + ".linear_fc1.bias", {stored}, 0, 0, live);
        add(stem + ".linear_fc2.weight", {out, stored}, 0, 1, live);
        add(stem + ".linear_fc2.bias", {out}, 0);
    }
};

const char* const kContainer = "npu_vision_tower_test.q4nx";
const char* const kConfig = "npu_vision_tower_test.config.json";

std::string write_fixture() {
    g_lcg = 20261110u;
    Writer w;
    w.add("patch_embed.proj.weight", {H, C, P, P}, 0);
    w.add("patch_embed.proj.bias", {H}, 0);
    w.add("pos_embed.weight", {SIDE * SIDE, H}, 0);
    for (int i = 0; i < L; ++i) {
        const std::string b = "blocks." + std::to_string(i) + ".";
        w.norm(b + "norm1", H);
        w.linear(b + "attn.qkv", 3 * H, H);
        w.linear(b + "attn.proj", H, H);
        w.norm(b + "norm2", H);
        w.padded_mlp(b + "mlp", H, I, I_STORED, H);
    }
    w.norm("vit_merger.norm1", H);
    w.linear("vit_merger.attn.qkv", 3 * H, H);
    w.linear("vit_merger.attn.proj", H, H);
    w.norm("vit_merger.pre_norm", 4 * H);
    w.padded_mlp("vit_merger.mlp", 4 * H, VI, VI_STORED, H);
    w.norm("merger.norm", H);
    w.norm("merger.pre_norm", 4 * H);
    w.linear("merger.linear_fc1", 4 * H, 4 * H);
    w.linear("merger.linear_fc2", O, 4 * H);
    const std::string head = w.table.dump();
    const uint64_t len = head.size();
    std::ofstream f(kContainer, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&len), 8);
    f.write(head.data(), std::streamsize(head.size()));
    f.write(reinterpret_cast<const char*>(w.payload.data()), std::streamsize(w.payload.size()));

    json vc;
    vc["MINICPM_V_PATCH_SIZE"] = P;
    vc["MINICPM_V_IMAGE_SIZE"] = SIDE * P;
    vc["MINICPM_V_VISION_EMBED_DIM"] = H;
    vc["MINICPM_V_VISION_NUM_HEADS"] = 2;
    vc["MINICPM_V_VISION_HEAD_DIM"] = 8;
    vc["MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE"] = I;
    vc["MINICPM_V_VISION_NUM_POSITION_EMBEDDINGS"] = SIDE * SIDE;
    vc["MINICPM_V_VISION_NUM_LAYERS"] = L;
    vc["MINICPM_V_VISION_LAYER_NORM_EPSILON"] = 1e-6;
    vc["MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE"] = VI;
    json c;
    c["hidden_size"] = O;
    c["insert_layer_id"] = 6;
    c["vision_config"] = vc;
    const std::string text = c.dump();
    std::ofstream cf(kConfig);
    cf << text;
    return text;
}

std::vector<float> test_pixels(int gh, int gw, int patch) {
    const int hp = gh * patch, wp = gw * patch;
    std::vector<float> px(size_t(3) * hp * wp);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < hp; ++y)
            for (int x = 0; x < wp; ++x)
                px[(size_t(c) * hp + y) * wp + x] = float(std::sin(0.37 * c + 0.11 * y + 0.05 * x + 0.003 * double(x) * y));
    return px;
}

// GOLDENS_BEGIN (transformers 5.22.2, torch 2.13.0+cpu, float32, eager attention)
const float kGoldenOut[2 * O] = {
    1.66476488f, -0.92423868f, 0.45880169f, 0.54087150f, 0.91351569f, -2.78720903f,
    1.69952393f, 0.54694951f,  -1.22599459f, 2.31425953f, 0.93287897f, 0.15626144f,
    1.68985367f, -0.91641307f, 0.10820520f, 0.28901398f, 0.99814737f, -2.51858306f,
    1.39385486f, 0.41435477f,  -1.09378004f, 2.35683727f, 0.77005690f, 0.12177789f,
};
const double kGoldenTowerSum = -0.859797;
// GOLDENS_END

double max_abs_diff(const std::vector<float>& a, const float* b, size_t n) {
    double m = a.size() == n ? 0.0 : 1e30;
    for (size_t i = 0; i < n && i < a.size(); ++i) m = std::fmax(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

void test_fixture(const std::string& cfg) {
    try {
        const vis::Pack p(kContainer, cfg);
        check(p.geometry().model_intermediate == I && p.geometry().intermediate == I_STORED && p.geometry().insert_layer == 6,
              "fixture: MLPs 12 wide stored 16 wide, vit_merger after block 6");
        const std::vector<float> px = test_pixels(GH, GW, P);
        std::vector<float> tower;
        const std::vector<float> out = vis::tower_forward(p, px.data(), GH, GW, &tower);
        check(out.size() == size_t(2 * O) && tower.size() == size_t(GH / 2) * (GW / 2) * H,
              "fixture: a 4 x 8 grid gives 8 tower rows and 2 output rows");
        const double d = max_abs_diff(out, kGoldenOut, size_t(2 * O));
        double sum = 0;
        for (float v : tower) sum += v;
        std::printf("  fixture: max |out - reference| = %.3g; tower sum %.6f (reference %.6f)\n", d, sum, kGoldenTowerSum);
        std::printf("  out[0,:6] %.8f %.8f %.8f %.8f %.8f %.8f\n", out[0], out[1], out[2], out[3], out[4], out[5]);
        check(d < 2e-5, "fixture: every output value is the reference's within 2e-5");
        check(std::fabs(sum - kGoldenTowerSum) < 2e-3, "fixture: the tower before the merger sums to the reference's");

        // Threads change the time and nothing else: forced onto 2, 5 and 16 threads with no size threshold (so even
        // this small tower is split), every output and every tower value is the same BITS as on one thread.
        {
            vis::tower_threading(1);
            std::vector<float> t1;
            const std::vector<float> o1 = vis::tower_forward(p, px.data(), GH, GW, &t1);
            bool same = o1 == out;
            for (int nt : {2, 5, 16}) {
                vis::tower_threading(nt, 0.0);
                std::vector<float> tn;
                const std::vector<float> on = vis::tower_forward(p, px.data(), GH, GW, &tn);
                same = same && on.size() == o1.size() && tn.size() == t1.size() &&
                       std::memcmp(on.data(), o1.data(), o1.size() * sizeof(float)) == 0 &&
                       std::memcmp(tn.data(), t1.data(), t1.size() * sizeof(float)) == 0;
            }
            vis::tower_threading(0);
            check(same, "fixture: 1, 2, 5 and 16 threads give bit-identical rows");
        }

        // grids the two 2 x 2 merges cannot take
        for (const auto& [gh, gw] : std::vector<std::pair<int, int>>{{4, 6}, {2, 8}, {0, 8}, {4, -4}}) {
            std::string why;
            try { vis::tower_forward(p, px.data(), gh, gw); } catch (const std::exception& e) { why = e.what(); }
            check(why.find("multiples of 4") != std::string::npos,
                  "a " + std::to_string(gh) + " x " + std::to_string(gw) + " grid is refused by name");
        }
    } catch (const std::exception& e) {
        check(false, std::string("the fixture runs (threw: ") + e.what() + ")");
    }
    // without a config there is no head count, and the forward says so instead of guessing one
    try {
        const vis::Pack p(kContainer, "");
        const std::vector<float> px = test_pixels(GH, GW, P);
        std::string why;
        try { vis::tower_forward(p, px.data(), GH, GW); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("head count") != std::string::npos, "no config: the forward refuses for want of a head count");
    } catch (const std::exception& e) {
        check(false, std::string("the fixture opens without a config (threw: ") + e.what() + ")");
    }
}

// The reference buckets y / gh (float32) against k / side; the integer form must give the same row and column. These
// are the reference's own indices for a 70-wide table (printed from get_vision_nearest_position_ids).
void test_nearest_position() {
    check(vis::nearest_position(0, 0, 8, 12, 70) == 0, "nearest position: the first patch is table row 0");
    check(vis::nearest_position(7, 11, 8, 12, 70) == 61 * 70 + 64, "nearest position: (7, 11) of 8 x 12 is (61, 64)");
    check(vis::nearest_position(3, 5, 8, 12, 70) == 26 * 70 + 29, "nearest position: (3, 5) of 8 x 12 is (26, 29)");
    check(vis::nearest_position(69, 69, 70, 70, 70) == 69 * 70 + 69, "nearest position: a full grid maps onto itself");
    check(vis::nearest_position(139, 0, 140, 4, 70) == 69 * 70, "nearest position: a grid taller than the table shares rows");
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<float> v(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
    return v;
}

void test_published(const std::string& pack_dir, const std::string& oracle) {
    try {
        int gh = 0, gw = 0;
        { std::ifstream g(oracle + "/grid.txt"); g >> gh >> gw; }
        const std::vector<float> px = read_f32(oracle + "/pixels.f32"), want = read_f32(oracle + "/out.f32"),
                                 want_tower = read_f32(oracle + "/tower.f32");
        const vis::Pack p(pack_dir);
        check(gh > 0 && gw > 0 && px.size() == size_t(3) * gh * gw * p.geometry().patch * p.geometry().patch,
              "published pack: the reference input is this grid's pixels");
        std::vector<float> tower;
        const std::vector<float> out = vis::tower_forward(p, px.data(), gh, gw, &tower);
        const auto stats = [](const std::vector<float>& a, const std::vector<float>& b, double* mx, double* corr, double* mean_abs) {
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            *mx = a.size() == b.size() ? 0 : 1e30;
            *mean_abs = 0;
            const size_t n = a.size() < b.size() ? a.size() : b.size();
            for (size_t i = 0; i < n; ++i) {
                *mx = std::fmax(*mx, std::fabs(double(a[i]) - b[i]));
                sa += a[i]; sb += b[i]; saa += double(a[i]) * a[i]; sbb += double(b[i]) * b[i]; sab += double(a[i]) * b[i];
                *mean_abs += std::fabs(double(b[i]));
            }
            *mean_abs /= double(n ? n : 1);
            const double va = saa - sa * sa / double(n), vb = sbb - sb * sb / double(n);
            *corr = (sab - sa * sb / double(n)) / std::sqrt(va * vb);
        };
        double mx, corr, mean_abs;
        stats(tower, want_tower, &mx, &corr, &mean_abs);
        std::printf("  published pack, %d x %d grid: tower   %zu values, max |d| %.3g, corr %.9f, reference mean |v| %.4f\n", gh, gw,
                    tower.size(), mx, corr, mean_abs);
        check(tower.size() == want_tower.size() && corr > 0.99999, "published pack: the tower before the merger matches the reference");
        stats(out, want, &mx, &corr, &mean_abs);
        std::printf("  published pack, %d x %d grid: output  %zu values, max |d| %.3g, corr %.9f, reference mean |v| %.4f\n", gh, gw,
                    out.size(), mx, corr, mean_abs);
        check(out.size() == want.size() && corr > 0.99999 && mx < 5e-3, "published pack: the output matches the reference");
    } catch (const std::exception& e) {
        check(false, std::string("the published pack runs (threw: ") + e.what() + ")");
    }
}

}  // namespace

int main() {
    const std::string cfg = write_fixture();
    test_fixture(cfg);
    test_nearest_position();
    if (!std::getenv("ONEBIT_VISION_KEEP_FIXTURE")) {
        std::remove(kContainer);
        std::remove(kConfig);
    }
    // engine#2324: the join's invariant on a REAL vision pack, with no oracle needed -- one text-side row per
    // placeholder, and the tower's own row count is the merger's second halving away from it. The reference arm below
    // needs ONEBIT_MINICPM_ORACLE (pixels.f32/tower.f32/out.f32/grid.txt); this one needs only the pack, which is the
    // thing a lane actually has. What it pins is the arithmetic that only minicpm_image_probe.cpp checked, and only on a
    // 21 GB model: `PiecePlan::rows` is "grid_h/4 * grid_w/4", so `out.size()/out_hidden` must equal it, and the tower
    // before the merger must be "grid_h/2 * grid_w/2" rows.
    {
        const char* pdir = std::getenv("ONEBIT_MINICPM_PACK");
        if (pdir && *pdir) {
            try {
                // `Pack(pack_dir)` reads config.json and the container it names. The plan is built at FOUR PATCHES PER SIDE
                // over the pack's own patch size, not at the pack's 980-pixel image: 4 * patch is the smallest geometry the
                // planner leaves alone (scale_resolution must be at least 4 * patch, and a bigger request is resized), and
                // this arm is about the two row counts, not about running a 70 x 70 grid on four cores.
                const std::string pdir_s = pdir;          // `Pack pk(std::string(pdir))` is a function declaration
                const vis::Pack pk(pdir_s);
                const vis::Geometry& g = pk.geometry();
                const int side = 4 * g.patch;
                const vis::ImagePlan plan = vis::plan_image(side, side, /*max_slice_nums=*/1,
                                                            /*scale_resolution=*/side, /*patch=*/g.patch);
                const vis::PiecePlan& src = plan.source;
                const int gh = src.grid_h, gw = src.grid_w;
                std::vector<float> px(size_t(3) * size_t(gh * g.patch) * size_t(gw * g.patch));
                for (size_t k = 0; k < px.size(); ++k) px[k] = float(int(k % 251) - 125) / 128.0f;
                std::vector<float> tower;
                const std::vector<float> out = vis::tower_forward(pk, px.data(), gh, gw, &tower);
                const size_t text_rows = g.out_hidden ? out.size() / size_t(g.out_hidden) : 0;
                check(int(text_rows) == src.rows,
                      "join (real pack): the merger hands the text model exactly PiecePlan::rows rows");
                check(tower.size() == size_t(gh / 2) * size_t(gw / 2) * size_t(g.hidden),
                      "join (real pack): the tower before the merger is grid/2 x grid/2 rows");
            } catch (const std::exception& e) {
                check(false, std::string("join (real pack): opened and ran (") + e.what() + ")");
            }
        } else {
            std::printf("  (ONEBIT_MINICPM_PACK not set: the join arm did not run)\n");
        }
    }
    const char* pack = std::getenv("ONEBIT_MINICPM_PACK");
    const char* oracle = std::getenv("ONEBIT_MINICPM_ORACLE");
    if (pack && *pack && oracle && *oracle) test_published(pack, oracle);
    else std::printf("  (ONEBIT_MINICPM_PACK / ONEBIT_MINICPM_ORACLE not set: the published-pack arm did not run)\n");
    std::printf("vision tower: %d checks, %d failed\n", g_arms, g_fail);
    return g_fail ? 1 : 0;
}
