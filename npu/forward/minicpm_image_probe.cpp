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
// One picture and one question through a MiniCPM-V 4.7 pack, every step the engine's own (engine#1788):
//
//   RGB bytes -> plan_image -> preprocess (resize, slices, normalise) -> tower_forward (host float32) per piece
//             -> the prompt's tokens (image_tokens) and its three position planes (prompt_positions)
//             -> the text model: step_batch_embeds() with the tower's rows where the placeholders are, then greedy decode.
//
//   minicpm_image_probe <pack dir> <kernels dir> (--image file.ppm | --solid R,G,B [--size HxW] | --halves R,G,B:R,G,B | --none)
//                       --prompt "question" [--decode N] [--expect-ids ids.i32] [--dump dir] [--chunk N] [--rows-argmax]
//
// A probe, not a route: the tower runs on the CPU (about half a minute for one 448 x 448 view on one thread), the image is
// a binary PPM or a generated colour so that no image decoder is involved, and the prompt is the one the pack's own chat
// template renders for a single user turn with one image and thinking off, written out here rather than rendered:
//     <|im_start|>user\n<image_id>0</image_id>[image]\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
// --expect-ids compares the token ids built here with a file of int32 ids (the reference processor's), and says so.
// --dump writes what the text model is about to be given (ids.i32, positions.i32, pooled.f32), --chunk N sends the
// prompt in calls of at most N rows as the served route does, and --rows-argmax prints the model's next-token choice at
// every row of a text-only prompt. The three exist to bisect a wrong answer against a reference run.
// The text model opens the device unless NPU_INFER_HOST_GEMM is set, so on the device this runs inside npu-hold.
#include "q4nx_forward.h"

#include "tokenizer.h"
#include "vision_image.h"
#include "vision_pack.h"
#include "vision_plan.h"
#include "vision_prompt.h"
#include "vision_tower.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vis = onebit::npu::vision;

static bool parse_rgb(const std::string& s, uint8_t out[3]) {
    int r = -1, g = -1, b = -1;
    if (std::sscanf(s.c_str(), "%d,%d,%d", &r, &g, &b) != 3) return false;
    if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) return false;
    out[0] = uint8_t(r);
    out[1] = uint8_t(g);
    out[2] = uint8_t(b);
    return true;
}

// A binary PPM (P6, maxval 255): the one image format that needs no decoder.
static bool read_ppm(const std::string& path, std::vector<uint8_t>& rgb, int& h, int& w) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    int maxval = 0;
    const auto token = [&]() {
        std::string t;
        for (;;) {
            f >> t;
            if (!f || t.empty() || t[0] != '#') break;
            std::string rest;
            std::getline(f, rest);
        }
        return t;
    };
    magic = token();
    if (magic != "P6") return false;
    w = std::atoi(token().c_str());
    h = std::atoi(token().c_str());
    maxval = std::atoi(token().c_str());
    f.get();  // the single whitespace byte after maxval
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384 || maxval != 255) return false;
    rgb.resize(size_t(h) * w * 3);
    f.read(reinterpret_cast<char*>(rgb.data()), std::streamsize(rgb.size()));
    return f.gcount() == std::streamsize(rgb.size());
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <pack dir> <kernels dir> (--image f.ppm | --solid R,G,B [--size HxW] | --halves R,G,B:R,G,B | "
                             "--none) --prompt \"question\" [--decode N] [--expect-ids ids.i32] [--dump dir] [--chunk N] [--rows-argmax]\n", argv[0]);
        return 2;
    }
    const std::string pack_dir = argv[1], kernels = argv[2];
    std::string image_path, prompt, expect_ids, dump_dir;
    bool none = false, have_solid = false, have_halves = false, rows_argmax = false;
    uint8_t solid[3] = {0, 0, 0}, left[3] = {0, 0, 0}, right[3] = {0, 0, 0};
    int height = 448, width = 448, decode = 8, chunk = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--image") image_path = value();
        else if (a == "--solid") { have_solid = parse_rgb(value(), solid); if (!have_solid) { std::fprintf(stderr, "--solid wants R,G,B\n"); return 2; } }
        else if (a == "--halves") {
            const std::string v = value();
            const size_t c = v.find(':');
            have_halves = c != std::string::npos && parse_rgb(v.substr(0, c), left) && parse_rgb(v.substr(c + 1), right);
            if (!have_halves) { std::fprintf(stderr, "--halves wants R,G,B:R,G,B\n"); return 2; }
        } else if (a == "--size") { if (std::sscanf(value().c_str(), "%dx%d", &height, &width) != 2 || height <= 0 || width <= 0 || height > 16384 || width > 16384) { std::fprintf(stderr, "--size wants HxW\n"); return 2; } }
        else if (a == "--none") none = true;
        else if (a == "--prompt") prompt = value();
        else if (a == "--decode") decode = std::atoi(value().c_str());
        else if (a == "--expect-ids") expect_ids = value();
        else if (a == "--dump") dump_dir = value();
        else if (a == "--chunk") chunk = std::atoi(value().c_str());
        else if (a == "--rows-argmax") rows_argmax = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (chunk < 0) { std::fprintf(stderr, "--chunk wants a row count (0: one call)\n"); return 2; }
    if (prompt.empty() || decode < 1 || decode > 512 || int(none) + int(have_solid) + int(have_halves) + int(!image_path.empty()) != 1) {
        std::fprintf(stderr, "give --prompt and exactly one of --image, --solid, --halves, --none\n");
        return 2;
    }

    try {
        const auto now = [] { return std::chrono::steady_clock::now(); };
        const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const onebit::npu::Tokenizer tok(pack_dir + "/tokenizer.json");
        vis::ImageMarkers m;
        m.image_start = tok.token_id("<image>");
        m.image_end = tok.token_id("</image>");
        m.slice_start = tok.token_id("<slice>");
        m.slice_end = tok.token_id("</slice>");
        m.image_pad = tok.token_id("<|image_pad|>");
        const std::vector<int> nl = tok.encode("\n");
        const int id_open = tok.token_id("<image_id>"), id_close = tok.token_id("</image_id>");
        if (m.image_start < 0 || m.image_end < 0 || m.slice_start < 0 || m.slice_end < 0 || m.image_pad < 0 || id_open < 0 || id_close < 0 ||
            nl.size() != 1)
            throw std::runtime_error("this pack's tokenizer lacks one of <image>, </image>, <slice>, </slice>, <|image_pad|>, "
                                     "<image_id>, </image_id>, or does not encode a newline as one token");
        m.newline = nl[0];

        // ── the picture: bytes -> pieces -> the tower's rows ──
        std::vector<float> rows;                          // every piece's rows, in order, out_hidden wide
        std::vector<std::pair<int, int>> grids;
        std::vector<int> image_ids;
        int out_hidden = 0;
        if (!none) {
            std::vector<uint8_t> rgb;
            if (!image_path.empty()) {
                if (!read_ppm(image_path, rgb, height, width)) throw std::runtime_error(image_path + " is not a binary PPM (P6, maxval 255)");
            } else {
                rgb.resize(size_t(height) * width * 3);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const uint8_t* c = have_solid ? solid : (x < width / 2 ? left : right);
                        std::memcpy(&rgb[(size_t(y) * width + x) * 3], c, 3);
                    }
            }
            const vis::Pack vp(pack_dir);
            out_hidden = vp.geometry().out_hidden;
            const vis::ImagePlan plan = vis::plan_image(height, width);
            if (plan.total_rows() > 4096) throw std::runtime_error("this image would be " + std::to_string(plan.total_rows()) + " rows; the probe stops at 4096");
            const auto t0 = now();
            const std::vector<vis::Piece> pieces = vis::preprocess(rgb.data(), height, width, plan);
            const auto t1 = now();
            for (const vis::Piece& p : pieces) {
                const std::vector<float> r = vis::tower_forward(vp, p.pixels.data(), p.grid_h, p.grid_w);
                rows.insert(rows.end(), r.begin(), r.end());
                grids.push_back({p.grid_h, p.grid_w});
            }
            const auto t2 = now();
            image_ids = vis::image_tokens(plan, m);
            std::printf("image %d x %d: %zu piece(s), %d rows; preprocess %.0f ms, tower (host) %.0f ms\n", height, width, pieces.size(),
                        plan.total_rows(), ms(t0, t1), ms(t1, t2));
        }

        // ── the prompt, as the pack's template renders one user turn (thinking off) ──
        std::vector<int> ids = tok.encode("<|im_start|>user\n");
        if (!none) {
            ids.push_back(id_open);
            const std::vector<int> zero = tok.encode("0");
            ids.insert(ids.end(), zero.begin(), zero.end());
            ids.push_back(id_close);
            ids.insert(ids.end(), image_ids.begin(), image_ids.end());
            ids.push_back(m.newline);
        }
        const std::vector<int> tail = tok.encode(prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        ids.insert(ids.end(), tail.begin(), tail.end());
        const vis::PromptPositions pos = vis::prompt_positions(ids, m, grids);
        const int n = int(ids.size());
        if (!expect_ids.empty()) {
            std::ifstream f(expect_ids, std::ios::binary | std::ios::ate);
            std::vector<int32_t> want(f ? size_t(f.tellg()) / 4 : 0);
            if (f) { f.seekg(0); f.read(reinterpret_cast<char*>(want.data()), std::streamsize(want.size() * 4)); }
            bool same = want.size() == ids.size();
            for (size_t i = 0; same && i < ids.size(); ++i) same = want[i] == ids[i];
            std::printf("prompt ids vs %s: %s (%zu here, %zu there)\n", expect_ids.c_str(), same ? "IDENTICAL" : "DIFFERENT", ids.size(), want.size());
        }

        // --dump <dir>: what the text model is about to be given, in hybrid_prompt_probe's file forms, so a wrong answer
        // can be bisected against a reference that writes the same files (ids.i32, positions.i32 as t|h|w, pooled.f32).
        if (!dump_dir.empty()) {
            const auto put = [&](const char* name, const void* data, size_t bytes) {
                std::ofstream f(dump_dir + "/" + name, std::ios::binary);
                f.write(static_cast<const char*>(data), std::streamsize(bytes));
                if (!f) throw std::runtime_error("cannot write " + dump_dir + "/" + name);
            };
            std::vector<int32_t> i32(ids.begin(), ids.end());
            put("ids.i32", i32.data(), i32.size() * 4);
            std::vector<int32_t> pl;
            pl.insert(pl.end(), pos.t.begin(), pos.t.end());
            pl.insert(pl.end(), pos.h.begin(), pos.h.end());
            pl.insert(pl.end(), pos.w.begin(), pos.w.end());
            put("positions.i32", pl.data(), pl.size() * 4);
            put("pooled.f32", rows.data(), rows.size() * sizeof(float));
            std::printf("dumped %zu ids, 3 x %zu positions and %zu row values to %s\n", ids.size(), ids.size(), rows.size(), dump_dir.c_str());
        }

        // ── the text model ──
        Q4nxNpuForward f;
        std::unique_ptr<xrt::device> dev;
        const std::string model = pack_dir + "/model.q4nx";
        bool ok;
        const auto t3 = now();
        if (std::getenv("NPU_INFER_HOST_GEMM")) {
            ok = f.init_host(model.c_str(), kernels.c_str());
        } else {
            dev = std::make_unique<xrt::device>(0);
            ok = f.init(*dev, model.c_str(), kernels.c_str());
        }
        if (!ok) throw std::runtime_error(std::string("text model: ") + f.last_error());
        std::vector<float> one;
        f.embed_row(ids[0], one);
        const size_t H = one.size();
        if (!none && size_t(out_hidden) != H) throw std::runtime_error("the tower projects to " + std::to_string(out_hidden) + " but the text model is " + std::to_string(H) + " wide");
        std::vector<float> embeds;
        embeds.reserve(size_t(n) * H);
        size_t next_row = 0;
        for (int i = 0; i < n; ++i) {
            if (ids[size_t(i)] == m.image_pad) {
                if ((next_row + 1) * H > rows.size()) throw std::runtime_error("the prompt holds more placeholders than the tower produced rows");
                embeds.insert(embeds.end(), rows.begin() + long(next_row * H), rows.begin() + long((next_row + 1) * H));
                ++next_row;
            } else {
                f.embed_row(ids[size_t(i)], one);
                embeds.insert(embeds.end(), one.begin(), one.end());
            }
        }
        if (next_row * H != rows.size()) throw std::runtime_error("the tower produced more rows than the prompt has placeholders");
        std::vector<int> planes;
        planes.reserve(size_t(3) * n);
        planes.insert(planes.end(), pos.t.begin(), pos.t.end());
        planes.insert(planes.end(), pos.h.begin(), pos.h.end());
        planes.insert(planes.end(), pos.w.begin(), pos.w.end());
        Q4nxNpuForward::HybridInputs in;
        in.position_ids = &planes;
        std::vector<float> logits;
        // --rows-argmax (text-only prompts): the model's own next-token choice at every prompt row, from one teacher-forced
        // pass over the prompt's ids. Compared row by row with a reference run of the same ids, it says whether a wrong
        // answer is one bad position or a uniform error in the layers; that is how engine#2149 was found (65 % agreement,
        // flat across positions). Images are refused: step_batch_all() takes token ids, not rows.
        if (rows_argmax && !none) throw std::runtime_error("--rows-argmax needs --none: the per-row pass takes token ids");
        if (rows_argmax) {
            std::vector<float> all;
            f.reset();
            if (!f.step_batch_all(ids, 0, all)) throw std::runtime_error(std::string("step_batch_all: ") + f.last_error());
            const size_t V = all.size() / size_t(n);
            std::printf("rows-argmax:");
            for (int r = 0; r < n; ++r) {
                const float* row = all.data() + size_t(r) * V;
                size_t b = 0;
                for (size_t i = 1; i < V; ++i) if (row[i] > row[b]) b = i;
                std::printf(" %zu", b);
            }
            std::printf("\n");
        }
        f.reset();
        const auto t4 = now();
        // The prompt goes in as chunks of at most `chunk` rows, as the served route sends a prefill (--chunk N; 0 sends
        // it as one call). Each call gets its own rows' embeddings and planes and starts at its cache slot.
        if (chunk <= 0) {
            if (!f.step_batch_embeds(embeds, 0, logits, in)) throw std::runtime_error(std::string("prefill: ") + f.last_error());
        } else {
            for (int off = 0; off < n; off += chunk) {
                const int take = off + chunk < n ? chunk : n - off;
                const std::vector<float> part(embeds.begin() + long(size_t(off) * H), embeds.begin() + long(size_t(off + take) * H));
                std::vector<int> pl;
                pl.insert(pl.end(), pos.t.begin() + off, pos.t.begin() + off + take);
                pl.insert(pl.end(), pos.h.begin() + off, pos.h.begin() + off + take);
                pl.insert(pl.end(), pos.w.begin() + off, pos.w.begin() + off + take);
                Q4nxNpuForward::HybridInputs ci;
                ci.position_ids = &pl;
                if (!f.step_batch_embeds(part, off, logits, ci)) throw std::runtime_error(std::string("prefill: ") + f.last_error());
            }
        }
        const auto t5 = now();
        const auto argmax = [](const std::vector<float>& v) {
            int b = 0;
            for (size_t i = 1; i < v.size(); ++i)
                if (v[i] > v[size_t(b)]) b = int(i);
            return b;
        };
        const int im_end = tok.token_id("<|im_end|>");
        std::vector<int> out;
        out.push_back(argmax(logits));
        // the rotary position resumes after the canvas (pos.next); the cache slot is the prompt's length
        for (int i = 1; i < decode && out.back() != im_end; ++i) {
            if (!f.step(out.back(), pos.next + i - 1, logits, n + i - 1)) throw std::runtime_error(std::string("step: ") + f.last_error());
            out.push_back(argmax(logits));
        }
        const auto t6 = now();
        std::printf("prompt %d tokens (%zu image rows), last position %d; init %.0f ms, prefill %.0f ms, %zu token(s) in %.0f ms\n", n, next_row,
                    pos.next - 1, ms(t3, t4), ms(t4, t5), out.size(), ms(t5, t6));
        std::printf("ids:");
        for (int t : out) std::printf(" %d", t);
        std::printf("\nanswer: %s\n", tok.decode(out).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "minicpm_image_probe: %s\n", e.what());
        return 1;
    }
}
