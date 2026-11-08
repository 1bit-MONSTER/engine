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

// A vision-conditioned prompt through the decoder (engine#1311/TQVL): the wiring gate 2 needs, on the engine side.
//
// Reads what the vision half produced -- the prompt's token ids, the image rows' embeddings, the 3-plane positions and
// the deepstack sets -- and runs the prefill through step_batch_embeds(), then greedily decodes N tokens. Text rows'
// embeddings come from the model's own table (embed_row), so the only foreign rows are the image ones.
//
//   hybrid_prompt_probe <model.q4nx> <kernels dir> <input dir> [--decode N]
//
// <input dir> holds ids.i32, positions.i32 (3 x rows), pooled.f32 (rows_img x H), deepstack.f32 (sets x rows_img x H)
// and meta.json {"image_lo": i, "image_hi": j}. Build them with the vision graph (tools/export_qwen3vl_vision.py)
// plus the tokenizer; this probe does not touch a tokenizer.
#include "q4nx_forward.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>

template <class T>
static bool read_vec(const std::string& path, std::vector<T>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const auto n = (size_t)f.tellg() / sizeof(T);
    f.seekg(0);
    out.resize(n);
    f.read((char*)out.data(), (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.q4nx> <kernels dir> <input dir> [--decode N]\n", argv[0]);
        return 2;
    }
    int decode = 8, top_k_n = 0;
    for (int i = 4; i + 1 < argc; i++)
        if (!std::strcmp(argv[i], "--decode")) decode = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--top-k") && i + 1 < argc) top_k_n = std::atoi(argv[++i]);
    const std::string dir = argv[3];

    Q4nxNpuForward f;
    std::unique_ptr<xrt::device> dev;
    bool ok;
    if (getenv("NPU_INFER_HOST_GEMM")) {
        ok = f.init_host(argv[1], argv[2]);
    } else {
        dev = std::make_unique<xrt::device>(0);
        ok = f.init(*dev, argv[1], argv[2]);
    }
    if (!ok) { std::fprintf(stderr, "init: %s\n", f.last_error()); return 1; }
    const int H = 0;   // read from a row, below
    (void)H;

    std::vector<int32_t> ids, positions;
    std::vector<float> pooled, deepstack;
    if (!read_vec(dir + "/ids.i32", ids) || !read_vec(dir + "/positions.i32", positions) ||
        !read_vec(dir + "/pooled.f32", pooled) || !read_vec(dir + "/deepstack.f32", deepstack)) {
        std::fprintf(stderr, "cannot read the input files from %s\n", dir.c_str());
        return 2;
    }
    // meta.json is tiny; the two numbers are read by position so the probe needs no JSON library
    int meta[2] = {-1, -1};
    {
        std::ifstream m(dir + "/meta.txt");
        if (m) m >> meta[0] >> meta[1];   // image_lo image_hi (one line each is fine)
    }
    // image_lo == image_hi is the text-only control: no image rows, nothing to inject.
    if (meta[0] < 0 || meta[1] < meta[0]) { std::fprintf(stderr, "bad meta (image_lo image_hi)\n"); return 2; }
    const int rows = (int)ids.size();
    std::vector<float> one;
    f.embed_row(11, one);
    const size_t Hd = one.size();
    const size_t rows_img = size_t(meta[1] - meta[0]);
    if (pooled.size() != rows_img * Hd || positions.size() != size_t(3) * size_t(rows)) {
        std::fprintf(stderr, "shapes: pooled %zu (want %zu), positions %zu (want %zu)\n", pooled.size() / Hd,
                     rows_img, positions.size(), size_t(3) * rows);
        return 2;
    }
    const int n_ds = f.n_deepstack_layers();
    if (n_ds > 0 && deepstack.size() != size_t(n_ds) * rows_img * Hd) {
        std::fprintf(stderr, "deepstack %zu, want %d sets x %zu rows x %zu\n", deepstack.size(), n_ds, rows_img, Hd);
        return 2;
    }

    // embedding rows: the model's own for text positions, the vision graph's for the image span
    std::vector<float> embeds;
    embeds.reserve(size_t(rows) * Hd);
    for (int r = 0; r < rows; r++) {
        if (r >= meta[0] && r < meta[1]) {
            const float* src = pooled.data() + size_t(r - meta[0]) * Hd;
            embeds.insert(embeds.end(), src, src + Hd);
        } else {
            f.embed_row(ids[size_t(r)], one);
            embeds.insert(embeds.end(), one.begin(), one.end());
        }
    }
    Q4nxNpuForward::HybridInputs in;
    in.position_ids = &positions;
    // A text-only prompt (the control) has no image rows, so there is no span to add deepstack sets at.
    if (n_ds > 0 && rows_img > 0) {
        in.deepstack = deepstack.data();
        in.deepstack_sets = n_ds;
        in.image_lo = meta[0];
        in.image_hi = meta[1];
    }

    std::vector<float> logits;
    f.reset();
    if (!f.step_batch_embeds(embeds, 0, logits, in)) { std::fprintf(stderr, "prefill: %s\n", f.last_error()); return 1; }
    auto argmax = [](const std::vector<float>& v) {
        int b = 0;
        for (size_t i = 1; i < v.size(); i++)
            if (v[i] > v[(size_t)b]) b = (int)i;
        return b;
    };
    // --top-k N prints the top-N logits wherever an argmax is taken, so a decode step can be diffed against the
    // reference number for number instead of token for token (two decoders that agree on the argmax can still differ by
    // a lot underneath, and that is what decides whether the next step is off).
    auto top_k = [&](const char* where) {
        if (top_k_n <= 0) return;
        std::vector<std::pair<float, int>> v;
        v.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); i++) v.emplace_back(logits[i], (int)i);
        const size_t k = std::min((size_t)top_k_n, v.size());
        std::partial_sort(v.begin(), v.begin() + (long)k, v.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
        std::printf("top%d %s:", top_k_n, where);
        for (size_t i = 0; i < k; i++) std::printf(" %d:%.4f", v[i].second, v[i].first);
        std::printf("\n");
        std::fflush(stdout);
    };
    std::vector<int> out;
    top_k("prefill");
    out.push_back(argmax(logits));
    int pos = int(positions[0 * rows + rows - 1]) + 1;   // the generation step's position (delta already applied)
    for (int i = 1; i < decode; i++) {
        if (!f.step(out.back(), pos + i - 1, logits)) { std::fprintf(stderr, "step: %s\n", f.last_error()); return 1; }
        top_k(i == 1 ? "step1" : "step");
        out.push_back(argmax(logits));
    }
    std::printf("prompt %d tokens (%zu image rows), deepstack %d set(s)\n", rows, rows_img, in.deepstack_sets);
    std::printf("generated %d token ids:", (int)out.size());
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    return 0;
}
