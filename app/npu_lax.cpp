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

#include "npu_lax.h"

#include "lax.h"
#include "model.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit {

namespace {

namespace lax = npu::lax;
using clk = std::chrono::steady_clock;
double sec_since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<float> v(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
    return v;
}

std::string sfx(int t) { return t ? "_t" + std::to_string(t) : ""; }

// The reference's metric (open_kernels model/compare_decode.py): Pearson correlation over
// the common length, argmax (first maximum) and top-5; PASS needs finite logits, corr >
// 0.9999 and the same argmax.
bool compare(int t, const std::vector<float>& ours, const std::vector<float>& ref) {
    const size_t n = std::min(ours.size(), ref.size());
    double ma = 0, mb = 0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) {
        finite = finite && std::isfinite(ours[i]);
        ma += ours[i];
        mb += ref[i];
    }
    ma /= double(n);
    mb /= double(n);
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = double(ours[i]) - ma, b = double(ref[i]) - mb;
        sab += a * b;
        saa += a * a;
        sbb += b * b;
    }
    const double corr = sab / std::sqrt(saa * sbb);
    auto top = [n](const std::vector<float>& v, int k) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[size_t(a)] > v[size_t(b)] || (v[size_t(a)] == v[size_t(b)] && a < b); });
        idx.resize(size_t(k));
        return idx;
    };
    const auto to = top(ours, 5), tr = top(ref, 5);
    auto list = [](const std::vector<int>& v) {
        std::string s;
        for (int x : v) s += (s.empty() ? "" : " ") + std::to_string(x);
        return s;
    };
    const bool ok = finite && corr > 0.9999 && to[0] == tr[0];
    std::printf("position %d: logits corr %.6f  argmax ours %d ref %d  top5 ours [%s] ref [%s]  %s\n", t, corr, to[0], tr[0],
                list(to).c_str(), list(tr).c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

int single_id(const npu::Tokenizer& tok, const char* text) {
    const auto ids = tok.encode(text);
    if (ids.size() != 1) throw std::runtime_error(std::string("the tokenizer has no single id for ") + text);
    return ids[0];
}

}  // namespace

int run_npu_lax(int argc, char** argv) {
    const auto t_start = clk::now();
    std::string model_dir, kernel_dir, parity, dump;
    int max_new = 256, tokens = 3, threads = 0, bench = 0;
    lax::Transport transport = lax::Transport::Elf;
    std::vector<std::string> prompts;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model_dir = next();
        else if (a == "--kernels") kernel_dir = next();
        else if (a == "-n" || a == "--max-new") max_new = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--parity") parity = next();
        else if (a == "--tokens") tokens = std::stoi(next());
        else if (a == "--dump") dump = next();
        else if (a == "--transport") transport = lax::parse_transport(next());
        else if (a == "--bench") bench = std::stoi(next());
        else if (a == "--help" || a == "-h") {
            std::printf(
                "usage: 1bit npu-lax --model <dir> --kernels <dir> [-n 256] [--threads N] [\"prompt\" ...]\n"
                "       1bit npu-lax --model <dir> --kernels <dir> --parity <ref dir> [--tokens 3] [--dump <dir>]\n"
                "       1bit npu-lax --model <dir> --kernels <dir> --bench N\n"
                "  --model      Qwen3.6-35B-A3B Q4NX directory (model.q4nx, config.json, tokenizer.json)\n"
                "  --kernels    scripts/build-lax.sh's <prefix>/kernels (lax_l, lax_a, ln, lm_head_q8)\n"
                "  --transport  elf (full ELFs, the default) or classic (xclbin + insts.bin)\n"
                "  --parity     make_decode.py --requant output: runs xres<t>.bin at position t and scores\n"
                "               the logits against ref_logits<_tN>.bin (corr > 0.9999, same argmax), then\n"
                "               again after reset(), which must give the same logits bit for bit\n"
                "  --bench      N tokens at positions 0..N-1, twice (new and cached position configs);\n"
                "               prints the mean host prep, 40-layer and head times\n"
                "Without prompts, reads one prompt per line from stdin; the conversation is one session.\n");
            return 0;
        } else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
        else prompts.push_back(a);
    }
    if (model_dir.empty() || kernel_dir.empty()) throw std::runtime_error("--model and --kernels are required");

    const npu::Model model(model_dir);
    const auto cfg = lax::Config::from_model(model, model_dir);
    std::fprintf(stderr, "[packing %s onto the NPU...]\n", model_dir.c_str());
    lax::Decoder dec(model, cfg, kernel_dir, transport, threads);
    const auto& ls = dec.load_stats();
    std::fprintf(stderr, "[ready in %.1f s since process start: buffers %.1f s, packing %.2f GB %.1f s; %s]\n",
                 sec_since(t_start), ls.buffers_ms / 1000, double(ls.packed_bytes) / 1e9, ls.pack_ms / 1000,
                 dec.kernels().c_str());

    if (!parity.empty()) {
        bool all = true;
        std::vector<float> lg;
        std::vector<std::vector<float>> first;
        for (int t = 0; t < tokens; ++t) {
            const auto x = read_f32(parity + "/xres" + std::to_string(t) + ".bin");
            if (x.size() < lax::kHidden) throw std::runtime_error("xres" + std::to_string(t) + ".bin is short");
            dec.set_residual(x.data());
            const auto t0 = clk::now();
            dec.run(size_t(t), true);
            const double ms = sec_since(t0) * 1000;
            dec.logits(lg);
            if (!dump.empty())
                std::ofstream(dump + "/y_logits" + sfx(t) + ".bin", std::ios::binary)
                    .write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * 4));
            std::fprintf(stderr, "[position %d: %.1f ms]\n", t, ms);
            all = compare(t, lg, read_f32(parity + "/ref_logits" + sfx(t) + ".bin")) && all;
            first.push_back(lg);
        }
        // A second sequence after reset() must see nothing of the first.
        dec.reset();
        for (int t = 0; t < tokens; ++t) {
            const auto x = read_f32(parity + "/xres" + std::to_string(t) + ".bin");
            dec.set_residual(x.data());
            dec.run(size_t(t), true);
            dec.logits(lg);
            const bool same = lg == first[size_t(t)];
            std::printf("position %d after reset: logits %s\n", t, same ? "bit-identical PASS" : "differ FAIL");
            all = same && all;
        }
        std::printf("%s\n", all ? "PASS" : "FAIL");
        return all ? 0 : 1;
    }

    if (bench > 0) {
        const npu::Tokenizer tok(model_dir + "/tokenizer.json");
        const int token = single_id(tok, "<|im_start|>");
        for (int pass = 0; pass < 2; ++pass) {
            dec.reset();
            lax::RunTimes sum;
            const auto t0 = clk::now();
            for (int p = 0; p < bench; ++p) {
                dec.feed(token);
                dec.run(size_t(p), true);
                const auto& r = dec.last_run();
                sum.prep_ms += r.prep_ms;
                sum.layers_ms += r.layers_ms;
                sum.head_ms += r.head_ms;
                sum.ahead_ms += r.ahead_ms;
            }
            const double total = sec_since(t0) * 1000 / bench;
            std::printf("bench %s: %d tokens, per token %.2f ms (%.1f tok/s): prep %.3f ms, 40 layers %.2f ms "
                        "(%.3f ms of it preparing the next), norm + lm head %.2f ms\n",
                        pass ? "cached positions" : "new positions", bench, total, 1000 / total, sum.prep_ms / bench,
                        sum.layers_ms / bench, sum.ahead_ms / bench, sum.head_ms / bench);
        }
        std::fprintf(stderr, "[%s]\n", dec.kernels().c_str());
        return 0;
    }

    const npu::Tokenizer tok(model_dir + "/tokenizer.json");
    const int vocab = tok.size();  // the lm head's rows past the tokenizer are padding
    const int im_end = single_id(tok, "<|im_end|>"), eot = single_id(tok, "<|endoftext|>");
    size_t pos = 0;
    auto turn = [&](const std::string& text) {
        // Qwen3.6's chat template, thinking off: an empty think block opens the answer.
        std::vector<int> ids = tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        if (pos) {
            auto nl = tok.encode("\n");  // after the previous turn's <|im_end|>
            ids.insert(ids.begin(), nl.begin(), nl.end());
        }
        if (pos + ids.size() + size_t(max_new) + 1 > size_t(lax::kMaxContext)) {
            std::fprintf(stderr, "[context full (%zu of %d positions)]\n", pos, lax::kMaxContext);
            return false;
        }
        const auto t1 = clk::now();
        for (size_t i = 0; i < ids.size(); ++i) {
            dec.feed(ids[i]);
            dec.run(pos++, i + 1 == ids.size());  // the head only where a next token is wanted
        }
        int next = dec.argmax(vocab);
        const double prompt_s = sec_since(t1);
        const auto t2 = clk::now();
        std::vector<int> out;
        std::string shown;
        while (next != im_end && next != eot && int(out.size()) < max_new) {
            out.push_back(next);
            const std::string s = tok.decode(out);
            std::fwrite(s.data() + shown.size(), 1, s.size() - shown.size(), stdout);
            std::fflush(stdout);
            shown = s;
            dec.feed(next);
            dec.run(pos++, true);
            next = dec.argmax(vocab);
        }
        const double gen_s = sec_since(t2);
        dec.feed(next);  // the end-of-turn token enters the cache too
        dec.run(pos++, false);
        std::printf("\n");
        std::fflush(stdout);
        std::fprintf(stderr, "[prompt %zu tok in %.2f s (%.1f tok/s), %zu tok in %.2f s = %.1f tok/s, context %zu]\n",
                     ids.size(), prompt_s, double(ids.size()) / prompt_s, out.size(), gen_s,
                     double(out.size()) / std::max(gen_s, 1e-9), pos);
        return true;
    };
    if (!prompts.empty()) {
        for (const auto& p : prompts)
            if (!turn(p)) break;
    } else {
        for (std::string line; std::getline(std::cin, line);)
            if (line.find_first_not_of(" \t\r") != std::string::npos && !turn(line)) break;
    }
    return 0;
}

}  // namespace onebit
