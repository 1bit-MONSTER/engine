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

// The embedding gate on the device (engine#1311 step 2): encode a fixed sentence list on the NPU through the
// engine's own int8 GEMM and compare each unit vector with the reference's, then report the rate.
//
//   ONEBIT_EP_NPU_ELFS="128x384x384=<Q_K384_N384.elf>,128x384x1536=<GU_K384_N1536.elf>,128x256x384=<D_K256_N384.elf>" \
//   embed_npu_gate_probe <container dir> <sentences.txt> <reference vectors.txt> [--host] [--concurrent N]
//
// --concurrent N runs N threads through the same list at once, the way the served route answers (cpp-httplib's
// worker pool), and requires every thread's every vector to be bit-identical to the single-threaded answer.
// That is the row for the shared weight cache: a lazily filled map raced from two threads either corrupts the
// map or launches with a half-written block, and both show up here as a difference, not as a crash.
//
// One sentence per line in both files; a reference line is 384 floats. `--host` runs the same loop with the
// host dot instead of the array, so the two rates come from one binary and one input.
// Exit: 0 both paths agree with the reference (min cosine >= 0.999), 1 a path disagrees, 2 usage, 3 device refused.
#include "embed_model.h"
#include "embed_npu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const bool host = argc > 4 && std::strcmp(argv[4], "--host") == 0;
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <container> <sentences.txt> <reference.txt> [--host]\n", argv[0]);
        return 2;
    }
    onebit::embed::Model m;
    std::string why;
    if (!m.load(argv[1], why)) {
        std::fprintf(stderr, "load: %s\n", why.c_str());
        return 2;
    }
    onebit::embed::NpuGemm npu;
    if (!host) {
        // pack every weight block before the first request, which is also what the served route does: the
        // cache is read-only afterwards, so a worker pool cannot race it
        if (!npu.prepare(m, why)) {
            std::fprintf(stderr, "prepare: %s\n", why.c_str());
            return 2;
        }
        m.set_gemm(npu.hook());
    }

    std::vector<std::string> sentences;
    {
        std::ifstream f(argv[2]);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            sentences.push_back(line);
        }
    }
    std::vector<std::vector<float>> ref;
    {
        std::ifstream f(argv[3]);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::vector<float> row;
            float v = 0;
            while (is >> v) row.push_back(v);
            if (!row.empty()) ref.push_back(std::move(row));
        }
    }
    if (sentences.size() != ref.size() || sentences.empty()) {
        std::fprintf(stderr, "input: %zu sentences vs %zu reference vectors\n", sentences.size(), ref.size());
        return 2;
    }
    // --concurrent N: N threads over the whole list, compared against the single-threaded answer below.
    std::vector<std::vector<std::vector<float>>> conc;
    int workers = 0;
    for (int i = 4; i < argc; ++i)
        if (std::strcmp(argv[i], "--concurrent") == 0 && i + 1 < argc) workers = std::atoi(argv[i + 1]);
    if (workers > 0) {
        conc.resize(size_t(workers));
        std::vector<std::thread> threads;
        std::atomic<int> ready{0};
        for (int w = 0; w < workers; ++w)
            threads.emplace_back([&, w] {
                conc[size_t(w)].reserve(sentences.size());
                ++ready;
                while (ready.load() < workers) std::this_thread::yield();   // start together: the race window is the start
                for (const auto& text : sentences) {
                    std::string w_why;
                    conc[size_t(w)].push_back(m.encode(text, w_why));
                }
            });
        for (auto& th : threads) th.join();
    }
    const auto t0 = std::chrono::steady_clock::now();
    double worst = 1.0, sum = 0.0;
    int below = 0, failed = 0;
    for (size_t i = 0; i < sentences.size(); ++i) {
        const auto v = m.encode(sentences[i], why);
        if (v.empty()) {
            std::fprintf(stderr, "encode %zu: %s\n", i, why.c_str());
            return 3;
        }
        const auto& r = ref[i];
        if (v.size() != r.size()) {
            std::fprintf(stderr, "encode %zu: %zu floats vs %zu reference\n", i, v.size(), r.size());
            return 2;
        }
        double c = 0, nv = 0, nr = 0;
        for (size_t k = 0; k < v.size(); ++k) {
            c += double(v[k]) * r[k];
            nv += double(v[k]) * v[k];
            nr += double(r[k]) * r[k];
        }
        c /= std::sqrt(nv * nr);
        worst = std::min(worst, c);
        sum += c;
        if (c < 0.999) ++below;
        if (!std::isfinite(c)) ++failed;
    }
    // The concurrency row: every worker against the single-threaded vector for the same sentence, bit for bit.
    long concurrent_mismatch = 0;
    double concurrent_worst = 0.0;
    if (workers > 0) {
        for (size_t i = 0; i < sentences.size(); ++i) {
            const auto single = m.encode(sentences[i], why);
            for (int w = 0; w < workers; ++w) {
                const auto& v = conc[size_t(w)][i];
                if (v.size() != single.size()) { ++concurrent_mismatch; continue; }
                for (size_t k = 0; k < v.size(); ++k)
                    if (v[k] != single[k]) { ++concurrent_mismatch; break; }
                if (v.size() == single.size())
                    for (size_t k = 0; k < v.size(); ++k)
                        concurrent_worst = std::max(concurrent_worst, double(std::fabs(v[k] - single[k])));
            }
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("path=%s sentences=%zu min_cosine=%.7f mean_cosine=%.9f below_0.999=%d non_finite=%d "
                "wall=%.3fs rate=%.2f/s launches=%ld\n",
                host ? "host" : (npu.used_device() ? "npu" : "npu-refused-fell-back"), sentences.size(), worst,
                sum / double(sentences.size()), below, failed, s, double(sentences.size()) / s, npu.launches());
    if (workers > 0) {
        std::printf("concurrent=%d sentences=%zu mismatched_vectors=%ld worst|dv|=%.3e\n", workers,
                    sentences.size(), concurrent_mismatch, concurrent_worst);
        std::printf("%s\n", concurrent_mismatch == 0 ? "concurrency: identical to the single-threaded answer"
                                                     : "concurrency: MISMATCH (shared state raced)");
    }
    if (!npu.status().empty()) std::fprintf(stderr, "npu: %s\n", npu.status().c_str());
    if (below || failed || concurrent_mismatch) return 1;
    return host || npu.used_device() ? 0 : 3;
}
