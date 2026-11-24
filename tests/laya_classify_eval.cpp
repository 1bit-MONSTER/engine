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

// laya_classify_eval <checkpoint_dir> <cases.json> [--limit N] [--min-accuracy PERCENT] [--per-case]
// Scores every labelled request (tests/laya_route_cases.json) with the request-class question
// (laya/route.h, request_class_question) and prints accuracy, the confusion matrix, per-class
// accuracy and accuracy by confidence band, plus the decision time. --per-case also prints every
// case's decision and probabilities, so a failing row names the request that moved. RFC #186.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "route.h"
#include "scorer.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: laya_classify_eval <checkpoint_dir> <cases.json> [--limit N] [--min-accuracy PERCENT] [--per-case]\n");
        return 2;
    }
    int limit = 1 << 30;
    double min_accuracy = 0;  // ctest laya_classify fails below it
    bool per_case = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (a == "--limit") {
            const char* v = next();
            if (!v) { std::fprintf(stderr, "--limit needs a value\n"); return 2; }
            limit = std::atoi(v);
        } else if (a == "--min-accuracy") {
            const char* v = next();
            if (!v) { std::fprintf(stderr, "--min-accuracy needs a value\n"); return 2; }
            min_accuracy = std::atof(v);
        } else if (a == "--per-case") {
            per_case = true;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    onebit::laya::Scorer scorer;
    if (!scorer.load(argv[1])) {
        std::fprintf(stderr, "load: %s\n", scorer.error().c_str());
        return 1;
    }
    std::ifstream in(argv[2]);
    const nlohmann::json cases = nlohmann::json::parse(in);
    const std::vector<std::string> classes = onebit::laya::request_classes();
    std::map<std::string, std::map<std::string, int>> confusion;
    std::map<std::string, int> right, total;
    std::vector<std::pair<float, bool>> conf;  // (confidence, correct)
    double ms = 0;
    int n = 0, ok = 0;
    for (const auto& c : cases["cases"]) {
        if (n >= limit) break;
        const std::string want = c["class"], text = c["text"];
        const auto t0 = std::chrono::steady_clock::now();
        const onebit::laya::RequestClass got = onebit::laya::classify_request(scorer, text);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (got.label.empty()) {
            std::fprintf(stderr, "score failed: %s\n", scorer.error().c_str());
            return 1;
        }
        ++n;
        confusion[want][got.label]++;
        total[want]++;
        const bool hit = got.label == want;
        right[want] += hit;
        ok += hit;
        conf.emplace_back(got.confidence, hit);
        if (per_case) {
            auto prob = [&](const char* key) {
                for (const auto& [k, v] : got.probabilities)
                    if (k == key) return v;
                return 0.0f;
            };
            std::string t = text.substr(0, 70);
            for (char& ch : t)
                if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            std::fprintf(stderr, "per-case: want=%-8s got=%-8s conf=%.2f code=%.2f prose=%.2f short=%.2f | %s%s\n",
                         want.c_str(), got.label.c_str(), got.confidence, prob("code"), prob("prose"),
                         prob("short"), t.c_str(), text.size() > t.size() ? "..." : "");
        }
    }
    std::printf("accuracy %d/%d = %.1f%%, %.0f ms per decision\n", ok, n, 100.0 * ok / std::max(1, n), ms / std::max(1, n));
    std::printf("%-10s", "want\\got");
    for (const auto& k : classes) std::printf(" %9s", k.c_str());
    std::printf("   accuracy\n");
    for (const auto& w : classes) {
        std::printf("%-10s", w.c_str());
        for (const auto& g : classes) std::printf(" %9d", confusion[w][g]);
        std::printf("   %.0f%%\n", 100.0 * right[w] / std::max(1, total[w]));
    }
    std::sort(conf.begin(), conf.end());
    std::printf("by confidence (quartiles, low to high):");
    for (int q = 0; q < 4; ++q) {
        const size_t a = conf.size() * q / 4, b = conf.size() * (q + 1) / 4;
        int h = 0;
        for (size_t i = a; i < b; ++i) h += conf[i].second;
        std::printf("  [%.2f-%.2f] %.0f%%", b > a ? conf[a].first : 0.f, b > a ? conf[b - 1].first : 0.f,
                    100.0 * h / std::max<size_t>(1, b - a));
    }
    std::printf("\n");
    if (100.0 * ok / std::max(1, n) < min_accuracy) {
        std::printf("FAIL: below the %.1f%% floor\n", min_accuracy);
        return 1;
    }
    return 0;
}
