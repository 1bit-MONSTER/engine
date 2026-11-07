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

// The 1-bit embedding container and forward (engine#1311), device-free, on a tiny fixture: load the container
// the importer writes, check the dims it declares, and encode two fixed token sequences to the vectors the
// documented arithmetic produces (tests/trit_embed_pack_test.py --write-fixture). The real model's gate -- a
// min cosine of 0.9997 against `trit_embed.py` on 1,000 sentences -- is the measured row in the PR, because a
// 13.9 MB container does not belong in the tree; this test is what keeps the plumbing from drifting.
#include "embed_model.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

int main() {
    const std::string dir = std::string(SRC_DIR) + "/tests/fixtures/trit_embed_tiny";
    nlohmann::json exp;
    {
        std::ifstream f(dir + "/expected.json");
        if (!f) {
            std::fprintf(stderr, "missing fixture: %s\n", dir.c_str());
            return 2;
        }
        f >> exp;
    }
    onebit::embed::Model m;
    std::string why;
    if (!m.load(dir, why)) {
        std::fprintf(stderr, "load: %s\n", why.c_str());
        return 1;
    }
    const auto& c = m.config();
    const auto& d = exp["dims"];
    if (c.vocab != d["vocab"] || c.d != d["d"] || c.heads != d["heads"] || c.ffn != d["ffn"] ||
        c.layers != d["layers"] || c.max_pos != d["max_pos"] || c.out_dim != d["out_dim"] ||
        c.max_tokens != d["max_pos"]) {
        std::fprintf(stderr, "dims: got vocab=%d d=%d heads=%d ffn=%d layers=%d max_pos=%d out=%d\n", c.vocab,
                     c.d, c.heads, c.ffn, c.layers, c.max_pos, c.out_dim);
        return 1;
    }
    int fails = 0;
    for (const auto& cs : exp["cases"]) {
        std::vector<int> ids = cs["ids"].get<std::vector<int>>();
        const auto v = m.encode_ids(ids, why);
        if (v.empty()) {
            std::fprintf(stderr, "encode_ids: %s\n", why.c_str());
            return 1;
        }
        const std::vector<float> want = cs["vector"].get<std::vector<float>>();
        if (v.size() != want.size()) {
            std::fprintf(stderr, "vector length %zu, expected %zu\n", v.size(), want.size());
            return 1;
        }
        float worst = 0.0f;
        double norm = 0.0;
        for (size_t i = 0; i < v.size(); ++i) worst = std::fmax(worst, std::fabs(v[i] - want[i]));
        for (float x : v) norm += double(x) * x;
        std::printf("case %zu: %zu tokens, worst |dv| %.3e, norm %.9f\n", ids.size(), v.size(), worst,
                    std::sqrt(norm));
        if (worst > 2e-5f) {
            std::fprintf(stderr, "case differs by %.3e (tolerance 2e-5)\n", worst);
            ++fails;
        }
        if (std::fabs(std::sqrt(norm) - 1.0) > 1e-5) {
            std::fprintf(stderr, "vector is not unit norm\n");
            ++fails;
        }
    }
    // refusals: an id outside the vocabulary, an empty sequence, and a sequence longer than the positions
    std::vector<int> bad_id = {1, 9999, 2};
    if (!m.encode_ids(bad_id, why).empty()) {
        std::fprintf(stderr, "an out-of-vocabulary id was accepted\n");
        ++fails;
    }
    if (!m.encode_ids({}, why).empty()) {
        std::fprintf(stderr, "an empty sequence was accepted\n");
        ++fails;
    }
    std::vector<int> too_long(size_t(c.max_pos) + 1, 1);
    if (!m.encode_ids(too_long, why).empty()) {
        std::fprintf(stderr, "a sequence past max_pos was accepted\n");
        ++fails;
    }
    if (fails) {
        std::fprintf(stderr, "embed_model: %d failure(s)\n", fails);
        return 1;
    }
    std::puts("embed_model: ok");
    return 0;
}
