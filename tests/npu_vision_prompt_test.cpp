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
// npu_vision_prompt_test
//
// The text side of an image (npu/vision_prompt.h, engine#1788) against transformers 5.22.2's MiniCPM-V 4.7 code:
// MiniCPMV4_7Processor.replace_image_token for the tokens and MiniCPMV4_7Model.get_rope_index for the positions.
// Eight prompts are built here from plan_image() and image_tokens(); the reference built the same prompts from its own
// processor arithmetic, and for each one this file holds its token count, piece count, next position and one FNV-1a
// hash over (ids, t, h, w). A wrong token, a wrong order or one wrong position changes the hash.
// Text tokens are 1000 + k; the marker ids are the published packs'.
#include "../npu/vision_prompt.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace vis = onebit::npu::vision;

int g_fail = 0, g_arms = 0;
void check(bool ok, const std::string& what) {
    ++g_arms;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL %s\n", what.c_str());
    }
}

const vis::ImageMarkers kM = {248078, 248079, 248088, 248089, 248056, 198};

struct Prompt {
    std::vector<int> ids;
    std::vector<std::pair<int, int>> grids;
    void text(int n, int base) {
        for (int i = 0; i < n; ++i) ids.push_back(1000 + base + i);
    }
    void raw(std::initializer_list<int> v) { ids.insert(ids.end(), v); }
    void image(int h, int w) {
        const vis::ImagePlan p = vis::plan_image(h, w);
        const std::vector<int> t = vis::image_tokens(p, kM);
        ids.insert(ids.end(), t.begin(), t.end());
        grids.push_back({p.source.grid_h, p.source.grid_w});
        for (int k = 0; k < p.slice_rows * p.slice_cols; ++k) grids.push_back({p.slice.grid_h, p.slice.grid_w});
    }
};

Prompt scenario(int k) {
    Prompt p;
    const int NL = kM.newline;
    switch (k) {
        case 0: p.text(7, 0); p.image(448, 448); p.text(5, 50); break;                      // one view, no slices
        case 1: p.text(3, 0); p.image(480, 640); p.raw({NL}); p.text(4, 50); break;          // 1 x 2 slices, a newline after
        case 2: p.text(2, 0); p.image(1080, 1920); p.image(100, 100); p.text(3, 50); break;  // two images touching
        case 3: p.text(1, 0); p.image(720, 1280); p.text(6, 50); p.image(3024, 4032); p.text(2, 80); break;
        case 4: p.text(4, 0); p.image(5000, 300); p.text(1, 50); break;                      // 9 x 1 slices
        case 5:                                                                              // <image_id>k</image_id> prefixes
            p.text(3, 0); p.raw({248090, 15, 248091}); p.image(600, 800); p.raw({NL, NL}); p.text(2, 50);
            p.raw({248090, 16, 248091}); p.image(224, 2000); p.raw({NL});
            break;
        case 6: p.text(9, 0); break;                                                         // no image at all
        case 7: p.text(1, 0); p.image(449, 448); p.raw({NL}); p.image(448, 449); break;      // two images, a newline between
    }
    return p;
}

// {tokens, pieces, next position, hash(ids, t, h, w)}, from the reference
struct Golden {
    int tokens, pieces, next;
    uint64_t hash;
};
const Golden kGolden[8] = {
    {78, 1, 21, 0xccb3904ded7cd15bULL},   {217, 3, 22, 0xece6c1beaeb65928ULL},  {668, 10, 47, 0xcd6d6c517226c2afULL},
    {1120, 17, 65, 0x9e0d9e3759a6b555ULL}, {693, 10, 105, 0xbfd3bd95f7383645ULL}, {662, 10, 79, 0xdd390fcf63fc8236ULL},
    {9, 0, 9, 0x2d9fde739c2c5f7cULL},     {407, 6, 27, 0xbe7321611eaa8facULL},
};
const uint64_t kLinspaceHash = 0x2f0e4ce031c6cf8eULL;  // round(linspace(0, c - 1, n)) for c, n in 1..160, by torch

uint64_t fnv(const std::vector<int>& v, uint64_t h) {
    for (int x : v)
        for (int b = 0; b < 4; ++b) h = (h ^ uint64_t((uint32_t(x) >> (8 * b)) & 0xff)) * 0x100000001b3ULL;
    return h;
}

}  // namespace

int main() {
    for (int k = 0; k < 8; ++k) {
        const Prompt p = scenario(k);
        const std::string name = "prompt " + std::to_string(k);
        try {
            const vis::PromptPositions pos = vis::prompt_positions(p.ids, kM, p.grids);
            uint64_t h = fnv(p.ids, 0xcbf29ce484222325ULL);
            h = fnv(pos.t, h);
            h = fnv(pos.h, h);
            h = fnv(pos.w, h);
            check(int(p.ids.size()) == kGolden[k].tokens && int(p.grids.size()) == kGolden[k].pieces,
                  name + ": " + std::to_string(p.ids.size()) + " tokens and " + std::to_string(p.grids.size()) + " pieces, as the reference lays it out");
            check(pos.next == kGolden[k].next, name + ": generation resumes at position " + std::to_string(pos.next));
            check(h == kGolden[k].hash, name + ": every token id and every (t, h, w) is the reference's");
        } catch (const std::exception& e) {
            check(false, name + " is laid out (threw: " + e.what() + ")");
        }
    }

    // prompt 1, spelled out: 3 text tokens, then a 480 x 640 image (a 7 x 9 view over 1 x 2 slices of 10 x 7)
    {
        const Prompt p = scenario(1);
        const vis::PromptPositions pos = vis::prompt_positions(p.ids, kM, p.grids);
        check(p.ids[3] == kM.image_start && pos.t[3] == 3 && pos.h[3] == 2 && pos.w[3] == 2,
              "<image> keeps the canvas origin on t and sits one step before it on h and w");
        check(pos.t[4] == 3 && pos.h[4] == 3 && pos.w[4] == 3 && pos.w[5] == 5 && pos.w[6] == 6 && pos.w[7] == 8,
              "the whole view is stretched over the slices' canvas: columns 0, 2, 3, 5 ... of 14 for 9 tokens");
        check(p.ids[67] == kM.image_end && pos.t[67] == 3 && pos.h[67] == 13 && pos.w[67] == 17,
              "</image> sits on the canvas's far corner (10 down, 14 across)");
        const size_t n = p.ids.size();
        check(pos.t[n - 4] == 18 && pos.t[n - 1] == 21 && pos.next == 22,
              "text resumes at origin + max(10, 14) + 1 = 18: the image costs its side, not its 203 rows");
        check(p.ids[n - 5] == kM.newline && pos.t[n - 5] == 3 && pos.h[n - 5] == 3,
              "a newline directly after the image belongs to it and takes the canvas origin");
    }

    {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int c = 1; c <= 160; ++c)
            for (int n = 1; n <= 160; ++n) {
                std::vector<int> v(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) v[size_t(i)] = vis::linspace_round(c - 1, n, i);
                h = fnv(v, h);
            }
        check(h == kLinspaceHash, "round(linspace(0, c - 1, n)) is torch's for every canvas and count up to 160");
    }

    // pads and grids that do not describe the same pieces
    {
        Prompt p = scenario(1);
        std::string why;
        auto g = p.grids;
        g.pop_back();
        try { vis::prompt_positions(p.ids, kM, g); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("more image pieces than grids") != std::string::npos, "a missing grid is refused");
        why.clear();
        g = p.grids;
        g.push_back({32, 32});
        try { vis::prompt_positions(p.ids, kM, g); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("more grids were given") != std::string::npos, "an extra grid is refused");
        why.clear();
        g = p.grids;
        g[0] = {32, 32};
        try { vis::prompt_positions(p.ids, kM, g); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("placeholders but its grid gives") != std::string::npos, "a grid that is not the piece's row count is refused");
        why.clear();
        g = p.grids;
        g[1] = {39, 28};
        try { vis::prompt_positions(p.ids, kM, g); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("multiple of 4") != std::string::npos, "a grid the tower cannot merge is refused");
        why.clear();
        try { vis::prompt_positions({kM.image_pad, 1000}, kM, {{4, 4}}); } catch (const std::exception& e) { why = e.what(); }
        check(why.find("opens the prompt") != std::string::npos, "a placeholder with nothing in front of it is refused");
    }
    std::printf("vision prompt: %d checks, %d failed\n", g_arms, g_fail);
    return g_fail ? 1 : 0;
}
