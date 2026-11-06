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

// npu_moe_container_test -- the two MoE container conventions the forward reads (engine#1035).
//
// The failure this pins is silent, so every check is a known answer rather than "it runs":
//  * the engine's converter (tools/gguf_to_q4nx.py) writes `mlp.experts.*_proj.weight` names and an
//    HF-shaped config; FastFlowLM-style containers use `*_exps_proj` names and a llama.cpp-shaped
//    config. Reading only the second made a converted Qwen3-Coder-30B run as a dense model and fail
//    with "MLP dequant : data_size 0".
//  * the engine converter's per-expert slice is a plain [N][K] tile grid; the FLM-style writer
//    transposes each expert, so its slice decodes on [K][N]. The wrong grid decodes a permutation:
//    on the real pack the converter slice correlates 0.99999 with the GGUF on [N][K] and 0.00855 on
//    [K][N]. The last check builds a q4_1 slice whose decoded values are known exactly and shows
//    only the geometry the format selects recovers them.
#include "moe_container.h"
#include "q4nx_dequant.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

void put_bf16(uint8_t* p, uint16_t bits) {
    p[0] = uint8_t(bits);
    p[1] = uint8_t(bits >> 8);
}

// One 5,120-byte q4_1 chunk for a 32x256 tile whose weight is `value(row, col)`, stored with a
// scale of 1.0 and a zero of 0 so the decoded weight is exactly the code.
void put_tile(uint8_t* t, int tr, int tc, int rows, int cols) {
    std::memset(t, 0, 5120);
    for (int r = 0; r < 32; ++r)
        for (int g = 0; g < 8; ++g) put_bf16(t + 2 * (g * 32 + r), 0x3F80);  // bf16(1.0)
    for (int r = 0; r < 32; ++r) {
        const int row = tr * 32 + r;
        if (row >= rows) continue;
        for (int j = 0; j < 256; ++j) {
            const int col = tc * 256 + j;
            if (col >= cols) continue;
            const uint8_t code = uint8_t(row % 16);   // known value: the row index mod 16
            uint8_t& byte = t[1024 + (r / 16) * 2048 + (j / 32) * 256 + (j % 32) * 8 + (r % 16) / 2];
            if (r % 2 == 0) byte = uint8_t((byte & 0xF0) | code);
            else byte = uint8_t((byte & 0x0F) | (code << 4));
        }
    }
}

}  // namespace

int main() {
    // ---- 1. the two spellings, and which grid each one's slices decode on ----
    const MoeNames eng = moe_names(MoeFormat::EngineConverter);
    const MoeNames flm = moe_names(MoeFormat::Flm);
    check(std::strcmp(eng.gate, "mlp.experts.gate_proj.weight") == 0 &&
              std::strcmp(eng.up, "mlp.experts.up_proj.weight") == 0 &&
              std::strcmp(eng.down, "mlp.experts.down_proj.weight") == 0 &&
              std::strcmp(eng.router, "mlp.gate.weight") == 0,
          "the engine converter's names are the mlp.experts.* / mlp.gate spellings");
    check(std::strcmp(flm.gate, "mlp.gate_exps_proj.weight") == 0 &&
              std::strcmp(flm.router, "moe_router.weight") == 0,
          "the FLM-style names are the *_exps_proj / moe_router spellings");
    check(eng.slice_nk && !flm.slice_nk,
          "only the engine converter's per-expert slice decodes on an [N][K] grid");

    // ---- 2. which convention a container is, from layer 0's gate tensor ----
    check(moe_format(false, true) == MoeFormat::EngineConverter, "converter gate alone -> EngineConverter");
    check(moe_format(true, false) == MoeFormat::Flm, "FLM gate alone -> Flm");
    check(moe_format(true, true) == MoeFormat::Flm, "both spellings -> FLM (the older convention wins)");
    check(moe_format(false, false) == MoeFormat::Flm, "neither -> Flm, which is what the old code assumed");

    // ---- 3. config keys: llama.cpp-style first, then the converter's HF names ----
    check(moe_expert_count(128, 0) == 128 && moe_expert_count(0, 128) == 128 &&
              moe_expert_count(128, 256) == 128 && moe_expert_count(0, 0) == 0,
          "expert count reads n_expert or num_experts, and 0 means 'not MoE'");
    check(moe_top_k(8, 0) == 8 && moe_top_k(0, 8) == 8 && moe_top_k(8, 4) == 8 && moe_top_k(0, 0) == 0,
          "top-k reads top_k or num_experts_per_tok");

    // ---- 4. the geometry the forward will decode an expert slice at ----
    int rows = 0, cols = 0;
    moe_slice_geometry(MoeFormat::EngineConverter, 768, 2048, &rows, &cols);
    check(rows == 768 && cols == 2048, "converter slice geometry is [N][K]");
    moe_slice_geometry(MoeFormat::Flm, 768, 2048, &rows, &cols);
    check(rows == 2048 && cols == 768, "FLM-style slice geometry is [K][N]");

    // ---- 5. known-answer decode: the selected grid is the one that recovers the weights ----
    // A 768 x 2048 expert (Qwen3-30B's ffn_gate_exps shape): 24 x 8 = 192 tiles. Scale 1.0,
    // zero 0, and every element of row n is the code n%16, so the correct decode is exactly
    // `row % 16` and any tile permutation shows up as a wrong value.
    const int N = 768, K = 2048;
    const int ntr = N / 32, ntc = K / 256;
    std::vector<uint8_t> slice((size_t)ntr * (size_t)ntc * 5120);
    for (int tr = 0; tr < ntr; ++tr)
        for (int tc = 0; tc < ntc; ++tc)
            put_tile(slice.data() + ((size_t)tr * ntc + tc) * 5120, tr, tc, N, K);

    std::vector<float> want((size_t)N * (size_t)K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) want[(size_t)n * K + k] = float(n % 16);

    std::vector<float> got((size_t)N * (size_t)K);
    moe_slice_geometry(MoeFormat::EngineConverter, N, K, &rows, &cols);
    check(q4nx_dequant_tensor_chunked(slice.data(), slice.size(), rows, cols, 5120, got.data()) == 0,
          "the converter's [N][K] geometry decodes the slice");
    check(got == want, "and it recovers the known weights exactly (the engine converter's container)");

    // The FLM-style grid on the same bytes is a permutation: reading it back as [N][K] must
    // disagree with the known weights, which is what makes the format choice load-bearing.
    std::vector<float> other((size_t)K * (size_t)N);
    moe_slice_geometry(MoeFormat::Flm, N, K, &rows, &cols);
    check(q4nx_dequant_tensor_chunked(slice.data(), slice.size(), rows, cols, 5120, other.data()) == 0,
          "the FLM-style [K][N] geometry also decodes these bytes");
    size_t differ = 0;
    for (size_t i = 0; i < want.size(); ++i) differ += (other[i] != want[i]);
    check(differ > want.size() / 4,
          "and it does NOT recover them (a silent permutation: " + std::to_string(differ) + " of " +
              std::to_string(want.size()) + " values differ)");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all MoE container-convention checks passed (names, config keys, slice geometry, "
                "known-answer decode)\n");
    return 0;
}
