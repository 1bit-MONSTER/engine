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
// npu_q4nx_test <dir with q4_1.bin q4k.bin q8.bin>
//
// The Q4NX chunk decoders (npu/q4nx.h) against 1bit-MONSTER's verified decoders
// (npu-infer/tools/q4nx_dequant.py at 35fafde8f, itself checked against the checkpoint's
// bf16 twin): one real tile of each kind, from Qwen3.5-4B (layer 0 qkv_proj tile 0, Q4_K;
// lm_head tile 0, Q8) and Qwen3-0.6B (layer 0 q_proj tile 0, q4_1). The expected output is
// the SHA-256 of the reference's 32 x 256 f32, so any bit that differs fails.
// Then Q4_K -> q4_1: the converted tile must decode within the bf16 rounding of its two
// scale products (half an ulp of S*scale times the largest code, plus half an ulp of M*min).
#include "../npu/q4nx.h"

#include "hrx_runtime.h"  // the one SHA-256 implementation, shared with serve's runtime line

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using onebit::npu::q4nx::Chunk;

std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}


float bf16(const uint8_t* p) {
    const uint32_t u = uint32_t(p[0] | (p[1] << 8)) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <golden dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    struct Case { const char* file; Chunk kind; const char* want; };
    const Case cases[] = {
        {"q4k.bin", Chunk::Q4K, "d2fc854e3b61bc0c77004724fefd441af57408519e8113bde87b63fd2ee5c4f5"},
        {"q8.bin", Chunk::Q8, "09ad9e7a61cad612ffcb9cc2a1d7dfd27b6813a6d184329404045d4691c5fdf2"},
        {"q4_1.bin", Chunk::Q4_1, "9f58d4201c4844abc937204c2b7f8d364ff44bc574b0bb87292928c54447d2c8"},
    };
    bool ok = true;
    std::vector<float> w(32 * 256), v(32 * 256);
    std::vector<uint8_t> q4k;
    for (const Case& c : cases) {
        const auto tile = slurp(dir + "/" + c.file);
        if (tile.size() != size_t(c.kind)) { std::printf("FAIL: %s is %zu bytes\n", c.file, tile.size()); return 1; }
        onebit::npu::q4nx::dequant_tile(c.kind, tile.data(), w.data(), 256);
        const std::string got = onebit::sha256_hex(reinterpret_cast<const uint8_t*>(w.data()), w.size() * 4);
        const bool same = got == c.want;
        ok &= same;
        std::printf("%s %-8s w[0][0] %.7g, w[31][255] %.7g\n", same ? "ok  " : "FAIL", c.file, w[0], w[32 * 256 - 1]);
        if (!same) std::printf("     sha256 %s, want %s\n", got.c_str(), c.want);
        if (c.kind == Chunk::Q4K) q4k = tile;
    }

    // Q4_K -> q4_1
    std::vector<uint8_t> q41(size_t(Chunk::Q4_1));
    onebit::npu::q4nx::q4k_to_q4_1(q4k.data(), q41.data());
    onebit::npu::q4nx::dequant_tile(Chunk::Q4K, q4k.data(), w.data(), 256);
    onebit::npu::q4nx::dequant_tile(Chunk::Q4_1, q41.data(), v.data(), 256);
    double worst = 0, worst_err = 0;
    for (int r = 0; r < 32; r++) {
        const float S = bf16(q4k.data() + 4608 + 2 * r), M = bf16(q4k.data() + 4672 + 2 * r);
        for (int c = 0; c < 256; c++) {
            const int i = (c / 32) * 32 + r;
            // half an ulp of each bf16 product: 2^(exponent - 8)
            auto half_ulp = [](double x) { return x == 0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(x)) - 8); };
            const double bound = 15 * half_ulp(double(S) * q4k[i]) + half_ulp(double(M) * q4k[256 + i]) + 1e-12;
            const double err = std::fabs(double(v[r * 256 + c]) - w[r * 256 + c]);
            worst = std::fmax(worst, err / bound);
            worst_err = std::fmax(worst_err, err);
        }
    }
    const bool conv = worst <= 1.0 + 1e-6;
    ok &= conv;
    std::printf("%s Q4_K -> q4_1: max |error| %.3g, at most %.2f of the bf16 rounding bound\n", conv ? "ok  " : "FAIL", worst_err, worst);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
