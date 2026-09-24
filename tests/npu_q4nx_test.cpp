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

// SHA-256 (FIPS 180-4), enough for one 32 KiB buffer
std::string sha256(const uint8_t* data, size_t len) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(data, data + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; i--) m.push_back(uint8_t((uint64_t(len) * 8) >> (8 * i)));
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = uint32_t(m[off + 4 * i]) << 24 | uint32_t(m[off + 4 * i + 1]) << 16 | uint32_t(m[off + 4 * i + 2]) << 8 | m[off + 4 * i + 3];
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char hex[65];
    for (int i = 0; i < 8; i++) std::snprintf(hex + 8 * i, 9, "%08x", h[i]);
    return hex;
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
        const std::string got = sha256(reinterpret_cast<const uint8_t*>(w.data()), w.size() * 4);
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
