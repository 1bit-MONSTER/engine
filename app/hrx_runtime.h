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

// engine#713 criterion 4 (docs/hrx.md, "The HSA runtime"): the HRX init line names the HSA runtime
// HRX will dlopen - its path and its sha256 - because several libhsa-runtime64.so.1 copies exist on a box
// and they are not interchangeable: a distro ROCr predates an entry point the driver resolves by name, and
// a copy of the right build without its librocm_sysdeps_* siblings cannot be loaded at all. Without naming
// it, a run can pass while a silent fallback to another copy executes, which is how the A/B behind
// engine#315's residual was only trustworthy - every arm named its runtime.
//
// Anything that needs the same identity (llama-server's log, a test) gets it from here, so the
// format is written once.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace onebit {

// SHA-256 (FIPS 180-4) of one buffer, as lower-case hex.
inline std::string sha256_hex(const uint8_t* data, size_t len) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(data, data + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; i--) m.push_back(uint8_t((uint64_t(len) * 8) >> (8 * i)));
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = uint32_t(m[off + 4 * i]) << 24 | uint32_t(m[off + 4 * i + 1]) << 16 |
                   uint32_t(m[off + 4 * i + 2]) << 8 | m[off + 4 * i + 3];
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

// SHA-256 of a whole file, or "" when it cannot be read: the HSA runtime is a few MiB, and one
// read per serve costs less than a wrong runtime costs a run.
inline std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    if (!f.good() && !f.eof()) return "";
    return sha256_hex(bytes.data(), bytes.size());
}

// The one line serve prints when HRX first resolves its runtime. `sha256` empty means the file could
// not be read, which is itself worth saying: HRX will still try to dlopen it.
inline std::string hrx_runtime_line(const std::string& path, const std::string& sha256) {
    if (sha256.empty())
        return "HRX: HSA runtime " + path + " (unreadable: no sha256, HRX may fail to open it)";
    return "HRX: HSA runtime " + path + " sha256 " + sha256;
}

}  // namespace onebit
