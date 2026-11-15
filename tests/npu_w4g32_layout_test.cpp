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

// npu_w4g32_layout_test -- the w4g32 weight image contract (q4nx_w4g32.h), checked without a device.
//
// The packed image goes straight into the weight BO, so a header that disagrees with the kernel's tap
// multiplies the wrong pairs and the output is silently garbage.  This test restates the contract
// from the b4 layout's own definition (npu-kernels layout_w4.py: codes [nt][ki][kb][nb][r][cc] from
// [ki][kb][r][nt][nb][cc]; mags [nt][ki][nb][g][cc]) rather than from the header:
//
//   1. the image size, MAGMAX(K) at the model's K values, and the int32 bound MAGMAX guarantees;
//   2. index() is a bijection: every code nibble is claimed once, every magnitude byte by exactly 32
//      elements of one column, and both agree with the restated layout element for element;
//   3. factor() recovers (c, s) from an assembled B = c * s and S, and refuses each way the input can
//      fail to be that (S not constant over a group, B/S not an integer, |c| out of int4, B != 0 at S == 0);
//   4. pack()/read() round-trip the codes, and the magnitudes are rint(s / cs) with cs = max|s| / MAGMAX;
//   5. pack_a() is the restated A order;
//   6. ref_dot() equals the dense int64 sum over (code, mag);
//   7. optional, when W4G32_VEC="<dir> M K N" names a gate vector set (W.bin, packed A.bin, C_ref.bin
//      from npu-kernels W4-g128/gen_q4.py, the set the NPU reproduced bit-exact): ref_dot() on that W
//      and A reproduces C_ref, so this header reads the device's own bytes the way the device does.
#include "q4nx_w4g32.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace w = q4nx_w4g32;

static int fails = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
            ++fails;                                      \
        }                                                 \
    } while (0)

// layout_w4.py, restated: byte of the code of (k, n) and byte of its magnitude.
static size_t restated_code_byte(int K, int k, int n, bool* high) {
    const int KTt = K / 128;
    const int ki = k / 128, kb = (k % 128) / 16, r = k % 16, nt = n / 128, nb = (n % 128) / 16, cc = n % 16;
    // flat index in [nt][ki][kb][nb][r][cc] over the code region of a tile, then the tile base
    const size_t e = ((((size_t)kb * 8 + nb) * 16 + r) * 16 + cc);
    *high = (e & 1) != 0;
    return ((size_t)nt * KTt + ki) * 8704 + e / 2;
}
static size_t restated_mag_byte(int K, int k, int n) {
    const int KTt = K / 128;
    const int ki = k / 128, g = (k % 128) / 32, nt = n / 128, nb = (n % 128) / 16, cc = n % 16;
    return ((size_t)nt * KTt + ki) * 8704 + 8192 + ((size_t)nb * 4 + g) * 16 + cc;
}

static void test_sizes() {
    CHECK(w::TILE == 8704 && w::CODE_BYTES == 8192, "tile %d", w::TILE);
    CHECK(w::w_bytes(5120, 17408) == 47349760u, "U image %zu", w::w_bytes(5120, 17408));  // = gen_q4's W
    CHECK(w::magmax(5120) == 127 && w::magmax(6144) == 127, "magmax 5120/6144");
    CHECK(w::magmax(17408) == 121, "magmax(17408) = %d", w::magmax(17408));
    for (int K = 128; K <= 65536; K += 128) {
        const long long worst = (long long)K * 127 * 8 * w::magmax(K);
        CHECK(worst <= 2147483647LL, "K=%d worst |C| %lld", K, worst);
        CHECK(w::magmax(K) == 127 || (long long)K * 127 * 8 * (w::magmax(K) + 1) > 2147483647LL, "K=%d not tight", K);
    }
    CHECK(!w::fits(5120, 100) && !w::fits(100, 1024) && w::fits(256, 128), "fits");
}

static void test_bijection() {
    const int K = 384, N = 256;
    std::vector<int> nib(w::w_bytes(K, N) * 2, 0), magc(w::w_bytes(K, N), 0), magcol(w::w_bytes(K, N), -1);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            const w::offsets o = w::index(K, N, k, n);
            bool hi;
            const size_t rb = restated_code_byte(K, k, n, &hi);
            CHECK(o.code == rb && o.high == hi, "code (%d,%d): %zu/%d vs %zu/%d", k, n, o.code, o.high, rb, hi);
            CHECK(o.mag == restated_mag_byte(K, k, n), "mag (%d,%d)", k, n);
            CHECK(o.code % w::TILE < (size_t)w::CODE_BYTES && o.mag % w::TILE >= (size_t)w::CODE_BYTES, "regions");
            nib[o.code * 2 + o.high]++;
            magc[o.mag]++;
            CHECK(magcol[o.mag] < 0 || magcol[o.mag] == n, "mag byte shared by two columns");
            magcol[o.mag] = n;
        }
    size_t codes = 0, mags = 0;
    for (size_t b = 0; b < w::w_bytes(K, N); b++) {
        if (b % w::TILE < (size_t)w::CODE_BYTES) {
            CHECK(nib[b * 2] == 1 && nib[b * 2 + 1] == 1, "byte %zu nibbles %d %d", b, nib[b * 2], nib[b * 2 + 1]);
            codes++;
        } else {
            CHECK(magc[b] == 32, "mag byte %zu claimed %d times", b, magc[b]);
            mags++;
        }
    }
    CHECK(codes * 2 == (size_t)K * N && mags * 32 == (size_t)K * N, "coverage");
}

static void test_factor_pack() {
    const int K = 256, N = 128;
    std::mt19937 rng(7);
    std::vector<int8_t> c(K * N);
    std::vector<float> s((K / 32) * N), B(K * N), S(K * N);
    for (auto& x : c) x = (int8_t)((int)(rng() % 16) - 8);
    for (auto& x : s) x = std::ldexp((float)((int)(rng() % 255) - 127), -14);   // signed, exact in bf16
    s[3] = 0.0f;                                                                // an all-zero group
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            const float sg = s[(k / 32) * N + n];
            S[k * N + n] = sg;
            B[k * N + n] = sg == 0.0f ? 0.0f : (float)c[k * N + n] * sg;
        }
    std::vector<int8_t> c2(K * N);
    std::vector<float> s2((K / 32) * N);
    CHECK(w::factor(B.data(), K, N, c2.data(), s2.data()), "factor refused a valid input");
    // any exact form is acceptable: check B = c2 * s2 element for element, and |c2| fits int4
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            const float sg = s2[(k / 32) * N + n];
            CHECK(c2[k * N + n] >= -8 && c2[k * N + n] <= 7, "code range");
            CHECK((float)c2[k * N + n] * sg == B[k * N + n], "factor (%d,%d) not exact", k, n);
        }
    // and where the group holds a -8 code the container's own (c, s) is the one found (q = 8 tried first)
    int same = 0, groups = 0;
    for (int g = 0; g < K / 32; g++)
        for (int n = 0; n < N; n++) {
            bool has8 = false;
            for (int kk = 0; kk < 32; kk++) has8 |= c[(g * 32 + kk) * N + n] == -8;
            if (!has8 || s[g * N + n] == 0.0f) continue;
            ++groups;
            same += s2[g * N + n] == s[g * N + n];
        }
    CHECK(groups > 0 && same == groups, "own scale recovered in %d of %d groups with a -8", same, groups);
    // a group of only even codes may be represented at twice the scale: still exact, checked above
    // refusals
    auto refused = [&](std::vector<float> b, const char* what) {
        CHECK(!w::factor(b.data(), K, N, c2.data(), s2.data()), "factor accepted: %s", what);
    };
    { auto b = B; size_t i = 40 * N + 1; while (S[i] == 0.0f) i++; b[i] = 0.37f * S[i]; refused(b, "a non-multiple element"); }
    { auto b = B; b[50 * N + 7] = std::nanf(""); refused(b, "NaN"); }
    { auto b = B; for (int kk = 0; kk < 32; kk++) b[(64 + kk) * N + 3] = (float)(kk * 9 - 140); refused(b, "a 32-level ramp (needs > 4 bits)"); }

    std::vector<uint8_t> img(w::w_bytes(K, N));
    std::vector<float> cs(N);
    CHECK(w::pack(c2.data(), s2.data(), K, N, img.data(), cs.data()), "pack");
    const int mm = w::magmax(K);
    for (int n = 0; n < N; n++) {
        float amax = 0;
        for (int g = 0; g < K / 32; g++) amax = std::fmax(amax, std::fabs(s2[g * N + n]));
        CHECK(cs[n] == (amax > 0 ? amax / mm : 1.0f), "cs[%d]", n);
    }
    int maxmag = 0;
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            int cc, mg;
            w::read(img.data(), K, N, k, n, &cc, &mg);
            CHECK(cc == c2[k * N + n], "read code (%d,%d)", k, n);
            const long want = std::lrint(s2[(k / 32) * N + n] / cs[n]);
            CHECK(mg == want, "mag (%d,%d): %d vs %ld", k, n, mg, want);
            maxmag = std::max(maxmag, std::abs(mg));
        }
    CHECK(maxmag == mm, "the column max reaches MAGMAX (%d vs %d)", maxmag, mm);
    std::vector<int8_t> bad = c2;
    bad[17] = 8;
    CHECK(!w::pack(bad.data(), s2.data(), K, N, img.data(), cs.data()), "pack accepted code 8");
}

static void test_a_and_ref() {
    const int M = 64, K = 256, N = 128, ld = K + 32;
    std::mt19937 rng(11);
    std::vector<int8_t> a(M * ld), ap(M * K), c(K * N);
    for (auto& x : a) x = (int8_t)((int)(rng() % 255) - 127);
    for (auto& x : c) x = (int8_t)((int)(rng() % 16) - 8);
    CHECK(w::pack_a(a.data(), ld, M, K, ap.data()), "pack_a");
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            // A.reshape(M/32, 8, 4, K/128, 8, 16).transpose(0, 3, 1, 4, 2, 5)
            const int rt = m / 32, mb = (m % 32) / 4, r = m % 4, ki = k / 128, kb = (k % 128) / 16, cc = k % 16;
            const size_t o = (((((size_t)rt * (K / 128) + ki) * 8 + mb) * 8 + kb) * 4 + r) * 16 + cc;
            CHECK(ap[o] == a[(size_t)m * ld + k], "pack_a (%d,%d)", m, k);
        }
    CHECK(!w::pack_a(a.data(), ld, 48, K, ap.data()), "pack_a accepted M=48");
    std::vector<float> s((K / 32) * N);
    for (auto& x : s) x = (float)((int)(rng() % 255) - 127);
    std::vector<uint8_t> img(w::w_bytes(K, N));
    std::vector<float> cs(N);
    CHECK(w::pack(c.data(), s.data(), K, N, img.data(), cs.data()), "pack");
    for (int m = 0; m < M; m += 7)
        for (int n = 0; n < N; n += 5) {
            int64_t dense = 0;
            for (int k = 0; k < K; k++) {
                int cc, mg;
                w::read(img.data(), K, N, k, n, &cc, &mg);
                dense += (int64_t)a[(size_t)m * ld + k] * cc * mg;
            }
            CHECK(w::ref_dot(a.data() + (size_t)m * ld, img.data(), K, N, n) == dense, "ref_dot (%d,%d)", m, n);
        }
}

static bool slurp(const std::string& p, std::vector<char>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    v.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

static void test_device_vectors() {
    const char* e = std::getenv("W4G32_VEC");
    if (!e || !*e) { std::printf("  (device-vector leg skipped: W4G32_VEC unset)\n"); return; }
    char dir[4096];
    int M, K, N;
    if (std::sscanf(e, "%4095s %d %d %d", dir, &M, &K, &N) != 4) { CHECK(false, "W4G32_VEC=\"dir M K N\""); return; }
    std::vector<char> W, A, C;
    if (!slurp(std::string(dir) + "/W.bin", W) || !slurp(std::string(dir) + "/A.bin", A) || !slurp(std::string(dir) + "/C_ref.bin", C)) {
        CHECK(false, "cannot read %s/{W,A,C_ref}.bin", dir);
        return;
    }
    CHECK(W.size() == w::w_bytes(K, N) && A.size() == (size_t)M * K && C.size() == (size_t)M * N * 4, "vector sizes");
    if (fails) return;
    // un-pack A with the restated order, check pack_a gives the file back, then sample C_ref
    std::vector<int8_t> a((size_t)M * K), ap((size_t)M * K);
    size_t o = 0;
    for (int rt = 0; rt < M / 32; rt++)
        for (int ki = 0; ki < K / 128; ki++)
            for (int mb = 0; mb < 8; mb++)
                for (int kb = 0; kb < 8; kb++)
                    for (int r = 0; r < 4; r++)
                        for (int cc = 0; cc < 16; cc++) a[(size_t)(rt * 32 + mb * 4 + r) * K + ki * 128 + kb * 16 + cc] = (int8_t)A[o++];
    w::pack_a(a.data(), K, M, K, ap.data());
    CHECK(std::memcmp(ap.data(), A.data(), A.size()) == 0, "pack_a does not reproduce the device's A.bin");
    const int32_t* cr = reinterpret_cast<const int32_t*>(C.data());
    int checked = 0;
    for (int m = 0; m < M; m += 61)
        for (int n = 0; n < N; n += 97) {
            const int64_t v = w::ref_dot(a.data() + (size_t)m * K, reinterpret_cast<const uint8_t*>(W.data()), K, N, n);
            CHECK(v == cr[(size_t)m * N + n], "C_ref (%d,%d): %lld vs %d", m, n, (long long)v, cr[(size_t)m * N + n]);
            ++checked;
        }
    std::printf("  device-vector leg: %s M=%d K=%d N=%d, %d C_ref entries reproduced\n", dir, M, K, N, checked);
}

int main() {
    test_sizes();
    test_bijection();
    test_factor_pack();
    test_a_and_ref();
    test_device_vectors();
    if (fails) { std::printf("npu_w4g32_layout_test: %d failure(s)\n", fails); return 1; }
    std::printf("npu_w4g32_layout_test: OK\n");
    return 0;
}
