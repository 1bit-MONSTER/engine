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

// npu_w4g32_gate_test -- the w4g32 prefill kernel, bit-exact on the device against the CPU reference.
//
//   W4G32_GATE="<full ELF> M K N"   (e.g. ".../full_i8_Z_K5120_N6144_w4g32_m512.elf 512 5120 6144")
//
// Unset, the test SKIPS (77): a box without the ELF or the device records a skip, not a red row.  Run it inside an
// npu-hold ticket.  Two weight sets, both built through q4nx_w4g32.h exactly as the engine builds an image:
//   seed  random codes in [-8, 7], signed random per-32 scales, random int8 activations;
//   edge  every code -8, every scale +max, every activation +127: each per-32 partial at |P| = 32*127*8 = 32512 (the
//         int16 narrowing bound inside the kernel) and |C| at K*127*8*MAGMAX(K), the int32 bound MAGMAX is chosen for.
// Each set runs three times with C poisoned (0x5a) before every launch, and every element of every run must equal the
// int64 reference computed from the packed image (q4nx_w4g32::read), with A taken through pack_a() from row-major rows.
#include "q4nx_w4g32.h"

#include "rt.h"   // the engine's own NPU runtime (ONEBIT_NPU_RT_DIR): the same launch API the forward's I8Ctx uses

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace w = q4nx_w4g32;

// The reference, from the image the device reads: C[m][n] = sum_g mag[g][n] * sum_{k in g} a[m][k] * code[k][n].
static void reference(const std::vector<int8_t>& a, const std::vector<uint8_t>& img, int M, int K, int N,
                      std::vector<int32_t>& C) {
    std::vector<int8_t> code((size_t)K * N);
    std::vector<int16_t> mag((size_t)(K / 32) * N);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            int c, m;
            w::read(img.data(), K, N, k, n, &c, &m);
            code[(size_t)k * N + n] = (int8_t)c;
            if (k % 32 == 0) mag[(size_t)(k / 32) * N + n] = (int16_t)m;
        }
    C.assign((size_t)M * N, 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++) {
        std::vector<int64_t> acc((size_t)N, 0);
        std::vector<int32_t> p((size_t)N);
        for (int g = 0; g < K / 32; g++) {
            std::fill(p.begin(), p.end(), 0);
            for (int kk = 0; kk < 32; kk++) {
                const int av = a[(size_t)m * K + g * 32 + kk];
                const int8_t* row = code.data() + (size_t)(g * 32 + kk) * N;
                for (int n = 0; n < N; n++) p[(size_t)n] += av * row[n];
            }
            const int16_t* mg = mag.data() + (size_t)g * N;
            for (int n = 0; n < N; n++) acc[(size_t)n] += (int64_t)p[(size_t)n] * mg[n];
        }
        for (int n = 0; n < N; n++) C[(size_t)m * N + n] = (int32_t)acc[(size_t)n];   // |acc| < 2^31 by MAGMAX
    }
}

int main() {
    const char* spec = std::getenv("W4G32_GATE");
    if (!spec || !*spec) {
        std::printf("npu_w4g32_gate: W4G32_GATE=\"<elf> M K N\" unset -- skipped\n");
        return 77;
    }
    char elf[4096];
    int M = 0, K = 0, N = 0;
    if (std::sscanf(spec, "%4095s %d %d %d", elf, &M, &K, &N) != 4 || M % 128 || !w::fits(K, N) || N % 1024) {
        std::printf("npu_w4g32_gate: bad W4G32_GATE \"%s\" (M %% 128, K %% 128, N %% 1024)\n", spec);
        return 1;
    }
    int fails = 0;
    try {
        xrt::device dev{0u};
        xrt::elf e(elf);
        xrt::hw_context hc(dev, e);
        xrt::ext::kernel kern(hc, "main:seq");
        const size_t ab = (size_t)M * K, wb = w::w_bytes(K, N), cb = (size_t)M * N * 4;
        xrt::ext::bo ba(dev, ab), bw(dev, wb), bc(dev, cb);
        for (int set = 0; set < 2; set++) {
            const bool edge = set == 1;
            std::mt19937 rng(20261115u + (unsigned)set);
            std::vector<int8_t> code((size_t)K * N), arow((size_t)M * K);
            std::vector<float> sc((size_t)(K / 32) * N);
            for (auto& c : code) c = edge ? (int8_t)-8 : (int8_t)((int)(rng() % 16) - 8);
            for (auto& s : sc) s = edge ? 1.0f : (float)((int)(rng() % 255) - 127) * (1.0f / 1024.0f);
            for (auto& x : arow) x = edge ? (int8_t)127 : (int8_t)((int)(rng() % 255) - 127);
            std::vector<uint8_t> img(wb);
            std::vector<float> cs((size_t)N);
            if (!w::pack(code.data(), sc.data(), K, N, img.data(), cs.data())) { std::printf("pack refused\n"); return 1; }
            std::vector<int8_t> ap(ab);
            w::pack_a(arow.data(), K, M, K, ap.data());
            std::vector<int32_t> ref;
            reference(arow, img, M, K, N, ref);
            int32_t rmax = 0;
            for (int32_t v : ref) rmax = std::max(rmax, v < 0 ? -v : v);
            std::memcpy(ba.map<char*>(), ap.data(), ab);
            std::memcpy(bw.map<char*>(), img.data(), wb);
            ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            for (int rep = 0; rep < 3; rep++) {
                std::memset(bc.map<char*>(), 0x5a, cb);
                bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                xrt::run r = kern(ba, bw, bc);   // (A, W, C), as I8Ctx::sync_and_launch
                const auto st = r.wait(std::chrono::milliseconds(30000));
                bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const int32_t* c = bc.map<int32_t*>();
                size_t bad = 0, first = 0;
                for (size_t i = 0; i < (size_t)M * N; i++)
                    if (c[i] != ref[i]) { if (!bad) first = i; ++bad; }
                std::printf("  %s rep %d state=%d mismatches=%zu/%zu max|C|=%d%s\n", edge ? "edge" : "seed", rep, (int)st,
                            bad, (size_t)M * N, rmax,
                            bad ? (" first m=" + std::to_string(first / N) + " n=" + std::to_string(first % N) +
                                   " dev=" + std::to_string(c[first]) + " ref=" + std::to_string(ref[first])).c_str() : "");
                fails += bad != 0;
            }
        }
    } catch (const std::exception& ex) {
        std::printf("npu_w4g32_gate: device error: %s\n", ex.what());
        return 1;
    }
    std::printf("npu_w4g32_gate: %s  (%s M=%d K=%d N=%d)\n", fails ? "FAIL" : "PASS, bit-exact x3 on seed and edge", elf, M,
                K, N);
    return fails ? 1 : 0;
}
