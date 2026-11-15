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
// act_outliers_test: the parts of the engine#2536 outlier split that need no device (npu/forward/include/act_outliers.h).
//
//   1. picking: an outlier row yields exactly its outlier channels; a flat row yields none, so a model without
//      outliers keeps the plain launch; kmax keeps the largest; tau <= 0 is off.
//   2. the arithmetic: on a row shaped like Qwen2.5-3B's ffn_down input (one channel ~100x rms), a per-row int8 GEMM
//      misses the f32 result badly, and the same GEMM with the split (outlier zeroed in the int8 row, x_c * W[:,c]
//      added in f32) lands within the int8 noise of the rest.
//   3. the column-block gather: decoding the gathered tiles of one 256-wide block as a [rows, 256] tensor gives the
//      same bits as that block of the full decode, on real q4_1 tiles.

#include "act_outliers.h"
#include "q4nx_dequant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(c, ...)                                         \
    do {                                                      \
        if (!(c)) {                                           \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                \
            std::fprintf(stderr, "\n");                       \
            ++failures;                                       \
        }                                                     \
    } while (0)

using onebit::npu::ActOutlier;

// The per-row int8 GEMM the i8 path computes: one scale amax/127 per row, codes in [-127, 127], f32 weights.
static void gemm_i8_row(const std::vector<float>& x, const std::vector<float>& W, int K, int N, std::vector<float>& y) {
    float amax = 0.0f;
    for (float v : x) amax = std::max(amax, std::fabs(v));
    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
    y.assign((size_t)N, 0.0f);
    for (int k = 0; k < K; k++) {
        const float q = std::round(x[(size_t)k] / s) * s;
        for (int n = 0; n < N; n++) y[(size_t)n] += q * W[(size_t)n * K + k];
    }
}

static double rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
    double e = 0.0, r = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        e += (double(a[i]) - ref[i]) * (double(a[i]) - ref[i]);
        r += double(ref[i]) * ref[i];
    }
    return std::sqrt(e / std::max(r, 1e-30));
}

static void test_pick() {
    const int w = 4096;   // the rms includes the outliers themselves: a narrow row would lift 20*rms over the second one
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> A(2 * (size_t)w);
    for (auto& v : A) v = g(rng);
    A[100] = 400.0f;    // row 0: two outliers
    A[700] = -250.0f;
    // row 1 stays flat: N(0,1), so nothing is above 20 * rms
    std::vector<ActOutlier> p;
    onebit::npu::act_outlier_pick(A.data(), 2, w, w, 20.0f, 16, p);
    CHECK(p.size() == 2, "outlier row: want 2 picks, got %zu", p.size());
    bool has100 = false, has700 = false;
    for (const auto& e : p) {
        CHECK(e.row == 0, "a pick on the flat row (row %d ch %d)", e.row, e.ch);
        if (e.ch == 100) has100 = (e.x == 400.0f);
        if (e.ch == 700) has700 = (e.x == -250.0f);
    }
    CHECK(has100 && has700, "the picks are not channels 100 (400) and 700 (-250)");

    std::vector<ActOutlier> flat;
    onebit::npu::act_outlier_pick(A.data() + w, 1, w, w, 20.0f, 16, flat);
    CHECK(flat.empty(), "flat row: want no picks (the plain launch), got %zu", flat.size());

    std::vector<ActOutlier> one;
    onebit::npu::act_outlier_pick(A.data(), 1, w, w, 20.0f, 1, one);
    CHECK(one.size() == 1 && one[0].ch == 100, "kmax 1 keeps the largest |x| (want ch 100)");

    std::vector<ActOutlier> off;
    onebit::npu::act_outlier_pick(A.data(), 2, w, w, 0.0f, 16, off);
    CHECK(off.empty(), "tau 0 must pick nothing, got %zu", off.size());

    // Row stride wider than the real width: the pad past w is never a pick.
    std::vector<float> B(2 * (size_t)(w + 256), 0.0f);
    for (int m = 0; m < 2; m++)
        for (int k = 0; k < w; k++) B[(size_t)m * (w + 256) + k] = A[(size_t)m * w + k];
    B[(size_t)1 * (w + 256) + w + 10] = 1e6f;   // in the pad of row 1
    std::vector<ActOutlier> ps;
    onebit::npu::act_outlier_pick(B.data(), 2, w + 256, w, 20.0f, 16, ps);
    CHECK(ps.size() == 2, "strided rows: want the same 2 picks, got %zu", ps.size());
}

static void test_arithmetic() {
    const int K = 2048, N = 256;
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> x((size_t)K), W((size_t)N * K);
    for (auto& v : x) v = g(rng);
    for (auto& v : W) v = 0.02f * g(rng);
    x[1234] = 100.0f;   // ~100x the row's rms, as measured on Qwen2.5-3B's layer-35 ffn_down input

    std::vector<float> ref((size_t)N, 0.0f);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) ref[(size_t)n] += x[(size_t)k] * W[(size_t)n * K + k];

    std::vector<float> plain;
    gemm_i8_row(x, W, K, N, plain);

    std::vector<ActOutlier> p;
    onebit::npu::act_outlier_pick(x.data(), 1, K, K, 20.0f, 16, p);
    CHECK(p.size() == 1 && p[0].ch == 1234, "the split must find channel 1234");
    std::vector<float> xz = x;
    for (const auto& e : p) xz[(size_t)e.ch] = 0.0f;
    std::vector<float> split;
    gemm_i8_row(xz, W, K, N, split);
    for (const auto& e : p)
        for (int n = 0; n < N; n++) split[(size_t)n] += e.x * W[(size_t)n * K + e.ch];

    const double ep = rel_err(plain, ref), es = rel_err(split, ref);
    std::printf("  int8 row vs f32: plain rel_err %.4f, split rel_err %.4f\n", ep, es);
    CHECK(es < 0.02, "split rel_err %.4f is not at int8 noise (< 0.02)", es);
    CHECK(es * 5.0 < ep, "split (%.4f) is not at least 5x better than plain (%.4f)", es, ep);
}

static uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static void test_gather() {
    const int TR = 6, TC = 5, rows = TR * Q4NX_TILE_ROWS, cols = TC * Q4NX_TILE_COLS;
    std::vector<uint8_t> raw((size_t)TR * TC * Q4NX_TILE_BYTES);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(0.001f, 0.05f), z(-0.5f, 0.5f);
    for (size_t t = 0; t < (size_t)TR * TC; t++) {
        uint8_t* tile = &raw[t * Q4NX_TILE_BYTES];
        // [512 B bf16 scales][512 B bf16 mins][4096 B codes]: sane scales so every value is finite.
        for (int i = 0; i < 256; i++) {
            const uint16_t s = bf16_of(u(rng)), m = bf16_of(z(rng));
            std::memcpy(tile + 2 * i, &s, 2);
            std::memcpy(tile + 512 + 2 * i, &m, 2);
        }
        for (int i = 1024; i < Q4NX_TILE_BYTES; i++) tile[i] = (uint8_t)(rng() & 0xff);
    }
    std::vector<float> full((size_t)rows * cols);
    CHECK(q4nx_dequant_tensor_chunked(raw.data(), raw.size(), rows, cols, Q4NX_TILE_BYTES, full.data()) == 0,
          "full decode refused the synthetic tensor");
    for (int tc = 0; tc < TC; tc++) {
        std::vector<uint8_t> buf;
        onebit::npu::act_outlier_gather_block(raw.data(), TR, TC, Q4NX_TILE_BYTES, tc, buf);
        CHECK(buf.size() == (size_t)TR * Q4NX_TILE_BYTES, "gathered block %d has the wrong size", tc);
        std::vector<float> blk((size_t)rows * Q4NX_TILE_COLS);
        CHECK(q4nx_dequant_tensor_chunked(buf.data(), buf.size(), rows, Q4NX_TILE_COLS, Q4NX_TILE_BYTES, blk.data()) == 0,
              "block decode refused block %d", tc);
        int diff = 0;
        for (int r = 0; r < rows; r++)
            if (std::memcmp(&blk[(size_t)r * Q4NX_TILE_COLS], &full[(size_t)r * cols + (size_t)tc * Q4NX_TILE_COLS],
                            Q4NX_TILE_COLS * sizeof(float)) != 0)
                ++diff;
        CHECK(diff == 0, "block %d: %d of %d rows differ from the full decode", tc, diff, rows);
    }
    // A wrong block index must NOT match (the check above is not vacuous).
    std::vector<uint8_t> buf;
    onebit::npu::act_outlier_gather_block(raw.data(), TR, TC, Q4NX_TILE_BYTES, 1, buf);
    std::vector<float> blk((size_t)rows * Q4NX_TILE_COLS);
    q4nx_dequant_tensor_chunked(buf.data(), buf.size(), rows, Q4NX_TILE_COLS, Q4NX_TILE_BYTES, blk.data());
    CHECK(std::memcmp(blk.data(), &full[0], Q4NX_TILE_COLS * sizeof(float)) != 0,
          "block 1 decoded equal to block 0: the gather check cannot fail");
}

// engine#2576: a FUSED [gate | up] GEMM (outputs gate rows first, then up rows, as q4nx_assemble_B lays them). The
// outlier channel feeds both halves, so each half's correction must land at its own offset. A correction applied with
// the up half at offset 0 (on top of gate) must not pass.
static void test_fused_gate_up() {
    const int K = 2048, R = 128, N = 2 * R;
    std::mt19937 rng(23);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> x((size_t)K), Wg((size_t)R * K), Wu((size_t)R * K);
    for (auto& v : x) v = g(rng);
    for (auto& v : Wg) v = 0.02f * g(rng);
    for (auto& v : Wu) v = 0.02f * g(rng);
    x[777] = 90.0f;   // one channel ~90x the row's rms: the post-attention norm output on CodeLlama
    std::vector<float> W((size_t)N * K);   // fused rows: [gate; up]
    std::memcpy(W.data(), Wg.data(), Wg.size() * sizeof(float));
    std::memcpy(W.data() + Wg.size(), Wu.data(), Wu.size() * sizeof(float));
    std::vector<float> ref((size_t)N, 0.0f);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) ref[(size_t)n] += x[(size_t)k] * W[(size_t)n * K + k];
    std::vector<ActOutlier> p;
    onebit::npu::act_outlier_pick(x.data(), 1, K, K, 40.0f, 16, p);
    CHECK(p.size() == 1 && p[0].ch == 777, "fused: the split finds channel 777 at tau 40");
    std::vector<float> xz = x;
    for (const auto& e : p) xz[(size_t)e.ch] = 0.0f;
    std::vector<float> colg((size_t)R), colu((size_t)R);
    for (int n = 0; n < R; n++) { colg[(size_t)n] = Wg[(size_t)n * K + 777]; colu[(size_t)n] = Wu[(size_t)n * K + 777]; }
    std::vector<float> plain, split, wrong;
    gemm_i8_row(x, W, K, N, plain);
    gemm_i8_row(xz, W, K, N, split);
    wrong = split;
    onebit::npu::act_outlier_correct(split.data(), N, p, 0, R, [&](int) { return colg.data(); });
    onebit::npu::act_outlier_correct(split.data(), N, p, R, R, [&](int) { return colu.data(); });
    onebit::npu::act_outlier_correct(wrong.data(), N, p, 0, R, [&](int) { return colg.data(); });
    onebit::npu::act_outlier_correct(wrong.data(), N, p, 0, R, [&](int) { return colu.data(); });   // up at the WRONG offset
    const double ep = rel_err(plain, ref), es = rel_err(split, ref), ew = rel_err(wrong, ref);
    std::printf("  fused gate|up vs f32: plain %.4f, split %.4f, up-at-offset-0 %.4f\n", ep, es, ew);
    CHECK(es < 0.02, "fused: split rel_err %.4f is not at int8 noise", es);
    CHECK(es * 5.0 < ep, "fused: split (%.4f) is not 5x better than plain (%.4f)", es, ep);
    CHECK(ew > 10.0 * es, "fused: the up half at offset 0 (%.4f) must be far worse than the right offset (%.4f)", ew, es);
    // out_off past N is a no-op, not a write past the row
    std::vector<float> guard = split;
    onebit::npu::act_outlier_correct(guard.data(), N, p, N, R, [&](int) { return colu.data(); });
    CHECK(guard == split, "fused: an offset at N writes nothing");
}

int main() {
    test_pick();
    test_arithmetic();
    test_gather();
    test_fused_gate_up();
    if (failures) {
        std::fprintf(stderr, "act_outliers_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("act_outliers_test: ok\n");
    return 0;
}
