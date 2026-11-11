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

// dflash_forward_test.cpp -- the DFlash draft's host forward (npu/dflash_forward.h, engine#1780).
//
// Three kinds of arm, none needing a device:
//   1. two cases small enough to work out by hand, written out here as plain arithmetic: one block row with no
//      context (norm, value path, residuals, MLP, final norm), and one block row with one context row (the fused and
//      normalised context, the query/key norms, the rotation and the two-key softmax);
//   2. properties of the structure on a small random model: the output does not depend on where the positions start,
//      a later block row changes an earlier one (not causal), the context is read and its scale does not matter, a
//      window of one position is the same as each row alone, and a query head reads the key/value head of its group;
//   3. the real Bolt draft, when a path is given (argv[1] or ONEBIT_TEST_DFLASH_DRAFT): it decodes and one pass over
//      a made-up context is finite and position-independent. That arm does not show the draft drafts well -- it has
//      no target model -- only that the real weights go through this code.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_forward.h"

namespace {

using namespace onebit::npu::dflash;

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

bool throws(const std::function<void()>& f, const char* needle) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

struct Rng {
    uint64_t s;
    float next() {   // uniform in [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return float(double((s >> 40) & 0xFFFFFF) / double(0x800000) - 1.0);
    }
    std::vector<float> vec(size_t n, float scale) {
        std::vector<float> v(n);
        for (float& f : v) f = next() * scale;
        return v;
    }
};

std::vector<float> identity(int n) {
    std::vector<float> m(size_t(n) * size_t(n), 0.0f);
    for (int i = 0; i < n; i++) m[size_t(i) * size_t(n) + size_t(i)] = 1.0f;
    return m;
}

// hidden 2, one head of width 2, one layer, eps 0, every projection the identity except v, gate and up. The gains that sit on
// the attention's inputs are all different, so a norm applied to the wrong rows (or not at all) moves the answer.
DraftWeights tiny_model() {
    DraftWeights w;
    w.hidden = 2; w.intermediate = 2; w.heads = 1; w.kv_heads = 1; w.head_dim = 2; w.taps = 1;
    w.rms_eps = 0.0f; w.rope_theta = 10000.0f;
    w.fc = identity(2);
    w.hidden_norm = {0.5f, 3.0f};
    w.norm = {1.0f, 1.0f};
    DraftLayerWeights L;
    L.q = L.k = L.o = L.down = identity(2);
    L.v = {2.0f, 0.0f, 0.0f, 3.0f};
    // gate and up differ, and gate drives its first output negative where up is positive, so silu(gate) * up and
    // silu(up) * gate are different numbers: which of the two is activated is part of the answer.
    L.gate = {-1.0f, 0.0f, 0.0f, 2.0f};
    L.up = {3.0f, 0.0f, 0.0f, 0.5f};
    L.input_norm = {2.0f, 0.5f};
    L.q_norm = {1.5f, 0.5f};
    L.k_norm = {0.25f, 2.0f};
    L.post_norm = {1.0f, 1.0f};
    w.layers.push_back(L);
    return w;
}

void rms2(double& a, double& b) {
    const double r = std::sqrt((a * a + b * b) / 2.0);
    a /= r; b /= r;
}
void rmsg(double& a, double& b, double g0, double g1) {   // RMS norm, then the two gains
    rms2(a, b);
    a *= g0; b *= g1;
}
double silu(double a) { return a / (1.0 + std::exp(-a)); }

// What follows the attention in the tiny model, given the block row and the attention's output.
void tiny_tail(double x0, double x1, double a0, double a1, double& o0, double& o1) {
    double y0 = x0 + a0, y1 = x1 + a1;          // residual (o_proj is the identity)
    double p0 = y0, p1 = y1;
    rms2(p0, p1);                                // post_norm
    y0 += silu(-1.0 * p0) * (3.0 * p0);          // silu(gate(p)) * up(p); down is the identity
    y1 += silu(2.0 * p1) * (0.5 * p1);
    rms2(y0, y1);                                // the final norm
    o0 = y0; o1 = y1;
}

void hand_cases() {
    const DraftWeights w = tiny_model();
    {   // one block row, no context: the only key is the row itself, so the attention returns v(h) exactly
        const float noise[2] = {3.0f, 4.0f};
        double h0 = 3.0, h1 = 4.0;
        rmsg(h0, h1, 2.0, 0.5);                  // input_norm
        double e0, e1;
        tiny_tail(3.0, 4.0, 2.0 * h0, 3.0 * h1, e0, e1);
        const std::vector<float> out = draft_forward(w, nullptr, 0, noise, 1);
        check(out.size() == 2 && std::fabs(out[0] - e0) < 1e-5 && std::fabs(out[1] - e1) < 1e-5,
              "one row, no context: the output is the hand-computed one");
        std::printf("hand case 1: got %.6f %.6f, expected %.6f %.6f\n", out[0], out[1], e0, e1);
    }
    {   // one context row and one block row: two keys, at positions 5 and 6
        const float tap[2] = {1.0f, 0.5f};
        const float noise[2] = {-2.0f, 1.0f};
        double c0 = 1.0, c1 = 0.5;
        rmsg(c0, c1, 0.5, 3.0);                  // fc is the identity, then hidden_norm; no input_norm on this row
        double h0 = -2.0, h1 = 1.0;
        rmsg(h0, h1, 2.0, 0.5);                  // input_norm, the block row only
        double qn0 = h0, qn1 = h1, kcn0 = c0, kcn1 = c1, khn0 = h0, khn1 = h1;
        rmsg(qn0, qn1, 1.5, 0.5);                // q_norm
        rmsg(kcn0, kcn1, 0.25, 2.0);             // k_norm, on both keys
        rmsg(khn0, khn1, 0.25, 2.0);
        auto rot = [](double a, double b, double ang, double& ra, double& rb) {
            ra = a * std::cos(ang) - b * std::sin(ang);
            rb = b * std::cos(ang) + a * std::sin(ang);
        };
        double q0, q1, kc0, kc1, kh0, kh1;
        rot(qn0, qn1, 6.0, q0, q1);              // head_dim 2: one pair, turned by its position
        rot(kcn0, kcn1, 5.0, kc0, kc1);
        rot(khn0, khn1, 6.0, kh0, kh1);
        const double sc = (q0 * kc0 + q1 * kc1) / std::sqrt(2.0), sh = (q0 * kh0 + q1 * kh1) / std::sqrt(2.0);
        const double wc = std::exp(sc) / (std::exp(sc) + std::exp(sh)), wh = 1.0 - wc;
        const double a0 = wc * 2.0 * c0 + wh * 2.0 * h0, a1 = wc * 3.0 * c1 + wh * 3.0 * h1;   // v = diag(2, 3)
        double e0, e1;
        tiny_tail(-2.0, 1.0, a0, a1, e0, e1);
        const std::vector<float> out = draft_forward(w, tap, 1, noise, 1, 5);
        check(out.size() == 2 && std::fabs(out[0] - e0) < 1e-5 && std::fabs(out[1] - e1) < 1e-5,
              "one context row and one block row: the output is the hand-computed one");
        std::printf("hand case 2: got %.6f %.6f, expected %.6f %.6f (context weight %.4f)\n", out[0], out[1], e0, e1, wc);
    }
}

DraftWeights random_model(uint64_t seed) {
    Rng r{seed};
    DraftWeights w;
    w.hidden = 8; w.intermediate = 12; w.heads = 4; w.kv_heads = 2; w.head_dim = 4; w.taps = 2;
    w.rms_eps = 1e-6f; w.rope_theta = 10000.0f;
    const size_t H = 8, I = 12, Q = 16, KV = 8;
    w.fc = r.vec(H * H * 2, 0.5f);
    auto gain = [&](size_t n) {
        std::vector<float> g = r.vec(n, 0.3f);
        for (float& f : g) f += 1.0f;
        return g;
    };
    w.hidden_norm = gain(H);
    w.norm = gain(H);
    for (int l = 0; l < 2; l++) {
        DraftLayerWeights L;
        L.q = r.vec(Q * H, 0.6f); L.k = r.vec(KV * H, 0.6f); L.v = r.vec(KV * H, 0.6f); L.o = r.vec(H * Q, 0.4f);
        L.gate = r.vec(I * H, 0.5f); L.up = r.vec(I * H, 0.5f); L.down = r.vec(H * I, 0.4f);
        L.input_norm = gain(H); L.post_norm = gain(H); L.q_norm = gain(4); L.k_norm = gain(4);
        w.layers.push_back(L);
    }
    return w;
}

void properties() {
    const DraftWeights w = random_model(7);
    Rng r{99};
    const int ctx = 3, block = 4, H = 8;
    const std::vector<float> taps = r.vec(size_t(ctx) * 16, 1.0f);
    const std::vector<float> noise = r.vec(size_t(block) * size_t(H), 1.0f);
    const std::vector<float> base = draft_forward(w, taps.data(), ctx, noise.data(), block);
    check(base.size() == size_t(block) * size_t(H), "the output is block x hidden");

    check(max_diff(base, draft_forward(w, taps.data(), ctx, noise.data(), block, 1000)) < 2e-4,
          "the output does not depend on where the positions start");

    {
        std::vector<float> n2 = noise;
        n2[size_t(block - 1) * size_t(H) + 2] += 0.5f;   // the last row
        const std::vector<float> o = draft_forward(w, taps.data(), ctx, n2.data(), block);
        double first_row = 0.0;
        for (int j = 0; j < H; j++) first_row = std::max(first_row, std::fabs(double(o[size_t(j)]) - double(base[size_t(j)])));
        check(first_row > 1e-4, "a change in the last block row reaches the first one: the draft is not causal");
    }
    {
        std::vector<float> t2 = taps;
        t2[5] += 0.5f;
        check(max_diff(base, draft_forward(w, t2.data(), ctx, noise.data(), block)) > 1e-4, "the context is read");
        check(max_diff(base, draft_forward(w, nullptr, 0, noise.data(), block)) > 1e-4,
              "a pass without context differs from one with it");
        std::vector<float> t7 = taps;
        for (float& f : t7) f *= 7.0f;
        check(max_diff(base, draft_forward(w, t7.data(), ctx, noise.data(), block)) < 2e-4,
              "the context's scale does not matter: it is normalised after fusion, before every layer");
    }
    {
        check(max_diff(base, draft_forward(w, taps.data(), ctx, noise.data(), block, 0, ctx + block)) == 0.0,
              "a window as wide as the pass hides nothing");
        const std::vector<float> w1 = draft_forward(w, taps.data(), ctx, noise.data(), block, 0, 1);
        std::vector<float> alone;
        for (int i = 0; i < block; i++) {
            const std::vector<float> one = draft_forward(w, nullptr, 0, noise.data() + size_t(i) * size_t(H), 1);
            alone.insert(alone.end(), one.begin(), one.end());
        }
        check(max_diff(w1, alone) < 2e-4, "a window of one position is each block row on its own");
        check(max_diff(w1, base) > 1e-4, "and that differs from the unwindowed pass");
    }
    {   // query heads 0,1 read key/value head 0 and heads 2,3 read head 1. Keep only head 2's columns of o_proj
        // (in a one-layer model), then the output must ignore key/value head 0 and follow head 1.
        DraftWeights g = random_model(11);
        g.layers.resize(1);
        const int D = 4, Q = 16;
        for (int n = 0; n < H; n++)
            for (int col = 0; col < Q; col++)
                if (col / D != 2) g.layers[0].o[size_t(n) * size_t(Q) + size_t(col)] = 0.0f;
        const std::vector<float> ref = draft_forward(g, taps.data(), ctx, noise.data(), block);
        DraftWeights g0 = g, g1 = g;
        for (int row = 0; row < D; row++)
            for (int col = 0; col < H; col++) {
                g0.layers[0].v[size_t(row) * size_t(H) + size_t(col)] += 0.25f;           // value head 0
                g1.layers[0].v[size_t(row + D) * size_t(H) + size_t(col)] += 0.25f;       // value head 1
            }
        check(max_diff(ref, draft_forward(g0, taps.data(), ctx, noise.data(), block)) == 0.0,
              "query head 2 does not read key/value head 0");
        check(max_diff(ref, draft_forward(g1, taps.data(), ctx, noise.data(), block)) > 1e-4,
              "query head 2 reads key/value head 1");
    }
}

void refusals() {
    const DraftWeights w = random_model(3);
    const std::vector<float> noise(8, 0.5f);
    check(throws([&] { draft_forward(w, nullptr, 0, noise.data(), 0); }, "empty block"), "an empty block is refused");
    check(throws([&] { draft_forward(w, nullptr, 2, noise.data(), 1); }, "without context rows"),
          "a context length with no rows is refused");
    check(throws([&] { draft_forward(w, nullptr, 0, noise.data(), 1, 0, -1); }, "negative window"),
          "a negative window is refused");
    {
        DraftWeights b = w;
        b.layers[1].k.pop_back();
        check(throws([&] { draft_forward(b, nullptr, 0, noise.data(), 1); }, "layer 1: k_proj"),
              "a projection of the wrong size is refused with its layer");
    }
    {
        DraftWeights b = w;
        b.kv_heads = 3;
        check(throws([&] { check_weights(b); }, "not a multiple"), "heads that do not divide into key/value heads are refused");
    }
    {
        DraftWeights b = w;
        b.head_dim = 3;
        check(throws([&] { check_weights(b); }, "odd"), "an odd head width is refused");
    }
    {
        std::vector<float> bad = noise;
        bad[3] = NAN;
        check(throws([&] { draft_forward(w, nullptr, 0, bad.data(), 1); }, "non-finite"), "a non-finite result is refused");
    }
}

void real_package(const std::string& dir) {
    DraftWeights w;
    try {
        w = decode_draft(load_draft(dir));
    } catch (const std::exception& e) {
        check(false, std::string("the real package decodes: ") + e.what());
        return;
    }
    const int ctx = 2, block = 8;
    Rng r{2026};
    const std::vector<float> taps = r.vec(size_t(ctx) * size_t(w.taps) * size_t(w.hidden), 1.0f);
    const std::vector<float> noise = r.vec(size_t(block) * size_t(w.hidden), 1.0f);
    try {
        const std::vector<float> a = draft_forward(w, taps.data(), ctx, noise.data(), block);
        const std::vector<float> b = draft_forward(w, taps.data(), ctx, noise.data(), block, 4096);
        double peak = 0.0;
        for (float f : a) peak = std::max(peak, std::fabs(double(f)));
        check(a.size() == size_t(block) * size_t(w.hidden), "the real package: one pass returns block x hidden");
        check(max_diff(a, b) < 1e-3 * std::max(1.0, peak), "the real package: the pass does not depend on where the positions start");
        std::printf("real package: %zu layers, hidden %d; one pass of %d context + %d block rows is finite, peak |x| %.3f, "
                    "moved by at most %.2e when the positions start at 4096\n",
                    w.layers.size(), w.hidden, ctx, block, peak, max_diff(a, b));
    } catch (const std::exception& e) {
        check(false, std::string("the real package runs: ") + e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    hand_cases();
    properties();
    refusals();
    const char* real = (argc > 1 && argv[1][0] != '\0') ? argv[1] : std::getenv("ONEBIT_TEST_DFLASH_DRAFT");
    if (real && real[0] != '\0') real_package(real);
    else std::printf("SKIP the real package: pass its directory, or set ONEBIT_TEST_DFLASH_DRAFT\n");
    if (g_failures) {
        std::printf("dflash_forward: %d of %d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    std::printf("dflash_forward: ok (%d checks)\n", g_checks);
    return 0;
}
