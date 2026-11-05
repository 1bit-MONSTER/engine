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

// gdn_row_bench.cpp -- where the GDN state recurrence's per-row cost actually goes.
//
// The M-row batched GDN path (`gdn_layer_batch`, npu/forward/src/q4nx_forward.cpp) measured
// `core` (the rank-1 delta recurrence) at 242.7 ms for a 4-row verify batch, i.e. ~61 ms/row over
// 48 layers, and TMTPLOOP-STATE.md read that as memory-bound state streaming (144 MB per row across
// all layers, ~2.4 GB/s).  That reading was arithmetic on a time, not a measurement: each thread
// only ever touches its OWN head's 64 KB state row, which fits in a core's L2, so the traffic story
// may be wrong and a loop exchange may buy nothing.  This bench isolates the recurrence from the
// array entirely -- it is host code on both paths -- and times four variants of it with the model's
// real shapes (Bonsai2-27B: 48 GDN layers, 48 v-heads, 128x128 state, heads 16/48 repeat 3):
//
//   A  today: per row, a parallel-for over heads, a per-head scratch vector allocated inside it.
//   B  A, with the per-head scratch hoisted to one buffer per thread (allocation removed).
//   C  B, with the loops EXCHANGED (head outer, row inner) so a head's 64 KB state row stays hot
//      across the batch's rows; the recurrence order per head is preserved, so it is exact.
//   D  C, plus host expf/softplus hoisted out of the row loop (they depend on ah/bh per row only).
//
// Deliberately absent: a variant that models the forward's L3 eviction (its weight stream sits between two
// rows of a layer).  A synthetic flush stream needs its cost measured in the same window as the arms to be
// subtracted fairly, and this box at load 28-37 swings identical code 2.6x (23.3 -> 60.5 ms/row for the same
// 1-thread section split), so that arm produced a meaningless ratio.  The eviction question is left
// unmeasured rather than answered wrongly.
//
// Each variant computes the same numbers: the bench checks B/C/D against A bit-for-bit and refuses to
// report timings if they differ, so a variant that is faster by being wrong cannot be quoted.
//
//   tools/gdn_row_bench [rows] [reps]        (device-free: this is host code on both paths)
//
// Threads follow the forward's own rule (3/4 of the procs, capped by OMP_NUM_THREADS), so the row
// numbers are comparable with what the forward actually pays with the same environment.

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

namespace {

// Bonsai2-27B's GDN geometry (from the forward's own log: "GDN linear attention: 48 layer(s),
// key_dim=2048 value_dim=6144 heads=16/48 hd=128/128 conv_k=4").
constexpr int NL = 48;     // GDN layers
constexpr int NVH = 48;    // v-heads
constexpr int KHD = 128;   // key head dim
constexpr int VHD = 128;   // value head dim
constexpr float EPS = 1e-6f;

float softplus_f(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

int heads_threads() {
    const int cap = (3 * omp_get_num_procs()) / 4;
    const int want = omp_get_max_threads();
    return want < cap ? (want > 0 ? want : 1) : (cap > 0 ? cap : 1);
}

// One layer's inputs for the batch, as gdn_layer_batch leaves them per row: q/k/v per head and the
// two scalars the gate needs.  Random but fixed, so every variant sees the same numbers.
struct Rows {
    int M;
    std::vector<float> q, k, v;   // [M][NVH][KHD or VHD], already L2-normalised
    std::vector<float> ah, bh;    // [M][NVH]
    std::vector<float> z, norm;   // [NVH][VHD], [VHD]: the gated-norm gate and weight
    std::vector<float> ssm_a, dt; // [NVH]
    std::vector<float> state0;    // [NVH][KHD][VHD] per layer, fresh per call
};

float frand(unsigned& s) {  // xorshift32 -> [-0.5, 0.5)
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float)(s % 100000) / 100000.0f - 0.5f;
}

Rows make_rows(int M) {
    Rows r;
    r.M = M;
    r.q.assign((size_t)M * NVH * KHD, 0.0f);
    r.k.assign((size_t)M * NVH * KHD, 0.0f);
    r.v.assign((size_t)M * NVH * VHD, 0.0f);
    r.ah.assign((size_t)M * NVH, 0.0f);
    r.bh.assign((size_t)M * NVH, 0.0f);
    r.z.assign((size_t)NVH * VHD, 0.0f);
    r.norm.assign((size_t)VHD, 0.0f);
    r.ssm_a.assign((size_t)NVH, 0.0f);
    r.dt.assign((size_t)NVH, 0.0f);
    r.state0.assign((size_t)NVH * KHD * VHD, 0.0f);
    unsigned s = 12345;
    // Scales that keep the recurrence in its real regime: a unit-norm q/k, a bounded v, a small state and
    // a gated-norm gate around 0.  Random values at O(1) diverge into NaN within a few rows and then the
    // timings are meaningless (and the exactness compare unusable, since NaN != NaN).
    for (auto& x : r.q) x = frand(s) * 0.088f;          // 1/sqrt(128) scale
    for (auto& x : r.k) x = frand(s) * 0.088f;
    for (auto& x : r.v) x = frand(s) * 0.5f;
    for (auto& x : r.z) x = frand(s) * 1.5f;
    for (auto& x : r.norm) x = 1.0f + frand(s) * 0.1f;
    for (auto& x : r.state0) x = frand(s) * 0.05f;
    for (int i = 0; i < NVH; i++) {
        r.ssm_a[(size_t)i] = -0.02f - 0.01f * (float)i / NVH;
        r.dt[(size_t)i] = 0.1f + 0.001f * (float)i;
        for (int m = 0; m < M; m++) {
            r.ah[(size_t)m * NVH + i] = frand(s) * 4.0f;
            r.bh[(size_t)m * NVH + i] = frand(s) * 4.0f;
        }
    }
    return r;
}

// One head, one row: the recurrence body, verbatim from gdn_layer_batch (the four simd loops), with
// the per-row gate scalars passed in so variant D can hoist them.
__attribute__((noinline)) void head_row(float* sh, const float* kh, const float* qh, const float* vh, float* crow,
                     const float* zv, const float* normv, float eg, float beta, float scale,
                     float* scratch) {
    float* kvm = scratch;
    float* delta = kvm + VHD;
#pragma omp simd
    for (int i = 0; i < KHD * VHD; i++) sh[i] *= eg;
    for (int vd = 0; vd < VHD; vd++) kvm[vd] = 0.0f;
    for (int kd = 0; kd < KHD; kd++) {
        const float k = kh[kd];
        const float* srow = sh + (size_t)kd * VHD;
#pragma omp simd
        for (int vd = 0; vd < VHD; vd++) kvm[vd] += srow[vd] * k;
    }
#pragma omp simd
    for (int vd = 0; vd < VHD; vd++) delta[vd] = (vh[vd] - kvm[vd]) * beta;
    for (int kd = 0; kd < KHD; kd++) {
        const float k = kh[kd];
        float* srow = sh + (size_t)kd * VHD;
#pragma omp simd
        for (int vd = 0; vd < VHD; vd++) srow[vd] += k * delta[vd];
    }
    float ss = 0;
    for (int vd = 0; vd < VHD; vd++) crow[vd] = 0.0f;
    for (int kd = 0; kd < KHD; kd++) {
        const float q = qh[kd];
        const float* srow = sh + (size_t)kd * VHD;
#pragma omp simd
        for (int vd = 0; vd < VHD; vd++) crow[vd] += srow[vd] * q;
    }
    for (int vd = 0; vd < VHD; vd++) { crow[vd] *= scale; ss += crow[vd] * crow[vd]; }
    const float inv = 1.0f / sqrtf(ss / (float)VHD + EPS);
    for (int vd = 0; vd < VHD; vd++) {
        const float zz = zv[vd];
        crow[vd] = crow[vd] * inv * normv[vd] * (zz / (1.0f + expf(-zz)));
    }
}

// A: today's shape -- row outer (parallel-for over heads per row), scratch allocated per head.
// The recurrence body is head_row() for every variant, so an exactness mismatch can only come from the
// loop structure being compared, never from two copies of the arithmetic drifting apart.
void variant_a(const Rows& r, int nl, std::vector<float>& core_out) {
    const int M = r.M, nt = heads_threads();
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> st = r.state0;   // one layer's state
    for (int l = 0; l < nl; l++) {
        float* rs = st.data();
        for (int m = 0; m < M; m++) {
            const float* qm = r.q.data() + (size_t)m * NVH * KHD;
            const float* km = r.k.data() + (size_t)m * NVH * KHD;
            const float* vm = r.v.data() + (size_t)m * NVH * VHD;
            float* cm = core_out.data() + ((size_t)(l * M + m)) * NVH * VHD;
            const float* ahm = r.ah.data() + (size_t)m * NVH;
            const float* bhm = r.bh.data() + (size_t)m * NVH;
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int hh = 0; hh < NVH; hh++) {
                std::vector<float> scratch((size_t)2 * VHD);
                const float eg = expf(r.ssm_a[(size_t)hh] * softplus_f(ahm[hh] + r.dt[(size_t)hh]));
                const float beta = 1.0f / (1.0f + expf(-bhm[hh]));
                head_row(rs + (size_t)hh * KHD * VHD, km + (size_t)hh * KHD, qm + (size_t)hh * KHD,
                         vm + (size_t)hh * VHD, cm + (size_t)hh * VHD, r.z.data() + (size_t)hh * VHD,
                         r.norm.data(), eg, beta, scale, scratch.data());
            }
        }
    }
}

// B: A with the scratch hoisted (one buffer per thread, no allocation in the loop).
void variant_b(const Rows& r, int nl, std::vector<float>& core_out) {
    const int M = r.M, nt = heads_threads();
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> st = r.state0;
    std::vector<float> scratch((size_t)nt * 2 * VHD);
    for (int l = 0; l < nl; l++) {
        float* rs = st.data();
        for (int m = 0; m < M; m++) {
            const float* qm = r.q.data() + (size_t)m * NVH * KHD;
            const float* km = r.k.data() + (size_t)m * NVH * KHD;
            const float* vm = r.v.data() + (size_t)m * NVH * VHD;
            float* cm = core_out.data() + ((size_t)(l * M + m)) * NVH * VHD;
            const float* ahm = r.ah.data() + (size_t)m * NVH;
            const float* bhm = r.bh.data() + (size_t)m * NVH;
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int hh = 0; hh < NVH; hh++) {
                float* sc = scratch.data() + (size_t)omp_get_thread_num() * 2 * VHD;
                const float eg = expf(r.ssm_a[(size_t)hh] * softplus_f(ahm[hh] + r.dt[(size_t)hh]));
                const float beta = 1.0f / (1.0f + expf(-bhm[hh]));
                head_row(rs + (size_t)hh * KHD * VHD, km + (size_t)hh * KHD, qm + (size_t)hh * KHD,
                         vm + (size_t)hh * VHD, cm + (size_t)hh * VHD, r.z.data() + (size_t)hh * VHD,
                         r.norm.data(), eg, beta, scale, sc);
            }
        }
    }
}

// C: loops exchanged -- head outer, row inner.  Parallel over heads as before, and each head's state
// row is read and written by one thread for the whole batch, so it stays in that core's cache.
void variant_cd(const Rows& r, int nl, std::vector<float>& core_out, bool hoist_gates) {
    const int M = r.M, nt = heads_threads();
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> st = r.state0;
    std::vector<float> scratch((size_t)nt * 2 * VHD);
    std::vector<float> gates;
    if (hoist_gates) gates.assign((size_t)M * NVH * 2, 0.0f);
    for (int l = 0; l < nl; l++) {
        float* rs = st.data();
        if (hoist_gates)
            for (int m = 0; m < M; m++)
                for (int hh = 0; hh < NVH; hh++) {
                    gates[(size_t)(m * NVH + hh) * 2] =
                        expf(r.ssm_a[(size_t)hh] *
                             softplus_f(r.ah[(size_t)m * NVH + hh] + r.dt[(size_t)hh]));
                    gates[(size_t)(m * NVH + hh) * 2 + 1] =
                        1.0f / (1.0f + expf(-r.bh[(size_t)m * NVH + hh]));
                }
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int hh = 0; hh < NVH; hh++) {
            float* sh = rs + (size_t)hh * KHD * VHD;
            float* sc = scratch.data() + (size_t)omp_get_thread_num() * 2 * VHD;
            for (int m = 0; m < M; m++) {   // rows stay in order: the state carries across them
                float eg, beta;
                if (hoist_gates) {
                    eg = gates[(size_t)(m * NVH + hh) * 2];
                    beta = gates[(size_t)(m * NVH + hh) * 2 + 1];
                } else {
                    eg = expf(r.ssm_a[(size_t)hh] *
                              softplus_f(r.ah[(size_t)m * NVH + hh] + r.dt[(size_t)hh]));
                    beta = 1.0f / (1.0f + expf(-r.bh[(size_t)m * NVH + hh]));
                }
                head_row(sh, r.k.data() + ((size_t)m * NVH + hh) * KHD,
                         r.q.data() + ((size_t)m * NVH + hh) * KHD,
                         r.v.data() + ((size_t)m * NVH + hh) * VHD,
                         core_out.data() + ((size_t)(l * M + m)) * NVH * VHD + (size_t)hh * VHD,
                         r.z.data() + (size_t)hh * VHD, r.norm.data(), eg, beta, scale, sc);
            }
        }
    }
}


// P: single-threaded attribution of the recurrence body itself, section by section (no parallel region,
// no allocator), so the four parts can be named.  Times are per (layer, row) summed over the batch.
struct Sections {
    double decay = 0, kvm = 0, upd = 0, out = 0;
};
Sections section_profile(const Rows& r, int nl) {
    const int M = r.M;
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> st = r.state0;
    std::vector<float> core((size_t)nl * M * NVH * VHD, 0.0f);
    std::vector<float> scratch((size_t)2 * VHD);
    Sections sec;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto add_ms = [](auto& acc, auto t0) {
        acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    for (int l = 0; l < nl; l++) {
        float* rs = st.data();
        for (int m = 0; m < M; m++) {
            for (int hh = 0; hh < NVH; hh++) {
                float* sh = rs + (size_t)hh * KHD * VHD;
                const float* kh = r.k.data() + ((size_t)m * NVH + hh) * KHD;
                const float* qh = r.q.data() + ((size_t)m * NVH + hh) * KHD;
                const float* vh = r.v.data() + ((size_t)m * NVH + hh) * VHD;
                float* crow = core.data() + ((size_t)(l * M + m)) * NVH * VHD + (size_t)hh * VHD;
                const float* zv = r.z.data() + (size_t)hh * VHD;
                const float eg = expf(r.ssm_a[(size_t)hh] *
                                      softplus_f(r.ah[(size_t)m * NVH + hh] + r.dt[(size_t)hh]));
                const float beta = 1.0f / (1.0f + expf(-r.bh[(size_t)m * NVH + hh]));
                float* kvm = scratch.data();
                float* delta = kvm + VHD;
                auto t0 = now();
                for (int i = 0; i < KHD * VHD; i++) sh[i] *= eg;
                add_ms(sec.decay, t0);
                t0 = now();
                for (int vd = 0; vd < VHD; vd++) kvm[vd] = 0.0f;
                for (int kd = 0; kd < KHD; kd++) {
                    const float k = kh[kd];
                    const float* srow = sh + (size_t)kd * VHD;
                    for (int vd = 0; vd < VHD; vd++) kvm[vd] += srow[vd] * k;
                }
                for (int vd = 0; vd < VHD; vd++) delta[vd] = (vh[vd] - kvm[vd]) * beta;
                add_ms(sec.kvm, t0);
                t0 = now();
                for (int kd = 0; kd < KHD; kd++) {
                    const float k = kh[kd];
                    float* srow = sh + (size_t)kd * VHD;
                    for (int vd = 0; vd < VHD; vd++) srow[vd] += k * delta[vd];
                }
                add_ms(sec.upd, t0);
                t0 = now();
                float ss = 0;
                for (int vd = 0; vd < VHD; vd++) crow[vd] = 0.0f;
                for (int kd = 0; kd < KHD; kd++) {
                    const float q = qh[kd];
                    const float* srow = sh + (size_t)kd * VHD;
                    for (int vd = 0; vd < VHD; vd++) crow[vd] += srow[vd] * q;
                }
                for (int vd = 0; vd < VHD; vd++) { crow[vd] *= scale; ss += crow[vd] * crow[vd]; }
                const float inv = 1.0f / sqrtf(ss / (float)VHD + EPS);
                for (int vd = 0; vd < VHD; vd++) {
                    const float zz = zv[vd];
                    crow[vd] = crow[vd] * inv * r.norm[(size_t)vd] * (zz / (1.0f + expf(-zz)));
                }
                add_ms(sec.out, t0);
            }
        }
    }
    return sec;
}


}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 4;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    if (M < 1 || M > 64 || reps < 1) {
        std::fprintf(stderr, "usage: %s [rows 1..64] [reps]\n", argv[0]);
        return 2;
    }
    const Rows r = make_rows(M);
    const size_t core_n = (size_t)NL * M * NVH * VHD;
    std::printf("gdn_row_bench: %d GDN layers x %d v-heads, state %dx%d, batch M=%d, reps %d, %d threads (omp max %d)\n",
                NL, NVH, KHD, VHD, M, reps, heads_threads(), omp_get_max_threads());
    std::printf("  per-row cost is what the forward pays for one row of a batched GDN layer set\n");

    std::vector<float> ref(core_n), got(core_n);
    using Fn = std::function<void(const Rows&, int, std::vector<float>&)>;
    const std::vector<std::pair<const char*, Fn>> variants = {
        {"A today (row outer, per-head alloc)", variant_a},
        {"B A + hoisted scratch", variant_b},
        {"C B + loops exchanged (head outer)", [](const Rows& r, int nl, std::vector<float>& o) { variant_cd(r, nl, o, false); }},
        {"D C + gates hoisted", [](const Rows& r, int nl, std::vector<float>& o) { variant_cd(r, nl, o, true); }},
    };

    // Exactness first: a variant that is faster by computing something else must not be quotable.
    variant_a(r, NL, ref);
    for (size_t i = 1; i < variants.size(); i++) {
        std::fill(got.begin(), got.end(), -12345.0f);
        variants[i].second(r, NL, got);
        if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) != 0) {
            size_t k = 0;
            while (k < got.size() &&
                   std::memcmp(&got[k], &ref[k], sizeof(float)) == 0) k++;
            std::printf("REFUSED  %s is not bit-identical to A -- no timings reported\n", variants[i].first);
            if (k < got.size())
                std::printf("  first mismatch at %zu (row %zu, head %zu): A %.9g vs %.9g\n", k,
                            k / (size_t)(NVH * VHD), (k % (size_t)(NVH * VHD)) / VHD, (double)ref[k],
                            (double)got[k]);
            return 1;
        }
    }

    // Warm every variant once (page faults, first-touch of the state copy) before the minimum-of-reps.
    for (const auto& v : variants) { std::fill(got.begin(), got.end(), 0.0f); v.second(r, NL, got); }

    std::vector<double> ms(variants.size(), 0.0);
    for (int k = 0; k < reps; k++) {
        for (size_t i = 0; i < variants.size(); i++) {
            std::fill(got.begin(), got.end(), 0.0f);
            const auto t0 = std::chrono::steady_clock::now();
            variants[i].second(r, NL, got);
            const double d =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms[i] == 0.0 || d < ms[i]) ms[i] = d;   // best of reps: contention only ever adds
        }
    }

    std::printf("\n  %-38s %9s %9s %8s\n", "variant", "ms/batch", "ms/row", "vs A");
    for (size_t i = 0; i < variants.size(); i++)
        std::printf("  %-38s %9.1f %9.2f %7.2fx\n", variants[i].first, ms[i], ms[i] / M, ms[0] / ms[i]);
    const Sections sp = section_profile(r, NL);
    const double stot = sp.decay + sp.kvm + sp.upd + sp.out;
    std::printf("\n  1-thread section split of the recurrence (whole layer set, batch M=%d):\n", M);
    auto puts = [&](const char* n, double v) {
        std::printf("    %-26s %8.1f ms  %5.1f%%  (%6.3f ms/row)\n", n, v, 100.0 * v / (stot > 0 ? stot : 1),
                    v / M);
    };
    puts("state decay (KHD*VHD mul)", sp.decay);
    puts("kvm = S.k + delta", sp.kvm);
    puts("state update S += k (x) d", sp.upd);
    puts("out = S.q, norm, gate", sp.out);
    puts("TOTAL", stot);
    std::printf("\n  ms/row is the whole GDN layer set (%d layers) for one row: the forward's `core` term is this\n",
                NL);
    return 0;
}
