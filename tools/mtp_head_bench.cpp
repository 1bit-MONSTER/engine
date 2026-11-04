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

// mtp_head_bench.cpp -- what one MTP head step costs on the HOST, and the hash of what it produces.
//
// The draft route's per-iteration cost has three terms: two base passes (the array), the drafter's logits calls
// (the array), and the head's own arithmetic (this host, npu/mtp/mtp_head.cpp).  Only the third one can be
// measured without the device, and until it was measured the split was arithmetic.  This tool exists so a
// change to the head's maths can be stated as a before/after on the same box, with the output hash proving the
// change did not move a single float:
//
//   mtp_head_bench --head-exact <pb16i-exact dir> --head-q4k <q4_k dir> [--steps 8] [--matvec 50]
//
// Prints per-matrix matvec milliseconds (the head is matvec-bound: 424 M MACs of the 424 M a step does), then
// per-step milliseconds with no lm head (as in the live loop's commit_rows / prime calls), and finally one FNV-1a
// hash over every output hidden and its bit pattern.  Same --steps/-matvec on both builds, same hash = the change
// is bit-exact; different hash = it is not, and any acceptance number taken with it is a new measurement.
#include "mtp/mtp_head.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace onebit::npu::mtp;

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// FNV-1a over the float's bytes: two runs that differ in one ULP hash differently.
uint64_t fnv(uint64_t h, const float* x, int n) {
    const uint8_t* p = (const uint8_t*)x;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Deterministic stand-in for hidden/embedding: same numbers on every build, so a timing comparison is the same
// work and a hash difference is a maths difference.
void fill(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = ((float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string exact, q4k;
    int steps = 8, matvec_rep = 50;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--head-exact") exact = val();
        else if (a == "--head-q4k") q4k = val();
        else if (a == "--steps") steps = std::atoi(val().c_str());
        else if (a == "--matvec") matvec_rep = std::atoi(val().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (exact.empty() || q4k.empty()) {
        std::fprintf(stderr, "usage: mtp_head_bench --head-exact <dir> --head-q4k <dir> [--steps N] [--matvec N]\n");
        return 2;
    }
    Weights w;
    std::string err;
    const double t_load0 = now_ms();
    if (!load_weights(exact, q4k, w, err)) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    std::printf("head loaded in %.1f ms\n", now_ms() - t_load0);

    // Per-matrix cost: one matvec each, on a vector of the right length.  The head's step is these eight calls
    // plus the attention/norm work, so this is where the 424 M MACs of a step live.
    struct Ent { const char* name; const Matrix* m; };
    const Ent ents[] = {{"fc", &w.fc},     {"q", &w.q},       {"k", &w.k},   {"v", &w.v},
                        {"o", &w.o},       {"gate", &w.gate}, {"up", &w.up}, {"down", &w.down}};
    double mv_tot = 0;
    std::printf("\nmatvec (N = %d reps, median of the reps):\n", matvec_rep);
    for (const Ent& e : ents) {
        std::vector<float> a((size_t)e.m->K), y((size_t)e.m->N);
        fill(a, 0x9e37u + (uint32_t)e.m->K);
        std::vector<double> ms;
        for (int r = 0; r < matvec_rep; r++) {
            const double t0 = now_ms();
            e.m->matvec(a.data(), y.data());
            ms.push_back(now_ms() - t0);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        mv_tot += med;
        std::printf("  %-4s K %5d x N %5d  %8.2f ms  (%6.1f M MAC, %5.1f GF/s)\n", e.name, e.m->K, e.m->N, med,
                    (double)e.m->K * e.m->N / 1e6, 2.0 * (double)e.m->K * e.m->N / (med * 1e6));
    }
    std::printf("  sum of the eight matvecs: %.2f ms\n", mv_tot);

    std::vector<float> hidden((size_t)kH), emb((size_t)kH), out, lg;
    fill(hidden, 0x1234u);
    fill(emb, 0x5678u);
    KvCache cache;
    uint64_t h = 1469598103934665603ULL;
    std::vector<double> step_ms;
    std::printf("\nhead_step, no lm head (the live loop's commit/prime path):\n");
    for (int s = 0; s < steps; s++) {
        const double t0 = now_ms();
        if (!head_step(w, hidden.data(), emb.data(), s, cache, out, nullptr, lg)) {
            std::fprintf(stderr, "head_step failed at %d\n", s);
            return 1;
        }
        const double ms = now_ms() - t0;
        step_ms.push_back(ms);
        h = fnv(h, out.data(), kH);
        std::printf("  step %d (pos %d, cache %d rows): %7.2f ms  out[0] %+.9e\n", s, s, cache.t, ms, out[0]);
        // next step's input is this step's output, as the drafter chains it
        hidden = out;
    }
    std::sort(step_ms.begin(), step_ms.end());
    std::printf("\nmedian %.2f ms/step, min %.2f, max %.2f, %d steps\n", step_ms[step_ms.size() / 2], step_ms.front(),
                step_ms.back(), (int)step_ms.size());
    std::printf("output hash (FNV-1a over %d x %d floats): %016llx\n", steps, kH, (unsigned long long)h);
    return 0;
}
