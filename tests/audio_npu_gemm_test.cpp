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
// tests/audio_npu_gemm_test.cpp -- the encoder's products on an NpuTarget (npu/forward/include/audio_npu_gemm.h, engine#1783),
// against a FAKE target that does the GEMM exactly from the staged Bmat and caches it by (layer, wkey) like XdnaTarget.
//
// Pinned: the role index is the design's position (never a default 0), each launch is at most the design's M rows,
// the weight is zero-padded to the staged K x N and sent once, the result is de-padded to the model's N, and an
// uncovered shape is declined by name.
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "audio_npu_gemm.h"

using namespace onebit::audio;
namespace t = ::npu::target;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct FakeTarget : t::NpuTarget {
    std::vector<NpuAudioDesign> d;          // what open_prefill adopted, by role index
    std::map<int, std::vector<float>> cache;   // wkey -> staged K x N weight
    std::vector<t::GemvArgs> seen;
    int with_bmat = 0;
    const char* name() const override { return "fake"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override { return true; }
    bool load_block(int, int, std::string&) override { return true; }
    t::Caps caps() const override { return t::Caps{}; }
    bool gemv(const t::GemvArgs& a, std::vector<float>& out, std::string& err) override {
        seen.push_back(a);
        if (a.role < 0 || a.role >= int(d.size())) { err = "no context for role"; return false; }
        const NpuAudioDesign& g = d[size_t(a.role)];
        if (a.M > g.M || a.K != g.K || a.N != g.N) { err = "launch does not fit the role's context"; return false; }
        const std::vector<float>* w = nullptr;
        if (a.Bmat != nullptr) {
            ++with_bmat;
            std::vector<float> b(a.Bmat, a.Bmat + size_t(a.K) * size_t(a.N));
            if (a.wkey >= 0) w = &(cache[a.wkey] = std::move(b));
            else w = &(cache[-1000 - int(seen.size())] = std::move(b));
        } else {
            const auto it = cache.find(a.wkey);
            if (it == cache.end()) { err = "no Bmat and nothing cached"; return false; }
            w = &it->second;
        }
        out.assign(size_t(a.M) * size_t(a.N), 0.0f);
        for (int m = 0; m < a.M; ++m)
            for (int k = 0; k < a.K; ++k) {
                const float x = k < a.a_k ? a.x[size_t(m) * size_t(a.a_k) + size_t(k)] : 0.0f;
                for (int n = 0; n < a.N; ++n) out[size_t(m) * size_t(a.N) + size_t(n)] += x * (*w)[size_t(k) * size_t(a.N) + size_t(n)];
            }
        return true;
    }
};

static void host_gemm(const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N, std::vector<float>* C) {
    C->assign(size_t(M) * size_t(N), 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n) (*C)[size_t(m) * N + n] += A[size_t(m) * K + k] * B[size_t(k) * N + n];
}

static double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

int main() {
    FakeTarget ft;
    // Small stand-ins on the real grid: role 0 = "O" K256 N512, role 1 = "conv" K256 N1024, role 2 = "D" K512 N512.
    ft.d = {{8, 256, 512}, {8, 256, 1024}, {8, 512, 512}};
    NpuAudioGemm g(ft, ft.d);

    auto fill = [](int n, float s) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) v[size_t(i)] = std::sin(0.37f * float(i + 1) * s) * 0.5f;
        return v;
    };

    // 1. a model-width projection (200 x 300 -> staged K256 N512) on role 0, 19 rows -> 3 launches (8, 8, 3)
    {
        const int M = 19, K = 200, N = 300;
        const auto A = fill(M * K, 1.0f), B = fill(K * N, 0.7f);
        std::vector<float> C, R;
        std::string why;
        check(g.gemm("w.o", A.data(), M, K, B.data(), N, &C, &why), "a padded projection runs: " + why);
        host_gemm(A, B, M, K, N, &R);
        check(maxdiff(C, R) < 1e-4, "it equals the host GEMM after de-padding (max " + std::to_string(maxdiff(C, R)) + ")");
        check(ft.seen.size() == 3 && ft.seen[0].M == 8 && ft.seen[1].M == 8 && ft.seen[2].M == 3,
              "19 rows at M=8 are three launches of 8, 8 and 3");
        check(ft.seen[0].role == 0 && ft.seen[0].K == 256 && ft.seen[0].N == 512 && ft.seen[0].a_k == 200,
              "role 0 is asked for the staged K256 N512, with the activation's real width 200");
        check(ft.seen[0].rows && ft.seen[2].rows, "every chunk is the batched body");
        check(ft.with_bmat == 1, "the padded weight is sent once and the next launches use the cache");
    }

    // 2. a second shape routes by its staged (K, N), not by order of first use or a default
    {
        ft.seen.clear();
        const int M = 5, K = 240, N = 768;   // conv1 in miniature: K240 -> K256, N768 -> N1024
        const auto A = fill(M * K, 1.3f), B = fill(K * N, 0.4f);
        std::vector<float> C, R;
        std::string why;
        check(g.gemm("conv1#conv", A.data(), M, K, B.data(), N, &C, &why), "a conv-stem shape runs: " + why);
        host_gemm(A, B, M, K, N, &R);
        check(ft.seen.size() == 1 && ft.seen[0].role == 1 && ft.seen[0].N == 1024, "K240 N768 goes to role 1 (K256 N1024)");
        check(maxdiff(C, R) < 1e-4, "and equals the host GEMM");
        check(ft.seen[0].wkey != 0, "a second weight gets its own residency key");
    }

    // 3. the same key a second time sends no weight
    {
        ft.seen.clear();
        const int before = ft.with_bmat;
        const int M = 4, K = 200, N = 300;
        const auto A = fill(M * K, 2.1f), B = fill(K * N, 0.7f);
        std::vector<float> C, R;
        std::string why;
        check(g.gemm("w.o", A.data(), M, K, B.data(), N, &C, &why), "a cached weight runs again");
        host_gemm(A, B, M, K, N, &R);
        check(ft.with_bmat == before && ft.seen[0].Bmat == nullptr, "the second call sends no Bmat (the target holds it)");
        check(maxdiff(C, R) < 1e-4, "and still equals the host GEMM");
    }

    // 4. a shape no design covers is declined by name, and the target is not called
    {
        ft.seen.clear();
        std::vector<float> C;
        std::string why;
        // K1000 stages to K1024: no design
        const auto A2 = fill(1000, 1.0f), B2 = fill(1000 * 10, 1.0f);
        check(!g.gemm("odd", A2.data(), 1, 1000, B2.data(), 10, &C, &why) &&
                  why.find("no design for staged K1024 N512") != std::string::npos,
              "an uncovered staged shape is declined by name: " + why);
        check(ft.seen.empty(), "and the target is not called");
    }

    // 5. a target failure is passed through with the role and key
    {
        ft.d[2].M = 0;   // the fake now refuses role 2
        NpuAudioGemm g2(ft, {{8, 256, 512}, {8, 256, 1024}, {8, 512, 512}});
        const auto A = fill(2 * 512, 1.0f), B = fill(512 * 512, 1.0f);
        std::vector<float> C;
        std::string why;
        check(!g2.gemm("w.d", A.data(), 2, 512, B.data(), 512, &C, &why) && why.find("role 2 (w.d)") != std::string::npos,
              "a refused launch names its role and weight: " + why);
    }

    check(g.launches() == 5 && g.products() == 3, "counts: 5 launches, 3 products");
    if (failures) {
        std::printf("audio_npu_gemm: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("audio_npu_gemm: all pass\n");
    return 0;
}
