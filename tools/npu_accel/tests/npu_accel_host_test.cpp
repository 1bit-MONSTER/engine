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
// npu_accel_host_test: the ACCEL backend's plumbing, with no device and no model. A ggml scheduler with the backend
// (ONEBIT_NPU_ACCEL_HOST=1: the same int8 codes and int32 sums on the host) in front of the CPU backend computes
//   a weight MUL_MAT        y  = W x          (Q8_0 weight, 3 activation rows)
//   a weight MUL_MAT_ID     ye = E[ids] x     (Q8_0 expert stack, 8 of 16 experts per token, 2 tokens, x broadcast)
// and the test checks (1) the scheduler gave both ops to the NPU backend, (2) the results agree with the CPU-only
// graph within W8A8 error (relative L2 < 2 %), (3) the expert ids it was handed are the ones the graph routed.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml_npu_accel.h"

namespace {
struct Case {
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor *W = nullptr, *E = nullptr;
};

void fill_q8(ggml_tensor* t, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> f(ggml_nelements(t));
    for (auto& v : f) v = nd(rng);
    std::vector<char> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nelements(t) / t->ne[0], t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// y (3 rows of W x) and the expert outputs, computed on the given backends.
bool run(Case& c, std::vector<ggml_backend_t> be, std::vector<float>& y, std::vector<float>& ye, int* npu_nodes) {
    const int K = 512, N = 1024, E = 16, NU = 8, T = 2;
    ggml_init_params ip = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 3);
    ggml_tensor* xe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NU, T);
    ggml_set_input(x); ggml_set_input(xe); ggml_set_input(ids);
    ggml_tensor* out = ggml_mul_mat(ctx, c.W, x);
    ggml_tensor* oute = ggml_mul_mat_id(ctx, c.E, xe, ids);
    ggml_set_output(out); ggml_set_output(oute);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, oute);
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto b : be) bufts.push_back(ggml_backend_get_default_buffer_type(b));
    ggml_backend_sched_t s = ggml_backend_sched_new(be.data(), bufts.data(), (int)be.size(), 256, false, true);
    if (!ggml_backend_sched_alloc_graph(s, gf)) return false;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(K * 3), xev(K * T);
    for (auto& v : xv) v = nd(rng);
    for (auto& v : xev) v = nd(rng);
    std::vector<int32_t> idv(NU * T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < NU; i++) idv[t * NU + i] = (i * 2 + t + 1) % E;   // 8 distinct experts per token
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
    ggml_backend_tensor_set(xe, xev.data(), 0, xev.size() * 4);
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * 4);
    if (ggml_backend_sched_graph_compute(s, gf) != GGML_STATUS_SUCCESS) return false;
    *npu_nodes = 0;
    if (be.size() > 1)
        for (ggml_tensor* n : {out, oute}) *npu_nodes += ggml_backend_sched_get_tensor_backend(s, n) == be[0];
    y.resize(ggml_nelements(out)); ye.resize(ggml_nelements(oute));
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * 4);
    ggml_backend_tensor_get(oute, ye.data(), 0, ye.size() * 4);
    ggml_backend_sched_free(s);
    ggml_free(ctx);
    return true;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) { num += (a[i] - b[i]) * (double)(a[i] - b[i]); den += (double)b[i] * b[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}
}  // namespace

int main() {
    setenv("ONEBIT_NPU_ACCEL_HOST", "1", 1);
    unsetenv("ONEBIT_NPU_ACCEL_KERNELS");
    if (!ggml_npu_accel_register()) { std::fprintf(stderr, "FAIL: backend did not register\n"); return 1; }
    ggml_backend_dev_t npu = ggml_backend_dev_by_name("NPU");
    if (!npu) { std::fprintf(stderr, "FAIL: no NPU device after registration\n"); return 1; }
    ggml_backend_t be_npu = ggml_backend_dev_init(npu, nullptr), be_cpu = ggml_backend_cpu_init();

    Case c;
    ggml_init_params ip = {ggml_tensor_overhead() * 4, nullptr, true};
    c.wctx = ggml_init(ip);
    c.W = ggml_new_tensor_2d(c.wctx, GGML_TYPE_Q8_0, 512, 1024);
    c.E = ggml_new_tensor_3d(c.wctx, GGML_TYPE_Q8_0, 512, 1024, 16);
    ggml_set_name(c.W, "blk.0.attn_q.weight"); ggml_set_name(c.E, "blk.0.ffn_gate_up_exps.weight");
    c.wbuf = ggml_backend_alloc_ctx_tensors(c.wctx, be_cpu);
    ggml_backend_buffer_set_usage(c.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(1);
    fill_q8(c.W, rng); fill_q8(c.E, rng);

    std::vector<float> y_ref, ye_ref, y, ye;
    int n_npu = 0, dummy = 0;
    if (!run(c, {be_cpu}, y_ref, ye_ref, &dummy) || !run(c, {be_npu, be_cpu}, y, ye, &n_npu)) {
        std::fprintf(stderr, "FAIL: graph compute\n"); return 1;
    }
    const double e1 = rel_l2(y, y_ref), e2 = rel_l2(ye, ye_ref);
    const bool ok = n_npu == 2 && e1 < 0.02 && e2 < 0.02;
    std::printf("npu_accel_host_test: %d of 2 ops on the NPU backend, MUL_MAT rel L2 %.4f, MUL_MAT_ID rel L2 %.4f -> %s\n",
                n_npu, e1, e2, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
