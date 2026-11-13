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

// npu_gemm_shape_guard_test (engine#1783): a GEMM call wider than the context it would launch on is refused by
// name, before any buffer is touched -- tested through XdnaTarget::gemv() itself, so removing the guard from
// either call site (npu_gemm, npu_gemm_rows) fails here. tests/launch_fit_test.cpp covers the rule
// (launch_fit.h) on its own; this covers that the target applies it.
//
// The defect (#2349 fixed it): XdnaTarget::gemv() never compared the call's M/K/N with the context's MD/KD/ND. A standalone harness
// that passed design index 0 for three designs (O K768 N1024, GU K768 N6144, D K3072 N1024) launched GU on O's
// context: the weight pack wrote 768 x 6144 into a 768 x 1024 weight buffer, and the process died with SIGSEGV and
// nothing on the device side to say why.
//
// This needs no device. The context is a bare I8Ctx with only its geometry set: its buffers were never allocated,
// so any path that reaches a pack, a quantise or a readback dereferences null and crashes the test: the refusal
// has to come first. That a call which fits still gets past it is the device rows' job (a fitting call needs real
// buffers): engine#2349's thread carries them (Bolt-E4B and the served 27B, byte-equal, no refusals).

#include "target/xdna/xdna_target.h"
#include "npu_engine_i8ctx_inc.h"   // I8Ctx

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using npu::target::GemvArgs;
using npu::target::xdna::XdnaTarget;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

// A context with a geometry and no buffers: the guard reads MD/KD/ND only.
static void bare_ctx(XdnaTarget& t, int role, int M, int K, int N) {
    auto c = std::make_unique<I8Ctx>();
    c->MD = M; c->KD = K; c->ND = N; c->NL = 1;
    t.ctx_ref(role) = std::move(c);
}

static bool refused_by_shape(XdnaTarget& t, GemvArgs a, std::vector<float>& out, std::string& err) {
    err.clear();
    const bool ok = t.gemv(a, out, err);
    return !ok && err.find("launch does not fit the role's context") != std::string::npos;
}

int main() {
    XdnaTarget t(nullptr);
    bare_ctx(t, 0, 128, 768, 1024);   // design index 0 is O: K768 N1024, 128 rows

    std::vector<float> x((size_t)1536 * 3072, 1.0f), w((size_t)3072 * 6144, 0.01f), out;
    std::string err;
    GemvArgs a;
    a.role = 0;
    a.layer = 0;
    a.x = x.data();
    a.Bmat = w.data();
    a.wkey = -1;
    a.tag = "probe";

    // The harness's crash, both bodies: GU's N and D's K on O's context.
    a.M = 1; a.K = 768; a.N = 6144; a.a_k = 768; a.rows = false;
    check(refused_by_shape(t, a, out, err), "one row, N 6144 on an N 1024 context: refused by shape");
    check(out.size() == 6144, "one row: the output is still sized to the call (N floats)");
    a.M = 128; a.rows = true;
    check(refused_by_shape(t, a, out, err), "rows, N 6144 on an N 1024 context: refused by shape");
    check(out.size() == (size_t)128 * 6144, "rows: the output is still sized to the call (M x N floats)");
    a.M = 128; a.K = 3072; a.N = 1024; a.a_k = 3072;
    check(refused_by_shape(t, a, out, err), "rows, K 3072 on a K 768 context: refused by shape");
    a.M = 1; a.rows = false;
    check(refused_by_shape(t, a, out, err), "one row, K 3072 on a K 768 context: refused by shape");
    // More rows than the context holds: the activation and result buffers are MD rows deep.
    a.M = 1536; a.K = 768; a.N = 1024; a.a_k = 768; a.rows = true;
    check(refused_by_shape(t, a, out, err), "rows, M 1536 on a 128-row context: refused by shape");
    check(err.find("M=1536 > 128") != std::string::npos && err.find("context is M=128 K=768 N=1024") != std::string::npos,
          "the message names the dimension that is over and the context's shape");
    // An empty launch (M=0) goes through the same check and is refused as empty before the buffers.
    a.M = 0; a.K = 768; a.N = 1024; a.rows = true;
    err.clear();
    check(!t.gemv(a, out, err) && err.find("empty dimension") != std::string::npos, "an empty launch: refused as empty");

    std::printf("npu_gemm_shape_guard: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
