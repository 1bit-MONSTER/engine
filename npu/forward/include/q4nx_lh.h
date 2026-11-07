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
// q4nx_lh.h — the NPU route's phase accounting (NPU_PHASE_PROF=1), shared by the model loop and the
// launch layer.
//
// Named RAII timers, one accumulator per slot.  Header-resident (inline variables) so that both the
// forward (which owns the request lifecycle and the layer loop) and a target's launch layer
// (npu/target/xdna, which times pack / quantise+launch / wait / readback) accumulate into the same
// array: a request can print the delta of this array and its rows sum to its wall time.
//
// Slot order matters: the first LH_LAYER_N entries are the layer-loop rows the per-token [lh] table
// ranks, the rest are request-level rows.
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace q4nx_lh {
enum LhSlot {
    LH_LOAD = 0,        // load_layer(): per-token f32 re-dequant
    LH_LOAD_PROJ,       //   of which: GDN qkv/z/out projections
    LH_LOAD_FFN,        //   of which: dense FFN tail (gate/up/down)
    LH_LOAD_GDN_SMALL,  //   of which: alpha/beta, conv1d, norm, ssm_a/dt
    LH_ASM,             // q4nx_assemble_B()
    LH_GEMM,            // npu_gemm() total outside gdn_layer, incl. amax scan + out deq
    LH_GEMM_GDN,        //   of which: npu_gemm() time spent inside gdn_layer()
    LH_ATTN,            // attention(): rope, softmax, KV append
    LH_GDN,             // gdn_layer() total
    LH_GDN_AB,          //   of which: host alpha/beta dot products
    LH_GDN_CORE,        //   of which: delta-rule recurrence + gated rmsnorm
    LH_RMS,             // rmsnorm()
    LH_RES,             // residual adds / layer_scale
    LH_MLP,             // mlp(): SiLU/GELU * up
    LH_LAYER_N,         // ---- end of the layer-loop slots ----
    // Request-level rows (forward_serve.cpp handle_generate).
    LH_RQ_ACCEPT = LH_LAYER_N,  // body parse + validate + options/grammar
    LH_RQ_TOKENIZE,     // prompt -> token ids
    LH_RQ_LOCK,         // wait for the engine mutex
    LH_RQ_RESET,        //   of which (inside prefill): KV/conv/recurrent reset
    LH_RQ_PREFILL,      // prefill chunk loop (contains reset + weight packs)
    LH_RQ_DECODE,       // decode step() total
    LH_RQ_SAMPLE,       //   of which: grammar mask + sample_token
    LH_RQ_EMIT,         //   of which: detokenise the emitted token
    LH_RQ_RESP,         // response serialise + write
    LH_PACK_UPLOAD,     // weight BO create + pack + upload (wcache_ miss only)
    LH_HEAD_ROWS,       // run_lm_head_rows(): the head for a batched verify pass
    LH_N
};
inline bool lh_on() {
    static const bool on = std::getenv("NPU_PHASE_PROF") != nullptr;
    return on;
}
inline double g_lh[LH_N];
inline long   g_lh_n[LH_N];
inline int    g_gdn_d = 0;   // >0 while inside a gdn_layer() scope
inline bool lh_in_gdn() { return g_gdn_d > 0; }
struct LhScope {
    int  s; bool on;
    std::chrono::steady_clock::time_point t;
    explicit LhScope(int s_)
        : s(s_), on(lh_on()),
          t(on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {
        if (on && s == LH_GDN) g_gdn_d++;
    }
    void stop() {
        if (!on) return;
        g_lh[s] += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t).count();
        g_lh_n[s]++;
        if (s == LH_GDN) g_gdn_d--;
        on = false;
    }
    ~LhScope() { stop(); }
};
// Table names, indexed by slot (shared by the per-token and per-request tables).
inline const char* lh_name(int s) {
    static const char* const kn[LH_N] = {
        "load_layer", "  of-which: GDN proj deq", "  of-which: FFN tail deq",
        "  of-which: GDN small deq", "assemble_B", "npu_gemm (total)",
        "  of-which: gemm in gdn", "attention", "gdn_layer",
        "  of-which: gdn alpha/beta", "  of-which: gdn recurrence", "rmsnorm",
        "residual adds", "mlp (act*up)",
        "request accept/validate", "tokenisation", "engine lock wait",
        "  of-which: KV/rec state reset", "prefill (chunks)",
        "decode (step loop)", "  of-which: sample+grammar",
        "  of-which: detokenise token", "response write",
        "  of-which: weight pack+upload (wcache miss)",
        "lm head (batch rows)"};
    return (s >= 0 && s < LH_N) ? kn[s] : "?";
}
}  // namespace q4nx_lh

// ---------------------------------------------------------------------------
// Phase attribution (NPU_PHASE_PROF=1).  One line per decode step whose named phases sum to that step's
// wall time.  Additive measurement only: nothing in the compute path depends on it.
// ---------------------------------------------------------------------------
namespace q4nx_lh {

struct StepPhases {
    double embed = 0, layer = 0, ffnorm = 0, head = 0, tail = 0;
    double g_pack = 0, g_ql = 0, g_wait = 0, g_rb = 0;   // layer GEMMs only
    double h_total = 0, h_memcpy = 0, h_pack = 0, h_ql = 0, h_wait = 0, h_rb = 0;
    long   launches = 0;
};
inline StepPhases g_ph;
inline bool       g_in_head = false;
inline bool phprof_on() {
    static const bool on = std::getenv("NPU_PHASE_PROF") != nullptr;
    return on;
}
inline double ph_ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t).count();
}


// Restored from main by the #1056 port: the batched-body flag, its scope guard and the ranked host-call-site
// dump are part of this accounting surface, so they live beside g_ph/g_lh rather than in the model .cpp (where
// main had them while the launch layer was still there). Main-only prose kept where it explains the numbers.
inline bool g_in_batch = false;
struct BatchPhaseScope {   // g_in_batch for the batched body, cleared on every return path
    bool prev;
    explicit BatchPhaseScope(bool on) : prev(g_in_batch) { if (on) g_in_batch = true; }
    ~BatchPhaseScope() { g_in_batch = prev; }
};
// Ranked layer-loop host call sites: what the `host=` residual is made of.
// Nested rows (of-which) are inside their parent; the TOP-LEVEL SUM is the
// one the residual is compared against.
// host part of npu_gemm = whole call minus the four device phases it times;
// host part of gdn_layer = the scope minus the gemm launches it contains.
// Shared by step() (unit "token") and step_batch() (unit "batch") so both residuals are read alike.
inline void lh_dump_layer_host(const char* what, const char* unit) {
    const double gs = g_ph.g_pack + g_ph.g_ql + g_ph.g_wait + g_ph.g_rb;
    const double gemm_all  = g_lh[LH_GEMM] + g_lh[LH_GEMM_GDN];
    const double gemm_host = gemm_all - gs;
    const double gdn_host  = g_lh[LH_GDN] - g_lh[LH_GEMM_GDN];
    const double top = g_lh[LH_LOAD] + g_lh[LH_ASM] + g_lh[LH_ATTN] + g_lh[LH_RMS] +
                       g_lh[LH_RES] + g_lh[LH_MLP] + gdn_host + gemm_host;
    fprintf(stderr, "[lh] %s layer-loop host call sites, ms/%s (ranked):\n", what, unit);
    bool done[LH_LAYER_N] = {false};
    for (int r = 0; r < LH_LAYER_N; r++) {
        int best = -1;
        for (int i = 0; i < LH_LAYER_N; i++) {
            if (done[i] || g_lh[i] <= 0.0) continue;
            if (best < 0 || g_lh[i] > g_lh[best]) best = i;
        }
        if (best < 0) break;
        done[best] = true;
        const double n = g_lh_n[best] ? (double)g_lh_n[best] : 1.0;
        fprintf(stderr, "[lh]   %-26s %8.2f ms  n=%4ld  %7.3f ms/call  %5.1f%%\n",
                lh_name(best), g_lh[best], g_lh_n[best], g_lh[best] / n,
                100.0 * g_lh[best] / (g_ph.layer > 0 ? g_ph.layer : 1.0));
    }
    // Deliberately not static: these are read from g_lh at call time.  A `static` array is initialised
    // once, so the table froze on its first call and every later line reprinted that first pass's numbers
    // (seen in ticket-batchprof-p0-before-20261104-142823.log: load_layer 8587.41 ms on every token while
    // the live ACCOUNTED row moved 654 -> 690 ms).
    const struct TlpRow { const char* n; double v; } tlp[] = {
        {"load_layer",                g_lh[LH_LOAD]},
        {"assemble_B",                g_lh[LH_ASM]},
        {"gemm host (amax+deq)",      gemm_host},
        {"gdn_layer (non-gemm)",      gdn_host},
        {"mlp (act*up)",              g_lh[LH_MLP]},
        {"rmsnorm",                   g_lh[LH_RMS]},
        {"attention",                 g_lh[LH_ATTN]},
        {"residual adds",             g_lh[LH_RES]},
    };
    fprintf(stderr, "[lh] TOP-LEVEL PARTITION (no double counting; npu_gemm %.2f "
                    "- phases %.2f = %.2f host):\n", gemm_all, gs, gemm_host);
    for (const TlpRow& r : tlp)
        fprintf(stderr, "[lh]   %-26s %8.2f ms  %5.1f%% of gemm-free host\n", r.n, r.v,
                100.0 * r.v / (g_ph.layer - gs > 0 ? g_ph.layer - gs : 1.0));
    fprintf(stderr, "[lh]   %-26s %8.2f ms\n", "ACCOUNTED", top);
    fprintf(stderr, "[lh]   %-26s %8.2f ms   (layers=%.2f gemm_phases=%.2f)\n",
            "UNATTRIBUTED", g_ph.layer - gs - top, g_ph.layer, gs);
}

}  // namespace q4nx_lh
