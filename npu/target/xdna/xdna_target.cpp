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
// npu/target/xdna/xdna_target.cpp — the XDNA2 launch layer (engine#799 P4 / #803).
//
// Moved out of npu/forward/src/q4nx_forward.cpp without a numerics change: the same quantisation, the same
// packed-weight BO cache, the same pb16i image path and Hadamard fold, the same kernel selection.  What
// changed is only who owns the state: the model keeps the maths and hands activations in here.
//
// docs/npu-p4-target-split.md records what moved in A1 and what A2 (plan-driven setup) still owes.
#include "target/xdna/xdna_target.h"
#include "batch_width.h"
#include "file_create.h"   // dumps are created 0600, never through a symlink

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>   // mkstemp/close: the pb16i activation probe's dump

#include "hadamard.h"                // the activation rotation: one implementation, shared with the model
#include "npu_engine_i8ctx_inc.h"   // I8Ctx
#include "q4nx_lh.h"                // q4nx_lh::LhScope, the phase accounting the model also writes
#include "q4nx_pack.h"

namespace npu::target::xdna {
using namespace q4nx_lh;   // the phase slots and g_ph are shared with the model (q4nx_lh.h)
namespace {

// The probe dump's own file: mkstemp() gives O_CREAT|O_EXCL and mode 0600 with a unique name in one step.
static FILE* open_probe_dump() {
    char path[] = "/tmp/a0-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return nullptr;
    FILE* f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
        return nullptr;
    }
    fprintf(stderr, "[pb16i-dump] %s\n", path);
    return f;
}

static const char* role_tag(Q4nxDesign r) {
    switch (r) {
        case Q4NX_DESIGN_QKV: return "QKV";
        case Q4NX_DESIGN_QKV2: return "QKV2";
        case Q4NX_DESIGN_O:   return "O";
        case Q4NX_DESIGN_O2:  return "O2";
        case Q4NX_DESIGN_GU:  return "GU";
        case Q4NX_DESIGN_G:   return "G";
        case Q4NX_DESIGN_U:   return "U";
        case Q4NX_DESIGN_PLEGATE: return "PLEGATE";
        case Q4NX_DESIGN_PLEPROJ: return "PLEPROJ";
        case Q4NX_DESIGN_GU2: return "GU2";
        case Q4NX_DESIGN_D2:  return "D2";
        case Q4NX_DESIGN_QKVLIN: return "QKV";
        case Q4NX_DESIGN_Z:   return "Z";
        case Q4NX_DESIGN_QKVFA_Q:  return "QKVFAQ";
        case Q4NX_DESIGN_QKVFA_KV: return "QKVFAKV";
        default:              return "D";
    }
}

}  // namespace

// The forward's own probe, kept identical (exists and non-empty): an ELF that is present but zero bytes
// is the failure this catches, and a target that decided differently from its caller would be a second
// definition of "the kernel is there".
static bool elf_present(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz >= 1;
}

XdnaTarget::XdnaTarget(xrt::device* dev) : dev_(dev) {}
XdnaTarget::~XdnaTarget() = default;

// The M=128 ELF set, moved out of Q4nxNpuForward::init_impl() unchanged (engine#803 A2b): the same
// probe, the same two refusals, the same per-role log line, and the same adopt-then-open order.  What
// changed is who owns it -- a missing or unloadable artifact is now refused by the target that would
// launch it, instead of by a caller that then hands it a context.
bool XdnaTarget::open_prefill(const Q4nxDesignGeom* designs, int n, std::string& err) {
    const std::string& dir = setup_.kernels_dir;
    const bool b8 = setup_.b8;
    const bool host_gemm = setup_.host_gemm;
    const std::string& plan_path = setup_.plan_path;
    if (!designs || n <= 0 || n > Q4NX_MAX_DESIGNS) {
        err = "open_prefill: no design set (" + std::to_string(n) + " designs)";
        return false;
    }
    for (int i = 0; i < n; i++) {
        const Q4nxDesignGeom* g = &designs[i];
        // The two full-attention-QKV-split designs exist only as pb16i kernels (the role has no int8
        // launch that fits), so they get no M=128 context and no int8 ELF on disk: their decode context
        // is built from the _pb16i ELF by the caller's twin pass.
        adopt_geometry(i, *g);
        if (role_is_pb16i_split(i)) continue;
        const std::string elf = dir + "/" + g->elf_name;
        if (host_gemm) {
            // Offline mode: host GEMMs are plain float matmuls over the decoded weights, so no ELF is
            // opened and none is required.  A model whose kernel set is not built yet can still be
            // loaded and prompted, which is what makes it a numeric reference for a new geometry
            // (engine#796).
            if (elf_present(elf))
                std::fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  %s (host-GEMM)\n",
                             q4nx_design_log_tag(g->role), g->M, g->K, g->N, elf.c_str());
            else
                std::fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  (host-GEMM, no ELF: %s)\n",
                             q4nx_design_log_tag(g->role), g->M, g->K, g->N, g->elf_name);
            continue;
        }
        if (!elf_present(elf)) {
            // The same wording the forward used for this case, because a diagnostic that changes shape
            // when code moves is a diagnostic someone has to re-learn.
            const char* role = q4nx_design_log_tag(g->role);
            // engine#1503: when the asked-for name is the plain one and the directory holds only the _b8 family, the missing
            // thing is the container's LAYOUT, not the artifact. Measured on a Ryzen AI 1.9 pack: the same container ran
            // correctly once it was re-imported with `--weight-layout b8`. Say that here, or the reader goes looking for a
            // kernel that was never supposed to exist.
            std::string hint;
            {
                const std::string nm(g->elf_name);
                if (nm.size() > 4 && nm.compare(nm.size() - 4, 4, ".elf") == 0) {
                    const std::string b8 = nm.substr(0, nm.size() - 4) + "_b8.elf";
                    if (elf_present(dir + "/" + b8))
                        hint = " - but the b8 variant \"" + b8 + "\" is there: this container's weight layout is "
                               "row-major (q4nx_weight_layout absent), and a row-major container asks for row-major "
                               "kernels. Re-import the pack with `npu-pack import --weight-layout b8`, or supply a "
                               "row-major ELF family (engine#1503).";
                }
            }
            err = (plan_path.empty() ? ("missing full ELF " + elf)
                                     : ("plan " + plan_path + ": this model's plan needs the " + role +
                                        " kernel \"" + g->elf_name + "\" in " + dir + " (missing)")) + hint;
            return false;
        }
        if (!dev_) {
            err = "open_prefill: no device for the ELF set";
            return false;
        }
        I8Ctx* ctx = make_ctx(i);
        if (!ctx) {
            err = "cannot create a context for role " + std::to_string(i);
            return false;
        }
        ctx->MD = g->M; ctx->KD = g->K; ctx->ND = g->N;
        ctx->blocked_b = b8;
        if (!ctx->init_elf(*dev_, elf.c_str(), g->M, g->K, g->N, 1)) {
            err = "I8Ctx init_elf failed for " + elf;
            return false;
        }
        std::fprintf(stderr, "  [forward] %-3s M=%d K=%d N=%d  %s\n", q4nx_design_log_tag(g->role),
                     g->M, g->K, g->N, elf.c_str());
    }
    return true;
}

// --------------------------------------------------------------------------
// NpuTarget
// --------------------------------------------------------------------------
bool XdnaTarget::open(const onebit::npu::plan::Plan& plan, const std::string& kernels_dir,
                      const char* family, std::string& err) {
    // A1 consumes the plan where it overlaps with what this target was handed: the geometry it is about
    // to launch on must be the geometry the plan describes, or the run is a different model than the one
    // the directory says it is.  A model directory with no plan keeps the legacy probing in the forward,
    // and this stays a "device attached?" check (that is what makes the plan additive).
    //
    // A2 (#803 §3) moves the derivation in here: then this becomes the source rather than the check.
    kernels_dir_ = kernels_dir;
    plan_ = plan;
    // Before the device checks: a plan that asks for a cache this target cannot store is refused whatever
    // the device is doing, and the refusal names the field.  Without this an int8 KV plan loads and is
    // quietly served an f16 cache -- the plan layer's own documented trap.
    if (!::npu::target::check_kv_caps(plan_.kv_key, plan_.kv_value, caps(), err)) return false;
    if (!dev_) { err = "xdna: no device attached (host-GEMM mode has no target)"; return false; }
    if (ndesigns_ == 0) { err = "xdna: no design adopted yet (A2: the plan supplies it)"; return false; }
    if (!plan_.loaded()) return true;

    if (plan_.layers > 0 && n_layers_ > 0 && plan_.layers != n_layers_)
        { err = "plan: " + std::to_string(plan_.layers) + " layers but the container has " +
                std::to_string(n_layers_); return false; }
    // The decode family is what this target picks between per launch, so a plan that states an M this
    // target has no kernel for is a refusal rather than a silent M=128 (or M=16) launch.
    int adopted_prefill = 0, adopted_decode = 0;
    for (int i = 0; i < ndesigns_; i++) {
        if (ctx_[i] && ctx_[i]->MD > adopted_prefill) adopted_prefill = ctx_[i]->MD;
        if (ctx16_[i] && ctx16_[i]->MD > adopted_decode) adopted_decode = ctx16_[i]->MD;
    }
    if (plan_.prefill_m > 0 && adopted_prefill > 0 && plan_.prefill_m != adopted_prefill)
        { err = "plan: prefill M=" + std::to_string(plan_.prefill_m) + " but the adopted kernels run M=" +
                std::to_string(adopted_prefill); return false; }
    if (plan_.decode_enabled && plan_.decode_m > 0 && adopted_decode > 0 &&
        plan_.decode_m != adopted_decode)
        { err = "plan: decode M=" + std::to_string(plan_.decode_m) + " but the adopted twin runs M=" +
                std::to_string(adopted_decode); return false; }
    // No refusal for "plan says decode is enabled but no M=16 twin was adopted": that is the normal case
    // for an rm container whose kernel set carries no _m16 twins, and the forward's own probe already
    // decided it (`skipped++; continue;   // stay on M=128`). plan.decode is a static field of P1's
    // generator (tools/npu_plan.py always writes enabled/M=16), not a claim that the files exist, so a
    // third refusal here would turn "no twins on disk" into a failed load. For pb16i the same condition
    // is already fatal earlier, by name, in the forward (`decode role ... needs "..." (missing)`).
    // Found by the #808 gate on 2026-11-02: the shipped Qwen3-0.6B plan (decode.M=16) plus a kernel set
    // with no m16 twins refused to load with this check in place.
    // The kernel family the plan names has to be the one this target was configured for.  Asked for by
    // name rather than recomputed from `pb16i_`/`weight_layout_b8_`: a t2s container sets `pb16i_` too
    // (its weights are the integer-exact ternary family), so a local copy of this rule refused every
    // `_t2s`/`_t2s2` plan as "this target runs family _pb16i".  Found by reading the tree against
    // main's #837/#843, which is where the t2s family tag stopped being a constant.
    const std::string want = family ? family : "";
    if (plan_.tag != want)
        { err = "plan: kernels.tag \"" + plan_.tag + "\" but this target runs family \"" +
                want + "\""; return false; }
    // The sealed ELF is plan-derived at the seam already (the forward takes it from the plan when the plan
    // names one), so there is nothing to reconcile here; A2 moves that decision in with the derivation.
    return true;
}

bool XdnaTarget::load_block(int, int, std::string& err) {
    // A2: weight residency per layer block (it is what npu_gemm's cache-miss path does today, one
    // projection at a time).  A1 keeps the lazy per-projection fill, so this is a no-op that says why.
    err = "xdna: load_block is A2 (weights still fill lazily per projection)";
    return false;
}

bool XdnaTarget::gemv(const GemvArgs& ar, std::vector<float>& out, std::string& err) {
    std::string e;
    if (ar.rows) npu_gemm_rows(ar, out, e);
    else         npu_gemm(ar, out, e);
    if (!e.empty()) { err = e; return false; }
    return true;
}

Caps XdnaTarget::caps() const {
    Caps c;
    c.max_rows = m16_ ? Q4NX_DESIGN_M_DECODE : Q4NX_DESIGN_M_PREFILL;
    c.prefill_rows = Q4NX_DESIGN_M_PREFILL;
    c.t2_expansion = false;      // P3 brings it
    c.gdn = false;               // the GDN recurrence is the model's today
    c.attn = false;
    c.lm_head = false;           // the head borrows a GEMM kernel; it is not a target op yet
    c.int8_kv = false;
    return c;
}

// --------------------------------------------------------------------------
// setup (the forward still derives; A2 moves this behind open())
// --------------------------------------------------------------------------
void XdnaTarget::set_device(xrt::device* dev, bool weight_layout_b8, bool weight_cache, int n_layers) {
    dev_ = dev;
    weight_layout_b8_ = weight_layout_b8;
    weight_cache_ = weight_cache;
    n_layers_ = n_layers;
}

void XdnaTarget::set_pb16i(bool on, std::string dir, std::string sealed) {
    pb16i_ = on;
    pb16i_dir_ = std::move(dir);
    pb16i_sealed_ = std::move(sealed);
}

bool XdnaTarget::weight_cached(int layer, int key) const {
    return weight_cache_ && layer >= 0 && layer < (int)wcache_.size() &&
           wcache_[(size_t)layer].find(key) != wcache_[(size_t)layer].end();
}

int XdnaTarget::max_rows() const {
    /* engine#1900/#1903: MIN over the loaded per-role widths, never the first populated one. This width IS the chunk
     * width, and the activation BO holds exactly that many rows -- an over-wide launch reads and poisons past the BO,
     * which is memory corruption rather than a slow path (engine#1372). Returning the FIRST entry meant a mixed set
     * answered whichever role happened to be index 0, so a 16/16/128 set could answer 128: the dangerous direction.
     * The pb16i_ case keeps main's fall-through (a pb16i set that adopted only its prefill kernels has no 16-row twin,
     * and answering 1 there would walk single-row passes), and #1904's plan-derived width for the sealed route stays
     * below exactly as merged. */
    const int n = ndesigns_ < Q4NX_MAX_DESIGNS ? ndesigns_ : Q4NX_MAX_DESIGNS;
    int widths16[Q4NX_MAX_DESIGNS] = {0};
    bool any16 = false;
    if (pb16i_)
        for (int i = 0; i < n; i++)
            if (ctx16_[i]) { widths16[i] = ctx16_[i]->MD; any16 = true; }
    if (any16) {
        // engine#1990: with a wide family adopted for every role, a chunk may be as wide as that family.
        return onebit::npu::chunk_rows(onebit::npu::batch_width_min(widths16, n, 1), wide_rows_);
    }
    int widths[Q4NX_MAX_DESIGNS] = {0};
    bool any_ctx = false;
    for (int i = 0; i < n; i++)
        if (ctx_[i]) { widths[i] = ctx_[i]->MD; any_ctx = true; }
    if (any_ctx) return onebit::npu::batch_width_min(widths, n, 1);
    // engine#1900: the sealed/mono-ELF generation adopts its kernel set through a path that fills neither array, so both
    // loops above find nothing and this returned 1. This is the CHUNK WIDTH (q4nx_forward's `blk = max_batch_rows()`), so
    // those packs batched ONE ROW per launch however wide their design is -- `models/npu/bonsai2-27b-NPU-b8` states prefill
    // M=128 and reported 1. Only reachable when no per-role context was adopted, so nothing that has one changes; and the
    // plan is the pack's own statement of the width its kernels were built for (a plan whose M disagrees with adopted
    // kernels is already refused above). Still 1 when the plan states nothing: guessing a width launches into a BO sized for
    // fewer rows, which is the memory-corruption path engine#1372 records.
    if (plan_.prefill_m > 0) return plan_.prefill_m;
    return 1;
}

int XdnaTarget::rows_for(int role, int want) const {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return want;
    if (m16_ && want >= 1 && want <= 16 && ctx16_[role]) return ctx16_[role]->MD;
    if (pb16i_ && ctxw_[role] && onebit::npu::wide_takes(want, Q4NX_DESIGN_M_DECODE, wide_rows_))
        return ctxw_[role]->MD;   // engine#1990
    return ctx_[role] ? ctx_[role]->MD : want;
}

I8Ctx* XdnaTarget::make_ctx(int role) {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    ctx_[role] = std::make_unique<I8Ctx>();
    return ctx_[role].get();
}
I8Ctx* XdnaTarget::ctx(int role) const {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    return ctx_[role].get();
}
std::unique_ptr<I8Ctx>& XdnaTarget::ctx_ref(int role) { return ctx_[role]; }

I8Ctx* XdnaTarget::make_ctx16(int role, const Q4nxDesignGeom& geom) {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    designs16_[role] = geom;
    ctx16_[role] = std::make_unique<I8Ctx>();
    return ctx16_[role].get();
}
I8Ctx* XdnaTarget::ctx16(int role) const {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    return ctx16_[role].get();
}
I8Ctx* XdnaTarget::make_ctx_wide(int role) {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    ctxw_[role] = std::make_unique<I8Ctx>();
    return ctxw_[role].get();
}
I8Ctx* XdnaTarget::ctx_wide(int role) const {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    return ctxw_[role].get();
}
bool XdnaTarget::adopt_wide(int rows) {
    int narrow[Q4NX_MAX_DESIGNS] = {0}, wide[Q4NX_MAX_DESIGNS] = {0};
    for (int i = 0; i < Q4NX_MAX_DESIGNS; i++) {
        if (ctx16_[i]) narrow[i] = ctx16_[i]->MD;
        if (ctxw_[i]) wide[i] = ctxw_[i]->MD;
    }
    if (!pb16i_ || !onebit::npu::wide_family_complete(narrow, wide, Q4NX_MAX_DESIGNS, rows)) {
        drop_wide();
        return false;
    }
    wide_rows_ = rows;
    return true;
}
I8Ctx* XdnaTarget::make_ctx_narrow(int role) {
    if (role < 0 || role >= Q4NX_MAX_DESIGNS) return nullptr;
    ctxn_[role] = std::make_unique<I8Ctx>();
    return ctxn_[role].get();
}
bool XdnaTarget::adopt_narrow(int rows) {
    int base[Q4NX_MAX_DESIGNS] = {0}, narrow[Q4NX_MAX_DESIGNS] = {0};
    for (int i = 0; i < Q4NX_MAX_DESIGNS; i++) {
        if (ctx16_[i]) base[i] = ctx16_[i]->MD;
        if (ctxn_[i]) narrow[i] = ctxn_[i]->MD;
    }
    if (!pb16i_ || !onebit::npu::narrow_family_complete(base, narrow, Q4NX_MAX_DESIGNS, rows)) {
        drop_narrow();
        return false;
    }
    narrow_rows_ = rows;
    return true;
}
void XdnaTarget::drop_narrow() {
    for (auto& c : ctxn_) c.reset();
    narrow_rows_ = 0;
}
void XdnaTarget::drop_wide() {
    for (auto& c : ctxw_) c.reset();
    wide_rows_ = 0;
}
void XdnaTarget::drop_ctx16(int role) {
    if (role >= 0 && role < Q4NX_MAX_DESIGNS) ctx16_[role].reset();
}
I8Ctx* XdnaTarget::make_head_ctx() {
    ctx_head_ = std::make_unique<I8Ctx>();
    return ctx_head_.get();
}
I8Ctx* XdnaTarget::head_ctx() const { return ctx_head_.get(); }
std::unique_ptr<I8Ctx>& XdnaTarget::head_ctx_ref() { return ctx_head_; }

void XdnaTarget::dump_profile() { prof_dump(); }

void XdnaTarget::note_head(int M, double dpk, double dql, double dwt, double drb) {
    prof_head_M_ = M;
    prof_add(&prof_head_, 0, dpk, dql, dwt, drb);
    prof_dec_n_++;
}

// --------------------------------------------------------------------------
// the moved launch layer
// --------------------------------------------------------------------------
int XdnaTarget::design_index(I8Ctx* c) const {
    if (!c) return -1;
    for (int i = 0; i < ndesigns_ && i < Q4NX_MAX_DESIGNS; i++)
        if (ctx_[i].get() == c || (ctx16_[i] && ctx16_[i].get() == c) || (ctxw_[i] && ctxw_[i].get() == c) ||
            (ctxn_[i] && ctxn_[i].get() == c))
            return i;
    return -1;
}

I8Ctx* XdnaTarget::ctx_for(I8Ctx* prefill, int rows, bool decode_twin) {
    // Rows that fit one M=16 row-block (decode's 1 row, or a short prompt's
    // prefill chunk of <= 16) run the cheap twin: the M=128 kernel streams the
    // same weights but walks 128 baked MAC rows to deliver <= 16, and a request
    // pays one full weight pass in prefill whatever its prompt length.  Chunks
    // wider than 16 keep M=128 (one launch per chunk), so long prefill is
    // unchanged.
    // decode_twin forces the narrow kernel even for a wide launch: the pb16i prefill path uses it for a
    // projection that has no M=128 kernel at all.
    if (!m16_) return prefill;
    if (!decode_twin && (rows < 1 || rows > 16)) return prefill;
    for (int i = 0; i < ndesigns_ && i < Q4NX_MAX_DESIGNS; i++)
        if (ctx_[i].get() == prefill && ctx16_[i]) return ctx16_[i].get();
    return prefill;
}

int64_t XdnaTarget::m16_check_gemm(I8Ctx& ref, I8Ctx& fast, const float* A, int K,
                                   int N, const float* Bmat, xrt::bo* wbo, float as) {
    // Same activation, same quantizer, same weight BO as the launch that just ran on
    // the M=16 kernel -- only the ELF differs, so any output difference is the
    // kernel's, not the plumbing's.
    xrt::run r;
    if (wbo) {
        r = ref.launch_async_with_bo(*wbo, A, 1, K, as);
    } else {
        float s = 1.0f;
        ref.packB(0, Bmat, K, N, s);
        ref.quantize_async(A, 1, K, as);
        r = ref.sync_and_launch(0);
    }
    ref.wait_kernel(r);
    ref.readback();
    // engine#666: a short read of the REFERENCE row would look like a kernel mismatch and be blamed on the
    // M=16 twin. Re-issue the reference once before comparing.
    if (ref.canary_short()) {
        ref.canary_retries_++;
        if (wbo) {
            r = ref.launch_async_with_bo(*wbo, A, 1, K, as);
        } else {
            float s = 1.0f;
            ref.packB(0, Bmat, K, N, s);
            ref.quantize_async(A, 1, K, as);
            r = ref.sync_and_launch(0);
        }
        ref.wait_kernel(r);
        ref.readback();
    }
    int64_t maxd = 0;
    long nmis = 0;
    for (int n = 0; n < N; n++) {
        const int64_t d = (int64_t)ref.Cm[n] - (int64_t)fast.Cm[n];
        const int64_t ad = d < 0 ? -d : d;
        if (ad) { nmis++; if (ad > maxd) maxd = ad; }
    }
    m16_checked_++;
    if (maxd > m16_worst_) m16_worst_ = maxd;
    if (maxd) {
        m16_bad_++;
        fprintf(stderr, "[m16check] MISMATCH N=%d maxdiff=%lld mismatched_cols=%ld "
                        "(K=%d)\n", N, (long long)maxd, nmis, K);
    } else if (m16_checked_ % 250 == 0) {
        fprintf(stderr, "[m16check] n=%ld maxdiff=0 (bit-exact)\n", m16_checked_);
    }
    return maxd;
}

int64_t XdnaTarget::m16_check_gemm_rows(I8Ctx& ref, I8Ctx& fast, const float* A,
                                        int M, int K, int N, xrt::bo* wbo,
                                        const std::vector<float>& ascales) {
    // Same M rows, same per-row scales, same weight BO: only the ELF differs.
    xrt::run r = wbo ? ref.launch_async_with_bo_rows(*wbo, A, M, K, ascales.data())
                     : ref.launch_async_rows(0, A, M, K, ascales.data());
    ref.wait_kernel(r);
    ref.readback();
    if (ref.canary_short()) {   // engine#666, same reasoning as m16_check_gemm
        ref.canary_retries_++;
        r = wbo ? ref.launch_async_with_bo_rows(*wbo, A, M, K, ascales.data())
                : ref.launch_async_rows(0, A, M, K, ascales.data());
        ref.wait_kernel(r);
        ref.readback();
    }
    int64_t maxd = 0;
    long nmis = 0;
    for (int m = 0; m < M; m++) {
        const int32_t* a = ref.Cm + (size_t)m * ref.ND;
        const int32_t* b = fast.Cm + (size_t)m * fast.ND;
        for (int n = 0; n < N; n++) {
            const int64_t d = (int64_t)a[n] - (int64_t)b[n];
            const int64_t ad = d < 0 ? -d : d;
            if (ad) { nmis++; if (ad > maxd) maxd = ad; }
        }
    }
    m16_checked_++;
    if (maxd > m16_worst_) m16_worst_ = maxd;
    if (maxd) {
        m16_bad_++;
        fprintf(stderr, "[m16check] MISMATCH rows=%d N=%d maxdiff=%lld "
                        "mismatched_cols=%ld (K=%d)\n", M, N, (long long)maxd, nmis, K);
    } else if (m16_checked_ % 250 == 0) {
        fprintf(stderr, "[m16check] n=%ld maxdiff=0 (bit-exact)\n", m16_checked_);
    }
    return maxd;
}

void XdnaTarget::prof_add(ProfAcc* acc, int design, double pack, double quant,
                              double wait, double rb) {
    // g_ph is what the [bs] (batch) and [ph] (decode step) lines print, and both read it. Only the DECODE
    // accumulator used to be counted, so a 219-row prefill -- which launches through prof_pre_ -- reported
    // "n=0 launches, host=93113 ms" for 224 launches and 93 s of device wait: the wait fell into the line's
    // host= residual and the launch count stayed zero, which reads exactly like a host fallback and sent one
    // lane (engine #1582) looking for a fallback that does not exist. The batched body sets g_in_batch for
    // its whole duration, so accepting prof_pre_ there counts those launches and keeps a decode-only
    // measurement decode-only.
    // The head is NEVER counted here, and that is the second half of this gate. `g_in_batch` stays set for the whole of
    // step_batch, and the batched head (run_lm_head_rows -> note_head -> prof_add(&prof_head_, ...)) runs inside that
    // window without setting g_in_head -- which only the single-row head does. So accepting "any accumulator while in a
    // batch" put the head's device time into g_ph as well, where [bs] computes host = layers - gsum: the same time was
    // then reported twice, once in gsum and once on the epilogue line's own h_* slices, and host was understated by it
    // (it can go negative) on exactly the multi-row passes this line exists for. The head keeps its own accounting.
    if (q4nx_lh::phprof_on() && acc != &prof_head_ &&
        (acc == prof_dec_ || (q4nx_lh::g_in_batch && acc == prof_pre_)) && !q4nx_lh::g_in_head) {
        q4nx_lh::g_ph.g_pack += pack; q4nx_lh::g_ph.g_ql += quant;
        q4nx_lh::g_ph.g_wait += wait; q4nx_lh::g_ph.g_rb += rb;
        q4nx_lh::g_ph.launches++;
    }
    if (design < 0 || design >= Q4NX_MAX_DESIGNS) return;
    ProfAcc& a = acc[design];
    a.pack += pack; a.quant += quant; a.wait += wait; a.rb += rb;
    a.n++;
}

void XdnaTarget::prof_dump() {
    fprintf(stderr, "[prof] ---- per-role launch cost (ms, window since last dump) ----\n");
    fprintf(stderr, "[prof] %-5s %-5s %2s %8s %7s %9s %7s %9s %9s\n", "phase", "role",
            "M", "launches", "pack", "quant+lnch", "wait", "readback", "ms/launch");
    for (int ph = 0; ph < 2; ph++) {
        ProfAcc* acc = ph ? prof_pre_ : prof_dec_;
        for (int i = 0; i < ndesigns_; i++) {
            const ProfAcc& a = acc[i];
            if (!a.n) continue;
            const double n = (double)a.n;
            const int M = (ctx16_[i] && ph == 0) ? designs16_[i].M : designs_[i].M;
            fprintf(stderr, "[prof] %-5s %-5s %2d %8ld %7.2f %9.2f %7.3f %9.3f %9.3f\n",
                    ph ? "pre" : "dec", role_tag(designs_[i].role), M, a.n,
                    a.pack / n, a.quant / n, a.wait / n, a.rb / n,
                    (a.pack + a.quant + a.wait + a.rb) / n);
        }
    }
    if (prof_head_.n) {
        const double n = (double)prof_head_.n;
        fprintf(stderr, "[prof] %-5s %-5s %2d %8ld %7.2f %9.2f %7.3f %9.3f %9.3f\n", "dec",
                "HEAD", prof_head_M_, prof_head_.n, prof_head_.pack / n, prof_head_.quant / n,
                prof_head_.wait / n, prof_head_.rb / n,
                (prof_head_.pack + prof_head_.quant + prof_head_.wait + prof_head_.rb) / n);
    }
    fprintf(stderr, "[prof] dispatch decode=%ld prefill=%ld tokens=%ld (decode/token=%.1f)"
                    " m16=%d\n", prof_dec_n_, prof_pre_n_, prof_tok_n_,
            prof_tok_n_ ? (double)prof_dec_n_ / (double)prof_tok_n_ : 0.0, (int)m16_);
    if (had_n_)
        fprintf(stderr, "[prof] hadamard host: %.1f ms over %ld row-launch(es) = %.1f ms/token\n",
                had_ms_, had_n_, prof_tok_n_ ? had_ms_ / (double)prof_tok_n_ : 0.0);
    if (m16_check_)
        fprintf(stderr, "[m16check] ---- checks=%ld mismatched=%ld worst_maxdiff=%lld"
                        " ----\n", m16_checked_, m16_bad_, (long long)m16_worst_);
    /* Window semantics: each dump reports the launches since the previous dump,
       so one burst of device contention cannot bias a whole run.  The dispatch /
       token totals above stay cumulative. */
    for (int i = 0; i < Q4NX_MAX_DESIGNS; i++) {
        prof_dec_[i] = ProfAcc{};
        prof_pre_[i] = ProfAcc{};
    }
    prof_head_ = ProfAcc{};
}

namespace {
// Same test the forward's file_exists() made: the file opens and is not empty.
bool decode_elf_exists(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz >= 1;
}
}  // namespace

// The log lines and refusals below are byte-for-byte the forward's: gates grep them.
bool XdnaTarget::open_decode_family(Q4nxDesignGeom* designs16, int n, const std::string& sealed_elf,
                                    int* made, int* skipped, std::string& err) {
    int made_n = 0, skipped_n = 0;
    for (int i = 0; i < n; i++) {
        if (designs16[i].M <= 0) { ++skipped_n; continue; }
        std::string elf, kname = "main:seq";
        if ((setup_.pb16i || setup_.b8) && !sealed_elf.empty()) {
            // One ELF for every phase: the file is the same for all roles and only the instance name
            // differs, so I8Ctx reuses the one hw_context. pb16i and b8 (engine#2208) each have their own
            // instance table.
            const char* inst = setup_.pb16i ? q4nx_pb16i_instance(designs16[i].role)
                                            : q4nx_i8_sealed_instance(designs16[i].role);
            if (!inst) {
                if (setup_.pb16i) {
                    std::fprintf(stderr, "  [forward] pb16i: the sealed ELF has no instance for "
                                         "role %s; this role has no decode kernel\n",
                                 q4nx_design_log_tag(designs16[i].role));
                    designs16[i].M = 0;
                    ++skipped_n;
                    continue;
                }
                // b8: an uncovered role is not a missing kernel; its per-role M=16 twin still serves it below.
                std::fprintf(stderr, "  [forward] b8: the sealed ELF has no instance for role %s; "
                                     "it keeps its per-role decode kernel\n",
                             q4nx_design_log_tag(designs16[i].role));
            } else {
                elf = setup_.kernels_dir + "/" + sealed_elf;
                kname = inst;
                if (!decode_elf_exists(elf)) {
                    err = plan_.loaded() ? ("plan " + setup_.plan_path + ": decode.sealed_elf \"" + sealed_elf +
                                            "\" is not in " + setup_.kernels_dir + " (missing)")
                                         : (std::string(setup_.pb16i ? "pb16i" : "b8") + ": missing sealed ELF " + elf);
                    return false;
                }
            }
        }
        if (elf.empty()) {   // not sealed, or a b8 role the sealed ELF does not cover
            for (const std::string& d : setup_.decode_dirs) {
                const std::string cand = d + "/" + designs16[i].elf_name;
                if (decode_elf_exists(cand)) { elf = cand; break; }
            }
        }
        if (elf.empty()) {
            // A pb16i role without its image kernel must stop the load: it is not a slower path, it is a missing
            // one (the int8 weights are not packed for these geometries under this layout).
            if (setup_.pb16i) {
                err = plan_.loaded() ? std::string("plan ") + setup_.plan_path + ": decode role " +
                                           q4nx_design_log_tag(designs16[i].role) + " needs \"" + designs16[i].elf_name +
                                           "\" (missing)"
                                     : std::string("pb16i: missing decode ELF ") + designs16[i].elf_name;
                return false;
            }
            ++skipped_n;
            continue;   // stay on M=128
        }
        I8Ctx* c16 = make_ctx16(i, designs16[i]);
        if (!c16) {
            err = "cannot create the decode context for role " + std::to_string(i);
            return false;
        }
        c16->MD = designs16[i].M;
        c16->KD = designs16[i].K;
        c16->ND = designs16[i].N;
        c16->blocked_b = setup_.b8;
        c16->pb16i = setup_.pb16i;
        c16->t2s = setup_.t2s;
        // Which pb16i ELFs leave two int32 planes (the arena's ACC64 family: D, O and the full-attention q half).
        // This mirrors the ELF family contract rather than inferring it, and load/launch refuse a tensor that
        // needs two planes on a one-plane ELF.
        if (setup_.pb16i) {
            const Q4nxDesign r = designs16[i].role;
            c16->pb16i_planes = (r == Q4NX_DESIGN_D || r == Q4NX_DESIGN_O || r == Q4NX_DESIGN_QKVFA_Q) ? 2 : 1;
        }
        if (!c16->init_elf(*dev_, elf.c_str(), designs16[i].M, designs16[i].K, designs16[i].N, 1, kname.c_str())) {
            if (setup_.pb16i) {
                err = "pb16i: init_elf failed for " + elf;
                return false;
            }
            std::fprintf(stderr, "  [forward] M=16 twin failed to load (%s); "
                                 "this role keeps M=128\n", designs16[i].elf_name);
            drop_ctx16(i);
            ++skipped_n;
            continue;
        }
        ++made_n;
        std::fprintf(stderr, "  [forward] %-7s M=%d K=%d N=%d  %s (decode)\n",
                     q4nx_design_log_tag(designs16[i].role), designs16[i].M, designs16[i].K, designs16[i].N, elf.c_str());
    }
    note_m16_made(made_n > 0);
    if (made) *made = made_n;
    if (skipped) *skipped = skipped_n;
    return true;
}

bool XdnaTarget::open_head(const Q4nxDesignGeom* designs, int n, std::string& err) {
    // run_lm_head packs int8 B tiles from the f32 head and launches them itself, so it
    // needs an int8 context with K=H.  Under pb16i every decode context is a pb16i one,
    // so build it here: the widest K=H design, at M=16 if its int8 twin is on disk.
    if (setup_.pb16i && !setup_.host_gemm) {
        int i_head = -1;
        for (int i = 0; i < n; i++) {
            if (!ctx(i) || designs[i].K != setup_.hidden || designs_[i].N <= 0) continue;
            if (i_head < 0 || designs[i].N > designs[i_head].N) i_head = i;
        }
        // The head's ternary images come first, when the converter produced them: output.weight
        // is PTQ1_0 (type 143) in the folded GGUF, so the image is exact rather than a
        // requantisation, it is 0.75 GB against the int8 head's 1.33 GB, and it runs the same
        // K=H kernel the MLP's GU launch runs.  Missing images are not an error: the head then
        // keeps the int8 family below.
        if (i_head >= 0 && !setup_.pb16i_dir.empty() && setup_.head_pb16i_images) {
            const Q4nxDesignGeom& g = designs[i_head];
            std::string stem(g.elf_name);
            stem.resize(stem.size() - 4);                     // drop ".elf"
            const std::string ft = setup_.family;
            const std::string cands[] = {stem + ft + "_m16.elf", stem + ft + ".elf"};
            std::string pelf;
            for (const std::string& c : cands) {
                for (const std::string& d : setup_.decode_dirs) {
                    if (elf_present(d + "/" + c)) { pelf = d + "/" + c; break; }
                }
                if (!pelf.empty()) break;
            }
            if (!pelf.empty()) {
                I8Ctx* hc = make_head_ctx();
                if (!hc) { err = "cannot create the lm-head context"; return false; }
                hc->MD = Q4NX_DESIGN_M_DECODE;
                hc->KD = g.K;
                hc->ND = g.N;
                hc->pb16i = true;
                hc->t2s = setup_.t2s;
                hc->pb16i_planes = 1;   // the K=H design is the arena's one-plane family
                if (hc->init_elf(*dev_, pelf.c_str(), Q4NX_DESIGN_M_DECODE, g.K, g.N, 1)) {
                    head_pb16i_ = true;
                    fprintf(stderr, "  [forward] lm head M=%d K=%d N=%d  %s (pb16i ternary "
                                    "images)\n", hc->MD, g.K, g.N, pelf.c_str());
                } else {
                    head_ctx_ref().reset();
                }
            }
        }
        if (!head_ctx() && i_head >= 0) {
            const Q4nxDesignGeom& g = designs[i_head];
            // The head streams 1.33 GB of int8 weights a token, so its layout matters more
            // than its row count: a blocked (_b8) ELF walks 8 KiB forward per tile at
            // 8.25 GB/s where the row-major one runs 128 separate 128-byte reads at
            // 3.3 GB/s (measured: 27 ms a launch either way for M=16 and M=128).  Try the
            // blocked M=16 twin first, then whatever int8 family is on disk.
            std::string stem(g.elf_name);
            stem.resize(stem.size() - 4);                     // drop ".elf"
            const std::string cands[] = {stem + "_b8_m16.elf", stem + "_m16.elf",
                                         stem + "_b8.elf", g.elf_name};
            std::string elf;
            bool blocked = setup_.b8;
            for (const std::string& c : cands) {
                for (const std::string& d : setup_.decode_dirs) {
                    if (elf_present(d + "/" + c)) { elf = d + "/" + c; break; }
                }
                if (!elf.empty()) {
                    blocked = c.find("_b8") != std::string::npos;
                    break;
                }
            }
            int m = elf.find("_m16.elf") != std::string::npos ? Q4NX_DESIGN_M_DECODE : g.M;
            if (elf.empty()) { elf = setup_.kernels_dir + "/" + g.elf_name; m = g.M; }
            I8Ctx* hc = make_head_ctx();
            if (!hc) { err = "cannot create the lm-head context"; return false; }
            hc->MD = m; hc->KD = g.K; hc->ND = g.N;
            hc->blocked_b = blocked;
            if (!hc->init_elf(*dev_, elf.c_str(), m, g.K, g.N, 1)) {
                fprintf(stderr, "  [forward] lm head: cannot load %s; the head keeps the "
                                "decode family\n", elf.c_str());
                head_ctx_ref().reset();
            } else {
                fprintf(stderr, "  [forward] lm head M=%d K=%d N=%d  %s\n", m, g.K, g.N,
                        elf.c_str());
            }
        }
    }

    // ---- the lm head on the q4_1 gemv family ----
    // That family has no int8 packer: its weight BO holds the container's own q4_1 tile stream and the
    // launch dequantises from those tiles, so the tiled int8 head above cannot express it -- packing an
    // int8 image into a tile BO both overruns the BO by (KD*ND - tile bytes) and means nothing to the
    // kernel.  The head is therefore ONE launch whose N is the vocabulary: the only width whose output
    // is the whole logits row, which is what run_lm_head_gemv() gathers the container's head tiles for.
    if (!setup_.host_gemm && !head_ctx()) {
        bool any_gemv = false;
        for (int i = 0; i < ndesigns_ && !any_gemv; i++)
            any_gemv = (ctx(i) && ctx(i)->gemv) || (ctx16(i) && ctx16(i)->gemv);
        if (any_gemv) {
            // The head ELF's N is on the gemv column grid, so try the padded vocabulary first and the
            // raw one after: the pack's vocab 151936 is 152064 on the grid.
            std::string elf;
            for (int nn : {q4nx_pad_n(setup_.vocab), setup_.vocab}) {
                for (const char* suf : {"_m16.elf", ".elf"}) {
                    const std::string n = "full_i8_LMHEAD_K" + std::to_string(setup_.kpad) + "_N" +
                                          std::to_string(nn) + suf;
                    for (const std::string& d : setup_.decode_dirs) {
                        if (elf_present(d + "/" + n)) { elf = d + "/" + n; break; }
                    }
                    if (!elf.empty()) break;
                }
                if (!elf.empty()) break;
            }
            if (elf.empty()) {
                err = "lm head: this container's kernels are the q4_1 gemv family, whose head is one "
                       "launch at N=vocab, and no head ELF is on disk (want full_i8_LMHEAD_K" +
                       std::to_string(setup_.kpad) + "_N" + std::to_string(q4nx_pad_n(setup_.vocab)) + "[_m16].elf)";
                return false;
            }
            const int m = elf.find("_m16.elf") != std::string::npos ? Q4NX_DESIGN_M_DECODE
                                                                   : Q4NX_DESIGN_M_PREFILL;
            const int n = (elf.find("_N" + std::to_string(q4nx_pad_n(setup_.vocab))) != std::string::npos)
                              ? q4nx_pad_n(setup_.vocab) : setup_.vocab;
            I8Ctx* hc = make_head_ctx();
            if (!hc) { err = "lm head: cannot create the gemv head context"; return false; }
            hc->MD = m; hc->KD = setup_.kpad; hc->ND = n;
            if (!hc->init_elf(*dev_, elf.c_str(), m, setup_.kpad, n, 1) || !hc->gemv) {
                err = "lm head: " + elf + " does not load as the gemv head (K=" +
                       std::to_string(setup_.kpad) + " N=" + std::to_string(n) + ")";
                head_ctx_ref().reset();
                return false;
            }
            fprintf(stderr, "  [forward] lm head M=%d K=%d N=%d  %s (q4_1 gemv tiles)\n", m, setup_.kpad, n,
                    elf.c_str());
        }
    }

    return true;
}

std::string XdnaTarget::pb16i_stem(int layer, const char* tag) const {
    return pb16i_dir_ + "/" + q4nx_pb16i::sidecar_stem(layer, tag);
}

bool XdnaTarget::load_hadamard(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "q4nx_hadamard: cannot open " + path; return false; }
    char magic[4] = {0};
    uint32_t n = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || std::memcmp(magic, "HAD1", 4) != 0 || n == 0 || n > 16) {
        err = "q4nx_hadamard: " + path + " is not a HAD1 sign table";
        return false;
    }
    std::vector<uint32_t> width(n), off(n);
    for (uint32_t i = 0; i < n; i++) {
        f.read(reinterpret_cast<char*>(&width[i]), 4);
        f.read(reinterpret_cast<char*>(&off[i]), 4);
    }
    std::vector<float> all;
    float v = 0.0f;
    while (f.read(reinterpret_cast<char*>(&v), 4)) all.push_back(v);
    for (uint32_t i = 0; i < n; i++) {
        if ((size_t)off[i] + width[i] > all.size()) { err = "q4nx_hadamard: short sign table"; return false; }
        had_signs_[int(width[i])].assign(all.begin() + off[i], all.begin() + off[i] + width[i]);
    }
    hadamard_ = true;
    fprintf(stderr, "  [forward] hadamard: %zu width(s) from %s, block %d (activations rotated "
                    "into the folded basis)\n", had_signs_.size(), path.c_str(), had_block_);
    return true;
}

void XdnaTarget::npu_gemm(const GemvArgs& ar, std::vector<float>& out, std::string& err) {
    I8Ctx* ctxp = ctx(ar.role);
    // A ternary-only design (the pb16i/t2s full-attention QKV halves) has no wide M=128 context: its own
    // entry IS the decode design, so ctx16_ is the only thing that can launch it.  Resolving that here rather
    // than at the call site is what keeps the model from having to know which roles the kernel set gave a wide
    // kernel to (main did it by passing *ctx16_[i] directly, and the move dropped that).
    if (!ctxp) ctxp = ctx16(ar.role);
    if (!ctxp) { err = "no context for role " + std::to_string(ar.role); out.assign((size_t)ar.N, 0.0f); return; }
    I8Ctx& ctx = *ctxp;
    const float* A = ar.x; const int K = ar.K, N = ar.N;
    std::vector<float> a_pad;   // keeps every A read inside the caller's buffer at a padded K
    if (ar.a_k > 0 && ar.a_k < K) {
        a_pad.resize((size_t)K);
        q4nx_pad_a_f32(A, 1, ar.a_k, K, a_pad.data());
        A = a_pad.data();
    }
    const float* Bmat = ar.Bmat; const int wkey = ar.wkey; const char* sc = ar.tag;
    q4nx_lh::LhScope _tlh_gemm(q4nx_lh::lh_in_gdn() ? LH_GEMM_GDN : LH_GEMM);  // amax scan, A quant, out deq
    float amax = 0.0f;
    for (int i = 0; i < K; i++) {
        float a = std::fabs(A[i]);
        if (a > amax) amax = a;
    }
    const float as = amax > 0.0f ? amax / 127.0f : 1.0f;
    // Phase-selected activation tile: a one-row call is decode, so it runs this
    // role's M=16 twin (one core row instead of 128 rows / 8 live tiles).
    I8Ctx* kctx1 = ctx_for(&ctx, 1, ar.decode_twin);
    // A decode step carries one row: with a narrow family it runs the kernel built for a few rows instead of 16.
    if (pb16i_ && ar.role >= 0 && ar.role < Q4NX_MAX_DESIGNS && ctxn_[ar.role] &&
        onebit::npu::narrow_takes(1, narrow_rows_))
        kctx1 = ctxn_[ar.role].get();
    I8Ctx& kctx = *kctx1;
    const bool prof = prof_;
    const auto tp = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // pb16i: the weight BO holds the sidecar image unchanged, so a cache miss is a
    // file read (no dequant, no quantise, no pack) and the result is dequantised with
    // the sidecar's own per-column 2^e_ref table.  `sc` names the tensor; a pb16i
    // context without one is a caller bug, not a fallback.
    if (pb16i_ && kctx.pb16i) {
        out.assign((size_t)N, 0.0f);   // sized before every refusal path below
        if (!sc) { err = "pb16i: no sidecar tag for this projection"; return; }
        PackedWeight* slot = nullptr;
        if (weight_cache_ && wkey >= 0 && ar.layer >= 0 && ar.layer < n_layers_ && dev_) {
            if (int(wcache_.size()) <= ar.layer) wcache_.resize(size_t(ar.layer) + 1);
            auto& m = wcache_[size_t(ar.layer)];
            auto  it = m.find(wkey);
            if (it == m.end()) {
                q4nx_lh::LhScope _tlh_pk(LH_PACK_UPLOAD);
                PackedWeight pw;
                pw.bo = kctx.make_weight_bo(*dev_, /*zero=*/false);
                std::vector<int16_t> eref;
                int mode = 0;
                const std::string stem = pb16i_stem(ar.layer, sc);
                if (!kctx.load_pb16i_into(*pw.bo, stem, eref, mode)) {
                    err = "pb16i: cannot load " + stem;
                    return;
                }
                pw.eref = std::move(eref);
                pw.pb16i_mode = mode;
                it = m.emplace(wkey, std::move(pw)).first;
            }
            slot = &it->second;
        } else {
            err = "pb16i needs the weight cache (routed or unlocked weights have no image)";
            return;
        }
        if (slot->pb16i_mode > kctx.pb16i_planes) {
            err = std::string("pb16i: ") + sc + " needs the two-plane accumulator but this "
                   "role's ELF is the int32 one";
            return;
        }
        // Folded basis: A' = H (A . s) with the sign vector of this input width, then the int8
        // scale is taken from the rotated vector (that is the vector the kernel actually sees).
        const float* Ab = A;
        float asb = as;
        if (hadamard_) {
            auto hs = had_signs_.find(K);
            if (hs == had_signs_.end()) {
                err = "pb16i: no Hadamard sign vector for K=" + std::to_string(K);
                return;
            }
            const auto h0 = std::chrono::steady_clock::now();
            had_buf_.assign(A, A + K);
            q4nx_hadamard::rotate(hs->second.data(), had_buf_.data(), K, had_block_);
            float am = 0.0f;
            for (int i = 0; i < K; i++) {
                const float a = std::fabs(had_buf_[(size_t)i]);
                if (a > am) am = a;
            }
            asb = am > 0.0f ? am / 127.0f : 1.0f;
            Ab = had_buf_.data();
            had_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - h0).count();
            had_n_++;
        }
        if (const char* dump = getenv("NPU_PB16I_DUMP")) {
            static bool done = false;
            if (!done && ar.layer == atoi(dump) && sc && std::string(sc) == "attn_qkv") {
                done = true;
                FILE* f = open_probe_dump();
                if (f) {
                    fwrite(&K, sizeof(int), 1, f);
                    fwrite(A, sizeof(float), (size_t)K, f);
                    fwrite(Ab, sizeof(float), (size_t)K, f);
                    fwrite(kctx.Am, sizeof(int8_t), (size_t)K, f);
                    fwrite(&asb, sizeof(float), 1, f);
                    fclose(f);
                }
                fprintf(stderr, "[pb16i-dump] wrote the dump for %s K=%d\n", sc, K);
            }
        }
        const auto tl = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        xrt::run r = kctx.launch_async_with_bo(*slot->bo, Ab, 1, K, asb);
        const auto tq = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        kctx.wait_kernel(r);
        const auto tw = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        kctx.readback();
        const auto tr = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        out.resize((size_t)N);
        kctx.dequant_pb16i(out.data(), 1, N, &asb, slot->eref, kctx.pb16i_planes);
        // NPU_PB16I_TRACE=<layer>: one line for the named layer's launch in the MODEL LOOP, with
        // the exact inputs and outputs, so the same A can be replayed through the frozen
        // harness.  If the harness reproduces C, the kernel and the images are fine and the bug
        // is what happens to C afterwards; if it does not, the activation quantisation is wrong.
        if (const char* tl = getenv("NPU_PB16I_TRACE")) {
            if (ar.layer == atoi(tl) && sc) {
                fprintf(stderr, "[pb16i-trace] %s raw[0..3]=", sc);
                for (int i = 0; i < 4; i++) fprintf(stderr, "%.9g ", (double)A[(size_t)i]);
                fprintf(stderr, " rot[0..3]=");
                for (int i = 0; i < 4; i++) fprintf(stderr, "%.9g ", (double)Ab[(size_t)i]);
                fprintf(stderr, " as=%.9g planes=%d A[0..7]=", (double)asb,
                        kctx.pb16i_planes);
                for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", (int)kctx.Am[i]);
                fprintf(stderr, " Cm[0..7]=");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", (int)kctx.Cm[i]);
                fprintf(stderr, " out[0..7]=");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.9g ", (double)out[(size_t)i]);
                fprintf(stderr, " eref[0..3]=%d %d %d %d\n", (int)slot->eref[0], (int)slot->eref[1],
                        (int)slot->eref[2], (int)slot->eref[3]);
            }
        }
        if (prof) {
            const double dpk = std::chrono::duration<double, std::milli>(tl - tp).count();
            const double dql = std::chrono::duration<double, std::milli>(tq - tl).count();
            const double dwt = std::chrono::duration<double, std::milli>(tw - tq).count();
            const double drb = std::chrono::duration<double, std::milli>(tr - tw).count();
            prof_add(prof_dec_, design_index(&kctx), dpk, dql, dwt, drb);
            if (++prof_dec_n_ % 100 == 0) prof_dump();
        }
        return;
    }

    // Token-invariant weights (wkey >= 0) get their own BO, packed once and
    // reused; every other call (wkey < 0: routed experts, whose weights change
    // per token) re-packs into layerB[0] as before.
    PackedWeight*             slot = nullptr;
    const std::vector<float>* gs = nullptr;
    if (weight_cache_ && wkey >= 0 && ar.layer >= 0 && ar.layer < n_layers_ && dev_) {
        if (int(wcache_.size()) <= ar.layer) wcache_.resize(size_t(ar.layer) + 1);
        auto& m = wcache_[size_t(ar.layer)];
        auto  it = m.find(wkey);
        if (it == m.end()) {
            // Cache miss only: this is the per-request re-pack/re-upload the
            // `prefill` row would hide.  It must be zero for a warm process.
            q4nx_lh::LhScope _tlh_pk(LH_PACK_UPLOAD);
            PackedWeight pw;
            pw.bo = kctx.make_weight_bo(*dev_);
            if (kctx.gemv) {
                // The gemv family dequantises from the container's own tiles: upload them unchanged.
                if (!ar.tiles) { err = "gemv: the model has no tile stream for this container"; return; }
                std::vector<uint8_t> blob;
                if (!ar.tiles(ar.tiles_self, ar.layer, ar.role, blob, err)) return;
                if (!kctx.upload_tiles_into(*pw.bo, blob.data(), blob.size())) {
                    err = "gemv: tile upload failed"; return;
                }
            } else {
                float sout = 1.0f;
                kctx.packB_into(*pw.bo, Bmat, K, N, sout, pw.scales);
            }
            it = m.emplace(wkey, std::move(pw)).first;
        }
        slot = &it->second;
        gs = &slot->scales;
    } else if (kctx.gemv) {
        err = "gemv: this family needs the weight cache (NPU_NO_WCACHE=1 is not supported with it)";
        return;
    } else {
        float sout = 1.0f;
        kctx.packB(0, Bmat, K, N, sout);
        kctx.quantize_async(A, 1, K, as);
        gs = &kctx.group_scales[0];
    }

    const auto tl = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto issue = [&]() -> xrt::run {
        return slot ? kctx.launch_async_with_bo(*slot->bo, A, 1, K, as)
                    : kctx.sync_and_launch(0);
    };
    xrt::run r = issue();
    const auto tq = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    kctx.wait_kernel(r);
    const auto tw = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    kctx.readback();
    // engine#666: the row about to be dequantised is still the poison the launch was supposed to overwrite,
    // so the writes are not in memory after the invalidate. Re-issue this launch once and read again, rather
    // than silently pushing zeros into the layer. The retry is counted and reported (NPU_CANARY_STATS=1).
    if (!kctx.gemv && kctx.canary_short()) {
        kctx.canary_retries_++;
        r = issue();
        kctx.wait_kernel(r);
        kctx.readback();
        if (q4nx_lh::phprof_on()) g_ph.h_rb += 0.0;   // the retry's cost is already inside the readback timer below
    }
    const auto tr = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    out.resize((size_t)N);
    if (kctx.gemv) {
        for (int n = 0; n < N; n++) out[(size_t)n] = kctx.Cg[n];
    } else {
        for (int n = 0; n < N; n++)
            out[(size_t)n] = (float)kctx.Cm[n] * as * (*gs)[(size_t)n];
    }

    if (!kctx.gemv && kctx.canary_short()) {
        // twice in a row: say so, loudly, and keep going -- an empty row here is a wrong activation either way
        fprintf(stderr, "[canary] SHORT RESULT twice: role K=%d N=%d layer=%d short=%d retries=%ld\n",
                K, N, ar.layer, kctx.short_row_, kctx.canary_retries_);
    }

    if (m16_check_ && &kctx != &ctx)
        m16_check_gemm(ctx, kctx, A, K, N, Bmat, slot ? slot->bo.get() : nullptr, as);

    if (prof) {
        const double dpk = std::chrono::duration<double, std::milli>(tl - tp).count();
        const double dql = std::chrono::duration<double, std::milli>(tq - tl).count();
        const double dwt = std::chrono::duration<double, std::milli>(tw - tq).count();
        const double drb = std::chrono::duration<double, std::milli>(tr - tw).count();
        prof_add(prof_dec_, design_index(&kctx), dpk, dql, dwt, drb);
        if (++prof_dec_n_ % 100 == 0) prof_dump();
    }
}

void XdnaTarget::npu_gemm_rows(const GemvArgs& ar, std::vector<float>& out, std::string& err) {
    I8Ctx* ctxp = ctx(ar.role);
    if (!ctxp) ctxp = ctx16(ar.role);   // same reason as npu_gemm: a ternary-only design lives in ctx16_
    if (!ctxp) { err = "no context for role " + std::to_string(ar.role); out.assign((size_t)ar.M * ar.N, 0.0f); return; }
    I8Ctx& ctx = *ctxp;
    const int M = ar.M, K = ar.K, N = ar.N;
    if (ctx.gemv) {
        // The gemv family is one row per launch (its `_m16` ELFs are the same one-row design), so a
        // batch is a loop of single-row launches.  Correct, and slower per row than a batched
        // kernel -- that gap is issue #671 item 2's m128 work, not something to hide here.
        out.assign((size_t)M * (size_t)N, 0.0f);
        const int astride = (ar.a_k > 0 && ar.a_k < K) ? ar.a_k : K;
        for (int m = 0; m < M; m++) {
            GemvArgs one = ar;
            one.rows = false;
            one.M = 1;
            one.x = ar.x + (size_t)m * (size_t)astride;
            std::vector<float> got;
            std::string e;
            npu_gemm(one, got, e);
            if (!e.empty()) { err = e; return; }
            std::copy(got.begin(), got.end(), out.begin() + (size_t)m * (size_t)N);
        }
        return;
    }
    const float* A = ar.x;
    std::vector<float> a_pad;   // as in npu_gemm(): A holds real_k columns, the kernel reads K
    if (ar.a_k > 0 && ar.a_k < K) {
        a_pad.resize((size_t)M * (size_t)K);
        q4nx_pad_a_f32(A, M, ar.a_k, K, a_pad.data());
        A = a_pad.data();
    }
    const float* Bmat = ar.Bmat; const int wkey = ar.wkey; const char* sc = ar.tag;
    // Per-row activation scales: row m is quantized with its own amax/127 so a
    // batch's rows each keep their dynamic range (issue #1699: a single shared
    // scale zeroed low-magnitude rows when one token's activations dwarfed the
    // rest).
    std::vector<float> ascales((size_t)M);
    // A Hadamard-folded container takes its per-row scale from the ROTATED row, in the block below, which overwrites
    // every entry this loop writes. So on that path this sweep over the unrotated activations is dead work -- and it
    // is not free: it is a scalar pass over M x K floats, measured at ~3.2 ms per 128-row launch at K=4864 and
    // ~2.3 ms at K=12288 (host bench, `pack_bench`), sitting inside the span the profile column calls `pack`.
    // engine #1416 reads that column as the row quantiser's cost; the quantiser is 0.16-0.79 ms per launch, and this
    // sweep plus the fold below are what the column is actually timing.
    // Skipped only where the block below is certain to overwrite it: that requires BOTH the fold and the pb16i
    // layout. `load_hadamard()` is called from the pb16i branch of the container parser, so in practice the fold
    // implies the layout -- but the condition names both, because leaving `ascales` at its value-initialised zeros
    // (a `std::vector<float>(M)` is all zeros) on any other path would change numerics silently rather than fail.
    if (!(hadamard_ && pb16i_)) {
        for (int m = 0; m < M; m++) {
            const float* row = A + (size_t)m * K;
            float amax = 0.0f;
            for (int k = 0; k < K; k++) {
                const float a = std::fabs(row[k]);
                if (a > amax) amax = a;
            }
            ascales[(size_t)m] = amax > 0.0f ? amax / 127.0f : 1.0f;
        }
    }
    // Rows > 1 is prefill and keeps the M=128 kernel (one launch per chunk); a
    // single row here is a decode step and gets the M=16 twin, exactly like
    // npu_gemm().
    I8Ctx* kctxp = ctx_for(&ctx, M, ar.decode_twin);
    // engine#1990: more rows than the 16-row family holds go to the wide family when the set has one. The weight
    // BO, its e_ref and its mode are the launch's (layer, key) slot, not the context's, so both families read the
    // same image; only the row count of the activation and result buffers differs.
    if (pb16i_ && ar.role >= 0 && ar.role < Q4NX_MAX_DESIGNS && ctxw_[ar.role] &&
        onebit::npu::wide_takes(M, Q4NX_DESIGN_M_DECODE, wide_rows_))
        kctxp = ctxw_[ar.role].get();
    else if (pb16i_ && ar.role >= 0 && ar.role < Q4NX_MAX_DESIGNS && ctxn_[ar.role] &&
             onebit::npu::narrow_takes(M, narrow_rows_))
        kctxp = ctxn_[ar.role].get();   // a short chunk (a verify pass of a few rows) fits the narrow kernel
    I8Ctx& kctx = *kctxp;
    const bool prof = prof_;
    const auto tp = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (pb16i_ && kctx.pb16i) {
        out.assign((size_t)M * (size_t)N, 0.0f);   // sized before every refusal path below
        if (M > kctx.MD) {
            // Never launch more rows than the kernel's buffers hold: that reads and writes past them (engine#1372).
            err = "pb16i: " + std::to_string(M) + " rows on a " + std::to_string(kctx.MD) + "-row kernel";
            return;
        }
        if (!sc) { err = "pb16i: no sidecar tag for this projection"; return; }
        PackedWeight* slot = nullptr;
        if (weight_cache_ && wkey >= 0 && ar.layer >= 0 && ar.layer < n_layers_ && dev_) {
            if (int(wcache_.size()) <= ar.layer) wcache_.resize(size_t(ar.layer) + 1);
            auto& m = wcache_[size_t(ar.layer)];
            auto  it = m.find(wkey);
            if (it == m.end()) {
                q4nx_lh::LhScope _tlh_pk(LH_PACK_UPLOAD);
                PackedWeight pw;
                pw.bo = kctx.make_weight_bo(*dev_, /*zero=*/false);
                std::vector<int16_t> eref;
                int mode = 0;
                const std::string stem = pb16i_stem(ar.layer, sc);
                if (!kctx.load_pb16i_into(*pw.bo, stem, eref, mode)) {
                    err = "pb16i: cannot load " + stem;
                    return;
                }
                pw.eref = std::move(eref);
                pw.pb16i_mode = mode;
                it = m.emplace(wkey, std::move(pw)).first;
            }
            slot = &it->second;
        } else {
            err = "pb16i needs the weight cache (routed or unlocked weights have no image)";
            return;
        }
        if (slot->pb16i_mode > kctx.pb16i_planes) {
            err = std::string("pb16i: ") + sc + " needs the two-plane accumulator but this "
                   "role's ELF is the int32 one";
            return;
        }
        // Folded basis, batched form: rotate every row of A into the container's basis and take
        // each row's int8 scale from the rotated vector, exactly as npu_gemm() does per launch.
        const float* Ab = A;
        if (hadamard_) {
            auto hs = had_signs_.find(K);
            if (hs == had_signs_.end()) {
                err = "pb16i: no Hadamard sign vector for K=" + std::to_string(K);
                return;
            }
            const auto h0 = std::chrono::steady_clock::now();
            had_buf_.assign(A, A + (size_t)M * (size_t)K);
            for (int m = 0; m < M; m++) {
                float* row = had_buf_.data() + (size_t)m * (size_t)K;
                q4nx_hadamard::rotate(hs->second.data(), row, K, had_block_);
                float am = 0.0f;
                for (int k = 0; k < K; k++) {
                    const float a = std::fabs(row[(size_t)k]);
                    if (a > am) am = a;
                }
                ascales[(size_t)m] = am > 0.0f ? am / 127.0f : 1.0f;
            }
            Ab = had_buf_.data();
            had_ms_ += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - h0).count();
            had_n_ += M;
        }
        const auto tl = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        xrt::run r = kctx.launch_async_with_bo_rows(*slot->bo, Ab, M, K, ascales.data());
        const auto tq = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        kctx.wait_kernel(r);
        const auto tw = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        kctx.readback();
        const auto tr = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        {
            // `out` is already M x N (sized before the refusal paths above) and every element is written here, so it
            // is not zero-filled a second time. Timed on its own: this runs after `tr`, so no launch phase column
            // ever counted it, and the batched [lh] table had it under UNATTRIBUTED (engine#1990).
            q4nx_lh::LhScope _t_deq(LH_DEQ_OUT);
            kctx.dequant_pb16i(out.data(), M, N, ascales.data(), slot->eref, kctx.pb16i_planes);
        }
        // The batched (prefill) twin of the trace above: a short prompt runs step_batch, so this
        // is the path a 1-token request actually takes.
        if (const char* tl = getenv("NPU_PB16I_TRACE")) {
            if (ar.layer == atoi(tl) && sc) {
                fprintf(stderr, "[pb16i-trace] %s M=%d as[0]=%.9g planes=%d A[0..7]=", sc, M,
                        (double)ascales[0], kctx.pb16i_planes);
                for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", (int)kctx.Am[i]);
                fprintf(stderr, " Cm[0..7]=");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", (int)kctx.Cm[i]);
                fprintf(stderr, " out[0..7]=");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.9g ", (double)out[(size_t)i]);
                fprintf(stderr, " eref[0..3]=%d %d %d %d\n", (int)slot->eref[0], (int)slot->eref[1],
                        (int)slot->eref[2], (int)slot->eref[3]);
            }
        }
        if (prof) {
            const double dpk = std::chrono::duration<double, std::milli>(tl - tp).count();
            const double dql = std::chrono::duration<double, std::milli>(tq - tl).count();
            const double dwt = std::chrono::duration<double, std::milli>(tw - tq).count();
            const double drb = std::chrono::duration<double, std::milli>(tr - tw).count();
            const int di = design_index(&kctx);
            if (M == 1) {
                prof_add(prof_dec_, di, dpk, dql, dwt, drb);
                prof_dec_n_++;
            } else {
                prof_add(prof_pre_, di, dpk, dql, dwt, drb);
                prof_pre_n_++;
            }
            // Dump on either family's count. The `prof_dec_n_ &&` guard meant a PREFILL-ONLY process never dumped at
            // all: no decode launch had happened, so the condition was false no matter how many prefill launches ran.
            // That is exactly the case that matters when a request stalls -- the run that never returns is a prefill,
            // so the profiler stayed silent and "no rows" could not be read as "no launches" (engine #1605: an 8-token
            // request on the pb16i arm returned nothing in 420 s and printed no profiler rows).
            if ((prof_dec_n_ + prof_pre_n_) % 100 == 0) prof_dump();
        }
        return;
    }
    PackedWeight*             slot = nullptr;
    const std::vector<float>* gs = nullptr;
    if (weight_cache_ && wkey >= 0 && ar.layer >= 0 && ar.layer < n_layers_ && dev_) {
        if (int(wcache_.size()) <= ar.layer) wcache_.resize(size_t(ar.layer) + 1);
        auto& m = wcache_[size_t(ar.layer)];
        auto  it = m.find(wkey);
        if (it == m.end()) {
            q4nx_lh::LhScope _tlh_pk(LH_PACK_UPLOAD);
            PackedWeight pw;
            pw.bo = kctx.make_weight_bo(*dev_);
            float sout = 1.0f;
            const auto w0 = std::chrono::steady_clock::now();       // engine#1416: the 22 ms candidate
            kctx.packB_into(*pw.bo, Bmat, K, N, sout, pw.scales);
            const auto w1 = std::chrono::steady_clock::now();
            wmiss_n_++;
            wmiss_ms_ += std::chrono::duration<double, std::milli>(w1 - w0).count();
            if (std::getenv("NPU_PACK_SPLIT") && wmiss_n_ % 20 == 0)
                fprintf(stderr, "[wm] n=%ld cache-miss repacks, avg packB_into=%.3f ms\n",
                        wmiss_n_, wmiss_ms_ / wmiss_n_);
            it = m.emplace(wkey, std::move(pw)).first;
        }
        slot = &it->second;
        gs = &slot->scales;
    } else {
        float sout = 1.0f;
        const auto w0 = std::chrono::steady_clock::now();           // engine#1416: uncached branch
        kctx.packB(0, Bmat, K, N, sout);
        const auto w1 = std::chrono::steady_clock::now();
        wunc_n_++;
        wunc_ms_ += std::chrono::duration<double, std::milli>(w1 - w0).count();
        if (std::getenv("NPU_PACK_SPLIT") && wunc_n_ % 20 == 0)
            fprintf(stderr, "[wm] n=%ld uncached packs, avg packB=%.3f ms\n",
                    wunc_n_, wunc_ms_ / wunc_n_);
        gs = &kctx.group_scales[0];
    }
    const auto tl = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto issue = [&]() -> xrt::run {
        return slot ? kctx.launch_async_with_bo_rows(*slot->bo, A, M, K, ascales.data())
                    : kctx.launch_async_rows(0, A, M, K, ascales.data());
    };
    xrt::run r = issue();
    const auto tq = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    kctx.wait_kernel(r);
    const auto tw = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    kctx.readback();
    if (kctx.canary_short()) {   // engine#666: re-issue once rather than dequantise an unwritten row
        kctx.canary_retries_++;
        r = issue();
        kctx.wait_kernel(r);
        kctx.readback();
        if (kctx.canary_short())
            fprintf(stderr, "[canary] SHORT RESULT (rows) twice: M=%d K=%d N=%d layer=%d short=%d\n",
                    M, K, N, ar.layer, kctx.short_row_);
    }
    const auto tr = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    out.resize((size_t)M * N);
    // engine#1808: the device's per-column weight scales, KEYED by (layer, N, K) so it can be paired with the host
    // arm's dump exactly. Index pairing was tried first and is meaningless -- the two sides reach the GEMM a
    // different number of times (384 device dumps against 1024 arm dumps on the same request).
    if (const char* gdir = getenv("NPU_I8_GS_DUMP")) {
        static int n = 0;
        char path[4096];
        std::snprintf(path, sizeof path, "%s/dev_L%d_N%d_K%d_%03d.bin", gdir, ar.layer, N, K, n++);
        if (std::FILE* f = onebit_fopen_private_wb(path)) {
            if (gs && !gs->empty()) std::fwrite(gs->data(), sizeof(float), (size_t)N, f);
            std::fclose(f);
        }
    }
    for (int m = 0; m < M; m++) {
        const int32_t* cm = kctx.Cm + (size_t)m * kctx.ND;
        const float asc = ascales[(size_t)m];
        float* o = out.data() + (size_t)m * N;
        for (int n = 0; n < N; n++)
            o[n] = (float)cm[n] * asc * (*gs)[(size_t)n];
    }
    if (m16_check_ && &kctx != &ctx)
        m16_check_gemm_rows(ctx, kctx, A, M, K, N, slot ? slot->bo.get() : nullptr,
                            ascales);
    if (prof) {
        const double dpk = std::chrono::duration<double, std::milli>(tl - tp).count();
        const double dql = std::chrono::duration<double, std::milli>(tq - tl).count();
        const double dwt = std::chrono::duration<double, std::milli>(tw - tq).count();
        const double drb = std::chrono::duration<double, std::milli>(tr - tw).count();
        const int di = design_index(&kctx);
        if (M == 1) {
            prof_add(prof_dec_, di, dpk, dql, dwt, drb);
            prof_dec_n_++;
        } else {
            prof_add(prof_pre_, di, dpk, dql, dwt, drb);
            prof_pre_n_++;
        }
        if ((prof_dec_n_ + prof_pre_n_) % 100 == 0) prof_dump();   // same reason as above: prefill-only runs
    }
}

}  // namespace npu::target::xdna
