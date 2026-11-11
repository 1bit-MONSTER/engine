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
// npu/target/xdna/xdna_target.h — the XDNA2 (AIE2P) target (engine#799 P4 / #803).
//
// Everything the engine does to drive this NPU lives here: the I8Ctx ELF contexts (M=128 prefill and the
// M=16 decode twin), the phase-selected launch, the packed-weight BO cache, the pb16i sidecar image path,
// the folded-basis (Hadamard) activation rotation that only the pb16i kernels need, the M=16 bit-equality
// gate, and the NPU_PROFILE accumulators.
//
// The model maths does NOT live here and never will: this class is handed activations and returns an
// output row, exactly like npu/target/target.h says.  The byte-identical move is phase A1 of
// docs/npu-p4-target-split.md; the plan-driven setup (open/load_block) is A2.
#ifndef ONEBIT_NPU_TARGET_XDNA_TARGET_H
#define ONEBIT_NPU_TARGET_XDNA_TARGET_H

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "target/target.h"
#include "plan/plan.h"   // onebit::npu::plan::Plan (engine #799 P1)
#include "q4nx_pack.h"       // Q4nxDesignGeom, Q4NX_MAX_DESIGNS
#include "rt.h"          // xrt::device, xrt::bo

struct I8Ctx;

namespace npu::target::xdna {

// A weight image resident on the device.  One per (layer, wkey): packed on first use, then reused, which
// is what makes a served answer possible at all (re-packing per token cost minutes per token).
struct PackedWeight {
    std::unique_ptr<xrt::bo> bo;
    std::vector<float>       scales;
    // pb16i: the sidecar's host side table.  Empty eref = an int8 pack.
    std::vector<int16_t>     eref;
    int                      pb16i_mode = 0;
};

// Per-role dispatch accounting for NPU_PROFILE=1.
struct ProfAcc {
    double pack = 0, quant = 0, wait = 0, rb = 0;
    long   n = 0;
};

class XdnaTarget final : public NpuTarget {
public:
    // The device is the XDNA target's business, not the interface's: passing it here is what keeps
    // NpuTarget::open(plan, kernels_dir, family, err) exactly as documented for a second target
    // (engine#803 A2b).  Null is legal and means "host GEMMs only" (init_host()).
    explicit XdnaTarget(xrt::device* dev = nullptr);
    ~XdnaTarget() override;
    XdnaTarget(const XdnaTarget&) = delete;
    XdnaTarget& operator=(const XdnaTarget&) = delete;

    // ---- NpuTarget ----
    // The name this target answers to, as a constant so the CLI can check a requested name against the
    // compiled set without constructing a target or opening a device (app/forward_serve.cpp).
    static constexpr const char* kName = "xdna";
    const char* name() const override { return kName; }
    // Consumes the plan (P1).  What A1 can check without owning the derivation: that the geometry handed
    // to this target is the geometry the plan describes (layer count, prefill M, decode M, kernel family),
    // and it refuses by name when they disagree.  A2 moves the derivation itself in here, at which point
    // this stops being a check and becomes the source.
    bool open(const onebit::npu::plan::Plan& plan, const std::string& kernels_dir,
              const char* family, std::string& err) override;
    bool load_block(int first_layer, int n_layers, std::string& err) override;
    bool gemv(const GemvArgs& a, std::vector<float>& out, std::string& err) override;
    bool supports(Op op) const override { return op == Op::gemv; }
    Caps caps() const override;

    // ---- setup, called by the forward while the derivation still lives there (A1) ----
    void set_device(xrt::device* dev, bool weight_layout_b8, bool weight_cache, int n_layers);
    void set_pb16i(bool on, std::string dir, std::string sealed);
    void set_hadamard_block(int block) { had_block_ = block; }
    bool load_hadamard(const std::string& path, std::string& err);
    long launches_decode() const { return prof_dec_n_; }
    long tokens_seen() const { return prof_tok_n_; }
    void note_head(int M, double dpk, double dql, double dwt, double drb);
    // The lm head's own images are folded the same way, and its launches are packed here too, so it needs
    // the same three bits of state until A3 moves the head behind NpuTarget::lm_head().
    bool weight_cache_on() const { return weight_cache_; }
    const std::vector<float>* had_signs_for(int width) const {
        auto it = had_signs_.find(width);
        return it == had_signs_.end() ? nullptr : &it->second;
    }
    void add_hadamard_cost(double ms, long n) { had_ms_ += ms; had_n_ += n; }
    void set_m16_check(bool on) { m16_check_ = on; }
    void set_profile(bool on) { prof_ = on; }
    bool profiling() const { return prof_; }
    void set_role_split(int qhalf, int kv) { i_qhalf_ = qhalf; i_kv_ = kv; }
    // The two full-attention-QKV-split roles exist only as pb16i kernels: they get no M=128 context.
    bool role_is_pb16i_split(int role) const { return role == i_qhalf_ || role == i_kv_; }
    int  role_split_qhalf() const { return i_qhalf_; }
    int  role_split_kv() const { return i_kv_; }
    bool pb16i() const { return pb16i_; }
    bool m16() const { return m16_; }
    bool weight_cached(int layer, int key) const;
    int  max_rows() const;               // widest launch this target will accept (decode twins included)
    int  rows_for(int role, int want) const;   // rows the phase-selected kernel for `role` can take

    // Everything the setup side reads, set once before open_*().  It exists because the remaining two
    // blocks (the M=16 decode family and the lm-head context) read a dozen of the forward's fields --
    // the plan, its path, the two given directories, the layout family, the sealed ELF, the t2s flag --
    // and a twelve-argument call would be a signature nobody can review.  Once these live here, moving
    // those blocks changes no call site (engine#803 A2b).
    struct Setup {
        std::string model_dir;      // the -m directory
        std::string kernels_dir;    // the --kernels directory
        std::string family;         // "", "_b8", "_pb16i", "_t2s", "_t2s2"
        std::string plan_path;      // the plan this set came from, when one did ("" = derived)
        bool b8 = false;            // blocked 8x8 tile layout
        bool pb16i = false;         // integer-exact ternary, sidecar images
        bool t2s = false;           // 2-bit ternary stream
        bool t2s_v2 = false;
        bool host_gemm = false;     // offline: open no ELF at all
        // The lm-head context needs the container's own naming, which is the forward's to compute and not
        // the target's: the sidecar directory, whether the head's ternary image exists at all (the test is
        // `head_pb16i_stem(0) + image_ext()` on the forward's side), and where decode twins are looked for.
        std::string pb16i_dir;
        bool head_pb16i_images = false;
        std::vector<std::string> decode_dirs;
        // The head geometry: hidden width, vocabulary, and the container's padded K.
        int hidden = 0, vocab = 0, kpad = 0;
        const onebit::npu::plan::Plan* plan = nullptr;   // copied into plan_ by configure(); null = none
    };
    // Setup carries a pointer to the forward's Plan; this copies it, so from here on the target's own
    // plan_ is THE copy the setup side reads (open() no longer has to be the one that filled it).  Two
    // live copies of the plan is the drift the plan layer exists to prevent, so there is one.
    void configure(const Setup& s) {
        setup_ = s;
        if (s.plan) plan_ = *s.plan;
    }
    const Setup& setup() const { return setup_; }

    // The M=128 ELF set: for each design, adopt its geometry, probe its artifact and open it, refusing by
    // name when an artifact is missing or will not load (engine#803 A2b).  This is the block that used to
    // live at the top of Q4nxNpuForward::init_impl(); it is here because it is ELF loading and its
    // refusals, which is what a target owns.  `host_gemm` opens nothing (the offline path needs no ELF)
    // and only names the artifacts it would have used.  `b8` is the container's blocked-tile layout.
    // `plan_path` is the plan the set came from, when one did: the missing-artifact refusal quotes it, so
    // the message a user sees is the same one the forward produced before this moved (an empty string
    // means no plan, i.e. a purely derived set).
    bool open_prefill(const Q4nxDesignGeom* designs, int n, std::string& err);

    // The lm-head context (engine#803 A2b, the last setup block): pb16i ternary images first, then the int8
    // ladder, then the q4_1 gemv head -- the body moved verbatim out of Q4nxNpuForward::open_lm_head_context.
    // It is a context and its refusals, which is what a target owns; the container's *naming* stayed in the
    // forward and arrives through Setup, because deciding which sidecar file the head has is the container's
    // rule and not a target's.
    bool open_head(const Q4nxDesignGeom* designs, int n, std::string& err);
    // engine#803 A2b: the M=16 decode family. `want_m16` is the NPU_DECODE_M16 guard, `dir` the kernels directory the
    // pb16i sealed elf and the <kernels>/m16 candidate hang off, and `decode_dirs` the container's candidate list.
    bool open_decode_family(const Q4nxDesignGeom* designs, int n, Q4nxDesignGeom* designs16, bool want_m16,
                            const std::string& dir, const std::vector<std::string>& decode_dirs, std::string& err);
    // Whether open_head() found and opened the pb16i ternary head.  The model path reads this flag in four
    // places, so it stays the forward's; the target writes it.
    bool head_is_pb16i() const { return head_pb16i_; }

    // Context construction: the caller configures and initialises the I8Ctx, then it stays here.
    I8Ctx* make_ctx(int role);
    I8Ctx* ctx(int role) const;
    std::unique_ptr<I8Ctx>& ctx_ref(int role);
    I8Ctx* make_ctx16(int role, const Q4nxDesignGeom& geom);
    I8Ctx* ctx16(int role) const;
    I8Ctx* make_head_ctx();
    I8Ctx* head_ctx() const;
    std::unique_ptr<I8Ctx>& head_ctx_ref();
    void   note_m16_made(bool any) { m16_ = m16_ || any; }
    void   drop_ctx16(int role);
    // The context a launch of `rows` rows on `role` would use: the M=16 twin when one was adopted and
    // the row count fits, otherwise the M=128 one. The forward needs this where it launches a GEMM
    // itself (the LM head) instead of going through gemv().
    I8Ctx* pick_ctx(int role, int rows, bool decode_twin = false) { return ctx_for(ctx(role), rows, decode_twin); }

    // Profiling / gates, driven from the forward's per-token loop.
    void note_token() { prof_tok_n_++; }
    void note_head_launch(int M) { prof_head_M_ = M; }
    void dump_profile();

private:
    I8Ctx* ctx_for(I8Ctx* prefill, int rows, bool decode_twin);
    int    design_index(I8Ctx* c) const;
    void   prof_add(ProfAcc* acc, int design, double pack, double quant, double wait, double rb);
    void   prof_dump();
    void   npu_gemm(const GemvArgs& a, std::vector<float>& out, std::string& err);
    void   npu_gemm_rows(const GemvArgs& a, std::vector<float>& out, std::string& err);
    // int64, not int32: the comparison subtracts two int32 accumulators, and this check exists to catch a
    // *corrupt* one -- whose value is by definition not bounded by the arithmetic.  At int32 width the
    // subtraction (and the negation of its INT32_MIN) is undefined behaviour in exactly the case the check is
    // for; at int64 it is defined and the report is right.  Nothing about a real launch moves: every
    // non-overflow difference is the same number at either width.
    int64_t m16_check_gemm(I8Ctx& ref, I8Ctx& fast, const float* A, int K, int N,
                           const float* Bmat, xrt::bo* wbo, float as);
    int64_t m16_check_gemm_rows(I8Ctx& ref, I8Ctx& fast, const float* A, int M, int K, int N,
                                xrt::bo* wbo, const std::vector<float>& ascales);
    std::string pb16i_stem(int layer, const char* tag) const;

    xrt::device* dev_ = nullptr;
    bool head_pb16i_ = false;   // written by open_head(), read back by the forward
    Setup setup_;   // see configure(): what the setup side reads, in one place
    int          n_layers_ = 0;
    std::string  kernels_dir_;
    onebit::npu::plan::Plan plan_;   // when loaded: what open() checked this target against
    bool         weight_layout_b8_ = false;
    bool         weight_cache_ = true;
    Q4nxDesignGeom designs_[Q4NX_MAX_DESIGNS];   // geometry, filled by adopt() from the forward's table
    int            ndesigns_ = 0;
    std::unique_ptr<I8Ctx> ctx_[Q4NX_MAX_DESIGNS];
    Q4nxDesignGeom designs16_[Q4NX_MAX_DESIGNS];
    std::unique_ptr<I8Ctx> ctx16_[Q4NX_MAX_DESIGNS];
    std::unique_ptr<I8Ctx> ctx_head_;
    bool m16_ = false;
    bool m16_check_ = false;
    bool pb16i_ = false;
    std::string pb16i_dir_, pb16i_sealed_;
    int  i_qhalf_ = -1, i_kv_ = -1;
    bool hadamard_ = false;
    int  had_block_ = 1024;
    std::vector<float> had_buf_;
    double had_ms_ = 0.0;
    long   had_n_ = 0;
    std::unordered_map<int, std::vector<float>> had_signs_;   // input width -> signs
    int64_t m16_worst_ = 0;
    long    m16_bad_ = 0, m16_checked_ = 0;
    // engine#1416 [wm]: weight-side cost in the same tp->tl window as the activation pack
    long    wmiss_n_ = 0, wunc_n_ = 0;
    double  wmiss_ms_ = 0, wunc_ms_ = 0;
    std::vector<int32_t> m16_ref_;
    bool   prof_ = false;
    ProfAcc prof_dec_[Q4NX_MAX_DESIGNS], prof_pre_[Q4NX_MAX_DESIGNS];
    ProfAcc prof_head_;
    int     prof_head_M_ = 0;
    long    prof_dec_n_ = 0, prof_pre_n_ = 0, prof_tok_n_ = 0;
    std::vector<std::unordered_map<int, PackedWeight>> wcache_;   // [layer][wkey]

public:
    // The forward still owns the design table (A2 moves it with the plan); this hands the geometry in.
    void adopt_geometry(int role, const Q4nxDesignGeom& g) {
        if (role >= 0 && role < Q4NX_MAX_DESIGNS) { designs_[role] = g; if (role + 1 > ndesigns_) ndesigns_ = role + 1; }
    }
    int ndesigns() const { return ndesigns_; }
    const Q4nxDesignGeom& geometry(int role) const { return designs_[role]; }
    const Q4nxDesignGeom& geometry16(int role) const { return designs16_[role]; }
};

}  // namespace npu::target::xdna

#endif  // ONEBIT_NPU_TARGET_XDNA_TARGET_H
