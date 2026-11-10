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
// Copyright (c) 2026 bong-water-water-bong
// q4nx_forward.cpp — see q4nx_forward.h.
//
// The host does RMSNorm / RoPE / GQA attention / SiLU in float32; the GEMMs run
// on the NPU through I8Ctx.  The design set is DERIVED from the model's own
// architecture dims (q4nx_derive_designs): four kernels when gate/up are fused,
// five when they are split.  No model name or tag appears in this file; the
// full-ELF names are geometry-derived.
//
// Numerics follow tools/q4nx_forward_ref.py so the two can be compared
// element-wise.
#include "q4nx_forward.h"
#include "q4nx_state_planes.h"   // the plane-copy half of a snapshot, device-free and unit-tested
#include "model/op_dispatch.h"   // the optional-op fallback seam (engine#803 A3)
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>   // mkstemp/close: the pb16i activation probe's dump

#include <chrono>
#include <cmath>
#include <cstdio>
#include "batch_check_compare.h"
#include "ops-dump.h"
#include "residual-dump.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "rt.h"  // the own NPU runtime (ONEBIT_NPU_RT_DIR)

#include "npu_engine_i8ctx_inc.h"   // I8Ctx
#include "hadamard.h"                // the activation rotation, SIMD path + scalar reference
#include "attn_math.h"               // attention's score dot + value accumulation, SIMD + portable
#include "q4nx_dequant.h"
#include "q4nx_device.h"  // an imported q4nx-2 container's device form (engine#1185)
#include "q4nx_semantics.h"
#include "q4nx_pack.h"
#include "q4nx_head_image.h"   // the lm-head tile cache (engine#1716)

#if defined(_OPENMP)
#include <omp.h>
#endif
// How many threads one head-parallel loop in the GDN layer may take: three quarters of the cores, because the
// forward's own array-submit thread and the serve's HTTP thread need one each.  OMP_NUM_THREADS below that wins.
static inline int omp_heads_threads() {
#if defined(_OPENMP)
    const int cap = (3 * omp_get_num_procs()) / 4;
    const int want = omp_get_max_threads();
    return want < cap ? (want > 0 ? want : 1) : (cap > 0 ? cap : 1);
#else
    return 1;
#endif
}

#ifdef NPU_INFER_DEBUG_DUMPS
// Debug dumps to a path named by an environment variable (NPU_INFER_DUMP_ATTN, _LN/_L0,
// _HEAD): compiled only into debug builds (cmake -DONEBIT_NPU_DEBUG_DUMPS=ON), written
// owner-only (0600) and never through a symlink.
static FILE* open_debug_dump(const char* path) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    return fd >= 0 ? fdopen(fd, "wb") : nullptr;
}
#endif

// The pb16i activation probe (NPU_PB16I_DUMP=<layer>) is runtime-gated, not debug-build gated,
// so it makes its own file: mkstemp() gives O_CREAT|O_EXCL and mode 0600 with a unique name in
// one step.  A fixed name in the shared /tmp cannot promise either (fopen creates it 0666 via
// umask and follows a symlink), which is the file-creation-without-restricted-permissions
// alert CodeQL raised on this probe.
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


static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// softplus with the same overflow guard the Python reference uses.
static inline float softplus_f(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// A tensor's data as n bf16 values, or null when the tensor holds fewer than n
// (config.json's dims decide n; the file decides the size).
static const uint16_t* bf16_data(ModelWeights* mw, const TensorDesc* d, size_t n) {
    if (!d || (size_t)d->data_size / 2 < n) return nullptr;
    return (const uint16_t*)model_tensor_data(mw, const_cast<TensorDesc*>(d));
}

// A tensor's data when it holds at least `bytes`, else null.
static const uint8_t* data_of(ModelWeights* mw, const TensorDesc* d, size_t bytes) {
    if (!d || (size_t)d->data_size < bytes) return nullptr;
    return (const uint8_t*)model_tensor_data(mw, const_cast<TensorDesc*>(d));
}

// Which instance of the sealed ELF runs a role, or nullptr when the sealed ELF does not cover it.  The table
// moved to q4nx_pack.c (q4nx_pb16i_instance) because the target opens the decode family (engine#803 A2b) and a
// contract both sides must agree on cannot live on one side; this wrapper keeps the call sites unchanged.
static const char* pb16i_instance(Q4nxDesign r) { return q4nx_pb16i_instance(r); }

// Where an M=16 int8 twin is looked for, first hit wins: <model dir>/kernels-m16,
// <kernels>/m16, then <kernels> itself.
static std::vector<std::string> m16dirs_like_head(const std::string& mdir,
                                                  const std::string& dir,
                                                  const std::vector<std::string>& plan_dirs = {}) {
    // A plan that names its decode directories wins, the same rule the projection lookup uses: with
    // decode ELFs under e.g. <model>/kernels-m16-seq the projections load from there, and a head lookup
    // restricted to the built-in guesses cannot find its ELF beside them (engine#910 review).
    if (!plan_dirs.empty()) return plan_dirs;
    return {mdir + "/kernels-m16", dir + "/m16", dir};
}

// A <- H (A . s): the inverse of the fold a folded (PTQ1/Prism) container baked into its weights.
// The transform itself lives in hadamard.h, because it runs once per projection (~430 times a decode token)
// and its SIMD path has to be provable bit-identical to the scalar loop on every build, without a device.
static void hadamard_rotate(const float* signs, float* a, int n, int block) {
    q4nx_hadamard::rotate(signs, a, n, block);
}

static bool file_exists(const std::string& p, long min_size = 1) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz >= min_size;
}

// A sidecar directory comes from the container's config or from an environment variable, and every
// weight image is opened under it, so it is checked once here instead of trusted at each fopen: an
// absolute path with no ".." component and no NUL, which is what a converter writes (a relative one
// would silently resolve against the serve's cwd).
static bool valid_sidecar_dir(const std::string& d) {
    if (d.empty() || d.size() > 4096 || d[0] != '/') return false;
    if (d.find('\0') != std::string::npos) return false;
    for (size_t i = 0; i + 1 < d.size(); i++)
        if (d[i] == '.' && d[i + 1] == '.' && (i == 0 || d[i - 1] == '/') &&
            (i + 2 == d.size() || d[i + 2] == '/'))
            return false;
    return true;
}

// Decode a quantized [rows, cols] tensor whatever chunk size it is stored in:
// 5120 = q4_1, 4736 = OFLM Q4_K (flm_dtype_t 8, Qwen3.5-4B's projections),
// 8704 = Q8 (Qwen3.5's tied table / ssm_alpha / ssm_beta).  The chunk size is
// derived from data_size rather than the declared shape so a 2-D [ntiles, N]
// and a 3-D [ntr, ntc, N] tensor both work.  (src/q4nx_dequant.c; the Q4_K/Q8
// layouts are in docs/qwen35-4b-weight-format.md.)

static int dequant_any(const void* data, size_t nbytes, int rows, int cols, float* out) {
    if (!data || !out || rows <= 0 || cols <= 0) return -1;
    if (rows % Q4NX_TILE_ROWS || cols % Q4NX_TILE_COLS) return -1;
    const size_t n = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS);
    if (n == 0) return -1;
    const int cb = (int)(nbytes / n);
    if (cb == Q4NX_TILE_BYTES || cb == Q4NX_CHUNK_Q4K || cb == Q4NX_CHUNK_Q8)
        return q4nx_dequant_tensor_chunked((const uint8_t*)data, nbytes, rows, cols, cb, out);
    // The container may store this weight on a wider grid than the model's dimensions (engine#1097).
    // Read it on the container's grid and write only the model's [rows, cols], so buffers, row
    // strides and the kernel's B layout stay exactly what they were for an unpadded container.
    int pcb = 0, prp = 0, pcp = 0;
    if (q4nx_packed_grid(nbytes, rows, cols, &pcb, &prp, &pcp))
        return q4nx_dequant_tensor_packed((const uint8_t*)data, nbytes, rows, cols, prp, pcp, pcb, out);
    return q4nx_dequant_tensor((const uint8_t*)data, nbytes, rows, cols, out);
}

// rope_type "llama3" (Llama-3.1/3.2/3.3) from config.json's rope_scaling: the frequency band that
// transformers' _compute_llama3_parameters applies.  `on == 0` unless the config names the type, so
// every other architecture builds exactly the table it built before.  Keys are searched with their
// quotes, which is what keeps "factor" from matching inside "high_freq_factor".
// Dynamic NTK: `rope_scaling: {type: "dynamic"|"dynamic_ntk", factor: F}` and `max_position_embeddings`. Hunyuan's
// 1.9 packs declare factor 1.0 with a 262,144 window, i.e. a no-op until a prompt passes that window -- which is why
// accepting it is safe rather than a guess: below the window the base IS theta, so the served bytes are plain rope.
Q4nxRopeDynamic read_rope_dynamic(const std::string& cfg_path) {
    Q4nxRopeDynamic d{};
    std::ifstream f(cfg_path);
    if (!f) return d;
    std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    char type[64];
    q4nx_rope_declared_type(t.c_str(), type, (int)sizeof type);
    if (std::strcmp(type, "dynamic") && std::strcmp(type, "dynamic_ntk")) return d;
    auto number = [&](const char* key) {
        const std::string k = std::string("\"") + key + "\":";
        const size_t at = t.find(k);
        if (at == std::string::npos) return 0.0;
        return std::strtod(t.c_str() + at + k.size(), nullptr);
    };
    d.factor = number("factor");
    d.original_max_position = number("max_position_embeddings");
    d.on = (d.factor > 0.0 && d.original_max_position > 0.0) ? 1 : 0;
    return d;
}

static Q4nxRopeLlama3 read_rope_llama3(const std::string& cfg_path) {
    Q4nxRopeLlama3 s{};
    std::ifstream f(cfg_path);
    if (!f) return s;
    std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!strstr(t.c_str(), "llama3")) return s;
    auto num = [&t](const char* key, double def) -> double {
        const char* p = strstr(t.c_str(), key);
        if (!p) return def;
        const char* c = strchr(p + strlen(key), ':');
        if (!c) return def;
        char* end = nullptr;
        const double v = strtod(c + 1, &end);
        return end == c + 1 ? def : v;
    };
    s.factor = num("\"factor\"", 0.0);
    s.high_freq_factor = num("\"high_freq_factor\"", 0.0);
    s.low_freq_factor = num("\"low_freq_factor\"", 0.0);
    s.original_max_position = num("\"original_max_position_embeddings\"", 0.0);
    s.on = s.factor > 0.0 && s.original_max_position > 0.0 && s.high_freq_factor > s.low_freq_factor;
    return s;
}

// One top-level string field out of a small sidecar JSON (decision_config.json's `attention_mode`).
// Empty when the file, the key or the value is absent -- the same "absent means not declared" reading
// tools/npu_plan.py uses, so the two cannot disagree about what a missing field means.
static std::string read_json_string(const std::string& path, const char* key) {
    std::ifstream f(path);
    if (!f) return std::string();
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string needle = std::string("\"") + key + "\"";   // the key, not a mention of it in a prompt string
    const char* p = strstr(s.c_str(), needle.c_str());
    if (!p) return std::string();
    const char* c = strchr(p + needle.size(), ':');
    if (!c) return std::string();
    const char* b = strchr(c, '"');
    if (!b) return std::string();
    const char* e = strchr(b + 1, '"');
    if (!e) return std::string();
    return std::string(b + 1, e);
}

// LongRoPE short_factor from config.json's rope_scaling (MiniCPM4): the
// per-half-dim frequency multipliers.  Empty when the config declares none.
// Ported from the engine route's npu/forward (commit 599ac77) so both copies of
// the model-generic forward agree.
static std::vector<float> read_rope_short_factor(const std::string& cfg_path) {
    std::vector<float> out;
    std::ifstream f(cfg_path);
    if (!f) return out;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = strstr(s.c_str(), "short_factor");
    if (!p) return out;
    const char* b = strchr(p, '[');
    if (!b) return out;
    const char* e = strchr(b, ']');
    if (!e) return out;
    const char* q = b + 1;
    while (q < e) {
        char* end = nullptr;
        const float v = strtof(q, &end);
        if (end == q) break;
        out.push_back(v);
        q = end;
    }
    return out;
}

// A scalar from config.json (MiniCPM4's embedding_scale / residual_scale).
static float read_config_float(const std::string& cfg_path, const char* key, float dflt) {
    std::ifstream f(cfg_path);
    if (!f) return dflt;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = strstr(s.c_str(), key);
    if (!p) return dflt;
    const char* c = strchr(p, ':');
    if (!c) return dflt;
    return (float)strtod(c + 1, nullptr);
}

// The device-side weight layout a container declares (`q4nx_weight_layout`).
// Absent or "rm" is the layout every container built before today has: packB
// writes a padded row-major [KD][ND] int8 image, and the ELF's B tap is the
// 4-level [<k/8, 8N>, <n/8, 8>, <8, N>, <8, 1>] scatter that lands 64 runs of
// 128 contiguous bytes per 8 KiB tile.  "b8" is the same bytes pre-blocked into
// the microkernel's own [kb][nb][8][8] tile order, tile-n-major first, so each
// tile is ONE contiguous 8 KiB burst (measured 3.32 -> 8.13 GB/s on the 27B GU
// phase, and 2.4-2.6x on every other phase).  It is a layout, not a different
// result: the quantizer, the scales and the int8 codes are identical, so the
// int32 output is bit-identical.
static std::string read_config_string(const std::string& cfg_path, const char* key,
                                      const char* dflt) {
    std::ifstream f(cfg_path);
    if (!f) return dflt;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = strstr(s.c_str(), key);
    if (!p) return dflt;
    const char* c = strchr(p, ':');
    if (!c) return dflt;
    c++;
    while (*c == ' ' || *c == '\t') c++;
    if (*c != '"') return dflt;
    const char* q = c + 1;
    const char* e = strchr(q, '"');
    if (!e) return dflt;
    return std::string(q, e);
}

// The named phase scopes live in q4nx_forward.h (namespace q4nx_lh) so the HTTP
// handler can time request-level phases with the same machinery; keep every use
// here unqualified.
using namespace q4nx_lh;

Q4nxNpuForward::Q4nxNpuForward() {}
Q4nxNpuForward::~Q4nxNpuForward() = default;

size_t Q4nxNpuForward::StateSnapshot::bytes() const {
    size_t n = 0;
    for (const auto* per_layer : {&k, &v, &conv, &rec})
        for (const auto& layer : *per_layer) n += layer.size() * sizeof(float);
    return n;
}

bool Q4nxNpuForward::snapshot_state(int tokens, StateSnapshot& out) const {
    if (!ready_ || tokens <= 0) return false;
    const size_t per = size_t(c_.NKV) * size_t(c_.HD);
    size_t held = 0;  // positions the KV caches hold (a layer that shares its KV holds none)
    for (const auto& layer : kcache_) held = std::max(held, layer.size() / per);
    if (size_t(tokens) > held) return false;
    out.tokens = tokens;
    out.per_pos = per;
    const size_t want = size_t(tokens) * per;
    // KV is per position, so a snapshot of the first `tokens` positions keeps exactly that many; the GDN conv
    // window and recurrent state are fixed-size blocks updated per position, so they go in whole.  Cutting them
    // with `want` truncated them (23 % of the recurrent plane at 180 tokens) and made restore_state refuse the
    // snapshot -- the prefix cache's `slot has no usable state`, which cost a full re-prefill on every request.
    q4nx_state::take_per_pos(kcache_, out.k, want);
    q4nx_state::take_per_pos(vcache_, out.v, want);
    q4nx_state::take_whole(conv_state_, out.conv);
    q4nx_state::take_whole(rec_state_, out.rec);
    return true;
}

bool Q4nxNpuForward::restore_state(const StateSnapshot& in, int tokens) {
    if (!ready_ || tokens <= 0 || tokens > in.tokens) return false;
    const size_t per = size_t(c_.NKV) * size_t(c_.HD);
    if (in.per_pos != per) return false;
    const size_t want = size_t(tokens) * per;
    // Same split as snapshot_state: KV is restored per position (a layer whose KV is shared carries none of its
    // own, so an empty plane leaves the destination empty rather than filling it with the owner's), and the GDN
    // planes are restored whole -- refusing a truncated one instead of installing it short.
    if (!q4nx_state::give_per_pos(in.k, kcache_, want)) return false;
    if (!q4nx_state::give_per_pos(in.v, vcache_, want)) return false;
    if (!q4nx_state::give_whole(in.conv, conv_state_)) return false;
    if (!q4nx_state::give_whole(in.rec, rec_state_)) return false;
    return true;
}

void Q4nxNpuForward::reset() {
    LhScope _tlh_rst(LH_RQ_RESET);
    for (auto& v : kcache_) v.clear();
    for (auto& v : vcache_) v.clear();
    // A GDN layer carries a conv window and a recurrent state across positions.
    // Clearing only the KV caches left those advanced, so a second request on the
    // same process -- and the NPU_BATCH_CHECK oracle, which replays from 0 --
    // started from the previous stream's state.
    for (auto& v : conv_state_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : rec_state_) std::fill(v.begin(), v.end(), 0.0f);
}

bool Q4nxNpuForward::extend_context(int n_tokens) {
    if (!ready_ || n_tokens <= 0) return false;
    if (kcache_.empty() || vcache_.empty()) return false;
    const size_t per = (size_t)c_.NKV * (size_t)c_.HD;   // floats per layer per token
    for (int l = 0; l < c_.NL; l++) {
        kcache_[l].reserve(kcache_[l].size() + (size_t)n_tokens * per);
        vcache_[l].reserve(vcache_[l].size() + (size_t)n_tokens * per);
        kcache_[l].insert(kcache_[l].end(), (size_t)n_tokens * per, 0.0f);
        vcache_[l].insert(vcache_[l].end(), (size_t)n_tokens * per, 0.0f);
    }
    return true;
}

int Q4nxNpuForward::idx_of(Q4nxDesign role) const {
    for (int i = 0; i < ndesigns_; i++)
        if (designs_[i].role == role) return i;
    return -1;
}

// The log tag lives with the geometry it describes (q4nx_pack.c) so the target's own setup messages and
// this file's can never drift apart; it is the historical mapping, not the unique plan name.
static const char* role_tag(Q4nxDesign r) { return q4nx_design_log_tag(r); }

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------
bool Q4nxNpuForward::init(xrt::device& dev, const char* model_path,
                          const std::string& dir) {
    return init_impl(&dev, model_path, dir);
}

bool Q4nxNpuForward::init_host(const char* model_path, const std::string& dir) {
    host_gemm_ = true;
    return init_impl(nullptr, model_path, dir);
}

bool Q4nxNpuForward::init_impl(xrt::device* dev, const char* model_path,
                               const std::string& dir) {
    if (const char* hg = getenv("NPU_INFER_HOST_GEMM")) {
        host_gemm_ = true;
        // engine#1808: "i8" selects the integer arm, whose int32 accumulation is the only thing a
        // kernel family can be bit-exact against (an int8 device cannot match an f32 host by
        // construction). Any other value keeps the f32 path, which is what every existing caller and
        // every row measured so far used.
        host_gemm_i8_ = (hg[0] == 'i' && hg[1] == '8');
    }
    // The launch layer.  A1: everything below that touches a device talks to this, and the settings the
    // container declares (weight layout, pb16i image dir, Hadamard fold) are pushed into it as they are
    // parsed.  A2 turns this into NpuTarget::open(plan, kernels_dir).
    xd_ = std::make_unique<npu::target::xdna::XdnaTarget>(dev);
    int  had_block = 1024;
    bool weight_cache = true;
    if (getenv("NPU_NO_WCACHE")) weight_cache = false;   // A/B: disable the packed-weight cache
    if (const char* v = getenv("NPU_FFN_CACHE")) {
        ffn_cache_ = std::atoi(v) != 0 ? 1 : 0;   // engine#587: force the dense-tail cache
    }
    dev_ = dev;

    // ---- config.json FIRST, then model_load ----
    // model_load allocates `layers` from the ModelConfig it is handed, so the
    // model's real depth has to be known before loading.  (Passing the built-in
    // 0.6B config made model_load allocate 28 layers while the rest of init used
    // the model's declared depth, so load_norms indexed past the end of the
    // array - a general-protection fault for Qwen3-VL-4B, which declares 36.)
    std::string mdir, cfg_path;
    {
        std::string mp(model_path);
        size_t slash = mp.find_last_of('/');
        mdir = slash == std::string::npos ? std::string(".") : mp.substr(0, slash);
        cfg_path = mdir + "/config.json";
    }

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    // Device-side weight layout this container declares (see read_config_string).
    // "b8" makes the ELF set the _b8 family and packB write the blocked image;
    // "pb16i" makes the decode family read pb16i sidecar images (integer-exact
    // ternary weights, no packing at all); anything else keeps the layout every
    // earlier container has.
    {
        const std::string lay = read_config_string(cfg_path, "\"q4nx_weight_layout\"", "rm");
        weight_layout_b8_ = (lay == "b8");
        t2s_ = (lay == "t2s");
        pb16i_ = (lay == "pb16i") || t2s_;
        if (pb16i_) {
            // Where the images live: blk_<layer>_<tag>_weight.{pb16i|t2s,ref}.  The env
            // variable wins so a run can point at a fresh set without editing the
            // container's config.
            // The image format has a version so a future change cannot be read as this
            // one: the config may pin it, and anything but a version this build knows is
            // refused (4 = pb16i's 4,608-B tiles, 5 = t2s v1, 6 = t2s v2).
            const int ver = q4nx_config_int(cfg_path.c_str(), "q4nx_weight_layout_version", 4);
            if (t2s_) {
                // v1 and v2 carry the same weights in different byte orders, at the same
                // size, and reading one with the other's kernel is a silent permutation of
                // the matrix.  So the version picks the artifact family, not just a flag:
                // v1 kernels are `_t2s_m16`, v2 kernels are `_t2s2_m16`, and a container
                // can only ever name the one its images were packed for.
                if (ver != 5 && ver != 6) {
                    err_ = "q4nx_weight_layout_version " + std::to_string(ver) +
                           " is not a t2s stream this build reads (5 = v1, 6 = v2)";
                    return false;
                }
                t2s_v2_ = (ver == 6);
            } else if (ver != 4) {
                err_ = "q4nx_weight_layout_version " + std::to_string(ver) +
                       " is not an image format this build reads (4 is pb16i)";
                return false;
            }
            const char* e = getenv(t2s_ ? "Q4NX_T2S_DIR" : "Q4NX_PB16I_DIR");
            pb16i_dir_ = e ? e : read_config_string(cfg_path,
                                                    t2s_ ? "\"q4nx_t2s_dir\"" : "\"q4nx_pb16i_dir\"", "");
            // The sealed-ELF trick (one full ELF, every ternary instance, one hw_context) was pb16i's
            // first, and it is worth as much to a t2s container: the same measurement applies, because it
            // is the hw_context switch that costs and not the layout.  Measured 2026-11-02 in-loop: a t2s
            // decode family shipped as eight per-role ELFs paid **+2.0-2.3 ms on every launch** against
            // this engine's own documented ~2.7 ms between two hw_contexts (include/q4nx_forward.h), which
            // is more than the kernels' own device time and turned a 1.76x-faster kernel into a 1.8x
            // slower model.  Each layout names its own key so a container cannot hand a t2s stream to a
            // pb16i ELF (which would read 4,608-B tiles out of 2,176-B images).
            pb16i_sealed_ = read_config_string(
                cfg_path, t2s_ ? "\"q4nx_t2s_sealed_elf\"" : "\"q4nx_pb16i_sealed_elf\"", "");
            // A folded container also declares the fold: `q4nx_hadamard` names the sign table and
            // `q4nx_hadamard_block` the Hadamard block.  Without it the sidecar images would be
            // read in the stored (rotated) basis against un-rotated activations.
            const std::string hf = read_config_string(cfg_path, "\"q4nx_hadamard\"", "");
            had_block = q4nx_config_int(cfg_path.c_str(), "q4nx_hadamard_block", 1024);
            if (!hf.empty()) {
                const std::string hp = hf[0] == '/' ? hf : mdir + "/" + hf;
                if (!xd_->load_hadamard(hp, err_)) return false;
            }
            if (pb16i_dir_.empty()) {
                err_ = std::string("q4nx_weight_layout is ") + (t2s_ ? "t2s" : "pb16i") +
                       " but no sidecar directory is set (config q4nx_" +
                       (t2s_ ? "t2s_dir" : "pb16i_dir") + ", or Q4NX_" +
                       (t2s_ ? "T2S_DIR" : "PB16I_DIR") + ")";
                return false;
            }
            if (!valid_sidecar_dir(pb16i_dir_)) {
                err_ = std::string("the ") + (t2s_ ? "t2s" : "pb16i") +
                       " sidecar directory must be an absolute path without \"..\": " + pb16i_dir_;
                return false;
            }
        }
        fprintf(stderr, "  [forward] weight layout: %s (%s)\n",
                t2s_ ? (t2s_v2_ ? "t2s v2 2-bit ternary stream, expanded on the core (no shuffle)"
                                : "t2s v1 2-bit ternary stream, interleaved on the core")
                     : (pb16i_ ? "pb16i integer-exact ternary, sidecar images"
                               : (weight_layout_b8_ ? "b8 blocked 8x8 microtiles, tile-n-major"
                                                    : "rm row-major")),
                pb16i_ ? pb16i_dir_.c_str()
                       : (weight_layout_b8_ ? "kernels carry the _b8 suffix"
                                            : "q4nx_weight_layout absent/rm"));
    }
    // plan.json (engine #799 P1).  The model dir may carry one; when it does, the three decisions this
    // file used to make by looking at which files exist are read from it instead.  A model without one
    // keeps the probes below, which is what makes the plan additive rather than a migration.
    plan_path_ = mdir + "/plan.json";
    if (file_exists(plan_path_)) {
        try {
            plan_ = onebit::npu::plan::Plan::load(plan_path_);
        } catch (const std::exception& ex) {
            err_ = ex.what();     // the loader names the field: never a default for a broken plan
            return false;
        }
        // The plan may not disagree with the container about the kernel family, or the run would take
        // a different artifact than the weights need.
        const std::string want = family_tag();
        if (plan_.tag != want) {
            err_ = "plan " + plan_path_ + ": kernels.tag \"" + plan_.tag + "\" but the container is \"" +
                   (want.empty() ? "rm" : want.substr(1)) + "\" (expected \"" + want + "\")";
            return false;
        }
        fprintf(stderr, "  [plan] %s: %zu layers, gate/up %s, family \"%s\", %zu decode dir(s) from the plan\n",
                plan_path_.c_str(), plan_.layer_plans.size(), plan_.fused_gu ? "fused" : "split",
                plan_.tag.empty() ? "rm" : plan_.tag.c_str(), plan_.decode_dirs.size());
        noncausal_attn_ = plan_.noncausal();
        if (noncausal_attn_)
            fprintf(stderr, "  [plan] attention.mode noncausal_full_attention: full-attention layers read "
                            "the whole prompt (a prompt-encoding model; it cannot generate)\n");
    }
    // The pack's own decision_config.json is where a d3 pack declares the mask it was trained with
    // (`attention_mode`), and tools/npu_plan.py copies that declaration into plan.json. A directory
    // converted with `pack` + `decision` and no plan step therefore used to be answered causally --
    // measured on the real d3 pack, worst |dp| 0.3702 and a choice flip against 0.0246 with the plan
    // (engine#796) -- so the declaration decides when there is no plan, and a plan that contradicts it
    // is refused as stale (the rule lives in q4nx_semantics.c, tests/npu_rmsnorm_offset_test.c covers it).
    {
        const std::string mode = read_json_string(mdir + "/decision_config.json", "attention_mode");
        char why[512];
        const int nc = q4nx_attention_mode(mode.empty() ? nullptr : mode.c_str(), plan_.loaded() ? 1 : 0,
                                           noncausal_attn_ ? 1 : 0, why, sizeof why);
        if (nc < 0) {
            err_ = why;
            return false;
        }
        if (nc && !noncausal_attn_) {
            noncausal_attn_ = true;
            fprintf(stderr, "  [plan] attention.mode noncausal_full_attention from %s/decision_config.json "
                            "(no plan.json): full-attention layers read the whole prompt; run "
                            "tools/npu_plan.py <model dir> --kernels <elf dir> for the kernel-set decisions\n",
                    mdir.c_str());
        }
    }
    // The plan is the runtime's copy of the sealed-ELF decision; when it names one it wins, so a plan
    // and a container cannot disagree about which file the decode roles run.
    if (plan_.loaded() && !plan_.sealed_elf.empty()) pb16i_sealed_ = plan_.sealed_elf;
    // Hand the layout family to the launch layer now that the plan (or the probes) has settled it.
    xd_->set_pb16i(pb16i_, pb16i_dir_, pb16i_sealed_);
    xd_->set_hadamard_block(had_block);
    cfg.hidden_size = q4nx_config_int(cfg_path.c_str(), "hidden_size", cfg.hidden_size);
    cfg.num_layers = q4nx_config_int(cfg_path.c_str(), "num_hidden_layers", cfg.num_layers);
    cfg.num_attention_heads = q4nx_config_int(cfg_path.c_str(), "num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = q4nx_config_int(cfg_path.c_str(), "num_key_value_heads", cfg.num_key_value_heads);
    cfg.intermediate_size = q4nx_config_int(cfg_path.c_str(), "intermediate_size", cfg.intermediate_size);
    cfg.vocab_size = q4nx_config_int(cfg_path.c_str(), "vocab_size", cfg.vocab_size);
    {
        int hd = q4nx_config_int(cfg_path.c_str(), "head_dim", 0);
        if (hd <= 0 && cfg.num_attention_heads > 0)
            hd = cfg.hidden_size / cfg.num_attention_heads;
        if (hd > 0) cfg.head_dim = hd;
    }
    rope_scale_ = read_rope_short_factor(cfg_path);  // LongRoPE (MiniCPM4)
    rope_llama3_ = read_rope_llama3(cfg_path);       // rope_type "llama3" (engine#1048)
    rope_dynamic_ = read_rope_dynamic(cfg_path);     // rope_type "dynamic" (NTK by base, engine#1740 W2)
    {
        // A declared scaling this forward does not apply is refused BY NAME (engine#1087): serving it
        // would run plain rope, stay fluent and be wrong past the original window, with nothing in the
        // log to say so.  "" (no rope_scaling block) is every model that never declared one.
        std::ifstream f(cfg_path);
        std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        char declared[64];
        q4nx_rope_declared_type(t.c_str(), declared, (int)sizeof declared);
        if (!q4nx_rope_type_supported(declared, &rope_llama3_, rope_scale_.empty() ? 0 : 1, rope_dynamic_.on)) {
            if (!std::strcmp(declared, "llama3"))
                err_ = "config.json declares rope_type \"llama3\" but its numbers are not usable (factor > 0, "
                       "original_max_position_embeddings > 0, high_freq_factor > low_freq_factor), so the "
                       "scaling cannot be applied; serving it would run plain RoPE";
            else
                err_ = std::string("config.json declares rope scaling \"") + declared +
                       "\", which this forward does not implement (only llama3 and longrope short_factor "
                       "are), so it would be served plain RoPE and reply wrongly past the model's "
                       "original window (engine#1048, #1087)";
            return false;
        }
    }
    embed_scale_ = read_config_float(cfg_path, "\"embedding_scale\"", 1.0f);
    residual_scale_ = read_config_float(cfg_path, "\"residual_scale\"", 1.0f);
    logit_scale_ = read_config_float(cfg_path, "\"logit_scale\"", 1.0f);
    if (logit_scale_ == 0.0f) logit_scale_ = 1.0f;
    // engine#1185: an OGA-imported container stores its I8 weights as 6,144-byte q4_1f chunks
    // (fp32 scale/zero), which the kernels this forward drives do not read. Load the device form:
    // every q4_1f tensor repacked through q4nx::as_q4_1() into the Q4NX cache, logged once there.
    // A directory already in the device's form -- every GGUF-derived container -- is returned
    // unchanged and nothing is written.
    std::string load_path(model_path);
    try {
        const onebit::npu::q4nx::DeviceForm form = onebit::npu::q4nx::device_form(model_path, std::string());
        if (form.converted) load_path = form.model;
    } catch (const std::exception& e) {
        err_ = std::string("cannot put the imported container in the device's form: ") + e.what();
        return false;
    }
    mw_ = model_load(load_path.c_str(), cfg);
    if (!mw_) { err_ = "model_load failed"; return false; }
    // engine#1035: two MoE container conventions exist on disk.  The engine's own converter
    // (tools/gguf_to_q4nx.py) names the routed stacks model.layers.N.mlp.experts.*_proj.weight with
    // a plain [N][K] per-expert tile grid and a mlp.gate.weight router; FastFlowLM-style containers
    // use *_exps_proj names whose writer transposes each expert's tiles.  Which one this is, and
    // therefore how an expert slice decodes, is read from the container's own layer-0 tensor
    // (moe_container.h) rather than assumed -- either mistake is silent.
    moe_format_ = moe_format(
        model_tensor_by_name(mw_, "model.layers.0.mlp.gate_exps_proj.weight") != nullptr,
        model_tensor_by_name(mw_, "model.layers.0.mlp.experts.gate_proj.weight") != nullptr);
    if (moe_format_ == MoeFormat::EngineConverter)
        fprintf(stderr, "  [moe] container layout: engine-converter names, per-expert [N][K] tiles\n");
    // model_load sized its layer table from the tensors; config.json may not promise more.
    if (cfg.num_layers <= 0 || cfg.num_layers > mw_->config.num_layers) {
        err_ = "config.json num_hidden_layers (" + std::to_string(cfg.num_layers) + ") does not match the " +
               std::to_string(mw_->config.num_layers) + " layers in model.q4nx";
        return false;
    }

    // forward dims mirror the config the weights were loaded with
    c_.H = cfg.hidden_size;
    c_.NL = cfg.num_layers;
    // The target needs the depth for its weight-residency bound (the packed-weight cache is keyed by
    // (layer, projection), so it must know how many layers exist) - set once c_ is final.
    xd_->set_device(dev, weight_layout_b8_, weight_cache, c_.NL);
    c_.NH = cfg.num_attention_heads;
    c_.NKV = cfg.num_key_value_heads;
    c_.HD = cfg.head_dim;
    c_.IM = cfg.intermediate_size;
    c_.NV = cfg.vocab_size;
    fprintf(stderr, "  [forward] dims from %s: H=%d NL=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
            cfg_path.c_str(), c_.H, c_.NL, c_.NH, c_.NKV, c_.HD, c_.IM, c_.NV);

    const int H = c_.H, NL = c_.NL, HD = c_.HD, NV = c_.NV;
    // Embedding: BF16 for most models, but Gemma4 stores it QUANTIZED (dtype I8).
    // Reading a quantized tensor as bf16 yields ~zero, which invalidated the whole
    // forward and the tied LM head (all-zero logits); dequantize it once instead.
    /* Engine#881: the K grid has to be known before the FIRST dequant, not just before the
     * projections are sized: a quantised embedding table dequantises to kpad_ wide rows, and the
     * assignment below used to sit after this block (engine#845 put it next to the design grid),
     * so kpad_ was still 0 there and every container with an I8 embedding - the 27B rm/b8/pb16i
     * packs - failed with "embedding dtype I8 but dequant failed". */
    kpad_ = q4nx_pad_k(c_.H);
    impad_ = q4nx_pad_k(c_.IM);
    if (kpad_ != c_.H || impad_ != c_.IM)
        fprintf(stderr, "  [forward] container K grid: H %d -> %d, IM %d -> %d (pad columns are zero)\n",
                c_.H, kpad_, c_.IM, impad_);

    /* engine#1784: `embed_` (the bf16 copy) is no longer filled: the table is read in place. It stays for nothing
     * and is sized to zero, so a stale read would be a bounds error rather than a silent wrong answer. */
    {
        /* A tied model may store the table ONLY as `lm_head.weight` (Qwen3.5-4B's
         * container has no `model.embed_tokens.weight`); it doubles as both the
         * token embedding and the head.  Reading the unset descriptor instead
         * memcpy'd 1.3 GB from a bogus offset and the whole forward went NaN. */
        TensorDesc* etd = &mw_->embed_tokens;
        if (etd->data_size == 0 || etd->name[0] == '\0') {
            TensorDesc* alt = model_tensor_by_name(mw_, "lm_head.weight");
            if (alt && alt->data_size) etd = alt;
        }
        const void* e = model_tensor_data(mw_, etd);
        if (!e || etd->data_size == 0) { err_ = "no embed_tokens/lm_head.weight"; return false; }
        const char* dt = etd->dtype;
        const size_t en = (size_t)NV * (size_t)H;
        if (dt && dt[0] && strncmp(dt, "BF16", 4) != 0) {
            /* A quantised table is stored on the container's K grid, so it dequantises to kpad_
             * wide rows (engine#845); the pad columns are zero, so the head's dot product over
             * c_.H columns is unchanged. */
            /* engine#1784: nothing is expanded until it is asked for. The plain int8 + per-32-column-scale table --
             * the dtype the 27B and the gemma-class containers use -- is read element-wise straight from the mapping
             * by embed_at(), so 0.74 GiB of container no longer becomes 2.96 GiB of anonymous memory. Only the chunked
             * formats (which have no element-wise read yet) still decode into embed_f_. */
            lm_w_ = kpad_;
            const size_t esz = (size_t)etd->data_size;
            int rc;
            if (esz == en) {
                /* Plain int8 with a per-32-column scale (Gemma4).  The tensor is
                 * stored PRE-SCALED by the architecture's embedding scale, so
                 * the forward applies none here; the LM head divides it back
                 * out (see run_lm_head). */
                TensorDesc* sd = model_tensor_by_name(mw_, "model.embed_tokens.weight.scale");
                const float* sc = sd ? (const float*)data_of(mw_, sd, en / 32 * sizeof(float)) : nullptr;
                if (sd && !sc) { err_ = "model.embed_tokens.weight.scale is too small"; return false; }
                embed_i8_     = (const int8_t*)e;
                embed_sc_     = sc;
                embed_groups_ = H / Q4NX_GROUP_SIZE;
                rc            = 0;
            } else {
                /* engine#1784: the chunked formats are tiles of 5,120 B (the 27B's 0.625 B/element packed table is
                 * 155,200 of them), so a row is read out of the `cols/256` tiles that cover it instead of the whole
                 * table being decoded into f32 -- which for that pack is 0.74 GiB of container becoming 4.74 GiB.
                 * Same tile grid and ordering as q4nx_dequant_tensor_chunked(), pinned by the row test. */
                /* engine#1842: the grid is checked with the row reader's own rule, not `esz % tiles == 0`. That test let a
                 * table with no supported chunk kind, or an NV / kpad_ off the 32 x 256 grid, take this path; the reader
                 * then refused every row and embed_at() served 0.0f with no message. A table it would refuse stays on
                 * the whole-table decode below, which reports its own failure. */
                const int cb = q4nx_tile_row_kind(esz, NV, kpad_);
                if (cb != 0) {
                    embed_tile_       = (const uint8_t*)e;
                    embed_tile_bytes_ = esz;
                    embed_tile_rows_  = NV;
                    embed_tile_cols_  = kpad_;
                    embed_tile_cb_    = cb;
                    embed_row_buf_.assign((size_t)kpad_, 0.0f);
                    embed_row_tok_    = -1;
                }
                rc = 0;   // the buffer below is the fallback, and it reports its own failure
            }
            if (rc != 0) {
                err_ = std::string("embedding dtype ") + dt + " but dequant failed";
                return false;
            }
            if (!embed_tile_ && !embed_i8_) {   // no element-wise read: keep the whole-table buffer (unchanged path)
                embed_f_.resize((size_t)NV * (size_t)kpad_);
                if (dequant_any((const uint8_t*)e, esz, NV, kpad_, embed_f_.data()) != 0) {
                    err_ = std::string("embedding dtype ") + dt + " but dequant failed";
                    return false;
                }
            }
            embed_f32_ = true;
            if (embed_i8_ || embed_tile_)
                fprintf(stderr, "  [forward] embedding dtype %s -> read in place from the container, one row per token "
                                "(engine#1784): %.2f GiB not materialised\n", dt,
                        (double)NV * (double)kpad_ * 4.0 / (1024.0 * 1024.0 * 1024.0));
            else
                fprintf(stderr, "  [forward] embedding dtype %s -> dequantized (%zu floats, %.2f GiB)\n",
                        dt, embed_f_.size(),
                        (double)embed_f_.size() * sizeof(float) / (1024.0 * 1024.0 * 1024.0));
        } else {
            if (etd->data_size < (size_t)NV * H * 2) {
                err_ = "embedding too small for NV*H bf16";
                return false;
            }
            /* engine#1784: read the table in place out of the container's read-only mapping, one row per token,
             * instead of copying it. `e` points into `mw_`'s mapping, which outlives this forward, and embed_at()
             * indexes it at the container's own row stride. */
            embed_bf16_ = (const uint16_t*)e;
            lm_w_ = H;   // raw bf16: the table has no pad
        }
    }

    // ---- architecture semantics from the model's own config.json ----
    {
        std::string mp(model_path);
        size_t slash = mp.find_last_of('/');
        std::string mdir = slash == std::string::npos ? std::string(".") : mp.substr(0, slash);
        std::string cfg = mdir + "/config.json";
        if (q4nx_semantics_from_config(cfg.c_str(), mdir.c_str(), &sem_) != 0) {
            err_ = "cannot read " + cfg;
            return false;
        }
        if (sem_.eps <= 0) sem_.eps = c_.eps;
        q4nx_semantics_dump(&sem_);
        // A layer kind nothing implements is refused BY NAME, here beside the other precise refusals (engine#1087's
        // shape): q4nx_semantics maps an unrecognised layer_types entry to Q4NX_LAYER_OTHER and the attention path
        // treats OTHER as full attention, so a container declaring, say, a conv attention would be served as a plain
        // decoder - fluent, wrong at every position, silent.  Every container on this box declares only
        // full_attention / linear_attention (sliding-window included), so this refuses nothing that runs today.
        if (sem_.unknown_layer_type[0]) {
            err_ = std::string("config.json layer_types names \"") + sem_.unknown_layer_type +
                   "\", a layer kind this forward does not implement (it runs full_attention, sliding-window and "
                   "linear/GDN layers), so serving it would run those layers as full attention and be wrong at every "
                   "position; file the kind you need instead of converting it";
            return false;
        }
        // Refuse rather than silently apply full-attention math to a model whose
        // config declares layer types (or RoPE variants) this forward lacks.
        // Refusal is now precise.  Only layers whose attention MATH we do not
        // have (linear_attention / SSM) and interleaved/multimodal RoPE are
        // refused.  Sliding-window layers and per-layer-type RoPE are executed:
        // a sliding window is a mask and per-layer-type RoPE is a per-layer
        // selection, both host-side, so Gemma4 needs no new kernel.
        const int n_linear = q4nx_semantics_linear_layers(&sem_);
        // GDN (qwen3_5 linear_attention) IS implemented: the fused q|k|v and the
        // z projection are derived from the config's linear_* dims and run through
        // the same full-ELF GEMM path; conv/alpha/beta/recurrence are host-side.
        // Text-only mRoPE reduces to the ordinary partial-RoPE angles (all three
        // position components are equal), so it is executed too.  Refuse only when
        // the linear geometry is underivable from the config.
        lin_kd_ = lin_vd_ = lin_nkh_ = lin_nvh_ = lin_khd_ = lin_vhd_ = 0;
        if (n_linear > 0) {
            lin_nkh_ = q4nx_config_int(cfg_path.c_str(), "linear_num_key_heads", 0);
            lin_nvh_ = q4nx_config_int(cfg_path.c_str(), "linear_num_value_heads", 0);
            lin_khd_ = q4nx_config_int(cfg_path.c_str(), "linear_key_head_dim", 0);
            lin_vhd_ = q4nx_config_int(cfg_path.c_str(), "linear_value_head_dim", 0);
            lin_ck_ = q4nx_config_int(cfg_path.c_str(), "linear_conv_kernel_dim", 4);
            lin_kd_ = lin_nkh_ * lin_khd_;
            lin_vd_ = lin_nvh_ * lin_vhd_;
            if (lin_kd_ <= 0 || lin_vd_ <= 0 || lin_ck_ < 1) {
                char mix[128];
                q4nx_semantics_layer_mix(&sem_, mix, sizeof(mix));
                char msg[384];
                snprintf(msg, sizeof(msg),
                         "unsupported attention layers (%s): %d layer(s) are "
                         "linear_attention but the config has no usable "
                         "linear_num_*/linear_*_head_dim dims",
                         mix, n_linear);
                err_ = msg;
                return false;
            }
            fprintf(stderr, "  [forward] GDN linear attention: %d layer(s), "
                    "key_dim=%d value_dim=%d heads=%d/%d hd=%d/%d conv_k=%d\n",
                    n_linear, lin_kd_, lin_vd_, lin_nkh_, lin_nvh_, lin_khd_,
                    lin_vhd_, lin_ck_);
        }
        attn_gate_ = q4nx_config_int(cfg_path.c_str(), "attn_output_gate", 0);
        if (attn_gate_) fprintf(stderr, "  [forward] attn_output_gate=1 (q_proj emits q|gate)\n");
        // The config's RoPE base supersedes the built-in default.
        c_.theta = sem_.rope_theta;
    }

    kcache_.assign(NL, {});
    vcache_.assign(NL, {});
    // Unconditional: this runs BEFORE the MLA detection below sets mla_, so
    // gating it on mla_ left the MLA caches empty and mla_attn segfaulted.
    mla_kn_.assign(NL, {}); mla_kr_.assign(NL, {}); mla_v_.assign(NL, {});

    // ---- derive the design set from the architecture dims ----
    // fused vs split gate/up is an artifact property, so probe for it rather
    // than keying on a model name: a GU artifact means fused, G/U means split.
    // Zero-initialised: the optionally-derived fields (ple_dim, double_wide_mlp,
    // NKV2/HD2, lin_kd/lin_vd, attn_output_gate) must be 0 when absent, not
    // garbage from the stack.
    /* MLA + MoE (deepseek2 / GLM-4.7-Flash).  A deepseek2 config's
     * head_dim/key_length describe the KV LATENT, not a per-head width, so the
     * MLA geometry comes from its own keys; then HD is re-pointed at the per-head
     * VALUE width so the standard O derivation gives K=NH*HD=NH*v_head, and IM at
     * the per-EXPERT intermediate so G/U/D ARE the expert MLP. */
    mla_ = q4nx_config_int(cfg_path.c_str(), "mla", 0);
    if (mla_) {
        q_lora_ = q4nx_config_int(cfg_path.c_str(), "q_lora_rank", 0);
        kv_lora_ = q4nx_config_int(cfg_path.c_str(), "kv_lora_rank", 0);
        rope_dim_ = q4nx_config_int(cfg_path.c_str(), "rope_dim", 0);
        qk_nope_ = q4nx_config_int(cfg_path.c_str(), "qk_nope_head_dim", 0);
        v_head_ = q4nx_config_int(cfg_path.c_str(), "v_head_dim", 0);
        if (q_lora_ <= 0 || kv_lora_ <= 0 || rope_dim_ <= 0 || qk_nope_ <= 0 ||
            v_head_ <= 0) {
            fprintf(stderr, "  [mla] config incomplete (q_lora=%d kv_lora=%d rope=%d nope=%d "
                            "v_head=%d)\n", q_lora_, kv_lora_, rope_dim_, qk_nope_, v_head_);
            err_ = "MLA config incomplete";
            return false;
        }
        // k_b's Q4NX slice is [kv_lora][nope] with the nope axis padded to the
        // 256-column block (192 -> 256), which is what its 16 tiles/head show.
        kb_rows_ = qk_nope_;   /* the re-converted Q4NX packs k_b at [192][512], 12 tiles/head (verified corr 0.996926 vs the GGUF) */
        c_.HD = v_head_;
        fprintf(stderr, "  [mla] MLA: q_lora=%d kv_lora=%d rope=%d nope=%d v_head=%d NH=%d\n",
                q_lora_, kv_lora_, rope_dim_, qk_nope_, v_head_, c_.NH);
    }
    // MoE params are parsed INDEPENDENTLY of mla_ so a non-MLA MoE arch (qwen3moe /
    // qwen35moe) gets its expert geometry too; reading them only inside `if (mla_)`
    // left n_expert_/top_k_/expert_im_ zero and the standard MoE path dead.
    // expert_im_ is the per-EXPERT intermediate, which c_.IM already holds for a MoE
    // config (the converter writes the per-expert FFN width as intermediate_size).
    // MoE containers name these keys in one of two conventions: llama.cpp-style `n_expert`/`top_k`
    // (FastFlowLM-style containers) or the HF names the engine's own converter writes --
    // `num_experts`/`num_experts_per_tok` (engine#1035).  Reading only the first made a converted
    // Qwen3-Coder-30B-A3B run as a dense model, whose first layer then asked for a
    // mlp.gate_proj.weight the container does not carry: it surfaced as
    // "MLP dequant : data_size 0 at geometry 768x2048 ... the container and config.json disagree".
    n_expert_ = moe_expert_count(q4nx_config_int(cfg_path.c_str(), "n_expert", 0),
                                 q4nx_config_int(cfg_path.c_str(), "num_experts", 0));
    if (n_expert_ > 0) {
        top_k_ = moe_top_k(q4nx_config_int(cfg_path.c_str(), "top_k", 0),
                           q4nx_config_int(cfg_path.c_str(), "num_experts_per_tok", 0));
        const int moe_im = q4nx_config_int(cfg_path.c_str(), "moe_intermediate_size", 0);
        expert_im_ = moe_im > 0 ? moe_im : c_.IM;
        dense_im_ = q4nx_config_int(cfg_path.c_str(), "dense_intermediate_size", 0);
        leading_dense_ = q4nx_config_int(cfg_path.c_str(), "leading_dense_block_count", 0);
        expert_gating_sigmoid_ =
            q4nx_config_int(cfg_path.c_str(), "expert_gating_func", 1) == 2;  // 2 = sigmoid
        expert_w_scale_ = read_config_float(cfg_path, "\"expert_weights_scale\"", 1.0f);
        if (top_k_ <= 0 || top_k_ > n_expert_ || expert_im_ <= 0) {
            fprintf(stderr, "  [moe] config incomplete (n_expert=%d top_k=%d im=%d)\n",
                    n_expert_, top_k_, expert_im_);
            err_ = "MoE config incomplete";
            return false;
        }
        fprintf(stderr, "  [moe] %sMoE: NH=%d n_expert=%d top%d im=%d dense_im=%d gating=%s\n",
                mla_ ? "MLA+" : "", c_.NH, n_expert_, top_k_, expert_im_, dense_im_,
                expert_gating_sigmoid_ ? "sigmoid" : "softmax");
    }

    // attention() shares each KV head across NH / NKV query heads (kh = h / (NH / NKV)), so the
    // KV heads must divide the query heads; Gemma4's global layers have their own KV count.
    if (!mla_) {
        auto bad_kv = [&](int nkv) {
            if (nkv > 0 && nkv <= c_.NH && c_.NH % nkv == 0) return false;
            err_ = "config.json has " + std::to_string(c_.NH) + " attention heads and " + std::to_string(nkv) +
                   " key/value heads; the key/value heads must divide the attention heads";
            return true;
        };
        if (bad_kv(c_.NKV)) return false;
        if (sem_.num_global_kv_heads > 0 && bad_kv(sem_.num_global_kv_heads)) return false;
    }

    Q4nxModelDims dims{};
    dims.mla = mla_;
    dims.q_lora = q_lora_; dims.kv_lora = kv_lora_; dims.rope_dim = rope_dim_;
    dims.qk_nope = qk_nope_; dims.dense_im = dense_im_;
    dims.H = c_.H; dims.NH = c_.NH; dims.NKV = c_.NKV; dims.HD = c_.HD;
    dims.IM = c_.IM;
    // Activation tile of the PREFILL family.  It used to be the only family (a
    // bare 128 here decided the whole engine's M); decode now selects the M=16
    // twin instead -- see the m16 block after q4nx_derive_designs() and
    // ctx_for().  Keep 128: prefill packs a whole chunk of rows per launch.
    dims.M = Q4NX_DESIGN_M_PREFILL;
    dims.ple_dim = sem_.hidden_size_per_layer_input;
    dims.double_wide_mlp = sem_.double_wide_mlp;
    /* A second attention geometry when the head dim varies by layer type. */
    dims.NKV2 = sem_.num_global_kv_heads;
    dims.HD2 = sem_.global_head_dim;
    /* GDN linear-attention projections (qwen3_5): fused q|k|v and z. */
    dims.lin_kd = lin_kd_;
    dims.lin_vd = lin_vd_;
    dims.attn_output_gate = attn_gate_;
    if (plan_.loaded()) {
        // What the plan says, not what happens to be in --kernels: this is the probe whose answer
        // changed when the same model was served against a different kernel directory.
        dims.fused_gu = plan_.fused_gu ? 1 : 0;
    } else {
        // The artifact probes must use the same tag the designs will carry, or a
        // b8 container would silently conclude "no fused GU kernel" and fall back
        // to split G/U on a model that ships one.
        const char* tag = family_tag();
        char gu[512], g[512];
        /* On the kernel's grids, exactly as q4nx_derive_designs() will name them (engine#845). */
        snprintf(gu, sizeof(gu), "%s/full_i8_GU_K%d_N%d%s.elf", dir.c_str(), q4nx_pad_k(dims.H),
                 q4nx_pad_n(2 * dims.IM), tag);
        snprintf(g, sizeof(g), "%s/full_i8_G_K%d_N%d%s.elf", dir.c_str(), q4nx_pad_k(dims.H),
                 q4nx_pad_n(dims.IM), tag);
        dims.fused_gu = file_exists(gu) || !file_exists(g) ? 1 : 0;
    }
    ndesigns_ = q4nx_derive_designs(&dims, designs_);
    if (ndesigns_ <= 0) { err_ = "design derivation failed"; return false; }
    // Batched expert gate/up (MoE): one GEMM per projection over ALL top-k experts
    // at once, instead of one per expert.  Derived here rather than in
    // q4nx_derive_designs because it depends on top_k, which the dims struct does
    // not carry.  The design's N is a G/U kernel width that holds top_k*IM; the
    // caller passes the real N = top_k*IM, so only those columns are packed.
    if (n_expert_ > 0 && top_k_ > 1 && expert_im_ > 0 && !mla_ && ndesigns_ + 2 <= Q4NX_MAX_DESIGNS) {
        const int need = top_k_ * expert_im_;
        int nb = 8192;   // the only K=H G/U kernel width >= 4096 in the set
        bool batched = false;
        if (plan_.loaded()) {
            batched = plan_.moe_batched_gu.enabled;
            if (batched) nb = plan_.moe_batched_gu.N;
        } else {
            char gb[512], ub[512];
            snprintf(gb, sizeof(gb), "%s/full_i8_G_K%d_N%d%s.elf", dir.c_str(), c_.H, nb,
                     weight_layout_b8_ ? "_b8" : "");
            snprintf(ub, sizeof(ub), "%s/full_i8_U_K%d_N%d%s.elf", dir.c_str(), c_.H, nb,
                     weight_layout_b8_ ? "_b8" : "");
            batched = need <= nb && file_exists(gb) && file_exists(ub);
        }
        if (batched && need <= nb) {
            memset(&designs_[ndesigns_], 0, sizeof(designs_[ndesigns_]));
            designs_[ndesigns_].M = dims.M; designs_[ndesigns_].K = c_.H; designs_[ndesigns_].N = nb;
            designs_[ndesigns_].role = Q4NX_DESIGN_GB;
            snprintf(designs_[ndesigns_].elf_name, Q4NX_ELF_NAME_MAX, "full_i8_G_K%d_N%d.elf", c_.H, nb);
            ndesigns_++;
            memset(&designs_[ndesigns_], 0, sizeof(designs_[ndesigns_]));
            designs_[ndesigns_].M = dims.M; designs_[ndesigns_].K = c_.H; designs_[ndesigns_].N = nb;
            designs_[ndesigns_].role = Q4NX_DESIGN_UB;
            snprintf(designs_[ndesigns_].elf_name, Q4NX_ELF_NAME_MAX, "full_i8_U_K%d_N%d.elf", c_.H, nb);
            ndesigns_++;
            fprintf(stderr, "  [forward] batched expert gate/up: N=%d (top_k=%d IM=%d) via K%d_N%d kernels\n",
                    need, top_k_, expert_im_, c_.H, nb);
        }
    }

    size_t maxB = 0;
    for (int i = 0; i < ndesigns_; i++)
        maxB = std::max(maxB, (size_t)designs_[i].K * (size_t)designs_[i].N);
    B_.resize(maxB);

    // Per-layer projection scratch, sized from the model's own dims (the earlier
    // version hardcoded 0.6B's NH*HD=2048, and the rewrite dropped the sizing
    // entirely, leaving these vectors empty: wq_.data() was null, so
    // q4nx_dequant_tensor's !out guard returned -1 and every step reported
    // "dequant failed").
    // Size the per-layer scratch for the WIDEST attention type (Gemma4's full
    // layers are 2x wider than its sliding ones) so one buffer serves both.
    const int max_hd = (sem_.global_head_dim > c_.HD) ? sem_.global_head_dim : c_.HD;
    const int max_nkv = (sem_.num_global_kv_heads > c_.NKV) ? sem_.num_global_kv_heads : c_.NKV;
    const size_t q_dim = (size_t)c_.NH * (size_t)max_hd;
    const size_t kv_dim = (size_t)max_nkv * (size_t)max_hd;
    // Widths are the container's padded K (engine#845): the f32 copy of a projection is stored the
    // way the tensor is stored, and assemble_B's source stride is the same number.
    wq_.resize(q_dim * (size_t)kpad_ * (attn_gate_ ? 2 : 1));   // q|gate doubles it
    wk_.resize(kv_dim * (size_t)kpad_);
    wv_.resize(kv_dim * (size_t)kpad_);
    wo_.resize((size_t)H * (size_t)q4nx_pad_k((int)q_dim));
    wg_.resize((size_t)c_.IM * (size_t)kpad_);
    wu_.resize((size_t)c_.IM * (size_t)kpad_);
    wd_.resize((size_t)H * (size_t)impad_);

    // GDN scratch + per-layer recurrence state (qwen3_5 linear_attention).
    if (lin_kd_ > 0) {
        const size_t CD = 2 * (size_t)lin_kd_ + (size_t)lin_vd_;
        wqkvlin_.resize(CD * (size_t)H);
        wz_.resize((size_t)lin_vd_ * (size_t)H);
        woutlin_.resize((size_t)H * (size_t)lin_vd_);
        gdn_small_.resize((size_t)NL);
        for (auto& g : gdn_small_) {
            g.wa.resize((size_t)lin_nvh_ * (size_t)H);
            g.wb.resize((size_t)lin_nvh_ * (size_t)H);
            g.conv.resize((size_t)lin_ck_ * CD);
            g.norm.resize((size_t)lin_vhd_);
            g.ssm_a.resize((size_t)lin_nvh_);
            g.dt.resize((size_t)lin_nvh_);
        }
        if (const char* v = getenv("NPU_GDN_CACHE")) {
            gdn_cache_ = std::atoi(v) != 0;
            std::fprintf(stderr, "[gdn] per-layer small-member cache %s (NPU_GDN_CACHE=%s)\n",
                         gdn_cache_ ? "on" : "off", v);
        }
        conv_state_.assign(NL, std::vector<float>((size_t)CD * (size_t)(lin_ck_ - 1), 0.0f));
        rec_state_.assign(NL, std::vector<float>((size_t)lin_nvh_ * (size_t)lin_khd_ * (size_t)lin_vhd_, 0.0f));
    }

    fprintf(stderr, "  [forward] derived %d designs (fused_gu=%d, H=%d NH=%d NKV=%d HD=%d IM=%d)\n",
            ndesigns_, dims.fused_gu, dims.H, dims.NH, dims.NKV, dims.HD, dims.IM);

    // The container states which design family it was imported for, and a serve that derives a different one refuses
    // rather than answering garbage. Two containers with the same hidden size can differ in IM, head count or head dim
    // (so in every K/N the kernels are looked up by), and pointing one at the other's ELF set used to be a silent
    // wrong answer. The field is written by `1bit npu-pack import` as "ROLE:KxN,ROLE:KxN,..."; a container without it
    // (everything imported before this) is unaffected.
    {
        const std::string want = read_config_string(cfg_path, "\"q4nx_designs\"", "");
        if (!want.empty()) {
            std::string have;
            for (int i = 0; i < ndesigns_; i++) {
                have += (have.empty() ? "" : ",") + std::string(role_tag(designs_[i].role)) + ":" +
                        std::to_string(designs_[i].K) + "x" + std::to_string(designs_[i].N);
            }
            if (have != want) {
                err_ = "this container declares its kernel family as \"" + want + "\", but this model derives \"" +
                       have + "\": the ELFs it was imported for are not the ones this geometry needs, so it would "
                       "answer wrong text rather than fail (re-import, or serve the container that matches)";
                return false;
            }
        }
    }

    // pb16i has no kernel for a wide fused full-attention QKV: its q half needs the
    // int64 accumulator, and 14336/32 = 448 int64 columns per core do not fit L1.  The
    // role therefore runs as the two q halves (N = NH*HD each, i.e. q and the output
    // gate) plus k and v (N = NKV*HD) -- four launches per full-attention layer, which
    // is the split the arena measured at 1.49 ms/layer.  They are designs of their own
    // so the launch path, the profiling and the sidecar key stay one shape per ELF.
    if (pb16i_ && !mla_ && lin_kd_ > 0 && c_.NH > 0 && c_.HD > 0 && c_.NKV > 0 &&
        ndesigns_ + 2 <= Q4NX_MAX_DESIGNS) {
        const int nq = c_.NH * c_.HD, nkv = c_.NKV * c_.HD;
        const bool geo_ok = t2s_ ? (q4nx_t2s::fits(c_.H, nq) && q4nx_t2s::fits(c_.H, nkv))
                                 : (q4nx_pb16i::fits(c_.H, nq) && q4nx_pb16i::fits(c_.H, nkv));
        if (geo_ok) {
            auto add = [&](Q4nxDesign role, int N, const char* role_name) {
                Q4nxDesignGeom& g = designs_[ndesigns_];
                memset(&g, 0, sizeof(g));
                g.M = Q4NX_DESIGN_M_DECODE;
                g.K = c_.H;
                g.N = N;
                g.role = role;
                snprintf(g.elf_name, Q4NX_ELF_NAME_MAX, "full_i8_%s_K%d_N%d%s.elf",
                         role_name, g.K, g.N, family_tag());
                return ndesigns_++;
            };
            const int i_qhalf = add(Q4NX_DESIGN_QKVFA_Q, nq, "QKVFAQ");
            const int i_kv = add(Q4NX_DESIGN_QKVFA_KV, nkv, "QKVFAKV");
            xd_->set_role_split(i_qhalf, i_kv);
            fprintf(stderr, "  [forward] %s: full-attention QKV as 2 launches of K%d_N%d "
                            "(q|gate) + 2 of K%d_N%d (k, v)\n", t2s_ ? "t2s" : "pb16i",
                    c_.H, nq, c_.H, nkv);
        } else {
            fprintf(stderr, "  [forward] %s: no full-attention QKV split (K=%d N=%d/%d is "
                            "not a %s geometry)\n", t2s_ ? "t2s" : "pb16i", c_.H, nq, nkv,
                    t2s_ ? "t2s" : "pb16i");
        }
    }

    // A blocked container needs the blocked-tap kernels: the tap lives in the
    // ELF, so a row-major ELF on a blocked buffer would read the weights
    // permuted and produce silent garbage.  Tagging every design's artifact
    // name keeps the two families in one directory and makes a missing artifact
    // a loud error instead of a wrong number.  The twins derive from these
    // names, so they become "..._b8_m16.elf".  A t2s container is NOT tagged here: like pb16i, its
    // M=128 designs still run the int8 family (the stream is a decode family only), and only the
    // decode designs get the `_t2s` suffix, below.
    if (weight_layout_b8_) {
        for (int i = 0; i < ndesigns_; i++) {
            char* n = designs_[i].elf_name;
            const size_t len = strlen(n);
            if (len < 5 || len + 4 > Q4NX_ELF_NAME_MAX - 1 ||
                strcmp(n + len - 4, ".elf") != 0) {
                err_ = std::string("cannot tag the kernel name for the b8 layout: ") + n;
                return false;
            }
            std::string tagged(n, len - 4);
            tagged += "_b8.elf";
            snprintf(n, Q4NX_ELF_NAME_MAX, "%s", tagged.c_str());
        }
    }


    // The plan may name this very set (engine#803 A2a).  Checked HERE and not at the end of the
    // derivation: the pb16i family appends its split designs and the b8 family rewrites every name
    // with the _b8 suffix between the derivation and this loop, so this is the first point at
    // which designs_[] is what a target would actually open.  When it does, it has to be the set the
    // container derives -- same roles, same M/K/N, same artifact -- or the run would take an ELF the
    // weights were not written for.  A plan without the section agrees trivially, which is what keeps
    // every plan written before the field meaningful.
    if (plan_.loaded()) {
        std::vector<onebit::npu::plan::Plan::Design> derived;
        derived.reserve((size_t)ndesigns_);
        for (int i = 0; i < ndesigns_; i++) {
            onebit::npu::plan::Plan::Design d;
            // The plan's role is a key, so it takes the unique name; the log keeps role_tag()'s
            // short tag (where QKVLIN prints as "QKV").
            d.role = q4nx_design_role_name(designs_[i].role);
            d.M = designs_[i].M;
            d.K = designs_[i].K;
            d.N = designs_[i].N;
            d.elf = designs_[i].elf_name;
            derived.push_back(std::move(d));
        }
        std::string why;
        if (!plan_.designs_agree(derived, why)) {
            err_ = "plan " + plan_path_ + ": " + why;
            return false;
        }
    }

    // The M=128 ELF set is the target's to open (engine#803 A2b): it adopts the geometry, probes the
    // artifact and refuses by name, and the forward no longer touches an I8Ctx to do it.
    {
        // What the setup side reads, set once: the remaining two blocks (decode family, lm-head) read
        // the same fields, so moving them later needs no new plumbing (engine#803 A2b).
        npu::target::xdna::XdnaTarget::Setup setup;
        setup.model_dir = mdir;
        setup.kernels_dir = dir;
        setup.family = family_tag();
        setup.plan_path = plan_.loaded() ? plan_path_ : std::string();
        setup.b8 = weight_layout_b8_;
        setup.pb16i = pb16i_;
        setup.t2s = t2s_;
        setup.t2s_v2 = t2s_v2_;
        setup.host_gemm = host_gemm_;
        // The lm-head block's container-side inputs: the sidecar dir, whether the head's ternary image is
        // there at all (this file owns that path rule), where decode twins are looked for, and the geometry.
        setup.pb16i_dir = pb16i_dir_;
        setup.head_pb16i_images =
            pb16i_ && !pb16i_dir_.empty() && file_exists(head_pb16i_stem(0) + image_ext());
        setup.decode_dirs = m16dirs_like_head(mdir, dir, plan_decode_dirs_);
        setup.hidden = c_.H;
        setup.vocab = c_.NV;
        setup.kpad = kpad_;
        setup.plan = plan_.loaded() ? &plan_ : nullptr;
        xd_->configure(setup);
        std::string why;
        // Import-time completeness (engine#799 rule (c)): a plan that names an op whose kernel is not here
        // is refused HERE, naming the design and the artifact, instead of at the first launch or -- worse --
        // by quietly running that role on the host loop.  The target's own probe stays as the second line:
        // this one runs before anything is opened at all.
        if (!host_gemm_ && plan_.loaded() && !plan_.check_artifacts({dir}, why)) {
            err_ = std::move(why);
            return false;
        }
        if (!xd_->open_prefill(designs_, ndesigns_, why)) {
            err_ = std::move(why);
            return false;
        }
    }

    // ---- M=16 decode family ----
    // Decode is one row, but every design above is M=128, so the host zero-pads
    // 128 activation rows and the kernel streams all 128 through the array to
    // deliver 1: 8 of 32 compute tiles hold data and 1 of 128 baked MAC rows is
    // useful (0.78%).  The M=16 twins are the SAME role geometries at the
    // smallest vectorized mmul M -- one core row, all 8 columns, column groups
    // still looped inside the ELF, so one launch per role is unchanged -- and a
    // launch measured ~0.34 ms against 3.4-4.9 ms for M=128.  Selection is by
    // row count (ctx_for): rows <= 16 is decode or a short prefill chunk,
    // anything wider keeps M=128 so long prefill does not regress.  A role with
    // no twin ELF on disk simply stays on M=128, so the artifact directory can
    // carry both families (or only the old one).  NPU_DECODE_M16=0 forces the
    // all-M=128 path for A/B runs.
    //
    // Where the twins are looked for, first hit wins: <model dir>/kernels-m16,
    // then <kernels>/m16, then <kernels> itself.  The point of the first two is
    // that a model directory ships its own decode family, so a plain
    // `--kernels <model>/kernels` serve runs M=16 with no extra flag; a shared
    // kernel dir (one whose ELF set serves several models) stays untouched.
    // Both candidates are derived from paths the serve was already given
    // (`-m` and `--kernels`), so no new user-controlled path reaches fopen:
    // an earlier draft also read an NPU_KERNELS_M16 override, which is exactly
    // the cpp/path-injection shape and is not needed for the documented use.
    xd_->set_profile(getenv("NPU_PROFILE") != nullptr);
    xd_->set_m16_check(getenv("NPU_M16_CHECK") != nullptr);
    const char* m16_env = getenv("NPU_DECODE_M16");
    const bool want_m16 = !m16_env || atoi(m16_env) != 0;
    if (want_m16 && !host_gemm_) {
        q4nx_derive_decode_designs(designs_, ndesigns_, designs16_);
        // An integer-exact ternary container's decode family is its own kernel set, not the int8
        // M=16 twins: same M=16 activation tile, but the image (and so the tap) is pb16i's or
        // t2s's, the ELF name carries that suffix, and the geometry test is the image's
        // (pb16i: K%256, N%1024; t2s: K%256, N%32) rather than the int8 twin's tiling.  A design
        // that is ternary-only skips the derivation entirely: its own entry already IS the decode
        // design.
        if (pb16i_) {
            const std::string ft = family_tag();
            for (int i = 0; i < ndesigns_; i++) {
                if (i == xd_->role_split_qhalf() || i == xd_->role_split_kv()) {
                    designs16_[i] = designs_[i];
                    continue;
                }
                // The wide fused full-attention QKV has no ternary kernel (its int64 half
                // needs 448 columns per core); when the split designs exist that role is
                // covered by them, so this design simply has no decode kernel.
                const Q4nxDesign r16 = designs_[i].role;
                if (xd_->role_split_qhalf() >= 0 && (r16 == Q4NX_DESIGN_QKV || r16 == Q4NX_DESIGN_QKV2)) {
                    designs16_[i].M = 0;
                    continue;
                }
                const bool geo_ok = t2s_ ? q4nx_t2s::fits(designs16_[i].K, designs16_[i].N)
                                         : q4nx_pb16i::fits(designs16_[i].K, designs16_[i].N);
                if (designs16_[i].M <= 0 || !geo_ok) {
                    // Say WHY, once per role.  A geometry the ternary tile cannot hold is not a
                    // missing artifact and not an error: the role keeps its M=128 kernel and the
                    // model is correct, only slower -- so the only symptom is a decode step that
                    // costs what a prefill step costs, with nothing in the log.  That is the same
                    // "nothing logs the decline" gap the HRX prefill N%16 note names for a
                    // different dispatcher, and it is cheap to close here.
                    if (designs16_[i].M > 0 && !geo_ok)
                        fprintf(stderr,
                                "  [forward] dec %-3s: no %s twin (K=%d N=%d does not fit the tile), "
                                "staying on M=128\n",
                                role_tag(designs16_[i].role), t2s_ ? "t2s" : "pb16i",
                                designs16_[i].K, designs16_[i].N);
                    designs16_[i].M = 0;
                    continue;
                }
                // Build the new name in a copy: snprintf must not read and write the
                // same buffer.
                std::string n16(designs16_[i].elf_name);
                n16.resize(n16.size() - 4);                  // drop ".elf"
                if (n16.size() >= 4 && n16.compare(n16.size() - 4, 4, "_m16") == 0)
                    n16.resize(n16.size() - 4);              // and the m16 twin's suffix
                n16 += ft;
                // pb16i's decode family carries no `_m16` marker: M=16 is its only design and those
                // files already exist under that name on every deployed container.  A t2s container's
                // M=128 designs keep their int8 names (untagged), so its decode kernels are told
                // apart by the marker the arena uses: full_i8_GU_K5120_N17408_t2s_m16.elf.
                if (t2s_) n16 += "_m16";
                n16 += ".elf";
                snprintf(designs16_[i].elf_name, Q4NX_ELF_NAME_MAX, "%s", n16.c_str());
            }
        }
        std::vector<std::string> m16dirs;
        plan_decode_dirs_.clear();
        if (plan_.loaded() && !plan_.decode_dirs.empty()) {
            // The plan says which of the two given paths each candidate hangs off, so the runtime
            // stops guessing which base a bare directory name meant.
            for (const auto& d : plan_.decode_dirs)
                plan_decode_dirs_.push_back((d.under_model ? mdir : dir) + "/" + d.path);
            for (const auto& p : plan_decode_dirs_) m16dirs.push_back(p);
        } else {
            m16dirs.push_back(mdir + "/kernels-m16");
            m16dirs.push_back(dir + "/m16");
            m16dirs.push_back(dir);
        }
        int made = 0, skipped = 0;
        for (int i = 0; i < ndesigns_; i++) {
            if (designs16_[i].M <= 0) { skipped++; continue; }
            std::string elf, kname = "main:seq";
            if (pb16i_ && !pb16i_sealed_.empty()) {
                // One ELF for every ternary phase: the file is the same for all roles and
                // only the instance name differs, so I8Ctx reuses the one hw_context.
                const char* inst = pb16i_instance(designs16_[i].role);
                if (!inst) {
                    fprintf(stderr, "  [forward] pb16i: the sealed ELF has no instance for "
                                    "role %s; this role has no decode kernel\n",
                            role_tag(designs16_[i].role));
                    designs16_[i].M = 0;
                    skipped++;
                    continue;
                }
                elf = dir + "/" + pb16i_sealed_;
                kname = inst;
                if (!file_exists(elf)) {
                    err_ = plan_.loaded() ? ("plan " + plan_path_ + ": decode.sealed_elf \"" +
                                             pb16i_sealed_ + "\" is not in " + dir + " (missing)")
                                          : ("pb16i: missing sealed ELF " + elf);
                    return false;
                }
            } else {
                for (const std::string& d : m16dirs) {
                    std::string cand = d + "/" + designs16_[i].elf_name;
                    if (file_exists(cand)) { elf = cand; break; }
                }
            }
            if (elf.empty()) {
                // A pb16i role without its image kernel must stop the load: it is not
                // a slower path, it is a missing one (the int8 weights are not packed
                // for these geometries under this layout).
                if (pb16i_) {
                    err_ = plan_.loaded()
                               ? std::string("plan ") + plan_path_ + ": decode role " +
                                     role_tag(designs16_[i].role) + " needs \"" + designs16_[i].elf_name +
                                     "\" (missing)"
                               : std::string("pb16i: missing decode ELF ") + designs16_[i].elf_name;
                    return false;
                }
                skipped++;
                continue;   // stay on M=128
            }
            I8Ctx* c16 = xd_->make_ctx16(i, designs16_[i]);
            if (!c16) { err_ = "cannot create the decode context for role " + std::to_string(i); return false; }
            c16->MD = designs16_[i].M;
            c16->KD = designs16_[i].K;
            c16->ND = designs16_[i].N;
            c16->blocked_b = weight_layout_b8_;
            c16->pb16i = pb16i_;
            c16->t2s = t2s_;
            // Which pb16i ELFs leave two int32 planes (the arena's ACC64 family: D, O and
            // the full-attention q half).  This mirrors the ELF family contract rather
            // than inferring it, and load/launch refuse a tensor that needs two planes on
            // a one-plane ELF.
            if (pb16i_) {
                const Q4nxDesign r = designs16_[i].role;
                c16->pb16i_planes =
                    (r == Q4NX_DESIGN_D || r == Q4NX_DESIGN_O || r == Q4NX_DESIGN_QKVFA_Q) ? 2 : 1;
            }
            if (!c16->init_elf(*dev, elf.c_str(), designs16_[i].M,
                               designs16_[i].K, designs16_[i].N, 1, kname.c_str())) {
                if (pb16i_) {
                    err_ = "pb16i: init_elf failed for " + elf;
                    return false;
                }
                fprintf(stderr, "  [forward] M=16 twin failed to load (%s); "
                                "this role keeps M=128\n", designs16_[i].elf_name);
                xd_->drop_ctx16(i);
                skipped++;
                continue;
            }
            made++;
            fprintf(stderr, "  [forward] %-7s M=%d K=%d N=%d  %s (decode)\n",
                    role_tag(designs_[i].role), designs16_[i].M, designs16_[i].K,
                    designs16_[i].N, elf.c_str());
        }
        xd_->note_m16_made(made > 0);
        if (pb16i_ && made < ndesigns_ - 0) {
            fprintf(stderr, "  [forward] %s: %d of %d designs have a decode kernel\n",
                    t2s_ ? "t2s" : "pb16i", made, ndesigns_);
        }
        fprintf(stderr, "  [forward] decode family: %d role(s) at M=%d, %d on M=%d%s\n",
                made, Q4NX_DESIGN_M_DECODE, skipped, Q4NX_DESIGN_M_PREFILL,
                want_m16 ? "" : " (NPU_DECODE_M16=0: disabled)");
    }

    // The lm head's own context (pb16i ternary images, the int8 ladder, or the q4_1 gemv head).  Cut out
    // into its own method with its body unchanged, so the next step (moving it behind XdnaTarget, as the
    // prefill set already is) is a copy rather than a surgery inside init_impl.
    // The lm-head context is the target's now (engine#803 A2b): the body moved verbatim into
    // XdnaTarget::open_head(), and what stayed here is the container's naming, because which sidecar file
    // the head has is the container's rule.  The flag the model path reads is written by the target.
    if (!xd_->open_head(designs_, ndesigns_, err_)) return false;
    head_pb16i_ = xd_->head_is_pb16i();

    // ---- Gemma4 per-layer embeddings (PLE) + KV sharing ----
    // Every tensor here is read out of the model's own index (name + dtype checked
    // rather than assumed); the shapes are BF16 except the per-layer token table,
    // which is I8 [vocab, NL*ple_dim].
    ple_dim_ = sem_.hidden_size_per_layer_input;
    gemma4_layer_ = (ple_dim_ > 0) || sem_.extra_layer_norms;
    // A model with tie_word_embeddings = false ships a SEPARATE lm_head.weight
    // (packed I8); dequantize it once here.  Nanbeige4.1-3B is such a model and
    // its head is unrelated to the embedding (corr 0.013), so decoding logits
    // from the embedding returns '的' instead of 'Paris'.
    if (!sem_.tie_embeddings) {
        TensorDesc* ltd = model_tensor_by_name(mw_, "lm_head.weight");
        if (ltd) {
            lm_head_f_.assign((size_t)NV * (size_t)kpad_, 0.0f);
            lm_w_ = kpad_;
            const uint8_t* lp = (const uint8_t*)model_tensor_data(mw_, ltd);
            if (!lp || dequant_any(lp, (size_t)ltd->data_size, NV, kpad_,
                                   lm_head_f_.data()) != 0) {
                err_ = "lm_head dequant failed";
                return false;
            }
            fprintf(stderr, "  [forward] separate lm_head.weight -> dequantized "
                            "(%zu floats)\n", lm_head_f_.size());
        }
    }
    // Norms are loaded HERE, once the semantics and ple_dim_/gemma4_layer_ are
    // known: the Gemma4 extras (pre/post feedforward norms, post-PLE norm, layer
    // scalar) are only read when gemma4_layer_ is set, so loading them before the
    // config was parsed left those vectors empty and crashed the first step.
    if (!load_norms()) return false;
    // qwen3_5 declares one model_type for two RMSNorm conventions and this box carries one of each (the
    // official 0.8B release is zero-centered; the hybrid 27B packs are plain gains). Resolve it from the
    // weight that was just loaded, and print the evidence so a surprising answer is visible rather than
    // buried. Every other model_type keeps the convention it already decided above.
    if (sem_.rmsnorm_offset_probe && !in_norm_.empty() && c_.H > 0) {
        const std::vector<float>& w0 = in_norm_[0];
        double sum = 0.0;
        float mn = w0[0];
        for (int i = 0; i < c_.H; i++) {
            sum += (double)w0[(size_t)i];
            if (w0[(size_t)i] < mn) mn = w0[(size_t)i];
        }
        sem_.rmsnorm_offset = q4nx_rmsnorm_offset_from_norm(w0.data(), c_.H);
        fprintf(stderr,
                "  [semantics] rmsnorm_offset=%d  <- model.layers.0.input_layernorm.weight: mean %.4f "
                "min %.4f (qwen3_5 ships both the zero-centered (1+w) form and the plain w form; the "
                "weight decides)\n",
                sem_.rmsnorm_offset, sum / (double)c_.H, (double)mn);
    }
    if (ple_dim_ > 0) {
        const int H = c_.H, NL = c_.NL;
        auto read_named = [&](const char* nm, float* dst, int n) -> bool {
            TensorDesc* td = model_tensor_by_name(mw_, nm);
            if (!td || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, td, (size_t)n);
            if (!p) return false;
            for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
            return true;
        };

        ple_norm_.assign((size_t)ple_dim_, 0.0f);
        if (!read_named("model.per_layer_proj_norm.weight", ple_norm_.data(), ple_dim_)) {
            err_ = "missing model.per_layer_proj_norm.weight";
            return false;
        }
        // per_layer_model_proj.weight_layer{l} is [ple_dim][H], stored
        // ROW-GROUPED as [ple_dim/32, H, 32] (the converter's
        // reshape_matrix_to_block_matrix_for_mvm); stacked over the layers it is
        // the full [NL*ple_dim][H] context projection.
        ple_model_proj_.assign((size_t)NL * (size_t)ple_dim_ * (size_t)H, 0.0f);
        for (int l = 0; l < NL; l++) {
            char nm[160];
            snprintf(nm, sizeof(nm), "model.per_layer_model_proj.weight_layer%d", l);
            TensorDesc* td = model_tensor_by_name(mw_, nm);
            if (!td) { err_ = std::string("missing ") + nm; return false; }
            const uint16_t* p = bf16_data(mw_, td, (size_t)ple_dim_ * (size_t)H);
            if (!p || (ple_dim_ % 32) != 0) {
                err_ = std::string("bad ") + nm;
                return false;
            }
            float* dst = &ple_model_proj_[(size_t)l * ple_dim_ * H];
            for (int a = 0; a < ple_dim_ / 32; a++)
                for (int c = 0; c < H; c++)
                    for (int g = 0; g < 32; g++)
                        dst[(size_t)(a * 32 + g) * H + c] =
                            bf16_to_f32(p[((size_t)a * H + c) * 32 + g]);
        }
        ple_input_.assign((size_t)NL * (size_t)ple_dim_, 0.0f);

        // KV sharing: layers from (NL - num_kv_shared_layers) on carry no k/v
        // weights and reuse the K/V of the last non-shared layer of their own
        // attention type (HF's `store_full_length_kv`).
        const int first_shared = NL - sem_.num_kv_shared_layers;
        kv_owner_.assign((size_t)NL, 0);
        for (int l = 0; l < NL; l++) {
            if (l < first_shared || first_shared <= 0) { kv_owner_[l] = l; continue; }
            const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
            int owner = l;
            for (int p = first_shared - 1; p >= 0; p--) {
                const int pt = (p < sem_.n_layer_types) ? sem_.layer_types[p]
                                                       : Q4NX_LAYER_FULL;
                if (pt == lt) { owner = p; break; }
            }
            kv_owner_[l] = owner;
        }
        fprintf(stderr, "  [forward] gemma4 PLE: ple_dim=%d, %d layers KV-shared "
                        "(from layer %d), double_wide_mlp=%d\n",
                ple_dim_, sem_.num_kv_shared_layers, first_shared, sem_.double_wide_mlp);
    }

    // engine#587: resolve the dense-tail cache once the config is known, and say
    // which way it went, so a measurement can tell "off" from "never asked".
    if (ffn_cache_ < 0) ffn_cache_ = ffn_cache_default() ? 1 : 0;
    std::fprintf(stderr, "[ffn] dense-tail cache %s (%s; lin_kd=%d IM=%d H=%d)\n",
                 ffn_cache_ ? "on" : "off",
                 getenv("NPU_FFN_CACHE") ? "NPU_FFN_CACHE" : "default",
                 lin_kd_, c_.IM, c_.H);

    // The target gets the plan now that the geometry it launches on is final: it checks the two against
    // each other and refuses by name when they disagree (P4 consuming P1's loader, docs/npu-p4-target-split.md
    // §3 A1).  Host-GEMM mode has no target to open - that path is the model's own fallback.
    if (!host_gemm_ && !xd_->open(plan_, dir, family_tag(), err_)) return false;

    backend_ = "full ELF";
    ready_ = true;
    /* engine#1900: say the batch width once at load, the way the block note does, so a log can be told apart from a
     * forward that never batched. #1904 gives the sealed route its plan's width; this line is what makes the difference
     * visible either way. */
    std::fprintf(stderr, "  [forward] batch width %d row(s) per launch\n", max_batch_rows());
    return true;
}

float Q4nxNpuForward::embed_at(int token, int k) const {
    if (token < 0 || token >= c_.NV || k < 0 || k >= c_.H) return 0.0f;
    // engine#1784: the two dtypes that can be read in place, without a table-sized buffer, are read in place. Pad
    // columns (k >= c_.H) are already refused above and read as 0.0f, which is what the expanded table held there.
    if (embed_i8_)
        return q4nx_plain_i8_at((const uint8_t*)embed_i8_, embed_sc_, c_.NV, c_.H, token, k);
    if (embed_tile_) {
        // One row per token: the callers walk k innermost, so this gathers the row once and serves the H reads from it.
        if (token != embed_row_tok_) {
            if (q4nx_dequant_tile_row(embed_tile_, embed_tile_bytes_, embed_tile_rows_, embed_tile_cols_,
                                      embed_tile_cb_, token, embed_row_buf_.data()) != 0)
                return 0.0f;
            embed_row_tok_ = token;
        }
        return embed_row_buf_[(size_t)k];
    }
    if (embed_bf16_) return bf16_to_f32(embed_bf16_[(size_t)token * (size_t)c_.H + (size_t)k]);
    if (embed_f32_) return embed_f_[(size_t)token * (size_t)lm_w_ + (size_t)k];
    return 0.0f;
}

/* LM-head weight for output n, column k: the separate lm_head.weight when the
 * model does not tie it, otherwise the (pre-scaled) token embedding. */
float Q4nxNpuForward::lm_at(int n, int k) const {
    if (n < 0 || n >= c_.NV || k < 0 || k >= c_.H) return 0.0f;
    if (!lm_head_f_.empty())
        return lm_head_f_[(size_t)n * (size_t)lm_w_ + (size_t)k];
    return embed_at(n, k);
}

bool Q4nxNpuForward::load_norms() {
    const int H = c_.H, HD = c_.HD, NL = c_.NL;
    // Gemma4's q/k norms are per-head and sized by THAT layer type's head_dim
    // (256 sliding / 512 full), so every slot is sized for the widest type and
    // each layer reads only its own head_dim.
    const int max_hd = (sem_.global_head_dim > c_.HD) ? sem_.global_head_dim : c_.HD;
    in_norm_.assign(NL, std::vector<float>(H));
    post_norm_.assign(NL, std::vector<float>(H));
    q_norm_.assign(NL, std::vector<float>(max_hd));
    k_norm_.assign(NL, std::vector<float>(max_hd));
    const int max_nkv = (sem_.num_global_kv_heads > c_.NKV) ? sem_.num_global_kv_heads : c_.NKV;
    q_bias_.assign(NL, std::vector<float>((size_t)c_.NH * max_hd));
    k_bias_.assign(NL, std::vector<float>((size_t)max_nkv * max_hd));
    v_bias_.assign(NL, std::vector<float>((size_t)max_nkv * max_hd));
    final_norm_.assign(H, 0.0f);

    auto read_bf16 = [&](const TensorDesc* d, float* dst, int n) -> bool {
        if (n < 0) return false;
        // A tensor the model does not have (no name, no data) reads as zeros, as before
        // it read whatever sat at the start of the data; its slot is unused.
        if (d->name[0] == '\0' && d->data_size == 0) {
            std::fill(dst, dst + n, 0.0f);
            return true;
        }
        const uint16_t* p = bf16_data(mw_, d, (size_t)n);
        if (!p) {
            err_ = std::string("tensor ") + d->name + " is smaller than config.json's dims";
            return false;
        }
        for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
        return true;
    };

    auto absent = [](const TensorDesc* d) {
        return d->name[0] == '\0' || d->data_size == 0;
    };
    for (int l = 0; l < NL; l++) {
        int hd_l = HD, nkv_l = c_.NKV;
        layer_attn_dims(l, hd_l, nkv_l);
        if (!read_bf16(&mw_->layers[l].input_layernorm_weight, in_norm_[l].data(), H)) return false;
        if (!read_bf16(&mw_->layers[l].post_attention_layernorm_weight, post_norm_[l].data(), H)) return false;
        if (!read_bf16(&mw_->layers[l].q_norm_weight, q_norm_[l].data(), hd_l)) return false;
        // Qwen2.5-style attention biases (q/k/v projection biases); absent on the
        // bias-free families, so a miss is not an error.  Loaded BEFORE the k_norm
        // continue below, or a k_norm-free model (Qwen2.5, Llama) would skip them.
        if (!absent(&mw_->layers[l].q_proj_bias))
            read_bf16(&mw_->layers[l].q_proj_bias, q_bias_[l].data(), (int)c_.NH * hd_l);
        if (!absent(&mw_->layers[l].k_proj_bias))
            read_bf16(&mw_->layers[l].k_proj_bias, k_bias_[l].data(), nkv_l * hd_l);
        if (!absent(&mw_->layers[l].v_proj_bias))
            read_bf16(&mw_->layers[l].v_proj_bias, v_bias_[l].data(), nkv_l * hd_l);
        // Gemma4 KV-shared layers carry no k_norm at all; that is structural, not
        // an error, so it must not fail the load.
        if (absent(&mw_->layers[l].k_norm_weight)) continue;
        if (!read_bf16(&mw_->layers[l].k_norm_weight, k_norm_[l].data(), hd_l)) return false;
    }
    if (!read_bf16(&mw_->norm_weight, final_norm_.data(), H)) return false;

    // ---- Gemma4 extras: two feedforward norms, the post-PLE norm, the layer
    // scalar, and the per-layer gate/projection weights (all BF16). ----
    if (gemma4_layer_) {
        pre_ff_norm_.assign(NL, std::vector<float>(H));
        post_ff_norm_.assign(NL, std::vector<float>(H));
        post_ln_.assign(NL, std::vector<float>(H));
        layer_scale_.assign(NL, 1.0f);
        inp_gate_.assign(NL, std::vector<float>());
        ple_proj_.assign(NL, std::vector<float>());
        for (int l = 0; l < NL; l++) {
            char nm[160];
            auto rd = [&](const char* fmt, float* dst, int n) -> bool {
                snprintf(nm, sizeof(nm), fmt, l);
                TensorDesc* td = model_tensor_by_name(mw_, nm);
                if (!td || n < 0) return false;
                const uint16_t* p = bf16_data(mw_, td, (size_t)n);
                if (!p) return false;
                for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(p[i]);
                return true;
            };
            // The PLE gate / projection are stored ROW-GROUPED as
            // [rows/32, cols, 32] (the converter's
            // reshape_matrix_to_block_matrix_for_mvm), so a flat read would
            // scramble every row: logical(a*32+g, c) = stored(a, c, g).
            auto rd_grouped = [&](const char* fmt, float* dst, int rows, int cols) -> bool {
                snprintf(nm, sizeof(nm), fmt, l);
                TensorDesc* td = model_tensor_by_name(mw_, nm);
                if (!td || rows < 0 || cols < 0) return false;
                const uint16_t* p = bf16_data(mw_, td, (size_t)rows * (size_t)cols);
                if (!p || (rows % 32) != 0) return false;
                for (int a = 0; a < rows / 32; a++)
                    for (int c = 0; c < cols; c++)
                        for (int g = 0; g < 32; g++)
                            dst[(size_t)(a * 32 + g) * cols + c] =
                                bf16_to_f32(p[((size_t)a * cols + c) * 32 + g]);
                return true;
            };
            if (sem_.extra_layer_norms) {
                if (!rd("model.layers.%d.pre_feedforward_layernorm.weight",
                        pre_ff_norm_[l].data(), H)) return false;
                if (!rd("model.layers.%d.post_feedforward_layernorm.weight",
                        post_ff_norm_[l].data(), H)) return false;
                if (!rd("model.layers.%d.post_layernorm.weight", post_ln_[l].data(), H)) return false;
                if (!rd("model.layers.%d.layer_output_scale.weight", &layer_scale_[l], 1)) return false;
            }
            if (ple_dim_ > 0) {
                inp_gate_[l].assign((size_t)ple_dim_ * (size_t)H, 0.0f);
                ple_proj_[l].assign((size_t)H * (size_t)ple_dim_, 0.0f);
                if (!rd_grouped("model.layers.%d.inp_gate.weight", inp_gate_[l].data(),
                                ple_dim_, H)) return false;
                if (!rd_grouped("model.layers.%d.per_layer_projection.weight",
                                ple_proj_[l].data(), H, ple_dim_)) return false;
            }
        }
    }
    return true;
}

void Q4nxNpuForward::layer_attn_dims(int l, int& hd, int& nkv) const {
    const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
    if (lt != Q4NX_LAYER_SLIDING && sem_.global_head_dim > 0) {
        hd = sem_.global_head_dim;
        nkv = (sem_.num_global_kv_heads > 0) ? sem_.num_global_kv_heads : c_.NKV;
    } else {
        hd = c_.HD;
        nkv = c_.NKV;
    }
}

// engine#587: with NPU_FFN_CACHE unset the cache is on where it is safe and worth
// it -- a GDN model (lin_kd_ > 0, the one case step()'s `dense_weights_cached`
// can never cover, since that guard stops at lin_kd_ == 0), packed weights
// actually being cached, and a dense tail big enough for the per-token reload to
// be material.  Qwen3.8-27B-NPU2 (IM 17408 x H 5120) clears it with 1.07 GB per
// layer, and measured 117.4 -> 104.5 s wall and 63.3 -> 22.9 s CPU per token on
// an 8-token request with identical text (~/lb/587-cache-20261030).
// The floor is a size guard, not a second measurement: below it the reload is
// small enough that the behaviour change is not worth taking unasked.
bool Q4nxNpuForward::ffn_cache_default() const {
    if (host_gemm_ || !xd_->weight_cache_on() || lin_kd_ <= 0) return false;
    const double tail_bytes = 3.0 * (double)c_.IM * (double)c_.H * 4.0;   // wg_+wu_+wd_
    return tail_bytes >= 256.0 * 1024.0 * 1024.0;
}
// ---- layer-loop host attribution ------------------------------------------
// The `host=` residual of the [ph] line is layer time that is NOT any of the
// gemm phases: 368 launches x ~6.5 ms with no timer on it.  These named
// per-call-site scopes sum it back up so it can be ranked.  Purely additive:
// nothing in the compute path reads any of this.

// The three f32 FFN buffers (wg_/wu_/wd_) are read only by q4nx_assemble_B(),
// which the MLP sections of step()/step_batch() skip once that layer's packed
// int8 weight is on the device (weight_cached(2|3|4)) -- the same argument as
// proj_cached in load_gdn_layer, which skips the three GDN projection deqs and
// leaves ~1 GB/layer/token of FFN dequant in place.
//
// step() never satisfies `dense_weights_cached` for a hybrid GDN model
// (lin_kd_ > 0 is model-level), so a GDN model re-dequantised the 17408-wide
// dense tail of all 64 layers on every token: 3 * 17408 * 5120 * 4 B = 1.07 GB
// per layer, ~68 GB of f32 writes per token.  Gating on the packed-BO state
// rather than assuming it keeps the first use of every layer correct, and a
// host-GEMM run (which never packs) keeps its legacy behaviour.
//
// Keys: down is always 4; a fused gate/up design packs gate+up as key 2
// (N = 2*IM), a split one as gate 2 + up 3.  Require 3 only when the model has
// the split designs, so the guard is not silently dead on a fused-GU model.
bool Q4nxNpuForward::ffn_tail_cached() const {
    if (!ffn_cache_ || host_gemm_) return false;
    if (!weight_cached(2) || !weight_cached(4)) return false;
    if (idx_of(Q4NX_DESIGN_G) >= 0 && idx_of(Q4NX_DESIGN_U) >= 0)
        return weight_cached(3);
    return true;
}

bool Q4nxNpuForward::load_layer(int l) {
    // MLA + MoE (deepseek2) has an entirely different tensor set.
    if (mla_) {
        const int H = c_.H, NH = c_.NH;
        const int qh = qk_nope_ + rope_dim_;
        const int kv_real = kv_lora_ + rope_dim_;
        char nm[160];
        auto td_of = [&](const char* fmt) -> TensorDesc* {
            snprintf(nm, sizeof(nm), fmt, l);
            return model_tensor_by_name(mw_, nm);
        };
        // A 2D packed tensor -> [rows][cols] floats.
        auto load2 = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { fprintf(stderr, "  [mla] missing " ); snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr,"%s\n", nm); return false; }
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        // A bf16 tensor (norms, router) is stored raw, not in 5120-byte tiles, so
        // it bypasses dequant_any.
        auto load_bf16 = [&](const char* fmt, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm, sizeof(nm), fmt, l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint16_t* p = n < 0 ? nullptr : bf16_data(mw_, d, (size_t)n);
            if (!p) { snprintf(nm, sizeof(nm), fmt, l); fprintf(stderr, "  [mla] %s is too small\n", nm); return false; }
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load2("model.layers.%d.self_attn.q_a_proj.weight", w_qa_, q_lora_, H)) return false;
        if (!load2("model.layers.%d.self_attn.q_b_proj.weight", w_qb_, NH * qh, q_lora_)) return false;
        if (!load2("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", w_kva_, kv_real, H)) return false;
        if (!load2("model.layers.%d.self_attn.o_proj.weight", w_o_, H, NH * v_head_)) return false;
        if (!load_bf16("model.layers.%d.self_attn.q_a_layernorm.weight", mla_qan_, q_lora_)) return false;
        if (!load_bf16("model.layers.%d.self_attn.kv_a_layernorm.weight", mla_kvan_, kv_lora_)) return false;
        // k_b/v_b are pooled per head: [NH][tiles][5120].
        auto load_heads3 = [&](const char* fmt, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            const size_t per = (size_t)d->data_size / (size_t)NH;
            v.resize((size_t)NH * (size_t)rows * (size_t)cols);
            for (int h = 0; h < NH; h++)
                if (dequant_any(raw + (size_t)h * per, per, rows, cols,
                                &v[(size_t)h * (size_t)rows * (size_t)cols]) != 0) return false;
            return true;
        };
        if (!load_heads3("model.layers.%d.self_attn.k_b_proj.weight", w_kb_, kb_rows_, kv_lora_)) return false;
        // v_b's Q4NX grid is [kv_lora][v_head] = [512][256] (NOT [256][512]).  Pinned
        // empirically against llama.cpp's own tensor dump: reading it as a [512][256]
        // grid and indexing flat[k*vh + n] reproduces the oracle's layer-0 post-attention
        // sum to 5 decimals (-0.227843 vs -0.228092), where the [256][512] grid gave
        // -0.454611.  The tile GRID (not just the flat order) matters: [512][256] and
        // [256][512] have the same tile count but a different tile-to-element mapping.
        if (!load_heads3("model.layers.%d.self_attn.v_b_proj.weight", w_vb_, kv_lora_, v_head_)) return false;
        // Router + bias (bf16).
        // Router, routed experts and the shared expert exist ONLY on the MoE
        // layers: with leading_dense_block_count = 1 the leading dense layer(s)
        // carry a plain MLP instead (46 of GLM-4.7-Flash's 47 layers have them).
        if (l >= leading_dense_) {
        if (!load_bf16("model.layers.%d.moe_router.weight", moe_router_, n_expert_ * H)) return false;
        if (!load_bf16("model.layers.%d.moe_router.bias", moe_router_b_, n_expert_)) return false;
        // Routed experts stay packed until selected.  The Q4NX pools them per
        // layer as [n_expert][tiles][5120], so hold the pointer plus the
        // per-expert byte count and dequant one slice at a time.
        auto pool = [&](const char* fmt, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td_of(fmt);
            if (!d) { snprintf(nm,sizeof(nm),fmt,l); fprintf(stderr, "  [mla] missing %s\n", nm); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw;
            *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool("model.layers.%d.mlp.gate_exps_proj.weight", &exp_g_, &exp_g_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.up_exps_proj.weight", &exp_u_, &exp_u_bytes_)) return false;
        if (!pool("model.layers.%d.mlp.down_exps_proj.weight", &exp_d_, &exp_d_bytes_)) return false;
        // Shared expert ([tiles][5120], a plain 2D [im][H] / [H][im] matrix).
        if (!load2("model.layers.%d.mlp.share_gate_exps_proj.weight", share_g_, expert_im_, H)) return false;
        if (!load2("model.layers.%d.mlp.share_up_exps_proj.weight", share_u_, expert_im_, H)) return false;
        if (!load2("model.layers.%d.mlp.share_down_exps_proj.weight", share_d_, H, expert_im_)) return false;
        }   // end MoE-layer tensors
        // The leading dense layer has no experts.
        if (l < leading_dense_) {
            if (!load2("model.layers.%d.mlp.gate_proj.weight", dense_g_, dense_im_, H)) return false;
            if (!load2("model.layers.%d.mlp.up_proj.weight", dense_u_, dense_im_, H)) return false;
            if (!load2("model.layers.%d.mlp.down_proj.weight", dense_d_, H, dense_im_)) return false;
        }
        return true;
    }
    // A GDN (qwen3_5 linear_attention) layer has a different tensor set.
    if (lin_kd_ > 0 && l < sem_.n_layer_types &&
        sem_.layer_types[l] == Q4NX_LAYER_LINEAR)
        return load_gdn_layer(l);
    auto deq = [&](TensorDesc* d, float* out, int rows, int cols) -> bool {
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw) { err_ = std::string("MLP tensor ") + d->name + " has no data"; return false; }
        if (dequant_any(raw, (size_t)d->data_size, rows, cols, out) != 0) {
            // Name it: the bare "dequant failed" this used to leave cost an afternoon on engine#1097.
            const size_t tiles = (size_t)(rows / Q4NX_TILE_ROWS) * (size_t)(cols / Q4NX_TILE_COLS);
            char why[256];
            snprintf(why, sizeof why, "MLP dequant %s: data_size %zu at geometry %dx%d = %zu tiles "
                     "(chunk %zu B is not a kind, and no padded grid fits): the container and "
                     "config.json disagree", d->name, (size_t)d->data_size, rows, cols, tiles,
                     tiles ? (size_t)d->data_size / tiles : 0);
            err_ = why;
            return false;
        }
        return true;
    };
    const int H = c_.H, IM = c_.IM, NH = c_.NH;
    // Gemma4 varies the attention width by layer type, so the projection rows
    // come from THIS layer's head_dim (the scratch buffers are sized for the
    // widest type at init).
    int hd = 0, nkv = 0;
    layer_attn_dims(l, hd, nkv);
    LayerWeights* lw = &mw_->layers[l];
    // A Gemma4 KV-shared layer has no k/v weights and reuses another layer's cache.
    const bool kv_shared = (l < (int)kv_owner_.size() && kv_owner_[l] != l);
    // Gemma4 with use_double_wide_mlp runs a 2*IM-wide MLP on the shared layers.
    const bool wide = kv_shared && sem_.double_wide_mlp;
    const int im = wide ? 2 * IM : IM;
    if (wide) {
        wg_.resize((size_t)im * (size_t)kpad_);
        wu_.resize((size_t)im * (size_t)kpad_);
        wd_.resize((size_t)H * (size_t)q4nx_pad_k(im));
    }
    // A pb16i container reads its decode weights from the sidecar images, so the f32
    // copies of the big projections are never needed: skipping them here is the
    // difference between a first token that dequantises the whole 17.9 GB model and
    // one that opens 13.4 GB of images.  The norms and every small member still load.
    if (!xd_->pb16i()) {
        if (!deq(&lw->q_proj_weight, wq_.data(),
                 NH * hd * (attn_gate_ ? 2 : 1), kpad_)) return false;   // q|gate rows
        if (!kv_shared) {
            if (!deq(&lw->k_proj_weight, wk_.data(), nkv * hd, kpad_)) return false;
            if (!deq(&lw->v_proj_weight, wv_.data(), nkv * hd, kpad_)) return false;
        }
        if (!deq(&lw->o_proj_weight, wo_.data(), H, q4nx_pad_k(NH * hd))) return false;
    }
    if (n_expert_ > 0) {
        // Qwen MoE layer (qwen3moe / qwen35moe): a router + top-k routed experts
        // instead of a dense gate/up/down.  The router is bf16; its bias is OPTIONAL
        // (Qwen MoE has none, deepseek2 does).  The experts stay pooled -- one
        // [N][K] slice is dequantized per selected expert at decode.
        // Two container conventions spell these tensors differently (engine#1035, moe_container.h):
        // the engine converter's `mlp.experts.*_proj.weight` + `mlp.gate.weight` with a plain
        // [N][K] per-expert grid, or the FLM-style `*_exps_proj.weight` + `moe_router.weight`.
        // Which one this container is was read from layer 0 in init_impl, so the names come from
        // one table rather than being assumed here.
        const MoeNames mn = moe_names(moe_format_);
        char nm2[160];
        auto td_named = [&](const char* suffix) -> TensorDesc* {
            snprintf(nm2, sizeof(nm2), "model.layers.%d.%s", l, suffix);
            return model_tensor_by_name(mw_, nm2);
        };
        auto load_bf16_named = [&](const char* suffix, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, d, (size_t)n);
            if (!p) return false;
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load_bf16_named(mn.router, moe_router_, n_expert_ * H)) { err_ = "moe router"; return false; }
        if (!load_bf16_named(mn.router_b, moe_router_b_, n_expert_)) moe_router_b_.clear();
        auto pool = [&](const char* suffix, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d) { fprintf(stderr, "  [moe] missing %s\n", nm2); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw; *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool(mn.gate, &exp_g_, &exp_g_bytes_)) return false;
        if (!pool(mn.up, &exp_u_, &exp_u_bytes_)) return false;
        if (!pool(mn.down, &exp_d_, &exp_d_bytes_)) return false;
        // Optional shared expert (qwen35moe): a 2D [expert_im][H] / [H][expert_im] pair
        // plus its [H] gate.  Cleared when absent (qwen3moe has none).
        auto load_share = [&](const char* suffix, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d) return false;
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        if (!load_share(mn.share_gate, share_g_, expert_im_, H)) share_g_.clear();
        if (!load_share(mn.share_up, share_u_, expert_im_, H)) { share_g_.clear(); share_u_.clear(); }
        if (!load_share(mn.share_down, share_d_, H, expert_im_)) { share_g_.clear(); share_u_.clear(); }
        if (!load_bf16_named(mn.share_router, share_router_, H)) share_router_.clear();
        return true;
    }
    if (!xd_->pb16i() && !ffn_tail_cached()) {
        LhScope _t_f(LH_LOAD_FFN);
        if (!deq(&lw->gate_proj_weight, wg_.data(), im, kpad_)) return false;
        if (!deq(&lw->up_proj_weight, wu_.data(), im, kpad_)) return false;
        if (!deq(&lw->down_proj_weight, wd_.data(), H, q4nx_pad_k(im))) return false;
    }
    return true;
}

// ── GDN (qwen3_5 linear_attention) ─────────────────────────────────────────
// Weight layouts are in docs/qwen35-4b-weight-format.md; the maths mirror
// tools/q4nx_forward_ref.py's forward_qwen35 (which predicts ' Paris').
bool Q4nxNpuForward::load_gdn_layer(int l) {
    const int H = c_.H;
    const int CD = 2 * lin_kd_ + lin_vd_;
    char nm[192];
    auto tensor = [&](const char* suffix) -> TensorDesc* {
        snprintf(nm, sizeof(nm), "model.layers.%d.%s", l, suffix);
        return model_tensor_by_name(mw_, nm);
    };
    auto deq = [&](const char* suffix, float* out, int rows, int cols) -> bool {
        TensorDesc* d = tensor(suffix);
        if (!d) { err_ = std::string("GDN: no tensor ") + nm; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw) { err_ = std::string("GDN: no data ") + nm; return false; }
        if (dequant_any(raw, (size_t)d->data_size, rows, cols, out) != 0) {
            char m[64];
            snprintf(m, sizeof(m), " (rows=%d cols=%d bytes=%llu)", rows, cols,
                     (unsigned long long)d->data_size);
            err_ = std::string("GDN: dequant ") + nm + m;
            return false;
        }
        return true;
    };
    // alpha/beta are stored row-padded to the 32-row tile (48 heads -> 64 rows,
    // Q8: 40 tiles x 8704 B = 348160 B), so dequantise at the stored height and
    // keep the first `rows`.  Short-circuits when rows already tile-align.
    auto deq_pad = [&](const char* suffix, float* out, int rows, int cols) -> bool {
        if (rows % Q4NX_TILE_ROWS == 0) return deq(suffix, out, rows, cols);
        const int prows = ((rows + Q4NX_TILE_ROWS - 1) / Q4NX_TILE_ROWS) * Q4NX_TILE_ROWS;
        std::vector<float> tmp((size_t)prows * (size_t)cols);
        if (!deq(suffix, tmp.data(), prows, cols)) return false;
        memcpy(out, tmp.data(), (size_t)rows * (size_t)cols * sizeof(float));
        return true;
    };
    auto bf16v = [&](const char* suffix, float* out, int n) -> bool {
        TensorDesc* d = tensor(suffix);
        if (!d) { err_ = std::string("GDN: no tensor ") + nm; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw || (size_t)d->data_size < (size_t)n * 2) {
            err_ = std::string("GDN: bad bf16 ") + nm;
            return false;
        }
        for (int i = 0; i < n; i++) out[i] = bf16_to_f32(((const uint16_t*)raw)[i]);
        return true;
    };
    // The three big GDN projections are only ever read through q4nx_assemble_B()
    // when their packed BO is not cached yet, so re-dequantising them on every
    // token is pure waste (~1.5 GB per layer per token).  The small members below
    // (alpha/beta, conv, norm, ssm_a/dt) are used on the host every token and are
    // still refreshed.
    // A pb16i container never assembles B for these three at all: their weights arrive
    // as images, so the f32 copies are dead from the first token.
    const bool proj_cached = xd_->pb16i() ||
                             (xd_->weight_cache_on() && weight_cached(0) && weight_cached(1) &&
                              weight_cached(6));
    LhScope _t_ldg(LH_LOAD_GDN_SMALL);
    if (!proj_cached) {
        LhScope _t_p(LH_LOAD_PROJ);
        if (!deq("linear_attn.qkv_proj.weight", wqkvlin_.data(), CD, H)) return false;
        if (!deq("self_attn.gate_proj.weight", wz_.data(), lin_vd_, H)) return false;
        if (!deq("linear_attn.ssm_out_proj.weight", woutlin_.data(), H, lin_vd_)) return false;
    }
    _t_ldg.stop();     // the three projections are done; the small members follow
    GdnSmall& gs = gdn_small_[(size_t)l];
    if (!gdn_cache_ || !gs.loaded) {
    if (!deq_pad("linear_attn.ssm_alpha_proj.weight", gs.wa.data(), lin_nvh_, H)) return false;
    if (!deq_pad("linear_attn.ssm_beta_proj.weight", gs.wb.data(), lin_nvh_, H)) return false;
    if (!bf16v("linear_attn.ssm_conv1d.weight", gs.conv.data(), lin_ck_ * CD)) return false;
    if (!bf16v("linear_attn.ssm_norm.weight", gs.norm.data(), lin_vhd_)) return false;
    {
        TensorDesc* da = tensor("linear_attn.ssm_a");
        TensorDesc* dd = tensor("linear_attn.ssm_dt.bias");
        if (!da || !dd) { err_ = std::string("GDN: no ssm_a/ssm_dt ") + nm; return false; }
        if ((size_t)da->data_size < (size_t)lin_nvh_ * 4 ||
            (size_t)dd->data_size < (size_t)lin_nvh_ * 4) {
            err_ = std::string("GDN: bad ssm_a/ssm_dt size ") + nm;
            return false;
        }
        memcpy(gs.ssm_a.data(), model_tensor_data(mw_, da), (size_t)lin_nvh_ * 4);
        memcpy(gs.dt.data(), model_tensor_data(mw_, dd), (size_t)lin_nvh_ * 4);
    }
    gs.loaded = true;
    }
    // The MLP runs on EVERY layer, but this path replaces load_layer()'s normal
    // tail for GDN layers -- without these three the MLP silently reused the
    // previous layer's (or uninitialised) weights, which is why the GDN
    // intermediates matched the reference exactly while the layer OUTPUT did not.
    LayerWeights* lw = &mw_->layers[l];
    auto deqT = [&](TensorDesc* d, float* out, int rows, int cols) -> bool {
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        if (!raw || d->data_size == 0) { err_ = std::string("GDN: missing MLP tensor ") + d->name; return false; }
        if (dequant_any(raw, (size_t)d->data_size, rows, cols, out) != 0) {
            err_ = std::string("GDN: MLP dequant ") + d->name;
            return false;
        }
        return true;
    };
    if (n_expert_ > 0) {
        // qwen35moe: a GDN layer's FFN is MoE, not a dense gate/up/down.  The
        // loading mirrors the standard path's MoE branch; the FFN itself is run by
        // the common MLP section (which sees n_expert_ > 0 and calls moe_mlp).
        // Same two conventions as the standard MoE branch (moe_container.h, engine#1035).
        const MoeNames mn = moe_names(moe_format_);
        char nm2[160];
        auto td_named = [&](const char* suffix) -> TensorDesc* {
            snprintf(nm2, sizeof(nm2), "model.layers.%d.%s", l, suffix);
            return model_tensor_by_name(mw_, nm2);
        };
        auto load_bf16_named = [&](const char* suffix, std::vector<float>& v, int n) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d || n < 0) return false;
            const uint16_t* p = bf16_data(mw_, d, (size_t)n);
            if (!p) return false;
            v.resize((size_t)n);
            for (int i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
            return true;
        };
        if (!load_bf16_named(mn.router, moe_router_, n_expert_ * H)) { err_ = "moe router"; return false; }
        if (!load_bf16_named(mn.router_b, moe_router_b_, n_expert_)) moe_router_b_.clear();
        auto pool = [&](const char* suffix, const uint8_t** p, size_t* bytes) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d) { fprintf(stderr, "  [moe] missing %s\n", nm2); return false; }
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            if (!raw) return false;
            *p = raw; *bytes = (size_t)d->data_size / (size_t)n_expert_;
            return true;
        };
        if (!pool(mn.gate, &exp_g_, &exp_g_bytes_)) return false;
        if (!pool(mn.up, &exp_u_, &exp_u_bytes_)) return false;
        if (!pool(mn.down, &exp_d_, &exp_d_bytes_)) return false;
        // Optional shared expert (qwen35moe): a 2D [expert_im][H] / [H][expert_im] pair
        // plus its [H] gate.  Cleared when absent.
        auto load_share = [&](const char* suffix, std::vector<float>& v, int rows, int cols) -> bool {
            TensorDesc* d = td_named(suffix);
            if (!d) return false;
            v.resize((size_t)rows * (size_t)cols);
            const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
            return raw && dequant_any(raw, (size_t)d->data_size, rows, cols, v.data()) == 0;
        };
        if (!load_share(mn.share_gate, share_g_, expert_im_, H)) share_g_.clear();
        if (!load_share(mn.share_up, share_u_, expert_im_, H)) { share_g_.clear(); share_u_.clear(); }
        if (!load_share(mn.share_down, share_d_, H, expert_im_)) { share_g_.clear(); share_u_.clear(); }
        if (!load_bf16_named(mn.share_router, share_router_, H)) share_router_.clear();
        return true;
    }
    if (!ffn_tail_cached()) {
        LhScope _t_f(LH_LOAD_FFN);
        if (!deqT(&lw->gate_proj_weight, wg_.data(), c_.IM, kpad_)) return false;
        if (!deqT(&lw->up_proj_weight, wu_.data(), c_.IM, kpad_)) return false;
        if (!deqT(&lw->down_proj_weight, wd_.data(), H, impad_)) return false;
    }
    return true;
}

bool Q4nxNpuForward::gdn_layer(int l, const std::vector<float>& xn,
                               std::vector<float>& out) {
    const int H = c_.H;
    const int KD = lin_kd_, VD = lin_vd_, CD = 2 * lin_kd_ + lin_vd_;
    const int NKH = lin_nkh_, NVH = lin_nvh_, KHD = lin_khd_, VHD = lin_vhd_;
    const int CK = lin_ck_;
    const int i_qkv = idx_of(Q4NX_DESIGN_QKVLIN);
    const int i_z = idx_of(Q4NX_DESIGN_Z);
    const int i_o = idx_of(Q4NX_DESIGN_O);
    if (i_qkv < 0 || i_z < 0 || i_o < 0) {
        err_ = "GDN designs missing (QKVLIN/Z/O)";
        return false;
    }
    std::vector<float> qkv, z;
    {
        const Q4nxDesignGeom* g = &designs_[i_qkv];
        Q4nxProjections p{};
        p.ple = wqkvlin_.data();
        if (!xd_->pb16i() && !weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn qkv"; return false; }
        npu_gemm(i_qkv, xn.data(), g->K, g->N, B_.data(), qkv, 0, "attn_qkv", g->real_k);
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_z];
        Q4nxProjections p{};
        p.ple = wz_.data();
        if (!xd_->pb16i() && !weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn z"; return false; }
        npu_gemm(i_z, xn.data(), g->K, g->N, B_.data(), z, 1, "attn_gate", g->real_k);
    }
    // depthwise causal conv1d (kernel CK) + SiLU over CD channels;
    // full = [state (CK-1) | current], new state = full[1:]
    std::vector<float> co(CD);
    float* st = conv_state_[l].data();
    for (int c = 0; c < CD; c++) {
        const float* cs = st + (size_t)c * (CK - 1);
        float s = 0;
        for (int j = 0; j < CK - 1; j++) s += cs[j] * gdn_small_[(size_t)l].conv[(size_t)j * CD + c];
        s += qkv[c] * gdn_small_[(size_t)l].conv[(size_t)(CK - 1) * CD + c];
        co[c] = s / (1.0f + expf(-s));
    }
    for (int c = 0; c < CD; c++) {
        float* cs = st + (size_t)c * (CK - 1);
        for (int j = 0; j + 1 < CK - 1; j++) cs[j] = cs[j + 1];
        if (CK > 1) cs[CK - 2] = qkv[c];
    }
    // alpha/beta on the host (N = NVH < 128, below the GEMM tile width)
    LhScope _t_ab(LH_GDN_AB);
    std::vector<float> ah(NVH), bh(NVH);
    for (int h = 0; h < NVH; h++) {
        const float* ra = gdn_small_[(size_t)l].wa.data() + (size_t)h * H;
        const float* rb = gdn_small_[(size_t)l].wb.data() + (size_t)h * H;
        float sa = 0, sb = 0;
        for (int i = 0; i < H; i++) { sa += xn[i] * ra[i]; sb += xn[i] * rb[i]; }
        ah[h] = sa;
        bh[h] = sb;
    }
    _t_ab.stop();
    // q/k L2-normalised with NVH/NKH head repetition; v per v-head
    const int rep = NVH / NKH;
    std::vector<float> ql((size_t)NVH * KHD), kl((size_t)NVH * KHD), vv((size_t)NVH * VHD);
    for (int h = 0; h < NVH; h++) {
        const int src = h / rep;
        for (int d = 0; d < KHD; d++) ql[(size_t)h * KHD + d] = co[(size_t)src * KHD + d];
        for (int d = 0; d < KHD; d++) kl[(size_t)h * KHD + d] = co[KD + (size_t)src * KHD + d];
        for (int d = 0; d < VHD; d++) vv[(size_t)h * VHD + d] = co[2 * KD + (size_t)h * VHD + d];
    }
    for (int h = 0; h < NVH; h++) {
        float sq = 0, sk = 0;
        for (int d = 0; d < KHD; d++) { sq += ql[(size_t)h*KHD+d]*ql[(size_t)h*KHD+d]; sk += kl[(size_t)h*KHD+d]*kl[(size_t)h*KHD+d]; }
        const float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
        for (int d = 0; d < KHD; d++) { ql[(size_t)h*KHD+d] *= iq; kl[(size_t)h*KHD+d] *= ik; }
    }
    // recurrent gated delta rule, then scale-free gated RMSNorm, then out_proj
    LhScope _t_core(LH_GDN_CORE);
    float* rs = rec_state_[l].data();
    const float scale = 1.0f / sqrtf((float)KHD);
    std::vector<float> core((size_t)NVH * VHD), kvm(VHD);
    for (int h = 0; h < NVH; h++) {
        const float eg = expf(gdn_small_[(size_t)l].ssm_a[h] * softplus_f(ah[h] + gdn_small_[(size_t)l].dt[h]));
        const float beta = 1.0f / (1.0f + expf(-bh[h]));
        float* sh = rs + (size_t)h * KHD * VHD;
        const float* kh = &kl[(size_t)h * KHD];
        const float* qh = &ql[(size_t)h * KHD];
        const float* vh = &vv[(size_t)h * VHD];
        for (int i = 0; i < KHD * VHD; i++) sh[i] *= eg;
        // The two matrix-vector products and the rank-1 update below were written with the
        // KEY index innermost, i.e. every inner step jumped VHD floats in `sh` (a 512-byte
        // stride): 107 ms a token, 21 % of the decode step, at ~1.4 GFLOP/s on a Zen 5 core.
        // The same arithmetic with the VALUE index innermost is contiguous and vectorisable,
        // and it is bit-for-bit the same: each output element still accumulates over kd in
        // ascending order, and the rank-1 update still writes each element once from the same
        // precomputed delta.  Only the loop nest changed.
        float* kvp = kvm.data();
        std::fill(kvp, kvp + VHD, 0.0f);
        for (int kd = 0; kd < KHD; kd++) {
            const float kk = kh[kd];
            const float* row = sh + (size_t)kd * VHD;
            for (int vd = 0; vd < VHD; vd++) kvp[vd] += kk * row[vd];
        }
        for (int vd = 0; vd < VHD; vd++) kvp[vd] = (vh[vd] - kvp[vd]) * beta;   // delta, in place
        for (int kd = 0; kd < KHD; kd++) {
            const float kk = kh[kd];
            float* row = sh + (size_t)kd * VHD;
            for (int vd = 0; vd < VHD; vd++) row[vd] += kk * kvp[vd];
        }
        float* cp = core.data() + (size_t)h * VHD;
        std::fill(cp, cp + VHD, 0.0f);
        for (int kd = 0; kd < KHD; kd++) {
            const float qk = qh[kd];
            const float* row = sh + (size_t)kd * VHD;
            for (int vd = 0; vd < VHD; vd++) cp[vd] += qk * row[vd];
        }
        float ss = 0;
        for (int vd = 0; vd < VHD; vd++) {
            cp[vd] *= scale;
            ss += cp[vd] * cp[vd];
        }
        const float inv = 1.0f / sqrtf(ss / (float)VHD + sem_.eps);
        for (int vd = 0; vd < VHD; vd++) {
            const float zz = z[(size_t)h * VHD + vd];
            core[(size_t)h * VHD + vd] = core[(size_t)h * VHD + vd] * inv *
                gdn_small_[(size_t)l].norm[vd] * (zz / (1.0f + expf(-zz)));
        }
    }
    _t_core.stop();
    if (getenv("NPU_INFER_DUMP_LAYERS")) {
        auto dmp = [&](const char* nm, const float* v, int n) {
            double s = 0;
            float mx = 0;
            for (int i = 0; i < n; i++) { float a = fabsf(v[i]); s += a; if (a > mx) mx = a; }
            fprintf(stderr, "  [G] l=%d %-5s %.6f %.4f\n", l, nm, s / n, mx);
        };
        dmp("qkv", qkv.data(), CD);
        dmp("co", co.data(), CD);
        dmp("z", z.data(), VD);
        dmp("a", ah.data(), NVH);
        dmp("b", bh.data(), NVH);
        dmp("core", core.data(), (int)core.size());
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_o];
        Q4nxProjections p{};
        p.o = woutlin_.data();
        // Key 6, not 2: key 2 is the FFN gate_proj of this same layer (N = IM),
        // so reusing it handed this N = H GEMM a BO packed for the other width.
        if (!xd_->pb16i() && !weight_cached(6) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gdn out"; return false; }
        npu_gemm(i_o, core.data(), g->K, g->N, B_.data(), out, 6, "ssm_out", g->real_k);
    }
    return true;
}


// Batched form of gdn_layer: the three projections run with M rows (one weight read
// per chunk), the conv/recurrence loop runs per row exactly as the per-token path
// does, so the numbers are identical and only the number of submits changes.
// A verify batch will run up to `rows` draft positions from `start_pos`.  Take the one snapshot this needs
// (conv ring + recurrent state per layer; the KV cache needs none -- it is position-indexed) and make room for
// each row's recurrence inputs.  Cheap by construction: the buffers are allocated once and reused.
void Q4nxNpuForward::begin_verify(int rows, int start_pos) {
    verify_pos_ = start_pos;
    verify_rows_ = rows;
    verify_open_ = true;
    if (snap_conv_.size() != conv_state_.size()) snap_conv_ = conv_state_;
    else
        for (size_t l = 0; l < conv_state_.size(); l++) snap_conv_[l] = conv_state_[l];
    if (snap_rec_.size() != rec_state_.size()) snap_rec_ = rec_state_;
    else
        for (size_t l = 0; l < rec_state_.size(); l++) snap_rec_[l] = rec_state_[l];
    row_in_.resize(rec_state_.size());
    for (auto& v : row_in_) v.assign((size_t)rows, GdnRowIn{});
}

// Keep the first `accepted` rows of the verify batch and drop the rest: restore the snapshot, then re-apply the
// accepted rows' recurrence (no projections, no GEMMs), and truncate the KV cache to the accepted prefix.
bool Q4nxNpuForward::commit_verify_rows(int accepted) {
    if (!verify_open_) { err_ = "commit_verify: no verify batch open"; return false; }
    if (accepted < 0 || accepted > verify_rows_) { err_ = "commit_verify: accepted out of range"; return false; }
    // NPU_PHASE_PROF=1: a commit is the per-row half of a verify batch, so its two terms are named separately --
    // the snapshot restore (M-independent) and the recurrence replay (per accepted row, engine#972).
    const bool cv_prof = lh_on();
    const auto cv_t0 = std::chrono::steady_clock::now();
    // NPU_REPLAY_CHECK=1: a FULL accept has nothing to roll back, so the snapshot-restore + replay must land on
    // exactly the state the batch's own recurrence left.  That equality is the invariant the deferred commit rests on
    // (and the one a loop-shape change could break silently).  Captured HERE, before the snapshot is restored under
    // it -- reported once per commit, off by default because the copy is 151 MB at this model's dimensions.
    static const bool replay_check = std::getenv("NPU_REPLAY_CHECK") != nullptr;
    std::vector<std::vector<float>> expect_rec, expect_conv;
    if (replay_check && accepted == verify_rows_ && accepted > 0) {
        expect_rec = rec_state_;
        expect_conv = conv_state_;
    }
    const size_t per = (size_t)c_.NKV * (size_t)c_.HD;
    const size_t keep = (size_t)(verify_pos_ + accepted) * per;
    for (size_t l = 0; l < kcache_.size(); l++) {
        if (!kcache_[l].empty() && kcache_[l].size() > keep) kcache_[l].resize(keep);
        if (!vcache_[l].empty() && vcache_[l].size() > keep) vcache_[l].resize(keep);
    }
    for (size_t l = 0; l < conv_state_.size(); l++) conv_state_[l] = snap_conv_[l];
    for (size_t l = 0; l < rec_state_.size(); l++) rec_state_[l] = snap_rec_[l];
    // The accepted prefix is the only thing a drafter may advance over: drop the rejected rows' hidden here, so
    // that last_hidden_rows() is exactly `accepted` rows starting at verify_pos_.  Leaving all verify_rows_ made
    // the draft loop's own size check fail, and it skipped the drafter advance without a word.
    if (verify_rows_ > 0 && !last_hidden_rows_.empty()) {
        const size_t stride = last_hidden_rows_.size() / (size_t)verify_rows_;
        last_hidden_rows_.resize(stride * (size_t)accepted);
    }
    const auto cv_t1 = std::chrono::steady_clock::now();
    verify_open_ = false;                        // the replay below must not re-record
    if (accepted > 0 && !gdn_apply_rows(0, accepted)) return false;
    if (cv_prof) {
        const auto cv_t2 = std::chrono::steady_clock::now();
        const double restore = std::chrono::duration<double, std::milli>(cv_t1 - cv_t0).count();
        const double replay = std::chrono::duration<double, std::milli>(cv_t2 - cv_t1).count();
        std::fprintf(stderr, "[cv] accepted %2d of %2d: restore %6.1f ms, replay %7.1f ms (%6.1f ms/row)\n",
                     accepted, verify_rows_, restore, replay, replay / (accepted > 0 ? accepted : 1));
    }
    if (!expect_rec.empty()) {
        double maxd = 0.0, scale = 0.0;
        for (size_t l = 0; l < expect_rec.size(); l++) {
            const size_t n = std::min(expect_rec[l].size(), rec_state_[l].size());
            for (size_t i = 0; i < n; i++) {
                const double d = std::fabs((double)rec_state_[l][i] - (double)expect_rec[l][i]);
                if (d > maxd) maxd = d;
                const double a = std::fabs((double)expect_rec[l][i]);
                if (a > scale) scale = a;
            }
        }
        for (size_t l = 0; l < expect_conv.size(); l++) {
            const size_t n = std::min(expect_conv[l].size(), conv_state_[l].size());
            for (size_t i = 0; i < n; i++) {
                const double d = std::fabs((double)conv_state_[l][i] - (double)expect_conv[l][i]);
                if (d > maxd) maxd = d;
            }
        }
        const bool ok_state = maxd <= 1e-6 * (scale > 0.0 ? scale : 1.0);
        std::fprintf(stderr, "[cv] replay-check: full accept of %d rows, max |replay - batch| %.3e (scale %.3e) %s\n",
                     accepted, maxd, scale, ok_state ? "OK" : "MISMATCH");
    }
    for (auto& v : row_in_) v.clear();
    return true;
}

// The recurrence alone, re-applied from the restored snapshot: conv ring, q/k/v normalisation from the stored
// conv input, the decay + rank-1 update.  The output is not recomputed -- the verify pass already produced the
// logits for these rows, and only the STATE has to be made consistent with the accepted prefix.
//
// This is the per-ROW term of a verify batch, so it has to cost like the batch does (engine#972).  It ran the
// pre-#944 shape -- one thread, and the rank-1 update walking the state row with stride VHD:
//
//   for (vd) { for (kd) ssum += sh[kd * VHD + vd] * kh[kd]; }     // every touch a different cache line
//
// which measured 85-95 ms per accepted row on the device (74-153 across arms) and 90 ms on a host probe at these
// dimensions (48 layers x 48 heads x 128x128 state = 144 MB), against 1.9-13.5 ms for the shape gdn_layer_batch()
// uses.  So the loops are exchanged exactly as #944 exchanged them (kd outer, vd inner, contiguous on sh; each
// output element still accumulates over kd in increasing order) and the heads run in the SAME
// one-region-per-replay, head-parallel form as the batch path, so a partial accept costs the batch's own per-row
// work instead of 48 single-threaded strided passes over the state.
bool Q4nxNpuForward::gdn_apply_rows(int from, int to) {
    const int KD = lin_kd_, CD = 2 * lin_kd_ + lin_vd_;
    const int NKH = lin_nkh_, NVH = lin_nvh_, KHD = lin_khd_, VHD = lin_vhd_;
    const int CK = lin_ck_;
    const int rep = (NKH > 0) ? std::max(1, NVH / NKH) : 1;
    std::vector<float> co((size_t)CD), ql((size_t)NVH * KHD), kl((size_t)NVH * KHD), vv((size_t)NVH * VHD);
    const int gdn_nt = omp_heads_threads();
    // ONE parallel region for the whole replay: the layer loop and the row loop stay sequential (the recurrence
    // carries state from row to row), and only the heads are shared out.  The per-row preparation writes the shared
    // conv ring and the shared q/k/v buffers, so exactly one thread does it and the others meet it at a barrier.
#pragma omp parallel num_threads(gdn_nt)
    {
    for (int l = 0; l < (int)row_in_.size(); l++) {
        if (row_in_[(size_t)l].empty() || rec_state_[(size_t)l].empty()) continue;
        const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
        if (!(lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR)) continue;    // only GDN layers carry this state
        for (int m = from; m < to; m++) {
            const GdnRowIn& rec = row_in_[(size_t)l][(size_t)m];
            if (rec.qkv.empty()) continue;
#pragma omp single
            {
            float* st = conv_state_[(size_t)l].data();
            for (int ch = 0; ch < CD; ch++) {
                const float* cs = st + (size_t)ch * (CK - 1);
                float ssum = 0;
                for (int j = 0; j < CK - 1; j++) ssum += cs[j] * gdn_small_[(size_t)l].conv[(size_t)j * CD + ch];
                ssum += rec.qkv[(size_t)ch] * gdn_small_[(size_t)l].conv[(size_t)(CK - 1) * CD + ch];
                co[(size_t)ch] = ssum / (1.0f + expf(-ssum));
            }
            for (int ch = 0; ch < CD; ch++) {
                float* cs = st + (size_t)ch * (CK - 1);
                for (int j = 0; j + 1 < CK - 1; j++) cs[j] = cs[j + 1];
                if (CK > 1) cs[CK - 2] = rec.qkv[(size_t)ch];
            }
            for (int hh = 0; hh < NVH; hh++) {
                const int src = hh / rep;
                for (int d = 0; d < KHD; d++) ql[(size_t)hh * KHD + d] = co[(size_t)src * KHD + d];
                for (int d = 0; d < KHD; d++) kl[(size_t)hh * KHD + d] = co[KD + (size_t)src * KHD + d];
                for (int d = 0; d < VHD; d++) vv[(size_t)hh * VHD + d] = co[2 * KD + (size_t)hh * VHD + d];
            }
            for (int hh = 0; hh < NVH; hh++) {
                float sq = 0, sk = 0;
                for (int d = 0; d < KHD; d++) {
                    sq += ql[(size_t)hh * KHD + d] * ql[(size_t)hh * KHD + d];
                    sk += kl[(size_t)hh * KHD + d] * kl[(size_t)hh * KHD + d];
                }
                const float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
                for (int d = 0; d < KHD; d++) {
                    ql[(size_t)hh * KHD + d] *= iq;
                    kl[(size_t)hh * KHD + d] *= ik;
                }
            }
            }
            float* rs = rec_state_[(size_t)l].data();
#pragma omp for schedule(static)
            for (int hh = 0; hh < NVH; hh++) {
                // As in gdn_layer_batch(): one scratch row per head, inside the region (a per-head allocation in
                // the serial loop is the mistake that cost the mla attempt 2x on 2026-11-03).
                std::vector<float> scratch((size_t)VHD);
                float* kvm = scratch.data();
                const float eg = expf(gdn_small_[(size_t)l].ssm_a[hh] *
                                      softplus_f(rec.ah[(size_t)hh] + gdn_small_[(size_t)l].dt[hh]));
                const float beta = 1.0f / (1.0f + expf(-rec.bh[(size_t)hh]));
                float* sh = rs + (size_t)hh * KHD * VHD;
                const float* kh = &kl[(size_t)hh * KHD];
                const float* vh = &vv[(size_t)hh * VHD];
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
                for (int vd = 0; vd < VHD; vd++) kvm[vd] = (vh[vd] - kvm[vd]) * beta;
                for (int kd = 0; kd < KHD; kd++) {
                    const float k = kh[kd];
                    float* srow = sh + (size_t)kd * VHD;
#pragma omp simd
                    for (int vd = 0; vd < VHD; vd++) srow[vd] += k * kvm[vd];
                }
            }
        }
    }
    }
    return true;
}

// Batch phase attribution (engine#972).  A batched verify pass pays every row's *host* work M times while
// its array launches are M-independent (a pb16i ELF runs its 16 rows whatever M is), so the marginal cost of
// a row is host-side and has to be named, not inferred.  Snapshots the same slots step() fills, prints the
// delta of one step_batch() call when NPU_PHASE_PROF is set.  Measurement only: nothing here is read back.
namespace {
struct BatchPhases {
    const bool on = lh_on();
    double v[LH_N]{};
    long n[LH_N]{};
    BatchPhases() {
        if (on) {
            std::copy(g_lh, g_lh + LH_N, v);
            std::copy(g_lh_n, g_lh_n + LH_N, n);
        }
    }
    double ms(int s) const { return on ? g_lh[s] - v[s] : 0.0; }
    void print(int M) const {
        if (!on) return;
        const double gemm = ms(LH_GEMM) + ms(LH_GEMM_GDN);
        const double row = M > 0 ? (double)M : 1.0;
        std::fprintf(stderr,
                     "[bl] M=%2d gdn %7.1f ab %7.1f core %7.1f attn %6.1f mlp %6.1f rms %5.1f "
                     "res %5.1f gemm %7.1f head %6.1f deq %6.1f | per row: gdn %6.1f attn %5.1f "
                     "gemm %6.1f head %5.1f\n",
                     M, ms(LH_GDN), ms(LH_GDN_AB), ms(LH_GDN_CORE), ms(LH_ATTN), ms(LH_MLP),
                     ms(LH_RMS), ms(LH_RES), gemm, ms(LH_HEAD_ROWS), ms(LH_DEQ_OUT),
                     ms(LH_GDN) / row, ms(LH_ATTN) / row, gemm / row, ms(LH_HEAD_ROWS) / row);
    }
};
}  // namespace

bool Q4nxNpuForward::gdn_layer_batch(int l, const std::vector<float>& xn, int M,
                                     std::vector<float>& out) {
    LhScope _tlh_gdn(LH_GDN);  // as in gdn_layer(): makes the gemm launches inside count as LH_GEMM_GDN
    const int H = c_.H;
    const int KD = lin_kd_, VD = lin_vd_, CD = 2 * lin_kd_ + lin_vd_;
    const int NKH = lin_nkh_, NVH = lin_nvh_, KHD = lin_khd_, VHD = lin_vhd_;
    const int CK = lin_ck_;
    const int i_qkv = idx_of(Q4NX_DESIGN_QKVLIN);
    const int i_z = idx_of(Q4NX_DESIGN_Z);
    const int i_o = idx_of(Q4NX_DESIGN_O);
    if (i_qkv < 0 || i_z < 0 || i_o < 0) {
        err_ = "GDN designs missing (QKVLIN/Z/O)";
        return false;
    }
    std::vector<float> QKV, Z;
    {
        const Q4nxDesignGeom* g = &designs_[i_qkv];
        Q4nxProjections p{};
        p.ple = wqkvlin_.data();
        if (!xd_->pb16i() && !weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) {
            err_ = "assemble gdn qkv"; return false;
        }
        npu_gemm_rows(i_qkv, xn.data(), M, g->K, g->N, B_.data(), QKV, 0, "attn_qkv", g->real_k);
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_z];
        Q4nxProjections p{};
        p.ple = wz_.data();
        if (!xd_->pb16i() && !weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) {
            err_ = "assemble gdn z"; return false;
        }
        npu_gemm_rows(i_z, xn.data(), M, g->K, g->N, B_.data(), Z, 1, "attn_gate", g->real_k);
    }

    // NPU_QKV_DUMP=<layer>: a GDN layer's projections as the engine sees them (q, k, v and the
    // attn_gate/Z), rows 0 and 3, on both paths.  Layer 0 is the first divergent layer and its
    // input is rmsnorm(embed(token)), identical in both paths, so this site answers "is the first
    // differing op the QKV/Z projection itself".
    if (const char* dl = getenv("NPU_QKV_DUMP")) {
        if (l == atoi(dl)) {
            for (int m = 0; m < M; m += (m == 0 ? 3 : 1)) {
                const float* r = QKV.data() + (size_t)m * CD;
                const float* zr = Z.data() + (size_t)m * VD;
                fprintf(stderr, "[qkv-dump] l=%d m=%d", l, m);
                for (int i = 0; i < 8; i++) fprintf(stderr, " q%d=%.6g", i, (double)r[KD + i]);
                for (int i = 0; i < 8; i++) fprintf(stderr, " k%d=%.6g", i, (double)r[2 * KD + i]);
                for (int i = 0; i < 8; i++) fprintf(stderr, " v%d=%.6g", i, (double)r[i]);
                for (int i = 0; i < 8; i++) fprintf(stderr, " gate%d=%.6g", i, (double)zr[i]);
                fprintf(stderr, "\n");
            }
        }
    }
    std::vector<float> CORE((size_t)M * VD, 0.0f);
    const bool dump_gb = getenv("NPU_INFER_DUMP_LAYERS") != nullptr;
    const int rep = NVH / NKH;
    const float scale = 1.0f / sqrtf((float)KHD);
    // ---- per-row preparation, then ONE parallel region for the whole layer ----
    // The recurrence below used to sit inside the row loop, which opened one `#pragma omp parallel for` per
    // (layer, row): 192 regions for an M=4 verify batch, 6144 for an M=128 prefill chunk.  A region is free when
    // the box is idle and expensive when it is not -- measured on this box, the same recurrence went 2 ms/row
    // idle to 252-301 ms/row with 8 background spinners, and region count alone (192 vs 48) was 16.8 vs 4.1
    // ms/row at equal contention (bench: ~/briefs/mtploop-gdn-row-bench.cpp, arms A vs C, bit-identical).
    // So the preparation stays row-major -- it carries conv_state_ and must stay in position order -- and the
    // recurrence becomes head-major with the rows inside it, which preserves every element's per-row operation
    // order.  NPU_GDN_CHECK=1 re-runs the old nesting through the same body and compares bit for bit.
    std::vector<float> co((size_t)CD);
    std::vector<float> AHS((size_t)M * NVH), BHS((size_t)M * NVH);
    std::vector<float> QLS((size_t)M * NVH * KHD), KLS((size_t)M * NVH * KHD),
        VVS((size_t)M * NVH * VHD);
    std::vector<float> co_dump;   // NPU_INFER_DUMP_LAYERS only: keeps `co` per row for the [GB] line
    if (dump_gb) co_dump.assign((size_t)M * CD, 0.0f);
    for (int m = 0; m < M; m++) {
        const float* qkv = QKV.data() + (size_t)m * CD;
        // depthwise causal conv1d (CK) + SiLU, state carried across positions
        float* st = conv_state_[l].data();
        for (int ch = 0; ch < CD; ch++) {
            const float* cs = st + (size_t)ch * (CK - 1);
            float s = 0;
            for (int j = 0; j < CK - 1; j++) s += cs[j] * gdn_small_[(size_t)l].conv[(size_t)j * CD + ch];
            s += qkv[ch] * gdn_small_[(size_t)l].conv[(size_t)(CK - 1) * CD + ch];
            co[(size_t)ch] = s / (1.0f + expf(-s));
        }
        for (int ch = 0; ch < CD; ch++) {
            float* cs = st + (size_t)ch * (CK - 1);
            for (int j = 0; j + 1 < CK - 1; j++) cs[j] = cs[j + 1];
            if (CK > 1) cs[CK - 2] = qkv[ch];
        }
        // alpha/beta (N = NVH, below the GEMM tile width) no longer runs here: it depends on nothing this loop
        // carries, so it is computed head-major inside the layer's one parallel region below (engine#1416).
        if (verify_open_ && m < verify_rows_ && (int)row_in_.size() > l)   // keep what a commit would need;
            row_in_[(size_t)l][(size_t)m].qkv.assign(qkv, qkv + CD);       // ah/bh are stored after the region
        // q/k L2-normalised with NVH/NKH head repetition; v per v-head
        float* qlm = &QLS[(size_t)m * NVH * KHD];
        float* klm = &KLS[(size_t)m * NVH * KHD];
        float* vvm = &VVS[(size_t)m * NVH * VHD];
        for (int hh = 0; hh < NVH; hh++) {
            const int src = hh / rep;
            for (int d = 0; d < KHD; d++) qlm[(size_t)hh * KHD + d] = co[(size_t)src * KHD + d];
            for (int d = 0; d < KHD; d++) klm[(size_t)hh * KHD + d] = co[KD + (size_t)src * KHD + d];
            for (int d = 0; d < VHD; d++) vvm[(size_t)hh * VHD + d] = co[2 * KD + (size_t)hh * VHD + d];
        }
        for (int hh = 0; hh < NVH; hh++) {
            float sq = 0, sk = 0;
            for (int d = 0; d < KHD; d++) {
                sq += qlm[(size_t)hh * KHD + d] * qlm[(size_t)hh * KHD + d];
                sk += klm[(size_t)hh * KHD + d] * klm[(size_t)hh * KHD + d];
            }
            const float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
            for (int d = 0; d < KHD; d++) {
                qlm[(size_t)hh * KHD + d] *= iq;
                klm[(size_t)hh * KHD + d] *= ik;
            }
        }
        if (dump_gb) std::copy(co.begin(), co.end(), co_dump.begin() + (size_t)m * CD);
    }
    // recurrent gated delta rule + scale-free gated RMSNorm.  One body for both nestings so that the check below
    // compares the LOOPS and nothing else; the per-head operation order is identical in either nesting because
    // rows are visited in order inside a head and heads own disjoint state rows and disjoint CORE slices.
    auto head_row = [&](int hh, int m, float* sh, float* crow, float* sc) {
        float* kvm = sc;
        float* delta = kvm + VHD;
        const float eg = expf(gdn_small_[(size_t)l].ssm_a[hh] *
                              softplus_f(AHS[(size_t)m * NVH + hh] + gdn_small_[(size_t)l].dt[hh]));
        const float beta = 1.0f / (1.0f + expf(-BHS[(size_t)m * NVH + hh]));
        const float* kh = &KLS[((size_t)m * NVH + hh) * KHD];
        const float* qh = &QLS[((size_t)m * NVH + hh) * KHD];
        const float* vh = &VVS[((size_t)m * NVH + hh) * VHD];
        const float* zv = Z.data() + (size_t)m * VD + (size_t)hh * VHD;
#pragma omp simd
        for (int i = 0; i < KHD * VHD; i++) sh[i] *= eg;
        // Loops EXCHANGED (kd outer, vd inner) so the inner loop walks sh and vh contiguously instead of
        // gathering with stride VHD.  Each output element still accumulates over kd in increasing order, so
        // the result is the same sequence of operations -- verified by the greedy text being byte-identical.
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
        const float inv = 1.0f / sqrtf(ss / (float)VHD + sem_.eps);
        for (int vd = 0; vd < VHD; vd++) {
            const float zz = zv[vd];
            crow[vd] = crow[vd] * inv * gdn_small_[(size_t)l].norm[vd] * (zz / (1.0f + expf(-zz)));
        }
    };
    // Never take the whole box.  The forward also needs a core for the array's submit thread and the serve
    // needs one for HTTP, so 32 threads on 32 cores measured 25% SLOWER than 16 on a 149-token prefill and
    // 24 was level with 16 (2026-11-04, same window, same binary).  OMP_NUM_THREADS still wins downward;
    // this only stops the default from oversubscribing.
    const int gdn_nt = omp_heads_threads();
    float* rs = rec_state_[l].data();
    const bool gdn_check = getenv("NPU_GDN_CHECK") != nullptr;
    std::vector<float> core_ref;   // NPU_GDN_CHECK only
    if (gdn_check) core_ref.assign(CORE.size(), 0.0f);
    std::vector<float> state_pre;   // NPU_GDN_CHECK only: the state the old nesting must start from
    if (gdn_check) state_pre = rec_state_[l];
    // alpha/beta on the host: for every (row, head) one dot product of the row against that head's alpha row and
    // one against its beta row.  It used to run inside the row loop above on ONE thread -- measured on the served
    // 27B t2s pack, 76 ms of a 730 ms 16-row prefill block (0.099 ms per row and layer, 48 GDN layers), the largest
    // single non-GEMM host term of a prefill pass once the pack column fell to 0.13 ms/launch (engine#1416).
    // Each (row, head) pair is independent and writes its own two elements, and each dot product is the same
    // sequential loop as before (strict IEEE: the build has no -ffast-math, so the summation order is the source
    // order), so sharing the pairs out moves no bit.  It takes NO new parallel region: it is a second worksharing
    // loop inside the one this layer already opens, because a region is what costs under contention (see above).
    // Head-major like the recurrence, so a thread keeps one head's two weight rows (2 x H floats) in L1/L2 across
    // the rows.  NPU_GDN_CHECK=1 recomputes it the old way, serially, and compares bit for bit.
    const float* const wa_all = gdn_small_[(size_t)l].wa.data();
    const float* const wb_all = gdn_small_[(size_t)l].wb.data();
    auto alpha_beta = [&](int hh, int m, float* ah, float* bh) {
        const float* xm = xn.data() + (size_t)m * H;
        const float* ra = wa_all + (size_t)hh * H;
        const float* rb = wb_all + (size_t)hh * H;
        float sa = 0, sb = 0;
        for (int i = 0; i < H; i++) { sa += xm[i] * ra[i]; sb += xm[i] * rb[i]; }
        ah[(size_t)m * NVH + hh] = sa;
        bh[(size_t)m * NVH + hh] = sb;
    };
    {
        // Timed on thread 0 at the two barriers, so `gdn alpha/beta` and `gdn recurrence` stay the two
        // disjoint columns they were (an LhScope inside the region would be written by every thread).
        const bool lh = q4nx_lh::lh_on();
        std::chrono::steady_clock::time_point t0, t1;
        if (lh) t0 = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(gdn_nt)
        {
            std::vector<float> scratch((size_t)2 * VHD);   // one allocation per thread, not per head
            float* sc = scratch.data();
#pragma omp for schedule(static)
            for (int hh = 0; hh < NVH; hh++)
                for (int m = 0; m < M; m++) alpha_beta(hh, m, AHS.data(), BHS.data());
            // implicit barrier: every AHS/BHS element is written before any head reads one
#if defined(_OPENMP)
            if (lh && omp_get_thread_num() == 0) t1 = std::chrono::steady_clock::now();
#else
            if (lh) t1 = std::chrono::steady_clock::now();
#endif
#pragma omp for schedule(static)
            for (int hh = 0; hh < NVH; hh++) {
                float* sh = rs + (size_t)hh * KHD * VHD;
                for (int m = 0; m < M; m++)   // rows in order: the state carries across them
                    head_row(hh, m, sh, CORE.data() + (size_t)m * VD + (size_t)hh * VHD, sc);
            }
        }
        if (lh) {
            const auto t2 = std::chrono::steady_clock::now();
            q4nx_lh::g_lh[LH_GDN_AB] += std::chrono::duration<double, std::milli>(t1 - t0).count();
            q4nx_lh::g_lh_n[LH_GDN_AB]++;
            q4nx_lh::g_lh[LH_GDN_CORE] += std::chrono::duration<double, std::milli>(t2 - t1).count();
            q4nx_lh::g_lh_n[LH_GDN_CORE]++;
        }
    }
    if (verify_open_ && (int)row_in_.size() > l)   // the rest of what a commit would need (qkv was stored above)
        for (int m = 0; m < M && m < verify_rows_; m++) {
            auto& rec = row_in_[(size_t)l][(size_t)m];
            rec.ah.assign(AHS.begin() + (size_t)m * NVH, AHS.begin() + (size_t)(m + 1) * NVH);
            rec.bh.assign(BHS.begin() + (size_t)m * NVH, BHS.begin() + (size_t)(m + 1) * NVH);
        }
    if (gdn_check) {
        // alpha/beta the old way: one thread, row-major, through the same body.  Compared as bit patterns, so a
        // NaN that both produce is equal and a -0.0 against +0.0 is not.
        std::vector<float> ah_ref(AHS.size()), bh_ref(BHS.size());
        for (int m = 0; m < M; m++)
            for (int hh = 0; hh < NVH; hh++) alpha_beta(hh, m, ah_ref.data(), bh_ref.data());
        const bool ab_same = std::memcmp(ah_ref.data(), AHS.data(), AHS.size() * sizeof(float)) == 0 &&
                             std::memcmp(bh_ref.data(), BHS.data(), BHS.size() * sizeof(float)) == 0;
        fprintf(stderr, "[gdncheck] l=%d M=%d alpha/beta shared vs serial: %zu values %s\n", l, M,
                2 * AHS.size(), ab_same ? "identical" : "MISMATCH");
        if (!ab_same) {
            err_ = "gdn check: the shared alpha/beta disagrees with the serial one at layer " + std::to_string(l);
            return false;
        }
        // The old nesting, through the same body: one region per row.  Bit-exactness is the whole claim of this
        // change, so it is asserted rather than argued -- the same form as NPU_REPLAY_CHECK in commit_verify_rows.
        std::copy(state_pre.begin(), state_pre.end(), rec_state_[l].begin());
#pragma omp parallel num_threads(gdn_nt)
        {
            std::vector<float> scratch((size_t)2 * VHD);
            float* sc = scratch.data();
            for (int m = 0; m < M; m++) {
#pragma omp for schedule(static)
                for (int hh = 0; hh < NVH; hh++)
                    head_row(hh, m, rs + (size_t)hh * KHD * VHD,
                             core_ref.data() + (size_t)m * VD + (size_t)hh * VHD, sc);
            }
        }
        int bad = 0;
        double worst = 0.0;
        for (size_t i = 0; i < CORE.size(); i++) {
            const double d = std::fabs((double)CORE[i] - (double)core_ref[i]);
            if (d > worst) worst = d;
            if (CORE[i] != core_ref[i]) bad++;
        }
        fprintf(stderr, "[gdncheck] l=%d M=%d exchanged vs row-major: %d/%zu differ, max|d|=%.3e %s\n", l, M,
                bad, CORE.size(), worst, bad ? "MISMATCH" : "identical");
        // An invariant, not a report: NPU_GDN_CHECK exists so that a test (and a device run) can assert the loop
        // exchange did not move a single element.  A mismatch must fail the call, not print a line nobody reads.
        if (bad) {
            err_ = "gdn check: the head-major nesting disagrees with row-major at layer " + std::to_string(l) +
                   " (" + std::to_string(bad) + " of " + std::to_string(CORE.size()) + ", max|d|=" +
                   std::to_string(worst) + ")";
            return false;
        }
    }
    if (dump_gb && l == 0) {
        for (int m = 0; m < M && m < 2; m++) {
            auto d2 = [&](const char* nm, const float* v, int n) {
                double sm = 0;
                float mx = 0;
                for (int i = 0; i < n; i++) { float a = fabsf(v[i]); sm += a; if (a > mx) mx = a; }
                fprintf(stderr, "  [GB] m=%d %-5s %.6f %.4f\n", m, nm, sm / n, mx);
            };
            d2("qkv", QKV.data() + (size_t)m * CD, CD);
            d2("co", co_dump.data() + (size_t)m * CD, CD);
            d2("z", Z.data() + (size_t)m * VD, VD);
            d2("a", AHS.data() + (size_t)m * NVH, NVH);
            d2("b", BHS.data() + (size_t)m * NVH, NVH);
            d2("core", CORE.data() + (size_t)m * VD, VD);
        }
    }
    {
        const Q4nxDesignGeom* g = &designs_[i_o];
        Q4nxProjections p{};
        p.o = woutlin_.data();
        // Key 6, not 2: key 2 is this layer's FFN gate_proj (N = IM).
        if (!xd_->pb16i() && !weight_cached(6) && q4nx_assemble_B(g, &p, B_.data()) != 0) {
            err_ = "assemble gdn out"; return false;
        }
        npu_gemm_rows(i_o, CORE.data(), M, g->K, g->N, B_.data(), out, 6, "ssm_out", g->real_k);
    }
    return true;
}

// --------------------------------------------------------------------------
// Gemma4 per-layer embeddings (PLE)
// --------------------------------------------------------------------------
// The per-layer token table is I8 [vocab, NL*ple_dim] = 262144 x 8960 for
// Gemma4-E2B, i.e. 9.4 GB if dequantized whole.  The packed grid is row-major
// over 32-row bands (each band of a [32, cols] tensor is a contiguous run of
// cols/256 tiles), so exactly the band holding the requested row is dequantized
// and cached; row-within-band is then a plain offset.
bool Q4nxNpuForward::load_ple_block(int row) {
    const int block = row / Q4NX_TILE_ROWS;
    if (block == ple_block_row_) return true;
    TensorDesc* td = model_tensor_by_name(mw_, "model.per_layer_token_embd.weight");
    if (!td) { err_ = "missing model.per_layer_token_embd.weight"; return false; }
    const int cols = c_.NL * ple_dim_;
    const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, td);
    if (!raw || block < 0) { err_ = "per_layer_token_embd has no data"; return false; }
    ple_block_.assign((size_t)Q4NX_TILE_ROWS * (size_t)cols, 0.0f);
    const size_t plain_bytes =
        (size_t)sem_.vocab_size_per_layer_input * (size_t)cols;
    // engine#1725: a RAW BF16 band. The multi-component packs state this table as f32 and the importer converts it to BF16
    // when it writes the container (a converting copy of the whole V x NL*P table would be 9.4 GiB), so the band here is
    // 32 rows of BF16 -- the shape the plain-int8 branch handles, with a 2-byte element instead of a code.
    //
    // The branch is chosen by the container's DECLARED dtype, not by arithmetic on data_size (review of #1825): a size test
    // needs `% bf16_bytes == 0 && != plain_bytes` to separate two layouts, and every multiple of the band satisfies it, so a
    // plain table whose byte count happens to divide would be read as BF16 bands -- plausible and wrong. This tree already
    // decides exactly this way elsewhere (`model.c`: `strcmp(dtype,"I8")`), and `q4nx_dequant.h`'s rule is that an
    // ambiguous layout is refused rather than resolved by which test happens to pass.
    const size_t bf16_bytes = (size_t)Q4NX_TILE_ROWS * (size_t)cols * 2;
    const char* pl_dtype = td->dtype;
    const bool declared_bf16 = pl_dtype[0] != '\0' && std::strcmp(pl_dtype, "BF16") == 0;
    const bool declared_i8 = pl_dtype[0] != '\0' && (std::strcmp(pl_dtype, "I8") == 0 || std::strcmp(pl_dtype, "Q4") == 0);
    if (!declared_bf16 && !declared_i8) {
        // Some other declared format, or none at all: refuse rather than fall through to a layout guess. A container
        // written by this importer always states one (`w.add()` writes dtype for every tensor), so an empty dtype means a
        // foreign or truncated container, not a layout this loader can infer.
        err_ = pl_dtype[0] != '\0'
                   ? std::string("per_layer_token_embd has dtype ") + pl_dtype +
                         ", which this loader does not read (BF16 or plain I8 only)"
                   : std::string("per_layer_token_embd declares no dtype, so its layout cannot be decided");
        return false;
    }
    if (declared_bf16) {
        if ((size_t)td->data_size < ((size_t)block + 1) * bf16_bytes ||
            (size_t)td->data_size % bf16_bytes != 0) {
            err_ = "per_layer_token_embd is declared BF16 but its size is not whole 32-row bands";
            return false;
        }
        const uint16_t* band = reinterpret_cast<const uint16_t*>(raw + (size_t)block * bf16_bytes);
        for (size_t i = 0; i < (size_t)Q4NX_TILE_ROWS * (size_t)cols; ++i) ple_block_[i] = bf16_to_f32(band[i]);
    } else if ((size_t)td->data_size == plain_bytes) {
        // PLAIN int8 + per-32-column scale (Gemma4): only the requested band of
        // 32 rows is needed, so it is dequantized in place rather than the whole
        // [vocab, NL*ple_dim] table.
        TensorDesc* sd =
            model_tensor_by_name(mw_, "model.per_layer_token_embd.weight.scale");
        // the band and its scales must lie inside their tensors
        if (((size_t)block + 1) * Q4NX_TILE_ROWS * (size_t)cols > plain_bytes) {
            err_ = "per_layer_token_embd row out of range";
            return false;
        }
        const float* sc = sd ? (const float*)data_of(mw_, sd, ((size_t)block + 1) * Q4NX_TILE_ROWS *
                                                                  (size_t)(cols / 32) * sizeof(float))
                             : nullptr;
        if (sd && !sc) { err_ = "per_layer_token_embd scale is too small"; return false; }
        const uint8_t* band = raw + (size_t)block * Q4NX_TILE_ROWS * (size_t)cols;
        const float* band_sc =
            sc ? sc + (size_t)block * Q4NX_TILE_ROWS * (size_t)(cols / 32) : nullptr;
        if (q4nx_dequant_plain_i8(band, band_sc, Q4NX_TILE_ROWS, cols,
                                  ple_block_.data()) != 0) {
            err_ = "per_layer_token_embd dequant failed";
            return false;
        }
    } else {
        const size_t band = q4nx_tensor_bytes(Q4NX_TILE_ROWS, cols);
        if (band == 0 || ((size_t)block + 1) * band > (size_t)td->data_size) {
            err_ = "per_layer_token_embd row out of range";
            return false;
        }
        if (q4nx_dequant_tensor(raw + (size_t)block * band, band, Q4NX_TILE_ROWS,
                                cols, ple_block_.data()) != 0) {
            err_ = "per_layer_token_embd dequant failed";
            return false;
        }
    }
    ple_block_row_ = block;
    return true;
}

// per_layer_input[l] = (context[l] + token_identity[l]) * (1/sqrt(2)) where
//   token_identity[l] = table[token][l*P:(l+1)*P] * sqrt(P)
//   context[l]        = rmsnorm( (emb @ W_l) * H**-0.5, per_layer_proj_norm )
// i.e. project -> scale by 1/sqrt(hidden) -> normalise -> combine.
void Q4nxNpuForward::build_ple_inputs(int token) {
    const int H = c_.H, NL = c_.NL, P = ple_dim_;
    if (P <= 0) return;
    if (!load_ple_block(token)) return;
    const int r = token % Q4NX_TILE_ROWS;
    const float* tok_row = &ple_block_[(size_t)r * (size_t)(NL * P)];
    // The stored table is already scaled by sqrt(P) (the converter bakes it in),
    // so no extra scale is applied here.
    const float tok_scale = 1.0f;
    const float inv_sqrt2 = 0.70710678118654752f;

    std::vector<float> emb((size_t)H);
    for (int k = 0; k < H; k++) emb[k] = embed_at(token, k);

    for (int l = 0; l < NL; l++) {
        const float* W = &ple_model_proj_[(size_t)l * P * H];   // [P][H]
        float* dst = &ple_input_[(size_t)l * P];
        for (int o = 0; o < P; o++) {
            const float* wr = &W[(size_t)o * H];
            double acc = 0.0;
            for (int k = 0; k < H; k++) acc += (double)wr[k] * (double)emb[k];
            dst[o] = (float)acc * (1.0f / std::sqrt((float)H));
        }
        rmsnorm(dst, ple_norm_.data(), P, dst);
        for (int i = 0; i < P; i++) {
            const float ti = tok_row[(size_t)l * P + i] * tok_scale;
            dst[i] = (dst[i] + ti) * inv_sqrt2;
        }
    }
}

// --------------------------------------------------------------------------
// M=16 decode-family selection
// --------------------------------------------------------------------------

// The phase accounting (q4nx_lh, in q4nx_lh.h) is shared: the launch layer in npu/target/xdna writes the
// layer-GEMM slots and this file writes the per-step and per-request ones. P4 moved the namespace out of
// q4nx_forward.h into q4nx_lh.h, and this file names its members through the file-scope `using namespace q4nx_lh`
// above, so no per-name using-declarations are needed here.

// main's model-side design_index()/ctx_for()/m16_check_gemm()/m16_check_gemm_rows()/prof_add()/prof_dump() are not
// here. P4 moved the launch layer, the design contexts (ctx_/ctx16_), the M=16 A/B check and the per-phase timers into
// npu/target/xdna, where each of them exists with live call sites (XdnaTarget::prof_add 8, prof_dump 6, design_index 5,
// m16_check_gemm 5, ctx_for 3, m16_check_gemm_rows 2). What stays in the model is the maths and the design GEOMETRY
// (designs_/designs16_), which is what the code below reads.

namespace {  // host-copy accounting (NPU_COPY_PROF=1), main #1142, kept where step/step_batch live
// Host-copy accounting (NPU_COPY_PROF=1).  The [ph]/[bs] lines say what the *device* did; these say what
// the host moved into and out of its own vectors on the same pass, per site, in bytes and milliseconds.
// That is the figure an audit of "what is left to remove" needs and a phase residual cannot give: the
// sites are host-side row gathers whose cost is invisible inside gemm/wait/readback.  Additive
// measurement only, exactly like the phases above: nothing in the compute path reads it, and with the
// variable unset nothing is counted and nothing is printed.
//
// Single writer, like `g_ph` beside it, and that is a property of the callers rather than of these
// globals: every serve-side entry into step()/step_batch() holds the engine's mutex
// (app/forward_serve.cpp:258 and :698), so one pass is in flight per process, and `--npu-streams N`
// is N processes sharing the device by flock (app/npu_device_lock.h), each with its own copy of this
// struct -- not N threads.  A caller that ever parallelises the layer loop would make cp_count() and
// the CpScope timings a data race and cp_reset() clear another pass's counts; the counters would
// then have to become per-pass objects, as would `g_ph`.
enum CopySite { CP_QKV, CP_GATE, CP_CTX, CP_N };
const char* const kCopySiteName[CP_N] = {"qkv_split", "attn_gate", "ctx_row"};
struct CopySites {
    double b[CP_N]{};
    double ms[CP_N]{};
    long   n[CP_N]{};
    int    rows = 0;      // rows a batched pass accounted, so bytes/row is read, not divided
};
CopySites g_cp;
inline bool cpprof_on() {
    static const bool on = getenv("NPU_COPY_PROF") != nullptr;
    return on;
}
inline void cp_count(int site, size_t bytes) {
    if (!cpprof_on()) return;
    g_cp.b[site] += (double)bytes; g_cp.n[site]++;
}
inline void cp_reset() {
    if (!cpprof_on()) return;
    g_cp = CopySites{};
}
// Times one site's region; its bytes are counted with cp_count() inside the same scope, so the site's
// ms and its bytes describe the same work.
struct CpScope {
    int  site;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit CpScope(int s)
        : site(s), on(cpprof_on()),
          t0(on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~CpScope() { if (on) g_cp.ms[site] += ph_ms(t0); }
};
void cp_print(const char* what) {
    if (!cpprof_on()) return;
    double tb = 0, tm = 0;
    fprintf(stderr, "[cp] %s host copies:", what);
    for (int i = 0; i < CP_N; i++) {
        if (!g_cp.n[i]) continue;
        fprintf(stderr, " %s=%.1fKiB/%ld/%.3fms", kCopySiteName[i], g_cp.b[i] / 1024.0, g_cp.n[i],
                g_cp.ms[i]);
        tb += g_cp.b[i]; tm += g_cp.ms[i];
    }
    fprintf(stderr, " total=%.1fKiB %.3fms", tb / 1024.0, tm);
    if (g_cp.rows > 1) fprintf(stderr, " (%.1fKiB/row)", tb / 1024.0 / (double)g_cp.rows);
    fprintf(stderr, "\n");
}
}  // namespace

bool Q4nxNpuForward::gemv_tiles(int layer, int role, std::vector<uint8_t>& out, std::string& err) {
    const int ri = idx_of((Q4nxDesign)role);
    if (ri < 0) { err = "gemv: no design for this role"; return false; }
    const Q4nxDesignGeom* g = &designs_[ri];
    if (g->K % 256 || g->N % 32) { err = "gemv: K/N are not on the tile grid"; return false; }
    const int kc = g->K / 256;
    const size_t tile = 5120;
    out.assign((size_t)kc * (size_t)(g->N / 32) * tile, 0);
    size_t at = 0;
    auto append = [&](const char* suffix, int rows) -> bool {
        char nm[192];
        std::snprintf(nm, sizeof nm, "model.layers.%d.%s", layer, suffix);
        TensorDesc* d = model_tensor_by_name(mw_, nm);
        if (!d) { err = std::string("gemv: missing ") + nm; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        const size_t need = (size_t)kc * (size_t)(rows / 32) * tile;
        if (!raw || (size_t)d->data_size < need) {
            err = std::string("gemv: ") + nm + " holds fewer tiles than the design needs";
            return false;
        }
        if (at + need > out.size()) { err = "gemv: the design's N is smaller than its parts"; return false; }
        std::memcpy(out.data() + at, raw, need);
        at += need;
        return true;
    };
    bool ok = true;
    switch (role) {
        case Q4NX_DESIGN_QKV:
            ok = append("self_attn.q_proj.weight", g->q_rows) &&
                 append("self_attn.k_proj.weight", g->k_rows) &&
                 append("self_attn.v_proj.weight", g->v_rows);
            break;
        case Q4NX_DESIGN_O: ok = append("self_attn.o_proj.weight", g->real_n > 0 ? g->real_n : g->N); break;
        case Q4NX_DESIGN_GU:
            ok = append("mlp.gate_proj.weight", g->gate_rows) && append("mlp.up_proj.weight", g->up_rows);
            break;
        case Q4NX_DESIGN_G: ok = append("mlp.gate_proj.weight", g->real_n > 0 ? g->real_n : g->N); break;
        case Q4NX_DESIGN_U: ok = append("mlp.up_proj.weight", g->real_n > 0 ? g->real_n : g->N); break;
        case Q4NX_DESIGN_D: ok = append("mlp.down_proj.weight", g->real_n > 0 ? g->real_n : g->N); break;
        default: err = "gemv: unsupported role"; return false;
    }
    if (!ok) return false;
    // rows past the real N stay zero tiles (scale 0, zero 0, codes 0 -> dequantises to 0)
    return true;
}

bool Q4nxNpuForward::gemv_tile_cb(void* self, int layer, int role, std::vector<uint8_t>& out,
                                  std::string& err) {
    return static_cast<Q4nxNpuForward*>(self)->gemv_tiles(layer, role, out, err);
}

// The head on the gemv family: the pack ships the head as q4_1 tiles of its own, so the same byte
// copy works.  Only a tied bf16 table (a GGUF-converted container) has no tile stream, and that
// case is refused by name rather than silently quantised here.
bool Q4nxNpuForward::run_lm_head_gemv(const std::vector<float>& h, std::vector<float>& logits) {
    const int H = c_.H, NV = c_.NV;
    I8Ctx* hctx = xd_ ? xd_->head_ctx() : nullptr;
    if (!hctx || !hctx->gemv) { err_ = "gemv: no gemv head context"; return false; }
    if (gemv_head_.empty()) {
        TensorDesc* d = model_tensor_by_name(mw_, "lm_head.weight");
        if (!d) { err_ = "gemv head: no lm_head.weight tensor"; return false; }
        const uint8_t* raw = (const uint8_t*)model_tensor_data(mw_, d);
        const size_t want = I8Ctx::gemv_w_bytes(kpad_, hctx->ND);
        // The container stores the head's tiles at the REAL vocabulary width, and the head ELF's N is
        // that width on the column grid (151936 -> 152064 for the 0.5B pack).  The tail between them is
        // columns past the vocabulary, so it stays zero tiles -- the same rule the fused designs use for
        // a padded tail -- and only the first NV logits are ever read.
        const size_t have = I8Ctx::gemv_w_bytes(kpad_, NV);
        // The gemv family reads q4_1 TILES. A BF16 or Q8 head can easily be at least `have` bytes, and
        // copying one into the tile stream produced invalid logits instead of a refusal (engine#910
        // review), so the dtype is checked by name and the length has to BE the stream size -- at the
        // real vocabulary width, or at the padded one when a container stores the padding too.
        const std::string dt(d->dtype);
        if (dt != "q4_1" && dt != "I8") {
            err_ = "gemv head: lm_head.weight is " + dt + ", not the q4_1 tile stream this family reads";
            return false;
        }
        const size_t stored = size_t(d->data_size);
        if (!raw || (stored != have && stored != want)) {
            err_ = "gemv head: lm_head.weight is " + std::to_string(stored) + " B, which is neither " +
                   std::to_string(have) + " (the real vocabulary width) nor " + std::to_string(want) +
                   " (the padded grid): not a q4_1 tile stream for K=" + std::to_string(kpad_) +
                   " N=" + std::to_string(NV);
            return false;
        }
        gemv_head_.assign(want, 0);
        // Bound by the DESTINATION, not the source: `want` is this buffer's size, and the tail past
        // the container's tiles is zero-fill by construction.
        std::memcpy(gemv_head_.data(), raw, std::min(want, have));
        if (!head_bo_) head_bo_ = hctx->make_weight_bo(*dev_);
        if (!hctx->upload_tiles_into(*head_bo_, gemv_head_.data(), gemv_head_.size())) {
            err_ = "gemv head: upload failed"; return false;
        }
    }
    auto r = hctx->launch_async_with_bo(*head_bo_, h.data(), 1, H, 1.0f);
    hctx->wait_kernel(r);
    hctx->readback();
    logits.assign((size_t)NV, 0.0f);
    const int n = std::min(NV, hctx->ND);
    for (int i = 0; i < n; i++) {
        float v = hctx->Cg[i];
        if (sem_.logit_softcap > 0) v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
        logits[(size_t)i] = v;
    }
    return true;
}

// --------------------------------------------------------------------------
// one NPU GEMM: C[1,N] = A[1,K] x B[K,N]  (row-major B)
//
// The launch itself is the target's (npu/target/xdna/xdna_target.h): phase-selected M=16/M=128 kernel,
// packed-weight BO cache, pb16i images, Hadamard fold.  What stays here is the model's own fallback, the
// plain float GEMM of init_host() -- that is a property of the model, not of any NPU.
// --------------------------------------------------------------------------
// The integer host arm (engine#1808/#1809). Two decisions here, both measured:
//  * the weight codes and their per-128-K multipliers are quantised ONCE per tensor (`wkey`) and cached --
//    doing it per call made a 7B token ~7.5 G scalar operations and the arm never finished a token inside a
//    9-minute ticket;
//  * the inner sum is exact int32 with no float in it (q4nx_host_gemm_i8_grouped), so the comparison against
//    the device is about accumulation and epilogue, not about a float both sides happen to share.
void Q4nxNpuForward::host_gemm_i8(const float* A, int a_k, int M, int K, int N, const float* Bmat,
                                  int wkey, float* out) {
    // THE KEY MUST CARRY THE LAYER AND THE SHAPE (engine#1830 review). `wkey` is a per-ROLE slot -- 0 QKV, 1 O/gate,
    // 2 GU, 3 up, 4 down -- so it is the same number in every layer, and `Bmat` is the one scratch buffer that
    // `load_layer()` decodes each layer into, so neither identifies a layer. Keyed on `wkey` alone, every layer after 0
    // ran on LAYER 0's weights: layer 0 scored cos 0.99923, layer 1 collapsed to 0.37, and 31 layers of wrong residual
    // updates compounded into the "explosion, not drift" the per-layer dump shows. The device cache this arm mirrors
    // says the rule in its own comment -- one weight BO per (layer, wkey) -- and carries `a.layer = cur_layer_`.
    // `wkey < 0` is the routed-expert case: those weights vary per TOKEN, which is why the device path passes -1; caching
    // them would serve every token's experts from the first token's, so that case packs a throwaway image instead.
    const bool cacheable = wkey >= 0;
    const auto key = std::make_tuple(cur_layer_, wkey, K, N);
    HostI8Tensor scratch;
    const HostI8Tensor* cached = nullptr;
    auto it = cacheable ? host_i8_cache_.find(key) : host_i8_cache_.end();
    if (!cacheable || it == host_i8_cache_.end()) {
        HostI8Tensor t;
        t.N = N;
        t.codes.assign((size_t)K * (size_t)N, 0);
        // RESOLVED toward the int8 packer (this branch) rather than main's corrected int4 rule (engine#1839).
        // Both are self-consistent; the question is what the DEVICE consumes. The device's body image comes from
        // packB_blocked() -- one multiplier per column, int8 codes -- and this branch measured the consequence:
        // device logits BIT-EXACT (max|d| = 0) over four steps, three repeats, plus an edge case. So the int8 packer
        // is the rule the reference has to share with the device, and q4nx_quant_int4_*() stays for the paths that
        // genuinely hold int4 codes.
        t.d.assign((size_t)N, 0.0f);
        // THE OPERANDS MUST BE THE ENGINE'S, NOT A PRIVATE RE-QUANTISATION (engine#1808/#1830). This arm exists to be
        // the integer reference a kernel family is gated against, and the device's weight image comes from
        // q4nx_pack_B_int8_blocked(): ONE multiplier per column (amax over the whole K), int8 codes in [-127, 127].
        // Quantising here with a different rule -- per 128-K group, int4 [-8, 7] -- computes a different model: measured
        // 2026-11-10, the arm answered "ikt" to "The capital of France is" and then locked onto one token, while the
        // per-GEMM A/B read cos 0.9999 against f32 on the same tensors. Same packer, same bytes, same model.
        double ssum = 0.0;
        // Blocked: the image is written in the blocked microtile order, which is the fast path AND the layout the
        // device actually reads, so the MAC below walks it with the same index. (blocked=0 was measured at 174.5 s of
        // warm-up on this container versus 32.2 s blocked -- the row-major scatter is the slow one.) But the image is
        // UNPADDED, K*N bytes, and blocked_index() addresses whole 64x128 tiles: an N that is not a multiple of 128 (or
        // a K not of 64) packs and reads past the vector (engine#1994). Those geometries take row-major, which holds
        // the same codes, so the result is unchanged and only an unaligned shape pays the slower pack.
        t.blocked = q4nx_layout::blocked_fits(K, N);
        q4nx_pack_B_int8_blocked(t.codes.data(), Bmat, K, N, K, N, t.blocked ? 1 : 0, t.d.data(), &ssum);
        if (!cacheable) {
            scratch = std::move(t);
            cached = &scratch;
        } else {
            cached = &host_i8_cache_.emplace(key, std::move(t)).first->second;
        }
    } else {
        cached = &it->second;
    }
    const HostI8Tensor& t = *cached;
    if (const char* gdir = getenv("NPU_I8_GS_DUMP")) {
        static int n = 0;
        char path[4096];
        std::snprintf(path, sizeof path, "%s/arm_L%d_N%d_K%d_%03d.bin", gdir, cur_layer_, N, K, n++);
        if (std::FILE* f = std::fopen(path, "wb")) {
            std::fwrite(t.d.data(), sizeof(float), std::min<size_t>((size_t)N, t.d.size()), f);
            std::fclose(f);
        }
    }
    std::vector<int8_t> a8((size_t)K);
    for (int m = 0; m < M; m++) {
        const float* a = A + (size_t)m * (size_t)a_k;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const float v = std::fabs(a[k]);
            if (std::isfinite(v) && v > amax) amax = v;
        }
        const float ascale = (amax >= 1e-12f) ? amax / 127.0f : 1.0f;
        const float tis = 1.0f / ascale;
        for (int k = 0; k < K; k++) {
            float v = a[k];
            if (!std::isfinite(v)) v = 0.0f;
            int q = (int)std::round(v * tis);
            a8[(size_t)k] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
        float* dst = out + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)((int)t.codes[q4nx_layout::image_index((size_t)k, (size_t)n, K, N, t.blocked)] *
                                 (int)a8[(size_t)k]);
            dst[(size_t)n] = t.d[(size_t)n] * (ascale * (float)acc);
        }
    }
}

// NPU_INFER_DUMP_OPS (ops-dump.h): one line per projection, tagged with the tag its caller already passes
// ("attn_qkv", "ssm_out", ...). Called where a projection's result is final -- both host arms and the device arm --
// so a host run and a device run print the same lines. Row 0 of the result; off is one cached bool.
static inline void op_line(const char* tag, int layer, int rows, const std::vector<float>& out, int n) {
    if (onebit::npu::ops_dump_enabled() && n > 0 && out.size() >= (size_t)n)
        onebit::npu::ops_dump(tag, layer, rows, out.data(), n);
}

void Q4nxNpuForward::npu_gemm(int role, const float* A, int K, int N, const float* Bmat,
                              std::vector<float>& out, int wkey, const char* sc, int a_k) {
    if (host_gemm_) {
        // The fallback reads A[k] for k < K exactly as the kernel does, so it needs the engine#845 pad
        // too: q4nx_host_gemm_f32_padded() owns the pad-and-read pair so neither leg can be handed an
        // unpadded A. (In review of this PR the split moved the pad into the target and left this
        // branch reading past the caller's buffer at a padded design.)
        out.assign((size_t)N, 0.0f);
        if (host_gemm_i8_) {
            host_gemm_i8(A, a_k, 1, K, N, Bmat, wkey, out.data());
            op_line(sc, cur_layer_, 1, out, N);
            return;
        }
        q4nx_host_gemm_f32_padded(A, a_k, 1, K, N, Bmat, out.data());
        op_line(sc, cur_layer_, 1, out, N);
        return;
    }
    npu::target::GemvArgs a;
    a.role = role; a.layer = cur_layer_; a.M = 1; a.K = K; a.N = N;
    a.x = A; a.Bmat = Bmat; a.wkey = wkey; a.tag = sc;
    a.a_k = a_k;
    a.tiles = &Q4nxNpuForward::gemv_tile_cb; a.tiles_self = this;
    std::string err;
    if (!xd_ || !xd_->gemv(a, out, err)) { err_ = err.empty() ? "npu_gemm: no target" : err; return; }
    op_line(sc, cur_layer_, 1, out, N);
}

void Q4nxNpuForward::npu_gemm_rows(int role, const float* A, int M, int K, int N,
                                   const float* Bmat, std::vector<float>& out, int wkey,
                                   const char* sc, int a_k, bool decode_twin) {
    if (host_gemm_) {
        // Same pad, and here it is also a stride: the rows form used to index A at `m * K` while the
        // caller's buffer is laid out at `m * a_k`, so every row past the first read the wrong row.
        out.assign((size_t)M * N, 0.0f);
        if (host_gemm_i8_) {
            host_gemm_i8(A, a_k, M, K, N, Bmat, wkey, out.data());
            op_line(sc, cur_layer_, M, out, N);
            return;
        }
        q4nx_host_gemm_f32_padded(A, a_k, M, K, N, Bmat, out.data());
        op_line(sc, cur_layer_, M, out, N);
        return;
    }
    // The device's activation BO holds exactly max_batch_rows() rows (the M=128 prefill design), and a wider
    // launch is not quantized at all -- the kernel then reads and poisons past the BO, which is memory
    // corruption, not a slow path (engine#1372: 4 of 4 segfaults on the d3 pack's real questions, and one run
    // that answered from rows the layers never processed). So a wider GEMM is walked in blocks here, at the one
    // choke point every batched projection goes through: the same launches in the same order, each within the
    // BO. Each row's result is independent of the others -- the row loops downstream (`row_attn`, the GDN
    // recurrence) never needed the whole prompt in one launch, only the GEMMs did.
    const int blk = max_batch_rows();
    if (blk > 0 && M > blk) {
        // engine#1582: say the block width out loud, once. A log that reads "0 launches, 93 s host" cannot be told
        // apart from a device that never ran the batch, and the width the batch was blocked at is the whole answer.
        static bool note_said = false;
        if (!note_said) {
            const char* note = q4nx_block_note(M, blk, xd_ ? xd_->caps().prefill_rows : 0);
            if (note && note[0]) { std::fprintf(stderr, "  [batch] %s\n", note); note_said = true; }
        }
        out.assign((size_t)M * N, 0.0f);
        // NPU_INFER_DUMP_OPS: the blocks are one projection, so they print one line (rows=M, row 0), as the host
        // arms do; the inner launches are muted for the length of the walk.
        onebit::npu::OpsDumpMute ops_mute;
        for (int b0 = 0; b0 < M && err_.empty(); b0 += blk) {
            const int rows = std::min(blk, M - b0);
            std::vector<float> part;
            // A's row stride is the caller's a_k, not the padded K (the same stride trap the host path documents).
            npu_gemm_rows(role, A + (size_t)b0 * a_k, rows, K, N, Bmat, part, wkey, sc, a_k, decode_twin);
            if (part.size() < (size_t)rows * N) {
                if (err_.empty()) err_ = std::string("npu_gemm_rows: a blocked ") + sc + " launch returned short";
                return;
            }
            std::copy(part.begin(), part.begin() + (size_t)rows * N, out.begin() + (size_t)b0 * N);
        }
        // Printed by hand: op_line() would be muted by this scope's own guard. A nested walk stays quiet.
        if (err_.empty() && onebit::npu::ops_dump_enabled() && onebit::npu::ops_dump_mute_depth() == 1) {
            char line[192];
            if (onebit::npu::ops_dump_line(line, sizeof line, sc, cur_layer_, M, out.data(), N) > 0)
                std::fputs(line, stderr);
        }
        return;
    }
    npu::target::GemvArgs a;
    a.role = role; a.layer = cur_layer_; a.M = M; a.K = K; a.N = N;
    a.x = A; a.Bmat = Bmat; a.wkey = wkey; a.tag = sc;
    a.a_k = a_k;
    a.rows = true; a.decode_twin = decode_twin;
    a.tiles = &Q4nxNpuForward::gemv_tile_cb; a.tiles_self = this;
    std::string err;
    if (!xd_ || !xd_->gemv(a, out, err)) { err_ = err.empty() ? "npu_gemm_rows: no target" : err; return; }
    op_line(sc, cur_layer_, M, out, N);
}

// The folded container's sign table: magic "HAD1", u32 count, count x (u32 width, u32 offset),
// then the f32 signs those offsets index (the GGUF's prism.hadamard.sign_widths/sign_values).
// The plain-text overload. It is declared in q4nx_forward.h and used by npu/forward/embeds_equiv_probe.cpp, but only the
// HybridInputs-taking definition existed, so every full build stopped at `undefined reference to
// Q4nxNpuForward::step_batch_embeds(std::vector<float>, int, std::vector<float>&)` — the onebit target links because it
// does not pull the probe in, which is why the breakage survived a release-ish build. Defaulting `in` is exactly the
// documented meaning of the three-argument form: no position planes, no deepstack sets, nothing vision-conditioned.
bool Q4nxNpuForward::step_batch_embeds(const std::vector<float>& embeds, int start_pos,
                                      std::vector<float>& logits) {
    return step_batch_embeds(embeds, start_pos, logits, HybridInputs{});
}

bool Q4nxNpuForward::step_batch_embeds(const std::vector<float>& embeds, int start_pos,
                                      std::vector<float>& logits, const HybridInputs& in) {
    if (!ready_) { err_ = "not initialised"; return false; }
    const size_t H = (size_t)c_.H;
    if (H == 0 || embeds.empty() || embeds.size() % H != 0) {
        err_ = "step_batch_embeds: " + std::to_string(embeds.size()) + " floats is not a whole number of " +
               std::to_string(H) + "-wide rows";
        return false;
    }
    for (float v : embeds)
        if (!std::isfinite(v)) { err_ = "step_batch_embeds: a row holds a non-finite value"; return false; }
    const int rows = int(embeds.size() / H);
    const std::vector<int>& position_ids = in.position_ids ? *in.position_ids : std::vector<int>();
    if (!position_ids.empty()) {
        if (!sem_.mrope) {
            err_ = "step_batch_embeds: position_ids given but this pack does not declare interleaved MRoPE";
            return false;
        }
        if (sem_.n_mrope_section < 3) {
            err_ = "step_batch_embeds: the pack declares MRoPE but no mrope_section, so the axis layout is "
                   "unknown; refusing rather than guessing which frequencies are (t, h, w)";
            return false;
        }
        if (position_ids.size() != size_t(3) * size_t(rows)) {
            err_ = "step_batch_embeds: position_ids is " + std::to_string(position_ids.size()) + " values, expected 3 x " +
                   std::to_string(rows);
            return false;
        }
    }
    // The body indexes tokens[m] only when nothing is injected, but it still needs the row count and the guards
    // above it read tokens; a zero vector of the right length keeps every check honest (0 is a valid id) while
    // the fill itself takes the injected rows.
    if (in.deepstack_sets > 0) {
        if (in.deepstack_sets != sem_.n_deepstack) {
            err_ = "step_batch_embeds: " + std::to_string(in.deepstack_sets) + " deepstack set(s) given but the pack "
                   "declares " + std::to_string(sem_.n_deepstack) + " deepstack layer(s)";
            return false;
        }
        if (!in.deepstack) { err_ = "step_batch_embeds: deepstack_sets without a buffer"; return false; }
        if (in.image_lo < 0 || in.image_hi <= in.image_lo || in.image_hi > rows) {
            err_ = "step_batch_embeds: deepstack sets need the image row span (image_lo/image_hi inside 0.." +
                   std::to_string(rows) + ")";
            return false;
        }
    }
    const std::vector<int> placeholder(size_t(rows), 0);
    struct Restore {
        const float*& p;
        const int*& q;
        int& n;
        const float*& d;
        int& dn;
        int& lo;
        int& hi;
        ~Restore() { p = nullptr; q = nullptr; n = 0; d = nullptr; dn = 0; lo = -1; hi = -1; }
    } restore{inject_embeds_, pos3_, pos3_rows_, deepstack_, deepstack_rows_, deepstack_lo_, deepstack_hi_};
    inject_embeds_ = embeds.data();
    pos3_ = position_ids.empty() ? nullptr : position_ids.data();
    pos3_rows_ = rows;
    deepstack_ = in.deepstack;
    deepstack_rows_ = in.image_hi - in.image_lo;
    deepstack_lo_ = in.image_lo;
    deepstack_hi_ = in.image_hi;
    return step_batch(placeholder, start_pos, logits);
}

bool Q4nxNpuForward::step_batch_all(const std::vector<int>& tokens, int start_pos,
                                    std::vector<float>& logits_all) {
    // A noncausal pack's full-attention layers read the whole prompt, so in a batch every row attends to
    // the rows after it.  The per-row logits this function exists to return are the prefix-conditioned ones
    // a draft verifier compares against, and a noncausal pass cannot produce them -- refuse by name, as
    // step() and step_batch(start_pos>0) do, instead of handing back rows that no draft may be checked
    // against.  Unreachable while d3 packs ship no draft head, which is why it is six lines and not a mode.
    if (noncausal_attn_) {
        err_ = "attention.mode noncausal_full_attention is a prompt-encoding mode and step_batch_all() "
               "verifies drafted tokens: every row of that batch attends to the rows after it, so its "
               "per-row logits are not the prefix-conditioned ones a draft is checked against (use "
               "step_batch() with the whole prompt at position 0)";
        return false;
    }
    std::vector<float> last;
    std::vector<float>* saved = batch_all_out_;
    batch_all_out_ = &logits_all;
    const bool ok = step_batch(tokens, start_pos, last);
    batch_all_out_ = saved;
    return ok;
}

bool Q4nxNpuForward::step_batch(const std::vector<int>& tokens, int start_pos,
                                std::vector<float>& logits) {
    if (!ready_) { err_ = "not initialised"; return false; }
    // A noncausal pack reads the whole prompt at once, so a chunk that starts above 0 was already
    // encoded causally by an earlier call and cannot be repaired here.  Named, not silently causal.
    if (noncausal_attn_ && start_pos > 0) {
        err_ = "attention.mode noncausal_full_attention encodes a prompt in one batched pass, but this "
               "call starts at position " + std::to_string(start_pos) +
               " (the positions before it were encoded causally); send the whole prompt as one batch";
        return false;
    }
    const int H = c_.H;
    const int M = (int)tokens.size();
    if (M <= 0) { err_ = "empty batch"; return false; }
    for (int t : tokens)
        if (t < 0 || t >= c_.NV) { err_ = "token out of range"; return false; }


    // A new request starts at position 0: clear the KV cache.  The cache is
    // append-only (kcache_[l].insert(end, ...)) but the attention indexes it by
    // `pos` (t in 0..pos), so without this a second request reads the PREVIOUS
    // request's K/V for positions 0..prompt_len and appends its own after them --
    // visible as cross-request echo (MiniCPM5-1B answered Japan, then echoed
    // "Tokyo" for France/Germany).  forward_serve.cpp's "each request resets it
    // from position 0" only ever held in the NPU_BATCH_CHECK debug path.
    if (start_pos == 0) reset();

    // Anything the batched body does not model falls back to per-token step():
    // MLA/MoE, GDN linear attention, Gemma4 (PLE + per-layer-type attention widths
    // + KV sharing), attn_output_gate, and multi-layer-type RoPE.
    bool kv_self = true;
    for (int l = 0; l < (int)kv_owner_.size() && l < c_.NL; l++)
        if (kv_owner_[l] != l) kv_self = false;
    // Two attention geometries (QKV2) and multiple layer types are fine: the
    // batched body selects the per-layer design and RoPE exactly as step() does.
    // MoE is fine too: its router/top-k are data-dependent per row and stay
    // per-token, while the attention GEMMs around them are batched. MLA and GDN
    // still fall back (their attention is not modelled here).
    // GDN layers are batched too (gdn_layer_batch: M-row projections, per-row
    // recurrence), so lin_kd_ no longer disqualifies a model, and attn_output_gate
    // is handled in the attention block below. MLA, per-layer embeddings, Gemma4
    // and shared-KV stacks still fall back.
    const bool simple_dense =
        !mla_ && ple_dim_ == 0 && !gemma4_layer_ && kv_self;
    if (!simple_dense || getenv("NPU_NO_BATCH")) {
        if (inject_embeds_) {
            err_ = "step_batch_embeds: this model's batched body falls back to per-token step(), which needs "
                   "token ids to look up; precomputed embeddings are refused rather than run on placeholders";
            return false;
        }
        if (getenv("NPU_BATCH_DBG"))
            std::fprintf(stderr,
                         "[batch] M=%d fallback: mla=%d lin=%d ple=%d gemma4=%d experts=%d "
                         "gate=%d kv_self=%d no_batch=%d\n",
                         M, mla_, lin_kd_, ple_dim_, (int)gemma4_layer_, n_expert_,
                         attn_gate_, (int)kv_self, (int)(getenv("NPU_NO_BATCH") != nullptr));
        std::vector<float> one;
        for (int m = 0; m < M; m++)
            if (!step(tokens[m], start_pos + m, one)) return false;
        logits = one;
        return true;
    }

    // Debug oracle (NPU_BATCH_CHECK=1): run the same tokens sequentially first,
    // capture the next-token argmax, clear the KV the sequential pass appended,
    // then let the batched pass run and compare. One process, both paths, same
    // device state -- localises a batched-vs-sequential divergence immediately.
    static const bool batch_check = getenv("NPU_BATCH_CHECK") != nullptr;
    int seq_argmax = -1;
    // engine#1781: the old oracle compared ONE argmax at ONE position -- the last row's -- which is not the question a
    // speculative decoder asks. Every drafted position's LOGITS matter (the accept rule consumes the target's
    // probability at the drafted token), so the sequential rows are kept whole and compared row for row, bitwise.
    // M x vocab floats: 4-7 MB on the models this runs, allocated only under the debug flag.
    std::vector<float> seq_rows, bat_rows;
    // Not with injected rows: the oracle's own sequential pass runs step(tokens[m]) on the placeholder ids this
    // entry point supplies, so it would compare against a zero-token pass and report a divergence that is not
    // there. A debug tool that lies is worse than one that says "not applicable".
    // engine#1905: the oracle's sequential replay must start from the SAME state the batched pass will, or the
    // comparison is between different prefixes. It used to walk `start_pos .. start_pos+M-1` and then `reset()`, so for
    // any chunk with `start_pos > 0` the batched pass ran from an empty cache and the replay from the real prefix: the
    // oracle reported MISMATCH for every chunk but the first (measured on `bonsai2-27b-NPU-b8`, width 1, four ids:
    // start=0 MATCH, start=1 seq=220 bat=653). That reads as a row-dependence bug in the kernels and sends the reader
    // to fix the wrong thing. Snapshot the state the prefix established and restore it after the replay -- the same
    // pair the prefix cache uses (`snapshot_state`/`restore_state`, and `q4nx_state_planes.h` is device-free).
    StateSnapshot oracle_snap;
    const bool oracle_snapped = batch_check && !inject_embeds_ && start_pos > 0 &&
                                snapshot_state(start_pos, oracle_snap);
    if (batch_check && !inject_embeds_) {
        std::vector<float> seq;
        if (c_.NV > 0) seq_rows.reserve((size_t)M * (size_t)c_.NV);
        for (int m = 0; m < M; m++) {
            if (!step(tokens[m], start_pos + m, seq)) return false;
            seq_rows.insert(seq_rows.end(), seq.begin(), seq.end());   // every row, not only the last
            if (m == M - 1)
                for (size_t i = 0; i < seq.size(); i++)
                    if (seq_argmax < 0 || seq[i] > seq[(size_t)seq_argmax]) seq_argmax = (int)i;
        }
        // Restore instead of reset() when the replay ran over a real prefix: `reset()` would leave the batched pass
        // starting from nothing while the replay started from `start_pos` rows, which is the false mismatch above. With
        // `start_pos == 0` there is nothing to restore and `reset()` is exactly right.
        if (oracle_snapped) {
            if (!restore_state(oracle_snap, start_pos)) {
                err_ = "batch check: the sequential replay's prefix could not be restored, so the batched pass would "
                       "not be comparable with it";
                return false;
            }
        } else {
            reset();
        }
    }

    // Batch phase attribution: the same partition step() prints, for an M-row verify pass, so the
    // batch term's interior is read per phase instead of inferred from the loop's total.
    //
    // The counters are zeroed BEFORE BatchPhases snapshots them.  BatchPhases reports this call's cost as a
    // delta against the values it saw at construction, so zeroing afterwards (the first version of this, #1056)
    // made every [bl] field a difference against the *previous* pass: `core -4.3` and `gemm -355.7` on a real
    // ticket (2026-11-05 01:31) are what that looks like.  The [bs] line never had the problem -- it reads the
    // g_ph it zeroes itself -- so the conclusions drawn from [bs] stand; the [bl] medians between #1056 and this
    // fix do not, and are re-taken.
    const bool phb = phprof_on();
    if (phb) {
        g_ph = StepPhases{};
        for (int i = 0; i < LH_LAYER_N; i++) { g_lh[i] = 0.0; g_lh_n[i] = 0; }
    }
    cp_reset();
    if (cpprof_on()) g_cp.rows = M;
    BatchPhases bph;  // measurement only (NPU_PHASE_PROF=1); prints this call's own delta at the end
    const auto tp_bs = phb ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    auto tp_lap = tp_bs;
    BatchPhaseScope _batch_scope(phb);
    const int i_qkv = idx_of(Q4NX_DESIGN_QKV);
    const int i_o   = idx_of(Q4NX_DESIGN_O);
    const int i_gu  = idx_of(Q4NX_DESIGN_GU);
    const int i_g   = idx_of(Q4NX_DESIGN_G);
    const int i_u   = idx_of(Q4NX_DESIGN_U);
    const int i_d   = idx_of(Q4NX_DESIGN_D);
    // A model with two attention geometries (e.g. MiniCPM4's sliding/full layers)
    // uses a different QKV/O design per layer type, as step() does.
    const int i_qkv2 = idx_of(Q4NX_DESIGN_QKV2);
    const int i_o2   = idx_of(Q4NX_DESIGN_O2);
    const bool two_widths = (i_qkv2 >= 0);
    const int im    = c_.IM;

    std::vector<float> X((size_t)M * H), XN((size_t)M * H);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < H; k++)
            X[(size_t)m * H + k] = (inject_embeds_ ? inject_embeds_[(size_t)m * H + k]
                                                   : embed_at(tokens[m], k)) *
                                   embed_scale_;
    if (phb) { g_ph.embed = ph_ms(tp_bs); tp_lap = std::chrono::steady_clock::now(); }

    // Deepstack, in the reference's place (transformers Qwen3VLTextModel.forward: `_deepstack_process` is called
    // after `decoder_layer(...)`, for layer_idx < len(sets)). Set `l` of `deepstack_` goes to the image rows of the
    // residual stream after text layer `l` runs. It is a lambda because a MoE layer ends in `continue`, which used to
    // skip the add entirely -- latent for this dense pack, wrong for the Qwen3-VL-*-A3B MoE models.
    auto deepstack_add = [&](int l) {
        if (!deepstack_ || l < 0 || l >= sem_.n_deepstack) return;
        const float* add = deepstack_ + size_t(l) * size_t(deepstack_rows_) * size_t(H);
        const int lo = std::max(0, deepstack_lo_), hi = std::min((int)M, deepstack_hi_);
        for (int r = lo; r < hi; r++)
            for (size_t k = 0; k < (size_t)H; k++)
                X[size_t(r) * (size_t)H + k] += add[size_t(r - lo) * (size_t)H + k];
    };

    std::vector<float> QKV, CTX, O, ACT, HH, D, qkv_row, ctx_row;
    for (int l = 0; l < c_.NL; l++) {
        cur_layer_ = l;   // the layer index the per-layer helpers read
        const bool cached = weight_cached(0);
        const int lt0 = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
        // A GDN layer's host parameters (conv window, alpha/beta, ssm_a/dt, gated
        // norm) live outside the weight cache and are per layer, so load_gdn_layer
        // must run every time; it skips its own big deqs when the BOs are cached.
        const bool gdn_l = (lin_kd_ > 0 && lt0 == Q4NX_LAYER_LINEAR);
        if ((gdn_l || !cached) && !load_layer(l)) {
            if (err_.empty()) err_ = "dequant failed";
            return false;
        }

        const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l] : Q4NX_LAYER_FULL;
        const bool l_full = (lt != Q4NX_LAYER_SLIDING);
        const int l_theta = l_full ? sem_.rope_theta : sem_.rope_theta_sliding;
        const int l_rot   = l_full ? sem_.rotary_dim : sem_.rotary_dim_sliding;
        const bool l_prop = l_full && sem_.proportional_rope;
        const int l_win   = l_full ? 0 : sem_.sliding_window;
        const int i_qkv_l = (two_widths && l_full) ? i_qkv2 : i_qkv;
        const int i_o_l   = (two_widths && l_full) ? i_o2 : i_o;

        // ---- attention ---- (a GDN layer goes through gdn_layer_batch instead) ----
        for (int m = 0; m < M; m++)
            rmsnorm(X.data() + (size_t)m * H, in_norm_[l].data(), H,
                    XN.data() + (size_t)m * H);
        O.assign((size_t)M * H, 0.0f);
        if (lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR) {
            if (!gdn_layer_batch(l, XN, M, O)) { err_ = "gdn batch failed"; return false; }
        } else {
            const Q4nxDesignGeom* g = &designs_[i_qkv_l];
            Q4nxProjections p{};
            p.q = wq_.data(); p.k = wk_.data(); p.v = wv_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble qkv"; return false;
            }
            if (xd_->pb16i() && xd_->role_split_qhalf() >= 0) {
                // Same four launches as step(): [q | gate] halves plus k, v.
                const int nhd = c_.NH * c_.HD, nkv = c_.NKV * c_.HD;
                const Q4nxDesignGeom* gq = &designs_[xd_->role_split_qhalf()];
                const Q4nxDesignGeom* gk = &designs_[xd_->role_split_kv()];
                std::vector<float> qh0, qh1, kvk, kvv;
                npu_gemm_rows(xd_->role_split_qhalf(), XN.data(), M, gq->K, gq->N, nullptr, qh0,
                              10, "attn_q_h0", gq->real_k);
                npu_gemm_rows(xd_->role_split_qhalf(), XN.data(), M, gq->K, gq->N, nullptr, qh1,
                              11, "attn_q_h1", gq->real_k);
                npu_gemm_rows(xd_->role_split_kv(), XN.data(), M, gk->K, gk->N, nullptr, kvk,
                              12, "attn_k", gk->real_k);
                npu_gemm_rows(xd_->role_split_kv(), XN.data(), M, gk->K, gk->N, nullptr, kvv,
                              13, "attn_v", gk->real_k);
                const size_t n_on = (size_t)2 * nhd + (size_t)2 * nkv;
                if (qh0.size() < (size_t)M * gq->N || qh1.size() < (size_t)M * gq->N ||
                    kvk.size() < (size_t)M * gk->N || kvv.size() < (size_t)M * gk->N ||
                    !err_.empty()) {
                    if (err_.empty()) err_ = "pb16i full-attention QKV split returned short";
                    return false;
                }
                QKV.assign((size_t)M * n_on, 0.0f);
                // Each of the four GEMMs wrote M rows of *its* design's N (padded, engine#854), so the
                // reads go through the design's stride -- the same class as O/D/gate-up below.
                const size_t sq = (size_t)gq->N, skv = (size_t)gk->N;
                CpScope _cp(CP_QKV);
                for (int m = 0; m < M; m++) {
                    float* d = QKV.data() + (size_t)m * n_on;
                    memcpy(d, qh0.data() + (size_t)m * sq, (size_t)nhd * sizeof(float));
                    memcpy(d + nhd, qh1.data() + (size_t)m * sq, (size_t)nhd * sizeof(float));
                    memcpy(d + 2 * nhd, kvk.data() + (size_t)m * skv, (size_t)nkv * sizeof(float));
                    memcpy(d + 2 * nhd + nkv, kvv.data() + (size_t)m * skv,
                           (size_t)nkv * sizeof(float));
                }
                cp_count(CP_QKV, (size_t)M * n_on * sizeof(float));
            } else {
                npu_gemm_rows(i_qkv_l, XN.data(), M, g->K, g->N, B_.data(), QKV, 0,
                              "attn_qkv", g->real_k);
            }
            // With attn_output_gate the fused QKV is [q | gate | k | v] and
            // g->q_rows covers q AND the gate, so the per-head width must divide
            // by NH * 2.  Using q_rows/NH gave 512 instead of 256 on the 27B
            // (12288/24), which made the context width 12288 instead of 6144 and
            // fed the out-projection a K that does not exist.
            const int q_heads_w = c_.NH * (attn_gate_ ? 2 : 1);
            const int l_hd = (c_.NH > 0 && q_heads_w > 0) ? g->q_rows / q_heads_w : 0;
            const int l_nkv = (l_hd > 0 && g->k_rows > 0) ? g->k_rows / l_hd : c_.NKV;
            const int nhd = c_.NH * l_hd;
            const int ctx_w = nhd;
            // NPU_QKV_DUMP=<layer>: the four parts of the assembled QKV as the engine sees them,
            // rows 0 and 3, on BOTH paths - the int8 fused launch and the pb16i four-launch
            // split fill the same [q|gate|k|v] layout, so this one site compares them directly.
            if (const char* dl = getenv("NPU_QKV_DUMP")) {
                if (l == atoi(dl)) {
                    const int kw = (g->k_rows > 0) ? g->k_rows : c_.NKV * l_hd;
                    for (int m = 0; m < M; m += (m == 0 ? 3 : 1)) {
                        const float* r = QKV.data() + (size_t)m * g->N;
                        fprintf(stderr, "[qkv-dump] l=%d m=%d", l, m);
                        for (int i = 0; i < 8; i++) fprintf(stderr, " q%d=%.6g", i, (double)r[i]);
                        for (int i = 0; i < 8; i++)
                            fprintf(stderr, " gate%d=%.6g", i, (double)r[nhd + i]);
                        for (int i = 0; i < 8; i++)
                            fprintf(stderr, " k%d=%.6g", i, (double)r[2 * nhd + i]);
                        for (int i = 0; i < 8; i++)
                            fprintf(stderr, " v%d=%.6g", i, (double)r[2 * nhd + kw + i]);
                        fprintf(stderr, "\n");
                    }
                }
            }
            CTX.assign((size_t)M * (size_t)ctx_w, 0.0f);
            std::vector<float> qkv_use, ctx_scratch;
            // One row of attention.  `append` is the K/V cache append, `kvlen` is how many cached
            // positions to read (0 = up to and including `pos`, i.e. causal), and `dst` receives the
            // context -- null when the pass only fills the cache (the noncausal first pass below).
            auto row_attn = [&](int m, float* dst, bool append, int kvlen,
                                std::vector<float>* q_keep = nullptr) -> bool {
                {
                    CpScope _cp(CP_QKV);
                    qkv_row.assign(QKV.begin() + (size_t)m * g->N,
                                   QKV.begin() + (size_t)(m + 1) * g->N);
                    cp_count(CP_QKV, (size_t)g->N * sizeof(float));
                }
                // attn_output_gate: the fused QKV is [q | gate | k | v], so attention()
                // must not see the gate half; the gate multiplies the context after.
                const float* gate_ptr = nullptr;
                const std::vector<float>* in = &qkv_row;
                if (attn_gate_) {
                    const int kvsz = g->k_rows + g->v_rows;
                    qkv_use.assign((size_t)nhd + (size_t)kvsz, 0.0f);
                    {
                        CpScope _cp(CP_GATE);
                        memcpy(qkv_use.data(), qkv_row.data(), (size_t)nhd * sizeof(float));
                        memcpy(qkv_use.data() + nhd, qkv_row.data() + 2 * nhd,
                               (size_t)kvsz * sizeof(float));
                        cp_count(CP_GATE, ((size_t)nhd + (size_t)kvsz) * sizeof(float));
                    }
                    gate_ptr = qkv_row.data() + nhd;
                    in = &qkv_use;
                }
                LhScope _t_attn(LH_ATTN);
                // pos3_ is plane-major (3 x rows): row m's three positions are three strided reads, not a pointer
                // offset, so they are gathered here into the trio attention() expects.
                int trio[3] = {0, 0, 0};
                if (pos3_) {
                    trio[0] = pos3_[m];
                    trio[1] = pos3_[pos3_rows_ + m];
                    trio[2] = pos3_[2 * pos3_rows_ + m];
                }
                if (!attention(*in, start_pos + m, pos3_ ? trio : nullptr, dst ? ctx_row : ctx_scratch,
                               l_hd, l_nkv, l_theta,
                               l_rot, l_prop, l_win, l, append, kvlen, q_keep)) {
                    err_ = "attention failed"; return false;
                }
                if (q_keep) return true;       // prepare only: q is in *q_keep, k/v are in the cache
                if (!dst) return true;
                if (gate_ptr)
                    for (int i = 0; i < nhd; i++)
                        ctx_row[(size_t)i] *= 1.0f / (1.0f + expf(-gate_ptr[i]));
                {
                    CpScope _cp(CP_CTX);
                    memcpy(dst, ctx_row.data(), (size_t)ctx_w * sizeof(float));
                    cp_count(CP_CTX, (size_t)ctx_w * sizeof(float));
                }
                return true;
            };
            // A noncausal pack: every full-attention layer reads the whole prompt, so the cache has to
            // hold all M rows before any of them attends -- a row's keys come from the same layer, and
            // the rows after it only exist once their projections ran.  The GDN layers keep their own
            // causal recurrence (the reference keeps it too), and a sliding layer stays causal.
            // engine#1990: one thread team for the layer's M rows instead of one per row. Every row's
            // position-local half runs first (q, and its k/v appended -- a causal row reads only positions up
            // to its own, so the later rows already being in the cache changes nothing it reads), then all
            // (row, head) pairs are one work-shared loop. NPU_ATTN_ROWS=0 keeps the per-row form for an A/B;
            // NPU_ATTN_CHECK=1 recomputes every row that way and fails the pass on any differing bit.
            static const bool attn_rows_on = [] {
                const char* e = getenv("NPU_ATTN_ROWS");
                return !(e && e[0] == '0');
            }();
            static const bool attn_check = [] {
                const char* e = getenv("NPU_ATTN_CHECK");
                return e && e[0] == '1';
            }();
            if (attn_rows_on && M > 1 && l_hd > 0 && c_.NH % l_nkv == 0) {
                const bool whole = noncausal_attn_ && l_full;
                const size_t qw = (size_t)nhd;
                std::vector<float> qall((size_t)M * qw), qm;
                std::vector<int> Tm((size_t)M), t0m((size_t)M);
                for (int m = 0; m < M; m++) {
                    if (!row_attn(m, nullptr, true, 0, &qm)) return false;
                    if (qm.size() != qw) { err_ = "attention rows: q width"; return false; }
                    memcpy(qall.data() + (size_t)m * qw, qm.data(), qw * sizeof(float));
                    const int T = whole ? start_pos + M : start_pos + m + 1;
                    Tm[(size_t)m] = T;
                    t0m[(size_t)m] = (l_win > 0 && T > l_win) ? T - l_win : 0;
                }
                {
                    LhScope _t_attn(LH_ATTN);
                    const float scale = sem_.attn_scaling_one ? 1.0f : 1.0f / std::sqrt((float)l_hd);
                    const int nt = ((long)M * (start_pos + M) * l_hd >= 8192) ? omp_heads_threads() : 1;
                    q4nx_attn::rows_context(qall.data(), kcache_[l].data(), vcache_[l].data(), M, c_.NH, l_nkv,
                                            l_hd, Tm.data(), t0m.data(), scale, CTX.data(), nt);
                    if (attn_gate_)
                        for (int m = 0; m < M; m++) {
                            const float* gate = QKV.data() + (size_t)m * g->N + nhd;
                            float* c = CTX.data() + (size_t)m * ctx_w;
                            for (int i = 0; i < nhd; i++) c[i] *= 1.0f / (1.0f + expf(-gate[i]));
                        }
                }
                if (attn_check) {
                    std::vector<float> ref((size_t)ctx_w);
                    size_t differing = 0;
                    for (int m = 0; m < M; m++) {
                        if (!row_attn(m, ref.data(), false, Tm[(size_t)m])) return false;
                        const float* c = CTX.data() + (size_t)m * ctx_w;
                        for (int i = 0; i < ctx_w; i++)
                            if (memcmp(&c[i], &ref[(size_t)i], sizeof(float)) != 0) differing++;
                    }
                    fprintf(stderr, "[attncheck] l=%d M=%d start=%d one-team vs per-row: %zu values %s\n", l, M,
                            start_pos, (size_t)M * ctx_w, differing ? "MISMATCH" : "identical");
                    if (differing) {
                        err_ = "attention rows: the one-team form differs from the per-row form";
                        return false;
                    }
                }
            } else if (noncausal_attn_ && l_full) {
                for (int m = 0; m < M; m++)
                    if (!row_attn(m, nullptr, true, 0)) return false;
                for (int m = 0; m < M; m++)
                    if (!row_attn(m, CTX.data() + (size_t)m * ctx_w, false, start_pos + M))
                        return false;
            } else {
                for (int m = 0; m < M; m++)
                    if (!row_attn(m, CTX.data() + (size_t)m * ctx_w, true, 0)) return false;
            }
            const Q4nxDesignGeom* go = &designs_[i_o_l];
            Q4nxProjections po{};
            po.o = wo_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(go, &po, B_.data()) != 0) {
                err_ = "assemble o"; return false;
            }
            npu_gemm_rows(i_o_l, CTX.data(), M, go->K, go->N, B_.data(), O, 1,
                          l_full ? "attn_output" : "ssm_out", go->real_k);
        }
        // The GEMM wrote M rows of the *design's* N, which is H only when H is already on the kernel's
        // column grid (engine#854 pads it otherwise): reading at H put every row after the first at the
        // wrong offset, which is why only position 0 of a prompt came out right (engine#855).
        {
            LhScope _t_res(LH_RES);
            const int nrow = q4nx_pad_n(H);
            for (int m = 0; m < M; m++)
                for (int k = 0; k < H; k++)
                    X[(size_t)m * H + k] += O[(size_t)m * nrow + k] * residual_scale_;
        }

        // ---- MLP ----
        for (int m = 0; m < M; m++)
            rmsnorm(X.data() + (size_t)m * H, post_norm_[l].data(), H,
                    XN.data() + (size_t)m * H);
        if (n_expert_ > 0) {
            // MoE FFN: the router/top-k pick different experts per row, so this
            // stays per-token (the attention GEMMs above are already batched).
            // Grouping rows by expert so each expert's weight read serves the
            // whole batch is the next step -- go_rows is the primitive for it.
            std::vector<float> xn_row, d_row;
            for (int m = 0; m < M; m++) {
                xn_row.assign(XN.begin() + (size_t)m * H,
                              XN.begin() + (size_t)(m + 1) * H);
                // Keep moe_mlp's own reason ("expert gate", "assemble expert down", "MoE designs
                // missing", ...): the bare "moe failed" this used to leave cost a whole ticket when
                // the expert tensors' convention was wrong (engine#1035).
                if (!moe_mlp(l, xn_row, d_row)) { err_ = "moe failed: " + err_; return false; }
                for (int k = 0; k < H; k++)
                    X[(size_t)m * H + k] += d_row[(size_t)k] * residual_scale_;
            }
            deepstack_add(l);   // a MoE layer's tail still owes the layer its set
            continue;
        }
        // ACT is read below at the *fused* design's N, so both branches have to hand it rows at that
        // stride.  The fused branch does that by construction (npu_gemm_rows writes M rows of the
        // design's N); the split branch packs its own buffer at the same width rather than at 2*im, so
        // there is one layout and one stride -- reading a real-width buffer at the design stride is
        // what the review of #885 caught (latent until a container has im % 256 != 0, then an OOB read).
        const size_t act_w = (size_t)q4nx_pad_n(2 * im);
        if (i_gu >= 0) {
            const Q4nxDesignGeom* g = &designs_[i_gu];
            Q4nxProjections p{};
            p.gate = wg_.data(); p.up = wu_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble gu"; return false;
            }
            npu_gemm_rows(i_gu, XN.data(), M, g->K, g->N, B_.data(), ACT, 2,
                          "ffn_gate_up", g->real_k);
        } else if (i_g >= 0 && i_u >= 0) {
            std::vector<float> gv, uv, av((size_t)M * act_w, 0.0f);
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections pg{}; pg.gate = wg_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(gg, &pg, B_.data()) != 0) {
                err_ = "assemble g"; return false;
            }
            npu_gemm_rows(i_g, XN.data(), M, gg->K, gg->N, B_.data(), gv, 2,
                          "ffn_gate", gg->real_k);
            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections pu{}; pu.up = wu_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(gu, &pu, B_.data()) != 0) {
                err_ = "assemble u"; return false;
            }
            npu_gemm_rows(i_u, XN.data(), M, gu->K, gu->N, B_.data(), uv, 3,
                          "ffn_up", gu->real_k);
            // Each GEMM wrote M rows of its own design's N (padded, engine#854), so both are read at
            // their design's stride and packed at act_w -- the same layout the fused branch produces.
            // The pad columns stay 0 (av is zeroed), matching what assemble_B leaves in the fused case.
            for (int m = 0; m < M; m++)
                for (int i = 0; i < im; i++) {
                    av[(size_t)m * act_w + i] = gv[(size_t)m * gg->N + i];
                    av[(size_t)m * act_w + im + i] = uv[(size_t)m * gu->N + i];
                }
            ACT.swap(av);
        } else {
            err_ = "no gate/up design";
            return false;
        }
        HH.assign((size_t)M * (size_t)im, 0.0f);
        {
            // act_w is the stride both branches above leave ACT at; only the first 2*im columns of a
            // row are real (the rest is the kernel's zero column pad), so the elementwise gate*up reads
            // that row in place and writes HH's row in place: no staging copy on either side.  The two
            // copies this replaces were 12.75 MiB of a 15.38 MiB row (NPU_COPY_PROF=1, 2026-11-05).
            for (int m = 0; m < M; m++)
                if (!mlp(ACT.data() + (size_t)m * act_w, HH.data() + (size_t)m * im, im)) {
                    err_ = "mlp failed"; return false;
                }
        }
        {
            const Q4nxDesignGeom* g = &designs_[i_d];
            Q4nxProjections p{};
            p.down = wd_.data();
            if (!xd_->pb16i() && !cached && q4nx_assemble_B(g, &p, B_.data()) != 0) {
                err_ = "assemble d"; return false;
            }
            npu_gemm_rows(i_d, HH.data(), M, g->K, g->N, B_.data(), D, 4,
                          "ffn_down", g->real_k);
        }
        {
            LhScope _t_res(LH_RES);
            const int nrow = q4nx_pad_n(H);   // as above: the design's N, identity when H is on the grid
            for (int m = 0; m < M; m++)
                for (int k = 0; k < H; k++)
                    X[(size_t)m * H + k] += D[(size_t)m * nrow + k] * residual_scale_;
        }

        // Same post-residual hidden-state dump as step() emits as [L], one row per
        // position, so the two paths can be compared layer by layer.
        if (getenv("NPU_INFER_DUMP_LAYERS")) {
            for (int m = 0; m < M; m++) {
                double ms = 0;
                float mmx = 0;
                for (int k = 0; k < H; k++) {
                    const float a = std::fabs(X[(size_t)m * H + k]);
                    ms += a;
                    if (a > mmx) mmx = a;
                }
                fprintf(stderr, "  [LB] m=%d l=%d %.6f %.4f\n", m, l, ms / H, mmx);
            }
        }
        deepstack_add(l);
        // NPU_INFER_DUMP_DIR=<dir> writes this layer's residual stream (M x H floats, row-major) as layer_<l>.f32, so a
        // per-layer cosine against a reference trace is possible at all: NPU_INFER_DUMP_LAYERS only prints mean/absmax,
        // which cannot localise a divergence to an op. TorchLens (Apache-2.0, reference side only) supplies the other
        // half; nothing from it ships here. Off unless the variable is set, and written with plain fopen/fwrite because
        // this runs in the forward's hot loop only when someone asked for it.
        if (const char* dd = getenv("NPU_INFER_DUMP_DIR")) {
            static bool told = false;
            if (!told) {
                told = true;
                std::fprintf(stderr, "  [dump] residual stream per layer -> %s (M=%d H=%d)\n", dd, M, H);
            }
            char path[4096];
            std::snprintf(path, sizeof path, "%s/layer_%02d.f32", dd, l);
            // engine#1726: every one of the three calls can fail, and all three used to fail silently -- a missing
            // directory, an unwritable one, or a full disk left the run printing that it was dumping with no files and
            // no error, and a missing layer_<l>.f32 then reads as "that layer never ran" to whoever analyses the dump.
            // Reported once per failure mode, not per layer: a full disk fails every layer, and 40 identical lines is how
            // the first one gets scrolled past. Still non-fatal -- a diagnostic must not fail a valid run.
            std::string dump_err;
            if (!onebit::npu::write_residual_dump(path, X.data(), (size_t) M * (size_t) H, &dump_err)) {
                static bool told_fail = false;
                if (!told_fail) {
                    told_fail = true;
                    std::fprintf(stderr, "  [dump] %s (no residual dump for this run)\n", dump_err.c_str());
                }
            }
        }
    }

    if (phb) { g_ph.layer = ph_ms(tp_lap); tp_lap = std::chrono::steady_clock::now(); }
    // The next-token logits come from the last row of the batch, after the final
    // norm -- step() applies final_norm_ before run_lm_head, and feeding the raw
    // residual here produced garbage logits (the M=1 self-check caught it).
    std::vector<float> fnorm(H);
    bool ok = true;
    last_residual_.assign(X.begin() + (size_t)(M - 1) * H, X.end());
    if (batch_all_out_) {
        // step_batch_all(): the verifier needs the model's own choice at every drafted position, so every
        // row is normed and the head runs ONCE for the batch (the M=16 twin streams its weights once).
        std::vector<float> norm_all((size_t)M * (size_t)H);
        for (int m = 0; m < M; m++)
            rmsnorm(&X[(size_t)m * H], final_norm_.data(), H, &norm_all[(size_t)m * H]);
        std::copy(norm_all.begin() + (size_t)(M - 1) * H, norm_all.end(), fnorm.begin());
        last_hidden_rows_ = norm_all;          // per-row normed hidden, for a drafter that must advance too
        {
            LhScope _t_head(LH_HEAD_ROWS);
            ok = run_lm_head_rows(norm_all, M, *batch_all_out_);
        }
        logits.assign(batch_all_out_->begin() + (size_t)(M - 1) * (size_t)c_.NV, batch_all_out_->end());
        // engine#1781: keep the batched rows so the oracle can compare all of them, not just the last one's argmax.
        if (batch_check) bat_rows.assign(batch_all_out_->begin(), batch_all_out_->end());
    } else {
        std::vector<float> last(X.begin() + (size_t)(M - 1) * H, X.end());
        rmsnorm(last.data(), final_norm_.data(), H, fnorm.data());
        ok = run_lm_head(fnorm, logits);
    }
    last_hidden_ = fnorm;  // for last_hidden(): the same vector the head reads, last position only
    if (getenv("NPU_INFER_DUMP_LAYERS")) {
        int best[5] = {-1, -1, -1, -1, -1};
        for (int i = 0; i < (int)logits.size(); i++) {
            for (int r = 0; r < 5; r++) {
                if (best[r] < 0 || logits[(size_t)i] > logits[(size_t)best[r]]) {
                    for (int q = 4; q > r; q--) best[q] = best[q - 1];
                    best[r] = i;
                    break;
                }
            }
        }
        fprintf(stderr, "  [TB] M=%d start=%d top:", M, start_pos);
        for (int r = 0; r < 5; r++) fprintf(stderr, " %d=%.4f", best[r], (double)logits[(size_t)best[r]]);
        fprintf(stderr, "\n");
    }
    if (batch_check) {
        int ba = 0;
        for (size_t i = 1; i < logits.size(); i++)
            if (logits[i] > logits[(size_t)ba]) ba = (int)i;
        // engine#1905: name the positions compared, not only the start. Before the snapshot fix this line said
        // `start=1 MATCH`-looking numbers for a comparison that was really "position 1 from scratch" against "position
        // 1 continuing a prefix", and the reader had to infer the range from `start` and `M`.
        std::fprintf(stderr, "[batchcheck] M=%d positions=%d..%d seq_argmax=%d batch_argmax=%d %s\n",
                     M, start_pos, start_pos + M - 1, seq_argmax, ba, seq_argmax == ba ? "MATCH" : "MISMATCH");
        // engine#1781: and the comparison the gate actually asks for -- every row, bitwise.
        if (!seq_rows.empty() && bat_rows.size() == seq_rows.size() && c_.NV > 0) {
            onebit::npu::RowMismatch mm;
            const int differing = onebit::npu::compare_rows_bitwise(seq_rows.data(), bat_rows.data(), M, c_.NV, &mm);
            std::fprintf(stderr, "[batchcheck] rows M=%d identical=%d differing=%d", M, M - differing, differing);
            if (differing > 0)
                std::fprintf(stderr, " first row=%d index=%d seq=0x%08x bat=0x%08x%s", mm.row, mm.index,
                             (unsigned)mm.seq_bits, (unsigned)mm.bat_bits, mm.nan_mismatch ? " (NaN)" : "");
            std::fprintf(stderr, "\n");
        }
    } else if (getenv("NPU_BATCH_DBG")) {
        std::fprintf(stderr, "[batch] M=%d start=%d batched (%d layers, two_widths=%d)\n",
                     M, start_pos, c_.NL, (int)two_widths);
    }
    bph.print(M);
    {
        char cp_what[48];
        snprintf(cp_what, sizeof cp_what, "batch M=%d start=%d", M, start_pos);
        cp_print(cp_what);
    }
    if (phb) {
        // epilogue = final norm + the head rows; head= is run_lm_head_rows' own timer, so
        // epilogue - head is the per-row final norm.  tail must be ~0: it is what the partition misses.
        const double total = ph_ms(tp_bs);
        const double epi = ph_ms(tp_lap);
        const double gsum = g_ph.g_pack + g_ph.g_ql + g_ph.g_wait + g_ph.g_rb;
        const double hsum = g_ph.h_memcpy + g_ph.h_pack + g_ph.h_ql + g_ph.h_wait + g_ph.h_rb;
        fprintf(stderr,
                // host_gemm and rows_max are here because a [bs] line with n=0 is ambiguous without them:
                // npu_gemm_rows() returns to the host GEMM on host_gemm_ BEFORE max_batch_rows() or the
                // blocked path is consulted (q4nx_forward.cpp:2979), and the flag is set only by
                // init_host() (:407) and NPU_INFER_HOST_GEMM (:413). So a row that says host_gemm=1 is a
                // host prefill by construction, not a device launch that went uncounted (engine#1582).
                "[bs] M=%2d start=%4d wall=%7.1f | host_gemm=%d rows_max=%d | embed=%5.1f | layers=%7.1f (gemm pack=%6.1f "
                "ql=%6.2f wait=%7.1f rb=%6.1f sum=%7.1f host=%7.1f n=%ld) | epilogue=%6.1f "
                "(head=%6.1f sum=%6.1f host=%5.1f) | tail=%5.1f\n",
                M, start_pos, total, (int)host_gemm_, max_batch_rows(), g_ph.embed, g_ph.layer, g_ph.g_pack, g_ph.g_ql, g_ph.g_wait,
                g_ph.g_rb, gsum, g_ph.layer - gsum, g_ph.launches, epi, g_ph.h_total, hsum,
                g_ph.h_total - hsum, total - g_ph.embed - g_ph.layer - epi);
        char what[48];
        snprintf(what, sizeof what, "batch M=%d start=%d", M, start_pos);
        lh_dump_layer_host(what, "batch");
    }
    return ok;
}


std::string Q4nxNpuForward::missing_kernel(const std::string& dir, const char* elf_name,
                                           Q4nxDesign role) const {
    if (plan_.loaded())
        return "plan " + plan_path_ + ": this model's plan needs the " + role_tag(role) + " kernel \"" +
               elf_name + "\" in " + dir + " (missing)";
    // engine#1503: a container imported without `--weight-layout b8` is row-major, so it asks for plain kernel names while
    // the only kernels on the box carry the _b8 suffix. Naming the layout instead of only the missing file is the
    // difference between "re-fetch the artifact" and "re-import the pack with a flag" - measured on a Ryzen AI 1.9 pack,
    // where the same container ran correctly with the b8 set and the flag.
    {
        const std::string name(elf_name);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".elf") == 0) {
            const std::string b8 = name.substr(0, name.size() - 4) + "_b8.elf";
            if (file_exists(dir + "/" + b8))
                return "missing full ELF " + dir + "/" + elf_name +
                       " - but the b8 variant \"" + b8 + "\" is there: this container's weight layout is row-major " +
                       "(q4nx_weight_layout absent), so it asks for row-major kernels and the directory holds the b8 " +
                       "family. Re-import the pack with `npu-pack import --weight-layout b8`, or supply row-major ELFs " +
                       "(engine#1503).";
        }
    }
    return "missing full ELF " + dir + "/" + elf_name;
}

int Q4nxNpuForward::max_batch_rows() const {
    // The target knows how wide its own kernels are (the M=16 decode family is what a pb16i container
    // packs images for); host-GEMM mode and an uninitialised forward keep batches trivially small.
    return xd_ ? xd_->max_rows() : 1;
}

// --------------------------------------------------------------------------
// host helpers (float32, matching the Python reference)
// --------------------------------------------------------------------------
float Q4nxNpuForward::rmsnorm(const float* x, const float* w, int n, float* out, const char* tag) const {
    LhScope _tlh(LH_RMS);
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + sem_.eps);
    if (sem_.rmsnorm_offset) {
        // Gemma convention: y = x/rms * (1 + w)
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.0f + w[i]);
    } else {
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    }
    // NPU_INFER_DUMP_OPS: the norm's own output, so a divergence can be put on the norm and not on its layer.
    if (onebit::npu::ops_dump_enabled()) onebit::npu::ops_dump(tag, cur_layer_, 1, out, n);
    return inv;
}

/* Scale-free RMSNorm: Gemma4's v_norm is built with_scale=False. */
float Q4nxNpuForward::rmsnorm_noweight(const float* x, int n, float* out) const {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + sem_.eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv;
    return inv;
}

// ---------------------------------------------------------------------------
// MLA + MoE (deepseek2 / GLM-4.7-Flash)
// ---------------------------------------------------------------------------
// Dequantize one expert's slice out of the pooled [n_expert][tiles][5120] tensor.
// MEASURED (not inferred): the converter's unpack TRANSPOSES, so although the GGUF
// stores {K, N, n_expert} (llama.cpp's create_tensor_gate_up_exps), the Q4NX slice
// is [N][K] -- corr 0.996969 against the GGUF read as [N][K] vs 0.0157 as [K][N].
// So dequant straight into [N][K], which is what assemble_B wants.
bool Q4nxNpuForward::mla_expert(const uint8_t* pool, size_t per_expert, int e,
                                int N, int K, std::vector<float>& out) {
    if (!pool || per_expert == 0 || e < 0) return false;
    // Dequantize straight into `out`: the old path materialized a `kn` temporary
    // and then copied it -- an extra (K*N) allocation plus a full copy per expert,
    // ~960 experts/token.
    out.resize((size_t)N * (size_t)K);
    // Do NOT transpose: both conventions decode straight into the [N][K] weight, but the tile GRID
    // differs with who wrote the container (moe_container.h, engine#1035).  The engine converter
    // writes a plain [N][K] grid over each expert; the FLM-style containers this path was written
    // for transpose each expert, so their slice decodes at (rows=K, cols=N) and that flat order IS
    // the logical [N][K] = [out][in] weight (verified elementwise against the GGUF per expert:
    // reshape(N,K) corr 0.999198 gate / 0.996949 down, transposing -0.001079 / 0.000417).
    // Reading the wrong grid is a silent permutation, not a rounding difference: on a converted
    // Qwen3-Coder-30B expert 0 the converter slice correlates 0.99999 with the GGUF on [N][K] and
    // 0.00855 on [K][N].
    int rows = K, cols = N;
    moe_slice_geometry(moe_format_, N, K, &rows, &cols);
    if (dequant_any(pool + (size_t)e * per_expert, per_expert, rows, cols, out.data()) != 0)
        return false;
    return true;
}

// MLA attention for one layer.  q_a -> norm -> q_b, kv_a -> norm -> k_b/v_b, with
// RoPE applied to the LAST rope_dim_ of each q head and to the single shared
// rope slice carried inside kv_a (LLM_ARCH_DEEPSEEK2's MLA).
bool Q4nxNpuForward::mla_attn(int l, const std::vector<float>& x, std::vector<float>& o,
                              int pos) {
    const int H = c_.H, NH = c_.NH;
    const int qh = qk_nope_ + rope_dim_;         // per-head q width (192+64)
    const int kv_real = kv_lora_ + rope_dim_;    // 576
    const int i_qa = idx_of(Q4NX_DESIGN_QA), i_qb = idx_of(Q4NX_DESIGN_QB);
    const int i_kva = idx_of(Q4NX_DESIGN_KVA), i_o = idx_of(Q4NX_DESIGN_O);
    if (i_qa < 0 || i_qb < 0 || i_kva < 0 || i_o < 0) { err_ = "MLA designs missing"; return false; }

    std::vector<float> qa, qan, q, kva;
    {   // q_a: [H] -> [q_lora]
        const Q4nxDesignGeom* g = &designs_[i_qa];
        Q4nxProjections p{}; p.q = w_qa_.data();
        if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble q_a"; return false; }
        npu_gemm(i_qa, x.data(), g->K, g->N, B_.data(), qa, 0, nullptr, g->real_k);
    }
    qan.resize(q_lora_);
    rmsnorm(qa.data(), mla_qan_.data(), q_lora_, qan.data());
    {   // q_b: [q_lora] -> [NH*qh]
        const Q4nxDesignGeom* g = &designs_[i_qb];
        Q4nxProjections p{}; p.k = w_qb_.data();
        if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble q_b"; return false; }
        npu_gemm(i_qb, qan.data(), g->K, g->N, B_.data(), q, 1, nullptr, g->real_k);
    }
    {   // kv_a: [H] -> [kv_real] (+ zero pad to the kernel's n-tile)
        const Q4nxDesignGeom* g = &designs_[i_kva];
        Q4nxProjections p{}; p.v = w_kva_.data();
        if (!weight_cached(6) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble kv_a"; return false; }
        npu_gemm(i_kva, x.data(), g->K, g->N, B_.data(), kva, 6, nullptr, g->real_k);
    }
    if ((int)kva.size() < kv_real) { err_ = "kv_a short"; return false; }
    std::vector<float> kl(kv_lora_);
    rmsnorm(kva.data(), mla_kvan_.data(), kv_lora_, kl.data());

    // RoPE on the shared slice and on each q head's tail.
    std::vector<float> kr(rope_dim_);
    for (int i = 0; i < rope_dim_; i++) kr[i] = kva[(size_t)kv_lora_ + i];
    if (!mla_rope(q, NH, qh, qk_nope_, kr, pos)) { err_ = "mla rope"; return false; }

    // Cache the reconstructed k_nope / shared k_rope / v for this position.
    const int kh = qk_nope_, vh = v_head_;
    std::vector<float>& kn = mla_kn_[l];
    std::vector<float>& krc = mla_kr_[l];
    std::vector<float>& vc = mla_v_[l];
    const size_t old = kn.size() / ((size_t)NH * kh);
    kn.resize((old + 1) * (size_t)NH * kh);
    vc.resize((old + 1) * (size_t)NH * vh);
    krc.resize((old + 1) * (size_t)rope_dim_);
    // k_b[h] : [kh][kv_lora] (stored padded to a 32-row tile, so read kh real rows)
    for (int h = 0; h < NH; h++) {
        // k_b: Q4NX tile grid [kv_lora][nope padded to 256], indexed flat[n*kv_lora+k].
        // v_b: the Q4NX tiles v_b with cols=256 while its LOGICAL row is 512 wide, so
        // the per-head [512][256] tile grid is NOT the logical matrix.  Decoding the
        // whole tensor at (rows=10240, cols=256) reproduces the GGUF's flat order
        // exactly (corr 0.996860 vs the 0.996870 a faithful Q4_1 requant scores),
        // i.e. the logical index is (h*256+n)*512 + k = out-major.  Indexing flat
        // [k*vh+n] instead permuted the whole MLA value path; the earlier check that
        // "pinned" it compared a single SUM, which a permutation can still match.
        const float* wk = &w_kb_[(size_t)h * (size_t)kv_lora_ * (size_t)kb_rows_];
        const float* wv = &w_vb_[(size_t)h * (size_t)kv_lora_ * (size_t)vh];
        float* dk = &kn[old * (size_t)NH * kh + (size_t)h * kh];
        float* dv = &vc[old * (size_t)NH * vh + (size_t)h * vh];
        for (int n = 0; n < kh; n++) {
            float s = 0.0f;
            for (int k = 0; k < kv_lora_; k++) s += wk[(size_t)n * kv_lora_ + k] * kl[k];
            dk[n] = s;
        }
        for (int n = 0; n < vh; n++) {
            float s = 0.0f;
            for (int k = 0; k < kv_lora_; k++) s += wv[(size_t)n * kv_lora_ + k] * kl[k];
            dv[n] = s;
        }
    }
    for (int i = 0; i < rope_dim_; i++) krc[old * (size_t)rope_dim_ + i] = kr[i];

    // Scores and softmax over the cache, then the weighted sum of v.
    const int T = pos + 1;
    const float scale = 1.0f / std::sqrt((float)(qh));
    std::vector<float> ctx((size_t)NH * vh, 0.0f);
    std::vector<float> score(T);
    for (int h = 0; h < NH; h++) {
        const float* qn = &q[(size_t)h * qh];
        const float* qr = qn + qk_nope_;
        float mx = -1e30f;
        for (int t = 0; t < T; t++) {
            const float* kt = &kn[(size_t)t * NH * kh + (size_t)h * kh];
            const float* rt = &krc[(size_t)t * rope_dim_];
            float s = 0.0f;
            for (int i = 0; i < kh; i++) s += qn[i] * kt[i];
            for (int i = 0; i < rope_dim_; i++) s += qr[i] * rt[i];
            score[t] = s * scale;
            if (score[t] > mx) mx = score[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < T; t++) { score[t] = std::exp(score[t] - mx); sum += score[t]; }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float* out = &ctx[(size_t)h * vh];
        for (int t = 0; t < T; t++) {
            const float pm = score[t] * inv;
            const float* vt = &vc[(size_t)t * NH * vh + (size_t)h * vh];
            for (int i = 0; i < vh; i++) out[i] += pm * vt[i];
        }
    }
    {   // o_proj: [NH*v_head] -> [H]
        const Q4nxDesignGeom* g = &designs_[i_o];
        Q4nxProjections p{}; p.o = w_o_.data();
        if (!weight_cached(7) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble o"; return false; }
        npu_gemm(i_o, ctx.data(), g->K, g->N, B_.data(), o, 7, nullptr, g->real_k);
    }
    return true;
}

// RoPE for MLA.  llama.cpp's llama_model_rope_type() returns LLAMA_ROPE_TYPE_NORM
// for LLM_ARCH_DEEPSEEK2 ("normal RoPE, operating on pairs of CONSECUTIVE head
// values"), so the rotation pairs (2i, 2i+1) -- NOT (i, i+half) as the paired/NeoX
// convention would.  Using the paired form here was a bug of exactly the class
// that broke MiniCPM4/MiniCPM5.
bool Q4nxNpuForward::mla_rope(std::vector<float>& q, int NH, int qh, int nope,
                              std::vector<float>& kr, int pos) {
    const int half = rope_dim_ / 2;
    std::vector<float> cs(half), sn(half);
    for (int i = 0; i < half; i++) {
        const double inv = 1.0 / std::pow((double)sem_.rope_theta, (double)i / (double)half);
        const double ang = (double)pos * inv;
        cs[i] = (float)std::cos(ang); sn[i] = (float)std::sin(ang);
    }
    auto rot = [&](float* v) {
        for (int i = 0; i < half; i++) {
            const float a = v[2 * i], b = v[2 * i + 1];
            v[2 * i] = a * cs[i] - b * sn[i];
            v[2 * i + 1] = a * sn[i] + b * cs[i];
        }
    };
    for (int h = 0; h < NH; h++) rot(&q[(size_t)h * qh + nope]);
    rot(kr.data());
    return true;
}

// MoE FFN: router (host, N=64 cannot tile) -> gating -> top-k -> per-expert G/U/D
// on the NPU -> weighted sum, plus the shared expert.
bool Q4nxNpuForward::moe_mlp(int l, const std::vector<float>& xn, std::vector<float>& d) {
    const int H = c_.H;
    const int i_g = idx_of(Q4NX_DESIGN_G), i_u = idx_of(Q4NX_DESIGN_U), i_d = idx_of(Q4NX_DESIGN_D);
    // The FFN needs D, and gate/up either split (G, U) or batched (GB, UB: one GEMM over the top-k
    // experts).  A fused-GU plan is the common case on a converted MoE container and has neither G
    // nor U, only GB/UB, so requiring G/U unconditionally refused a model whose kernels were all
    // there (engine#1035: "MoE designs missing" on a 6-design load).
    const int i_gb = idx_of(Q4NX_DESIGN_GB), i_ub = idx_of(Q4NX_DESIGN_UB);
    const bool expert_batched = i_gb >= 0 && i_ub >= 0 && !getenv("NPU_NO_EXPERT_BATCH");
    if (i_d < 0 || (!expert_batched && (i_g < 0 || i_u < 0))) {
        err_ = expert_batched ? std::string("MoE designs missing (D)")
                              : std::string("MoE designs missing (need D, and G+U or the batched GB/UB)");
        return false;
    }
    d.assign(H, 0.0f);

    // Router.  llama.cpp: probs = sigmoid(logits) UNBIASED; selection_probs =
    // probs + exp_probs_b (the bias is added AFTER the sigmoid and used ONLY for
    // the top-k selection -- "leave probs unbiased as it's later used to get
    // expert weights").  Adding the bias to the logit before the sigmoid, as I
    // had, corrupts both the selection and the weights.
    std::vector<float> probs(n_expert_), selprobs(n_expert_), logits(n_expert_);
    for (int e = 0; e < n_expert_; e++) {
        const float* row = &moe_router_[(size_t)e * H];
        float s = 0.0f;
        for (int i = 0; i < H; i++) s += row[i] * xn[i];
        logits[e] = s;
        // llama.cpp's deepseek2 expert_gating_func: 2 = sigmoid, 1 = softmax.
        probs[e] = expert_gating_sigmoid_ ? (1.0f / (1.0f + std::exp(-s))) : s;
    }
    if (!expert_gating_sigmoid_) {
        float mx = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (int e = 0; e < n_expert_; e++) { probs[e] = std::exp(probs[e] - mx); sum += probs[e]; }
        for (int e = 0; e < n_expert_; e++) probs[e] /= sum;
    }
    for (int e = 0; e < n_expert_; e++)
        selprobs[e] = probs[e] + (moe_router_b_.empty() ? 0.0f : moe_router_b_[e]);
    std::vector<int> idx(n_expert_);
    for (int e = 0; e < n_expert_; e++) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + top_k_, idx.end(),
                      [&](int a, int b) { return selprobs[a] > selprobs[b]; });
    std::vector<int> sel(idx.begin(), idx.begin() + top_k_);
    // Weights come from the UNBIASED probs, sum-normalised, then scaled.
    std::vector<float> r(n_expert_, 0.0f);
    for (int e : sel) r[e] = probs[e];
    // expert_weights_norm: llama.cpp's deepseek2 normalises the SELECTED weights by
    // their SUM (ggml_div(weights, ggml_sum_rows(weights))), not by their L2 norm,
    // and only then applies expert_weights_scale.  Using L2 here was a bug.
    float wsum = 0.0f;
    for (int e : sel) wsum += r[e];
    if (wsum > 0.0f) for (int e : sel) r[e] = r[e] / wsum * expert_w_scale_;

    // Per-layer MoE dump, laid out to compare directly against llama.cpp's
    // llama-eval-callback sums (ffn_moe_logits/probs/weights, ffn_moe_*).
    const bool dump = getenv("NPU_INFER_DUMP_MOE") && std::atoi(getenv("NPU_INFER_DUMP_MOE")) == l;
    if (dump) {
        auto sm = [](const std::vector<float>& v) { double s = 0; for (float f : v) s += f; return s; };
        double rawsum = 0; for (float f : logits) rawsum += f;
        double xnsq = 0; for (float f : xn) xnsq += (double)f * f;
        fprintf(stderr, "  [moe] L%d xn_sum=%.6f xn_rms=%.6f router_rms=%.6f\n",
                l, sm(xn), std::sqrt(xnsq / H),
                [&] { double s = 0; for (float f : moe_router_) s += (double)f * f;
                      return std::sqrt(s / moe_router_.size()); }());
        fprintf(stderr, "  [moe] L%d logits=%.6f probs=%.6f biased=%.6f sel=[%d,%d,%d,%d] "
                        "w=[%.6f,%.6f,%.6f,%.6f] wsum_raw=%.6f scaled=%.6f\n",
                l, rawsum, sm(probs), sm(selprobs), sel[0], sel[1], sel[2], sel[3],
                r[sel[0]], r[sel[1]], r[sel[2]], r[sel[3]], wsum, wsum * expert_w_scale_);
    }
    std::vector<float> g, u, dd, go, uo;
    double moe_g = 0, moe_u = 0, moe_sw = 0, moe_dn = 0;
    if (expert_batched) {
        // One gate GEMM and one up GEMM over ALL selected experts (they share xn),
        // instead of 2*top_k dispatches.  Expert j's weights occupy columns
        // [j*IM, (j+1)*IM) of a [K][top_k*IM] B; the design's N is wider, so
        // npu_gemm packs only the real columns and the tail stays zero.
        const int K = H, tk = (int)sel.size(), N = tk * expert_im_;
        static std::vector<float> Bg, Bu;   // reused across layers; fully written below
        Bg.resize((size_t)K * N);
        Bu.resize((size_t)K * N);
        std::vector<std::vector<float>> gs(tk), us(tk), ds(tk);
        for (int j = 0; j < tk; j++) {
            if (!mla_expert(exp_g_, exp_g_bytes_, sel[j], expert_im_, K, gs[j])) { err_ = "expert gate"; return false; }
            if (!mla_expert(exp_u_, exp_u_bytes_, sel[j], expert_im_, K, us[j])) { err_ = "expert up"; return false; }
            if (!mla_expert(exp_d_, exp_d_bytes_, sel[j], K, expert_im_, ds[j])) { err_ = "expert down"; return false; }
        }
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < tk; j++) {
            for (int n = 0; n < expert_im_; n++) {
                const float* rg = &gs[j][(size_t)n * K];
                const float* ru = &us[j][(size_t)n * K];
                for (int i = 0; i < K; i++) {
                    Bg[(size_t)i * N + (size_t)j * expert_im_ + n] = rg[i];
                    Bu[(size_t)i * N + (size_t)j * expert_im_ + n] = ru[i];
                }
            }
        }
        std::vector<float> go_all, uo_all;
        npu_gemm(i_gb, xn.data(), designs_[i_gb].K, N, Bg.data(), go_all, -1, nullptr, designs_[i_gb].real_k);
        npu_gemm(i_ub, xn.data(), designs_[i_ub].K, N, Bu.data(), uo_all, -1, nullptr, designs_[i_ub].real_k);
        for (int j = 0; j < tk; j++) {
            const int e = sel[j];
            std::vector<float> act((size_t)expert_im_);
            for (int i = 0; i < expert_im_; i++) {
                const float gv = go_all[(size_t)j * expert_im_ + i];
                act[i] = (gv / (1.0f + std::exp(-gv))) * uo_all[(size_t)j * expert_im_ + i];
            }
            const Q4nxDesignGeom* gd = &designs_[i_d];
            std::vector<float> ed;
            Q4nxProjections p{}; p.down = ds[j].data();
            if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble expert down"; return false; }
            npu_gemm(i_d, act.data(), gd->K, gd->N, B_.data(), ed, -1, nullptr, gd->real_k);
            if (dump) {
                for (int i = 0; i < expert_im_; i++) { moe_g += go_all[(size_t)j * expert_im_ + i]; moe_u += uo_all[(size_t)j * expert_im_ + i]; }
                for (float f : act) moe_sw += f;
                for (float f : ed) moe_dn += f;
            }
            for (int i = 0; i < H; i++) d[i] += r[e] * ed[i];
        }
    } else {
    for (int e : sel) {
        if (!mla_expert(exp_g_, exp_g_bytes_, e, expert_im_, H, g)) { err_ = "expert gate"; return false; }
        if (!mla_expert(exp_u_, exp_u_bytes_, e, expert_im_, H, u)) { err_ = "expert up"; return false; }
        if (!mla_expert(exp_d_, exp_d_bytes_, e, H, expert_im_, dd)) { err_ = "expert down"; return false; }
        // gate/up are GEMMs against xn (the dequantized matrices are weights, not
        // outputs): run them on the NPU through the G/U designs.
        {
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections p{}; p.gate = g.data();
            if (q4nx_assemble_B(gg, &p, B_.data()) != 0) { err_ = "assemble expert gate"; return false; }
            npu_gemm(i_g, xn.data(), gg->K, gg->N, B_.data(), go, -1, nullptr, gg->real_k);
        }
        {
            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections p{}; p.up = u.data();
            if (q4nx_assemble_B(gu, &p, B_.data()) != 0) { err_ = "assemble expert up"; return false; }
            npu_gemm(i_u, xn.data(), gu->K, gu->N, B_.data(), uo, -1, nullptr, gu->real_k);
        }
        std::vector<float> act((size_t)expert_im_);
        for (int i = 0; i < expert_im_; i++)
            act[i] = (go[i] / (1.0f + std::exp(-go[i]))) * uo[i];
        const Q4nxDesignGeom* gd = &designs_[i_d];
        std::vector<float> ed;
        Q4nxProjections p{}; p.down = dd.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble expert down"; return false; }
        npu_gemm(i_d, act.data(), gd->K, gd->N, B_.data(), ed, -1, nullptr, gd->real_k);
        if (dump) {
            for (float f : go) moe_g += f;
            for (float f : uo) moe_u += f;
            for (float f : act) moe_sw += f;
            for (float f : ed) moe_dn += f;
        }
        for (int i = 0; i < H; i++) d[i] += r[e] * ed[i];
    }
    }
    double moe_d_sum = 0; for (float f : d) moe_d_sum += f;
    // Shared expert, always on.
    double shexp_sum = 0;
    if (!share_g_.empty()) {
        std::vector<float> act((size_t)expert_im_);
        for (int i = 0; i < expert_im_; i++) {
            const float* rg = &share_g_[(size_t)i * H];
            const float* ru = &share_u_[(size_t)i * H];
            float sg = 0.0f, su = 0.0f;
            for (int k = 0; k < H; k++) { sg += rg[k] * xn[k]; su += ru[k] * xn[k]; }
            act[i] = (sg / (1.0f + std::exp(-sg))) * su;
        }
        const Q4nxDesignGeom* gd = &designs_[i_d];
        Q4nxProjections p{}; p.down = share_d_.data();
        if (!weight_cached(5) && q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble shared down"; return false; }
        std::vector<float> sd;
        npu_gemm(i_d, act.data(), gd->K, gd->N, B_.data(), sd, 5, nullptr, gd->real_k);
        if (dump) {
            double ss = 0; for (float f : act) ss += f;
            fprintf(stderr, "  [moe] L%d expert gate=%.6f up=%.6f swiglu=%.6f down=%.6f weighted=%.6f | "
                            "shexp_swiglu=%.6f\n", l, moe_g, moe_u, moe_sw, moe_dn, moe_d_sum, ss);
        }
        for (float f : sd) shexp_sum += f;
        // Qwen MoE (qwen35moe) gates the shared expert with sigmoid(x . g), a [H]
        // vector; deepseek2 has no such gate (share_router_ empty -> scale 1).
        float sgate = 1.0f;
        if (!share_router_.empty()) {
            float g = 0.0f;
            for (int i = 0; i < H; i++) g += share_router_[i] * xn[i];
            sgate = 1.0f / (1.0f + std::exp(-g));
        }
        for (int i = 0; i < H; i++) d[i] += sgate * sd[i];
    }
    if (dump) {
        double tot = 0; for (float f : d) tot += f;
        fprintf(stderr, "  [moe] L%d moe_out=%.6f shexp=%.6f ffn_out=%.6f\n", l, moe_d_sum, shexp_sum, tot);
    }
    return true;
}

// The leading dense layer (leading_dense_block_count=1): DG/DU/DD designs.
bool Q4nxNpuForward::dense_mlp(int l, const std::vector<float>& xn, std::vector<float>& d) {
    const int H = c_.H;
    const int i_dg = idx_of(Q4NX_DESIGN_DG), i_du = idx_of(Q4NX_DESIGN_DU), i_dd = idx_of(Q4NX_DESIGN_DD);
    if (i_dg < 0 || i_du < 0 || i_dd < 0) { err_ = "dense designs missing"; return false; }
    std::vector<float> g, u;
    {
        const Q4nxDesignGeom* gd = &designs_[i_dg];
        Q4nxProjections p{}; p.gate = dense_g_.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense gate"; return false; }
        npu_gemm(i_dg, xn.data(), gd->K, gd->N, B_.data(), g, -1, nullptr, gd->real_k);
    }
    {
        const Q4nxDesignGeom* gd = &designs_[i_du];
        Q4nxProjections p{}; p.up = dense_u_.data();
        if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense up"; return false; }
        npu_gemm(i_du, xn.data(), gd->K, gd->N, B_.data(), u, -1, nullptr, gd->real_k);
    }
    std::vector<float> act((size_t)dense_im_);
    for (int i = 0; i < dense_im_; i++)
        act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    if (getenv("NPU_INFER_DUMP_DENSE")) {
        auto sm = [](const std::vector<float>& v) {
            double s = 0; for (float f : v) s += f; return s;
        };
        fprintf(stderr, "  [dense] gate n=%zu sum=%.6f | up n=%zu sum=%.6f | act n=%zu sum=%.6f\n",
                g.size(), sm(g), u.size(), sm(u), act.size(), sm(act));
    }
    const Q4nxDesignGeom* gd = &designs_[i_dd];
    Q4nxProjections p{}; p.down = dense_d_.data();
    if (q4nx_assemble_B(gd, &p, B_.data()) != 0) { err_ = "assemble dense down"; return false; }
    npu_gemm(i_dd, act.data(), gd->K, gd->N, B_.data(), d, -1, nullptr, gd->real_k);
    return true;
}

bool Q4nxNpuForward::attention(const std::vector<float>& qkv, int pos, const int* pos3,
                               std::vector<float>& ctx, int head_dim,
                               int n_kv_heads, float theta, int rotary_dim,
                               bool proportional, int window,
                               int kv_layer, bool append_kv, int kv_len, std::vector<float>* q_out) {
    const int NH = c_.NH;
    const int HD = head_dim;      /* per layer type (Gemma4: 256 or 512) */
    const int NKV = n_kv_heads;
    const int gqa = NH / NKV;

    std::vector<float> q((size_t)NH * HD), k((size_t)NKV * HD), v((size_t)NKV * HD);
    for (int i = 0; i < NH * HD; i++) q[i] = qkv[i];
    for (int i = 0; i < NKV * HD; i++) k[i] = qkv[NH * HD + i];
    for (int i = 0; i < NKV * HD; i++) v[i] = qkv[NH * HD + NKV * HD + i];
    // Qwen2.5-style attention biases: the q/k/v projections carry a per-output
    // bias (config attention_bias=true). Added before qk-norm/RoPE.
    if (sem_.attention_bias) {
        for (int i = 0; i < NH * HD; i++) q[i] += q_bias_[cur_layer_][i];
        for (int i = 0; i < NKV * HD; i++) k[i] += k_bias_[cur_layer_][i];
        for (int i = 0; i < NKV * HD; i++) v[i] += v_bias_[cur_layer_][i];
    }

    // per-head RMSNorm on q and k, only for architectures that have it.  A KV-shared
    // layer (Gemma4) computed no k at all, so its k is never normalized or used.
    if (sem_.qk_norm) {
        std::vector<float> t(HD);
        for (int h = 0; h < NH; h++) {
            rmsnorm(&q[(size_t)h * HD], q_norm_[cur_layer_].data(), HD, t.data(), "q_norm");
            memcpy(&q[(size_t)h * HD], t.data(), sizeof(float) * HD);
        }
        if (append_kv) {
            for (int h = 0; h < NKV; h++) {
                rmsnorm(&k[(size_t)h * HD], k_norm_[cur_layer_].data(), HD, t.data(), "k_norm");
                memcpy(&k[(size_t)h * HD], t.data(), sizeof(float) * HD);
            }
        }
    }
    // Gemma4 normalises v with a SCALE-FREE RMSNorm (its v_norm has no weight).
    if (sem_.v_norm && append_kv) {
        std::vector<float> tv(HD);
        for (int h = 0; h < NKV; h++) {
            rmsnorm_noweight(&v[(size_t)h * HD], HD, tv.data());
            memcpy(&v[(size_t)h * HD], tv.data(), sizeof(float) * HD);
        }
    }

    // RoPE: half-split over the first `rot` channels (rot == HD means full).
    // Gemma4's full layers use "proportional" RoPE: the pairing spans the WHOLE
    // head_dim, but the frequencies past rot/2 are zero, so those pairs are the
    // identity (_compute_proportional_rope_parameters).
    const int rot = (rotary_dim > 0 && rotary_dim <= HD) ? rotary_dim : HD;
    {
        const int half = proportional ? HD / 2 : rot / 2;
        std::vector<float> cos_(half), sin_(half);
        // Dynamic NTK: the base moves with the position, so it is one pow per position rather than one per dim pair.
        // Below the model's original window this returns theta unchanged, which is what makes a short prompt
        // bit-identical to the same model served without the scaling.
        // engine#1792 review: the exponent is HF's `dim / (dim - 2)` over the ROTARY width (`dim = int(head_dim *
        // partial_rotary_factor)`), which is `rot` here -- passing `HD` was wrong for any model that declares dynamic
        // scaling together with partial_rotary_factor < 1. Unreachable in today's catalog, so this is a correctness
        // constant rather than a behaviour change for any pack that exists.
        const double rope_theta = q4nx_rope_dynamic_base((double)theta, &rope_dynamic_, (double)pos, rot);
        for (int i = 0; i < half; i++) {
            double inv;
            if (proportional) {
                inv = 1.0 / std::pow(rope_theta, 2.0 * (double)i / (double)HD);
                if (i >= rot / 2) inv = 0.0;
            } else {
                inv = 1.0 / std::pow(rope_theta, (double)i / (double)half);
                // LongRoPE (MiniCPM4): ggml's rope applies the factor as
                // `rope_yarn(theta / ff, ...)` (ggml-cpu/ops.cpp
                // ggml_rope_cache_init), i.e. the angle is DIVIDED by
                // freq_factors.  So divide the inv_freq here.  Multiplying (the
                // engine route's 599ac77) blows the high dims up by up to 31x
                // and yields garbage logits.  NPU_INFER_ROPE_FACTOR_MUL=1 keeps
                // the old multiply for A/B.
                if (i < (int)rope_scale_.size()) {
                    const char* mul = getenv("NPU_INFER_ROPE_FACTOR_MUL");
                    if (mul && mul[0] == '1') inv *= (double)rope_scale_[i];
                    else                       inv /= (double)rope_scale_[i];
                }
                // llama3 (Llama-3.1/3.2/3.3): the wavelength-band rule, on the FREQUENCY, so it costs
                // nothing per token.  No-op unless the config declares rope_type "llama3".
                inv = q4nx_rope_llama3_inv_freq(inv, &rope_llama3_);
            }
            // Interleaved MRoPE: each rotary frequency rotates by its own axis's position (t, h or w). With the
            // three planes equal this is exactly `pos * inv`, which is why text-only prompts are bit-identical
            // to plain RoPE (measured: 0 angle difference over positions 0..262143, engine#1311/TQVL).
            const double ang = (double)(pos3 ? pos3[mrope_plane(i)] : pos) * inv;
            cos_[i] = (float)std::cos(ang);
            sin_[i] = (float)std::sin(ang);
        }
        auto rot_half = [&](float* x) {
            // pairs (i, i + half) inside the rotated block; the rest passes through
            for (int i = 0; i < half; i++) {
                const float x1 = x[i], x2 = x[i + half];
                x[i] = x1 * cos_[i] - x2 * sin_[i];
                x[i + half] = x1 * sin_[i] + x2 * cos_[i];
            }
        };
        for (int h = 0; h < NH; h++) rot_half(&q[(size_t)h * HD]);
        for (int h = 0; h < NKV; h++) rot_half(&k[(size_t)h * HD]);
    }

    // append to cache (a KV-shared layer reuses its owner's, which the owner has
    // already appended earlier in this same step).
    if (append_kv) {
        kcache_[kv_layer].insert(kcache_[kv_layer].end(), k.begin(), k.end());
        vcache_[kv_layer].insert(vcache_[kv_layer].end(), v.begin(), v.end());
    }

    // engine#1990: a batched pass asks for the position-local half only -- the q this row attends with, and its
    // k/v in the cache -- and runs the read for all of its rows in one thread team (q4nx_attn::rows_context).
    if (q_out) {
        q_out->swap(q);
        return true;
    }

    // kv_len > 0 is a noncausal read: attend over that many cached positions instead of only the ones
    // up to `pos`.  Everything above (rope, qk-norm, the cache append) is position-local and unchanged.
    const int T = kv_len > 0 ? kv_len : pos + 1;
    // Gemma4 sets attention scaling to 1.0 (q/k carry an RMSNorm); everyone else
    // uses the usual 1/sqrt(head_dim).
    const float scale = sem_.attn_scaling_one ? 1.0f : 1.0f / std::sqrt((float)HD);
    // Sliding-window attention: ignore positions older than the window.
    const int t0 = (window > 0 && T > window)
                       ? T - window : 0;
    ctx.assign((size_t)NH * HD, 0.0f);
    // Heads are the unit the cores split here, and they are independent by construction: a head writes
    // only its own ctx rows and reads its own q plus the read-only KV cache.  The score buffer is
    // therefore per thread, and the ordering of nothing else changes.  The `if` keeps short contexts
    // serial -- one team per layer per token is not worth it for a handful of cached positions -- and
    // `omp_heads_threads()` is this file's existing convention (it leaves a core for the array-submit
    // thread and the serve's HTTP thread).
    const int nt = omp_heads_threads();
#pragma omp parallel num_threads(nt) if (nt > 1 && NH > 1 && (long)T * HD >= 8192)
    {
        std::vector<float> score(T);
#pragma omp for schedule(static)
        for (int h = 0; h < NH; h++)
            q4nx_attn::head_context(&q[(size_t)h * HD], kcache_[kv_layer].data(), vcache_[kv_layer].data(), NKV,
                                    h / gqa, HD, t0, T, scale, score.data(), &ctx[(size_t)h * HD]);
    }
    return true;
}

// SiLU(gate) * up from either a fused GU result or separate G and U results.
bool Q4nxNpuForward::mlp(const std::vector<float>& gu, std::vector<float>& h, int im) {
    h.resize(im);
    return mlp(gu.data(), h.data(), im);
}

bool Q4nxNpuForward::mlp(const float* gu, float* h, int im) {
    LhScope _tlh(LH_MLP);
    const int gelu = (sem_.mlp_act == Q4NX_ACT_GELU);
    for (int i = 0; i < im; i++) {
        const float g = gu[i];
        const float u = gu[im + i];
        float a;
        if (gelu) {
            // tanh approximation (gelu_pytorch_tanh), as HF uses for Gemma
            const float c = 0.7978845608028654f;   // sqrt(2/pi)
            a = 0.5f * g * (1.0f + std::tanh(c * (g + 0.044715f * g * g * g)));
        } else {
            a = g / (1.0f + std::exp(-g));         // SiLU
        }
        h[i] = a * u;
    }
    return true;
}

// --------------------------------------------------------------------------
// step
// --------------------------------------------------------------------------
bool Q4nxNpuForward::step(int token, int pos, std::vector<float>& logits, int cache_pos) {
    if (!ready_) { err_ = "not initialised"; return false; }
    // Generation is causal by construction (one position at a time, keys only in the past), so a pack
    // whose full-attention layers read the whole prompt cannot be run this way.  Refused by name
    // rather than answered with a mask the pack was not trained with.
    if (noncausal_attn_) {
        err_ = "attention.mode noncausal_full_attention is a prompt-encoding mode and step() generates: "
               "this model's full-attention layers read the whole prompt, so a one-token-at-a-time pass "
               "would use a mask it was not trained with (use step_batch() with the whole prompt at "
               "position 0)";
        return false;
    }
    const long step_d0 = xd_ ? xd_->launches_decode() : 0;
    // Phase attribution (NPU_PHASE_PROF=1): named phases summing to this step's wall.
    const bool ph = phprof_on();
    const auto tp_step = ph ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
    auto tp_lap = tp_step;
    if (ph) {
        g_ph = StepPhases{};
        for (int i = 0; i < LH_LAYER_N; i++) { g_lh[i] = 0.0; g_lh_n[i] = 0; }
    }
    cp_reset();
    const int H = c_.H;
    if (token < 0 || token >= c_.NV) { err_ = "token out of range"; return false; }

    const int i_qkv = idx_of(Q4NX_DESIGN_QKV);
    const int i_o = idx_of(Q4NX_DESIGN_O);
    const int i_gu = idx_of(Q4NX_DESIGN_GU);
    const int i_g = idx_of(Q4NX_DESIGN_G);
    const int i_u = idx_of(Q4NX_DESIGN_U);
    const int i_d = idx_of(Q4NX_DESIGN_D);
    const int i_qkv2 = idx_of(Q4NX_DESIGN_QKV2);   /* 2nd attention type (Gemma4) */
    const int i_o2 = idx_of(Q4NX_DESIGN_O2);

    std::vector<float> x(H), xn(H), o(H), d(H), h, ctx;
    // The stored embedding is PRE-SCALED by the architecture's embedding scale
    // (Gemma4 bakes sqrt(hidden_size) into the tensor), so no scale is applied
    // here; the LM head divides it back out (see run_lm_head).
    for (int k = 0; k < H; k++)
        x[k] = embed_at(token, k) * embed_scale_;
    if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
        double s = 0;
        float mx = 0;
        for (int k = 0; k < H; k++) {
            const float a = std::fabs(x[(size_t)k]);
            s += a;
            if (a > mx) mx = a;
        }
        fprintf(stderr, "  [L] -1 %.6f %.4f\n", s / H, mx);
    }

    // Gemma4 needs this token's per-layer inputs before the layer loop.
    if (ple_dim_ > 0) build_ple_inputs(token);
    if (ph) { g_ph.embed = ph_ms(tp_step); tp_lap = std::chrono::steady_clock::now(); }

    for (int l = 0; l < c_.NL; l++) {
        cur_layer_ = l;
        // Dense models: once this layer's projections are packed (after the
        // first token), the f32 weights are no longer read, so skip the per-token
        // Q4NX dequant.  GDN/MLA/PLE load members used outside the GEMMs, so they
        // keep loading every token.
        const bool dense_weights_cached = lin_kd_ == 0 && !mla_ && ple_dim_ == 0 && n_expert_ == 0 && weight_cached(0);
        {
            LhScope _tlh(LH_LOAD);
        if (!dense_weights_cached && !load_layer(l)) { if (err_.empty()) err_ = "dequant failed"; return false; }
        }
        // MLA + MoE layers have no plain q/k/v/o or single-MLP structure, so they
        // take their own path through this loop body and skip the rest.
        if (mla_) {
            // The attention input is the NORMALISED hidden state (input_layernorm),
            // exactly as the standard path does; feeding the raw residual here was
            // a bug.
            rmsnorm(x.data(), in_norm_[l].data(), H, xn.data());
            if (!mla_attn(l, xn, o, pos)) return false;
            { LhScope _tlh(LH_RES);
            for (int k = 0; k < H; k++) x[k] += o[k] * residual_scale_; }
#ifdef NPU_INFER_DEBUG_DUMPS
            if (l == 0 && pos == 0 && getenv("NPU_INFER_DUMP_ATTN")) {
                FILE* f = open_debug_dump(getenv("NPU_INFER_DUMP_ATTN"));
                if (f) { fwrite(x.data(), sizeof(float), (size_t)H, f); fclose(f); }
            }
#endif
            rmsnorm(x.data(), post_norm_[l].data(), H, xn.data());
            if (l < leading_dense_) {
                if (!dense_mlp(l, xn, d)) return false;
            } else {
                if (!moe_mlp(l, xn, d)) return false;
            }
            { LhScope _tlh(LH_RES);
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_; }
            // Dump one layer's post-residual hidden state for the Python reference
            // comparison (NPU_INFER_DUMP_LN names the layer).
#ifdef NPU_INFER_DEBUG_DUMPS
            if (getenv("NPU_INFER_DUMP_LN") && l == atoi(getenv("NPU_INFER_DUMP_LN")) && pos == 0) {
                const char* p = getenv("NPU_INFER_DUMP_L0");
                FILE* f = open_debug_dump(p ? p : "/tmp/npu_layer.bin");
                if (f) { fwrite(x.data(), sizeof(float), (size_t)H, f); fclose(f); }
                fprintf(stderr, "  [mla] dumped layer-%d x (%d floats)\n", l, H);
            }
#endif
            if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
                auto am = [&](const std::vector<float>& v) {
                    float m = 0.0f;
                    for (float f : v) m = std::max(m, std::fabs(f));
                    return m;
                };
                fprintf(stderr, "  [mla] L%d attn_absmax=%.4f mlp_absmax=%.4f x_absmax=%.4f\n",
                        l, am(o), am(d), am(x));
            }
            continue;
        }
        // KV sharing (Gemma4): a layer past num_kv_shared_layers carries no k/v
        // weights and reuses the K/V of the last non-shared layer of its own
        // attention type, which that layer already appended earlier this step.
        const bool kv_shared = (l < (int)kv_owner_.size() && kv_owner_[l] != l);
        const int kv_layer = (l < (int)kv_owner_.size()) ? kv_owner_[l] : l;
        // use_double_wide_mlp: the shared layers run a 2*IM-wide MLP (GU2/D2).
        const bool wide = kv_shared && sem_.double_wide_mlp;
        const int i_gu_l = wide ? idx_of(Q4NX_DESIGN_GU2) : i_gu;
        const int i_d_l = wide ? idx_of(Q4NX_DESIGN_D2) : i_d;
        const int im_l = wide ? 2 * c_.IM : c_.IM;

        // ---- per-layer-type attention geometry ----
        // Gemma4's full_attention layers use global_head_dim (512) where its
        // sliding layers use head_dim (256), so BOTH the attention width and the
        // QKV/O geometry differ by layer type; every other model has one type.
        const int lt = (l < sem_.n_layer_types) ? sem_.layer_types[l]
                                                : Q4NX_LAYER_FULL;
        const bool l_full = (lt != Q4NX_LAYER_SLIDING);
        const bool two_widths = (i_qkv2 >= 0);
        const int i_qkv_l = (two_widths && l_full) ? i_qkv2 : i_qkv;
        const int i_o_l = (two_widths && l_full) ? i_o2 : i_o;
        const int l_qrows_all = designs_[i_qkv_l].q_rows;   // q_proj rows (x2 if gated)
        const int l_hd = (attn_gate_ ? l_qrows_all / 2 : l_qrows_all) / c_.NH;
        const int l_nkv_heads = (l_hd > 0) ? designs_[i_qkv_l].k_rows / l_hd : c_.NKV;
        const float l_theta = l_full ? sem_.rope_theta : sem_.rope_theta_sliding;
        const int l_rot = l_full ? sem_.rotary_dim : sem_.rotary_dim_sliding;
        const bool l_prop = l_full && sem_.proportional_rope;
        const int l_win = l_full ? 0 : sem_.sliding_window;
        // Debug aid: with NPU_INFER_NANCHECK set, stop at the first layer that
        // produces a non-finite hidden state (the offline host-GEMM run made
        // NaN-hunting cheap this way).
        auto nancheck = [&](const char* where) -> bool {
            if (!getenv("NPU_INFER_NANCHECK")) return true;
            for (int k = 0; k < H; k++) {
                if (!std::isfinite(x[(size_t)k])) {
                    fprintf(stderr, "  [nan] pos=%d layer=%d (%s) after %s\n", pos, l,
                            (lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR) ? "linear" : "full", where);
                    return false;
                }
            }
            return true;
        };

        // ---- attention ---- (or, for a GDN layer, gdn_layer below)
        rmsnorm(x.data(), in_norm_[l].data(), H, xn.data());
        if (lin_kd_ > 0 && lt == Q4NX_LAYER_LINEAR) {
            LhScope _tlh(LH_GDN);
            if (!gdn_layer(l, xn, o)) return false;
        } else {
        {
            const Q4nxDesignGeom* g = &designs_[i_qkv_l];
            Q4nxProjections p{};
            p.q = wq_.data(); p.k = wk_.data(); p.v = wv_.data();
            // A shared layer has no k/v weights at all: only its q is real and the
            // k/v half of the GEMM output is discarded.  Reusing wq_ for those
            // slots keeps assemble_B's row counts in bounds (q_rows >= k_rows).
            if (kv_shared) { p.k = wq_.data(); p.v = wq_.data(); }
            if (!xd_->pb16i() && !weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble qkv"; return false; }
            std::vector<float> qkv;
            if (xd_->pb16i() && xd_->role_split_qhalf() >= 0) {
                // pb16i has no 14336-wide full-attention QKV kernel: four launches, the two
                // q halves (q, and the output gate) plus k and v on their own geometry.
                // They fill the same [q|gate|k|v] order the fused launch would.
                const int nhd = c_.NH * c_.HD, nkv = c_.NKV * c_.HD;
                std::vector<float> qh0, qh1, kvk, kvv;
                const Q4nxDesignGeom* gq = &designs_[xd_->role_split_qhalf()];
                const Q4nxDesignGeom* gk = &designs_[xd_->role_split_kv()];
                // These two roles exist only as decode-twin kernels, so each launch asks for the twin
                // explicitly (there is no M=128 context for them to fall back to).
                npu_gemm(xd_->role_split_qhalf(), xn.data(), gq->K, gq->N, nullptr, qh0, 10, "attn_q_h0", gq->real_k);
                npu_gemm(xd_->role_split_qhalf(), xn.data(), gq->K, gq->N, nullptr, qh1, 11, "attn_q_h1", gq->real_k);
                npu_gemm(xd_->role_split_kv(), xn.data(), gk->K, gk->N, nullptr, kvk, 12, "attn_k", gk->real_k);
                npu_gemm(xd_->role_split_kv(), xn.data(), gk->K, gk->N, nullptr, kvv, 13, "attn_v", gk->real_k);
                if ((int)qh0.size() < nhd || (int)qh1.size() < nhd || (int)kvk.size() < nkv ||
                    (int)kvv.size() < nkv || !err_.empty()) {
                    if (err_.empty()) err_ = "pb16i full-attention QKV split returned short";
                    return false;
                }
                qkv.assign((size_t)2 * nhd + (size_t)2 * nkv, 0.0f);
                {
                    CpScope _cp(CP_QKV);
                    memcpy(qkv.data(), qh0.data(), (size_t)nhd * sizeof(float));
                    memcpy(qkv.data() + nhd, qh1.data(), (size_t)nhd * sizeof(float));
                    memcpy(qkv.data() + 2 * nhd, kvk.data(), (size_t)nkv * sizeof(float));
                    memcpy(qkv.data() + 2 * nhd + nkv, kvv.data(), (size_t)nkv * sizeof(float));
                    cp_count(CP_QKV, ((size_t)2 * nhd + (size_t)2 * nkv) * sizeof(float));
                }
            } else {
                npu_gemm(i_qkv_l, xn.data(), g->K, g->N, B_.data(), qkv, 0, "attn_qkv", g->real_k);
            }
            // attn_output_gate: the fused QKV is [q | gate | k | v], so attention()
            // must not see the gate half; the gate multiplies ctx afterwards.
            std::vector<float> qkv_use;
            const int nhd = c_.NH * l_hd;
            const float* gate_ptr = nullptr;
            if (attn_gate_) {
                const int kvsz = g->k_rows + g->v_rows;
                qkv_use.resize((size_t)nhd + (size_t)kvsz);
                {
                    CpScope _cp(CP_GATE);
                    memcpy(qkv_use.data(), qkv.data(), (size_t)nhd * sizeof(float));
                    memcpy(qkv_use.data() + nhd, qkv.data() + 2 * nhd,
                           (size_t)kvsz * sizeof(float));
                    cp_count(CP_GATE, ((size_t)nhd + (size_t)kvsz) * sizeof(float));
                }
                gate_ptr = qkv.data() + nhd;
            } else {
                CpScope _cp(CP_GATE);
                qkv_use = qkv;
                cp_count(CP_GATE, qkv.size() * sizeof(float));
            }
            {
            LhScope _tlh(LH_ATTN);
            // The rotary position rotates q; the cache slot decides the window.  Passing the rotary position as the
            // window is what broke every image prompt: an image's rows walk a small h/w grid, so the position after the
            // image is far below the cache length, and attention then read only the first `pos + 1` rows -- dropping
            // the prompt's tail (the question) and never seeing the tokens it had just generated.
            const int kv_window = cache_pos >= 0 ? cache_pos + 1 : 0;
            if (!attention(qkv_use, pos, nullptr, ctx, l_hd, l_nkv_heads, l_theta, l_rot, l_prop,
                           l_win, kv_layer, !kv_shared, kv_window)) {
                err_ = "attention failed";
                return false;
            }
            }
            if (gate_ptr)
                for (int i = 0; i < nhd; i++)
                    ctx[i] *= 1.0f / (1.0f + expf(-gate_ptr[i]));
        }
        {
            const Q4nxDesignGeom* g = &designs_[i_o_l];
            Q4nxProjections p{};
            p.o = wo_.data();
            if (!xd_->pb16i() && !weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble o"; return false; }
            npu_gemm(i_o_l, ctx.data(), g->K, g->N, B_.data(), o, 1,
                     l_full ? "attn_output" : "ssm_out", g->real_k);
        }
        }  // end non-GDN attention
        if (gemma4_layer_) {
            // Gemma4 normalises the attention output BEFORE the residual add.
            rmsnorm(o.data(), post_norm_[l].data(), H, xn.data());
            for (int k = 0; k < H; k++) x[k] += xn[k];
        } else {
            { LhScope _tlh(LH_RES);
            for (int k = 0; k < H; k++) x[k] += o[k] * residual_scale_; }
        }
        if (!nancheck("attention")) return false;

        // ---- MLP ----
        // Gemma4 normalises with pre_feedforward_layernorm and applies
        // post_feedforward_layernorm to the down-projection output; the plain
        // decoder normalises x with post_attention_layernorm instead.
        if (gemma4_layer_) {
            rmsnorm(x.data(), pre_ff_norm_[l].data(), H, xn.data());
        } else {
            rmsnorm(x.data(), post_norm_[l].data(), H, xn.data());
        }
        if (n_expert_ > 0) {
            // Qwen MoE layer (qwen3moe / qwen35moe): the FFN is router -> top-k
            // experts, not a dense gate/up/down.  moe_mlp writes the whole FFN
            // delta into d.  (deepseek2/GLM MoE layers never reach here -- the MLA
            // branch above handles attn+MoE and continues.)
            if (!moe_mlp(l, xn, d)) return false;
            { LhScope _tlh(LH_RES);
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_; }
            if (!nancheck("moe")) return false;
            continue;
        }
        std::vector<float> act;
        if (i_gu_l >= 0) {
            const Q4nxDesignGeom* g = &designs_[i_gu_l];
            Q4nxProjections p{};
            p.gate = wg_.data(); p.up = wu_.data();
            if (!xd_->pb16i() && !weight_cached(2) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble gu"; return false; }
            npu_gemm(i_gu_l, xn.data(), g->K, g->N, B_.data(), act, 2, "ffn_gate_up", g->real_k);
        } else if (i_g >= 0 && i_u >= 0) {
            // split gate/up: two kernels, then SiLU(gate) * up
            std::vector<float> gv, uv;
            const Q4nxDesignGeom* gg = &designs_[i_g];
            Q4nxProjections pg{}; pg.gate = wg_.data();
            if (!xd_->pb16i() && !weight_cached(2) && q4nx_assemble_B(gg, &pg, B_.data()) != 0) { err_ = "assemble g"; return false; }
            npu_gemm(i_g, xn.data(), gg->K, gg->N, B_.data(), gv, 2, "ffn_gate", gg->real_k);

            const Q4nxDesignGeom* gu = &designs_[i_u];
            Q4nxProjections pu{}; pu.up = wu_.data();
            if (!xd_->pb16i() && !weight_cached(3) && q4nx_assemble_B(gu, &pu, B_.data()) != 0) { err_ = "assemble u"; return false; }
            npu_gemm(i_u, xn.data(), gu->K, gu->N, B_.data(), uv, 3, "ffn_up", gu->real_k);

            act.resize((size_t)im_l * 2);
            for (int i = 0; i < im_l; i++) {
                act[i] = gv[(size_t)i];
                act[im_l + i] = uv[(size_t)i];
            }
        } else {
            err_ = "no gate/up design";
            return false;
        }
        mlp(act, h, im_l);
        {
            const Q4nxDesignGeom* g = &designs_[i_d_l];
            Q4nxProjections p{};
            p.down = wd_.data();
            if (!xd_->pb16i() && !weight_cached(4) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble d"; return false; }
            npu_gemm(i_d_l, h.data(), g->K, g->N, B_.data(), d, 4, "ffn_down", g->real_k);
        }
        if (!gemma4_layer_) {
            { LhScope _tlh(LH_RES);
            for (int k = 0; k < H; k++) x[k] += d[k] * residual_scale_; }
            if (!nancheck("mlp")) return false;
            if (pos == 0 && getenv("NPU_INFER_DUMP_LAYERS")) {
                double s = 0;
                float mx = 0;
                for (int k = 0; k < H; k++) {
                    const float a = std::fabs(x[(size_t)k]);
                    s += a;
                    if (a > mx) mx = a;
                }
                fprintf(stderr, "  [L] %d %.6f %.4f\n", l, s / H, mx);
            }
            continue;
        }
        // post_feedforward_layernorm, then the residual add.
        rmsnorm(d.data(), post_ff_norm_[l].data(), H, xn.data());
        for (int k = 0; k < H; k++) x[k] += xn[k];

        // ---- per-layer embedding block (PLE) ----
        // x = x + post_per_layer_input_norm( per_layer_projection(
        //         act_fn(per_layer_input_gate(x)) * per_layer_input[l] ) )
        if (ple_dim_ > 0) {
            const int i_pg = idx_of(Q4NX_DESIGN_PLEGATE);
            const int i_pp = idx_of(Q4NX_DESIGN_PLEPROJ);
            std::vector<float> pg, pp;
            {
                const Q4nxDesignGeom* g = &designs_[i_pg];
                Q4nxProjections p{};
                p.ple = inp_gate_[l].data();
                if (!weight_cached(0) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble plegate"; return false; }
                npu_gemm(i_pg, x.data(), g->K, g->N, B_.data(), pg, 0, nullptr, g->real_k);
            }
            const int gelu = (sem_.mlp_act == Q4NX_ACT_GELU);
            for (int i = 0; i < ple_dim_; i++) {
                float a;
                if (gelu) {
                    const float cc = 0.7978845608028654f;
                    a = 0.5f * pg[i] *
                        (1.0f + std::tanh(cc * (pg[i] + 0.044715f * pg[i] * pg[i] * pg[i])));
                } else {
                    a = pg[i] / (1.0f + std::exp(-pg[i]));
                }
                pg[i] = a * ple_input_[(size_t)l * ple_dim_ + i];
            }
            {
                const Q4nxDesignGeom* g = &designs_[i_pp];
                Q4nxProjections p{};
                p.ple = ple_proj_[l].data();
                if (!weight_cached(1) && q4nx_assemble_B(g, &p, B_.data()) != 0) { err_ = "assemble pleproj"; return false; }
                npu_gemm(i_pp, pg.data(), g->K, g->N, B_.data(), pp, 1, nullptr, g->real_k);
            }
            rmsnorm(pp.data(), post_ln_[l].data(), H, xn.data());
            for (int k = 0; k < H; k++) x[k] += xn[k];
        }
        // layer_scalar (layer_output_scale) closes the Gemma4 layer.
        const float ls = (l < (int)layer_scale_.size()) ? layer_scale_[l] : 1.0f;
        if (ls != 1.0f)
            for (int k = 0; k < H; k++) x[k] *= ls;
    }
    if (ph) { g_ph.layer = ph_ms(tp_lap); tp_lap = std::chrono::steady_clock::now(); }

    last_residual_ = x;                        // one norm earlier, for a drafter that wants the residual
    rmsnorm(x.data(), final_norm_.data(), H, xn.data());
    last_hidden_ = xn;  // for last_hidden(): the post-final-norm state of this position
    if (ph) { g_ph.ffnorm = ph_ms(tp_lap); tp_lap = std::chrono::steady_clock::now(); }
    if (getenv("NPU_INFER_DUMP_HIDDEN")) {
        char hp[256];
        snprintf(hp, sizeof(hp), "/tmp/q4nx_hidden_%d.bin", pos);
        // private to the user, and never through a pre-planted link in /tmp
        const int fd = open(hp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
        FILE* fp = fd >= 0 ? fdopen(fd, "wb") : nullptr;
        if (fp) { fwrite(xn.data(), sizeof(float), (size_t)H, fp); fclose(fp); }
        fprintf(stderr, "  [debug] hidden pos %d mean=%.6f absmax=%.6f -> %s\n",
                pos, [&]{ double s=0; for(float v:xn) s+=std::fabs((double)v); return s/H; }(),
                [&]{ float m=0; for(float v:xn) m=std::max(m, std::fabs(v)); return m; }(), hp);
    }
#ifdef NPU_INFER_DEBUG_DUMPS
    if (getenv("NPU_INFER_DUMP_HEAD")) {
        FILE* f = open_debug_dump(getenv("NPU_INFER_DUMP_HEAD"));
        if (f) { fwrite(xn.data(), sizeof(float), (size_t)xn.size(), f); fclose(f); }
    }
#endif
    if (!run_lm_head(xn, logits)) return false;
    if (ph) { g_ph.head = ph_ms(tp_lap); tp_lap = std::chrono::steady_clock::now(); }
    if (getenv("NPU_INFER_DUMP_LAYERS")) {
        double s = 0; float mx = 0;
        for (float f : xn) { s += std::fabs(f); if (std::fabs(f) > mx) mx = std::fabs(f); }
        fprintf(stderr, "  [mla] final xn: mean|x|=%.6f absmax=%.6f (n=%zu)\n",
                s / (double)xn.size(), mx, xn.size());
        // top-3 before the final norm, to separate a collapsed hidden state from a
        // broken head
        double s2 = 0; float mx2 = 0;
        for (float f : x) { s2 += std::fabs(f); if (std::fabs(f) > mx2) mx2 = std::fabs(f); }
        fprintf(stderr, "  [mla] pre-norm x: mean|x|=%.6f absmax=%.6f\n",
                s2 / (double)x.size(), mx2);
    }
    // MiniCPM4 logit_scale: fitted against the llama.cpp oracle, this forward's
    // logits were ~16x the oracle's (oracle top-1 prob 0.928 vs effectively
    // one-hot here) and the best-fit temperature was 0.060 ~ 1/16.7 against a
    // GGUF logit_scale of 16.  Constant divisor -> cannot move the argmax, so
    // the captured anchor answers are unchanged.
    if (logit_scale_ != 1.0f)
        for (size_t i = 0; i < logits.size(); i++) logits[i] /= logit_scale_;
    if (xd_->profiling()) {
        // Dispatches for this token, i.e. the count the decode budget is made of.
        xd_->note_token();
        // NPU_PROFILE_TOKEN=1 dumps the per-role table per token instead of every 100
        // launches, so the decode budget can be summed from the SAME tokens as the wall
        // (a 100-launch window straddles tokens, and its medians are not the token's cost).
        if (getenv("NPU_PROFILE_TOKEN")) xd_->dump_profile();
        fprintf(stderr, "[prof] token %ld: decode dispatches=%ld (score=%zu of %zu)\n",
                xd_->tokens_seen(), xd_->launches_decode() - step_d0, logits.size(), logits.size());
    }
    if (ph) {
        const double total = ph_ms(tp_step);
        const double tail = ph_ms(tp_lap);
        const double gsum = g_ph.g_pack + g_ph.g_ql + g_ph.g_wait + g_ph.g_rb;
        const double hsum = g_ph.h_memcpy + g_ph.h_pack + g_ph.h_ql + g_ph.h_wait + g_ph.h_rb;
        fprintf(stderr,
                "[ph] tok %d pos %d wall=%.1f | host_gemm=%d | embed=%.1f | layers=%.1f (gemm pack=%.1f "
                "ql=%.1f wait=%.1f rb=%.1f sum=%.1f host=%.1f n=%ld) | ffnorm=%.1f | "
                "head=%.1f (memcpy=%.1f pack=%.1f ql=%.1f wait=%.1f rb=%.1f sum=%.1f "
                "host=%.1f) | tail=%.1f\n",
                token, pos, total, (int)host_gemm_, g_ph.embed, g_ph.layer, g_ph.g_pack, g_ph.g_ql, g_ph.g_wait,
                g_ph.g_rb, gsum, g_ph.layer - gsum, g_ph.launches, g_ph.ffnorm, g_ph.head,
                g_ph.h_memcpy, g_ph.h_pack, g_ph.h_ql, g_ph.h_wait, g_ph.h_rb, hsum,
                g_ph.head - hsum, tail);
        // Top-1 for the greedy-identity gate: <id> plus the margin over second.
        int am = 0;
        float a0 = -INFINITY, a1 = -INFINITY;
        for (size_t i = 0; i < logits.size(); i++) {
            const float v = logits[(size_t)i];
            if (v > a0) { a1 = a0; a0 = v; am = (int)i; } else if (v > a1) { a1 = v; }
        }
        fprintf(stderr, "[ph] argmax=%d top=%.6f second=%.6f margin=%.6f\n",
                am, a0, a1, a0 - a1);
        // The two C syncs' cumulative cost, from I8Ctx's own counters, when the narrow-vs-whole A/B asks for
        // it (NPU_C_SYNC_PROF=1). Cumulative per process, printed once per token: a token's flushes are the
        // delta between two lines, and the bytes are exact (bo::sync is a clflush loop over its range).
        if (getenv("NPU_C_SYNC_PROF")) I8Ctx::c_sync_print(token);
        if (getenv("NPU_A_SYNC_PROF")) I8Ctx::a_sync_print(token);
        char lhwhat[48];
        snprintf(lhwhat, sizeof lhwhat, "tok %d", token);
        lh_dump_layer_host(lhwhat, "token");
    }
    // Independent of NPU_PHASE_PROF: the copy accounting answers a different question and must be readable
    // with only NPU_COPY_PROF set.
    {
        char cp_what[48];
        snprintf(cp_what, sizeof cp_what, "tok %d pos %d", token, pos);
        cp_print(cp_what);
    }
    return true;
}

// tiled LM head: logits[K*t + n] = dot(embed[K*t + n], h); K = H, tile = QKV N
bool Q4nxNpuForward::run_lm_head_rows(const std::vector<float>& h_rows, int M,
                                     std::vector<float>& logits_all) {
    const int H = c_.H, NV = c_.NV;
    if (M <= 0) { err_ = "lm head rows: empty batch"; return false; }
    if (xd_ && xd_->head_ctx() && xd_->head_ctx()->gemv) {
        // one row per launch on this family; the batch is a loop
        logits_all.assign((size_t)M * (size_t)NV, 0.0f);
        std::vector<float> one;
        for (int m = 0; m < M; m++) {
            const std::vector<float> row(h_rows.begin() + (size_t)m * H,
                                        h_rows.begin() + (size_t)(m + 1) * H);
            if (!run_lm_head_gemv(row, one)) return false;
            std::copy(one.begin(), one.end(), logits_all.begin() + (size_t)m * (size_t)NV);
        }
        return true;
    }
    if (M == 1 || h_rows.size() != (size_t)M * (size_t)H) {
        std::vector<float> one(h_rows.begin(), h_rows.begin() + H);
        std::vector<float> lg;
        if (!run_lm_head(one, lg)) return false;
        logits_all = std::move(lg);
        return true;
    }
    logits_all.assign((size_t)M * (size_t)NV, 0.0f);
    // One single-row call first: it prepares the packed tiles / f32 gather and refuses loudly (missing
    // images, two-plane tile on a one-plane ELF) exactly as the token path does, so this function never
    // has to duplicate that setup.  Its logits are the last row's.
    {
        std::vector<float> last(h_rows.begin() + (size_t)(M - 1) * H, h_rows.begin() + (size_t)M * H);
        std::vector<float> lg;
        if (!run_lm_head(last, lg)) return false;
        std::copy(lg.begin(), lg.end(), logits_all.begin() + (size_t)(M - 1) * (size_t)NV);
    }
    // ── the int8 packed head, one launch per vocabulary tile for the whole batch (engine#1341) ──
    // The fallback below issues M x ntiles SINGLE-ROW launches, and a single-row launch is latency-bound:
    // measured on the 0.5B b8 pack, a 1-row head launch streams its 9.96 MB tile at 0.33 GB/s where a
    // 16-row launch of the same tile reaches 4.9 GB/s, so a 16-row verify pass spent 536 of its 678 ms in
    // the head.  This is the batching the layers already do, and the one the ternary head (rows_supported,
    // below) already had; the int8 packed head never got it.
    //
    // The arithmetic is per-row identical to the single-row path: each row keeps its own activation scale
    // (amax/127 of that row), the tile's per-column weight scales are shared, and the dequantisation
    // multiplies the same three factors -- measured bit-identical (0/16 rows differ in argmax).
    if (!host_gemm_ && xd_ && !head_pb16i_ && xd_->weight_cache_on() && !lm_head_packed_.empty() &&
        M > 1 && h_rows.size() == (size_t)M * (size_t)H) {
        int i_wide = -1;
        for (int d = 0; d < ndesigns_; d++) {
            if (designs_[d].K != kpad_ || designs_[d].N <= 0) continue;
            if (i_wide < 0 || designs_[d].N > designs_[i_wide].N) i_wide = d;
        }
        const int T = i_wide >= 0 ? designs_[i_wide].N : 0;
        const int ntiles = T > 0 ? (NV + T - 1) / T : 0;
        I8Ctx* hctx = i_wide >= 0 ? xd_->pick_ctx(i_wide, M) : nullptr;
        // A verify batch may be wider than the twin's row capacity (mtp_acceptance measures up to 128 rows).
        // The launch is then made in CHUNKS of at most MD rows rather than falling back to one launch per
        // row: the weights are already resident, so a chunk costs one launch, and M=128 becomes 8 launches
        // instead of 128.  A row's activation scale is its own either way, so the chunking cannot move a
        // value -- the numeric gate is the same per-row comparison.
        // NPU_HEAD_NO_CHUNK=1 runs the same code with one row per chunk, which is what this path did before
        // chunking existed: the A/B for the claim, on the same artifacts and in the same process.
        static const bool chunk_on = getenv("NPU_HEAD_NO_CHUNK") == nullptr;
        const int chunk = (!chunk_on || !hctx || hctx->MD <= 0) ? (chunk_on ? 0 : 1) : hctx->MD;
        if (hctx && !hctx->gemv && chunk > 0 && (int)lm_head_packed_.size() == ntiles &&
            lm_head_packed_T_ == T) {
            I8Ctx& lctx = *hctx;
            const float inv_es = (sem_.tie_embeddings && sem_.embed_scale > 0)
                                     ? 1.0f / sem_.embed_scale : 1.0f;
            std::vector<float> scales((size_t)M, 1.0f);
            std::vector<float> hs((size_t)M * (size_t)H);
            for (int m = 0; m < M; m++) {
                const float* hrow = &h_rows[(size_t)m * H];
                float amax = 0.0f;
                for (int k = 0; k < H; k++) {
                    const float a = std::fabs(hrow[(size_t)k] * inv_es);
                    hs[(size_t)m * (size_t)H + (size_t)k] = hrow[(size_t)k] * inv_es;
                    if (a > amax) amax = a;
                }
                scales[(size_t)m] = amax > 0 ? amax / 127.0f : 1.0f;
            }
            bool ok_all = true;
            for (int r0 = 0; r0 < M && ok_all; r0 += chunk) {
                const int rows = std::min(chunk, M - r0);
                for (int t = 0; t < ntiles && ok_all; t++) {
                const int n0 = t * T, n1 = std::min(NV, n0 + T);
                const auto t0 = std::chrono::steady_clock::now();
                xrt::run r = lctx.launch_async_with_bo_rows(*lm_head_packed_[(size_t)t].bo,
                                                           hs.data() + (size_t)r0 * (size_t)H,
                                                           rows, H, scales.data() + r0);
                const auto t1 = std::chrono::steady_clock::now();
                lctx.wait_kernel(r);
                const auto t2 = std::chrono::steady_clock::now();
                lctx.readback();
                const auto t3 = std::chrono::steady_clock::now();
                if (xd_->profiling())
                    xd_->note_head(lctx.MD, 0.0,
                                   std::chrono::duration<double, std::milli>(t1 - t0).count(),
                                   std::chrono::duration<double, std::milli>(t2 - t1).count(),
                                   std::chrono::duration<double, std::milli>(t3 - t2).count());
                if (lctx.canary_short()) {
                    fprintf(stderr, "[canary] SHORT RESULT (head rows): tile=%d M=%d short=%d\n", t, M,
                            lctx.short_row_);
                    ok_all = false;
                    break;
                }
                const std::vector<float>& gs = lm_head_packed_[(size_t)t].scales;
                for (int m = 0; m < rows; m++) {
                    const int32_t* cr = lctx.Cm + (size_t)m * (size_t)lctx.ND;
                    float* dst = &logits_all[(size_t)(r0 + m) * (size_t)NV];
                    const float as = scales[(size_t)(r0 + m)];
                    for (int n = n0; n < n1; n++) {
                        float v = (float)cr[n - n0] * as * gs[(size_t)(n - n0)];
                        if (sem_.logit_softcap > 0)
                            v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
                        dst[(size_t)n] = v;
                    }
                }
                }
            }
            if (ok_all) return true;
            std::fill(logits_all.begin(), logits_all.end(), 0.0f);  // a short tile: fall through, don't half-fill
        }
    }

    const bool rows_supported = !host_gemm_ && head_pb16i_ && xd_->head_ctx() != nullptr && !lm_head_packed_.empty();
    if (!rows_supported) {
        // host_gemm and the int8 head have no per-row launcher; the rows are rare (speculative verify only),
        // so they go one at a time rather than growing a second code path in the hot token routine.
        for (int m = 0; m + 1 < M; m++) {
            std::vector<float> one(h_rows.begin() + (size_t)m * H, h_rows.begin() + (size_t)(m + 1) * H);
            std::vector<float> lg;
            if (!run_lm_head(one, lg)) return false;
            std::copy(lg.begin(), lg.end(), logits_all.begin() + (size_t)m * (size_t)NV);
        }
        return true;
    }
    // The pb16i head, once for the whole batch: per-row rotation + per-row activation scale (the launcher and
    // the dequantiser both carry per-row scales), one launch per tile, then the host finishing each row.
    int i_qkv = -1;
    for (int d = 0; d < ndesigns_; d++)
        if (designs_[d].K == kpad_ && designs_[d].N > 0 && (i_qkv < 0 || designs_[d].N > designs_[i_qkv].N))
            i_qkv = d;
    if (i_qkv < 0) { err_ = "LM head rows: no GEMM design on the kernel K grid (kpad=" + std::to_string(kpad_) + ")";
        return false; }
    const int T = designs_[i_qkv].N;
    const int ntiles = (NV + T - 1) / T;
    if ((int)lm_head_packed_.size() != ntiles) { err_ = "LM head rows: packed tiles not resident"; return false; }
    const std::vector<float>* hsig = xd_->had_signs_for(H);
    if (!hsig) {
        err_ = "pb16i head rows: no Hadamard sign vector for K=" + std::to_string(H);
        return false;
    }
    std::vector<float> hr((size_t)M * (size_t)H), scales((size_t)M);
    for (int m = 0; m < M; m++) {
        const float* src = &h_rows[(size_t)m * H];
        float* dst = &hr[(size_t)m * H];
        std::copy(src, src + H, dst);
        hadamard_rotate(hsig->data(), dst, H, 1024);
        float amax = 0.0f;
        for (int k = 0; k < H; k++) {
            const float a = std::fabs(dst[(size_t)k]);
            if (a > amax) amax = a;
        }
        scales[(size_t)m] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
    I8Ctx& lctx = *xd_->head_ctx();
    // A verify batch can be wider than this head context's row capacity -- the tri-ternary head context is
    // built on the decode design (MD = 16 rows) -- and the launcher's quantiser refuses am > MD while the
    // launch still runs, so rows MD..M-1 were read and written past the activation BO (engine#1386, the
    // same failure as #1372 on the head instead of the layers). Chunk at MD exactly as the int8 head does
    // (engine#1364): the weights are already resident, so a chunk is one launch, and a row keeps its own
    // activation scale either way, so the chunking cannot move a value.
    const int chunk = lctx.MD > 0 ? lctx.MD : M;
    std::vector<float> got((size_t)std::min(chunk, M) * (size_t)T);
    for (int r0 = 0; r0 < M; r0 += chunk) {
        const int rows = std::min(chunk, M - r0);
        for (int t = 0; t < ntiles; t++) {
            const int n0 = t * T, n1 = std::min(NV, n0 + T);
            xrt::run run = lctx.launch_async_with_bo_rows(*lm_head_packed_[(size_t)t].bo,
                                                         hr.data() + (size_t)r0 * (size_t)H, rows, H,
                                                         scales.data() + r0);
            lctx.wait_kernel(run);
            lctx.readback();
            // Mirror the int8 head's guard: this is the change that makes it worth having, because a chunk
            // that comes back short would otherwise feed dequant_pb16i rows nobody wrote -- poison_canary()
            // and a zeroed buffer make a plausible-looking row, so the logits go quietly wrong. Named here.
            if (lctx.canary_short()) {
                fprintf(stderr, "[canary] SHORT RESULT (head rows, pb16i): tile=%d M=%d rows=%d short=%d\n", t,
                        M, rows, lctx.short_row_);
                err_ = "LM head rows: a pb16i chunk came back short (" + std::to_string(rows) +
                       " rows asked, row " + std::to_string(lctx.short_row_) + " short)";
                return false;
            }
            lctx.dequant_pb16i(got.data(), rows, T, scales.data() + r0, lm_head_packed_[(size_t)t].eref,
                               xd_->head_ctx()->pb16i_planes);
            for (int m = 0; m < rows; m++) {
                for (int n = n0; n < n1; n++) {
                    float v = got[(size_t)m * T + (n - n0)];
                    if (sem_.logit_softcap > 0) v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
                    logits_all[(size_t)(r0 + m) * (size_t)NV + (size_t)n] = v;
                }
            }
        }
    }
    return true;
}

bool Q4nxNpuForward::run_lm_head(const std::vector<float>& h, std::vector<float>& logits) {
    const int H = c_.H, NV = c_.NV;
    // The target first, and only if it both claims and succeeds (npu/model/op_dispatch.h): `lm_head` is the one
    // optional op whose contract needs no residency -- one hidden row [H] to [V], idempotent, weights the model's
    // -- so a target that owns a head context can run it here instead of the launch path below. No target on this
    // box claims it yet (XdnaTarget's supports() is gemv-only), so today this falls through and the numerics are
    // unchanged by construction; a target that claims it and fails still falls through, which is the case the
    // dispatch is written for.
    // `supports()` first and separately: the target's output buffer is [V] floats (1 MB at 248k vocab), and asking
    // for it on every token when no target claims the op would be a per-token allocation for nothing.
    if (xd_ && !host_gemm_ && (int)h.size() == H && xd_->supports(npu::target::Op::lm_head)) {
        std::vector<float> via_target((size_t)NV, 0.0f);
        if (npu::model::run_lm_head(*xd_, h.data(), via_target.data())) { logits = std::move(via_target); return true; }
    }
    if (!host_gemm_ && xd_ && xd_->head_ctx() && xd_->head_ctx()->gemv) return run_lm_head_gemv(h, logits);
    if (host_gemm_ && host_gemm_i8_) {
        // engine#1808: the head's integer twin. Without this the arm's head is a plain f32 matvec ("No I8Ctx is
        // touched" below) while the device's head reads a packed quantised table, which is exactly the 0.30 max|d|
        // that was left once every body GEMM had been shown bit-exact. The device packs its head tiles with
        // lctx.packB_into(...) -> packB_blocked, so the arm builds them with the same rule through
        // q4nx_pack_B_int8_blocked(): one multiplier per column, int8 codes. Chunked by vocabulary block, because a
        // whole head image is H x V bytes (622 MB at this vocabulary).
        const float inv_es = (sem_.tie_embeddings && sem_.embed_scale > 0) ? 1.0f / sem_.embed_scale : 1.0f;
        const int TB = 8192;   // vocabulary block; a tile's image is H x TB bytes
        if ((int)lm_head_i8_.size() != (NV + TB - 1) / TB) lm_head_i8_.assign((NV + TB - 1) / TB, HostI8Tensor());
        logits.assign((size_t)NV, 0.0f);
        std::vector<float> a8v((size_t)H);
        float amax = 0.0f;
        for (int k = 0; k < H; k++) {
            const float v = std::fabs(h[(size_t)k] * inv_es);
            if (v > amax) amax = v;
        }
        const float ascale = amax > 0 ? amax / 127.0f : 1.0f;
        for (int k = 0; k < H; k++) {
            int q = (int)std::round(h[(size_t)k] * inv_es / ascale);
            a8v[(size_t)k] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
        std::vector<float> bt((size_t)H * (size_t)TB);
        std::vector<int8_t> a8i((size_t)H);
        for (size_t i = 0; i < a8v.size(); i++) a8i[i] = a8v[i];
        for (int t = 0; t < (int)lm_head_i8_.size(); t++) {
            const int n0 = t * TB, n1 = std::min(NV, n0 + TB), Tn = n1 - n0;
            HostI8Tensor& c = lm_head_i8_[(size_t)t];
            if (c.codes.empty()) {
                for (int n = n0; n < n1; n++)
                    for (int k = 0; k < H; k++) bt[(size_t)k * Tn + (n - n0)] = lm_at(n, k);
                c.N = Tn;
                c.codes.assign((size_t)H * (size_t)Tn, 0);
                c.d.assign((size_t)Tn, 0.0f);
                double ssum = 0.0;
                // The last vocabulary block is rarely whole tiles (phi-3-mini: 32064 % 8192 = 7488, not a multiple of
                // 128), and this image is unpadded, so it takes the layout its geometry fits (engine#1994).
                c.blocked = q4nx_layout::blocked_fits(H, Tn);
                q4nx_pack_B_int8_blocked(c.codes.data(), bt.data(), H, Tn, H, Tn, c.blocked ? 1 : 0, c.d.data(), &ssum);
            }
            for (int n = 0; n < Tn; n++) {
                int32_t acc = 0;
                for (int k = 0; k < H; k++)
                    acc += (int32_t)((int)c.codes[q4nx_layout::image_index((size_t)k, (size_t)n, H, Tn, c.blocked)] *
                                     (int)a8i[(size_t)k]);
                float v = c.d[(size_t)n] * (ascale * (float)acc);
                if (sem_.logit_softcap > 0) v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
                logits[(size_t)(n0 + n)] = v;
            }
        }
        return true;
    }
    if (host_gemm_) {
        // Offline mode: plain float matvec against the (possibly dequantized)
        // tied table, matching the Python reference.  No I8Ctx is touched.
        const float inv_es0 = (sem_.tie_embeddings && sem_.embed_scale > 0)
                                  ? 1.0f / sem_.embed_scale : 1.0f;
        logits.assign((size_t)NV, 0.0f);
        for (int n = 0; n < NV; n++) {
            float s = 0;
            for (int k = 0; k < H; k++) s += h[(size_t)k] * inv_es0 * lm_at(n, k);
            if (sem_.logit_softcap > 0)
                s = sem_.logit_softcap * std::tanh(s / sem_.logit_softcap);
            logits[(size_t)n] = s;
        }
        return true;
    }
    // Reuse an existing GEMM geometry for the head (the variable keeps its old name
    // for the block below).  QKV when the arch has one; the MLA/deepseek2 arch
    // (GLM-4.7-Flash) has QA/QB/KVA/O instead, so idx_of(QKV) returned -1 and T was
    // read out of designs_[-1] -- garbage, and the ntiles division below raised
    // SIGFPE.  Only this NPU path uses T (the host path is a plain matvec), which is
    // why GLM crashed here the first time the ELF path was exercised.
    /* Pick the WIDEST design whose K is the kernel grid this container's hidden width lands on, kpad_
     * (engine#845): fewer/larger tiles mean fewer synchronous launches for the same math.  It used to
     * compare against H, which is the same rule only while a model's hidden is already on the grid -- after
     * the K pad it matched nothing, so the first OGA pack (hidden 896) failed right here with
     * "no GEMM design with K=H available" although all five designs had loaded. */
    int i_qkv = -1;
    for (int d = 0; d < ndesigns_; d++) {
        if (designs_[d].K != kpad_ || designs_[d].N <= 0) continue;
        if (i_qkv < 0 || designs_[d].N > designs_[i_qkv].N) i_qkv = d;
    }
    if (i_qkv < 0) {
        err_ = "LM head: no GEMM design on the kernel K grid (kpad=" + std::to_string(kpad_) + ")";
        return false;
    }
    const int T = designs_[i_qkv].N;
    const int ntiles = (NV + T - 1) / T;
    logits.assign(NV, 0.0f);

    // The stored embedding is PRE-SCALED by the architecture's embedding scale
    // (Gemma4: sqrt(hidden_size)), so the LM head must divide that scale back out
    // to recover the raw tied weight.  embed_scale is 1 for every other model.
    const float inv_es = (sem_.tie_embeddings && sem_.embed_scale > 0)
                             ? 1.0f / sem_.embed_scale : 1.0f;
    std::vector<float> hs((size_t)H);
    for (int k = 0; k < H; k++) hs[(size_t)k] = h[(size_t)k] * inv_es;

    /* The kernel's K is the padded width, so the packed B tile is kpad_ rows tall; lm_at() returns
     * 0 for k >= c_.H, which fills the pad rows with zeros (engine#845). */
    const size_t tile_elems = (size_t)kpad_ * (size_t)T;
    // The head is ONE row of activations (the last token's hidden state), so it is
    // decode and runs the M=16 twin of this geometry.  The head reuses the widest
    // K=H design, so the twin is the same kernel the MLP's GU launch uses -- same
    // N, same per-launch cost.
    // The head keeps the int8 family even in a pb16i container: lm_head is not a
    // PTQ1_0 ternary tensor and has no image, and packB below is the int8 packer.
    I8Ctx& lctx = xd_->head_ctx() ? *xd_->head_ctx() : *xd_->pick_ctx(i_qkv, 1);
    // Belt for the gemv family: packB_into writes a KD*ND int8 image, which is larger than that
    // family's tile-stream BO.  The head context above handles this family; reaching here means a
    // container with no head ELF, and a loud refusal beats a heap overrun (measured: it aborted a
    // 0.5B pack in run_lm_head, ASan named the memset).
    if (!host_gemm_ && lctx.gemv) {
        err_ = "lm head: the int8 tiled head cannot run on the q4_1 gemv family (no head ELF loaded)";
        return false;
    }
    // ── Pack the head's B tiles ONCE ──────────────────────────────────────────
    // Before this, every token copied all 15 f32 tiles out of lm_head_tiles_ (356 MB
    // a tile) and re-ran packB -- a per-column int8 quantisation of all 89 M head
    // weights -- on each one: 0.82 s of the 11.5 s decode step (measured: 203 ms of
    // f32 copy + 618 ms of packB), the only per-token pack left in the model because
    // wcache_ already holds every layer's projections.  Packing into dedicated
    // weight BOs at first use makes it a load-time cost instead.
    const bool head_packed = xd_->weight_cache_on() && dev_ != nullptr;
    const bool head_pb16i = head_pb16i_ && head_packed && xd_->head_ctx() != nullptr;
    if (head_packed &&
        ((int)lm_head_packed_.size() != ntiles || lm_head_packed_T_ != T)) {
        lm_head_packed_.clear();
        if (head_pb16i) {
            // The image is the file: no dequant, no pack, nothing to quantise on the host.  A
            // tile whose side table asks for the two-plane accumulator is refused, not guessed:
            // the K=H ELF family is a one-plane design.
            for (int t = 0; t < ntiles; t++) {
                PackedWeight pw;
                pw.bo = lctx.make_weight_bo(*dev_, /*zero=*/false);
                if (!lctx.load_pb16i_into(*pw.bo, head_pb16i_stem(t), pw.eref, pw.pb16i_mode)) {
                    err_ = "lm head: cannot load the ternary image for tile " + std::to_string(t);
                    return false;
                }
                if (pw.pb16i_mode > xd_->head_ctx()->pb16i_planes) {
                    err_ = "lm head: tile " + std::to_string(t) +
                           " needs the two-plane accumulator, this ELF has one plane";
                    return false;
                }
                lm_head_packed_.push_back(std::move(pw));
            }
            lm_head_packed_T_ = T;
            lm_head_tiles_.clear();
            lm_head_tiles_.shrink_to_fit();
            lm_head_f_.clear();
            lm_head_f_.shrink_to_fit();
            fprintf(stderr, "  [forward] lm head: %d ternary image(s) resident (H=%d T=%d), "
                            "f32 head released\n", ntiles, H, T);
        } else {
        // engine#1716: this loop used to gather and pack every tile on EVERY serve (10-17 s on a quiet box for the
        // h4096 shapes, minutes on a contended one). The image is a pure function of the head's stored bytes and the
        // kernel set's geometry, so a serve now asks q4nx_head_tile_image() for it: a previous serve's file is used
        // when it reads back whole AND a sample of its columns re-packs to the same bytes from the weights held here;
        // anything else packs exactly as before (same packer, same padded KD x ND image, same per-column scales as
        // the per-token packB below) and leaves the file for the next serve. ONEBIT_HEAD_IMAGE_CACHE=0 turns it off.
        const auto head_t0 = std::chrono::steady_clock::now();
        std::string head_dir;
        Q4nxHeadImageKey head_key = {(uint32_t)kpad_, (uint32_t)T, lctx.blocked_b ? 1u : 0u, 0};
        {
            const char* off = getenv("ONEBIT_HEAD_IMAGE_CACHE");
            const bool cache_on = !(off && off[0] == '0' && off[1] == 0);
            // The head's stored bytes: the separate lm_head.weight of an untied model, else the token table lm_at()
            // falls back to (which a tied container may itself store as lm_head.weight).
            TensorDesc* src = !lm_head_f_.empty() ? model_tensor_by_name(mw_, "lm_head.weight") : nullptr;
            if (!src) {
                src = &mw_->embed_tokens;
                if (src->data_size == 0 || src->name[0] == '\0') src = model_tensor_by_name(mw_, "lm_head.weight");
            }
            const void* sp = (cache_on && src) ? model_tensor_data(mw_, src) : nullptr;
            // An image the layout cannot hold as kd x nd bytes of T columns is not cached: it packs as before.
            if (sp && src->data_size > 0 && (int)lctx.ND == T && (int)lctx.KD >= kpad_) {
                // What changes the decoded head without changing its bytes: the vocabulary, the width it is read at,
                // whether it is the tied table, and the dtype the decoder dispatches on.
                uint32_t extra[8] = {(uint32_t)NV, (uint32_t)H, (uint32_t)lm_w_, lm_head_f_.empty() ? 1u : 0u, 0, 0, 0, 0};
                std::memcpy(&extra[4], src->dtype, std::min(sizeof src->dtype, (size_t)16));
                head_key.head_hash = q4nx_head_source_hash(sp, (size_t)src->data_size, extra, 8);
                try {
                    head_dir = onebit::npu::q4nx::cache_root() + "/head-images";
                    std::error_code ec;
                    std::filesystem::create_directories(head_dir, ec);
                    if (ec) head_dir.clear();
                } catch (const std::exception&) {
                    head_dir.clear();   // no home and no ONEBIT_Q4NX_CACHE: pack, as before
                }
            }
        }
        struct HeadCol {
            const Q4nxNpuForward* self;
            int k;
        } head_col = {this, kpad_};
        const Q4nxHeadColFn head_col_fn = [](void* ctx, int n, float* out) {
            const HeadCol* c = (const HeadCol*)ctx;
            for (int k = 0; k < c->k; k++) out[k] = c->self->lm_at(n, k);
        };
        int head_hits = 0, head_wrote = 0;
        for (int t = 0; t < ntiles; t++) {
            const int n0 = t * T, n1 = std::min(NV, n0 + T);
            PackedWeight pw;
            pw.bo = lctx.make_weight_bo(*dev_);
            pw.scales.assign((size_t)lctx.ND > (size_t)T ? (size_t)lctx.ND : (size_t)T, 0.0f);
            Q4nxHeadTileResult hres;
            bool ok = false;
            lctx.fill_weight_bo(*pw.bo, [&](int8_t* Bm) {
                ok = q4nx_head_tile_image(head_dir.empty() ? nullptr : head_dir.c_str(), &head_key, t, n0, n1 - n0, kpad_,
                                          T, lctx.KD, lctx.ND, head_col_fn, &head_col, Bm, pw.scales.data(), &hres) == 1;
            });
            if (!ok) {
                err_ = std::string("lm head: ") + hres.why;
                return false;
            }
            if (hres.refused)
                fprintf(stderr, "  [forward] lm head: tile %d's cached image was NOT used and is replaced: %s\n", t, hres.why);
            head_hits += hres.hit;
            head_wrote += hres.wrote;
            pw.scales.resize((size_t)T);
            lm_head_packed_.push_back(std::move(pw));
        }
        lm_head_packed_T_ = T;
        const double head_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count();
        // Both f32 copies are now dead on this path: lm_head_tiles_ was only ever
        // read by the memcpy below, and lm_at() is reachable only from the
        // host_gemm_ branch (which returns above) and from the gather just done.
        lm_head_tiles_.clear();
        lm_head_tiles_.shrink_to_fit();
        lm_head_f_.clear();
        lm_head_f_.shrink_to_fit();
        fprintf(stderr, "  [forward] lm head: %d packed weight tile(s) resident "
                        "(H=%d T=%d), f32 head + f32 tile gather released; %d from the image cache, %d packed "
                        "(%d written for the next serve) in %.2f s\n",
                ntiles, H, T, head_hits, ntiles - head_hits, head_wrote, head_s);
        }
    }
    std::vector<float> Bt;
    if (!head_packed) {
        if (lm_head_tiles_.size() != (size_t)ntiles * tile_elems || lm_head_tiles_T_ != T) {
            // Once per model: [tile][H][T], so each token copies sequentially instead
            // of gathering the whole head with a T-float stride.
            lm_head_tiles_.assign((size_t)ntiles * tile_elems, 0.0f);
            for (int t = 0; t < ntiles; t++) {
                const int n0 = t * T, n1 = std::min(NV, n0 + T);
                float* dst = lm_head_tiles_.data() + (size_t)t * tile_elems;
                for (int n = n0; n < n1; n++)
                    for (int k = 0; k < kpad_; k++)
                        dst[(size_t)k * T + (n - n0)] = lm_at(n, k);
            }
            lm_head_tiles_T_ = T;
        }
        Bt.assign(tile_elems, 0.0f);
    }
    // Phase attribution for the NPU head (NPU_PHASE_PROF=1): h_memcpy is the
    // per-tile f32 copy, h_pack/h_ql/h_wait/h_rb come from the launch timers
    // below, h_total is the whole call.
    const bool ph = phprof_on();
    g_in_head = ph;
    g_ph.h_memcpy = g_ph.h_pack = g_ph.h_ql = g_ph.h_wait = g_ph.h_rb = 0;
    const auto hp0 = ph ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
    // ---- the ternary head: rotate A once for all 15 tiles ---------------------
    // Same fold and same scale rule as npu_gemm's pb16i branch (A' = H (A . s), int8 with its own
    // column scale); the side table's per-column 2^e_ref finishes the arithmetic on the host, and
    // the device only sums integers.
    std::vector<float> hr;
    std::vector<float> head_got;
    float hasb = 1.0f;
    if (head_pb16i) {
        const std::vector<float>* hsig = xd_->had_signs_for(H);
        if (!hsig) {
            err_ = "pb16i head: no Hadamard sign vector for K=" + std::to_string(H);
            return false;
        }
        const auto hh0 = std::chrono::steady_clock::now();
        hr = hs;
        hadamard_rotate(hsig->data(), hr.data(), H, 1024);
        float amax = 0.0f;
        for (int k = 0; k < H; k++) {
            const float a = std::fabs(hr[(size_t)k]);
            if (a > amax) amax = a;
        }
        hasb = amax > 0.0f ? amax / 127.0f : 1.0f;
        const double had_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - hh0).count();
        xd_->add_hadamard_cost(had_ms, 1);
        head_got.assign((size_t)T, 0.0f);
    }
    for (int t = 0; t < ntiles; t++) {
        const int n0 = t * T, n1 = std::min(NV, n0 + T);
        const auto hm0 = ph ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
        if (!head_packed)
            std::memcpy(Bt.data(), lm_head_tiles_.data() + (size_t)t * tile_elems,
                        tile_elems * sizeof(float));
        if (ph) g_ph.h_memcpy += ph_ms(hm0);
        if (head_pb16i) {
            const auto q0 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            xrt::run hr_run = lctx.launch_async_with_bo(*lm_head_packed_[(size_t)t].bo, hr.data(),
                                                        1, H, hasb);
            const auto q1 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            lctx.wait_kernel(hr_run);
            const auto q2 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            lctx.readback();
            const auto q3 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            lctx.dequant_pb16i(head_got.data(), 1, T, &hasb, lm_head_packed_[(size_t)t].eref,
                               xd_->head_ctx()->pb16i_planes);
            for (int n = n0; n < n1; n++) {
                float v = head_got[(size_t)(n - n0)];
                if (sem_.logit_softcap > 0)
                    v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
                logits[(size_t)n] = v;
            }
            if (xd_->profiling()) {
                const double dql = std::chrono::duration<double, std::milli>(q1 - q0).count();
                const double dwt = std::chrono::duration<double, std::milli>(q2 - q1).count();
                const double drb = std::chrono::duration<double, std::milli>(q3 - q2).count();
                xd_->note_head(lctx.MD, 0.0, dql, dwt, drb);
                if (ph) { g_ph.h_ql += dql; g_ph.h_wait += dwt; g_ph.h_rb += drb; }
            }
            continue;
        }
        const auto h0 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        float amax = 0.0f;
        for (int k = 0; k < H; k++) { float a = std::fabs(hs[(size_t)k]); if (a > amax) amax = a; }
        const float as = amax > 0 ? amax / 127.0f : 1.0f;
        auto issue = [&]() -> xrt::run {
            if (head_packed)
                return lctx.launch_async_with_bo(*lm_head_packed_[(size_t)t].bo,
                                                 hs.data(), 1, H, as);
            float sout = 1.0f;
            lctx.packB(0, Bt.data(), H, T, sout);
            lctx.quantize_async(hs.data(), 1, H, as);
            return lctx.sync_and_launch(0);
        };
        xrt::run r = issue();
        const auto h1 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        lctx.wait_kernel(r);
        const auto h2 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        lctx.readback();
        if (lctx.canary_short()) {   // engine#666: the head tile's row is not in memory; re-issue once
            lctx.canary_retries_++;
            r = issue();
            lctx.wait_kernel(r);
            lctx.readback();
            if (lctx.canary_short())
                fprintf(stderr, "[canary] SHORT RESULT (head) twice: tile=%d T=%d short=%d\n", t, T, lctx.short_row_);
        }
        const auto h3 = xd_->profiling() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const std::vector<float>& gs = head_packed ? lm_head_packed_[(size_t)t].scales
                                                   : lctx.group_scales[0];
        for (int n = n0; n < n1; n++) {
            float v = (float)lctx.Cm[n - n0] * as * gs[(size_t)(n - n0)];
            if (sem_.logit_softcap > 0)
                v = sem_.logit_softcap * std::tanh(v / sem_.logit_softcap);
            logits[(size_t)n] = v;
        }
        if (xd_->profiling()) {
            // h0 -> h1 is the weight work (packB in the fallback, a no-op amax scan +
            // BO launch when packed), h1 -> h2 the kernel, h2 -> h3 the readback.
            const double dpk = std::chrono::duration<double, std::milli>(h1 - h0).count();
            const double dql = 0.0;
            const double dwt = std::chrono::duration<double, std::milli>(h2 - h1).count();
            const double drb = std::chrono::duration<double, std::milli>(h3 - h2).count();
            xd_->note_head(lctx.MD, dpk, dql, dwt, drb);
            if (ph) {
                g_ph.h_pack += dpk; g_ph.h_ql += dql;
                g_ph.h_wait += dwt; g_ph.h_rb += drb;
            }
        }
    }
    if (ph) { g_ph.h_total = ph_ms(hp0); g_in_head = false; }
    return true;
}
