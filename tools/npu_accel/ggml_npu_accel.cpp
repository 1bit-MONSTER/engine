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
// ggml_npu_accel.cpp -- an ACCEL ggml backend that runs a model's weight matmuls on the XDNA2 NPU.
//
// Why this shape: a model llama.cpp already runs (on the CPU) gets its projections and expert matmuls on the NPU
// without porting its maths into the NPU forward. It is the mechanism llama.cpp's own BLAS backend uses: an ACCEL
// device whose buffer type is the CPU's, so the scheduler gives it every MUL_MAT / MUL_MAT_ID whose weight it accepts
// (supports_op) and keeps every other op -- norms, convolutions, attention, routing -- on the CPU backend, reading
// the same host memory with no copies. First user: the Inkling 2 family (short convolutions, relative-position bias,
// depth mixing, a looped core), which the NPU forward has no ops for.
//
// Arithmetic: W8A8. Each weight is dequantised from its GGUF type and requantised to int8 per output column by the
// production packer (q4nx_pack_B_int8_blocked, blocked 8x8 tiles); each activation row is quantised to int8 with its
// own max-abs scale; the kernel accumulates int32; the result is C * row_scale * col_scale. Weights are packed on
// first use and stay resident (one NPU buffer per weight; an expert stack is one buffer with a sub-buffer per expert).
//
// Kernels: the per-op int8 family (full_i8_<ROLE>_K<K>_N<N>_b8.elf for 128 rows, ..._b8_m16.elf for 16), one hardware
// context per ELF, found in $ONEBIT_NPU_ACCEL_KERNELS. A weight is taken only when an ELF of exactly its staged shape
// (K padded to 256, N to 512) is there; anything else stays on the CPU, so a missing kernel is slower, never wrong.
//
// Environment:
//   ONEBIT_NPU_ACCEL_KERNELS=<dir>  required; the kernel directory
//   ONEBIT_NPU_ACCEL_CHECK=<n>      compare every n-th launch against a host int32 GEMM of the same codes (0 = off)
//   ONEBIT_NPU_ACCEL_STATS=1        print per-shape launch counts and times at exit
//   ONEBIT_NPU_ACCEL_SKIP=<list>    comma list of tensor-name substrings to leave on the CPU (e.g. "ffn_" or "attn_q")
//   ONEBIT_NPU_ACCEL_IDS=1          append the expert ids of every MUL_MAT_ID on a *gate_up* / *_gate_* weight to
//                                   ./npu-accel-ids.txt (working directory), one line per op:
//                                   "<weight> <n_tok> <n_used> ids..." (the routing-agreement gate)
//   (a private add-on can claim matmuls first through npu_accel_ext.h)
//   ONEBIT_NPU_ACCEL_MAX_GB=<n>     ceiling on NPU weight-buffer bytes (default 20): packing past it aborts with a message
//                                   instead of growing until the box runs out of memory (22 Nov: the box OOMed)
//   ONEBIT_NPU_ACCEL_OBSERVE=1      no device: take only MUL_MAT_ID, dump its ids (ONEBIT_NPU_ACCEL_IDS) and compute it on
//                                   ggml's own CPU backend -- the CPU arm's exact routing, for the routing-agreement gate
//   ONEBIT_NPU_ACCEL_REF=1          no device: take only MUL_MAT_ID and compute it in f32 from the dequantised weights
//                                   (the routing baseline for ONEBIT_NPU_ACCEL_IDS; nothing is cached)
//   ONEBIT_NPU_ACCEL_HOST=1         no device: the same int8 codes and int32 sums on the host (accuracy/plumbing check
//                                   off the box); without ONEBIT_NPU_ACCEL_KERNELS it then takes every weight shape
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include "rt.h"
#include "q4nx_pack.h"
#include "q4nx_weight_layout.h"
#include "file_create.h"
#include "npu_accel_ext.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/mman.h>
#include <unistd.h>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml_npu_accel.h"

namespace {

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

inline int pad_to(int v, int m) { return (v + m - 1) / m * m; }

constexpr const char* kIdsFile = "npu-accel-ids.txt";   // ONEBIT_NPU_ACCEL_IDS=1 appends here

// One ELF of the int8 family: its own hardware context and the activation/result buffers it launches with.
struct Kern {
    std::string path;
    int M = 0, KD = 0, ND = 0;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::hw_context> ctx;
    std::unique_ptr<xrt::ext::kernel> k;
    std::unique_ptr<xrt::ext::bo> A, C;
    std::vector<int8_t> hA;     // host mode
    std::vector<int32_t> hC;
    long launches = 0, rows = 0;
    double ms_wait = 0, ms_quant = 0, ms_out = 0;
};

// One packed weight slice (a 2-D weight, or one expert of a stack).
struct Packed {
    xrt::bo bo;                 // KD x ND int8, blocked
    std::vector<float> cs;      // ND column scales (0 past the real N)
    std::vector<int8_t> host;   // the image, kept only when ONEBIT_NPU_ACCEL_CHECK is on
    std::vector<int8_t> hostT;  // host mode: codes as [N][K] rows
};

struct WeightEntry {
    std::unique_ptr<xrt::ext::bo> stack;   // the whole tensor's image (n_slices x KD x ND)
    std::vector<std::unique_ptr<Packed>> slices;
};

struct Shape {
    int KD, ND;
    bool operator<(const Shape& o) const { return KD != o.KD ? KD < o.KD : ND < o.ND; }
};

struct Accel {
    std::mutex mu;
    bool opened = false, failed = false;
    std::string why;
    std::string dir;
    std::unique_ptr<xrt::device> dev;
    std::map<Shape, std::unique_ptr<Kern>> k16, k128;   // by staged shape
    std::unordered_map<const void*, WeightEntry> weights; // by src0->data
    size_t bo_bytes = 0, bo_cap = 0;
    long gc_calls = 0, gc_nodes = 0;
    double gc_ms = 0;   // wall time inside graph_compute (NPU launches + this backend's host work)
    std::vector<std::string> skip;
    int check_every = 0;
    bool stats = false;
    bool host = false, any_shape = false, ref = false, observe = false;
    ggml_backend_t cpu = nullptr;
    FILE* ids_out = nullptr;
    long n_checked = 0, n_check_bad = 0, n_pack = 0;
    double ms_pack = 0;
    std::atomic<long> calls{0};
    std::map<std::string, std::pair<long, double>> per_op;  // "K x N" -> launches, ms (wall, incl. quant)
};

// ---- the private extension hook (npu_accel_ext.h) ----
const onebit::npu_accel::Ext*& ext_slot() { static const onebit::npu_accel::Ext* e = nullptr; return e; }
const onebit::npu_accel::Ext* ext() { return ext_slot(); }
bool& ext_configured() { static bool b = false; return b; }
bool ext_on() { return ext() && ext_configured(); }
std::unique_ptr<xrt::ext::bo> host_weight_bo(size_t bytes, const char* what);
void host_release_pages(const void* p, size_t n);

Accel& G() {
    static Accel a;
    return a;
}

// Kernel directory scan: what shapes exist. Done once, without opening the device, so supports_op can answer
// during graph reservation; the device is opened on the first real launch.
bool scan_dir(Accel& a) {
    const char* d = std::getenv("ONEBIT_NPU_ACCEL_KERNELS");
    a.ref = std::getenv("ONEBIT_NPU_ACCEL_REF") != nullptr;
    a.observe = std::getenv("ONEBIT_NPU_ACCEL_OBSERVE") != nullptr;
    if (a.observe) a.ref = true;   // same supports_op (MUL_MAT_ID only, any shape, no device)
    a.host = a.ref || std::getenv("ONEBIT_NPU_ACCEL_HOST") != nullptr;
    a.check_every = std::getenv("ONEBIT_NPU_ACCEL_CHECK") ? std::atoi(std::getenv("ONEBIT_NPU_ACCEL_CHECK")) : 0;
    a.stats = std::getenv("ONEBIT_NPU_ACCEL_STATS") != nullptr;
    a.bo_cap = (size_t)(std::getenv("ONEBIT_NPU_ACCEL_MAX_GB") ? std::atof(std::getenv("ONEBIT_NPU_ACCEL_MAX_GB")) : 20.0) << 30;
    if (const char* f = std::getenv("ONEBIT_NPU_ACCEL_IDS"); f && *f && std::strcmp(f, "0") != 0) {
        // a dump log several runs append to, at a FIXED name in the working directory (no path is taken from the
        // environment): 0600, no symlink following (file_create.h)
        a.ids_out = onebit_fopen_private_ab(kIdsFile);
        if (!a.ids_out) std::fprintf(stderr, "npu-accel: cannot open the ids dump ./%s\n", kIdsFile);
    }
    if ((!d || !*d) && a.host) { a.any_shape = true; a.dir = "(host, every shape)"; return true; }
    if (!d || !*d) { a.why = "ONEBIT_NPU_ACCEL_KERNELS is not set"; return false; }
    a.dir = d;
    DIR* dp = opendir(d);
    if (!dp) { a.why = std::string("cannot open ") + d; return false; }
    while (dirent* e = readdir(dp)) {
        const std::string f = e->d_name;
        if (f.rfind("full_i8_", 0) != 0) continue;
        int K = 0, N = 0;
        // full_i8_<ROLE>_K<k>_N<n>...: the role may itself start with K ("KV"), so take the last "_K" before "_N<digit>"
        size_t pn = std::string::npos;
        for (size_t q = f.find("_N"); q != std::string::npos; q = f.find("_N", q + 1))
            if (q + 2 < f.size() && std::isdigit((unsigned char)f[q + 2])) { pn = q; break; }
        const size_t pk = pn == std::string::npos ? std::string::npos : f.rfind("_K", pn);
        if (pk == std::string::npos || pn == std::string::npos) continue;
        K = std::atoi(f.c_str() + pk + 2);
        N = std::atoi(f.c_str() + pn + 2);
        if (K <= 0 || N <= 0) continue;
        const std::string tail = f.substr(f.find('_', pn + 2));
        auto kern = std::make_unique<Kern>();
        kern->path = a.dir + "/" + f; kern->KD = K; kern->ND = N;
        if (tail == "_b8_m16.elf") { kern->M = 16; a.k16[{K, N}] = std::move(kern); }
        else if (tail == "_b8.elf") { kern->M = 128; a.k128[{K, N}] = std::move(kern); }
    }
    closedir(dp);
    if (a.k16.empty() && a.k128.empty()) { a.why = a.dir + " holds no full_i8_*_b8[_m16].elf"; return false; }
    if (const char* s = std::getenv("ONEBIT_NPU_ACCEL_SKIP")) {
        std::string cur;
        for (const char* p = s;; ++p) {
            if (*p == ',' || *p == 0) { if (!cur.empty()) a.skip.push_back(cur); cur.clear(); if (!*p) break; }
            else cur += *p;
        }
    }
    if (const char* c = std::getenv("ONEBIT_NPU_ACCEL_CHECK")) a.check_every = std::atoi(c);
    a.stats = std::getenv("ONEBIT_NPU_ACCEL_STATS") != nullptr;
    if (ext()) ext_configured() = ext()->configure();
    return true;
}

const onebit::npu_accel::Host& host(Accel& a) {
    static onebit::npu_accel::Host h;
    h.dev = a.dev.get(); h.weight_bo = host_weight_bo; h.release_pages = host_release_pages;
    return h;
}

bool scanned() {
    static int state = 0;   // 0 = not yet, 1 = ok, 2 = no kernels
    if (state == 0) {
        state = scan_dir(G()) ? 1 : 2;
        if (state == 2) std::fprintf(stderr, "npu-accel: off: %s\n", G().why.c_str());
        else std::fprintf(stderr, "npu-accel: %zu shapes at 16 rows, %zu at 128 rows in %s\n", G().k16.size(),
                          G().k128.size(), G().dir.c_str());
    }
    return state == 1;
}

bool open_kern(Accel& a, Kern& k) {
    if (a.host) {
        if (k.hA.empty()) { k.hA.assign((size_t)k.M * k.KD, 0); k.hC.assign((size_t)k.M * k.ND, 0); }
        return true;
    }
    if (k.k) return true;
    k.elf = std::make_unique<xrt::elf>(k.path);
    k.ctx = std::make_unique<xrt::hw_context>(*a.dev, *k.elf);
    k.k = std::make_unique<xrt::ext::kernel>(*k.ctx, "main:seq");
    k.A = std::make_unique<xrt::ext::bo>(*a.dev, (size_t)k.M * k.KD);
    k.C = std::make_unique<xrt::ext::bo>(*a.dev, (size_t)k.M * k.ND * 4);
    std::memset(k.A->map(), 0, (size_t)k.M * k.KD);
    return true;
}

bool ensure_device(Accel& a) {
    if (a.opened) return !a.failed;
    a.opened = true;
    if (a.host) return true;
    try {
        a.dev = std::make_unique<xrt::device>(0);
    } catch (const std::exception& e) {
        a.failed = true; a.why = std::string("device open failed: ") + e.what();
        std::fprintf(stderr, "npu-accel: %s\n", a.why.c_str());
        return false;
    }
    if (ext_on()) {
        std::string why;
        if (!ext()->open(host(a), &why)) {
            std::fprintf(stderr, "npu-accel: extension %s off: %s\n", ext()->name, why.c_str());
            ext_configured() = false;
        }
    }
    return true;
}

// Every weight buffer goes through here: counted against ONEBIT_NPU_ACCEL_MAX_GB.
std::unique_ptr<xrt::ext::bo> weight_bo(Accel& a, size_t bytes, const char* what) {
    if (a.bo_bytes + bytes > a.bo_cap)
        GGML_ABORT("npu-accel: weight buffers would reach %.1f GB (> ONEBIT_NPU_ACCEL_MAX_GB %.1f) packing %s; refusing to grow",
                   (a.bo_bytes + bytes) / 1073741824.0, a.bo_cap / 1073741824.0, what);
    a.bo_bytes += bytes;
    return std::make_unique<xrt::ext::bo>(*a.dev, bytes);
}

// The GGUF pages a packed weight came from: the CPU never reads them again (the op is ours), so hand them back to the
// kernel instead of keeping a second copy of every packed weight resident.
std::unique_ptr<xrt::ext::bo> host_weight_bo(size_t bytes, const char* what) { return weight_bo(G(), bytes, what); }
void host_release_pages(const void* p, size_t n);

void release_pages(const void* p, size_t n) {
    static const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t b = ((uintptr_t)p + pg - 1) & ~(pg - 1), e = ((uintptr_t)p + n) & ~(pg - 1);
    if (e > b) madvise((void*)b, e - b, MADV_DONTNEED);
}

// Is this a weight matmul we can take? (src0 is a weight in host memory, its staged shape has a kernel.)
void host_release_pages(const void* p, size_t n) { release_pages(p, n); }

bool weight_ok(const ggml_tensor* src0, const ggml_tensor* src1) {
    if (!src0 || !src1 || !src0->buffer) return false;
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
    if (!ggml_backend_buffer_is_host(src0->buffer)) return false;
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float)) return false;
    if (!ggml_is_contiguous(src0)) return false;
    if (src0->type != GGML_TYPE_F32 && ggml_get_type_traits(src0->type)->to_float == nullptr) return false;
    if (src0->ne[3] != 1) return false;
    const int KD = pad_to((int)src0->ne[0], 256), ND = pad_to((int)src0->ne[1], 512);
    Accel& a = G();
    const bool ext_takes = ext_on() && (src0->ne[2] == 1 ? ext()->takes_mul_mat(src0) : ext()->takes_mul_mat_id(src0));
    if (!a.any_shape && !ext_takes && !a.k16.count({KD, ND}) && !a.k128.count({KD, ND})) return false;
    for (const auto& s : a.skip)
        if (std::strstr(src0->name, s.c_str())) return false;
    return true;
}

// Pack slice `e` of weight src0 (rows of K, N rows per slice) into its resident image.
Packed* packed_slice(Accel& a, const ggml_tensor* src0, int e) {
    WeightEntry& we = a.weights[src0->data];
    const int K = (int)src0->ne[0], N = (int)src0->ne[1];
    const int nsl = (int)src0->ne[2];
    const int KD = pad_to(K, 256), ND = pad_to(N, 512);
    const size_t img = (size_t)KD * ND;
    if (we.slices.empty()) {
        we.slices.resize((size_t)nsl);
        if (!a.host) we.stack = weight_bo(a, img * nsl, src0->name);
    }
    if (we.slices[(size_t)e]) return we.slices[(size_t)e].get();
    const auto t0 = clk::now();
    auto p = std::make_unique<Packed>();
    std::vector<int8_t> himg;
    if (a.host) himg.assign(img, 0);
    else p->bo = xrt::bo(*we.stack, img, img * (size_t)e);
    // dequantise rows [N][K] then transpose to the packer's [K][N]
    std::vector<float> rowsf((size_t)N * K), w((size_t)K * N);
    const char* base = (const char*)src0->data + (size_t)e * src0->nb[2];
    const auto* tt = ggml_get_type_traits(src0->type);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const char* row = base + (size_t)n * src0->nb[1];
        if (src0->type == GGML_TYPE_F32) std::memcpy(&rowsf[(size_t)n * K], row, (size_t)K * 4);
        else tt->to_float(row, &rowsf[(size_t)n * K], K);
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) w[(size_t)k * N + n] = rowsf[(size_t)n * K + k];
    int8_t* B = a.host ? himg.data() : p->bo.map<int8_t*>();
    std::memset(B, 0, img);
    p->cs.assign((size_t)ND, 0.0f);
    double ssum = 0;
    q4nx_pack_B_int8_blocked(B, w.data(), K, N, KD, ND, 1, p->cs.data(), &ssum);
    for (int n = N; n < ND; n++) p->cs[(size_t)n] = 0.0f;
    if (a.host) {
        p->hostT.resize((size_t)N * KD);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < KD; k++) p->hostT[(size_t)n * KD + k] = B[q4nx_layout::blocked_index(k, n, KD, ND)];
    } else {
        p->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (a.check_every > 0) p->host.assign(B, B + img);
    }
    a.n_pack++;
    a.ms_pack += ms_since(t0);
    we.slices[(size_t)e] = std::move(p);
    return we.slices[(size_t)e].get();
}

// rows x K activations (row stride lda floats) times one packed slice -> rows x N into out (row stride ldo floats).
bool launch_rows(Accel& a, const Packed& p, int K, int N, int rows, const float* x, size_t lda, float* out, size_t ldo,
                 const char* name) {
    const int KD = pad_to(K, 256), ND = pad_to(N, 512);
    for (int r0 = 0; r0 < rows;) {
        const int left = rows - r0;
        Kern* kern = nullptr;
        if (a.any_shape && !a.k16.count({KD, ND})) {
            auto k = std::make_unique<Kern>();
            k->M = 16; k->KD = KD; k->ND = ND; k->path = "(host)";
            a.k16[{KD, ND}] = std::move(k);
        }
        auto i16 = a.k16.find({KD, ND});
        auto i128 = a.k128.find({KD, ND});
        if (left <= 16 && i16 != a.k16.end()) kern = i16->second.get();
        else if (i128 != a.k128.end()) kern = i128->second.get();
        else if (i16 != a.k16.end()) kern = i16->second.get();
        if (!kern) return false;
        open_kern(a, *kern);
        const int m = left < kern->M ? left : kern->M;
        const auto t0 = clk::now();
        int8_t* A = a.host ? kern->hA.data() : kern->A->map<int8_t*>();
        std::vector<float> rs((size_t)m);
#pragma omp parallel for schedule(static) if (m > 4)
        for (int r = 0; r < m; r++) {
            const float* xr = x + (size_t)(r0 + r) * lda;
            float am = 0.0f;
            for (int i = 0; i < K; i++) { const float v = std::fabs(xr[i]); if (v > am && std::isfinite(v)) am = v; }
            const float s = am > 0.0f ? am / 127.0f : 1.0f;
            const float is = 1.0f / s;
            rs[(size_t)r] = s;
            int8_t* ar = A + (size_t)r * KD;
            for (int i = 0; i < K; i++) {
                float v = xr[i] * is;
                v = std::isfinite(v) ? v : 0.0f;
                int q = (int)std::lrintf(v);
                ar[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
            }
            std::memset(ar + K, 0, (size_t)(KD - K));
        }
        // rows past m keep whatever the last launch left: their results are never read
        const auto t1 = clk::now();
        const int32_t* C = nullptr;
        if (a.host) {
            int32_t* hc = kern->hC.data();
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; n++) {
                const int8_t* w = &p.hostT[(size_t)n * KD];
                for (int r = 0; r < m; r++) {
                    const int8_t* ar = A + (size_t)r * KD;
                    int32_t acc = 0;
                    for (int i = 0; i < K; i++) acc += (int32_t)ar[i] * (int32_t)w[i];
                    hc[(size_t)r * ND + n] = acc;
                }
            }
            C = hc;
        } else {
            kern->A->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            xrt::run run = (*kern->k)(*kern->A, p.bo, *kern->C);
            const int st = (int)run.wait();
            if (st != 4) {
                std::fprintf(stderr, "npu-accel: %s K%d N%d: launch ended in state %d\n", name, KD, ND, st);
                return false;
            }
            kern->C->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            C = kern->C->map<int32_t*>();
        }
        const auto t2 = clk::now();
        const long call = ++a.calls;
        if (a.check_every > 0 && call % a.check_every == 0 && !p.host.empty()) {
            long bad = 0;
            for (int r = 0; r < m; r++)
                for (int n = 0; n < N; n++) {
                    int32_t acc = 0;
                    for (int i = 0; i < K; i++)
                        acc += (int32_t)A[(size_t)r * KD + i] * (int32_t)p.host[q4nx_layout::blocked_index(i, n, KD, ND)];
                    bad += acc != C[(size_t)r * ND + n];
                }
            a.n_checked++;
            if (bad) {
                a.n_check_bad++;
                std::fprintf(stderr, "npu-accel: CHECK FAIL %s K%d N%d rows %d: %ld int32 differ\n", name, KD, ND, m, bad);
            }
        }
#pragma omp parallel for schedule(static) if (m > 4)
        for (int r = 0; r < m; r++) {
            float* o = out + (size_t)(r0 + r) * ldo;
            const int32_t* c = C + (size_t)r * ND;
            const float s = rs[(size_t)r];
            for (int n = 0; n < N; n++) o[n] = (float)c[n] * s * p.cs[(size_t)n];
        }
        const auto t3 = clk::now();
        kern->launches++;
        kern->rows += m;
        kern->ms_quant += std::chrono::duration<double, std::milli>(t1 - t0).count();
        kern->ms_wait += std::chrono::duration<double, std::milli>(t2 - t1).count();
        kern->ms_out += std::chrono::duration<double, std::milli>(t3 - t2).count();
        r0 += m;
    }
    return true;
}

void mul_mat(Accel& a, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const int K = (int)src0->ne[0], N = (int)src0->ne[1];
    GGML_ASSERT(src0->ne[2] == 1);
    if (ext_on() && ext()->mul_mat(host(a), dst)) return;
    Packed* p = packed_slice(a, src0, 0);
    // src1: [K, ne11, ne12, ne13] -> rows of K floats
    for (int64_t i13 = 0; i13 < src1->ne[3]; i13++)
        for (int64_t i12 = 0; i12 < src1->ne[2]; i12++) {
            const float* x = (const float*)((const char*)src1->data + i12 * src1->nb[2] + i13 * src1->nb[3]);
            float* d = (float*)((char*)dst->data + i12 * dst->nb[2] + i13 * dst->nb[3]);
            const auto t0 = clk::now();
            if (!launch_rows(a, *p, K, N, (int)src1->ne[1], x, src1->nb[1] / 4, d, dst->nb[1] / 4, src0->name))
                GGML_ABORT("npu-accel: %s launch failed", src0->name);
            auto& po = a.per_op[std::to_string(K) + "x" + std::to_string(N)];
            po.first++; po.second += ms_since(t0);
        }
}

void dump_ids(Accel& a, const ggml_tensor* as, const ggml_tensor* ids) {
    const int n_used = (int)ids->ne[0], n_tok = (int)ids->ne[1];
    if (a.ids_out && (std::strstr(as->name, "gate_up") || std::strstr(as->name, "_gate_"))) {
        std::fprintf(a.ids_out, "%s %d %d", as->name, n_tok, n_used);
        for (int t = 0; t < n_tok; t++)
            for (int i = 0; i < n_used; i++)
                std::fprintf(a.ids_out, " %d",
                             *(const int32_t*)((const char*)ids->data + (size_t)i * ids->nb[0] + (size_t)t * ids->nb[1]));
        std::fputc('\n', a.ids_out);
        std::fflush(a.ids_out);
    }
}

void mul_mat_id(Accel& a, ggml_tensor* dst) {
    if (a.ids_out) dump_ids(a, dst->src[0], dst->src[2]);
    if (a.observe) {
        if (!a.cpu) {
            a.cpu = ggml_backend_cpu_init();
            const char* t = std::getenv("ONEBIT_NPU_ACCEL_OBSERVE_THREADS");
            ggml_backend_cpu_set_n_threads(a.cpu, t ? std::atoi(t) : 4);
        }
        static ggml_context* gctx = nullptr;
        if (!gctx) { ggml_init_params ip = {ggml_graph_overhead() + 1024, nullptr, true}; gctx = ggml_init(ip); }
        ggml_cgraph* gf = ggml_new_graph(gctx);
        ggml_graph_add_node(gf, dst);
        if (ggml_backend_graph_compute(a.cpu, gf) != GGML_STATUS_SUCCESS) GGML_ABORT("npu-accel observe: CPU compute failed");
        ggml_free(gctx); gctx = nullptr;
        return;
    }
    if (ext_on() && ext()->mul_mat_id(host(a), dst)) return;
    const ggml_tensor* as = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    const ggml_tensor* ids = dst->src[2];
    const int K = (int)as->ne[0], N = (int)as->ne[1], E = (int)as->ne[2];
    const int n_used = (int)ids->ne[0], n_tok = (int)ids->ne[1];
    std::vector<std::vector<std::pair<int, int>>> by_e((size_t)E);
    for (int t = 0; t < n_tok; t++)
        for (int i = 0; i < n_used; i++) {
            const int e = *(const int32_t*)((const char*)ids->data + (size_t)i * ids->nb[0] + (size_t)t * ids->nb[1]);
            GGML_ASSERT(e >= 0 && e < E);
            by_e[(size_t)e].emplace_back(i, t);
        }
    if (a.ref) {
        const auto* tt = ggml_get_type_traits(as->type);
        for (int e = 0; e < E; e++) {
            const auto& v = by_e[(size_t)e];
            if (v.empty()) continue;
            const char* base = (const char*)as->data + (size_t)e * as->nb[2];
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; n++) {
                std::vector<float> wr((size_t)K);
                const char* row = base + (size_t)n * as->nb[1];
                if (as->type == GGML_TYPE_F32) std::memcpy(wr.data(), row, (size_t)K * 4);
                else tt->to_float(row, wr.data(), K);
                for (const auto& it : v) {
                    const float* x = (const float*)((const char*)b->data + (size_t)(it.first % b->ne[1]) * b->nb[1] +
                                                    (size_t)it.second * b->nb[2]);
                    double acc = 0;
                    for (int k = 0; k < K; k++) acc += (double)wr[(size_t)k] * x[k];
                    *(float*)((char*)dst->data + (size_t)n * 4 + (size_t)it.first * dst->nb[1] + (size_t)it.second * dst->nb[2]) =
                        (float)acc;
                }
            }
        }
        return;
    }
    std::vector<float> xa, ya;
    for (int e = 0; e < E; e++) {
        const auto& v = by_e[(size_t)e];
        if (v.empty()) continue;
        Packed* p = packed_slice(a, as, e);
        const int rows = (int)v.size();
        xa.resize((size_t)rows * K);
        ya.resize((size_t)rows * N);
        for (int r = 0; r < rows; r++) {
            const int i = v[(size_t)r].first, t = v[(size_t)r].second;
            const float* src = (const float*)((const char*)b->data + (size_t)(i % b->ne[1]) * b->nb[1] + (size_t)t * b->nb[2]);
            std::memcpy(&xa[(size_t)r * K], src, (size_t)K * 4);
        }
        const auto t0 = clk::now();
        if (!launch_rows(a, *p, K, N, rows, xa.data(), (size_t)K, ya.data(), (size_t)N, as->name))
            GGML_ABORT("npu-accel: %s expert %d launch failed", as->name, e);
        auto& po = a.per_op[std::to_string(K) + "x" + std::to_string(N) + " (expert)"];
        po.first++; po.second += ms_since(t0);
        for (int r = 0; r < rows; r++) {
            const int i = v[(size_t)r].first, t = v[(size_t)r].second;
            float* d = (float*)((char*)dst->data + (size_t)i * dst->nb[1] + (size_t)t * dst->nb[2]);
            std::memcpy(d, &ya[(size_t)r * N], (size_t)N * 4);
        }
    }
}

void dump_stats() {
    Accel& a = G();
    if (!a.stats) return;
    std::fprintf(stderr, "npu-accel stats: graph_compute calls %ld (matmul nodes %ld), %.1f s inside\n", a.gc_calls,
                 a.gc_nodes, a.gc_ms / 1000.0);
    std::fprintf(stderr, "npu-accel stats: weight buffers %.2f GB (cap %.1f)\n", a.bo_bytes / 1073741824.0, a.bo_cap / 1073741824.0);
    std::fprintf(stderr, "npu-accel stats: %ld weight slices packed in %.1f s; checks %ld (%ld failed)\n", a.n_pack,
                 a.ms_pack / 1000.0, a.n_checked, a.n_check_bad);
    for (auto* m : {&a.k16, &a.k128})
        for (auto& kv : *m) {
            const Kern& k = *kv.second;
            if (!k.launches) continue;
            std::fprintf(stderr, "  K%-5d N%-5d M%-3d launches %8ld rows %9ld  quant %.3f  wait %.3f  out %.3f ms/launch\n",
                         k.KD, k.ND, k.M, k.launches, k.rows, k.ms_quant / k.launches, k.ms_wait / k.launches,
                         k.ms_out / k.launches);
        }
    if (ext_on() && ext()->stats) ext()->stats();
    for (auto& kv : a.per_op)
        std::fprintf(stderr, "  op %-18s calls %8ld  %.3f ms/call\n", kv.first.c_str(), kv.second.first,
                     kv.second.second / kv.second.first);
}

// ---- backend ----
const char* be_name(ggml_backend_t) { return "NPU"; }
void be_free(ggml_backend_t b) { delete b; }

ggml_status be_graph_compute(ggml_backend_t, ggml_cgraph* g) {
    Accel& a = G();
    std::lock_guard<std::mutex> lk(a.mu);
    if (!ensure_device(a)) return GGML_STATUS_FAILED;
    const auto t_gc = clk::now();
    struct Done { Accel& a; clk::time_point t; ~Done() { a.gc_ms += ms_since(t); a.gc_calls++; } } done{a, t_gc};
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i]->op == GGML_OP_MUL_MAT || g->nodes[i]->op == GGML_OP_MUL_MAT_ID) a.gc_nodes++;
        ggml_tensor* node = g->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT: mul_mat(a, node); break;
            case GGML_OP_MUL_MAT_ID: mul_mat_id(a, node); break;
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: break;
            default: GGML_ABORT("npu-accel: unsupported op %s", ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

ggml_guid_t be_guid() {
    static ggml_guid guid = {0x1b, 0x17, 0x6e, 0x70, 0x75, 0x61, 0x63, 0x63, 0x65, 0x6c, 0x2d, 0x78, 0x64, 0x6e, 0x61, 0x32};
    return &guid;
}

ggml_backend_i be_iface = {
    /* .get_name                = */ be_name,
    /* .free                    = */ be_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ be_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ---- device ----
const char* dev_name(ggml_backend_dev_t) { return "NPU"; }
const char* dev_desc(ggml_backend_dev_t) { return "XDNA2 NPU, int8 per-op GEMM (own runtime)"; }
void dev_memory(ggml_backend_dev_t, size_t* f, size_t* t) { *f = 0; *t = 0; }
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
void dev_props(ggml_backend_dev_t d, ggml_backend_dev_props* p) {
    p->name = dev_name(d);
    p->description = dev_desc(d);
    p->type = dev_type(d);
    dev_memory(d, &p->memory_free, &p->memory_total);
    p->caps = {false, false, true, false, true};
}
ggml_backend_dev_t the_dev();
ggml_backend_t dev_init(ggml_backend_dev_t, const char*) {
    return new ggml_backend{be_guid(), be_iface, the_dev(), nullptr};
}
ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }
ggml_backend_buffer_t dev_from_ptr(ggml_backend_dev_t, void* ptr, size_t size, size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}
bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    if (!scanned()) return false;
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            if (G().ref) return false;
            return op->src[0]->ne[2] == 1 && weight_ok(op->src[0], op->src[1]) && ggml_is_contiguous_rows(op->src[1]);
        case GGML_OP_MUL_MAT_ID:
            return weight_ok(op->src[0], op->src[1]) && op->src[2] && op->src[2]->type == GGML_TYPE_I32 &&
                   op->src[1]->nb[0] == sizeof(float);
        default:
            return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return ggml_backend_buft_is_host(buft); }

const ggml_backend_device_i dev_iface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_desc,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init,
    /* .get_buffer_type      = */ dev_buft,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ dev_from_ptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---- reg ----
const char* reg_name(ggml_backend_reg_t) { return "NPU"; }
size_t reg_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t reg_dev(ggml_backend_reg_t, size_t) { return the_dev(); }
void* reg_proc(ggml_backend_reg_t, const char*) { return nullptr; }
const ggml_backend_reg_i reg_iface = {reg_name, reg_count, reg_dev, reg_proc};

ggml_backend_reg_t the_reg() {
    static ggml_backend_reg r = {GGML_BACKEND_API_VERSION, reg_iface, nullptr};
    return &r;
}
ggml_backend_dev_t the_dev() {
    static ggml_backend_device d = {dev_iface, the_reg(), nullptr};
    return &d;
}

}  // namespace

extern "C" bool ggml_npu_accel_register() {
    if (!scanned()) return false;
    ggml_backend_register(the_reg());
    std::atexit(dump_stats);
    return true;
}

void onebit::npu_accel::register_ext(const Ext* e) { ext_slot() = e; }
