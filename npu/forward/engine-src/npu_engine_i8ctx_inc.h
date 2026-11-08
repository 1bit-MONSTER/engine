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
// npu_engine_i8ctx_inc.h — I8Ctx: one int8 GEMM design, run from a full ELF.
//
// The kernel is an xrt::ext::kernel built from the full ELF, signature
// (A, B, C): one activation BO (A), one weight BO per layer (B), one output
// BO (C). The ELF carries the PDIs and the control code; the engine loads no
// xclbin and no instruction file.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <climits>
#include <cstdlib>
#include <map>
#include <string>

#include "q4nx_weight_layout.h"   // bpos(): the container's device weight layout
#include "q4nx_pb16i.h"          // the pb16i image: integer-exact ternary decode weights
#include "q4nx_t2s.h"            // the t2s stream: the same weights at 2.125 bits/weight
#include "q4nx_pack.h"           // q4nx_elf_arg_sizes(): args_fit_elf() checks the ELF's own widths
#include "quant_i8.h"             // the A-row quantiser: scalar reference + an AVX2 path, byte-identical
#include <algorithm>
#include "rt.h"  // the own NPU runtime (ONEBIT_NPU_RT_DIR)


// One full ELF, one PDI, one hw_context -- and one kernel handle per instance name
// ("main:<instance>").  This is not a convenience: switching hw_context costs ~2.7 ms on
// this box, independent of the stream size (measured 2026-10-31: 0.488 ms per launch back
// to back in one context, 3.005 alternating two, 3.203 alternating seven), so every design
// whose ELF carries several instances shares one context and only the kernel handle
// differs.  The pool below is process-wide because the engine opens one device and one
// model; it is keyed by the ELF's canonical path, so two roles pointing at the same
// artifact (G and U share one pb16i ELF today) also share its context.
struct ElfHost {
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::hw_context> hc;
    std::map<std::string, std::unique_ptr<xrt::ext::kernel>> kernels;
    xrt::ext::kernel* get(const std::string& inst) {
        auto it = kernels.find(inst);
        if (it != kernels.end()) return it->second.get();
        auto k = std::make_unique<xrt::ext::kernel>(*hc, inst);
        xrt::ext::kernel* raw = k.get();
        kernels.emplace(inst, std::move(k));
        return raw;
    }
};

struct I8Ctx {
    int MD, KD, ND, NL;
    // Consume a `aiecc --generate-full-elf` artifact directly: the ELF bundles
    // the PDIs + TXN control code, so there is no instruction file. The
    // hw_context is created from the ELF, and kext is the resulting
    // xrt::ext::kernel whose signature is (A, B, C). Requires the design's
    // runtime sequence to load the ELF's PDI (aiex.npu.load_pdi; see
    // generators/build_zaya_m8.sh), otherwise the runtime never configures the array.
    std::shared_ptr<ElfHost> host_;
    xrt::ext::kernel* kext = nullptr;
    std::unique_ptr<xrt::bo> bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;     // weight BOs
    int8_t* Am;
    int32_t* Cm;
    std::vector<std::vector<float>> group_scales;
    bool initialized = false;

    ~I8Ctx() {}

    bool isReady() { return initialized && kext && bA && bC; }

    // Device weight layout: false = padded row-major [KD][ND] (every container
    // before q4nx_weight_layout "b8"), true = the blocked 8x8-microtile image.
    // Set by Q4nxNpuForward from the container's config before the first pack.
    bool blocked_b = false;

    // Weight image "pb16i" (q4nx_weight_layout "pb16i"): the BO holds a pb16i sidecar
    // unchanged (codes + int16 multipliers), there is no packing and no int8 scale --
    // the per-column 2^e_ref table and the int32/int64 mode come from the sidecar's
    // own `.ref` file, and the host applies the single rounding.  The output BO grows
    // to two int32 planes for the int64 mode (S = hi*2^21 + lo), which the kernel
    // writes as [M][2N]; bC is allocated for both modes so one BO serves either.
    bool pb16i = false;
    // Weight stream "t2s" (q4nx_weight_layout "t2s"): the same integer-exact ternary contract as
    // pb16i, with the codes at 2 bits and the multiplier once per 128-K block -- the kernel expands
    // them to the int4 nibbles the mmul takes.  Everything else (the sidecar `.ref` side table, the
    // int32/int64 accumulation, the host's single rounding) is pb16i's, so `pb16i` is set too and
    // only the image's file extension and size differ.  A container never sets both layouts; the
    // flag is on the context so a load cannot read a 4,608-B image as a 2,176-B one.
    bool t2s = false;
    // Weight stream "gemv": the forge-built q4_1 GEMV family.  Its runtime sequence declares
    // (arg0 = the tensor's own q4_1 tile stream, arg1 = K bf16 activations, arg2 = N f32 outputs) --
    // another order and another dtype set than the int8 per-tile family this launcher was written
    // for, so the family is detected from the artifact's own declared widths at load (see
    // args_fit_elf) and this flag carries it.  One row per launch: the family has no M dimension (its
    // `_m16` ELFs are byte-identical to the one-row design), which is why the batched paths fall back
    // to per-row launches.  Measured 2026-11-02 on the OGA 0.5B pack's five ELFs.
    bool gemv = false;
    uint16_t* Ag = nullptr;   // gemv: the bf16 activation BO, mapped
    float*    Cg = nullptr;   // gemv: the f32 output BO, mapped

    // Bytes of the gemv weight stream: one 5,120-byte q4_1 tile per (32 rows, 256 K columns).
    static size_t gemv_w_bytes(int K, int N) {
        return (size_t)(K / 256) * (size_t)(N / 32) * 5120u;
    }

    // int32 planes the pb16i ELF leaves its sum in: 1 (plain int32) or 2 (ACC64:
    // lo = sum of (t & 2^21-1), hi = sum of (t >> 21), S = hi*2^21 + lo).  Both are
    // exact, so a tensor whose own bound fits int32 may still run on the two-plane
    // kernel; the reverse is not allowed, and load_pb16i_into refuses it.
    int pb16i_planes = 1;

    // Rows of C the host will read after the launch that is in flight: the activation rows the
    // quantiser wrote. Both C syncs (the canary's writeback and readback()'s invalidate) cover
    // exactly these rows instead of the whole BO -- see c_host_bytes().
    int last_am_ = 1;
    // engine#1416: [ps] pack-split accumulators (NPU_PACK_SPLIT=1)
    using pack_clock = std::chrono::steady_clock;
    // Bucketed by row count: the first version printed the LAST call's M with a CUMULATIVE average,
    // which is how a prefill window came back reading "M=1" while prefill was demonstrably batched.
    struct PackBucket { long n = 0; double q = 0, sy = 0, cn = 0, lc = 0; };
    std::map<long, PackBucket> pack_buckets_;
    inline void pack_split_add(double q, double sy, double cn, double lc, int am, int ak) {
        PackBucket& b = pack_buckets_[(long)am * 100000 + ak];
        b.n++; b.q += q; b.sy += sy; b.cn += cn; b.lc += lc;
        pack_split_n_++;
        if (pack_split_n_ % 100 == 0) {
            fprintf(stderr, "[ps] total=%ld launches, by rows:", pack_split_n_);
            for (const auto& kv : pack_buckets_)
                fprintf(stderr, " | M=%ld K=%ld n=%ld quant=%.3f sync=%.3f canary=%.3f launch=%.3f ms",
                        kv.first / 100000, kv.first % 100000, kv.second.n, kv.second.q / kv.second.n,
                        kv.second.sy / kv.second.n, kv.second.cn / kv.second.n,
                        kv.second.lc / kv.second.n);
            fprintf(stderr, "\n");
        }
    }
    long pack_split_n_ = 0;
    double pack_split_q_ = 0, pack_split_sync_ = 0, pack_split_canary_ = 0, pack_split_launch_ = 0;
    // A client that writes A itself instead of going through the quantisers has to say how many rows it
    // wrote, or the narrow invalidation would be sized for one row: onnx/ort/onebit_npu_gemm.cpp memcpys the
    // graph's int8 rows straight into Am, so it calls this with the rows of the launch it is about to make.
    void set_activation_rows(int rows) {
        last_am_ = rows > 0 ? rows : 1;
        // ...but its A writeback stays the whole BO: that client memsets all MD rows first (the pad rows have to
        // read as zero) and only then memcpys `rows` of them, so every line of bA is dirty. Narrowing this to
        // `rows` would leave the memset's zeroes in cache to be evicted *after* the kernel read them.
        a_dirty(MD);
    }

    // Rows of bA the host has written since the last A writeback (audit row 17). The array reads rows
    // [0, MD), but only lines the host touched can be dirty in cache, and every other row was flushed
    // when it was last written -- so the writeback's range is the dirty span, not the BO. See a_host_bytes().
    int a_dirty_rows_ = 0;
    inline void a_dirty(int rows) { if (rows > a_dirty_rows_) a_dirty_rows_ = rows; }

    // Bytes of one weight image: the int8 layouts are KD*ND, pb16i and t2s are their own sizes.
    size_t w_bytes() const {
        if (gemv) return gemv_w_bytes(KD, ND);
        if (t2s) return q4nx_t2s::w_bytes(KD, ND);
        return pb16i ? q4nx_pb16i::w_bytes(KD, ND) : (size_t)KD * (size_t)ND;
    }

    // ── Init: consume a FULL ELF directly ──
    // Kernel signature: (bo0, bo1, bo2) == (A, B, C). No instruction BO: the
    // TXN control stream lives in the ELF's .ctrltext. Kernel name is
    // "<kernel>:<instance>" from the aiecc full-ELF config ("main:seq" for the
    // v27 designs).
    bool init_elf(xrt::device& d, const char* ep, int M, int K, int N,
                  int nlayers, const char* kname = "main:seq") {
        MD = M; KD = K; ND = N; NL = nlayers;
        fprintf(stderr, "  I8Ctx::init_elf ep=%s M=%d K=%d N=%d nl=%d\n",
                ep, M, K, N, nlayers);
        // The blocked image is built from whole 64x128 tiles: a geometry that is
        // not a multiple of those has no blocked mapping, and silently packing it
        // would index outside the tile grid.
        if (blocked_b && ((size_t)KD % 64 != 0 || (size_t)ND % 128 != 0)) {
            fprintf(stderr, "  I8Ctx::init_elf: the b8 layout needs K%%64==0 and N%%128==0 "
                            "(K=%d N=%d)\n", KD, ND);
            return false;
        }
        if (t2s && !q4nx_t2s::fits(KD, ND)) {
            fprintf(stderr, "  I8Ctx::init_elf: the t2s layout needs K%%256==0 and N%%32==0 "
                            "(K=%d N=%d)\n", KD, ND);
            return false;
        }
        if (pb16i && !t2s && !q4nx_pb16i::fits(KD, ND)) {
            fprintf(stderr, "  I8Ctx::init_elf: the pb16i layout needs K%%256==0 and N%%1024==0 "
                            "(K=%d N=%d)\n", KD, ND);
            return false;
        }
        try {
            // One context per artifact: a second design on the same ELF reuses it, so
            // going from one role to another inside that ELF does not pay a switch.
            char canon[PATH_MAX];
            const std::string key = realpath(ep, canon) ? std::string(canon) : std::string(ep);
            static std::map<std::string, std::shared_ptr<ElfHost>> pool;
            auto it = pool.find(key);
            if (it != pool.end()) {
                host_ = it->second;
            } else {
                auto h = std::make_shared<ElfHost>();
                h->elf = std::make_unique<xrt::elf>(key);
                if (!h->elf->is_full_elf()) {
                    fprintf(stderr, "  I8Ctx::init_elf: %s is not a full ELF\n", ep);
                    return false;
                }
                fprintf(stderr, "  full ELF ok: uuid=%s\n",
                        h->elf->get_cfg_uuid().to_string().c_str());
                h->hc = std::make_unique<xrt::hw_context>(d, *h->elf);
                host_ = h;
                pool.emplace(key, h);
            }
            kext = host_->get(std::string(kname));
            fprintf(stderr, "  I8Ctx::init_elf %s: %s instance %s (%zu in this context)\n",
                    ep, key.c_str(), kname, host_->kernels.size());
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx::init_elf failed: %s\n", ex.what());
            return false;
        }
        // A group-less xrt::bo is rejected by the amdxdna driver ("unsupported
        // buffer type: none flag") under the system XRT; xrt::ext::bo picks a
        // valid host-accessible memory group for the device.
        // Which launch family is this artifact?  The declared widths answer it: the gemv family takes
        // (weights = its tile stream, activations = K bf16, outputs = N f32), the int8 families take
        // (activations = MD*K int8, weights, outputs = MD*N int32).  Decided here, before any BO is
        // sized, because the widths follow from it.
        gemv = elf_is_gemv(ep);
        if (gemv && (KD % 256 || ND % 32)) {
            fprintf(stderr, "  I8Ctx::init_elf: the gemv family needs K%%256==0 and N%%32==0 (K=%d N=%d)\n",
                    KD, ND);
            return false;
        }
        auto mkbo = [&](size_t sz) {
            xrt::ext::bo b{d, sz};
            return std::make_unique<xrt::bo>(b);
        };
        bA = mkbo(gemv ? (size_t)KD * 2 : (size_t)MD * KD);
        // pb16i int64 mode returns two int32 planes ([M][2N]); allocate for both modes
        // so the tensor's own mode byte, not the geometry, decides what is used.
        bC = mkbo(gemv ? (size_t)ND * 4 : (size_t)MD * ND * 4 * (pb16i ? 2 : 1));
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();
        if (gemv) { Ag = (uint16_t*)Am; Cg = (float*)Cm; }
        layerB.resize(NL);
        group_scales.resize(NL);   // packB() writes group_scales[l]
        for (int l = 0; l < NL; l++)
            layerB[l] = mkbo(w_bytes());
        if (!args_fit_elf(ep)) return false;
        initialized = true;
        return true;
    }

    // ── The launch contract, checked against the artifact ──
    // A full ELF declares each argument's byte width in .dynsym (the driver patches those very
    // symbols), and the kernel's shim DMA touches exactly that much.  Passing a narrower buffer does
    // not fail: the DMA walks past the mapping, the IOMMU logs IO_PAGE_FAULTs and the run returns
    // garbage (measured 2026-11-02 on the 896-hidden OGA pack, whose ELF family is
    // (weights, activations, output) = (655,360, 2,048, 4,096) B while the int8 launcher passed
    // (16,384, 1,048,576, 65,536) B).  The check is one-directional on purpose: a declared width may
    // be *smaller* than the buffer (the tile-family ELFs address a tile of a larger BO through the
    // control text), but it may never be larger.
    // Does this artifact declare the gemv family's widths?  All three have to match: a tile stream for
    // arg0, 2*K bf16 activations for arg1, N f32 outputs for arg2.  Anything else is one of the int8
    // families (whose contract the guard below checks).
    bool elf_is_gemv(const char* ep) {
        std::vector<uint8_t> raw;
        std::FILE* f = std::fopen(ep, "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        const long len = std::ftell(f);
        std::rewind(f);
        if (len <= 0) { std::fclose(f); return false; }
        raw.resize((size_t)len);
        const size_t got = std::fread(raw.data(), 1, raw.size(), f);
        std::fclose(f);
        if (got != raw.size()) return false;
        uint64_t declared[3] = {0, 0, 0};
        if (q4nx_elf_arg_sizes(raw.data(), raw.size(), declared) < 0) return false;
        // The .dynsym symbols are PER CORE (the driver patches each core's own object): the gemv
        // family's arg0 and arg2 are 1/cores of the stream and of the output, while arg1 (the whole
        // activation row) is not split.  Measured on these five ELFs 2026-11-02: QKV K=1024 N=1536
        // declares 245,760 / 2,048 / 1,536 (= w/4, 2K, N) and the head 97,320,960/4 / 2,048 / 152,064.
        // So the core count is inferred from arg0 and the other two are checked against it, rather
        // than assuming four.
        return q4nx_gemv_family(declared, KD, ND) != 0;
    }

    bool args_fit_elf(const char* ep) {
        std::vector<uint8_t> raw;
        {
            std::FILE* f = std::fopen(ep, "rb");
            if (!f) return true;                       // cannot read it: nothing to check against
            std::fseek(f, 0, SEEK_END);
            const long len = std::ftell(f);
            std::rewind(f);
            if (len <= 0) { std::fclose(f); return true; }
            raw.resize((size_t)len);
            const size_t got = std::fread(raw.data(), 1, raw.size(), f);
            std::fclose(f);
            if (got != raw.size()) return true;
        }
        uint64_t declared[3] = {0, 0, 0};
        if (q4nx_elf_arg_sizes(raw.data(), raw.size(), declared) < 0) return true;   // not our shape
        // passed[] is what this launcher hands the kernel for the family in use: the gemv family takes
        // (weights, activations, outputs) in that order, everything else (activations, weights, outputs).
        uint64_t passed[3] = {
            gemv ? (uint64_t)w_bytes() : (uint64_t)MD * (uint64_t)KD,
            gemv ? (uint64_t)KD * 2u : (uint64_t)w_bytes(),
            gemv ? (uint64_t)ND * 4u
                 : (uint64_t)MD * (uint64_t)ND * 4u * (pb16i ? 2u : 1u)};
        // I8Ctx carries no error string; the caller's err_ names the ELF, so print the numbers here.
        static const char* const kWhat[3] = {"activations", "weights", "output"};
        for (int i = 0; i < 3; i++) {
            if (!declared[i] || declared[i] <= passed[i]) continue;
            std::fprintf(stderr,
                         "  I8Ctx::init_elf: %s declares argument %d (%s) = %llu B, this launch would "
                         "pass %llu B -- the kernel's DMA would run past the buffer (M=%d K=%d N=%d). "
                         "The artifact's argument width is its contract and this ELF family is not the "
                         "one this launcher drives; refusing before any dispatch.\n",
                         ep, i, kWhat[i], (unsigned long long)declared[i],
                         (unsigned long long)passed[i], MD, KD, ND);
            return false;
        }
        return true;
    }

    // ── The gemv weight image: the container's own q4_1 tile stream, copied, never repacked ──
    // The kernel dequantises with each tile's own bf16 scale/zero, so passing the container's bytes
    // unchanged is both the cheapest thing to do and the only lossless one (packB's int8 image would
    // throw those parameters away).  blen is the tensor's byte count; the rest of the BO is left as
    // the zero tiles the row pad needs.
    bool upload_tiles_into(xrt::bo& bo, const uint8_t* blob, size_t blen) {
        if (!gemv) return false;
        if (blen > bo.size()) {
            std::fprintf(stderr, "  I8Ctx::upload_tiles_into: %zu B of tiles does not fit a %llu B BO\n",
                         blen, (unsigned long long)bo.size());
            return false;
        }
        uint8_t* p = (uint8_t*)bo.map();
        std::memset(p, 0, bo.size());
        if (blob && blen) std::memcpy(p, blob, blen);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return true;
    }

    // ── Resident-expert (MoE) helpers: pack/launch against an arbitrary BO ──
    // Decode is M=1 with top-1 routing; re-streaming the selected expert's
    // weights into a shared per-layer BO every token costs a memcpy + sync on
    // the critical path (~30ms/tok for 20 layers). Instead allocate one
    // weight BO per (layer, expert) at startup, pack+sync once, and pass the
    // BO handle directly at decode.
    // `zero` is false only where the whole BO is about to be overwritten by an image read
    // (load_pb16i_into zeroes whatever tail it does not write itself), because for the
    // pb16i/t2s containers the memset was 6.8 GB of writes per serve that nothing read.
    std::unique_ptr<xrt::bo> make_weight_bo(xrt::device& d, bool zero = true) {
        // The amdxdna driver rejects a group-less xrt::bo, so use the same
        // xrt::ext::bo allocation init_elf's mkbo uses.
        xrt::ext::bo b{d, w_bytes()};
        auto bo = std::make_unique<xrt::bo>(b);
        if (zero) if (void* m = bo->map()) memset(m, 0, w_bytes());
        return bo;
    }

    // ── pb16i / t2s: the weight image is the sidecar file, copied unchanged ──
    // `<dir>/<stem>.pb16i` (or `.t2s`) is the image and `<dir>/<stem>.ref` the side table
    // (e_ref int16[N] + mode byte, the same file for both layouts).  Nothing is requantised, so
    // a mismatch is a wrong file, not a rounding difference: every check here is loud.
    bool load_pb16i_into(xrt::bo& bo, const std::string& stem,
                         std::vector<int16_t>& eref, int& mode) {
        const char* const tag = t2s ? "t2s" : "pb16i";
        const size_t want = t2s ? q4nx_t2s::w_bytes(KD, ND) : q4nx_pb16i::w_bytes(KD, ND);
        const std::string imgp = stem + (t2s ? ".t2s" : ".pb16i"), refp = stem + ".ref";
        // The image IS the BO's contents, so read it straight into the mapping: the old path
        // zero-filled a `want`-byte vector, then memcpy'd it in, which is four passes over the
        // whole 6.8 GB of decode weights per serve (measured with NPU_PB16I_TIMING=1; see the
        // 2026-11-02 cold-start entry).  NPU_FILL_COPY=1 keeps the copying path for an A/B.
        if (bo.size() < want) {   // before the read: the direct path reads into the mapping
            fprintf(stderr, "  %s: weight BO is %zu B, needs %zu\n", tag, (size_t)bo.size(), want);
            return false;
        }
        const bool direct = getenv("NPU_FILL_COPY") == nullptr;
        std::vector<uint8_t> img(direct ? 0 : want);
        uint8_t* buf = direct ? (uint8_t*)bo.map() : img.data();
        if (!buf) { fprintf(stderr, "  %s: cannot map the weight BO\n", tag); return false; }
        const auto f0 = std::chrono::steady_clock::now();
        FILE* f = fopen(imgp.c_str(), "rb");
        if (!f) { fprintf(stderr, "  %s: cannot open %s\n", tag, imgp.c_str()); return false; }
        const size_t got = fread(buf, 1, want, f);
        const int extra = fgetc(f);
        fclose(f);
        const auto f1 = std::chrono::steady_clock::now();
        if (got != want || extra != EOF) {
            fprintf(stderr, "  %s: %s is not %zu bytes (K=%d N=%d)\n", tag, imgp.c_str(), want,
                    KD, ND);
            return false;
        }
        f = fopen(refp.c_str(), "rb");
        if (!f) { fprintf(stderr, "  %s: cannot open %s\n", tag, refp.c_str()); return false; }
        std::vector<uint8_t> ref(q4nx_pb16i::ref_bytes(ND) + 1);
        const size_t rgot = fread(ref.data(), 1, ref.size(), f);
        fclose(f);
        if (rgot != q4nx_pb16i::ref_bytes(ND) ||
            !q4nx_pb16i::parse_ref(ref.data(), rgot, ND, &eref, &mode)) {
            fprintf(stderr, "  %s: %s is not e_ref[%d] + one mode byte\n", tag, refp.c_str(), ND);
            return false;
        }
        if (direct) {
            if (bo.size() > want)   // the padded tail the kernel never writes: keep it zero
                memset((uint8_t*)bo.map() + want, 0, (size_t)bo.size() - want);
        } else {
            memcpy(bo.map(), img.data(), want);
        }
        const auto f2 = std::chrono::steady_clock::now();
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const auto f3 = std::chrono::steady_clock::now();
        if (getenv("NPU_PB16I_TIMING")) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            fprintf(stderr, "[fill] %-8s %s bytes=%zu read=%.2f copy=%.2f sync=%.2f ms\n",
                    direct ? "direct" : "copy", stem.c_str(), want, ms(f0, f1), ms(f1, f2),
                    ms(f2, f3));
        }
        // NPU_PB16I_HASH=1: one line per image actually loaded, so a run can answer "is this
        // tensor's image non-zero, and is it different from its neighbours" without a device
        // probe.  The stem carries (layer, tag), which is what a per-launch gate cannot see.
        if (getenv("NPU_PB16I_HASH")) {
            unsigned long long h = 1469598103934665603ULL;
            const size_t n = want < 4096 ? want : 4096;
            for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ULL; }
            unsigned long long nz = 0;
            for (size_t i = 0; i < n; i++) nz += (buf[i] != 0);
            fprintf(stderr, "[%s-image] %s mode=%d planes=%d first4k_fnv=%016llx nonzero=%llu/%zu\n",
                    tag, stem.c_str(), mode, pb16i_planes, h, nz, n);
        }
        return true;
    }

    // C = float32(S * 2^e_ref) * ascale, the only rounding in the chain.  Row 0 is
    // the decode row; am rows are written, each row being an int32 plane or the
    // pair [lo | hi] of the int64 mode, exactly as the kernel wrote it, and each
    // row is dequantised with its own activation scale.
    void dequant_pb16i(float* out, int am, int an, const float* ascales,
                       const std::vector<int16_t>& eref, int planes) {
        const int row = planes > 1 ? 2 * ND : ND;
        for (int m = 0; m < am; m++) {
            const int32_t* lo = Cm + (size_t)m * (size_t)row;
            const float asc = ascales[m];
            for (int n = 0; n < an; n++) {
                const int64_t s = q4nx_pb16i::sum_of(lo + n, lo + ND + n, planes > 1 ? 1 : 0);
                out[(size_t)m * (size_t)an + n] =
                    q4nx_pb16i::scale_i64(s, eref[(size_t)n]) * asc;
            }
        }
    }

    // Cache-friendly per-column int8 quantization into a [KD][ND] padded buffer.
    // The original column-at-a-time loop reads w[i*N + j] with a stride of N
    // floats -- 2 KB at N=512 -- so nearly every element is its own cache line;
    // the pack measured ~11 ms per routed-expert call (K*N = 1M). Walk row-major
    // in blocks of CB adjacent columns instead (each block's CB floats are
    // contiguous, so a row costs ~CB/16 lines, not CB) and parallelize over the
    // disjoint column blocks. Same amax, per-column scale, rounding and clamps:
    // the packed bytes and scales are bit-identical; only the order of the
    // diagnostic `ssum` accumulation differs.
    // Where element (i,j) of the padded [KD][ND] weight image lives in the BO.
    // The mapping (and why the blocked layout is worth 2.4-2.6x) is documented in
    // q4nx_weight_layout.h, which is also what tests/npu_blocked_layout_test.cpp
    // checks; this is just the layout switch.
    inline size_t bpos(size_t i, size_t j) const {
        return blocked_b ? q4nx_layout::blocked_index(i, j, KD, ND)
                         : q4nx_layout::rowmajor_index(i, j, KD, ND);
    }

    void packB_blocked(int8_t* Bm, const float* w, int K, int N,
                       std::vector<float>& col, double& ssum) {
        const int CB = 64;
        #pragma omp parallel for reduction(+:ssum) schedule(static)
        for (int jb = 0; jb < N; jb += CB) {
            const int jn = (N - jb < CB) ? (N - jb) : CB;
            float amax[CB], tis[CB];
            for (int b = 0; b < jn; b++) amax[b] = 0.0f;
            for (int i = 0; i < K; i++) {
                const float* row = w + (size_t)i * N + jb;
                for (int b = 0; b < jn; b++) {
                    float a = fabsf(row[b]);
                    if (std::isfinite(a) && a > amax[b]) amax[b] = a;
                }
            }
            for (int b = 0; b < jn; b++) {
                float am = amax[b];
                if (am < 1e-12f) am = 1.0f;
                col[jb + b] = am / 127.0f;
                tis[b] = 127.0f / am;
                ssum += (double)(am / 127.0f);
            }
            for (int i = 0; i < K; i++) {
                const float* row = w + (size_t)i * N + jb;
                for (int b = 0; b < jn; b++) {
                    float v = row[b];
                    if (!std::isfinite(v)) v = 0.0f;
                    int x = (int)roundf(v * tis[b]);
                    if (x > 127) x = 127;
                    else if (x < -127) x = -127;
                    Bm[bpos((size_t)i, (size_t)jb + b)] = (int8_t)x;
                }
            }
        }
    }

    // Pack weights into an arbitrary (already-allocated) weight BO.
    void packB_into(xrt::bo& bo, const float* w, int K, int N,
                    float& sout, std::vector<float>& col_out) {
        auto* Bm = (int8_t*)bo.map();
        memset(Bm, 0, (size_t)KD * ND);
        std::vector<float> col(N);
        double ssum = 0;
        packB_blocked(Bm, w, K, N, col, ssum);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        col_out = std::move(col);
        sout = (float)(ssum / N);
    }

    // Async launch with an arbitrary weight BO (resident-expert path).
    inline xrt::run launch_async_with_bo(xrt::bo& wbo, const float* A,
                                         int am, int ak, float ascale) {
        quantize_async(A, am, ak, ascale);
        sync_a_to_device();
        if (!gemv) poison_canary(ND, am);   // engine#666: so readback() can tell "not written" from "written as 0"
        if (gemv) return (*kext)(wbo, *bA, *bC);
        return (*kext)(*bA, wbo, *bC);
    }

    // Batched variant of launch_async_with_bo: one activation quantization with
    // PER-ROW scales, so each row of a prompt keeps its own dynamic range (a
    // single shared scale zeroes low-magnitude rows when one token's activations
    // dwarf the rest -- issue #1699). Dequantize with the same per-row scales.
    // engine#1416 [ps]: the prefill `pack` column was 24 ms/launch on a build whose profiler is now
    // gone, and a device-free bench of the row quantiser alone says 0.203 ms for 128x4864 -- so ~99 %
    // of that figure sits in one of the other three terms below. This splits them, no behaviour change.
    inline xrt::run launch_async_with_bo_rows(xrt::bo& wbo, const float* A,
                                              int am, int ak,
                                              const float* ascales) {
        static const bool split = std::getenv("NPU_PACK_SPLIT") != nullptr;
        if (!split) {
            quantize_async_rows(A, am, ak, ascales);
            sync_a_to_device();
            poison_canary(ND, am);
            return (*kext)(*bA, wbo, *bC);
        }
        const auto t0 = pack_clock::now();
        quantize_async_rows(A, am, ak, ascales);
        const auto t1 = pack_clock::now();
        sync_a_to_device();
        const auto t2 = pack_clock::now();
        poison_canary(ND, am);
        const auto t3 = pack_clock::now();
        xrt::run r = (*kext)(*bA, wbo, *bC);
        const auto t4 = pack_clock::now();
        {
            using namespace std::chrono;
            const double q  = duration<double, std::milli>(t1 - t0).count();
            const double sy = duration<double, std::milli>(t2 - t1).count();
            const double cn = duration<double, std::milli>(t3 - t2).count();
            const double lc = duration<double, std::milli>(t4 - t3).count();
            pack_split_q_ += q; pack_split_sync_ += sy; pack_split_canary_ += cn;
            pack_split_launch_ += lc; pack_split_n_++;
            if (pack_split_n_ % 100 == 0)
                fprintf(stderr, "[ps] n=%ld M=%d K=%d per-launch: quant=%.3f sync=%.3f "
                                "canary=%.3f launch=%.3f ms\n", pack_split_n_, am, ak,
                        pack_split_q_ / pack_split_n_, pack_split_sync_ / pack_split_n_,
                        pack_split_canary_ / pack_split_n_, pack_split_launch_ / pack_split_n_);
        }
        return r;
    }

    // ── Pack weights for layer l into contiguous BO ──
    // K×N are the logical (unpadded) weight dims; the BO is KD×ND (padded to 128).
    // Zero-init ensures padded regions contribute zero to the GEMM output.
    void packB(int l, const float* w, int K, int N, float& sout) {
        // Weight-content checksum (NPU_DBG=1), placed INSIDE the implementation so it fires for
        // whichever context FLM_PACKB selects. My previous two attempts sat at call sites and never
        // fired, because Nanbeige packs through a branch I misread. This checks the DEQUANTIZED
        // weights -- the last host input not yet proven stable across runs.
        // RESULTS-coverage-multifamily 67.
        if (getenv("NPU_DBG") && l < 3) {
            unsigned long long hh = 1469598103934665603ULL;
            const unsigned char* pp = (const unsigned char*)w;
            for (size_t i = 0; i < (size_t)K * N * sizeof(float); i++) { hh ^= pp[i]; hh *= 1099511628211ULL; }
            fprintf(stderr, "[WCHK i8] l=%d K=%d N=%d fnv=%016llx\n", l, K, N, hh);
        }
        // Per-output-column weight scales: each column j is quantized with its
        // own amax_j/127 and dequantized with group_scales[l][j]. A single
        // per-tensor scale packed low-magnitude columns (Qwen3 v_proj rms
        // ~0.007 vs q/k ~0.02-0.03) onto ~10 int8 levels -> ~5% output error
        // that compounds over 28 layers and flips the final token.
        auto* Bm = (int8_t*)layerB[l]->map();
        memset(Bm, 0, (size_t)KD * ND);
        std::vector<float> col(N);
        double ssum = 0;
        packB_blocked(Bm, w, K, N, col, ssum);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        group_scales[l] = std::move(col);
        sout = (float)(ssum / N);
    }

    // ── Quantize activations into bA ──
    // bf16 of one activation row, round-to-nearest-even: the gemv family takes bf16 in, and the kernel
    // does its own quantisation, so there is no scale for the host to choose.
    static inline uint16_t f32_to_bf16_bits(float v) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
        return (uint16_t)(u >> 16);
    }
    inline uint16_t* quantize_bf16(const float* A, int am, int ak) {
        if (am != 1) { std::fprintf(stderr, "  I8Ctx: the gemv family takes one row per launch (am=%d)\n", am); return Ag; }
        last_am_ = am;
        a_dirty(1);
        const int n = ak < KD ? ak : KD;
        for (int k = 0; k < n; k++) {
            float v = A[k];
            if (!std::isfinite(v)) v = 0.0f;
            Ag[k] = f32_to_bf16_bits(v);
        }
        for (int k = n; k < KD; k++) Ag[k] = 0;   // K pad columns read as 0, like the container's
        return Ag;
    }

    // ── Activation BO bookkeeping (target 2) ──
    // The M=128 stream reads rows [am, MD) on every launch, so those rows must be zero -- but they only need
    // to become zero again when a previous launch wrote real data there. On the decode path am is always 1,
    // so tracking which rows the host has written turns a per-launch memset of the whole BO into a one-off.
    // Bytes left in the BO are identical either way: rows [0, am) are written by the quantiser, rows
    // [am, a_written_rows_) are zeroed here, the rest are already zero.
    //
    // a_written_rows_ is set to `am` (not left at its old value) after every call: once rows [am, ...) have been
    // zeroed and rows [0, am) quantised, the only rows that may hold host data are [0, am). Leaving it at MD was
    // the bug in the first version of this change -- it re-zeroed 127 of 128 rows on every decode launch, i.e.
    // 99.2 % of the memset it was written to remove, which the target-2 gate measured as no change at all.
    //
    // -1, not MD: MD is declared without an initialiser (`int MD, KD, ND, NL` above) and is only assigned in
    // init_elf(), while a non-static member initialiser here runs at construction. Contexts are built with
    // std::make_unique<I8Ctx>() and given their geometry afterwards (q4nx_forward.cpp:931, onnx/onebit_npu_gemm.cpp:205);
    // make_unique value-initialises, so the zero-fill sets MD to 0 before this initialiser runs -- measured, not
    // assumed: `Ctx{int MD; int w = MD;}` printed `MD=0 w=0` through make_unique and through placement-new on 0xAA.
    // So `= MD` starts at 0, "no rows known", and the first ak == KD launch zeroes nothing, leaving the pad rows
    // to whatever the BO was allocated with. That is the defect this line closes: -1 means "nothing known yet",
    // which costs one whole-BO zero on the first launch and cannot depend on the allocation or on member order.
    int a_written_rows_ = -1;   // rows [0, a_written_rows_) may hold host data; -1 = nothing known yet

    inline void prepare_activation(int am, int ak) {
        if (ak != KD || a_written_rows_ < 0) {
            // (a) a K shorter than the padded geometry leaves a column tail in rows [0, am) that the quantiser does
            //     not write, and (b) before the first launch nobody knows what the BO holds: both fall back to the
            //     whole-BO zero, exactly as the code did unconditionally before this change
            memset(Am, 0, (size_t)MD * KD);
            a_written_rows_ = am;
            a_dirty(MD);   // the whole BO is dirty after this zero
            return;
        }
        if (a_written_rows_ > am) {
            memset(Am + (size_t)am * KD, 0, (size_t)(a_written_rows_ - am) * KD);
            a_dirty(a_written_rows_);   // the rows this zeroed
        }
        a_dirty(am);                    // what the caller's quantiser loop writes next
        a_written_rows_ = am;
    }

    // ── Target 2: are the BO syncs required at all? ──
    // An XDNA2 BO is cacheable host memory, so bo::sync() is `mfence; clflush ...; mfence` over the range
    // (disassembled 2026-10-31) -- cache maintenance, not a copy. The brief's question is *when the XDNA
    // cache model needs it*, and it can only be answered by running the same prompts with the syncs off and
    // comparing logits. These switches exist for exactly that arm; both default to on (the shipped path is
    // untouched) and both are only ever set inside an npu-hold window.
    //   NPU_NO_A_FLUSH=1   skip bA's writeback before the launch (host wrote A, the array reads it)
    //   NPU_NO_C_INVAL=1   skip bC's invalidate before the host reads it (the array wrote C)
    static bool env_off(const char* name) {
        const char* v = getenv(name);
        return v != nullptr && atoi(v) != 0;
    }
    inline void sync_a_to_device() {
        static const bool skip = env_off("NPU_NO_A_FLUSH");
        static const bool whole = getenv("NPU_NO_A_NARROW") != nullptr || getenv("NPU_DBG") != nullptr;
        if (skip) return;
        const auto t0 = a_sync_prof() ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
        if (whole) bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        else bA->sync(XCL_BO_SYNC_BO_TO_DEVICE, a_host_bytes(), 0);
        a_sync_note(t0, whole ? bA->size() : a_host_bytes());
        a_dirty_rows_ = 0;   // the rows the host had touched are in memory now
    }

    // Bytes of bA the host has written since the last writeback: rows [0, a_dirty_rows_), each KD int8 (one
    // KD-wide bf16 row in the gemv family). bo::sync() is a per-64-B clflush loop over its range -- the same
    // instrument that measured the C syncs -- so its cost IS the range, and on the decode path the range is
    // one row instead of MD. NPU_NO_A_NARROW=1 restores the whole-BO range for an A/B, NPU_DBG forces it
    // because bochk checksums the full extent.
    inline size_t a_host_bytes() const {
        const size_t row = gemv ? (size_t)KD * 2u : (size_t)KD;
        const size_t bytes = row * (size_t)(a_dirty_rows_ > 0 ? a_dirty_rows_ : 1);
        return bytes < bA->size() ? bytes : bA->size();
    }

    inline int8_t* quantize_async(const float* A, int am, int ak, float ascale) {
        last_am_ = am;
        if (gemv) { quantize_bf16(A, am, ak); return Am; }
        float ais = 1.0f / ascale;
        prepare_activation(am, ak);
        for (int m = 0; m < am; m++)
            q4nx_quant_i8::row(A + (size_t)m * ak, ak, ais, Am + (size_t)m * KD);
        return Am;
    }

    // Per-row activation scales (batched MoE prefill): row m quantized with
    // 1/ascales[m], so each token keeps its own dynamic range. Dequant must
    // use the matching per-row scale (dequant_only_rows).
    inline int8_t* quantize_async_rows(const float* A, int am, int ak,
                                       const float* ascales) {
        // bA holds exactly MD rows. The i8 ("fallback") prefill passes am == npt with no cap
        // (the bf16 path caps at 'cap'; this one does not), so any prompt longer than MD wrote
        // PAST the end of bA -- and the kernel, launched for MD rows, never processed rows
        // MD..npt-1 at all. The boot token is taken from h_b[npt-1], which is therefore still the
        // RAW EMBEDDING, never passed through the layers: a context-free prediction, which is why
        // this path returns small scattered tokens (12-19 for 0.6B, 151 for Nanbeige) instead of
        // the reference. The overrun is also the second, independent source of nondeterminism.
        // RESULTS-coverage-multifamily 82. Refuse loudly instead of corrupting memory silently.
        if (am > MD) {
            fprintf(stderr,
                    "quantize_async_rows: am=%d exceeds bA capacity MD=%d -- refusing to write "
                    "past the activation BO (see RESULTS-coverage-multifamily 82: the prefill "
                    "must walk the prompt in MD-row blocks)\n", am, MD);
            return Am;
        }
        last_am_ = am;
        prepare_activation(am, ak);
        for (int m = 0; m < am; m++)
            q4nx_quant_i8::row(A + (size_t)m * ak, ak, 1.0f / ascales[m], Am + (size_t)m * KD);
        return Am;
    }


    inline void sync_A(int /*l*/) { sync_a_to_device(); }

    // ── Launch kernel for layer l ──
    // Kernel signature: (opcode, instr_bo, ninstr, bo0, bo1, bo2, bo3, bo4)
    // BOCHK: checksum the three BOs this kernel reads, immediately before the launch. Two independent
    // facts pin this as the live path -- I8Ctx::packB is what fires (so cq is the selected context),
    // and the banner reports GU_split=1 (so the single-launch fused FFN is not used). Section 68
    // exonerated every host INPUT, so whichever of these three differs across runs is a buffer the
    // kernel reads without anyone writing it.
    inline void bochk(int l) {
        if (!getenv("NPU_DBG")) return;
        // Layers 0-1 and the LAST four. The first two proved every pre-launch byte identical across runs
        // (8/8 checksums) while the boot token varied, so the divergence begins somewhere later -- and
        // covering the tail is what localizes it. RESULTS-coverage-multifamily 68.
        if (!(l <= 1 || l >= (int)layerB.size() - 4)) return;
        static int n = 0;
        if (n >= 40) return;
        n++;
        fprintf(stderr, "[BOCHK] l=%d bA=%016llx W=%016llx bC=%016llx\n",
                l, bo_fnv(*bA), bo_fnv(*layerB[l]), bo_fnv(*bC));
    }
    inline xrt::run launch(int l) {
        bochk(l);
        return (*kext)(*bA, *layerB[l], *bC);
    }

    inline xrt::run sync_and_launch(int l) {
        sync_a_to_device();
        if (!gemv) poison_canary(ND);   // the canary is an int32 sentinel; the gemv output is f32
        bochk(l);
        // The gemv family's argument order is (weights, activations, outputs); every int8 family
        // this launcher was written for takes (activations, weights, outputs).
        if (gemv) return (*kext)(*layerB[l], *bA, *bC);
        return (*kext)(*bA, *layerB[l], *bC);
    }

    inline void wait_kernel(xrt::run& r) { r.wait(); }

    // Per-section output scales for the fused QKV GEMM (fix #1699: llama
    // v_proj rms ~0.007 vs q/k ~0.02-0.03 — a single weight scale packs the
    // small v section onto ~10 int8 levels, ~5% output error that compounds
    // over 32 layers and flips the final token). When sec_scales is set
    // (size 3: [ts_q, ts_k, ts_v]), dequant_qkv_rows applies each section's
    // scale; sec_n0/sec_n1 are the q/k output lengths.
    std::vector<std::vector<float>> sec_scales;  // per-layer [ts_q, ts_k, ts_v]
    int sec_n0 = 0, sec_n1 = 0;

    inline bool pack_qkv_sec(int l, const float* w, int K, int N,
                             int nq, int nk, std::vector<float>& out_scales) {
        auto* Bm = (int8_t*)layerB[l]->map();
        memset(Bm, 0, (size_t)KD * ND);
        auto pack_sec = [&](int j0, int j1, float& ts) {
            float amax = 0;
            for (int j = j0; j < j1; j++)
                for (int i = 0; i < K; i++) {
                    float a = fabsf(w[(size_t)i * N + j]);
                    if (std::isfinite(a) && a > amax) amax = a;
                }
            if (amax < 1e-12f) amax = 1.0f;
            ts = amax / 127.0f;
            float tis = 127.0f / amax;
            for (int j = j0; j < j1; j++)
                for (int i = 0; i < K; i++) {
                    float v = w[(size_t)i * N + j];
                    if (!std::isfinite(v)) v = 0;
                    int x = (int)roundf(v * tis);
                    if (x > 127) x = 127;
                    else if (x < -127) x = -127;
                    Bm[(size_t)i * ND + j] = (int8_t)x;
                }
        };
        float tsq = 0, tsk = 0, tsv = 0;
        pack_sec(0, nq, tsq);
        pack_sec(nq, nq + nk, tsk);
        pack_sec(nq + nk, N, tsv);
        layerB[l]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        out_scales = { tsq, tsk, tsv };
        return true;
    }

    inline void dequant_qkv_rows(xrt::run& r, float* C, int am, int an,
                                 const float* ascales, int layer = -1) {
        r.wait();
        readback();
        if (layer >= 0 && (size_t)layer < sec_scales.size() && sec_scales[layer].size() == 3 && sec_n0 + sec_n1 < an) {
            const std::vector<float>& ss = sec_scales[layer];
            for (int m = 0; m < am; m++) {
                float cs = ascales[m];
                const int32_t* src = Cm + (size_t)m * ND;
                float* dst = C + (size_t)m * an;
                for (int n = 0; n < sec_n0; n++) dst[n] = (float)src[n] * (cs * ss[0]);
                for (int n = 0; n < sec_n1; n++) dst[sec_n0 + n] = (float)src[sec_n0 + n] * (cs * ss[1]);
                for (int n = sec_n0 + sec_n1; n < an; n++) dst[n] = (float)src[n] * (cs * ss[2]);
            }
        } else {
            dequant_only_rows(C, am, an, ascales, 0, layer);
        }
    }

    // ── Readback + dequantize output ──
    // readback() is the ONE point where the host takes the kernel's result. Section 68 and the BOCHK
    // sweep localized the divergence exactly here: across runs the weight BO is ALWAYS identical, bC
    // is identical immediately before each launch, and yet the dequantized bA -- the host's reading of
    // that same bC -- differs. So the kernel's output buffer is the same and what the host gets back
    // from it is not. Checksumming right AFTER the sync isolates the transfer.
    //
    // engine#666: that sync IS the invalidate the issue asks for, and it is not enough. bo::sync() is
    // `mfence; clflush x N; mfence` over the range (disassembled from libonebit_npu_rt.a, 2026-10-31), so
    // the host's cache lines for bC are dropped before every read -- a stale line cannot survive it. A
    // read that still comes back short therefore means the writes are NOT in memory yet (or never
    // landed), not that the host read under them. The mitigation is the canary below: poison the row we
    // read before the launch, and if that row is still poisoned after the invalidate, re-issue the launch
    // once instead of dequantising garbage into a layer.
    // Bytes of C the host actually reads: rows [0, last_am_), each ND int32 (two planes in the int64
    // mode). bo::sync() is a per-64-B clflush loop over its range -- disassembled from
    // libonebit_npu_rt.a -- so its cost IS the range, and the device writes C through the IOMMU into
    // memory: only lines the host touched can be stale, and the host touches rows [0, am).  A decode
    // launch therefore invalidates its one row (ND x 4 B, i.e. up to 128x less than the MD-row BO).
    // NPU_NO_C_NARROW=1 restores the whole-BO range for an A/B, and NPU_DBG forces it because
    // bochk/RBCHK checksum the full extent.
    inline size_t c_host_bytes() const {
        size_t row = (size_t)ND * 4u;
        if (!gemv && (pb16i || pb16i_planes > 1)) row *= 2u;
        const size_t bytes = row * (size_t)(last_am_ > 0 ? last_am_ : 1);
        return bytes < bC->size() ? bytes : bC->size();
    }
    inline bool c_sync_whole() const {
        return getenv("NPU_NO_C_NARROW") != nullptr || getenv("NPU_DBG") != nullptr;
    }

    // ── Optional instrument for the narrow-vs-whole A/B (NPU_C_SYNC_PROF=1) ──
    // bo::sync() is a clflush loop over the range it is given, so the bytes are exact and the ms is the
    // cost of flushing them. Process-global on purpose: every context's flushes land in the same process,
    // and the printer is per token, so consecutive lines give that token's flushes. Off unless asked.
    static bool& c_sync_prof() { static bool v = getenv("NPU_C_SYNC_PROF") != nullptr; return v; }
    static double& c_sync_ms() { static double v = 0; return v; }
    static long& c_sync_calls() { static long v = 0; return v; }
    static size_t& c_sync_bytes() { static size_t v = 0; return v; }
    static void c_sync_note(const std::chrono::steady_clock::time_point& t0, size_t bytes) {
        c_sync_ms() += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
        c_sync_calls()++;
        c_sync_bytes() += bytes;
    }
    static void c_sync_print(int token) {
        fprintf(stderr, "[csync] tok %d calls=%ld bytes=%zu mb=%.2f ms=%.2f\n", token, c_sync_calls(),
                c_sync_bytes(), (double)c_sync_bytes() / 1e6, c_sync_ms());
    }

    // ── The same instrument for the A writeback (audit row 17, NPU_A_SYNC_PROF=1) ──
    static bool& a_sync_prof() { static bool v = getenv("NPU_A_SYNC_PROF") != nullptr; return v; }
    static double& a_sync_ms() { static double v = 0; return v; }
    static long& a_sync_calls() { static long v = 0; return v; }
    static size_t& a_sync_bytes() { static size_t v = 0; return v; }
    static void a_sync_note(const std::chrono::steady_clock::time_point& t0, size_t bytes) {
        if (!a_sync_prof()) return;
        a_sync_ms() += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
        a_sync_calls()++;
        a_sync_bytes() += bytes;
    }
    static void a_sync_print(int token) {
        fprintf(stderr, "[async] tok %d calls=%ld bytes=%zu mb=%.2f ms=%.2f\n", token, a_sync_calls(),
                a_sync_bytes(), (double)a_sync_bytes() / 1e6, a_sync_ms());
    }

    inline void readback() {
        static const bool skip_inval = env_off("NPU_NO_C_INVAL");
        const bool whole = c_sync_whole();
        if (!skip_inval) {
            const auto cs0 = c_sync_prof() ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
            if (whole) bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            else bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_host_bytes(), 0);
            if (c_sync_prof()) c_sync_note(cs0, whole ? bC->size() : c_host_bytes());
        }
        if (canary_on_ && !gemv) verify_canary();
        if (getenv("NPU_DBG")) {
            static int n = 0;
            if (n < 8) { n++; fprintf(stderr, "[RBCHK] bC=%016llx\n", bo_fnv(*bC)); }
        }
    }

    // ── engine#666 canary: is the row we are about to dequantise actually written? ──
    // One poison store per element of row 0 (the only row the decode path reads) before the launch, and a
    // compare of the same row after the invalidate. A genuinely short tile leaves thousands of sentinels
    // standing; an accumulator that happens to equal the sentinel is one element, so a threshold of 8 keeps
    // false positives out while any real short tile trips it. Cost is two passes over N int32 (<= 68 KiB)
    // per launch, measured below; ONEBIT_NPU_CANARY=0 turns it off, NPU_CANARY_STATS=1 reports it.
    static constexpr int32_t kCanary = (int32_t)0x80000000;  // INT32_MIN: no accumulator of ours lands there by rounding
    int canary_on_ = []() { const char* v = getenv("ONEBIT_NPU_CANARY"); return v == nullptr || atoi(v) != 0; }();
    int short_row_ = 0;      // sentinels still standing in row 0 after the last invalidate
    long canary_launches_ = 0;
    long canary_retries_ = 0;
    long canary_short_launches_ = 0;

    // one sentinel every kCanaryStride elements: a dropped tile is hundreds of elements wide, so sampling
    // catches it while the poison+scan stays a few KiB per launch instead of the whole 68 KiB row
    static constexpr int kCanaryStride = 16;

    // Row 0 only: it is the row every decode GEMM and every head tile reads back, and it is where this issue's
    // own failures land (first_idx=832 and 9344 against N=17408). A batched (prefill) launch is NOT canaried
    // beyond row 0: measured 2026-10-31 in the window, poisoning row am-1 there fires on every prefill launch
    // (short=59..1304 survivors) because the kernel does not write the padded last row of the activation BO --
    // a false positive that also doubles prefill work through the retry. Prefill numerics are NPU_M16_CHECK's job.
    inline void poison_canary(int an, int /*am*/ = 1) {
        if (!canary_on_ || Cm == nullptr) return;
        const auto cs0 = c_sync_prof() ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        const int n = std::min(an, ND);
        for (int i = 0; i < n; i += kCanaryStride) Cm[i] = kCanary;
        // Write the sentinels back NOW. Without this the poisoned lines sit dirty in the host cache and are
        // evicted after the kernel has already written the same memory, so the eviction overwrites the device's
        // result with the sentinel -- the canary then "detects" a short read it caused itself (measured: 1,416
        // false positives on the shipping M=128 decode path, one per launch group, before this line existed).
        if (c_sync_whole()) bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        else bC->sync(XCL_BO_SYNC_BO_TO_DEVICE, c_host_bytes(), 0);
        if (c_sync_prof())
            c_sync_note(cs0, c_sync_whole() ? bC->size() : c_host_bytes());
    }

    inline void verify_canary() {
        int bad = 0;
        for (int i = 0; i < ND; i += kCanaryStride) bad += (Cm[i] == kCanary);
        short_row_ = bad > 8 ? bad : 0;
        canary_launches_++;
        if (short_row_) canary_short_launches_++;
    }

    inline bool canary_short() const { return short_row_ != 0; }

    inline void canary_report(const char* tag, int l) const {
        if (canary_short_launches_ == 0 && !getenv("NPU_CANARY_STATS")) return;
        fprintf(stderr, "[canary] %s l=%d launches=%ld short=%ld retries=%ld last_short=%d\n", tag, l,
                canary_launches_, canary_short_launches_, canary_retries_, short_row_);
    }

    inline void dequant_only(float* C, int am, int an, float ascale,
                             float Bscale, int layer = -1) {
        if (gemv) {   // no scales left to apply: the kernel returns N f32 outputs for the one row
            for (int n = 0; n < an && n < ND; n++) C[n] = Cg[n];
            return;
        }
        const float* gs = nullptr;
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            (int)group_scales[layer].size() == an)
            gs = group_scales[layer].data();
        for (int m = 0; m < am; m++)
            for (int n = 0; n < an; n++) {
                float cs = ascale * (gs ? gs[n] : Bscale);
                float val = (float)((int32_t)Cm[m * ND + n]) * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
    }

    // Per-row dequant (batched MoE prefill): row m scaled by ascales[m].
    inline void dequant_only_rows(float* C, int am, int an,
                                  const float* ascales, float Bscale,
                                  int layer = -1) {
        if (gemv) {   // the family is one row per launch, so a batched read never happens here
            for (int n = 0; n < an && n < ND; n++) C[n] = Cg[n];
            return;
        }
        const float* gs = nullptr;
        if (layer >= 0 && (size_t)layer < group_scales.size() &&
            (int)group_scales[layer].size() == an)
            gs = group_scales[layer].data();
        for (int m = 0; m < am; m++) {
            for (int n = 0; n < an; n++) {
                float cs = ascales[m] * (gs ? gs[n] : Bscale);
                float val = (float)((int32_t)Cm[m * ND + n]) * cs;
                if (!std::isfinite(val)) val = 0;
                C[m * an + n] = val;
            }
        }
    }

    inline void dequantize(xrt::run& r, float* C, int am, int an,
                           float ascale, float Bscale, int layer = -1) {
        r.wait();
        readback();
        dequant_only(C, am, an, ascale, Bscale, layer);
    }

    inline void sync_back_and_dequant(float* C, int am, int an,
                                      float ascale, float Bscale,
                                      int layer = -1) {
        readback();
        dequant_only(C, am, an, ascale, Bscale, layer);
    }

    // ── Synchronous go() ──
    inline bool go(int l, const float* A, int am, int ak, float ascale,
                   float Bscale, float* C, int an) {
        auto t0 = std::chrono::steady_clock::now();
        quantize_async(A, am, ak, ascale);
        auto t1 = std::chrono::steady_clock::now();
        auto r = sync_and_launch(l);
        auto t2 = std::chrono::steady_clock::now();
        r.wait();
        auto t3 = std::chrono::steady_clock::now();
        dequantize(r, C, am, an, ascale, Bscale, l);
        auto t4 = std::chrono::steady_clock::now();
        if (getenv("NPU_GO_STATS"))
            fprintf(stderr, "[go] q=%.2f sync+launch=%.2f wait=%.2f deq=%.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(),
                    std::chrono::duration<double, std::milli>(t3 - t2).count(),
                    std::chrono::duration<double, std::milli>(t4 - t3).count());
        return true;
    }

    // Synchronous go() with per-row activation scales (batched MoE prefill):
    // row m of A quantized with ascales_q[m], row m of C dequantized with
    // ascales_d[m] (GU: q==d; D: q=asu, d=asu*d_sc so per-token dequant
    // matches sequential's per-token expert-mean scale).
    inline bool go_rows(int l, const float* A, int am, int ak,
                        const float* ascales_q, const float* ascales_d,
                        float Bscale, float* C, int an) {
        auto t0 = std::chrono::steady_clock::now();
        quantize_async_rows(A, am, ak, ascales_q);
        auto t1 = std::chrono::steady_clock::now();
        auto r = sync_and_launch(l);
        auto t2 = std::chrono::steady_clock::now();
        r.wait();
        auto t3 = std::chrono::steady_clock::now();
        readback();
        dequant_only_rows(C, am, an, ascales_d, Bscale, l);
        auto t4 = std::chrono::steady_clock::now();
        if (getenv("NPU_GO_STATS"))
            fprintf(stderr, "[go_rows] q=%.2f sync+launch=%.2f wait=%.2f readback+deq=%.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t2 - t1).count(),
                    std::chrono::duration<double, std::milli>(t3 - t2).count(),
                    std::chrono::duration<double, std::milli>(t4 - t3).count());
        return true;
    }

    inline xrt::run launch_async(int l, const float* A, int am, int ak,
                                 float ascale) {
        // engine#1416: prefill's M=128 layer launches come through here, not through the
        // with_bo_rows variant, which is why the first [ps] window only ever saw M=1.
        static const bool split = std::getenv("NPU_PACK_SPLIT") != nullptr;
        if (!split) {
            quantize_async(A, am, ak, ascale);
            return sync_and_launch(l);
        }
        const auto t0 = pack_clock::now();
        quantize_async(A, am, ak, ascale);
        const auto t1 = pack_clock::now();
        xrt::run r = sync_and_launch(l);
        const auto t2 = pack_clock::now();
        using namespace std::chrono;
        pack_split_add(duration<double, std::milli>(t1 - t0).count(), 0.0, 0.0,
                       duration<double, std::milli>(t2 - t1).count(), am, ak);
        return r;
    }

    // Async launch with per-row activation scales (batched prefill fix,
    // #1699): row m quantized with ascales_q[m]. Dequant must use the same
    // per-row scales (finish_async_rows). Prevents the shared-batch ascale
    // from zeroing low-magnitude rows when one token's activations dwarf the
    // rest (Qwen3-0.6B: pos0 su max ~3671 vs pos1-3 max ~5 -> rows 1-3 were
    // quantized to all-zero int8 and the D GEMM emitted zeros).
    inline xrt::run launch_async_rows(int l, const float* A, int am, int ak,
                                      const float* ascales_q) {
        static const bool split = std::getenv("NPU_PACK_SPLIT") != nullptr;
        if (!split) {
            quantize_async_rows(A, am, ak, ascales_q);
            return sync_and_launch(l);
        }
        const auto t0 = pack_clock::now();
        quantize_async_rows(A, am, ak, ascales_q);
        const auto t1 = pack_clock::now();
        xrt::run r = sync_and_launch(l);
        const auto t2 = pack_clock::now();
        using namespace std::chrono;
        pack_split_add(duration<double, std::milli>(t1 - t0).count(), 0.0, 0.0,
                       duration<double, std::milli>(t2 - t1).count(), am, ak);
        return r;
    }

    inline void finish_async_rows(xrt::run& r, float* C, int am, int an,
                                  const float* ascales, float Bscale,
                                  int layer = -1) {
        r.wait();
        readback();
        dequant_only_rows(C, am, an, ascales, Bscale, layer);
    }

    inline void finish_async(xrt::run& r, float* C, int am, int an,
                             float ascale, float Bscale, int layer = -1) {
        r.wait();
        dequantize(r, C, am, an, ascale, Bscale, layer);
    }

    // FNV-1a over a BO's FULL extent. Section 68 exonerated every host INPUT (the dequantized weights
    // are byte-identical across runs while the boot token varies), so what remains is the kernel
    // reading a buffer nobody writes. Checking a prefix would miss exactly that, since the unwritten
    // regions in this code are the gs scale TAILS.
    static inline unsigned long long bo_fnv(xrt::bo& b) {
        const unsigned char* p = (const unsigned char*)b.map();
        size_t n = b.size();
        unsigned long long h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
        return h;
    }
};
