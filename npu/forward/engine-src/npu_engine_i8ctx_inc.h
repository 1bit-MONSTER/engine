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
// npu_engine_i8ctx_inc.h — I8Ctx GEMM context using xrt::kernel (classic API).
//
// Matches the actual xclbin kernel interface:
//   kernel(opcode, instr_bo, ninstr, bo0, bo1, bo2, bo3, bo4)
//
// One contiguous weight BO per layer (bo1). One activation BO (bo0).
// One output BO (bo2). Instructions loaded from pre-generated .txt files
// (blob_instr_transaction format), one BO per layer.
//
// Per-op xclbins or full ELFs, loaded from the kernel directory at runtime.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>


// Include npu_sequence for init_with_generator (may already be included by caller)
#if __has_include("npu_utils/npu_instr_utils.hpp")
#include "npu_utils/npu_instr_utils.hpp"
#else
struct npu_sequence;  // the gemm_generate_sequence_i8 decl below needs the name only
#endif

// Forward decl (defined in gemm_npu_instructions.cpp)
void gemm_generate_sequence_i8(
    npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N,
    uint32_t a_ddr_offset, uint32_t b_base_offset,
    bool add_bias, int activation, uint32_t bias_offset, uint32_t output_offset);

struct I8Ctx {
    int MD, KD, ND, NL;
    int bC_nd = 0;   // bo2 (C2) size override: MD*bC_nd*4 bytes when > 0, for a
                     // kernel whose output is wider than the logical ND.
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    // ── Full-ELF mode (no xclbin) ──
    // Consume a `aiecc --generate-full-elf` artifact directly: the ELF bundles
    // the PDIs + TXN control code, so there is no xclbin and no instruction
    // .txt. The hw_context is created from the ELF, and kext is the resulting
    // xrt::ext::kernel whose signature is (A, B, C). Requires the design's
    // runtime sequence to load the ELF's PDI (aiex.npu.load_pdi; see
    // generators/build_zaya_m8.sh), otherwise XRT never configures the array.
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> kext;
    bool elf_mode = false;
    std::unique_ptr<xrt::bo> bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;     // weight BOs
    std::vector<std::unique_ptr<xrt::bo>> layerInstr;  // instruction BOs
    std::vector<std::vector<uint32_t>> layerInstrData; // raw instruction data
    int8_t* Am;
    int32_t* Cm;
    std::vector<std::vector<float>> group_scales;
    bool initialized = false;

    ~I8Ctx() {}

    bool isReady() { return initialized && (k || kext) && bA && bC; }

    // ── Init with generated instructions (no pre-gen'd .txt files needed) ──
#if __has_include("npu_utils/npu_instr_utils.hpp")
    bool init_with_generator(xrt::device& d, const char* xp,
                             int M, int K, int N, int nlayers) {
        MD = M; KD = K; ND = N; NL = nlayers;
        fprintf(stderr, "  I8Ctx::init_with_generator xp=%s M=%d K=%d N=%d\n", xp, M, K, N);

        // The generated sequence matches only some kernel builds; pairing it
        // with a kernel that expects a different instruction stream silently
        // computes the wrong result rather than failing.  Xclbins built by run_build.sh always ship their
        // instruction file, so this path is only reached when that file is
        // missing.
        fprintf(stderr, "  WARN: generating single-core-row instructions; if %s\n"
                        "        was built multi-row (v27), its .txt instruction file is\n"
                        "        required and results will be wrong without it.\n", xp);

        // Generate instruction sequence
        npu_sequence seq(device_npu2);
        gemm_generate_sequence_i8(&seq, (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                  0, 0, false, 0, 0, 0);
        // Never cmds2seq() here: it appends a stale header after the raw payload.
        std::vector<uint32_t>& raw = seq.raw_seq();
        uint32_t ncmds = raw.back(); raw.pop_back();
        std::vector<uint32_t> ins;
        ins.reserve(raw.size() + 4);
        ins.push_back(0x06040100);
        ins.push_back(0x00000108);
        ins.push_back(ncmds);
        ins.push_back((uint32_t)(raw.size() * 4 + 16));
        ins.insert(ins.end(), raw.begin(), raw.end());
        fprintf(stderr, "  generated %zu instr bytes (%zu words)\n",
                ins.size() * sizeof(uint32_t), ins.size());

        // Register xclbin
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        int grp_a   = k->group_id(3);
        int grp_w   = k->group_id(4);
        int grp_c   = k->group_id(5);
        int grp_ins = k->group_id(1);

        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_a);
        bC = std::make_unique<xrt::bo>(d, (size_t)MD * ND * 4,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_c);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();
        // Both BOs are XRT_BO_FLAGS_HOST_ONLY and NOTHING zeroed them. Am is fully written by
        // quantize_async (memset(Am,0,MD*KD)) before every launch, so bA was safe -- but Cm is
        // the GEMM OUTPUT: the kernel writes only the valid rows of each launch, while the host
        // reads MD rows. On the FIRST launch the rows the kernel did not write were whatever the
        // device allocator handed back, which is why Nanbeige's boot token was nondeterministic
        // (1214 / 131718 / 145029 / 42438 ... for one command, RESULTS-coverage-multifamily 59,
        // 62). Zeroing makes the initial contents defined; section 61 made the same fix on the
        // bf16 path's KV BO, which is a different set of buffers.
        memset(Am, 0, (size_t)MD * KD);
        memset(Cm, 0, (size_t)MD * ND * 4);

        layerB.resize(NL);
        layerInstr.resize(NL);
        layerInstrData.resize(NL);
        group_scales.resize(NL);

        for (int l = 0; l < NL; l++) {
            layerB[l] = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                                   XRT_BO_FLAGS_HOST_ONLY, grp_w);
        }
        // ONE instruction BO per context: the instruction stream is identical
        // for every layer of the same GEMM (the kernel selects the layer via
        // layerB[l]); per-layer BOs exhaust device memory on large models.
        layerInstr.resize(1);
        layerInstrData.resize(1);
        layerInstrData[0] = ins;
        layerInstr[0] = std::make_unique<xrt::bo>(
            d, ins.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(layerInstr[0]->map(), ins.data(),
               ins.size() * sizeof(uint32_t));
        layerInstr[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        initialized = true;
        return true;
    }

    // ── M is ALWAYS 128 — there is no valid M<128 instruction stream ──
    // The kernels are built for M=128, so gemm_generate_sequence_i8 ignores
    // its M argument: the stream depends only on (K, N). A stream built for a
    // smaller M does not complete.
    //
    // Smaller batches MUST reuse the M=128 stream and pass the real row count
    // as `am` to go()/launch_*: quantize_async zero-pads rows [am, 128) so only
    // rows [0, am) are valid. That is what npu_engine_universal, zaya_decode
    // and zaya_npu_runner do for single-token decode. Real small-M streams
    // require per-shape small-M xclbins (build_xclbins.sh Peano path), not a
    // runtime regen. (A former regen_insts(int M) re-uploaded an identical
    // M=128 stream and claimed to resize the batch — removed as misleading.)
#else
    // Stub: npu_instr_utils.hpp not available — use init() with pre-gen'd files
    bool init_with_generator(xrt::device&, const char*, int, int, int, int) {
        fprintf(stderr, "  I8Ctx: init_with_generator unavailable (no npu_instr_utils)\n");
        return false;
    }
#endif

    // ── Init: consume a FULL ELF directly (no xclbin) ──
    // Kernel signature: (bo0, bo1, bo2) == (A, B, C). No instruction BO: the
    // TXN control stream lives in the ELF's .ctrltext. Kernel name is
    // "<kernel>:<instance>" from the aiecc full-ELF config ("main:seq" for the
    // v27 designs).
    bool init_elf(xrt::device& d, const char* ep, int M, int K, int N,
                  int nlayers, const char* kname = "main:seq") {
        MD = M; KD = K; ND = N; NL = nlayers;
        fprintf(stderr, "  I8Ctx::init_elf ep=%s M=%d K=%d N=%d nl=%d\n",
                ep, M, K, N, nlayers);
        try {
            elf = std::make_unique<xrt::elf>(std::string(ep));
            if (!elf->is_full_elf()) {
                fprintf(stderr, "  I8Ctx::init_elf: %s is not a full ELF\n", ep);
                return false;
            }
            fprintf(stderr, "  full ELF ok: uuid=%s\n",
                    elf->get_cfg_uuid().to_string().c_str());
            hc = std::make_unique<xrt::hw_context>(d, *elf);
            kext = std::make_unique<xrt::ext::kernel>(*hc, std::string(kname));
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx::init_elf failed: %s\n", ex.what());
            return false;
        }
        // A group-less xrt::bo is rejected by the amdxdna driver ("unsupported
        // buffer type: none flag") under the system XRT; xrt::ext::bo picks a
        // valid host-accessible memory group for the device.
        auto mkbo = [&](size_t sz) {
            xrt::ext::bo b{d, sz};
            return std::make_unique<xrt::bo>(b);
        };
        bA = mkbo((size_t)MD * KD);
        bC = mkbo((size_t)MD * ND * 4);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();
        layerB.resize(NL);
        group_scales.resize(NL);   // packB() writes group_scales[l]; the xclbin init does too
        for (int l = 0; l < NL; l++)
            layerB[l] = mkbo((size_t)KD * ND);
        elf_mode = true;
        initialized = true;
        return true;
    }

    // ── Init: load xclbin + per-layer instruction files ──
    bool init(xrt::device& d, const char* xp, const char* ip,
              int /*gid_B*/, int nlayers) {
        NL = nlayers;
        fprintf(stderr, "  I8Ctx::init xp=%s ip=%s\n", xp, ip);
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "  fopen failed: %s\n", ip); return false; }
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        fprintf(stderr, "  instr file size=%ld\n", sz);
        std::vector<uint32_t> ins(sz / 4);
        fread(ins.data(), 4, ins.size(), f);
        fclose(f);

        // Register xclbin
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
        } catch (std::exception& ex) {
            fprintf(stderr, "  I8Ctx: xclbin/kernel init failed: %s\n", ex.what());
            return false;
        }

        // Get kernel group IDs for BO allocation
        int grp_a   = k->group_id(3);  // bo0
        int grp_w   = k->group_id(4);  // bo1
        int grp_c   = k->group_id(5);  // bo2
        int grp_ins = k->group_id(1);  // instr
        fprintf(stderr, "  grp_a=%d grp_w=%d grp_c=%d grp_ins=%d\n", grp_a, grp_w, grp_c, grp_ins);

        // One activation BO + one output BO (shared across layers)
        fprintf(stderr, "  creating bA size=%zu (MD=%d KD=%d)\n", (size_t)MD * KD, MD, KD);
        bA = std::make_unique<xrt::bo>(d, (size_t)MD * KD,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_a);
        size_t bc_bytes = bC_nd > 0 ? (size_t)MD * bC_nd * 4 : (size_t)MD * ND * 4;
        fprintf(stderr, "  creating bC size=%zu (MD=%d ND=%d bC_nd=%d)\n", bc_bytes, MD, ND, bC_nd);
        bC = std::make_unique<xrt::bo>(d, bc_bytes,
                                       XRT_BO_FLAGS_HOST_ONLY, grp_c);
        Am = (int8_t*)bA->map();
        Cm = (int32_t*)bC->map();
        // See the note in the first init overload: neither BO was zeroed, and Cm is the GEMM
        // output, so rows the kernel did not write on the first launch were uninitialized
        // device memory -- the source of the nondeterministic boot token.
        memset(Am, 0, (size_t)MD * KD);
        memset(Cm, 0, bc_bytes);

        // Per-layer weight BOs + instruction BOs
        // ONE weight BO per context, NOT one per layer.  Every caller in the
        // model-generic forward packs and launches index 0 (npu_gemm /
        // run_lm_head in q4nx_forward.cpp, lm_gemm in engine.cpp), so NL-1 of
        // these BOs were pure waste -- and unlike the ELF path's ext BOs these
        // are xclbin-group BOs, so Qwen3-8B allocated ~32 x (4096*12288 etc.)
        // ~= 6 GB and exhausted the device heap; its kernel then made no
        // progress (0 tokens in 27 min, where the ELF path did 5 in 1329 s) and
        // llama-3.2-1B died with 'double free or corruption (out)'.  Mirrors the
        // instruction-BO fix below.
        layerB.resize(1);
        layerInstr.resize(1);
        layerInstrData.resize(1);
        group_scales.resize(NL);
        layerB[0] = std::make_unique<xrt::bo>(d, (size_t)KD * ND,
                                              XRT_BO_FLAGS_HOST_ONLY, grp_w);
        // ONE instruction BO per context: the instruction stream is identical
        // for every layer of the same GEMM (the kernel selects the layer via
        // layerB[l]); per-layer BOs exhaust device memory on large models.
        layerInstr.resize(1);
        layerInstrData.resize(1);
        layerInstrData[0] = ins;
        layerInstr[0] = std::make_unique<xrt::bo>(
            d, ins.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE, grp_ins);
        memcpy(layerInstr[0]->map(), ins.data(),
               ins.size() * sizeof(uint32_t));
        layerInstr[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        initialized = true;
        return true;
    }

    // ── Resident-expert (MoE) helpers: pack/launch against an arbitrary BO ──
    // Decode is M=1 with top-1 routing; re-streaming the selected expert's
    // weights into a shared per-layer BO every token costs a memcpy + sync on
    // the critical path (~30ms/tok for 20 layers). Instead allocate one
    // weight BO per (layer, expert) at startup, pack+sync once, and pass the
    // BO handle directly at decode.
    std::unique_ptr<xrt::bo> make_weight_bo(xrt::device& d) {
        // The xclbin path picks a group off the kernel object; the ELF path has
        // no kernel object (`k` is null) and the amdxdna driver rejects a
        // group-less xrt::bo, so use the same xrt::ext::bo allocation
        // init_elf's mkbo uses.
        if (elf_mode) {
            xrt::ext::bo b{d, (size_t)KD * ND};
            auto bo = std::make_unique<xrt::bo>(b);
            if (void* m = bo->map()) memset(m, 0, (size_t)KD * ND);
            return bo;
        }
        int grp_w = k->group_id(4);
        // Weight BOs are written once (packB_into) and read every token by the
        // shim DMA. HOST_ONLY forces the device through the slow cache-coherent
        // path (~3.6 GB/s measured); try normal/cacheable/svm for faster reads.
        uint32_t fl = XRT_BO_FLAGS_HOST_ONLY;
        if (const char* f = getenv("NPU_WBO_FLAGS")) {
            int v = atoi(f);
            if (v == 0) fl = 0;
            else if (v == 1) fl = XRT_BO_FLAGS_CACHEABLE;
            else if (v == 2) fl = XRT_BO_FLAGS_SVM;
        }
        auto bo = std::make_unique<xrt::bo>(d, (size_t)KD * ND, fl, grp_w);
        // packB_into memsets the mapped buffer before packing, so this BO is covered in practice;
        // zeroing at allocation as well costs one memset and removes the ordering assumption.
        if (void* m = bo->map()) memset(m, 0, (size_t)KD * ND);
        return bo;
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
                int8_t* brow = Bm + (size_t)i * ND + jb;
                for (int b = 0; b < jn; b++) {
                    float v = row[b];
                    if (!std::isfinite(v)) v = 0.0f;
                    int x = (int)roundf(v * tis[b]);
                    if (x > 127) x = 127;
                    else if (x < -127) x = -127;
                    brow[b] = (int8_t)x;
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
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (elf_mode) return (*kext)(*bA, wbo, *bC);
        return (*k)((unsigned)3, *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, wbo, *bC);
    }

    // Batched variant of launch_async_with_bo: one activation quantization with
    // PER-ROW scales, so each row of a prompt keeps its own dynamic range (a
    // single shared scale zeroes low-magnitude rows when one token's activations
    // dwarf the rest -- issue #1699). Dequantize with the same per-row scales.
    inline xrt::run launch_async_with_bo_rows(xrt::bo& wbo, const float* A,
                                              int am, int ak,
                                              const float* ascales) {
        quantize_async_rows(A, am, ak, ascales);
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (elf_mode) return (*kext)(*bA, wbo, *bC);
        return (*k)((unsigned)3, *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, wbo, *bC);
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
    inline int8_t* quantize_async(const float* A, int am, int ak, float ascale) {
        float ais = 1.0f / ascale;
        // Zero-pad ALL MD rows: the M=128 stream reads rows
        // [am, MD) every launch; zeroing only rows [0, am) left stale BO
        // memory (the previous launch's A) in [am, MD) — an
        // uninitialized-read-class hazard for any kernel with cross-row
        // interaction.
        memset(Am, 0, (size_t)MD * KD);
        for (int m = 0; m < am; m++)
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
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
        memset(Am, 0, (size_t)MD * KD);
        for (int m = 0; m < am; m++) {
            float ais = 1.0f / ascales[m];
            for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais);
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        }
        return Am;
    }


    inline void sync_A(int /*l*/) { bA->sync(XCL_BO_SYNC_BO_TO_DEVICE); }

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
        if (elf_mode) return (*kext)(*bA, *layerB[l], *bC);
        return (*k)((unsigned)3,
                    *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, *layerB[l], *bC);
    }

    inline xrt::run sync_and_launch(int l) {
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bochk(l);
        if (elf_mode) return (*kext)(*bA, *layerB[l], *bC);
        return (*k)((unsigned)3,
                    *layerInstr[0],
                    (unsigned)(layerInstrData[0].size()),
                    *bA, *layerB[l], *bC);
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
    inline void readback() {
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (getenv("NPU_DBG")) {
            static int n = 0;
            if (n < 8) { n++; fprintf(stderr, "[RBCHK] bC=%016llx\n", bo_fnv(*bC)); }
        }
    }

    inline void dequant_only(float* C, int am, int an, float ascale,
                             float Bscale, int layer = -1) {
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
        quantize_async(A, am, ak, ascale);
        return sync_and_launch(l);
    }

    // Async launch with per-row activation scales (batched prefill fix,
    // #1699): row m quantized with ascales_q[m]. Dequant must use the same
    // per-row scales (finish_async_rows). Prevents the shared-batch ascale
    // from zeroing low-magnitude rows when one token's activations dwarf the
    // rest (Qwen3-0.6B: pos0 su max ~3671 vs pos1-3 max ~5 -> rows 1-3 were
    // quantized to all-zero int8 and the D GEMM emitted zeros).
    inline xrt::run launch_async_rows(int l, const float* A, int am, int ak,
                                      const float* ascales_q) {
        quantize_async_rows(A, am, ak, ascales_q);
        return sync_and_launch(l);
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
