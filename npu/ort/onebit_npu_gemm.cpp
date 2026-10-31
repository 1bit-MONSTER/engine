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
// onebit_npu_gemm.cpp — see onebit_npu_gemm.h for the why.
#include "onebit_npu_gemm.h"

// No direct XRT include on purpose: whichever NPU tree is configured decides the runtime
// (npu/rt.h aliases the private runtime's npu_rt_api as xrt::, so this file names xrt:: only). I8Ctx
// carries the rest.
#include "npu_engine_i8ctx_inc.h"  // I8Ctx — the forward route's GEMM context

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace onebit::npu {
namespace {

// One kernel: an ELF built for one design (M rows, K, N) and the last weight packed into it.
// A transformer needs one per projection shape, so the EP holds a small set rather than one.
struct Kernel {
    int m = 0, k = 0, n = 0;
    std::string elf;
    std::unique_ptr<I8Ctx> ctx;
    // Last packed weight, so repeated MatMuls against the same tensor do not re-quantise it.
    // shortcut: one slot per kernel; two different weights of the same shape repack per call.
    const float* w_ptr = nullptr;
    int64_t w_k = 0, w_n = 0;
    float w_scale = 1.0f;
};

// One QDQ-path kernel: the same ELF, but the weight BO holds the quantised graph's own int8 bytes and
// nothing here knows about scales. Created on demand for the (K, N) a fused node asks for, because a
// second I8Ctx per shape costs its BOs (bA + bC + one layerB) and most graphs ask for a few shapes.
struct Q8Kernel {
    int m = 0, k = 0, n = 0;
    std::string elf;
    std::unique_ptr<I8Ctx> ctx;
    const int8_t* w_ptr = nullptr;  // the graph's weights whose bytes are in layerB[0]
    int64_t w_k = 0, w_n = 0;
};

// One process-wide NPU context: the device, the kernels, and the lock that keeps other processes
// out. The EP is single-session in practice, so one is enough.
struct State {
    std::mutex mu;
    bool tried = false, ok = false;
    std::string why;
    int lock_fd = -1;
    std::unique_ptr<xrt::device> dev;
    std::vector<std::unique_ptr<Kernel>> kernels;
    // Configured designs (MxKxN -> ELF), kept so a QDQ kernel can be built for a shape later without
    // re-reading the environment or re-opening the device.
    std::vector<std::pair<std::string, std::string>> specs;
    std::vector<std::unique_ptr<Q8Kernel>> q8_kernels;
    // Shapes whose QDQ kernel could not be built (a bad ELF, a design M that does not match it). Remembered
    // because building one means parsing an ELF and creating a hw_context -- ~50 ms -- and retrying that on
    // every call turns a fallback into a 20x slowdown: measured, 8 decode tokens went from 4 s to 87 s when
    // every group paid a failed init_elf.
    std::vector<std::pair<int64_t, int64_t>> q8_unavailable;
};

State& state() {
    static State s;
    return s;
}

const char* env(const char* name) {
    const char* v = getenv(name);
    return (v && *v) ? v : nullptr;
}

bool env_int(const char* name, int* out) {
    const char* v = env(name);
    if (!v) return false;
    char* end = nullptr;
    const long n = strtol(v, &end, 10);
    if (end == v || n <= 0 || n > (1 << 20)) return false;
    *out = (int)n;
    return true;
}

bool fail(State& s, const std::string& why) {
    s.why = why;
    s.ok = false;
    return false;
}

// Caller holds s.mu.
bool init(State& s) {
    if (s.tried) return s.ok;
    s.tried = true;

    // 1. the device lock, non-blocking: never evict the owner's forward serve. Same file, same
    // flock(2) and the same open semantics as app/npu_device_lock.h (engine#640):
    // kNpuDeviceLock = "/tmp/1bit-npu-device.lock", O_CREAT|O_RDONLY|O_NOFOLLOW|O_CLOEXEC, 0644 --
    // a read fd is enough for flock, 0644 keeps the file openable by other users, and O_NOFOLLOW
    // refuses a symlink planted in /tmp. The difference is what happens on contention: that helper
    // queues (wait=true) unless asked not to, while this one always gives up, because a plugin EP
    // inside an inference session must fall back to the CPU instead of blocking the session on
    // another process' serve.
    // shortcut: three lines copied from that helper rather than including it, because #640 is not
    // merged yet and a .so cannot depend on a header that is not on main; fold this into
    // take_npu_device_lock(1, onebit::kNpuDeviceLock, /*wait=*/false) once #640 lands.
    const char* lock_path = "/tmp/1bit-npu-device.lock";
    s.lock_fd = open(lock_path, O_CREAT | O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (s.lock_fd < 0)
        return fail(s, std::string("cannot open device lock ") + lock_path);
    if (flock(s.lock_fd, LOCK_EX | LOCK_NB) != 0) {
        close(s.lock_fd);
        s.lock_fd = -1;
        return fail(s, std::string("device busy: ") + lock_path +
                           " is held by another process (falling back to CPU)");
    }

    // 2. the kernels. Each ELF is built for one design, so the shapes come from configuration:
    //    ONEBIT_EP_NPU_ELFS="MxKxN=<path>,MxKxN=<path>" for a model, or the single
    //    ONEBIT_EP_NPU_ELF plus ONEBIT_EP_NPU_M/K/N for one projection.
    std::vector<std::pair<std::string, std::string>> spec;  // (MxKxN, path)
    if (const char* list = env("ONEBIT_EP_NPU_ELFS")) {
        std::string rest = list;
        while (!rest.empty()) {
            const size_t comma = rest.find(',');
            const std::string item = rest.substr(0, comma);
            rest = comma == std::string::npos ? "" : rest.substr(comma + 1);
            const size_t eq = item.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 >= item.size())
                return fail(s, std::string("ONEBIT_EP_NPU_ELFS: expected MxKxN=path, got '") + item + "'");
            spec.emplace_back(item.substr(0, eq), item.substr(eq + 1));
        }
        if (spec.empty()) return fail(s, "ONEBIT_EP_NPU_ELFS is empty");
    } else {
        const char* elf = env("ONEBIT_EP_NPU_ELF");
        if (!elf) return fail(s, "no NPU kernel configured (ONEBIT_EP_NPU_ELFS or ONEBIT_EP_NPU_ELF)");
        int m = 0, k = 0, n = 0;
        if (!env_int("ONEBIT_EP_NPU_M", &m) || !env_int("ONEBIT_EP_NPU_K", &k) ||
            !env_int("ONEBIT_EP_NPU_N", &n))
            return fail(s, "ONEBIT_EP_NPU_M/K/N must name the ELF's design dims");
        spec.emplace_back(std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n), elf);
    }

    // 3. the device
    for (int id = 0; id < 4 && !s.dev; id++) {
        try {
            s.dev = std::make_unique<xrt::device>(id);
        } catch (const std::exception&) {
        }
    }
    if (!s.dev) return fail(s, "no XRT device (amdxdna) found");

    s.specs = spec;  // a QDQ kernel is built from these on first use
    for (const auto& [dims, elf] : spec) {
        int m = 0, k = 0, n = 0;
        if (sscanf(dims.c_str(), "%dx%dx%d", &m, &k, &n) != 3 || m <= 0 || k <= 0 || n <= 0)
            return fail(s, "ONEBIT_EP_NPU_ELFS: bad design dims '" + dims + "'");
        auto kernel = std::make_unique<Kernel>();
        kernel->m = m;
        kernel->k = k;
        kernel->n = n;
        kernel->elf = elf;
        kernel->ctx = std::make_unique<I8Ctx>();
        if (!kernel->ctx->init_elf(*s.dev, elf.c_str(), m, k, n, 1))
            return fail(s, "I8Ctx init_elf failed for " + elf + " (" + dims + ")");
        s.kernels.push_back(std::move(kernel));
    }

    s.ok = true;
    s.why.clear();
    return true;
}

// #701 puts a canary in I8Ctx::readback(): a short result row is then detected instead of copied out as a
// huge negative int32. This branch predates it, so the probe is written to compile on both trees and to start
// working the moment the canary exists -- an `if constexpr` on a dependent type is not instantiated when the
// method is absent. What must NOT happen is the silent version: a poisoned row going through the shared
// epilogue and into the graph, which is the failure mode the canary exists to stop.
template <class Ctx, class = void>
struct has_canary : std::false_type {};
template <class Ctx>
struct has_canary<Ctx, std::void_t<decltype(std::declval<Ctx&>().canary_short())>> : std::true_type {};

template <class Ctx>
bool short_read(Ctx& ctx) {
    if constexpr (has_canary<Ctx>::value) {
        return ctx.canary_short();
    } else {
        (void)ctx;
        return false;  // no canary in this tree: nothing to ask
    }
}

// The QDQ kernel for one (K, N), built the first time it is asked for. Caller holds s.mu.
Q8Kernel* q8_kernel(State& s, int64_t k, int64_t n) {
    for (auto& kern : s.q8_kernels)
        if (kern->k == (int)k && kern->n == (int)n) return kern.get();
    for (const auto& [fk, fn] : s.q8_unavailable)
        if (fk == k && fn == n) return nullptr;  // known bad: do not parse the ELF again
    for (const auto& [dims, elf] : s.specs) {
        int m = 0, kk = 0, nn = 0;
        if (sscanf(dims.c_str(), "%dx%dx%d", &m, &kk, &nn) != 3) continue;
        if (kk != (int)k || nn != (int)n) continue;
        auto kern = std::make_unique<Q8Kernel>();
        kern->m = m;
        kern->k = kk;
        kern->n = nn;
        kern->elf = elf;
        kern->ctx = std::make_unique<I8Ctx>();
        if (!kern->ctx->init_elf(*s.dev, elf.c_str(), m, kk, nn, 1)) {
            s.why = "I8Ctx init_elf failed for " + elf + " (" + dims + ") -- check the design M";
            s.q8_unavailable.emplace_back(k, n);
            return nullptr;
        }
        s.q8_kernels.push_back(std::move(kern));
        return s.q8_kernels.back().get();
    }
    return nullptr;
}

}  // namespace

bool gemm_q8_raw(const int8_t* a, int64_t m, int64_t k, int64_t n, const int8_t* b, int32_t* c,
                 std::string* err) {
    State& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    if (!init(s)) {
        if (err) *err = s.why;
        return false;
    }
    Q8Kernel* kern = q8_kernel(s, k, n);
    if (!kern) {
        if (err) *err = "no QDQ kernel for " + std::to_string(k) + "x" + std::to_string(n);
        return false;
    }
    I8Ctx& ctx = *kern->ctx;

    // Weights: the graph's int8 bytes, in the device's own layout, exactly as packB would have written
    // them from floats -- but without the amax/127 step, because these bytes already carry the graph's
    // per-column scale.
    // ONNX initializers are immutable, so the tensor's address is its identity: a graph that rewrote the bytes
    // in place would keep the old ones here, and re-uploading on every call would cost what this cache saves.
    if (kern->w_ptr != b || kern->w_k != k || kern->w_n != n) {
        auto* Bm = (int8_t*)ctx.layerB[0]->map();
        std::memset(Bm, 0, (size_t)ctx.KD * ctx.ND);
        for (int64_t p = 0; p < k; ++p)
            for (int64_t j = 0; j < n; ++j)
                Bm[ctx.bpos((size_t)p, (size_t)j)] = b[p * n + j];
        ctx.layerB[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kern->w_ptr = b;
        kern->w_k = k;
        kern->w_n = n;
    }

    // Activations: the graph's int8 bytes, M rows per launch (the ELF's design). Rows past `rows` are
    // zeroed, which is what the kernel reads for them and what the host path also assumes.
    for (int64_t row0 = 0; row0 < m; row0 += ctx.MD) {
        const int rows = (int)std::min<int64_t>(ctx.MD, m - row0);
        std::memset(ctx.Am, 0, (size_t)ctx.MD * ctx.KD);
        for (int r = 0; r < rows; ++r)
            std::memcpy(ctx.Am + (size_t)r * ctx.KD, a + (row0 + r) * k, (size_t)k);
        auto launch_and_read = [&ctx] {
            ctx.sync_A(0);
            auto run = ctx.launch(0);
            ctx.wait_kernel(run);
            ctx.readback();
        };
        launch_and_read();
        if (short_read(ctx)) {
            // Once, as #701's own call sites do: a short read is a race with the completion of the last
            // row, not a systematic failure.
            launch_and_read();
            if (short_read(ctx)) {
                if (err) *err = "short read (canary) for " + std::to_string(k) + "x" + std::to_string(n) +
                                 " at row " + std::to_string(row0);
                return false;  // gemm_q8 redoes every row on the host; never write a poisoned accumulator
            }
        }
        for (int r = 0; r < rows; ++r)
            for (int64_t j = 0; j < n; ++j) c[(row0 + r) * n + j] = ctx.Cm[(size_t)r * ctx.ND + j];
    }
    return true;
}

bool has_kernel(int64_t k, int64_t n) {
    if (const char* list = env("ONEBIT_EP_NPU_ELFS")) {
        std::string rest = list;
        while (!rest.empty()) {
            const size_t comma = rest.find(',');
            const std::string item = rest.substr(0, comma);
            rest = comma == std::string::npos ? "" : rest.substr(comma + 1);
            const size_t eq = item.find('=');
            if (eq == std::string::npos) continue;
            int m = 0, kk = 0, nn = 0;
            if (sscanf(item.substr(0, eq).c_str(), "%dx%dx%d", &m, &kk, &nn) != 3) continue;
            if (kk == k && nn == n) return true;
        }
        return false;
    }
    int m = 0, kd = 0, nd = 0;
    if (!env("ONEBIT_EP_NPU_ELF") || !env_int("ONEBIT_EP_NPU_M", &m) ||
        !env_int("ONEBIT_EP_NPU_K", &kd) || !env_int("ONEBIT_EP_NPU_N", &nd))
        return false;
    return kd == k && nd == n;
}

bool available(std::string* why) {
    State& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    const bool ok = init(s);
    if (!ok && why) *why = s.why;
    return ok;
}

bool gemm_f32(const float* a, int64_t m, int64_t k, int64_t n, const float* b,
              float* c, std::string* err) {
    State& s = state();
    std::lock_guard<std::mutex> g(s.mu);
    if (!init(s)) {
        if (err) *err = s.why;
        return false;
    }
    Kernel* kern = nullptr;
    for (auto& kk : s.kernels)
        if (kk->k == (int)k && kk->n == (int)n) kern = kk.get();
    if (!kern) {
        if (err) {
            *err = "no NPU kernel for " + std::to_string(k) + "x" + std::to_string(n) + " (configured: ";
            for (size_t i = 0; i < s.kernels.size(); i++)
                *err += std::string(i ? ", " : "") + std::to_string(s.kernels[i]->k) + "x" +
                        std::to_string(s.kernels[i]->n);
            *err += ")";
        }
        return false;
    }

    // Weights: int8 per column, one scale per column (packB).
    // ONNX initializers are immutable, so the tensor's address is its identity: a graph that rewrote the bytes
    // in place would keep the old ones here, and re-uploading on every call would cost what this cache saves.
    if (kern->w_ptr != b || kern->w_k != k || kern->w_n != n) {
        kern->ctx->packB(0, b, (int)k, (int)n, kern->w_scale);
        kern->w_ptr = b;
        kern->w_k = k;
        kern->w_n = n;
    }

    // Activations: one row group of the kernel's M at a time. Rows past `rows` are zero-padded
    // inside quantize_async_rows and never read back.
    std::vector<float> scales((size_t)kern->m);
    for (int64_t row0 = 0; row0 < m; row0 += kern->m) {
        const int rows = (int)std::min<int64_t>(kern->m, m - row0);
        for (int r = 0; r < rows; r++) {
            const float* row = a + (row0 + r) * k;
            float amax = 0.0f;
            for (int64_t i = 0; i < k; i++) amax = std::max(amax, std::fabs(row[i]));
            scales[(size_t)r] = amax > 0.0f ? amax / 127.0f : 1.0f;
        }
        if (!kern->ctx->go_rows(0, a + row0 * k, rows, (int)k, scales.data(), scales.data(),
                                1.0f, c + row0 * n, (int)n)) {
            if (err) *err = "I8Ctx go_rows failed";
            return false;
        }
    }
    return true;
}

}  // namespace onebit::npu
