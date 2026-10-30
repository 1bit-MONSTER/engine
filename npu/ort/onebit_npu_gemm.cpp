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
    // One resident weight per initialiser: the runtime keeps a weight BO per layer (layerB[l]), so a
    // weight is packed once, on first use, and a steady-state token does no packing at all. Keyed by
    // the initialiser pointer, which is stable for the life of the session's weights.
    std::vector<const float*> slots;
    std::vector<float> scales;
};

// Weight slots per kernel. A transformer has one weight per projection shape per layer, so this is
// the depth a single shared NPU context needs at the moment; K*N*slots bytes of device memory.
constexpr int kSlots = 24;

// A fused QKV kernel is not a plain projection: its weights are packed per section (q/k/v, three
// scale sets) and read back per section (I8Ctx::pack_qkv_sec / dequant_qkv_rows, the fix for #1699).
// Driving one through packB/go_rows quantises three sections with one scale, which is exactly the
// ~5%-per-layer error that flips tokens, so such ELFs are ignored until section support lands.
bool sectioned_elf(const std::string& elf) { return elf.find("QKV") != std::string::npos; }

// One process-wide NPU context: the device, the kernels, and the lock that keeps other processes
// out. The EP is single-session in practice, so one is enough.
struct State {
    std::mutex mu;
    bool tried = false, ok = false;
    std::string why;
    int lock_fd = -1;
    std::unique_ptr<xrt::device> dev;
    std::vector<std::unique_ptr<Kernel>> kernels;
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

    for (const auto& [dims, elf] : spec) {
        int m = 0, k = 0, n = 0;
        if (sscanf(dims.c_str(), "%dx%dx%d", &m, &k, &n) != 3 || m <= 0 || k <= 0 || n <= 0)
            return fail(s, "ONEBIT_EP_NPU_ELFS: bad design dims '" + dims + "'");
        if (sectioned_elf(elf)) {
            fprintf(stderr,
                    "onebit_npu: ignoring %s -- fused QKV kernel needs per-section weights "
                    "(pack_qkv_sec), not a plain projection; that shape falls back to the CPU EP\n",
                    elf.c_str());
            continue;
        }
        auto kernel = std::make_unique<Kernel>();
        kernel->m = m;
        kernel->k = k;
        kernel->n = n;
        kernel->elf = elf;
        kernel->ctx = std::make_unique<I8Ctx>();
        if (!kernel->ctx->init_elf(*s.dev, elf.c_str(), m, k, n, kSlots))
            return fail(s, "I8Ctx init_elf failed for " + elf + " (" + dims + ")");
        s.kernels.push_back(std::move(kernel));
    }

    s.ok = true;
    s.why.clear();
    return true;
}

}  // namespace

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
            if (sectioned_elf(item.substr(eq + 1))) continue;
            if (kk == k && nn == n) return true;
        }
        return false;
    }
    int m = 0, kd = 0, nd = 0;
    if (!env("ONEBIT_EP_NPU_ELF") || sectioned_elf(env("ONEBIT_EP_NPU_ELF")) ||
        !env_int("ONEBIT_EP_NPU_M", &m) ||
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

    // Weights: int8 per column, one scale per column (packB), packed once per initialiser.
    int slot = -1;
    for (size_t i = 0; i < kern->slots.size(); i++)
        if (kern->slots[i] == b) slot = (int)i;
    if (slot < 0) {
        if ((int)kern->slots.size() >= kSlots) {
            if (err)
                *err = "no free NPU weight slot (" + std::to_string(kSlots) +
                       " per shape) -- more distinct weights than the context holds";
            return false;
        }
        slot = (int)kern->slots.size();
        kern->slots.push_back(b);
        kern->scales.resize((size_t)slot + 1, 1.0f);
        kern->ctx->packB(slot, b, (int)k, (int)n, kern->scales[(size_t)slot]);
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
        if (!kern->ctx->go_rows(slot, a + row0 * k, rows, (int)k, scales.data(), scales.data(),
                                1.0f, c + row0 * n, (int)n)) {
            if (err) *err = "I8Ctx go_rows failed";
            return false;
        }
    }
    return true;
}

}  // namespace onebit::npu
