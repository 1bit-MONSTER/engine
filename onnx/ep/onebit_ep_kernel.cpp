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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// limitations under the License.

// The one seam between the ORT plugin EP and whatever actually multiplies.
//
// Deliverable 2 ran every MatMul here on the host, so the EP ABI and the partitioning were what
// got tested with nothing moving underneath them. With ONEBIT_EP_NPU=ON the multiply goes to the
// existing NPU runtime GEMM instead (npu/ort/onebit_npu_gemm.*: I8Ctx, the forward route's
// packB + amax/127 quantisation, the device lock), and the host loop below stays as the fallback:
// a busy device, no kernel for this K x N, or any XRT failure must leave the session working,
// never fail the graph. Signature and contract are identical either way — row-major A[M,K] x
// B[K,N] -> C[M,N], C fully written — so the EP's numerics comparison stays the same test.
//
// This is deliberately not an interface or a dispatch table. One function, one call site.

#include "onebit_ep_kernel.h"

#include <cstdarg>
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#ifdef ONEBIT_EP_NPU
#include "onebit_npu_gemm.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#endif

namespace onebit_ep {
namespace {

// Every note of this file goes through here.
//
// open(0600) + fdopen, not fopen("a"): fopen creates with 0666 & ~umask, so under a permissive umask the
// note is world-writable in /tmp (CodeQL cpp/world-writable-file-creation), and it would follow a symlink
// planted at that path. This is exactly what the EP's own trace does (onebit_ep.cpp, tracef), so the two
// writers agree on the mode as well as the file.
//
// `stderr_too` exists for the one caller whose only output this is (ONEBIT_EP_QDQ_PROF): a knob that prints
// nowhere when ONEBIT_EP_TRACE is off is the same footgun as the path-vs-flag bug above, so that line is
// never silently dropped.
void trace_line(const char* line, bool stderr_too) {
    char path[64];
    std::snprintf(path, sizeof(path), "/tmp/onebit_ep_trace.%ld.log", (long)::getpid());
    bool written = false;
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd >= 0) {
        FILE* f = ::fdopen(fd, "a");
        if (f) {
            std::fputs(line, f);
            std::fputc('\n', f);
            std::fclose(f);
            written = true;
        } else {
            ::close(fd);
        }
    }
    if (!written || stderr_too) std::fprintf(stderr, "%s\n", line);
}

// ONEBIT_EP_TRACE is a *flag* -- onebit_ep.cpp opens /tmp/onebit_ep_trace.<pid>.log when it is set --
// so a fallback note has to be appended to that same file. Reading the variable as a path (which this
// file did until it cost me a debugging session) opens a file literally named "1" in whatever the
// current directory is, and the note is invisible exactly when someone is looking for it.
void trace_fallback(const char* fmt, ...) {
    const char* on = std::getenv("ONEBIT_EP_TRACE");
    if (!on || !*on || *on == '0') return;
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    trace_line(line, /*stderr_too=*/false);
}

// ONEBIT_EP_QDQ_PROF=1: where the fused group's wall time goes, as one running total written when the
// process exits. #1023 measured the device arm 1.13-1.16x *slower* than the host arm at model level while
// the per-design probe, at the same shapes, had the device kernels 1.7x ahead; the two readings reconcile
// only if the per-call dispatch or the probe's synthetic operands account for the gap, and nothing
// separated them. This does: the same clock on both legs, over the graph's own int8 bytes, in the arm that
// runs them. Off unless asked, one line per process, so an unmeasured run pays one getenv and one branch.
struct Q8Prof {
    const bool on = std::getenv("ONEBIT_EP_QDQ_PROF") != nullptr;
    double dev_call_ms = 0, dev_epi_ms = 0, host_ms = 0;
    // Where dev_call_ms goes, from the device leg's own timers (Q8RawSplit). `other` is what is left:
    // the device lock, the kernel lookup, and the EP's own frame around the call. Printed rather than
    // subtracted by a reader, so the line adds up on its face.
    double dev_wpack_ms = 0, dev_aup_ms = 0, dev_async_ms = 0, dev_launch_ms = 0, dev_rdb_ms = 0;
    long calls_dev = 0, calls_host = 0, rows_dev = 0, rows_host = 0;
    ~Q8Prof() {
        if (!on || (calls_dev == 0 && calls_host == 0)) return;
        const double split = dev_wpack_ms + dev_aup_ms + dev_async_ms + dev_launch_ms + dev_rdb_ms;
        const double other = dev_call_ms > split ? dev_call_ms - split : 0.0;
        char line[512];
        std::snprintf(line, sizeof(line),
                      "qdq_prof calls_dev=%ld calls_host=%ld rows_dev=%ld rows_host=%ld"
                      " dev_call_ms=%.1f dev_epi_ms=%.1f host_ms=%.1f"
                      " dev_split_wpack=%.1f dev_split_aupload=%.1f dev_split_async=%.1f"
                      " dev_split_launch=%.1f dev_split_readback=%.1f dev_split_other=%.1f",
                      calls_dev, calls_host, rows_dev, rows_host, dev_call_ms, dev_epi_ms, host_ms,
                      dev_wpack_ms, dev_aup_ms, dev_async_ms, dev_launch_ms, dev_rdb_ms, other);
        // Bare stderr when ONEBIT_EP_TRACE is off: this line is the knob's whole output, and losing the one
        // measurement that answers the question is not worth saving a trace file.
        trace_line(line, /*stderr_too=*/std::getenv("ONEBIT_EP_TRACE") == nullptr);
    }
};
Q8Prof& q8_prof() {
    static Q8Prof p;
    return p;
}

}  // namespace

// Returns true when the NPU did the multiply. Always writes all of C.
bool gemm_f32(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n) {
#ifdef ONEBIT_EP_NPU
    std::string why;
    if (onebit::npu::gemm_f32(a, m, k, n, b, c, &why)) return true;
    trace_fallback("npu_fallback k=%lld n=%lld why=%s", (long long)k, (long long)n, why.c_str());
#endif
    std::memset(c, 0, (size_t)m * (size_t)n * sizeof(float));
    // i-k-j instead of i-j-k: each B row is walked once per A row, which is what keeps this
    // within a small factor of MLAS on shapes that matter here (K,N in the hundreds).
    for (int64_t i = 0; i < m; ++i) {
        float* crow = c + i * n;
        for (int64_t p = 0; p < k; ++p) {
            const float aik = a[i * k + p];
            if (aik == 0.0f) continue;
            const float* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) crow[j] += aik * brow[j];
        }
    }
    return false;
}

// One output element of the folded QuantizeLinear: round half to even (nearbyint under the default
// FE_TONEAREST), add the zero point, saturate to int8. The order matters -- rounding after adding an
// integer zero point would move exact ties to the wrong neighbour.
static inline int8_t q8_quantize(float v, int32_t zp) {
    float r = std::nearbyint(v);
    // The output is int8 whatever happens next: keep the integer add in range.
    if (r > 1.0e6f) r = 1.0e6f;
    if (r < -1.0e6f) r = -1.0e6f;
    const int32_t q = zp + (int32_t)r;
    if (q > 127) return 127;
    if (q < -128) return -128;
    return (int8_t)q;
}

void gemm_q8(const int8_t* a, const int8_t* b, int8_t* y, int64_t m, int64_t k, int64_t n,
             const Q8Epilogue& e) {
    // The whole dequantise/requantise chain collapses to one scale per output element; the column half
    // of it (and 1/s_y) is folded once, up front.
    std::vector<float> wmul((size_t)n);
    const float sw = e.s_w * e.inv_s_y;
    for (int64_t j = 0; j < n; ++j) wmul[(size_t)j] = e.s_w_col ? e.s_w_col[j] * e.inv_s_y : sw;

    const bool asym = e.zp_x != 0 || e.zp_w != 0 || e.zp_x_row || e.zp_w_col || e.zp_y != 0;
    const bool per_tensor = !asym && !e.s_x_row && !e.s_w_col;
    const float scale = e.s_x * e.s_w * e.inv_s_y;  // per-tensor symmetric path, hoisted
    const bool needs_rowsum = e.zp_w != 0 || e.zp_w_col != nullptr;

    // One row of exact int32 accumulators -> int8. Shared by the device path and the host loop, so the
    // scales, zero points and rounding exist in exactly one place: a device run and a host run of the
    // same graph differ only where the accumulator itself would differ (never, it is exact integer).
    auto epilogue_row = [&](const int32_t* acc, int64_t i, int8_t* yrow) {
        if (per_tensor) {
            for (int64_t j = 0; j < n; ++j) yrow[j] = q8_quantize((float)acc[j] * scale, 0);
            return;
        }
        const float sx_row = e.s_x_row ? e.s_x_row[i] : e.s_x;
        const int64_t zx = e.zp_x_row ? e.zp_x_row[i] : e.zp_x;
        int64_t rowsum = 0;  // sum_k a_q[m][k]: only a non-zero zp_w needs it
        if (asym && needs_rowsum) {
            const int8_t* arow = a + i * k;
            for (int64_t p = 0; p < k; ++p) rowsum += arow[p];
        }
        for (int64_t j = 0; j < n; ++j) {
            int64_t c = acc[j];
            if (asym) {
                const int64_t zw = e.zp_w_col ? e.zp_w_col[j] : e.zp_w;
                if (zx) c -= zx * (int64_t)e.colsum_w[j];
                if (zw) {
                    c -= zw * rowsum;
                    c += (int64_t)k * zx * zw;
                }
            }
            yrow[j] = q8_quantize((float)c * sx_row * wmul[(size_t)j], e.zp_y);
        }
    };

#ifdef ONEBIT_EP_NPU
    // The device leg: int8 in, exact int32 accumulators out, the epilogue above on the way back. A
    // graph whose (K, N) has no ELF, or a busy device, leaves the row blocks to the host loop below.
    // Below kDeviceMinRows the host loop is the faster leg -- the call is dispatch-bound, not work-bound --
    // so that case is a policy choice, logged with its own name: a run is a device run only when neither
    // this line nor npu_qdq_fallback is written for the shape.
    if (m < kDeviceMinRows) {
        trace_fallback("npu_qdq_host_rows m=%lld k=%lld n=%lld min=%lld", (long long)m, (long long)k,
                       (long long)n, (long long)kDeviceMinRows);
    } else {
        std::vector<int32_t> acc((size_t)m * (size_t)n);
        std::string why;
        Q8Prof& p = q8_prof();
        // Where the call's time goes, but only when the profiler is on: a plain run passes null and the
        // device leg pays one branch per timed section instead of five clock reads per launch block.
        onebit::npu::Q8RawSplit split;
        const auto t_dev = p.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (onebit::npu::gemm_q8_raw(a, m, k, n, b, acc.data(), &why, p.on ? &split : nullptr)) {
            const auto t_epi =
                p.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (int64_t i = 0; i < m; ++i) epilogue_row(acc.data() + i * n, i, y + i * n);
            if (p.on) {
                const auto t_end = std::chrono::steady_clock::now();
                p.dev_call_ms += std::chrono::duration<double, std::milli>(t_epi - t_dev).count();
                p.dev_epi_ms += std::chrono::duration<double, std::milli>(t_end - t_epi).count();
                p.dev_wpack_ms += split.wpack_ms;
                p.dev_aup_ms += split.aupload_ms;
                p.dev_async_ms += split.async_ms;
                p.dev_launch_ms += split.launch_ms;
                p.dev_rdb_ms += split.readback_ms;
                p.calls_dev++;
                p.rows_dev += m;
            }
            return;
        }
        trace_fallback("npu_qdq_fallback k=%lld n=%lld why=%s", (long long)k, (long long)n,
                       why.c_str());
    }
#endif

    // Also where a refused or busy device call ends up, so host_ms is the whole host leg, not just the
    // calls that were policy-routed here.
    Q8Prof& p = q8_prof();
    const auto t_host = p.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<int32_t> acc((size_t)n);
    for (int64_t i = 0; i < m; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        // i-k-j: exact int32 accumulation, one B row walked once per A row.
        for (int64_t p = 0; p < k; ++p) {
            const int32_t aik = a[i * k + p];
            if (aik == 0) continue;
            const int8_t* brow = b + p * n;
            for (int64_t j = 0; j < n; ++j) acc[(size_t)j] += aik * brow[j];
        }
        epilogue_row(acc.data(), i, y + i * n);
    }
    if (p.on) {
        p.host_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_host).count();
        p.calls_host++;
        p.rows_host += m;
    }
}

}  // namespace onebit_ep
