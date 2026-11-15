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
// audio_parallel.h -- the fork-join the host audio executor splits independent work over (engine#1931).
//
// Deterministic by construction: unit u always runs fn(u) exactly once, and a unit writes only its own slice of the
// output, so every element's arithmetic is unchanged by the thread count and the result is bit-identical to the serial
// loop. That is the property a reference implementation has to keep, and it is asserted, not assumed:
// tests/audio_parallel_test.cpp compares the executor's MatMul byte for byte across thread counts.
//
// A unit that throws does not take the process down. A std::thread whose function lets an exception escape calls
// std::terminate, so each worker catches, every worker is joined, and the first exception (lowest worker index, which
// makes the choice repeatable) is rethrown on the calling thread, where the executor turns it into a refusal with a
// reason like every other failure.
//
// Header-only and free of the executor's types, so the test needs no pack and no device.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace onebit::npu::pack_import {

// How many threads the audio executor may use. ONEBIT_AUDIO_THREADS=N sets it; anything that is not a positive number
// means 1 (serial, the control for measuring the split), so a typo cannot turn into "as many as the parse produced".
// Unset: the hardware's count, at most 8.
inline size_t audio_thread_count() {
    if (const char* env = std::getenv("ONEBIT_AUDIO_THREADS")) {
        const long n = std::strtol(env, nullptr, 10);
        return n > 0 ? size_t(n) : 1;
    }
    // hardware_concurrency() answers "how many CPUs does this machine have", not "how many may this process use". Under a
    // cpuset -- a thermal-run scope, a container, a systemd AllowedCPUs= -- it over-reports: measured on this box it says 32
    // while the scope allows 4, so the capped-at-8 default spawned 8 threads for 4 CPUs. The affinity mask is what the
    // scheduler enforces, so that is what this follows.
    // MEASURED SCALING (engine#1783, whisper-small encoder, 447 nodes, 32-core box, host build of main,
    // ONEBIT_AUDIO_THREADS=N, single run per count, box spread 7-8 %):
    //     threads   1        2        4        6        8
    //     encoder   11403.8  6627.8   4001.0   4467.4   4515.6  ms
    //     speed-up  1.00x    1.72x    2.85x    2.55x    2.53x
    // The knee is at 4: 6 and 8 threads are SLOWER than 4, which is what a memory-bound kernel does when more
    // workers compete for the same bandwidth and L3 rather than a work-splitting shortfall. The 29 points of
    // efficiency lost at 4 are therefore not the harness: the threaded spawn tax measured 0.018 ms/call, 2.6 %
    // of threaded time (engine#2505). Do not raise this default without a new interleaved table -- the
    // remaining lever is the GEMM's memory traffic, not more threads (73 % of the gap is MatMul).
    // What this is NOT: a measured speedup. Three interleaved repetitions at 4 CPUs (8 threads / 4 threads) gave encoder
    // 10.38/10.40 s and decode 2.58/2.69 s per token -- within the box's own noise, and my first, non-interleaved comparison
    // of 8 threads against serial (5.21 against 4.24 s/token) was load, not threading. The reason to prefer the mask is that
    // oversubscribing a cpuset cannot help and can only queue, not that it measured faster.
    size_t hw = 0;
#ifdef __linux__
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof set, &set) == 0) {
            const int allowed = CPU_COUNT(&set);
            if (allowed > 0) hw = size_t(allowed);
        }
    }
#endif
    if (hw == 0) hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : (hw > 8 ? 8 : hw);
}

struct par_overhead;
inline par_overhead & par_overhead_stats();

// Run fn(0) .. fn(units-1), each exactly once, on at most `want` threads. Worker t takes units t, t+want, t+2*want, ...
template <typename Fn>
void parallel_units(size_t units, size_t want, Fn&& fn) {
    if (want > units) want = units;
    if (want <= 1) {
        for (size_t u = 0; u < units; ++u) fn(u);
        return;
    }
    // engine#2366: instrumented because the question could not be answered any other way. Whole-model A/B runs
    // cannot resolve a 2 % effect against this box's 8-11 % run-to-run spread (measured), and a micro-benchmark
    // measures thread CREATION on an idle box rather than the cost inside a graph. So: time each threaded call and
    // each worker's own work; the summed (call - slowest_worker) is exactly what a persistent pool would remove.
    // Measured this way on a whisper decode: 16079 calls, 292.4 ms of tax, 0.018 ms/call, 2.0 % of wall.
    // It also caught a pool implementation that was 54 % SLOWER, which no wall-clock A/B would have attributed.
    const bool measure = std::getenv("ONEBIT_AUDIO_PAR_OVERHEAD") != nullptr;
    const auto t_call = std::chrono::steady_clock::now();
    std::vector<std::exception_ptr> failed(want);
    std::vector<std::thread> pool;
    std::vector<std::chrono::steady_clock::time_point> w_stop(want);
    pool.reserve(want);
    for (size_t t = 0; t < want; ++t)
        pool.emplace_back([&, t]() {
            try {
                for (size_t u = t; u < units; u += want) fn(u);
            } catch (...) {
                failed[t] = std::current_exception();
            }
            if (measure) w_stop[t] = std::chrono::steady_clock::now();
        });
    for (auto& th : pool) th.join();
    if (measure) {
        using ns = std::chrono::nanoseconds;
        const auto t_done = std::chrono::steady_clock::now();
        uint64_t work = 0;
        for (size_t t = 0; t < want; ++t) {
            if (w_stop[t].time_since_epoch().count() == 0) continue;
            const uint64_t d = (uint64_t) std::chrono::duration_cast<ns>(w_stop[t] - t_call).count();
            if (d > work) work = d;   // the slowest worker bounds how long the join could usefully wait
        }
        auto& s = par_overhead_stats();
        s.calls.fetch_add(1);
        s.ns_total.fetch_add((uint64_t) std::chrono::duration_cast<ns>(t_done - t_call).count());
        s.ns_work_max.fetch_add(work);
    }
    for (const auto& ep : failed)
        if (ep) std::rethrow_exception(ep);
}

// engine#2366: see parallel_units. One line at exit when ONEBIT_AUDIO_PAR_OVERHEAD is set; nothing is measured
// when it is unset (the branch is a single getenv at the top of each call, and the setenv result is stable).
struct par_overhead {
    std::atomic<uint64_t> calls{0}, ns_total{0}, ns_work_max{0};
    ~par_overhead() {
        if (std::getenv("ONEBIT_AUDIO_PAR_OVERHEAD") == nullptr) return;
        const uint64_t c = calls.load(), tot = ns_total.load(), work = ns_work_max.load();
        if (c == 0) return;
        const double ms_tot = double(tot) / 1e6, ms_work = double(work) / 1e6, ms_tax = ms_tot - ms_work;
        std::fprintf(stderr,
                     "[par-overhead] %llu threaded calls: total %.1f ms, slowest-worker work %.1f ms, tax %.1f ms "
                     "(%.1f%% of threaded time, %.3f ms/call)\n",
                     (unsigned long long) c, ms_tot, ms_work, ms_tax, ms_tot > 0 ? 100.0 * ms_tax / ms_tot : 0.0,
                     c ? ms_tax / double(c) : 0.0);
    }
};
inline par_overhead & par_overhead_stats() {
    static par_overhead s;
    return s;
}

template <typename Fn>
void parallel_units(size_t units, Fn&& fn) {
    parallel_units(units, audio_thread_count(), static_cast<Fn&&>(fn));
}

}  // namespace onebit::npu::pack_import
