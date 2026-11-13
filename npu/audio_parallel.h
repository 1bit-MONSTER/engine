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

// Run fn(0) .. fn(units-1), each exactly once, on at most `want` threads. Worker t takes units t, t+want, t+2*want, ...
template <typename Fn>
void parallel_units(size_t units, size_t want, Fn&& fn) {
    if (want > units) want = units;
    if (want <= 1) {
        for (size_t u = 0; u < units; ++u) fn(u);
        return;
    }
    std::vector<std::exception_ptr> failed(want);
    std::vector<std::thread> pool;
    pool.reserve(want);
    for (size_t t = 0; t < want; ++t)
        pool.emplace_back([&, t]() {
            try {
                for (size_t u = t; u < units; u += want) fn(u);
            } catch (...) {
                failed[t] = std::current_exception();
            }
        });
    for (auto& th : pool) th.join();
    for (const auto& ep : failed)
        if (ep) std::rethrow_exception(ep);
}

template <typename Fn>
void parallel_units(size_t units, Fn&& fn) {
    parallel_units(units, audio_thread_count(), static_cast<Fn&&>(fn));
}

}  // namespace onebit::npu::pack_import
