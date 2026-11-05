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

// The NPU route's load retry (engine #1010, docs/npu.md "When a load follows an evict").
//
// On the appliance a model load that starts right after an evict can die at device / context creation while the previous
// backend's hardware context is still being torn down: the c22 board's log shows two children exiting 1 within 100 ms,
// so `POST /api/v1/load` answers 500 and the user's model switch fails, while the next request loads the same model fine.
// Lemonade's half of that race is its fixed 2 s wait after "waiting for GPU driver cleanup" being a delay and not a
// readiness signal (1bit-MONSTER/lemonade). The engine's half is this: a context-creation failure was fatal on the first
// attempt, when it is exactly the kind of failure that clears up in a few hundred milliseconds.
//
// So the NPU routes open with a short, bounded retry: a transient failure becomes a normal start, a real one still fails
// — with every attempt's reason in the log, which is what the appliance could not say before engine #999.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace onebit {

struct LoadRetryPolicy {
    // Long enough to outlast a hardware-context teardown (the appliance waits 2 s before it even starts us), short
    // enough that a genuinely broken NPU still answers the load request promptly.
    double window_s = 2.5;
    double first_delay_s = 0.15;
    double max_delay_s = 0.5;
    int max_attempts = 12;   // a bound of its own: the delay schedule leaves ~6 attempts inside the window
};

// The delay before attempt `attempt` (1-based: attempt 1 is the immediate first try and sleeps nothing).
inline double load_retry_delay(const LoadRetryPolicy& p, int attempt) {
    double d = p.first_delay_s;
    for (int i = 1; i < attempt; ++i) d = std::min(d * 2.0, p.max_delay_s);
    return std::min(d, p.max_delay_s);
}

// Whether another attempt is allowed after `attempts` failures that took `elapsed_s` so far.
inline bool load_retry_more(const LoadRetryPolicy& p, int attempts, double elapsed_s) {
    if (attempts >= p.max_attempts) return false;
    return elapsed_s + load_retry_delay(p, attempts) <= p.window_s;
}

// The one load failure an operator has to act on instead of waiting out: the driver refusing to create a hardware
// context. The runtime reports it as code 102 (`errc::ctx_create`, npu_rt_errors.h) with the ioctl's errno in the
// message, and "No such device" (ENODEV) is the state a TDR leaves behind when it could not restart a context: amdxdna
// keeps refusing every new context until it is reloaded (measured on strixhalo 2026-11-04, the wedge in engine #1051).
// A retry cannot fix that, and neither can the supervisor that restarts us -- so say what does.
inline bool npu_contexts_refused(const std::string& reason) {
    return reason.find("code 102") != std::string::npos && reason.find("No such device") != std::string::npos;
}

inline const char* npu_contexts_refused_hint() {
    return "the NPU array is refusing hardware contexts: reload the driver (modprobe -r amdxdna && modprobe amdxdna)";
}

// Run `open` until it succeeds, the window closes, or it has thrown `max_attempts` times. A caller that wants the log
// silent can pass a no-op `report`; the default names each attempt with its own reason, which is the whole point.
template <class Open, class Report, class Sleep>
void open_with_retry(const LoadRetryPolicy& p, const char* what, Open&& open, Report&& report, Sleep&& sleep) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (int attempt = 1;; ++attempt) {
        try {
            open();
            if (attempt > 1)
                report("recovered on attempt " + std::to_string(attempt) + " after " +
                       std::to_string(std::chrono::duration<double, std::milli>(clock::now() - t0).count()) + " ms");
            return;
        } catch (const std::exception& ex) {
            const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
            if (!load_retry_more(p, attempt, elapsed)) {
                report(std::string(what) + " failed " + std::to_string(attempt) + " attempt(s) in " +
                       std::to_string(int(elapsed * 1000)) + " ms, last reason: " + ex.what());
                if (npu_contexts_refused(ex.what())) report(npu_contexts_refused_hint());
                throw;   // the caller's own error path; the reasons above are the evidence the appliance lacked
            }
            const double delay = load_retry_delay(p, attempt);
            report(std::string(what) + " attempt " + std::to_string(attempt) + " failed after " +
                   std::to_string(int(elapsed * 1000)) + " ms (" + ex.what() + "); retrying in " +
                   std::to_string(int(delay * 1000)) + " ms");
            sleep(delay);
        }
    }
}

// The two-argument form the routes use: report to stderr, sleep for real.
template <class Open>
void open_with_retry(const LoadRetryPolicy& p, const char* what, Open&& open) {
    open_with_retry(p, what, std::forward<Open>(open),
                    [](const std::string& line) { std::fprintf(stderr, "1bit unified: %s\n", line.c_str()); },
                    [](double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); });
}

}  // namespace onebit
