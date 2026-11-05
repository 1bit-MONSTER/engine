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
    // A *refused context* (code 102/104 with ENODEV) is a queue, not a fault: with the per-client partition fix
    // (engine #1085, orchestrator decision 2026-11-05 item 2) a second client is refused with the same code a dead
    // array answers, and the only way to tell them apart is time -- a holder frees the array when it exits, a dead
    // array never does. So wait up to this budget before declaring it dead; `1bit serve --load-queue-ms` sets it.
    double refusal_window_s = 45.0;
    double refusal_max_delay_s = 2.0;
    int refusal_max_attempts = 60;
};

// The delay before attempt `attempt` (1-based: attempt 1 is the immediate first try and sleeps nothing).
inline double load_retry_delay(const LoadRetryPolicy& p, int attempt, double max_delay_s) {
    double d = p.first_delay_s;
    for (int i = 1; i < attempt; ++i) d = std::min(d * 2.0, max_delay_s);
    return std::min(d, max_delay_s);
}

inline double load_retry_delay(const LoadRetryPolicy& p, int attempt) {
    return load_retry_delay(p, attempt, p.max_delay_s);
}

// Whether another attempt is allowed after `attempts` failures that took `elapsed_s` so far.
inline bool load_retry_more(const LoadRetryPolicy& p, int attempts, double elapsed_s, double window_s,
                            double max_delay_s, int max_attempts) {
    if (attempts >= max_attempts) return false;
    return elapsed_s + load_retry_delay(p, attempts, max_delay_s) <= window_s;
}

inline bool load_retry_more(const LoadRetryPolicy& p, int attempts, double elapsed_s) {
    return load_retry_more(p, attempts, elapsed_s, p.window_s, p.max_delay_s, p.max_attempts);
}

// The one NPU failure an operator has to act on instead of waiting out: the driver deciding the array is dead. The
// runtime reports it with the ioctl's errno in the message -- "No such device" (ENODEV) -- and the code says which call
// was refused: 102 `errc::ctx_create` (`aie2_message.c`) or 104 `errc::submit` (`aie2_ctx.c`). Both are the state a TDR
// leaves behind when it could not restart a context, and amdxdna keeps refusing until it is reloaded (measured on
// strixhalo 2026-11-04, the wedge in engine #1051). A retry cannot fix that, and neither can the supervisor that
// restarts us -- so say what does.
inline bool npu_contexts_refused(const std::string& reason) {
    if (reason.find("No such device") == std::string::npos) return false;
    // 102 ctx_create and 104 submit: the driver refuses BOTH once hw_dead is latched (aie2_message.c's create path and
    // aie2_ctx.c's submit path both return -ENODEV), and which one you see depends only on whether this process already
    // had a context when the array died -- a backend that was loaded fails at submit, a fresh one at create.
    return reason.find("code 102") != std::string::npos || reason.find("code 104") != std::string::npos;
}

inline const char* npu_contexts_refused_hint() {
    return "the NPU refuses hardware contexts: another client may hold the array (wait for it, or stop it); if nothing "
           "else is serving, reload the driver (modprobe -r amdxdna && modprobe amdxdna)";
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
            // A refused context is queued (another client may own the array); every other failure keeps the short
            // window that exists for the evict-then-load race.
            const bool queued = npu_contexts_refused(ex.what());
            const double window = queued ? p.refusal_window_s : p.window_s;
            const double max_delay = queued ? p.refusal_max_delay_s : p.max_delay_s;
            const int max_attempts = queued ? p.refusal_max_attempts : p.max_attempts;
            const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
            if (!load_retry_more(p, attempt, elapsed, window, max_delay, max_attempts)) {
                report(std::string(what) + " failed " + std::to_string(attempt) + " attempt(s) in " +
                       std::to_string(int(elapsed * 1000)) + " ms" +
                       (queued ? " over a " + std::to_string(int(window * 1000)) + " ms queue window" : "") +
                       ", last reason: " + ex.what());
                if (queued) report(npu_contexts_refused_hint());
                throw;   // the caller's own error path; the reasons above are the evidence the appliance lacked
            }
            const double delay = load_retry_delay(p, attempt, max_delay);
            report(std::string(what) + " attempt " + std::to_string(attempt) + " failed after " +
                   std::to_string(int(elapsed * 1000)) + " ms (" + ex.what() + "); retrying in " +
                   std::to_string(int(delay * 1000)) + " ms" +
                   (queued ? " (queued: another NPU client may hold the array)" : ""));
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
