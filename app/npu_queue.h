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

// `1bit unified`'s wait for the NPU (engine#1024).
//
// The NPU route serves one generation at a time, so a second request waits on the engine's mutex. That wait used to
// be 5 s and then a 503, because a *silent* 10 s wait is what Lemonade's streaming proxy reported as a hung backend
// (engine#911: "0 chunks in 5.9 s", "Backend did not send [DONE] marker"). On a single-user appliance the first
// request is routinely a whole generation, so the wait has to cover one - and the client has to be told, every
// `kQueueTickMs`, that the request is queueing rather than stuck. `tick` is where a stream puts that byte on the wire.
#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>

namespace onebit {

// How often a queued request says it is still waiting. One second is well inside every proxy's no-data timer that
// matters here (Lemonade's is ~6 s) and costs a queued generation ~60 bytes a minute.
constexpr int kQueueTickMs = 1000;

// Wait up to `total_ms` for `mu`. Returns true holding the lock. `tick` runs once when the wait starts (the request
// found the device busy) and after every `kQueueTickMs` slice, so a caller can keep its client informed; it does not
// run when the device is free, and it does not run at all for `total_ms <= 0` (refuse at once).
inline bool wait_for_device(std::timed_mutex& mu, int total_ms, const std::function<void()>& tick) {
    if (mu.try_lock()) return true;
    if (total_ms <= 0) return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(total_ms);
    if (tick) tick();
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) return false;
        if (mu.try_lock_for(std::chrono::milliseconds(std::min<long long>(kQueueTickMs, left)))) return true;
        if (tick) tick();
    }
}

}  // namespace onebit
