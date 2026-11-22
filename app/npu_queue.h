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
// The NPU route serves one generation at a time (or, since engine#3318, as many as its model's max_concurrent()), so a
// request past that waits on the engine's admission. That wait used to
// be 5 s and then a 503, because a *silent* 10 s wait is what Lemonade's streaming proxy reported as a hung backend
// (engine#911: "0 chunks in 5.9 s", "Backend did not send [DONE] marker"). On a single-user appliance the first
// request is routinely a whole generation, so the wait has to cover one - and the client has to be told, every
// `kQueueTickMs`, that the request is queueing rather than stuck. `tick` is where a stream puts that byte on the wire.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>

namespace onebit {

// How often a queued request says it is still waiting. One second is well inside every proxy's no-data timer that
// matters here (Lemonade's is ~6 s) and costs a queued generation ~60 bytes a minute.
constexpr int kQueueTickMs = 1000;

// engine#3318: the NPU route's admission when a model serves several requests at once (a private route's
// PrivateSession::max_concurrent(), e.g. dx's decoder slots). A counting lock with the timed mutex's member names, so
// wait_for_device() takes either: try_lock() takes one of `capacity` places, unlock() gives it back and wakes one
// waiter. Capacity 1 is the timed mutex this replaces, exactly.
class DeviceSlots {
public:
    explicit DeviceSlots(int capacity = 1) { set_capacity(capacity); }
    DeviceSlots(const DeviceSlots&) = delete;
    DeviceSlots& operator=(const DeviceSlots&) = delete;

    // Set before the serve answers its first request (the model decides it at load).
    void set_capacity(int capacity) {
        if (capacity < 1) throw std::invalid_argument("DeviceSlots: capacity must be at least 1");
        std::lock_guard<std::mutex> g(mu_);
        capacity_ = capacity;
        cv_.notify_all();
    }
    int capacity() const {
        std::lock_guard<std::mutex> g(mu_);
        return capacity_;
    }
    int in_use() const {
        std::lock_guard<std::mutex> g(mu_);
        return used_;
    }
    bool try_lock() {
        std::lock_guard<std::mutex> g(mu_);
        if (used_ >= capacity_) return false;
        used_++;
        return true;
    }
    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& d) {
        std::unique_lock<std::mutex> g(mu_);
        if (!cv_.wait_for(g, d, [this] { return used_ < capacity_; })) return false;
        used_++;
        return true;
    }
    void unlock() {
        std::lock_guard<std::mutex> g(mu_);
        if (used_ > 0) used_--;
        cv_.notify_one();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    int capacity_ = 1, used_ = 0;
};

// Wait up to `total_ms` for `mu` (a std::timed_mutex, or DeviceSlots for one of its places). Returns true holding it.
// `tick` runs once when the wait starts (the request found the device busy) and after every `kQueueTickMs` slice, so a
// caller can keep its client informed; it does not run when the device is free, and it does not run at all for
// `total_ms <= 0` (refuse at once).
template <class TimedLockable>
inline bool wait_for_device(TimedLockable& mu, int total_ms, const std::function<void()>& tick) {
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
