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

// `1bit unified`'s wait for the NPU (app/npu_queue.h, engine#1024). Device-free: the whole point of the ticket is
// that the wait is bounded and that the client hears about it, and both are properties of this function alone.
#include "../app/npu_queue.h"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace {

using clock_now = std::chrono::steady_clock;

long ms_since(const clock_now::time_point& t0) {
    return long(std::chrono::duration_cast<std::chrono::milliseconds>(clock_now::now() - t0).count());
}

#define CHECK(cond)                                                                                        \
    do {                                                                                                   \
        if (!(cond)) {                                                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

}  // namespace

int main() {
    // The device is free: the lock is taken at once and nothing is sent to the client.
    {
        std::timed_mutex mu;
        int ticks = 0;
        CHECK(onebit::wait_for_device(mu, 300000, [&] { ticks++; }));
        mu.unlock();
        CHECK(ticks == 0);
    }
    // The device stays busy past the wait: the request gives up (503 + Retry-After, not a silent pause) and the
    // client was told it was queueing, once when the wait began and again on every slice.
    {
        std::timed_mutex mu;
        mu.lock();
        int ticks = 0;
        const auto t0 = clock_now::now();
        CHECK(!onebit::wait_for_device(mu, 200, [&] { ticks++; }));
        const long ms = ms_since(t0);
        mu.unlock();
        CHECK(ticks >= 2);
        CHECK(ms >= 180 && ms < 2000);
    }
    // Released while the request waits (the normal case on the appliance): the lock is taken, so the second chat
    // queueing behind a full generation is answered instead of refused.
    {
        std::timed_mutex mu;
        std::atomic<bool> held{false};
        std::thread t([&] {
            mu.lock();
            held = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            mu.unlock();
        });
        while (!held) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        int ticks = 0;
        const auto t0 = clock_now::now();
        CHECK(onebit::wait_for_device(mu, 5000, [&] { ticks++; }));
        const long ms = ms_since(t0);
        t.join();
        mu.unlock();
        CHECK(ticks >= 1);
        CHECK(ms >= 250);  // it really waited for the holder rather than taking the lock early
    }
    // --queue-ms 0 still refuses at once, and does not claim to be queueing.
    {
        std::timed_mutex mu;
        mu.lock();
        int ticks = 0;
        const auto t0 = clock_now::now();
        CHECK(!onebit::wait_for_device(mu, 0, [&] { ticks++; }));
        const long ms = ms_since(t0);
        mu.unlock();
        CHECK(ticks == 0);
        CHECK(ms < 100);
    }
    std::puts("npu_queue: ok");
    return 0;
}
