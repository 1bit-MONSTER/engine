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
#include <vector>

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
    // engine#3318: DeviceSlots, the admission when a model serves several requests at once.
    // Capacity 1 behaves as the timed mutex it replaces: a second taker is refused at once and waits like one.
    {
        onebit::DeviceSlots s(1);
        CHECK(s.try_lock());
        CHECK(!s.try_lock());
        int ticks = 0;
        CHECK(!onebit::wait_for_device(s, 0, [&] { ticks++; }));
        CHECK(ticks == 0);
        CHECK(!onebit::wait_for_device(s, 150, [&] { ticks++; }));
        CHECK(ticks >= 1);
        s.unlock();
        CHECK(s.in_use() == 0);
        CHECK(onebit::wait_for_device(s, 0, nullptr));
        s.unlock();
    }
    // Capacity 3: three requests are admitted at once, without a tick (nobody queued); the fourth queues and is
    // admitted the moment one of the three leaves.
    {
        onebit::DeviceSlots s(3);
        int ticks = 0;
        for (int i = 0; i < 3; i++) CHECK(onebit::wait_for_device(s, 5000, [&] { ticks++; }));
        CHECK(ticks == 0);
        CHECK(s.in_use() == 3);
        std::thread t([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            s.unlock();
        });
        const auto t0 = clock_now::now();
        CHECK(onebit::wait_for_device(s, 5000, [&] { ticks++; }));
        const long ms = ms_since(t0);
        t.join();
        CHECK(ticks >= 1);
        CHECK(ms >= 250 && ms < 3000);
        CHECK(s.in_use() == 3);
        for (int i = 0; i < 3; i++) s.unlock();
        CHECK(s.in_use() == 0);
    }
    // Many threads against capacity 2: never more than two inside at once, and every one of them gets in.
    {
        onebit::DeviceSlots s(2);
        std::atomic<int> inside{0}, peak{0}, served{0};
        std::vector<std::thread> ts;
        for (int i = 0; i < 8; i++)
            ts.emplace_back([&] {
                if (!onebit::wait_for_device(s, 10000, nullptr)) return;
                const int n = ++inside;
                int p = peak.load();
                while (n > p && !peak.compare_exchange_weak(p, n)) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                --inside;
                served++;
                s.unlock();
            });
        for (auto& t : ts) t.join();
        CHECK(served == 8);
        CHECK(peak <= 2);
        CHECK(peak == 2);   // and the second place was really used
    }
    // Raising the capacity at load wakes a waiter (the serve sets it once the model says how many it serves).
    {
        onebit::DeviceSlots s(1);
        CHECK(s.try_lock());
        std::thread t([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            s.set_capacity(2);
        });
        CHECK(onebit::wait_for_device(s, 5000, nullptr));
        t.join();
        CHECK(s.in_use() == 2);
        s.unlock();
        s.unlock();
        bool threw = false;
        try { s.set_capacity(0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    std::puts("npu_queue: ok");
    return 0;
}
