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
// stream_bridge_test: app/stream_bridge.h, the hand-off between the thread that generates a streamed reply and the
// thread that writes it (engine#1785).  No server, no model, no device: two threads and a clock.

#include "stream_bridge.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using onebit::npu::StreamBridge;
using clk = std::chrono::steady_clock;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main() {
    // 1. The point of the bridge: a frame is readable when it is pushed, not when the producer finishes.
    {
        StreamBridge b;
        const auto t0 = clk::now();
        std::thread producer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            b.push("one");
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            b.push("two");
            b.finish();
        });
        const bool started = b.wait_first();
        const double first_ms = ms_since(t0);
        std::string f;
        std::vector<std::string> got;
        std::vector<double> at;
        while (b.next(f)) {
            got.push_back(f);
            at.push_back(ms_since(t0));
        }
        producer.join();
        check(started, "wait_first reports a stream once a frame exists");
        check(first_ms < 300, "wait_first returns at the first frame, not at the end (" + std::to_string(first_ms) + " ms of ~450)");
        check(got == std::vector<std::string>({"one", "two"}), "frames arrive in order, each once");
        check(at.size() == 2 && at[0] < 300 && at[1] >= 400, "the first frame is taken well before the second exists");
        check(b.finished() && b.started(), "finished and started are both true at the end");
    }
    // 2. A producer that ends without a frame: no stream, so the caller can still answer with a status code.
    {
        StreamBridge b;
        std::thread producer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            b.finish();
        });
        check(!b.wait_first(), "wait_first is false when the producer ended with no frame");
        std::string f;
        check(!b.next(f) && !b.started(), "and there is nothing to read");
        producer.join();
    }
    // 3. Frames pushed before anyone waits are kept, and the end is seen only after they are drained.
    {
        StreamBridge b;
        b.push("a");
        b.push("b");
        b.finish();
        b.push("late");   // after finish: ignored
        std::string f;
        check(b.wait_first() && b.next(f) && f == "a" && b.next(f) && f == "b" && !b.next(f),
              "queued frames survive finish() and a push after finish() is dropped");
    }
    // 4. cancel(): the consumer stops, a blocked next() returns, and the producer is told.
    {
        StreamBridge b;
        std::atomic<bool> producer_saw{false};
        std::thread producer([&] {
            b.push("x");
            for (int i = 0; i < 200 && !b.cancelled(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            producer_saw = b.cancelled();
            check(!b.push("after cancel"), "push reports the cancellation to the producer");
            b.finish();
        });
        std::string f;
        check(b.next(f) && f == "x", "the frame before the cancel is delivered");
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            b.cancel();
        });
        const auto t0 = clk::now();
        check(!b.next(f), "next() returns false once cancelled");
        check(ms_since(t0) < 500, "and a next() that was blocked is released by the cancel");
        canceller.join();
        producer.join();
        check(producer_saw.load(), "the producer sees cancelled()");
    }
    // 5. Many frames from a fast producer: none lost, none repeated, order kept.
    {
        StreamBridge b;
        const int n = 20000;
        std::thread producer([&] {
            for (int i = 0; i < n; i++) b.push(std::to_string(i));
            b.finish();
        });
        std::string f;
        int expect = 0;
        bool ordered = true;
        while (b.next(f)) {
            ordered = ordered && f == std::to_string(expect);
            ++expect;
        }
        producer.join();
        check(ordered && expect == n, "20000 frames cross in order with none lost (" + std::to_string(expect) + ")");
    }
    if (failures == 0) std::printf("stream_bridge: ok\n");
    return failures == 0 ? 0 : 1;
}
