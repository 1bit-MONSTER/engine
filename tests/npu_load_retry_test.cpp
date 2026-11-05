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

// npu_load_retry_test: the bounded load retry of app/npu_load_retry.h (engine #1010). On the appliance a model load
// started right after an evict can die at device / context creation while the previous backend's hardware context is
// still being torn down — two children exiting 1 within 100 ms, `POST /api/v1/load` 500, and the next request loads the
// same model fine. The retry turns the transient case into a normal start and keeps every attempt's reason in the log.
//
// Device-free and model-free: the "open" is a functor that throws what the NPU path throws, and the clock is real but
// the policy windows here are milliseconds, so this runs in the ordinary test suite.
#include "npu_load_retry.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void expect(bool cond, const std::string& what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

int main() {
    using onebit::LoadRetryPolicy;
    using onebit::load_retry_delay;
    using onebit::load_retry_more;
    using onebit::open_with_retry;

    // 1. The delay schedule: immediate first attempt, then 150 ms doubling to a 500 ms cap.
    {
        LoadRetryPolicy p;
        const std::vector<double> want{0.15, 0.30, 0.50, 0.50, 0.50};
        for (size_t i = 0; i < want.size(); ++i) {
            const double got = load_retry_delay(p, int(i) + 1);
            expect(std::abs(got - want[i]) < 1e-9,
                   "delay before attempt " + std::to_string(i + 2) + " is " + std::to_string(got));
        }
    }

    // 2. The window holds: attempts fit inside 2.5 s, and nothing is retried once it would run past it.
    {
        LoadRetryPolicy p;
        double elapsed = 0;
        int attempts = 1;
        while (load_retry_more(p, attempts, elapsed)) {
            elapsed += load_retry_delay(p, attempts);
            ++attempts;
        }
        expect(attempts >= 4 && attempts <= p.max_attempts, "attempts inside the 2.5 s window: " + std::to_string(attempts));
        expect(elapsed <= p.window_s, "the schedule never runs past the window");
        expect(!load_retry_more(p, attempts, p.window_s), "a closed window allows no further attempt");
        expect(!load_retry_more(p, p.max_attempts, 0.0), "the attempt cap holds even inside the window");
    }

    // 3. A transient failure is retried and the caller sees a success (the appliance's real case: 2 failures, then ok).
    {
        LoadRetryPolicy p;
        p.window_s = 0.25;
        p.first_delay_s = 0.01;
        p.max_delay_s = 0.02;
        int calls = 0;
        bool opened = false;
        std::vector<std::string> said;
        open_with_retry(p, "opening the private route's NPU context",
                        [&] {
                            ++calls;
                            if (calls <= 2) throw std::runtime_error("npu runtime: device error (code 102): No such device");
                            opened = true;
                        },
                        [&](const std::string& line) { said.push_back(line); },
                        [](double) {});
        expect(opened, "opened after the transient failures");
        expect(calls == 3, "three attempts (2 failures + 1 success), got " + std::to_string(calls));
        expect(said.size() == 3, "one log line per attempt, got " + std::to_string(said.size()));
        expect(!said.empty() && said[0].find("No such device") != std::string::npos,
               "the first line carries the attempt's own reason");
        expect(!said.empty() && said.back().find("recovered on attempt 3") != std::string::npos,
               "the last line says it recovered and on which attempt");
    }

    // 4. A real failure still fails, and the error says how many attempts and why (the appliance's missing sentence).
    {
        LoadRetryPolicy p;
        p.window_s = 0.05;
        p.first_delay_s = 0.01;
        p.max_delay_s = 0.02;
        int calls = 0;
        std::string last_report, thrown;
        try {
            open_with_retry(p, "opening the layer lane's NPU context",
                            [&] {
                                ++calls;
                                throw std::runtime_error("the driver refused to create a context");
                            },
                            [&](const std::string& line) { last_report = line; },
                            [](double) {});
        } catch (const std::exception& ex) {
            thrown = ex.what();
        }
        expect(calls > 1, "a persistent failure is retried before giving up, attempts " + std::to_string(calls));
        expect(thrown == "the driver refused to create a context", "the caller's error is the last attempt's reason");
        expect(last_report.find("failed") != std::string::npos && last_report.find("attempt(s)") != std::string::npos,
               "the final report names the attempt count: " + last_report);
        expect(last_report.find("the driver refused") != std::string::npos, "and the reason: " + last_report);
    }

    // 5. The one load failure that needs an operator action: the driver refusing a hardware context (ENODEV after a TDR
    //    the driver could not recover). The give-up report must say what to do, because no retry and no supervisor
    //    restart can fix it.
    {
        const onebit::LoadRetryPolicy p0;
        expect(onebit::npu_contexts_refused("npu runtime: device error (code 102): No such device"),
               "the measured ENODEV context-creation message is recognised");
        expect(!onebit::npu_contexts_refused("npu runtime: invalid call (code 614): state 8, run 0 of 28"),
               "a failed runlist (614) is not a refused context");
        expect(!onebit::npu_contexts_refused("npu runtime: device error (code 102): Cannot allocate memory"),
               "a different errno on the same code is not the reload case");
        expect(onebit::npu_contexts_refused("npu runtime: device error (code 104): No such device"),
               "a submit refused by a dead array is the same state as a refused context creation");
        expect(!onebit::npu_contexts_refused("npu runtime: device error (code 104): Input/output error"),
               "a submit that failed with an ordinary errno is not the dead-array case");
        expect(std::string(onebit::npu_contexts_refused_hint()).find("modprobe") != std::string::npos,
               "the hint names the one action that works");

        LoadRetryPolicy p;
        p.window_s = 0.05;
        p.first_delay_s = 0.01;
        p.max_delay_s = 0.02;
        std::vector<std::string> said;
        try {
            open_with_retry(p, "opening",
                            [] { throw std::runtime_error("npu runtime: device error (code 102): No such device"); },
                            [&](const std::string& line) { said.push_back(line); }, [](double) {});
        } catch (const std::exception&) {
        }
        bool hinted = false;
        for (const auto& line : said)
            if (line.find("modprobe -r amdxdna") != std::string::npos) hinted = true;
        expect(hinted, "the give-up path prints the operator hint for a refused context");
    }

    // 6. A zero window means one attempt: an operator who wants no retry gets none.
    {
        LoadRetryPolicy p;
        p.window_s = 0.0;
        int calls = 0;
        try {
            open_with_retry(p, "opening", [&] { ++calls; throw std::runtime_error("x"); },
                            [](const std::string&) {}, [](double) {});
        } catch (const std::exception&) {
        }
        expect(calls == 1, "no retry with a zero window, attempts " + std::to_string(calls));
    }

    if (failures == 0) std::printf("npu_load_retry_test: PASS\n");
    return failures == 0 ? 0 : 1;
}
