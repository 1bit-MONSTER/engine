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
// tests/audio_launch_plan_test.cpp -- the launch loop's routing and its ACCOUNTING (npu/forward/include/audio_launch_plan.h,
// engine#1783), driven against a fake target and a fake host: no device, no container, no kernels.
//
// The report is the measurement the row is scored on (`device_gemm=<X> s + host_remainder=<Y> s`), so the arms here are about
// where a launch is COUNTED, not only whether it ran: a launch that falls back must land in the host column with the device's
// own reason beside it, and a launch both paths refuse must land in neither column -- counting it as host would quietly move an
// unexplained projection into the number the route is judged by.

#include <cstdio>
#include <string>
#include <vector>

#include "audio_launch_plan.h"
using namespace onebit::audio;
namespace t = ::npu::target;
static int failures = 0;
static void check(bool ok, const std::string& what) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str()); if (!ok) ++failures; }
struct FakeTarget : t::NpuTarget {
    t::Caps c; int calls = 0; bool refuse_all = false;
    const char* name() const override { return "fake"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override { return true; }
    bool load_block(int, int, std::string&) override { return true; }
    t::Caps caps() const override { return c; }
    bool supports(t::Op op) const override { return op == t::Op::gemv; }
    bool gemv(const t::GemvArgs& a, std::vector<float>& out, std::string& err) override {
        ++calls;
        if (refuse_all) { err = "no kernel for this shape"; return false; }
        out.assign(size_t(a.M) * size_t(a.N), 1.0f);
        return true;
    }
};
int main() {
    std::vector<float> x(128 * 768, 1.0f);
    std::vector<LaunchSpec> specs = {
        {0, 1, 0, 768, 768, 128, "O"},
        {1, 0, 0, 768, 3072, 128, "GU"},
        {2, 2, 0, 3072, 768, 128, "D"},
    };
    auto input = [&](const LaunchSpec&) { return x.data(); };
    int host_calls = 0;
    auto host = [&](const LaunchSpec&, std::vector<float>*, std::string*) { ++host_calls; return true; };

    FakeTarget ft; ft.c.max_rows = 16; ft.c.prefill_rows = 128;
    auto r = run_launches(&ft, specs, input, host);
    check(r.ok() && r.device_launches == 3 && r.host_launches == 0, "with a target, every launch goes to the device");
    check(ft.calls == 3 && host_calls == 0, "...the device was called three times and the host not at all");
    check(r.per_launch[0].device && r.per_launch[2].device, "...and each launch records which column it went in");

    ft.refuse_all = true; host_calls = 0;
    auto r2 = run_launches(&ft, specs, input, host);
    check(r2.ok() && r2.host_launches == 3 && r2.device_launches == 0, "a target that refuses every shape falls back to the host");
    check(r2.per_launch[1].why == "no kernel for this shape", "...keeping the target's own reason per launch");
    check(host_calls == 3, "...the host ran all three");

    ft.refuse_all = false; host_calls = 0;
    auto r3 = run_launches(nullptr, specs, input, host);
    check(r3.ok() && r3.host_launches == 3, "no target at all is the host path, counted as such");
    check(r3.per_launch[0].why == "no target", "...and says why it is not on the device");

    auto bad_host = [&](const LaunchSpec&, std::vector<float>*, std::string* why) { *why = "host GEMM refused"; return false; };
    auto r4 = run_launches(nullptr, specs, input, bad_host);
    check(!r4.ok() && r4.failed == 3 && r4.device_launches == 0 && r4.host_launches == 0,
          "a launch both paths refuse counts in NEITHER column");
    check(r4.failures.size() == 3 && r4.failures[0].find("launch 0 O") == 0 && r4.failures[0].find("host GEMM refused") != std::string::npos,
          "...and the failure names the launch, its tag and both reasons");
    // The close condition's own line, built from a report rather than typed. A split with a silent third category (failed) is a
    // split that adds up to less than the run, so the failure count is part of the string.
    {
        ft.refuse_all = true;                                     // every launch falls back to the host
        auto r5 = run_launches(&ft, specs, input, host);
        const std::string line = format_split_run(r5, 163.0, 2050.0);
        check(line == "device_gemm=0.163 s + host_remainder=2.050 s per window (0 device launch(es), 3 host, 0 failed)",
              "the split line reads exactly as the close condition asks: " + line);
        auto r6 = run_launches(nullptr, specs, input, bad_host);
        const std::string bad = format_split_run(r6, 0.0, 0.0);
        check(bad.find("3 failed") != std::string::npos, "...and a run whose launches all failed says so: " + bad);
    }
    std::printf(failures ? "audio launch plan: %d FAILED\n" : "audio launch plan: all cases ok\n", failures);
    return failures ? 1 : 0;
}
