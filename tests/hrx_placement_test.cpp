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

// engine #390's load-time placement report (app/serve_metrics.h): the line ggml-hrx's placement
// observer prints goes in, serve's own line and the --strict-device refusal come out. No GPU, no
// backend, no model: the input is the observer's text, which the fork's ggml-hrx.cpp writes.
#include "hrx_strict.h"
#include "serve_metrics.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// the observer's own wording: "%s: placement: %d nodes, %lld off HRX (%s), %d graph splits"
static const char* kOffHrx =
    "placement_observer: placement: 512 nodes, 34 off HRX (MUL_MAT x 30, GET_ROWS x 4), 7 graph splits";
static const char* kAllOnHrx = "placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits";

int main() {
    {
        // every node on HRX: the report says so, and nothing refuses
        onebit::BackendLogFacts f;
        f.feed(kAllOnHrx);
        expect(f.placement_lines == 1, "clean: the line is a placement report");
        expect(f.placement_nodes && *f.placement_nodes == 512, "clean: 512 nodes");
        expect(f.placement_splits && *f.placement_splits == 1, "clean: 1 graph split");
        expect(f.cpu_fallback_total && *f.cpu_fallback_total == 0, "clean: 0 off HRX (not the split count)");
        expect(f.cpu_fallback_ops.empty(), "clean: no op types");
        expect(onebit::placement_summary(f) == "all 512 nodes on HRX, 1 graph splits",
               "clean: summary " + onebit::placement_summary(f));
    }
    {
        // ops on the CPU: counts per op type, the total and the splits
        onebit::BackendLogFacts f;
        f.feed(kOffHrx);
        expect(f.placement_lines == 1, "off: one placement report");
        expect(f.placement_nodes && *f.placement_nodes == 512, "off: 512 nodes");
        expect(f.placement_splits && *f.placement_splits == 7, "off: 7 graph splits");
        expect(f.cpu_fallback_total && *f.cpu_fallback_total == 34, "off: 34 nodes off HRX");
        expect(f.cpu_fallback_ops.size() == 2 && f.cpu_fallback_ops.at("MUL_MAT") == 30 &&
                   f.cpu_fallback_ops.at("GET_ROWS") == 4,
               "off: 30 MUL_MAT and 4 GET_ROWS");
        const std::string summary = onebit::placement_summary(f);
        expect(summary == "34 of 512 nodes off HRX (GET_ROWS x 4, MUL_MAT x 30), 7 graph splits",
               "off: summary " + summary);
        const std::string why = onebit::placement_refusal(f);
        expect(why.rfind("--strict-device: ", 0) == 0, "off: refusal names the flag " + why);
        expect(why.find("34 of 512") != std::string::npos && why.find("refusing to start") != std::string::npos,
               "off: refusal counts " + why);
    }
    {
        // lines that are not the report change nothing
        onebit::BackendLogFacts f;
        for (const char* line : {"llama_model_loader: loaded meta data with 32 key-value pairs",
                                 "placement_observer: placement: nothing to say",
                                 "srv  update_slots: all slots are idle"})
            f.feed(line);
        expect(f.placement_lines == 0 && !f.placement_nodes && !f.cpu_fallback_total,
               "other lines: no placement report");
        // an older/other backend's line with only a total still counts for the metric
        f.feed("CPU fallback: 12 ops");
        expect(f.cpu_fallback_total && *f.cpu_fallback_total == 12, "total-only line: 12");
    }
    {
        // serve prints the report once per worse graph, and the accessor is what --strict-device reads
        onebit::ServeMetrics metrics("hrx");
        expect(!metrics.placement(), "serve: no report before the backend speaks");
        metrics.backend_line("srv  load_model: loading model");
        expect(!metrics.placement(), "serve: an ordinary line is not a report");
        metrics.backend_line(kOffHrx);
        const auto report = metrics.placement();
        expect(report && report->cpu_fallback_total.value_or(0) == 34, "serve: the report is kept");
        expect(report && onebit::placement_summary(*report) ==
                             "34 of 512 nodes off HRX (GET_ROWS x 4, MUL_MAT x 30), 7 graph splits",
               "serve: the report is formatted for the log");
    }
    {
        // a clean report is not a refusal, however few nodes it names
        onebit::ServeMetrics metrics("hrx");
        metrics.backend_line(kAllOnHrx);
        const auto report = metrics.placement();
        expect(report && report->cpu_fallback_total.value_or(-1) == 0, "serve: clean graph, 0 off HRX");
        // and the metric renders the per-op series from the same numbers
        onebit::Lifecycle life("test-hrx-placement");
        const std::string text = metrics.render(life);
        expect(text.find("onebit_cpu_fallback_ops{op=\"all\"} 0") != std::string::npos,
               "metrics: onebit_cpu_fallback_ops{op=\"all\"} 0");
    }

    {
        // --placement-report now sees every graph (GGML_HRX_REPORT=2): the warmup graph is clean, the
        // multi-sequence graph is not, and the worst graph is what serve prints and --strict-device
        // refuses on (census 2026-10-20, finding 1: LFM2 --parallel 4, ZAYA1-8B).
        onebit::BackendLogFacts f;
        f.feed(kAllOnHrx);   // warmup
        expect(onebit::placement_summary(f) == "all 512 nodes on HRX, 1 graph splits", "warmup is clean");
        f.feed("placement_observer: placement: 354 nodes, 20 off HRX (MUL_MAT x 20), 5 graph splits");
        expect(f.cpu_fallback_total && *f.cpu_fallback_total == 20, "par4: 20 off HRX");
        expect(f.placement_nodes && *f.placement_nodes == 354, "par4: 354 nodes, not the warmup's 512");
        expect(f.placement_splits && *f.placement_splits == 5, "par4: 5 graph splits");
        const std::string worst = onebit::placement_summary(f);
        expect(worst == "20 of 354 nodes off HRX (MUL_MAT x 20), 5 graph splits, worst of 2 graphs",
               "par4: " + worst);
        expect(onebit::placement_refusal(f).find("20 of 354") != std::string::npos,
               "par4: --strict-device refuses on the worst graph");
        // a later clean graph does not hide the one that is not
        f.feed(kAllOnHrx);
        expect(f.cpu_fallback_total && *f.cpu_fallback_total == 20, "a later clean graph keeps the worst");
        expect(f.placement_lines == 3, "three reports seen");
    }

    {
        // strict placement is the default on HRX (app/hrx_strict.h); the opt-out restores the fallback
        using onebit::StrictDevice;
        auto has = [](const std::vector<std::string>& env, const std::string& kv) {
            return std::find(env.begin(), env.end(), kv) != env.end();
        };
        expect(onebit::strict_for(StrictDevice::Default, "hrx"), "default: strict on hrx");
        expect(!onebit::strict_for(StrictDevice::Default, "cpu"), "default: not strict on cpu");
        expect(!onebit::strict_for(StrictDevice::Off, "hrx"), "--no-strict-device: not strict on hrx");
        expect(onebit::strict_for(StrictDevice::On, "hrx"), "--strict-device: strict on hrx");

        const auto def = onebit::hrx_placement_env(StrictDevice::Default, false);
        expect(has(def, "GGML_HRX_STRICT=1") && has(def, "GGML_HRX_REPORT=1"),
               "default env: GGML_HRX_STRICT=1, first-graph report");
        expect(!onebit::hrx_placement_verbose(StrictDevice::Default, false), "default: no -lv 4");
        const auto off = onebit::hrx_placement_env(StrictDevice::Off, false);
        expect(!has(off, "GGML_HRX_STRICT=1") && has(off, "GGML_HRX_REPORT=1"), "opt-out env: no GGML_HRX_STRICT");
        const auto on = onebit::hrx_placement_env(StrictDevice::On, false);
        expect(has(on, "GGML_HRX_STRICT=1") && has(on, "GGML_HRX_REPORT=2"), "--strict-device env: strict, every graph");
        expect(onebit::hrx_placement_verbose(StrictDevice::On, false), "--strict-device: -lv 4");
        const auto off_report = onebit::hrx_placement_env(StrictDevice::Off, true);
        expect(!has(off_report, "GGML_HRX_STRICT=1") && has(off_report, "GGML_HRX_REPORT=2") &&
                   onebit::hrx_placement_verbose(StrictDevice::Off, true),
               "--no-strict-device --placement-report: counts, no abort");

        expect(onebit::strict_from_env(nullptr) == StrictDevice::Default, "ONEBIT_STRICT_DEVICE unset: default");
        expect(onebit::strict_from_env("") == StrictDevice::Default, "ONEBIT_STRICT_DEVICE empty: default");
        expect(onebit::strict_from_env("0") == StrictDevice::Off, "ONEBIT_STRICT_DEVICE=0: off");
        expect(onebit::strict_from_env("1") == StrictDevice::On, "ONEBIT_STRICT_DEVICE=1: on");
        bool threw = false;
        try { (void) onebit::strict_from_env("maybe"); } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "ONEBIT_STRICT_DEVICE=maybe: refused");

        // ggml-hrx's abort line (GGML_ABORT: file:line: func: message) becomes serve's refusal
        const std::string abort_line =
            "/src/ggml/src/ggml-hrx/ggml-hrx.cpp:444: placement_observer: GGML_HRX_STRICT=1: \"attn_out-0\" is "
            "MUL_MAT f32 ne=[1024,7,1,1] and is not placed on HRX; 1 of 512 nodes off HRX (MUL_MAT x 1), 3 splits";
        const std::string detail = onebit::strict_abort_detail(abort_line + "\n");
        expect(detail.rfind("\"attn_out-0\" is MUL_MAT f32 ne=[1024,7,1,1]", 0) == 0 && detail.back() == 's',
               "abort: detail " + detail);
        expect(onebit::strict_abort_detail(kOffHrx).empty(), "a report line is not an abort");
        const std::string msg = onebit::strict_refusal_message(detail);
        expect(msg.find("MUL_MAT f32 ne=[1024,7,1,1]") != std::string::npos &&
                   msg.find("--no-strict-device") != std::string::npos &&
                   msg.find("ONEBIT_STRICT_DEVICE=0") != std::string::npos,
               "refusal names the op, the shape and the opt-out: " + msg);

        onebit::ServeMetrics metrics("hrx");
        metrics.backend_line(kAllOnHrx);
        expect(metrics.strict_refusal().empty(), "serve: no refusal from a clean report");
        metrics.backend_line(abort_line);
        expect(metrics.strict_refusal() == detail, "serve: the abort is kept for the exit line");
    }

    if (failures == 0) std::printf("hrx_placement: OK\n");
    return failures == 0 ? 0 : 1;
}
