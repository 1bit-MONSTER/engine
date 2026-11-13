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
// audio_launch_plan.h -- walk an encoder's launch list, device first, host for what the device cannot take (engine#1783).
//
// The route this belongs to (item 2 of the issue's close condition) is not "run the encoder on the NPU": it is a loop over the
// projections in graph order that sends each one to the device when the device can take it and to the host when it cannot, and
// REPORTS WHICH. That report is the measurement the row is scored on -- `device_gemm=<X> s + host_remainder=<Y> s` -- so the
// accounting is part of the contract, not a log line: a launch that silently fell back is a launch whose time is being counted
// in the wrong column, and the whole point of measuring the split is to know which column is worth attacking next.
//
// It is deliberately not a planner: the specs come from `audio_kernel_shapes --write-plan` (already emitted, 72 launches for
// whisper-small, dense order, staged widths), and the host path is a CALLBACK so this file neither duplicates the executor's
// GEMM nor decides what "host" means.
//
// Why a callback for the activations too: each launch reads the previous one's output, so the caller owns the buffers. A
// function that took a single big X would be a function that only works for a stack of independent projections.

#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "audio_device_gemm.h"

namespace onebit::audio {

struct LaunchSpec {
    int         index = -1;     // the plan's own `order`, so a report can be compared with the plan line for line
    int         role = 0;
    int         layer = -1;
    int         k = 0;
    int         n = 0;
    int         rows = 0;
    const char* tag = nullptr;
};

struct LaunchRun {
    bool        device = false;      // ran on the device
    bool        host = false;        // ran on the host (either because there is no target, or after a device refusal)
    std::string why;                 // why it did not run on the device, when it did not
};

struct LaunchReport {
    int device_launches = 0;
    int host_launches = 0;
    int failed = 0;
    std::vector<LaunchRun> per_launch;                    // parallel to the specs, so a caller can print them in order
    std::vector<std::string> failures;                    // "<index> <tag>: <device reason> / <host reason>"
    bool ok() const { return failed == 0; }
};

// A host path for one launch: writes the caller's width (n), returns false and fills why when it cannot.
using HostLaunchFn = std::function<bool(const LaunchSpec&, std::vector<float>*, std::string*)>;
// The activations for one launch, supplied by the caller because each launch reads the previous one's result.
using LaunchInputFn = std::function<const float*(const LaunchSpec&)>;

// Runs every spec in order. A launch goes to the device when `target` is not null AND `device_projection` accepts it; otherwise
// the host callback runs it and the refusal reason is kept. A launch both paths refuse is a failure and does not silently
// count as either column.
inline LaunchReport run_launches(::npu::target::NpuTarget* target, const std::vector<LaunchSpec>& specs,
                                 const LaunchInputFn& input_of, const HostLaunchFn& host) {
    LaunchReport rep;
    rep.per_launch.resize(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        const LaunchSpec& s = specs[i];
        LaunchRun& r = rep.per_launch[i];
        if (target != nullptr) {
            DeviceProjection p;
            p.role = s.role;
            p.layer = s.layer;
            p.k = s.k;
            p.n = s.n;
            p.rows = s.rows;
            p.x = input_of != nullptr ? input_of(s) : nullptr;
            p.tag = s.tag;
            std::vector<float> out;
            std::string why;
            if (device_projection(*target, p, &out, &why)) {
                r.device = true;
                ++rep.device_launches;
                continue;
            }
            r.why = why;
        } else {
            r.why = "no target";
        }
        std::vector<float> out;
        std::string hwhy;
        if (host != nullptr && host(s, &out, &hwhy)) {
            r.host = true;
            ++rep.host_launches;
            continue;
        }
        ++rep.failed;
        rep.failures.push_back("launch " + std::to_string(s.index) + " " + (s.tag != nullptr ? s.tag : "?") +
                               ": device: " + (r.why.empty() ? std::string("not attempted") : r.why) +
                               " / host: " + (hwhy.empty() ? std::string("no host path") : hwhy));
    }
    return rep;
}

// The bar this route is scored on, formatted from a run rather than assembled by hand at the end of a session: the issue's close
// condition asks for `device_gemm=<X> s + host_remainder=<Y> s per 30 s window`, and a number typed from memory is not evidence.
// A launch that is in NEITHER column (failed) is printed explicitly -- a split with a silent third category is a split that adds
// up to less than the run.
inline std::string format_split_run(const LaunchReport& r, double device_ms, double host_ms) {
    char buf[220];
    std::snprintf(buf, sizeof buf,
                  "device_gemm=%.3f s + host_remainder=%.3f s per window (%d device launch(es), %d host, %d failed)",
                  device_ms / 1000.0, host_ms / 1000.0, r.device_launches, r.host_launches, r.failed);
    return std::string(buf);
}

}  // namespace onebit::audio
