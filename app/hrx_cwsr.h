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

// engine#315 candidate 1 (docs/hrx.md, "CWSR save-area placement"): where ROCr puts each queue's
// CWSR context save area on a box that runs amdgpu with cwsr_enable=1.
//
// By default libhsakmt allocates the save area as a pageable SVM range. On strixhalo with CWSR on,
// GLM-4.7-Flash at the #315 composition faulted 4 of 16 runs that way. With HSA_USE_SVM=0 and
// HSA_USERPTR_FOR_PAGED_MEM=0 in the backend's environment the save area is a KFD GTT buffer
// object instead, and the same cell ran 46 of 46 without a fault (42 clean in band; every save
// area checked as a GTT BO). ROCr then reports no SVM, so HRX also needs HRX_ALLOW_NO_SVM=1 (our
// hrx-system gate, util/no_svm_gate.c), or it registers no HRX0.
//
// serve adds the three variables to every HRX backend's environment (the chat server, the RAG
// servers and the Laya scorer) before it starts, so they are in place before ROCr initialises in
// that process. --hrx-cwsr-placement picks the mode:
//   auto (default)  only when /sys/module/amdgpu/parameters/cwsr_enable reads 1 and the user set
//                   none of the three variables; CWSR off (the shipped default) changes nothing
//   on              always (a variable the user set still wins, as for every serve variable)
//   off             never
// One line in serve's log says what was decided.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit {

enum class CwsrPlacement { Auto, On, Off };

inline constexpr const char* kCwsrEnablePath = "/sys/module/amdgpu/parameters/cwsr_enable";

// Points read_cwsr_enable() at another file. serve never sets it: it exists so a test can drive the
// CWSR-on branch on a box whose amdgpu runs with CWSR off, which is the shipped default - otherwise
// that branch is only ever exercised on a CWSR-on boot (tests/hrx_cwsr_test.cpp, tests/hrx_cwsr_route.sh).
inline constexpr const char* kCwsrEnablePathEnv = "ONEBIT_CWSR_ENABLE_FILE";

// The file serve reads the CWSR state from: $ONEBIT_CWSR_ENABLE_FILE, else amdgpu's own parameter.
inline std::string cwsr_enable_path() {
    if (const char* p = std::getenv(kCwsrEnablePathEnv); p && *p) return p;
    return kCwsrEnablePath;
}
inline constexpr const char* kCwsrPlacementVars[] = {"HSA_USE_SVM", "HSA_USERPTR_FOR_PAGED_MEM", "HRX_ALLOW_NO_SVM"};
inline constexpr const char* kCwsrPlacementValues[] = {"0", "0", "1"};

inline CwsrPlacement parse_cwsr_placement(const std::string& v) {
    if (v == "auto") return CwsrPlacement::Auto;
    if (v == "on") return CwsrPlacement::On;
    if (v == "off") return CwsrPlacement::Off;
    throw std::runtime_error("--hrx-cwsr-placement takes auto, on or off (got \"" + v + "\")");
}

// The amdgpu module's cwsr_enable: 1 or 0, -1 when the file cannot be read (no amdgpu loaded,
// another OS) or holds anything else.
inline int read_cwsr_enable(const std::string& path = kCwsrEnablePath) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return -1;
    char buf[16] = {};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    std::string s(buf, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    if (s == "1") return 1;
    if (s == "0") return 0;
    return -1;
}

struct CwsrDecision {
    std::vector<std::string> env;  // NAME=VALUE for every HRX backend; empty: nothing changes
    std::string log;               // the one line serve prints
};

using EnvLookup = std::function<const char*(const char*)>;

// mode: --hrx-cwsr-placement; cwsr_enable: read_cwsr_enable(); gate_built: the hrx-system this
// build links has HRX_ALLOW_NO_SVM (without it, HSA_USE_SVM=0 leaves no HRX0); getenv_fn: serve's
// environment. Throws for `on` without the gate, since that request cannot be honoured.
inline CwsrDecision decide_cwsr_placement(CwsrPlacement mode, int cwsr_enable, bool gate_built,
                                          const EnvLookup& getenv_fn) {
    CwsrDecision d;
    std::string user_set;
    for (const char* v : kCwsrPlacementVars)
        if (getenv_fn(v)) user_set += (user_set.empty() ? "" : ", ") + std::string(v);
    auto place = [&]() {
        for (size_t i = 0; i < sizeof kCwsrPlacementVars / sizeof *kCwsrPlacementVars; ++i)
            d.env.push_back(std::string(kCwsrPlacementVars[i]) + "=" + kCwsrPlacementValues[i]);
    };
    if (mode == CwsrPlacement::Off) {
        d.log = "HRX: --hrx-cwsr-placement=off; CWSR save-area placement left to ROCr (engine#315)";
        return d;
    }
    if (mode == CwsrPlacement::On) {
        if (!gate_built)
            throw std::runtime_error("--hrx-cwsr-placement=on: this build's hrx-system has no HRX_ALLOW_NO_SVM gate, "
                                     "so HSA_USE_SVM=0 would leave no HRX0 (engine#315)");
        place();
        d.log = "HRX: --hrx-cwsr-placement=on; CWSR save area placed in GTT (engine#315)";
        if (!user_set.empty()) d.log += "; kept from the environment: " + user_set;
        return d;
    }
    if (cwsr_enable != 1) {
        d.log = cwsr_enable == 0 ? "HRX: CWSR is off; CWSR save-area placement unchanged (engine#315)"
                                 : "HRX: CWSR state unknown (no readable amdgpu cwsr_enable); placement unchanged (engine#315)";
        return d;
    }
    if (!user_set.empty()) {
        d.log = "HRX: CWSR is on; " + user_set + " set in the environment, so serve leaves the placement to it (engine#315)";
        return d;
    }
    if (!gate_built) {
        d.log = "HRX: CWSR is on, but this build's hrx-system has no HRX_ALLOW_NO_SVM gate; placement unchanged (engine#315)";
        return d;
    }
    place();
    d.log = "HRX: CWSR is on; CWSR save area placed in GTT (engine#315)";
    return d;
}

}  // namespace onebit
