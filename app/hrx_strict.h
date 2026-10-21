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

// Strict device placement on HRX (docs/serve.md, "Off-HRX placement"). Since the HRX census closed
// with every runnable cell all on HRX (2026-10-20), a backend on HRX runs strict by default: ggml-hrx
// aborts on the first node the scheduler would put on the CPU (GGML_HRX_STRICT=1, llama.cpp fork
// #102) instead of serving it slower with nothing said. --no-strict-device or ONEBIT_STRICT_DEVICE=0
// keeps the old fallback; --strict-device also asks for the every-graph report and checks it.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace onebit {

enum class StrictDevice {
    Default,   // strict on HRX, nothing elsewhere
    On,        // --strict-device: strict, plus the every-graph report (GGML_HRX_REPORT=2, -lv 4)
    Off,       // --no-strict-device / ONEBIT_STRICT_DEVICE=0: nodes HRX declines run on the CPU
};

// ONEBIT_STRICT_DEVICE: unset or empty = default, 0 = off, 1 = on.
inline StrictDevice strict_from_env(const char* value) {
    if (value == nullptr || *value == '\0') return StrictDevice::Default;
    const std::string v(value);
    if (v == "0") return StrictDevice::Off;
    if (v == "1") return StrictDevice::On;
    throw std::runtime_error("ONEBIT_STRICT_DEVICE=" + v + ": use 0 (let HRX fall back to the CPU) or 1");
}

// whether a backend on `device` ("hrx", "cpu", ...) runs strict
inline bool strict_for(StrictDevice s, const std::string& device) {
    return device == "hrx" && s != StrictDevice::Off;
}

// what an HRX backend's environment gets for placement; a value the user set wins (Child, and
// --print-backend's env lists only what the caller's environment does not set)
inline std::vector<std::string> hrx_placement_env(StrictDevice s, bool report) {
    // 2 = every graph: the warmup graph alone is not the model (census 2026-10-20, finding 1)
    std::vector<std::string> env{report || s == StrictDevice::On ? "GGML_HRX_REPORT=2" : "GGML_HRX_REPORT=1"};
    if (strict_for(s, "hrx")) env.push_back("GGML_HRX_STRICT=1");
    return env;
}

// llama-server drops the library's INFO lines below -lv 4, so the report reaches serve only when a
// flag reads it. The default strict mode does not need it: the backend's abort is an error line.
inline bool hrx_placement_verbose(StrictDevice s, bool report) { return report || s == StrictDevice::On; }

// ggml-hrx's abort line: "...: GGML_HRX_STRICT=1: \"<node>\" is <OP> <type> ne=[...] and is not placed
// on HRX; N of M nodes off HRX (...), K splits". Returns what follows the marker, or "".
inline std::string strict_abort_detail(const std::string& line) {
    static const std::string marker = "GGML_HRX_STRICT=1: ";
    const size_t at = line.find(marker);
    if (at == std::string::npos) return "";
    std::string detail = line.substr(at + marker.size());
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r' || detail.back() == ' ')) detail.pop_back();
    return detail;
}

// serve's line when a strict backend refused a graph
inline std::string strict_refusal_message(const std::string& detail) {
    return "HRX strict placement refused the model: " +
           (detail.empty() ? std::string("a node would run off HRX (the backend's own line above names it)") : detail) +
           ". Pass --no-strict-device (or set ONEBIT_STRICT_DEVICE=0) to run those nodes on the CPU instead";
}

}  // namespace onebit
