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

// ort_diagnose_test — engine#1436: a session that cannot start says why in one line.
//
// The acceptance on that issue is "a row that ships an ONNX-native int8 export either runs, or says in one
// line that the pinned runtime has no kernel for it and which export of the repo does run. No silent 'does
// not load'."  This pins the second half on ORT's own message text, which is what the runner gets.
//
// The messages below are quoted from the pinned runtime (1.23.2, distro package) as it printed them for a
// 139-byte synthetic ConvInteger graph and for the real public export; they are the input the runner parses,
// so they belong in the test verbatim.
#include "ort_diagnose.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

void check_contains(const std::string& hay, const std::string& needle, const std::string& what) {
    if (hay.find(needle) != std::string::npos) return;
    ++failures;
    std::printf("FAIL %s (looking for \"%s\" in \"%s\")\n", what.c_str(), needle.c_str(), hay.c_str());
}

}  // namespace

int main() {
    using namespace onebit_ep;

    // Reproduced for this issue with a 139-byte synthetic ConvInteger graph (no download, no AGPL model):
    //   onebit-ort-run: FAIL create session: Could not find an implementation for ConvInteger(10) node …
    const std::string convint =
        "Could not find an implementation for ConvInteger(10) node with name 'conv_int8'";
    check(is_missing_kernel_message(convint), "ORT's no-kernel message is recognised");
    check(missing_kernel_op(convint) == "ConvInteger", "the op is extracted from the message");

    const std::string hint = session_failure_hint(convint, "1.23.2");
    check(!hint.empty(), "the ConvInteger failure gets a hint");
    check_contains(hint, "ConvInteger", "the hint names the op");
    check_contains(hint, "1.23.2", "the hint names the pinned runtime's version");
    check_contains(hint, "ONNX-native int8", "the hint says WHICH flavour of quantised ONNX this is");
    check_contains(hint, "fp32 export", "the hint names the route that works");
    check_contains(hint, "Quantized ONNX is two formats", "the hint points at the page that explains it");
    // It must not read as a model failure or an EP failure: both are wrong for this case.
    check(hint.find("model fails") == std::string::npos, "the hint does not blame the model");

    // The other integer ops get the same treatment, because the fix is the same answer.
    for (const char* op : {"MatMulInteger", "QLinearConv", "QLinearMatMul", "DynamicQuantizeLinear"}) {
        const std::string msg = std::string("Could not find an implementation for ") + op + "(10) node…";
        const std::string h = session_failure_hint(msg, "1.23.2");
        check(missing_kernel_op(msg) == op, std::string(op) + ": op extracted");
        check_contains(h, "ONNX-native int8", std::string(op) + ": gets the integer-op reading");
    }

    // A missing kernel for an op that is NOT an integer op is still answered -- a reader needs a route out
    // -- but must not be told the graph is ONNX-native int8, because it is not.
    {
        const std::string msg = "Could not find an implementation for GridSample(20) node…";
        const std::string h = session_failure_hint(msg, "1.23.2");
        check(missing_kernel_op(msg) == "GridSample", "an arbitrary op name is extracted");
        check(!h.empty(), "an arbitrary missing kernel still gets a route out");
        check(h.find("ONNX-native int8") == std::string::npos,
              "an arbitrary op is not mislabelled as ONNX-native int8");
        check_contains(h, "fp32 export", "the route still points at the fp32 export");
    }

    // Anything that is not that failure gets no hint, so the runner's own message is not decorated.
    for (const char* other : {"Invalid model. Node 'x' has 3 inputs but 2 were expected",
                              "Load model from /nope.onnx failed:no such file",
                              ""}) {
        check(!is_missing_kernel_message(other), std::string("not a no-kernel message: ") + other);
        check(session_failure_hint(other, "1.23.2").empty(), std::string("no hint for: ") + other);
    }

    // An empty version string must not leave a stray "()" in the line.
    check_contains(session_failure_hint(convint, ""), "pinned ORT has", "the version is optional");

    std::printf("%s\n", failures ? "FAILED" : "ort_diagnose: ok");
    return failures ? 1 : 0;
}
