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
#include "ort_diagnose.h"

#include <cstddef>

namespace onebit_ep {
namespace {

// ORT's text, from onnxruntime/core/framework/kernel_registry.cc: the message is
//   "Could not find an implementation for <Op>(<version>) node with name '<name>'"
// parsed by hand rather than by a regex, because this file is compiled into a runner that links
// nothing but ORT and the C++ standard library.
const char kNoKernel[] = "Could not find an implementation for ";

// The ops whose absence from the pinned runtime is "ONNX-native int8", i.e. the case this hint exists
// for. A missing kernel for one of these is a runtime-coverage answer, not a model or EP answer.
bool is_integer_op(const std::string& op) {
    static const char* kIntegerOps[] = {"ConvInteger",   "MatMulInteger", "QLinearConv",
                                       "QLinearMatMul",  "DynamicQuantizeLinear"};
    for (const char* candidate : kIntegerOps)
        if (op == candidate) return true;
    return false;
}

}  // namespace

bool is_missing_kernel_message(const std::string& msg) {
    return msg.find(kNoKernel) != std::string::npos;
}

std::string missing_kernel_op(const std::string& msg) {
    const size_t at = msg.find(kNoKernel);
    if (at == std::string::npos) return "";
    size_t i = at + sizeof(kNoKernel) - 1;
    const size_t end = msg.find('(', i);
    if (end == std::string::npos || end == i) return "";
    return msg.substr(i, end - i);
}

std::string session_failure_hint(const std::string& msg, const std::string& ort_version) {
    if (!is_missing_kernel_message(msg)) return "";
    const std::string op = missing_kernel_op(msg);
    const std::string who = op.empty() ? std::string("this op") : ("'" + op + "'");
    std::string hint = "onebit-ort-run: " + who + " is not a kernel the pinned ORT has";
    if (!ort_version.empty()) hint += " (" + ort_version + ")";
    hint += ", so the session cannot start. ";
    if (op.empty() || is_integer_op(op)) {
        // The case this exists for: a public ONNX-native int8 export, which is a different format from
        // the fp32-plus-Q/DQ graphs this engine runs.  Naming the format is what stops a row reporting
        // "the model does not load" when the runtime simply has no kernel for that vocabulary.
        hint += "This graph is ONNX-native int8 (ConvInteger / MatMulInteger / QLinear*/"
                "DynamicQuantizeLinear), which is a different format from fp32 weights with in-graph "
                "QuantizeLinear/DequantizeLinear -- the second kind is what this engine runs and what our "
                "EP claims. Use the repository's fp32 export for this measurement (docs/onnx.md, "
                "\"Quantized ONNX is two formats\").";
    } else {
        hint += "This is the pinned runtime's op coverage, not the model and not our EP: use the fp32 "
                "export of the same repository, or a runtime that implements the op (docs/onnx.md, "
                "\"Quantized ONNX is two formats\").";
    }
    return hint;
}

}  // namespace onebit_ep
