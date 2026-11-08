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
// onnx/ort_diagnose.h — one line that says WHY a session would not start (engine#1436).
//
// ORT's own message for a missing kernel is accurate and useless to a reader:
//
//   Could not find an implementation for ConvInteger(10) node with name '/model.0/conv/Conv_quant'
//
// It says nothing about whether the model, the graph, our EP or the *runtime's op coverage* is at
// fault — and "quantized ONNX" is at least two formats, which decides the answer:
//
//   fp32 weights + in-graph QuantizeLinear/DequantizeLinear  -> runs here; our EP claims the MatMuls
//   ONNX-native integer ops (ConvInteger, MatMulInteger, ...)  -> the pinned ORT has no CPU kernel
//
// The second case is not a model failure and not an EP failure, and a row that reports "does not load"
// teaches a reader nothing.  This turns that message into the one line the issue asks for, naming the
// op and the route that does work.
#pragma once

#include <string>

namespace onebit_ep {

// True when `msg` is ORT's "no kernel for <Op>(<version>)" session-creation error.
bool is_missing_kernel_message(const std::string& msg);

// The op that message names ("ConvInteger"), or "" when it names none.
std::string missing_kernel_op(const std::string& msg);

// A one-line explanation to print after ORT's own message, or "" when this is not that kind of failure.
// `ort_version` is the pinned runtime's version string, printed so the reader can check it.
//
// The wording has one job: leave no doubt that the *runtime's op coverage* is what stopped the session,
// that an fp32 export of the same repository is the route that works, and where the difference between
// the two formats is written down.
std::string session_failure_hint(const std::string& msg, const std::string& ort_version);

}  // namespace onebit_ep
