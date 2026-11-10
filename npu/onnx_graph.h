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
// onnx_graph.h -- a *reader* of an ONNX graph, as opposed to the census walk in npu/pack_import.cpp, which keeps only
// the MatMul-family nodes because that is all a quantised LLM pack needs. An audio pack (the Whisper encoder) is a plain
// f32 graph: 32 transformer blocks of Conv/MatMul/Add/LayerNormalization/Gelu/Softmax/Transpose/Mul, and to execute it the
// engine needs every node in order, with its inputs, outputs and integer attributes, and the set of initializer names.
//
// Nothing here executes anything. It exists so the executor's ordering assumption -- "a node's inputs are graph inputs,
// initializers, or outputs of an earlier node" -- is a *checked* fact about a real pack rather than a hope, and so a pack
// that breaks it is refused by name.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::npu::pack_import {

struct OnnxNodeInfo {
    std::string op;
    std::string name;
    std::vector<std::string> ins;
    std::vector<std::string> outs;
    std::map<std::string, int64_t> attrs;   // integer attributes only (axis, perm as a list is not an int attr)
    std::map<std::string, std::vector<int64_t>> attrs_list;   // `perm`, `pads`, `strides`, `dilations`
    std::map<std::string, float> attrs_float;                 // `alpha`, `epsilon` (as stored)
};

// One initializer: its shape, dtype and where its bytes are (inline `raw_data`, or a side file via external_data).
struct OnnxInit {
    std::string name;
    int dtype = 0;                     // TensorProto.DataType: 1 = FLOAT, 7 = INT64, 10 = FLOAT16, ...
    std::vector<int64_t> dims;
    const uint8_t* raw = nullptr;      // valid only while the mapped file is open; the caller copies what it needs
    size_t raw_len = 0;
    std::string location;              // external_data side file, relative to the graph's directory
    uint64_t offset = 0;
    uint64_t length = 0;
};

struct OnnxGraphInfo {
    std::string dir;                   // directory of the .onnx file, for resolving external_data
    std::vector<OnnxInit> init_tensors;
    std::vector<std::string> inputs;    // graph inputs, in declaration order
    std::vector<std::string> outputs;   // graph outputs
    std::vector<OnnxNodeInfo> nodes;    // graph order, which ONNX guarantees is topological
    std::vector<std::string> initializers;   // weight/bias names
    std::vector<std::string> external;       // initializers whose data is in a side file (location set)
};

// Throws std::runtime_error with the path and the protobuf complaint on a malformed file.
OnnxGraphInfo parse_onnx_graph(const std::string& path);

}  // namespace onebit::npu::pack_import
