// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
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

struct OnnxGraphInfo {
    std::vector<std::string> inputs;    // graph inputs, in declaration order
    std::vector<std::string> outputs;   // graph outputs
    std::vector<OnnxNodeInfo> nodes;    // graph order, which ONNX guarantees is topological
    std::vector<std::string> initializers;   // weight/bias names
    std::vector<std::string> external;       // initializers whose data is in a side file (location set)
};

// Throws std::runtime_error with the path and the protobuf complaint on a malformed file.
OnnxGraphInfo parse_onnx_graph(const std::string& path);

}  // namespace onebit::npu::pack_import
