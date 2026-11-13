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
// audio_exec.h -- run an audio pack's graph on the host in f32. Not a general ONNX runtime: it covers exactly the ops a
// Whisper encoder uses (measured on both exports -- 447 nodes / 11 op types for whisper-small, 1031 / 10 for large-v3), and
// it refuses an op it does not implement *by name* rather than producing a plausible activation from a silently skipped node.
//
// The four f32 kernels this stands on are in npu/forward/include/audio_ops.h, which is device-free and known-answer tested;
// this file adds the shape bookkeeping and the wiring. The caller compares its output against the reference, and against
// onnxruntime, before anything here is used to argue a device number.

#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "onnx_graph.h"

namespace onebit::npu::pack_import {

// A device for the graph's matrix products (engine#1783). The executor stays host-only and device-free: it offers each
// product whose right operand is a CONSTANT weight to `gemm`, and runs it on the host itself when the device declines.
//
// Offered: every MatMul whose B is a rank-2 initializer (the projections q, k, v, o, fc1, fc2), with A's leading axes
// flattened into rows, and every Conv, as its im2col GEMM (A = (tout, cin*k), B = the kernel as (cin*k, cout)).
// Not offered: a MatMul whose B is an activation (attention's q.k^T and scores.v). Its operand changes every call, so it
// has no weight to stay resident, and the reason is recorded like any other host product.
//
// `key` names the weight and is stable across calls (the initializer's name; "<name>#conv" for a Conv's transposed
// kernel), so an implementation may cache the packed weight by it. Return false with `why` to leave the product to
// the host. The executor then runs it, counts it under `host`, and keeps the reason in `host_why`, so a run whose
// products silently fell back to the host is visible in the counts.
struct AudioGemm {
    std::function<bool(const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                       std::vector<float>* C, std::string* why)>
        gemm;
    int device = 0;                           // products the device ran
    int host = 0;                             // products the host ran (declined, or not offered)
    std::map<std::string, int> host_why;      // reason -> count
};

struct AudioRunResult {
    std::vector<int64_t> dims;      // dims of the requested output
    std::vector<float> data;        // row-major f32 payload
    std::vector<int64_t> i64;       // int64 payload, when the requested tensor is an index/token tensor
    std::vector<uint8_t> b;         // bool payload, when it is a mask
};

// Two entry points, because a Whisper pack has two graphs: the encoder consumes the mel window as f32, the decoder consumes
// a token window as int64 plus the encoder's output. Everything else -- node order, weights, refusals -- is shared.
//
// mel: (mel_bins, frames) row-major f32, the encoder's graph input in the layout its own first Conv consumes.
// out_name: the graph output to return (empty = the graph's declared output).
// Returns true and fills `out`; on failure returns false and sets `why` to the node and the reason.
bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames,
                     const std::string& out_name, AudioRunResult* out, std::string* why);
// The same, with the matrix products offered to `dev` (null = host only, identical to the overload above).
bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames,
                     const std::string& out_name, AudioRunResult* out, std::string* why, AudioGemm* dev);

// The decoder: `tokens` is the int64 token window for the graph's first input; `xa` (the encoder output, row-major f32) is
// bound to the graph's second input when the graph has one. `xa_dims` may be empty when there is no cross-attention input.
bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why);

}  // namespace onebit::npu::pack_import
