// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// audio_exec.h -- run an audio pack's graph on the host in f32. Not a general ONNX runtime: it covers exactly the ops a
// Whisper encoder uses (measured on both exports -- 447 nodes / 11 op types for whisper-small, 1031 / 10 for large-v3), and
// it refuses an op it does not implement *by name* rather than producing a plausible activation from a silently skipped node.
//
// The four f32 kernels this stands on are in npu/forward/include/audio_ops.h, which is device-free and known-answer tested;
// this file adds the shape bookkeeping and the wiring. The caller compares its output against the reference, and against
// onnxruntime, before anything here is used to argue a device number.

#pragma once

#include <string>
#include <vector>

#include "onnx_graph.h"

namespace onebit::npu::pack_import {

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

// The decoder: `tokens` is the int64 token window for the graph's first input; `xa` (the encoder output, row-major f32) is
// bound to the graph's second input when the graph has one. `xa_dims` may be empty when there is no cross-attention input.
bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why);

}  // namespace onebit::npu::pack_import
