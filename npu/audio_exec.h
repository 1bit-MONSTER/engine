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
    std::vector<int64_t> dims;   // dims of the requested output
    std::vector<float> data;     // row-major
};

// mel: (mel_bins, frames) row-major f32, the encoder's graph input in the layout its own first Conv consumes.
// out_name: the graph output to return (empty = the graph's declared output).
// Returns true and fills `out`; on failure returns false and sets `why` to the node and the reason.
bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames,
                     const std::string& out_name, AudioRunResult* out, std::string* why);

}  // namespace onebit::npu::pack_import
