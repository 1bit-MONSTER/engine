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

#include "audio_epilogue.h"
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
    // engine#1783: true when the device already holds `key`'s packed weight, so the executor can offer the product with
    // B = nullptr and skip loading the f32 initializer (a whisper-small encoder pass re-read ~340 MB of weights the device
    // had cached since the first window). Unset = never cached, the old behaviour.
    std::function<bool(const std::string& key)> cached;
    // engine#1783: the product with the projection's tail folded in (bias, GELU, residual; audio_epilogue.h). C must come back
    // with the tail already applied. Unset = the executor runs the tail as nodes, the old behaviour.
    std::function<bool(const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                       std::vector<float>* C, const onebit::audio::Epilogue& ep, std::string* why)>
        gemm_ep;
    // engine#1783, ONEBIT_AUDIO_NPU_ATTN=1 (default off): one layer's attention, out = softmax(q kt) v for every head,
    // offered by the fused attention path with q [H][S][D], kt [H][D][S], v [H][S][D] and out [H][S][D]. The device
    // computes it in bf16, so the encoder output is NOT bit-identical to the host's; the switch exists for the accuracy
    // gate and the A/B. False + why leaves the group to the host's fused path. Unset = never offered.
    std::function<bool(const float* q, const float* kt, const float* v, int64_t H, int64_t S, int64_t D, float* out,
                       std::string* why)>
        attn;
    int attn_groups = 0;                      // attention groups the device ran (each is two products)
    int device = 0;                           // products the device ran
    int host = 0;                             // products the host ran (declined, or not offered)
    std::map<std::string, int> host_why;      // reason -> count
};

struct AudioRunResult {
    std::vector<int64_t> dims;      // dims of the requested output
    std::vector<float> data;        // row-major f32 payload
    std::vector<int64_t> i64;       // int64 payload, when the requested tensor is an index/token tensor
    std::vector<uint8_t> b;         // bool payload, when it is a mask
    // engine#1783: how many MatMul -> Softmax -> MatMul groups this run computed fused, so a test or a measurement can
    // state which attention path it took (0 = node by node, e.g. ONEBIT_AUDIO_FUSED_ATTN=0).
    size_t fused_attention_groups = 0;
    // engine#1783: projections whose bias / GELU / residual nodes ran folded into the product (0 = node by node, e.g.
    // ONEBIT_AUDIO_FOLD_EPILOGUE=0 or a device without gemm_ep).
    size_t folded_epilogues = 0;
    // engine#1783: of the fused attention groups, how many the device ran (ONEBIT_AUDIO_NPU_ATTN=1; 0 = all on the host).
    size_t npu_attention_groups = 0;
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
// engine#1783: offer every weight product of `g` to `dev` once with an all-zero activation of `rows` rows, so the device
// caches each packed weight before the first window. Returns the number offered, -1 + why on a load failure.
int prefill_audio_device(const OnnxGraphInfo& g, int64_t rows, AudioGemm* dev, std::string* why);
bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why);

// engine#1783: what stays the same across a decoder's steps. A greedy decode runs the whole decoder graph once per token
// with a new token window and the SAME encoder output, so every node whose inputs are initializers, constants or the
// encoder output -- above all every layer's cross-attention key and value projections, 1500 x 768 x 768 each -- computes
// the same tensor at every step. The first run with a cache keeps the step-invariant tensors that a step-dependent node
// reads; every later run with that cache starts from them and skips the step-invariant nodes. The skipped nodes would
// have run the same operations on the same inputs, so the output is bit-identical (tests/audio_exec_test.cpp memcmps the
// logits). A cache belongs to ONE encoder output: use a new one per window. A failed run empties it.
struct DecoderStepCache {
    struct Saved {
        std::vector<int64_t> dims;
        std::vector<float> data;
        std::vector<int64_t> i64;
        std::vector<uint8_t> b;
    };
    bool filled = false;
    std::map<std::string, Saved> frontier;   // step-invariant tensors that step-dependent nodes read
    size_t invariant_nodes = 0;              // nodes a filled run skips
    // The one row (position) of the graph output the caller reads, -1 = all. The MatMul that makes the output (whisper's
    // vocabulary projection, 128 x 768 x 51865) then runs only the four-row unit holding that row: the same unit and the
    // same code as a full run, so that row's bytes are unchanged. PRECONDITION: the caller reads that row only. Every
    // other row of the output was never computed (left 0 here), so a logits dump, a beam or a sampler over other rows
    // must leave this at -1.
    int64_t output_row = -1;
    // Incremental decode (a self-attention KV cache without changing the graph). Between greedy steps only one token
    // position changes (`changed_row`, the new token), and causal masking makes every row before it the same as in the
    // previous step. With `incremental` on and a previous step kept, every product whose rows are the token positions
    // (A's row count == `window`) starts from its previous output and recomputes only the four-row unit holding
    // `changed_row`, in every head, with the same code. The rows before it keep their values, which equal a full run's;
    // the rows after it are pad positions nothing reads. Every other node still runs in full, on rows that are equal
    // to a full run's up to `changed_row`. Exactness is CHECKED, not argued: whisper_transcribe's
    // ONEBIT_WHISPER_DECODE_VERIFY=1 and tests/audio_exec_test.cpp memcmp the logits against the full path.
    bool incremental = true;
    int64_t changed_row = -1;                // -1 = a full run (and the run whose tensors the next step starts from)
    int64_t window = 0;                      // the token window (set by run_decoder_graph)
    std::map<std::string, Saved> prev;       // the previous step's outputs of every step-dependent node
    std::map<std::string, Saved> inits;      // initializers loaded by earlier steps (loaded once per window)
    size_t incremental_products = 0;         // products the last run computed for one row unit only
};
bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why, DecoderStepCache* cache);

}  // namespace onebit::npu::pack_import
