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
// `1bit npu-pack`: turn a Ryzen AI OGA NPU model pack (`*_rai_*_npu_*`, the
// `model.onnx` + `model.onnx.data` + `dd_metastate_*.fconst` + `genai_config.json`
// set AMD publishes) into the engine's Q4NX model directory. The pack's weights
// and its tokenizer/config contract are the model; every line of this reader is
// written from the documented facts in
// amd-oss/npu-re: docs/FACTS-oga-pack-weight-layout.md (pack-side layout) and
// docs/npu.md here (container side). No vendor library, wheel or compiled
// program is called, linked or shipped.
//
// Two commands:
//   npu-pack info   <pack dir>                 what the pack says, as JSON
//   npu-pack import <pack dir> -o <out dir>    write model.q4nx + config.json
//
// A pack the importer cannot represent is refused by name, with the reason and
// the tensor, and nothing is written: a container that silently carries 1 % of
// a model is worse than no container. `--allow-incomplete` writes what is
// representable and records the rest in the container's manifest, so the gap is
// visible to whatever loads it next.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::npu::pack_import {

// The model geometry the container's config.json needs.
struct Dims {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
    float rope_theta = 0;
    std::string model_type;
    std::vector<int> eos;
};

// One tensor the pack describes, and where its values would come from.
struct Planned {
    std::string name;     // Q4NX tensor name (model.layers.N.<...>.weight)
    std::string source;   // "model.onnx.data", "fconst:Llm_Token_Token", "fconst:Llm_Prefill", "config.json"
    std::string dtype;    // "q4_1" tiles, or "BF16"
    std::vector<int64_t> shape;
    std::string refusal;  // empty when importable; otherwise the named reason
};

struct Plan {
    std::string dir;
    std::string pack;  // *_rai_1.8.0_npu_4K, from the directory name
    Dims dims;
    std::vector<Planned> tensors;

    std::vector<std::string> refused() const;
    std::string manifest() const;  // JSON: format, layout version, source, geometry, tensors
};

// Reads the pack (model.onnx, its external data, both .fconst headers, config.json).
// Throws only when the pack is structurally unreadable.
Plan plan(const std::string& pack_dir);

struct Options {
    bool allow_incomplete = false;  // write what is representable and name the rest
};

struct Report {
    std::string out_dir;
    int tensors = 0;
    long long tiles = 0;
    std::vector<std::string> refused;

    // Self-check 1: the 4-bit decoder against the pack's own independent data. With
    // tie_word_embeddings the pack's lm_head must reproduce its fp16 embedding, so the
    // decode is checked against the embedding, and two wrong decoders are run as controls.
    double lmhead_corr = 0, lmhead_rel = 0, ctrl_corr_nibble = 0, ctrl_corr_nozp = 0;

    // Self-check 2: the written container, read back, against the pack's dequantised values.
    double q4nx_rel = 0, q4nx_max_abs = 0, zero_exact_pct = 0;
    long long codes_exact = 0, codes_total = 0;

    // fconst norm anchors: per-layer RMSNorm weights are located by fixed offsets; a block
    // is aligned right when its head carries the bf16 epsilon.
    int norm_anchors_ok = 0, norm_anchors_total = 0;
};

Report import_pack(const std::string& pack_dir, const std::string& out_dir, const Options&);

// `1bit npu-pack <info|import> ...`; argv[0] is the subcommand.
int run(int argc, char** argv);

// Exposed for the tests: a decoded MatMulNBits tensor as (codes, scale, zero) per
// 32-column group, ready to tile.
struct Q4 {
    int N = 0, K = 0, block = 0;
    std::vector<uint8_t> codes;  // [N, K], 0..15
    std::vector<float> scale;    // [N, K/32]
    std::vector<float> zero;     // [N, K/32]
};
Q4 decode_matmulnbits(const uint8_t* qweight, const float* scales, const uint8_t* zeros,
                      int N, int K, int block, bool swap_nibbles = false, bool drop_zero = false);

// Q4NX q4_1 tiles of a Q4 tensor: [N/32 * K/256, 5120] bytes, plus the parameters
// that were not exactly representable in bf16 (counted, not fixed up).
void tile_q4_1(const Q4& q, std::vector<uint8_t>& out, long long* zero_inexact = nullptr);

}  // namespace onebit::npu::pack_import
