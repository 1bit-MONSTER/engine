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
// dflash_draft.h -- load a DFlash draft package as a model source (engine#1780, the drafts half of F2).
//
// A draft is a two-file source: `config.json` in the DFlash schema (model_type "dflash": the draft's own geometry, the
// block size, the mask token, the target layers it reads) and a Q4NX container with its weights. The container is in
// forms this tree already decodes (tools/dflash_draft_info.py is the gate that established it), so this is a loader
// and not a format: it checks every tensor against the geometry the config gives, and hands each one out decoded.
//
// What it guarantees, and refuses by name otherwise:
//   - the config is the DFlash schema with every key the loader needs; another architecture (DFlash2, with selector
//     and convolution tensors) is refused rather than half-read;
//   - every projection of every layer is present and is exactly the q4_1 tile grid of the N x K the config implies;
//   - every norm and the fusion matrix are present at 2 bytes per element in the declared shape;
//   - the container's own scalars agree with config.json;
//   - no tensor is left unexplained: a tensor this loader does not know means an architecture it does not know.
//
// Nothing is decoded at load. A draft's projections are 1.3 GB as f32, so the model holds views into the mapped file
// and decode_*() produces one tensor at a time; the caller decides what to keep.
//
// Not here: the draft's forward (five non-causal layers over a masked block), the target-side taps, and anything on
// the device. This is the step before them.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace onebit::npu::dflash {

struct DraftConfig {
    int hidden = 0;            // hidden_size: the draft's width, and the target model's
    int intermediate = 0;      // intermediate_size
    int layers = 0;            // num_hidden_layers
    int heads = 0;             // num_attention_heads
    int kv_heads = 0;          // num_key_value_heads
    int head_dim = 0;
    int block_size = 0;        // positions per drafted block (the first is the last accepted token)
    int64_t mask_token_id = -1;
    std::vector<int> target_layer_ids;   // the target layers whose hidden states the draft reads, in fusion order
    int sliding_window = 0;
    float rms_eps = 0.0f;
    float rope_theta = 0.0f;
    float embedding_scale = 0.0f;
    bool causal = false;
    // Which draft this is. 1: DFlashDraftModel (model_type "dflash"). 2: DFlash2DraftModel, which adds a grouped
    // convolution around each layer's attention and MLP and a candidate selector, states its interface under
    // "dflash_config", and stores every projection (the fusion matrix included) as Q4_K chunks.
    int generation = 1;
    int conv_kernel = 0, conv_group = 0;       // generation 2
    int selector_rank = 0, selector_top_k = 0; // generation 2
    int vocab = 0;                             // generation 2: rows of the selector's codebooks
    std::string draft_weight;  // the container's file name, relative to the package
    std::string base_model;    // the target this draft was made for, as the package names it (may be empty)

    int taps() const { return int(target_layer_ids.size()); }
    int q_dim() const { return heads * head_dim; }
    int kv_dim() const { return kv_heads * head_dim; }
};

// One tensor of the container: where its bytes are and the matrix they describe.
struct TensorView {
    std::string name;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    int rows = 0;              // N (out); 1 for a vector
    int cols = 0;              // K (in), or the vector's length
    int chunk = 0;             // a projection's bytes per 32 x 256 tile: 0 or 5120 = q4_1, 4736 = Q4_K
    bool ok() const { return data != nullptr && bytes != 0; }
};

struct DraftLayer {
    TensorView q, k, v, o, gate, up, down;                 // q4_1 tile grids, N x K
    TensorView input_norm, post_norm, q_norm, k_norm;      // 2-byte gains
    // Generation 2, 2 bytes per element: base kernels [2][kernel][hidden], kernel projections
    // [2 * kernel * hidden / group][hidden] row-major.
    TensorView attn_conv_base, attn_conv_proj, mlp_conv_base, mlp_conv_proj;
};

struct DraftModel {
    DraftConfig cfg;
    TensorView fc;             // hidden x (taps * hidden): fuses the target layers' hidden states (generation 1: 2-byte
                               // elements in 32-row blocks, decode_fusion(); generation 2: a tile grid, decode_projection())
    // Generation 2: the candidate selector. hidden_projection [rank][hidden], the two codebooks [vocab][rank], all at
    // 2 bytes per element. The codebooks are 254 MB each as f32, so a caller reads rows out of the view (decode_row()).
    TensorView selector_hproj, selector_pred, selector_succ;
    TensorView hidden_norm;    // applied to the fused features
    TensorView norm;           // the final norm
    std::vector<DraftLayer> layers;
    size_t container_bytes = 0;
    int tensors = 0;           // tensors in the container, all of them accounted for
    std::shared_ptr<const void> keep;   // owns the mapping every view points into
};

// config.json of a draft package. Throws std::runtime_error naming the key or the value that is wrong.
DraftConfig read_draft_config(const std::string& package_dir);

// The whole package: config, container, every check above. Throws std::runtime_error with the tensor and the reason.
DraftModel load_draft(const std::string& package_dir);

// A projection as f32, row-major [rows][cols] (out[n * cols + k] multiplies input k into output n).
std::vector<float> decode_projection(const TensorView& t);

// A norm's gains as f32.
std::vector<float> decode_gain(const TensorView& t);

// A plain 2-byte-per-element tensor as f32, row-major [rows][cols].
std::vector<float> decode_plain(const TensorView& t);

// One row of a plain 2-byte-per-element tensor into out[cols]. Throws if the row is outside the tensor.
void decode_row(const TensorView& t, int row, float* out);

// The fusion matrix as f32, row-major [hidden][taps * hidden]. The container stores it as [hidden/32][taps*hidden][32]
// -- 32-row blocks, column by column, the row inside the block last -- and that is the order this undoes:
// out[(a*32 + r) * cols + k] = stored[(a * cols + k) * 32 + r].
std::vector<float> decode_fusion(const TensorView& t);

}  // namespace onebit::npu::dflash
