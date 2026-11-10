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
// The vision component of a two-component pack (engine#1788): `vision_weight.q4nx` beside `model.q4nx`.
//
// FastFlowLM's MiniCPM-V 4.7 packs ship the vision tower as its own container at the pack root. It is the same
// container as model.q4nx (8-byte little-endian length, a JSON table keyed by tensor name plus `__metadata__`, then
// the payload that `data_offsets` index), holding a different kind of content: a ViT, every tensor BF16, no tiles.
//
// What this reader is for: the forward that will run the tower must never size a buffer from config.json. Both packs
// on the catalogue state `MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE 4304` and `MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE
// 17216` while their tensors are 4608 and 17280 rows; a loader that trusted the config would read 304 rows into the
// next tensor and return plausible numbers. So the geometry here is DERIVED from the tensors' own shapes, every
// tensor in the file has to be one this reader can place (zero gaps), and the config is only consulted for what the
// tensors cannot say (the head count, the image size) and for the two widths that tie the tower to something outside
// it (the patch grid and the text model's hidden size).
//
// Two kinds of disagreement with config.json, on purpose:
//   refused  the head count does not divide the width, the patch grid is not the position table, the tower's output
//            is not the text model's hidden size, or the width / block count differ: the files do not belong together
//            or the container is cut short, and no forward can be right.
//   noted    an MLP intermediate width differs. The tensors decide it completely and nothing outside the tower
//            depends on it; the tensor wins and `notes()` says what the config claimed.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::npu::vision {

struct Tensor {
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;  // BF16, little-endian, row-major
    uint64_t bytes = 0;
    uint64_t offset = 0;            // from the payload start
    uint64_t elements() const { return bytes / 2; }
};

// Everything below is read from tensor shapes, except heads / head_dim / image_size (config.json, 0 when no config
// was given).
struct Geometry {
    int hidden = 0;                   // patch_embed.proj.weight rows
    int channels = 0, patch = 0;      // patch_embed.proj.weight [hidden, channels, patch, patch]
    int layers = 0;                   // blocks 0..layers-1, all present
    int intermediate = 0;             // blocks.*.mlp.linear_fc1 rows, the same in every block
    int pos_rows = 0, grid = 0;       // pos_embed.weight rows, and its square root (patches per side)
    int merge_hidden = 0;             // 4 * hidden: one 2x2 group of tokens, the width both mergers work at
    int vit_merger_intermediate = 0;  // vit_merger.mlp.linear_fc1 rows
    int out_hidden = 0;               // merger.linear_fc2 rows: the text model's hidden size
    int heads = 0, head_dim = 0, image_size = 0;
};

class Pack {
public:
    // `pack_dir` holds config.json and the container config.json names (`vision_model_weight`, default
    // vision_weight.q4nx). Throws std::runtime_error naming the file and the tensor or field on every refusal.
    explicit Pack(const std::string& pack_dir);
    // The container alone, with config.json's text ("" for none: nothing is cross-checked and heads stay 0).
    Pack(const std::string& container_path, const std::string& config_json);
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;

    const Geometry& geometry() const { return geom_; }
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }
    const Tensor& tensor(const std::string& name) const;  // throws when absent
    // The tensor as float32 (a BF16 value is the top half of a float32, so this is exact).
    std::vector<float> f32(const std::string& name) const;
    // Config fields the tensors overruled, one sentence each; empty when the config agrees or was not given.
    const std::vector<std::string>& notes() const { return notes_; }
    // `__metadata__[key]` as text, "" when absent.
    std::string metadata(const std::string& key) const;
    uint64_t payload_bytes() const { return payload_; }
    // Payload bytes no tensor covers. 0 on both published packs.
    uint64_t unaccounted_bytes() const { return unaccounted_; }
    const std::string& path() const { return path_; }

private:
    void open(const std::string& container_path, const std::string& config_json);
    const uint8_t* map_ = nullptr;
    size_t size_ = 0;
    std::string path_;
    std::map<std::string, Tensor> tensors_;
    std::map<std::string, std::string> meta_;
    std::vector<std::string> notes_;
    Geometry geom_;
    uint64_t payload_ = 0, unaccounted_ = 0;
};

// `1bit npu-pack vision <pack dir>`: the geometry, the tensor count, the provenance and the notes, on stdout.
// Returns 0, or throws what Pack throws.
int print_info(const std::string& pack_dir);

}  // namespace onebit::npu::vision
