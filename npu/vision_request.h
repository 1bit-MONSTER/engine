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

// engine#2324: a chat request's pictures, from decoded RGB to what the text model is given -- the steps
// minicpm_image_probe chains, as one call a serve route can make.
//
// The pack's chat template renders one `<|image_pad|>` per image content part, in order. expand_images() replaces the
// k-th pad with `<image_id>k</image_id>` and the image's own tokens (vision::image_tokens: `<image>`, one placeholder per
// row of the whole view, `</image>`, then the slices), runs the tower on every piece (the whole view, then the slices),
// and lays out the three position planes (vision::prompt_positions). The caller puts the tower's rows where the
// placeholders are and its own embedding everywhere else, and sends the prompt through step_batch_embeds with the
// planes. Decoding then resumes at `positions.next` on the rotary planes while the cache slot keeps counting from the
// prompt's length: an image moves the position by its side, not by its row count.
//
// `image_hash` covers every image's size and bytes in order. Two pictures of one size expand to identical ids, so a
// prompt cache keyed on ids alone would hand one picture's state to the other; the hash is what tells them apart.
#pragma once

#include "tokenizer.h"
#include "vision_pack.h"
#include "vision_prompt.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu::vision {

struct RgbImage {
    int height = 0, width = 0;
    std::vector<uint8_t> rgb;  // height * width * 3, R G B
};

struct ExpandedPrompt {
    std::vector<int> ids;                    // the rendered prompt with every pad expanded
    std::vector<float> rows;                 // the tower's rows, every piece in order, `width` floats each
    int width = 0;                           // the text model's hidden size, as the tower projects
    PromptPositions positions;               // (t, h, w) per id, and the rotary position decoding resumes at
    std::vector<std::pair<int, int>> grids;  // each piece's patch grid, as prompt_positions read them
    int image_pad = -1;                      // the placeholder id: a row of `rows` goes wherever it appears
    uint64_t image_hash = 0;                 // FNV-1a over every image's height, width and bytes, in order
    int images = 0;
};

// The markers a MiniCPM-V 4.7 tokenizer carries; false (with `why`) when one is missing.
bool image_markers(const Tokenizer& tok, ImageMarkers& m, int& id_open, int& id_close, std::string& why);

// Expand `rendered` (the template's ids, one pad per image) with `images`, in order. False with `why` when the pad
// count and the image count differ, a marker is missing, or an image would exceed `max_rows` placeholder rows.
// With `run_tower` false the ids, grids, positions and hash are filled and `rows` is left empty: everything a prompt
// cache needs to decide a hit, without the tower (the plan alone gives every piece's grid). tower_rows() fills them.
bool expand_images(const Tokenizer& tok, const Pack& pack, const std::vector<int>& rendered, const std::vector<RgbImage>& images,
                   ExpandedPrompt& out, std::string& why, int max_rows = 4096, bool run_tower = true);

// The tower's rows for `images` into `out.rows` (an ExpandedPrompt made with run_tower false), checked against its
// placeholder count.
bool tower_rows(const Pack& pack, const std::vector<RgbImage>& images, ExpandedPrompt& out, std::string& why);

uint64_t images_hash(const std::vector<RgbImage>& images);

}  // namespace onebit::npu::vision
