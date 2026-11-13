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

#include "vision_request.h"

#include "vision_image.h"
#include "vision_plan.h"
#include "vision_tower.h"

#include <stdexcept>

namespace onebit::npu::vision {

bool image_markers(const Tokenizer& tok, ImageMarkers& m, int& id_open, int& id_close, std::string& why) {
    m.image_start = tok.token_id("<image>");
    m.image_end = tok.token_id("</image>");
    m.slice_start = tok.token_id("<slice>");
    m.slice_end = tok.token_id("</slice>");
    m.image_pad = tok.token_id("<|image_pad|>");
    id_open = tok.token_id("<image_id>");
    id_close = tok.token_id("</image_id>");
    const std::vector<int> nl = tok.encode("\n");
    if (m.image_start < 0 || m.image_end < 0 || m.slice_start < 0 || m.slice_end < 0 || m.image_pad < 0 || id_open < 0 ||
        id_close < 0 || nl.size() != 1) {
        why = "this tokenizer lacks one of <image>, </image>, <slice>, </slice>, <|image_pad|>, <image_id>, </image_id>, or "
              "does not encode a newline as one token";
        return false;
    }
    m.newline = nl[0];
    return true;
}

uint64_t images_hash(const std::vector<RgbImage>& images) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    };
    for (const RgbImage& im : images) {
        const int32_t dims[2] = {im.height, im.width};
        mix(reinterpret_cast<const uint8_t*>(dims), sizeof dims);
        mix(im.rgb.data(), im.rgb.size());
    }
    return h;
}

bool tower_rows(const Pack& pack, const std::vector<RgbImage>& images, ExpandedPrompt& out, std::string& why) {
    out.rows.clear();
    try {
        for (const RgbImage& im : images) {
            const ImagePlan plan = plan_image(im.height, im.width);
            for (const Piece& p : preprocess(im.rgb.data(), im.height, im.width, plan)) {
                const std::vector<float> r = tower_forward(pack, p.pixels.data(), p.grid_h, p.grid_w);
                out.rows.insert(out.rows.end(), r.begin(), r.end());
            }
        }
    } catch (const std::exception& e) {
        why = e.what();
        return false;
    }
    size_t placeholders = 0;
    for (int t : out.ids) placeholders += t == out.image_pad;
    if (out.width <= 0 || placeholders * size_t(out.width) != out.rows.size()) {
        why = "the tower produced " + std::to_string(out.rows.size()) + " values for " + std::to_string(placeholders) +
              " placeholder rows of width " + std::to_string(out.width);
        return false;
    }
    return true;
}

bool expand_images(const Tokenizer& tok, const Pack& pack, const std::vector<int>& rendered, const std::vector<RgbImage>& images,
                   ExpandedPrompt& out, std::string& why, int max_rows, bool run_tower) {
    out = ExpandedPrompt{};
    ImageMarkers m;
    int id_open = -1, id_close = -1;
    if (!image_markers(tok, m, id_open, id_close, why)) return false;
    size_t pads = 0;
    for (int t : rendered) pads += t == m.image_pad;
    if (pads != images.size()) {
        why = "the rendered prompt holds " + std::to_string(pads) + " image placeholder(s) for " + std::to_string(images.size()) +
              " image(s)";
        return false;
    }
    out.width = pack.geometry().out_hidden;
    out.image_pad = m.image_pad;
    size_t k = 0;
    try {
        for (int t : rendered) {
            if (t != m.image_pad) { out.ids.push_back(t); continue; }
            const RgbImage& im = images[k];
            if (im.height <= 0 || im.width <= 0 || im.rgb.size() != size_t(im.height) * size_t(im.width) * 3) {
                why = "image " + std::to_string(k) + " is not height x width x 3 bytes";
                return false;
            }
            const ImagePlan plan = plan_image(im.height, im.width);
            if (plan.total_rows() > max_rows) {
                why = "image " + std::to_string(k) + " (" + std::to_string(im.height) + " x " + std::to_string(im.width) + ") is " +
                      std::to_string(plan.total_rows()) + " rows; this route takes at most " + std::to_string(max_rows);
                return false;
            }
            // Every piece's grid comes from the plan: the whole view, then each slice (all the same size).
            out.grids.push_back({plan.source.grid_h, plan.source.grid_w});
            for (int sl = 0; sl < plan.slice_rows * plan.slice_cols; ++sl) out.grids.push_back({plan.slice.grid_h, plan.slice.grid_w});
            // `<image_id>k</image_id>` is ordinary text in front of the image, as the reference processor writes it.
            out.ids.push_back(id_open);
            const std::vector<int> num = tok.encode(std::to_string(k));
            out.ids.insert(out.ids.end(), num.begin(), num.end());
            out.ids.push_back(id_close);
            const std::vector<int> img = image_tokens(plan, m);
            out.ids.insert(out.ids.end(), img.begin(), img.end());
            ++k;
        }
        out.positions = prompt_positions(out.ids, m, out.grids);
    } catch (const std::exception& e) {
        why = e.what();
        return false;
    }
    out.image_hash = images_hash(images);
    out.images = int(images.size());
    return run_tower ? tower_rows(pack, images, out, why) : true;
}

}  // namespace onebit::npu::vision
