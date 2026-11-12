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
// The text side of a MiniCPM-V 4.7 image (engine#1788): which tokens stand for it in the prompt, and which position
// every token of a prompt with images gets. Arithmetic on token ids; no tokenizer, no tensor, no file.
//
// TOKENS. One `<|image_pad|>` in the rendered prompt is replaced by
//     <image> pad x R0 </image>                                  the whole view, R0 = its rows after the tower
//     <slice> pad x R </slice> ... (one per slice of the row)      then each row of slices,
//     "\n"                                                          rows separated by one newline token
// (transformers' MiniCPMV4_7Processor.replace_image_token). The processor also puts `<image_id>k</image_id>` in front;
// that is ordinary text and is the caller's to tokenise.
//
// POSITIONS. The model's `mrope_mode: canvas`: text counts 1, 2, 3 on all three planes; an image is laid on a square
// canvas whose origin is the next free position, the whole view stretched over the canvas the slices tile, every slice
// at its own place, and the text after it resumes at origin + max(canvas height, canvas width) + 1. So an image costs
// positions by its side, not by its row count. This follows MiniCPMV4_7Model.get_rope_index line for line, including:
//   - `<image>` sits one step before the canvas (the "halo"), `</image>` on its far corner;
//   - the whole view's coordinates are torch.linspace(0, side - 1, n) in float32, rounded half to even;
//   - marker and newline tokens directly after an image belong to it (they take the canvas origin).
// The reference also groups frames that only markers separate and steps once more after text between two frames of a
// group. Both are kept below so the code can be read against it, but for prompts that hold only images neither can
// change a position (an image's trailing markers already belong to it, so no such gap exists): removing either leaves
// all eight tested prompts unchanged. They matter for video frames, which this does not handle.
// tests/npu_vision_prompt_test.cpp holds the reference's positions for eight prompts and its linspace for 25,600 cases.
#pragma once

#include "vision_plan.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onebit::npu::vision {

// The model's ids (config.json: image_start_id, image_end_id, slice_start_id, slice_end_id, image_token_id, newline_id).
struct ImageMarkers {
    int image_start = 0, image_end = 0, slice_start = 0, slice_end = 0, image_pad = 0, newline = 0;
    bool is_marker(int id) const {
        return id == slice_start || id == slice_end || id == image_start || id == image_end || id == newline;
    }
};

// The tokens for one image. Its pieces receive the tower's rows in order: the first run of pads is the whole view,
// then one run per slice, row by row.
inline std::vector<int> image_tokens(const ImagePlan& plan, const ImageMarkers& m) {
    std::vector<int> ids;
    ids.push_back(m.image_start);
    ids.insert(ids.end(), size_t(plan.source.rows), m.image_pad);
    ids.push_back(m.image_end);
    for (int r = 0; r < plan.slice_rows; ++r) {
        if (r) ids.push_back(m.newline);
        for (int c = 0; c < plan.slice_cols; ++c) {
            ids.push_back(m.slice_start);
            ids.insert(ids.end(), size_t(plan.slice.rows), m.image_pad);
            ids.push_back(m.slice_end);
        }
    }
    return ids;
}

// round(torch.linspace(0, last, n))[i] in float32: the first half counts up from 0, the second half down from `last`.
inline int linspace_round(int last, int n, int i) {
    if (n <= 1) return 0;
    const float step = float(last) / float(n - 1);
    const float v = i < n / 2 ? step * float(i) : float(last) - step * float(n - i - 1);
    return int(std::nearbyintf(v));
}

struct PromptPositions {
    std::vector<int> t, h, w;  // one per token
    int next = 0;              // the position of the first token generated after the prompt
};

// `grids` is the patch grid (height, width) of every piece in the order its pads appear: each image's whole view, then
// its slices. Throws when the pads and the grids do not describe the same pieces.
inline PromptPositions prompt_positions(const std::vector<int>& ids, const ImageMarkers& m,
                                        const std::vector<std::pair<int, int>>& grids) {
    const int n = int(ids.size());
    const int merge = 4;  // "16x": two 2 x 2 merges
    struct Frame {
        int thumb_start, thumb_end, end;
        std::vector<std::pair<int, int>> slices;
    };
    struct Group {
        int start, end;
        std::vector<Frame> frames;
    };
    std::vector<Group> groups;
    bool open = false;
    int previous_end = 0;
    for (int i = 0; i < n;) {
        if (ids[size_t(i)] != m.image_pad) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && ids[size_t(j)] == m.image_pad) ++j;
        if (i == 0) throw std::runtime_error("an image placeholder opens the prompt with no <image> in front of it");
        if (open && ids[size_t(i - 1)] == m.slice_start) {
            groups.back().frames.back().slices.push_back({i, j});
        } else {
            bool adjacent = !groups.empty();
            for (int k = previous_end; adjacent && k < i - 1; ++k) adjacent = m.is_marker(ids[size_t(k)]);
            if (!adjacent) groups.push_back({i - 1, 0, {}});
            groups.back().frames.push_back({i, j, 0, {}});
            open = true;
        }
        previous_end = j;
        i = j;
    }
    for (Group& g : groups) {
        for (Frame& f : g.frames) {
            int e = f.slices.empty() ? f.thumb_end : f.slices.back().second;
            while (e < n && m.is_marker(ids[size_t(e)])) ++e;
            f.end = e;
        }
        g.end = g.frames.back().end;
    }

    PromptPositions p;
    p.t.resize(size_t(n));
    p.h.resize(size_t(n));
    p.w.resize(size_t(n));
    for (int i = 0; i < n; ++i) p.t[size_t(i)] = p.h[size_t(i)] = p.w[size_t(i)] = i;
    const auto text = [&](int from, int to, int pos) {
        for (int i = from; i < to; ++i) p.t[size_t(i)] = p.h[size_t(i)] = p.w[size_t(i)] = pos + (i - from);
    };
    size_t next_grid = 0;
    const auto grid = [&]() {
        if (next_grid >= grids.size()) throw std::runtime_error("the prompt holds more image pieces than grids were given");
        const std::pair<int, int> g = grids[next_grid++];
        if (g.first <= 0 || g.second <= 0 || g.first % merge || g.second % merge)
            throw std::runtime_error("a piece's patch grid is not a positive multiple of 4 in both extents");
        return std::pair<int, int>{g.first / merge, g.second / merge};
    };
    // a piece's tokens: `count` of them, row-major on a (gh, gw) grid
    const auto piece = [&](int from, int to, std::pair<int, int> g, int canvas_h, int canvas_w, int h_off, int w_off, int origin) {
        if (to - from != g.first * g.second)
            throw std::runtime_error("a piece has " + std::to_string(to - from) + " placeholders but its grid gives " +
                                     std::to_string(g.first * g.second) + " rows");
        for (int y = 0; y < g.first; ++y)
            for (int x = 0; x < g.second; ++x) {
                const size_t at = size_t(from + y * g.second + x);
                p.t[at] = origin;
                p.h[at] = origin + h_off + linspace_round((canvas_h ? canvas_h : g.first) - 1, g.first, y);
                p.w[at] = origin + w_off + linspace_round((canvas_w ? canvas_w : g.second) - 1, g.second, x);
            }
    };

    int current_pos = 0, current_idx = 0;
    for (const Group& g : groups) {
        if (g.start > current_idx) {
            text(current_idx, g.start, current_pos);
            current_pos += g.start - current_idx;
        }
        int cursor = g.start;
        for (const Frame& f0 : g.frames) {
            const std::pair<int, int> thumb = grid();
            const int frame_start = f0.thumb_start - 1;
            if (frame_start > cursor) {
                text(cursor, frame_start, current_pos);
                current_pos += frame_start - cursor + 1;
            }
            const int frame_end = f0.end < g.end ? f0.end : g.end;
            const int origin = current_pos, halo = origin > 0 ? origin - 1 : 0;
            int canvas_h = thumb.first, canvas_w = thumb.second, slice_h = 0, slice_w = 0, rows = 0, cols = 0;
            std::pair<int, int> first_slice{0, 0};
            const int ns = int(f0.slices.size());
            if (ns) {
                cols = ns;
                for (int k = 0; k + 1 < ns; ++k)
                    if (f0.slices[size_t(k + 1)].first - f0.slices[size_t(k)].second > 2) {
                        cols = k + 1;
                        break;
                    }
                rows = cols > 0 ? ns / cols : 1;
                if (rows * cols != ns) {
                    rows = 1;
                    cols = ns;
                }
                first_slice = grid();
                slice_h = first_slice.first;
                slice_w = first_slice.second;
                canvas_h = rows * slice_h;
                canvas_w = cols * slice_w;
            }
            for (int i = frame_start; i < frame_end; ++i) p.t[size_t(i)] = p.h[size_t(i)] = p.w[size_t(i)] = origin;
            p.h[size_t(frame_start)] = p.w[size_t(frame_start)] = halo;
            if (f0.thumb_end < frame_end) {
                p.h[size_t(f0.thumb_end)] = origin + canvas_h;
                p.w[size_t(f0.thumb_end)] = origin + canvas_w;
            }
            piece(f0.thumb_start, f0.thumb_end, thumb, canvas_h, canvas_w, 0, 0, origin);
            for (int k = 0; k < ns; ++k) {
                const int h_off = (k / cols) * slice_h, w_off = (k % cols) * slice_w;
                const std::pair<int, int> sg = k == 0 ? first_slice : grid();
                const int s0 = f0.slices[size_t(k)].first, s1 = f0.slices[size_t(k)].second;
                if (s0 - 1 >= frame_start) {
                    p.h[size_t(s0 - 1)] = origin + h_off;
                    p.w[size_t(s0 - 1)] = origin + w_off;
                }
                if (s1 < frame_end) {
                    p.h[size_t(s1)] = origin + h_off + sg.first - 1;
                    p.w[size_t(s1)] = origin + w_off + sg.second - 1;
                }
                piece(s0, s1, sg, 0, 0, h_off, w_off, origin);
            }
            for (int k = 0; k + 1 < ns; ++k) {
                const int gap0 = f0.slices[size_t(k)].second, gap1 = f0.slices[size_t(k + 1)].first;
                if (gap1 - gap0 <= 2) continue;
                for (int i = gap0 + 1; i < gap1 - 1; ++i) {
                    p.h[size_t(i)] = origin + (k / cols + 1) * slice_h - 1;
                    p.w[size_t(i)] = origin + cols * slice_w;
                }
            }
            current_pos = origin + (canvas_h > canvas_w ? canvas_h : canvas_w) + 1;
            cursor = frame_end;
        }
        if (cursor < g.end) {
            text(cursor, g.end, current_pos);
            current_pos += g.end - cursor;
        }
        current_idx = g.end;
    }
    if (current_idx < n) text(current_idx, n, current_pos);
    if (next_grid != grids.size()) throw std::runtime_error("more grids were given than the prompt holds image pieces");
    int mx = -1;
    for (int i = 0; i < n; ++i) {
        if (p.t[size_t(i)] > mx) mx = p.t[size_t(i)];
        if (p.h[size_t(i)] > mx) mx = p.h[size_t(i)];
        if (p.w[size_t(i)] > mx) mx = p.w[size_t(i)];
    }
    p.next = mx + 1;
    return p;
}

}  // namespace onebit::npu::vision
