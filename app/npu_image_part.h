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

// npu_image_part.h -- turning an OpenAI `image_url` content part into the bytes the vision chain wants (engine#2324).
//
// The chain downstream of here is decoder-agnostic: npu/vision_image.h's preprocess() takes raw RGB plus its geometry, and
// npu/vision_tower.h's tower_forward() takes the same. What the serve route had no way to produce was those bytes, because an
// image part arrives as a data: URL and nothing in this tree decodes one.
//
// This decodes exactly one media type -- **image/x-portable-pixmap, P6, maxval 255** -- and refuses the rest *by name*, which is
// the same discipline as the route's refusal: a PNG handed to a route with no PNG decoder must not become "an image that produced
// nothing". P6 is the format npu/forward/minicpm_image_probe.cpp already reads, so a picture that works in the probe works here,
// and adding a real decoder later is a change to this one function.
//
// usage:
//     DecodedImage img;
//     if (!decode_image_data_url(url, img)) -> 400 with img.error (it names the media type)
//     preprocess(img.rgb.data(), img.height, img.width, plan)

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace onebit::npu {

struct DecodedImage {
    std::vector<uint8_t> rgb;   // height * width * 3, row-major
    int height = 0, width = 0;
    std::string error;          // set when the decode failed; names the media type when that was the reason
};

// RFC 4648 base64. Not a general-purpose encoder: the decoder is incremental-free and stops at the first bad character, which
// is what a request body needs -- anything after an invalid byte is not a picture.
inline bool base64_decode(const std::string& in, std::vector<uint8_t>& out) {
    static const int8_t kMap[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63, 52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, 15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, 41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    out.clear();
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char ch : in) {
        if (ch == '=') break;                       // padding ends the stream
        const int v = kMap[ch];
        if (v < 0) return false;
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t((acc >> bits) & 0xffu));
        }
    }
    return true;
}

// `data:image/x-portable-pixmap;base64,<payload>`, or `data:;base64,...` (the media type may be omitted, as browsers allow).
inline bool decode_image_data_url(const std::string& url, DecodedImage& out) {
    out = DecodedImage{};
    if (url.rfind("data:", 0) != 0) {
        out.error = "the image part is not a data: URL; this route does not fetch remote images (engine#2324)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) {
        out.error = "the image part has no ',' after its data: header";
        return false;
    }
    const std::string header = url.substr(5, comma - 5);
    const bool b64 = header.find(";base64") != std::string::npos;
    if (!b64) {
        out.error = "the image part is a data: URL that is not base64-encoded; only ;base64 is accepted (engine#2324)";
        return false;
    }
    std::string media = header.substr(0, header.find(';'));
    if (media.empty()) media = "image/x-portable-pixmap";   // `data:;base64,` means "you decide"
    if (media != "image/x-portable-pixmap" && media != "image/x-ppm" && media != "image/ppm") {
        out.error = "no decoder for media type '" + media + "': this route decodes image/x-portable-pixmap (P6) only, and "
                    "refuses the rest by name rather than answering without the picture (engine#2324)";
        return false;
    }
    std::vector<uint8_t> raw;
    if (!base64_decode(url.substr(comma + 1), raw)) {
        out.error = "the image part's base64 payload is not valid base64";
        return false;
    }
    // P6: "P6" whitespace W H whitespace 255 single-whitespace then binary RGB. Comments ('#') may appear in the header.
    size_t i = 0;
    auto skip_ws_and_comments = [&]() {
        while (i < raw.size()) {
            if (raw[i] == '#') { while (i < raw.size() && raw[i] != '\n') ++i; }
            else if (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\r' || raw[i] == '\n') ++i;
            else break;
        }
    };
    auto read_uint = [&](int& v) {
        skip_ws_and_comments();
        if (i >= raw.size() || raw[i] < '0' || raw[i] > '9') return false;
        v = 0;
        while (i < raw.size() && raw[i] >= '0' && raw[i] <= '9') { v = v * 10 + (raw[i] - '0'); ++i; }
        return true;
    };
    if (raw.size() < 2 || raw[0] != 'P' || raw[1] != '6') {
        out.error = "the decoded image is not a P6 PPM ('P6' magic missing)";
        return false;
    }
    i = 2;
    int h = 0, w = 0, maxval = 0;
    if (!read_uint(w) || !read_uint(h) || !read_uint(maxval)) {
        out.error = "the PPM header is truncated (width, height and maxval are required)";
        return false;
    }
    if (maxval != 255) {
        out.error = "the PPM has maxval " + std::to_string(maxval) + "; only 255 (one byte per sample) is accepted";
        return false;
    }
    if (w <= 0 || h <= 0) {
        out.error = "the PPM is " + std::to_string(w) + "x" + std::to_string(h) + ", which is not an image";
        return false;
    }
    ++i;   // exactly one whitespace byte after maxval, then binary samples
    const size_t need = size_t(w) * size_t(h) * 3u;
    if (raw.size() < i + need) {
        out.error = "the PPM is short: " + std::to_string(raw.size() - i) + " of " + std::to_string(need) + " sample bytes";
        return false;
    }
    out.rgb.assign(raw.begin() + long(i), raw.begin() + long(i + need));
    out.height = h;
    out.width = w;
    return true;
}

}  // namespace onebit::npu
