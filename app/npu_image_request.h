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

// engine#2324: the pictures of an OpenAI chat request, as RGB bytes.
//
// A picture is a content part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` (or `"url"` as
// a bare string), in message order; the pack's chat template renders one `<|image_pad|>` for each, in the same order,
// which is how the k-th picture finds its place (npu/vision_request.h). Only `data:` URLs are taken: the route fetches
// nothing from the network. PNG, JPEG and binary PPM are decoded (stb_image, app/stb_image_impl.cpp); anything else,
// a picture larger than kMaxImagePixels, and a part with no URL are refused by name.
#pragma once

#include "vision_request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
unsigned char* stbi_load_from_memory(const unsigned char* buffer, int len, int* x, int* y, int* channels_in_file,
                                     int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
const char* stbi_failure_reason(void);
}

namespace onebit::npu {

constexpr long kMaxImagePixels = 16L * 1024 * 1024;

inline std::vector<uint8_t> base64_decode(const std::string& s) {
    static const auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int v = value(c);
        if (v < 0) throw std::runtime_error("image_url: the data URL is not base64");
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// RGB bytes from encoded PNG / JPEG / PPM bytes.
inline vision::RgbImage decode_image_bytes(const std::vector<uint8_t>& bytes) {
    if (bytes.empty() || bytes.size() > size_t(1) << 30) throw std::runtime_error("image_url: an empty or oversized image");
    int w = 0, h = 0, ch = 0;
    unsigned char* px = stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &ch, 3);
    if (!px) throw std::runtime_error(std::string("image_url: not a PNG, JPEG or PPM image (") + stbi_failure_reason() + ")");
    vision::RgbImage im;
    im.height = h;
    im.width = w;
    const bool too_big = long(w) * long(h) > kMaxImagePixels;
    if (!too_big) im.rgb.assign(px, px + size_t(w) * size_t(h) * 3);
    stbi_image_free(px);
    if (too_big)
        throw std::runtime_error("image_url: a " + std::to_string(w) + " x " + std::to_string(h) + " image is more than " +
                                 std::to_string(kMaxImagePixels) + " pixels");
    return im;
}

inline vision::RgbImage image_from_data_url(const std::string& url) {
    if (url.rfind("data:", 0) != 0)
        throw std::runtime_error("image_url: only data: URLs are taken (this route fetches nothing from the network)");
    const size_t comma = url.find(',');
    if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos)
        throw std::runtime_error("image_url: a data URL must be base64 (data:image/png;base64,...)");
    return decode_image_bytes(base64_decode(url.substr(comma + 1)));
}

// Every picture in `messages`, in order. True when there is at least one.
template <class Json>
inline bool request_images(const Json& messages, std::vector<vision::RgbImage>& out) {
    out.clear();
    if (!messages.is_array()) return false;
    for (const auto& m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto& part : m["content"]) {
            if (!part.is_object()) continue;
            const std::string type = part.contains("type") && part["type"].is_string() ? part["type"].template get<std::string>() : "";
            // the set the MiniCPM-V template turns into a pad: an `image` or `image_url` key, or type "image"
            if (type != "image_url" && type != "image" && !part.contains("image_url") && !part.contains("image")) continue;
            std::string url;
            if (part.contains("image_url")) {
                const auto& iu = part["image_url"];
                if (iu.is_string()) url = iu.template get<std::string>();
                else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) url = iu["url"].template get<std::string>();
            }
            if (url.empty() && part.contains("image") && part["image"].is_string()) url = part["image"].template get<std::string>();
            if (url.empty()) throw std::runtime_error("image_url: a picture part without a url");
            out.push_back(image_from_data_url(url));
        }
    }
    return !out.empty();
}

// engine#2324: the digest a prefix-cache slot carries, so "same text, different picture" cannot be a cache hit.
// It covers the DIMENSIONS and the PIXELS, and the pictures in order: two identical images in a different order are a
// different prompt, and a re-encode of the same picture (PNG vs PPM) decodes to the same pixels and therefore to the same
// digest -- which is the behaviour we want, because the cache stores prefill state, not bytes on the wire.
inline uint64_t image_digest_of(const std::vector<vision::RgbImage>& images) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a, 64-bit: stable across runs and platforms, which a std::hash is not
    auto mix = [&h](const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    const uint64_t count = images.size();
    mix(&count, sizeof count);            // 0 images and 1 empty image must not collide
    for (const auto& im : images) {
        const int64_t wh[2] = {im.width, im.height};
        mix(wh, sizeof wh);
        if (!im.rgb.empty()) mix(im.rgb.data(), im.rgb.size());
    }
    return h;
}

// The digest for a whole request: 0 (the text-only value) when there is no picture.
template <class Json>
inline uint64_t request_image_digest(const Json& messages) {
    std::vector<vision::RgbImage> images;
    if (!request_images(messages, images)) return 0;
    return image_digest_of(images);
}


}  // namespace onebit::npu
