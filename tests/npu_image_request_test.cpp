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

// engine#2324: a chat request's pictures to RGB bytes (app/npu_image_request.h). A crafted PPM and a crafted PNG (stored
// zlib blocks, no compression needed) go in as data URLs; the bytes that come out must be exactly the pixels written.
#include "npu_image_request.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

static std::string b64(const std::vector<uint8_t>& d) {
    static const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < d.size(); i += 3) {
        uint32_t v = uint32_t(d[i]) << 16 | (i + 1 < d.size() ? uint32_t(d[i + 1]) << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
        o += A[(v >> 18) & 63]; o += A[(v >> 12) & 63];
        o += i + 1 < d.size() ? A[(v >> 6) & 63] : '='; o += i + 2 < d.size() ? A[v & 63] : '=';
    }
    return o;
}

static uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    for (size_t i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1))); }
    return c;
}
static uint32_t adler32(const std::vector<uint8_t>& d) {
    uint32_t a = 1, b = 0;
    for (uint8_t x : d) { a = (a + x) % 65521; b = (b + a) % 65521; }
    return b << 16 | a;
}
static void be32(std::vector<uint8_t>& o, uint32_t v) { for (int s = 24; s >= 0; s -= 8) o.push_back(uint8_t(v >> s)); }
static void chunk(std::vector<uint8_t>& o, const char* type, const std::vector<uint8_t>& data) {
    be32(o, uint32_t(data.size()));
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    o.insert(o.end(), td.begin(), td.end());
    be32(o, crc32(td.data(), td.size()) ^ 0xFFFFFFFFu);
}
// An RGB PNG with filter 0 rows in one stored (uncompressed) deflate block.
static std::vector<uint8_t> png(int w, int h, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> raw;
    for (int y = 0; y < h; y++) { raw.push_back(0); raw.insert(raw.end(), rgb.begin() + y * w * 3, rgb.begin() + (y + 1) * w * 3); }
    std::vector<uint8_t> z = {0x78, 0x01, 0x01};
    const uint16_t len = uint16_t(raw.size());
    z.push_back(uint8_t(len)); z.push_back(uint8_t(len >> 8)); z.push_back(uint8_t(~len)); z.push_back(uint8_t(~len >> 8));
    z.insert(z.end(), raw.begin(), raw.end());
    be32(z, adler32(raw));
    std::vector<uint8_t> o = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    be32(ihdr, uint32_t(w)); be32(ihdr, uint32_t(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(o, "IHDR", ihdr); chunk(o, "IDAT", z); chunk(o, "IEND", {});
    return o;
}

int main() {
    using namespace onebit::npu;
    const int w = 5, h = 3;
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); i++) rgb[i] = uint8_t(i * 37 + 11);

    // PPM (P6): what the probes read
    std::string hdr = "P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n";
    std::vector<uint8_t> ppm(hdr.begin(), hdr.end());
    ppm.insert(ppm.end(), rgb.begin(), rgb.end());
    vision::RgbImage a = image_from_data_url("data:image/x-portable-pixmap;base64," + b64(ppm));
    check(a.height == h && a.width == w && a.rgb == rgb, "a PPM data URL decodes to exactly its pixels");

    // PNG: what a chat client sends
    vision::RgbImage b = image_from_data_url("data:image/png;base64," + b64(png(w, h, rgb)));
    check(b.height == h && b.width == w && b.rgb == rgb, "a PNG data URL decodes to exactly its pixels");

    // the request shape: content parts in message order, both url spellings, text parts skipped
    nlohmann::json msgs = nlohmann::json::parse(R"([
      {"role": "system", "content": "be brief"},
      {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "image_url", "image_url": {"url": "X"}}]},
      {"role": "user", "content": [{"type": "image_url", "image_url": "Y"}]}])");
    msgs[1]["content"][1]["image_url"]["url"] = "data:image/png;base64," + b64(png(w, h, rgb));
    msgs[2]["content"][0]["image_url"] = "data:image/x-portable-pixmap;base64," + b64(ppm);
    std::vector<vision::RgbImage> got;
    check(request_images(msgs, got) && got.size() == 2 && got[0].rgb == rgb && got[1].rgb == rgb,
          "request_images finds both pictures, in order, in either url spelling");
    check(!request_images(nlohmann::json::parse(R"([{"role":"user","content":"no pictures"}])"), got) && got.empty(),
          "a text-only request has no pictures");

    // refusals, by name
    auto refuses = [](const std::string& url, const char* needle) {
        try { image_from_data_url(url); } catch (const std::exception& e) { return std::string(e.what()).find(needle) != std::string::npos; }
        return false;
    };
    check(refuses("https://example.com/cat.png", "only data: URLs"), "a network URL is refused");
    check(refuses("data:image/png,rawbytes", "must be base64"), "a non-base64 data URL is refused");
    check(refuses("data:image/png;base64," + b64({1, 2, 3, 4}), "not a PNG, JPEG or PPM"), "bytes that are no image are refused");
    check(vision::images_hash({a}) == vision::images_hash({b}) && vision::images_hash({a}) != vision::images_hash({vision::RgbImage{h, w, std::vector<uint8_t>(rgb.size(), 0)}}),
          "the image hash follows the pixels, not the encoding");

    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("ok: PPM and PNG data URLs decode exactly, request order kept, refusals named\n");
    return 0;
}
