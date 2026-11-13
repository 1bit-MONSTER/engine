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
// npu_image_part_test.cpp -- the data: URL decode in front of the vision chain (engine#2324).
//
// Downstream of this function the chain is decoder-agnostic (npu/vision_image.h takes raw RGB plus geometry), so this is the one
// place that knows what a picture arrives as. Both directions are pinned: a P6 PPM decodes to exactly its samples, and every way
// of not being one -- a PNG, a remote URL, a truncated header, a short payload, maxval 16, a 0x0 image -- is refused with a
// message, because "an image that produced nothing" is the failure this file exists to prevent.
#include <cstdio>
#include <string>
#include <vector>

#include "npu_image_part.h"

static int g_fail = 0, g_n = 0;
static void check(bool ok, const std::string& what) {
    ++g_n;
    if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++g_fail; }
}

static std::string b64(const std::string& raw) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    uint32_t acc = 0; int bits = 0;
    for (unsigned char c : raw) {
        acc = (acc << 8) | c; bits += 8;
        while (bits >= 6) { bits -= 6; out += t[(acc >> bits) & 0x3f]; }
    }
    if (bits) out += t[(acc << (6 - bits)) & 0x3f];
    while (out.size() % 4) out += '=';
    return out;
}

static std::string ppm(int w, int h, int maxval = 255, size_t trim = 0) {
    std::string s = "P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n" + std::to_string(maxval) + "\n";
    std::string body(size_t(w) * size_t(h) * 3u, '\x7f');
    for (size_t k = 0; k < body.size(); ++k) body[k] = char(k % 251);
    if (trim && trim <= body.size()) body.resize(body.size() - trim);
    return s + body;
}

int main() {
    using onebit::npu::decode_image_data_url;
    using onebit::npu::DecodedImage;

    // 1. the happy path: a 2x1 P6 decodes to exactly its six samples
    {
        DecodedImage img;
        const std::string p = ppm(2, 1);
        check(decode_image_data_url("data:image/x-portable-pixmap;base64," + b64(p), img),
              "a P6 PPM in a data URL decodes");
        check(img.width == 2 && img.height == 1, "the geometry is the PPM's own");
        check(img.rgb.size() == 6 && img.rgb[0] == uint8_t(p[p.size() - 6]),
              "the samples are the PPM's bytes, in order");
    }
    // 2. an omitted media type is accepted (browsers emit `data:;base64,`)
    {
        DecodedImage img;
        check(decode_image_data_url("data:;base64," + b64(ppm(1, 1)), img),
              "an omitted media type is decoded as P6");
    }
    // 3. everything that is not a P6 is refused, by name
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/png;base64," + b64("\x89PNG\r\n\x1a\n"), img),
              "a PNG is refused");
        check(img.error.find("image/png") != std::string::npos,
              "...and the refusal names the media type it cannot decode");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("https://example.test/a.png", img), "a remote URL is refused");
        check(img.error.find("does not fetch") != std::string::npos, "...and says it will not fetch");
    }
    // 4. malformed PPMs are refused rather than passed on as short buffers
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64("P6\n2 1\n"), img),
              "a PPM with no maxval is refused");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64(ppm(2, 2, 255, 5)), img),
              "a PPM whose sample bytes are short is refused");
        check(img.error.find("short") != std::string::npos, "...and says it is short");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64(ppm(1, 1, 16)), img),
              "a PPM with maxval 16 is refused (two bytes per sample is a different format)");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64(ppm(0, 1)), img),
              "a 0-wide PPM is refused");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64,!!!notbase64!!!", img),
              "invalid base64 is refused");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap," + ppm(1, 1), img),
              "a non-base64 data URL is refused");
    }
    // 4b. the hostile header: a dimension past 2^31-1 is refused as out of range, not as a wrapped int. This is a parser of
    // a REMOTE string, so the width is attacker-controlled; before the fix the multiply was signed-overflow UB and only the
    // `w <= 0` check afterwards caught the wrap.
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64("P6\n2147483648 2\n255\n"), img),
              "a width past 2^31-1 is refused");
        check(img.error.find("out of range") != std::string::npos,
              "...and says the field is out of range rather than calling the header truncated");
    }
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64("P6\n99999999999999 2\n255\n"), img),
              "a 14-digit width is refused too");
    }
    // 4c. trailing bytes ARE accepted, deliberately: a PPM stream may carry data after the samples. Pinned so the leniency
    // cannot be mistaken for the "exact" the earlier comment claimed, and so the buffer handed downstream stays exactly w*h*3.
    {
        DecodedImage img;
        const std::string p = ppm(2, 1) + "trailing";
        check(decode_image_data_url("data:image/x-portable-pixmap;base64," + b64(p), img),
              "a PPM with trailing bytes after the samples is still accepted");
        check(img.rgb.size() == 6, "...and the buffer handed downstream is exactly w*h*3, not the trailing bytes too");
    }

    // 5. the decoder is not fooled by a P3 (ASCII PPM) header
    {
        DecodedImage img;
        check(!decode_image_data_url("data:image/x-portable-pixmap;base64," + b64("P3\n1 1\n255\n0 0 0\n"), img),
              "an ASCII PPM (P3) is refused: the chain wants binary samples");
    }

    if (g_fail) { std::printf("npu_image_part: %d of %d checks FAILED\n", g_fail, g_n); return 1; }
    std::printf("npu_image_part: %d checks ok\n", g_n);
    return 0;
}
