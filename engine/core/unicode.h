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
// unicode.h: UTF-8 codec, character classes and NFC normalization.
//
// Tables come from tools/gen_unicode_tables.py (Python unicodedata); the
// version is recorded at the top of unicode_tables.inc.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::unicode {

// Decodes UTF-8; nullopt on any malformed, overlong or surrogate sequence.
std::optional<std::vector<char32_t>> decode_utf8(std::string_view s);
void append_utf8(std::string& out, char32_t cp);
std::string encode_utf8(const std::vector<char32_t>& cps);

bool is_letter(char32_t cp);      // general category L*
bool is_number(char32_t cp);      // general category N*
bool is_whitespace(char32_t cp);  // Unicode White_Space property

// Canonical composition (NFC) of a code point sequence.
std::vector<char32_t> nfc(const std::vector<char32_t>& in);

}  // namespace onebit::unicode
