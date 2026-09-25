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
#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace onebit {

// general.architecture from a GGUF's header, or "" when the file is not a GGUF or has none.
// Reads only the key/value section (GGUF v2/v3, little endian); the tensors are never touched.
inline std::string gguf_architecture(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    auto rd = [&](void* p, size_t n) { return static_cast<bool>(f.read(static_cast<char*>(p), n)); };
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(&magic, 4) || magic != 0x46554747u /* "GGUF" */ || !rd(&version, 4) || version < 2 ||
        !rd(&n_tensors, 8) || !rd(&n_kv, 8))
        return "";
    auto str = [&](std::string* out) {
        uint64_t n = 0;
        if (!rd(&n, 8) || n > (1u << 20)) return false;
        if (!out) return static_cast<bool>(f.seekg(static_cast<std::streamoff>(n), std::ios::cur));
        out->resize(n);
        return rd(out->data(), n);
    };
    // bytes of each scalar value type: u8 i8 u16 i16 u32 i32 f32 bool (string) (array) u64 i64 f64
    auto scalar = [](uint32_t t) -> int {
        static const int size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        return t < 13 ? size[t] : -1;
    };
    for (uint64_t i = 0; i < n_kv && i < 4096; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!str(&key) || !rd(&type, 4)) return "";
        if (type == 8) {  // string
            std::string v;
            if (!str(key == "general.architecture" ? &v : nullptr)) return "";
            if (key == "general.architecture") return v;
        } else if (type == 9) {  // array: element type, count, elements
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) return "";
            if (et == 8) {
                for (uint64_t j = 0; j < n; ++j)
                    if (!str(nullptr)) return "";
            } else if (scalar(et) > 0) {
                f.seekg(static_cast<std::streamoff>(n * scalar(et)), std::ios::cur);
            } else {
                return "";  // nested arrays do not occur in GGUF metadata
            }
        } else if (scalar(type) > 0) {
            f.seekg(scalar(type), std::ios::cur);
        } else {
            return "";
        }
        if (!f) return "";
    }
    return "";
}

}  // namespace onebit
