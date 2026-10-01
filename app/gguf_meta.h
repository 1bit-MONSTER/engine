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
#include <cstring>
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

// The integer value of metadata key `key` (any GGUF integer type), or -1 when the file is not a
// GGUF, lacks the key, or stores something else under it.
inline long long gguf_int(const std::string& path, const std::string& key) {
    std::ifstream f(path, std::ios::binary);
    auto rd = [&](void* p, size_t n) { return static_cast<bool>(f.read(static_cast<char*>(p), n)); };
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(&magic, 4) || magic != 0x46554747u || !rd(&version, 4) || version < 2 || !rd(&n_tensors, 8) || !rd(&n_kv, 8))
        return -1;
    auto str = [&](std::string* out) {
        uint64_t n = 0;
        if (!rd(&n, 8) || n > (1u << 20)) return false;
        if (!out) return static_cast<bool>(f.seekg(static_cast<std::streamoff>(n), std::ios::cur));
        out->resize(n);
        return rd(out->data(), n);
    };
    static const int size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    auto scalar = [](uint32_t t) -> int { return t < 13 ? size[t] : -1; };
    for (uint64_t i = 0; i < n_kv && i < 4096; ++i) {
        std::string k;
        uint32_t type = 0;
        if (!str(&k) || !rd(&type, 4)) return -1;
        if (k == key) {
            unsigned char b[8] = {};
            const int n = scalar(type);
            if (n <= 0 || type == 6 || type == 7 || type == 12 || !rd(b, n)) return -1;  // f32, bool, f64
            const bool is_signed = type == 1 || type == 3 || type == 5 || type == 11;
            long long v = 0;
            std::memcpy(&v, b, n);
            if (is_signed && n < 8 && (b[n - 1] & 0x80)) v -= 1LL << (8 * n);
            return v;
        }
        if (type == 8) {
            if (!str(nullptr)) return -1;
        } else if (type == 9) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) return -1;
            if (et == 8) {
                for (uint64_t j = 0; j < n; ++j)
                    if (!str(nullptr)) return -1;
            } else if (scalar(et) > 0) {
                f.seekg(static_cast<std::streamoff>(n * scalar(et)), std::ios::cur);
            } else {
                return -1;
            }
        } else if (scalar(type) > 0) {
            f.seekg(scalar(type), std::ios::cur);
        } else {
            return -1;
        }
        if (!f) return -1;
    }
    return -1;
}

// How many tensors of ggml type `type` (GGML_TYPE_Q4_0 = 2, ...) a GGUF holds, or -1 when the file
// is not a GGUF or its header cannot be read. Reads the key/value section and the tensor infos only.
inline long long gguf_tensor_type_count(const std::string& path, uint32_t type) {
    std::ifstream f(path, std::ios::binary);
    auto rd = [&](void* p, size_t n) { return static_cast<bool>(f.read(static_cast<char*>(p), n)); };
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(&magic, 4) || magic != 0x46554747u || !rd(&version, 4) || version < 2 || !rd(&n_tensors, 8) || !rd(&n_kv, 8))
        return -1;
    auto skip_str = [&]() {
        uint64_t n = 0;
        if (!rd(&n, 8) || n > (1u << 20)) return false;
        return static_cast<bool>(f.seekg(static_cast<std::streamoff>(n), std::ios::cur));
    };
    static const int size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    auto scalar = [](uint32_t t) -> int { return t < 13 ? size[t] : -1; };
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint32_t t = 0;
        if (!skip_str() || !rd(&t, 4)) return -1;
        if (t == 8) {
            if (!skip_str()) return -1;
        } else if (t == 9) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!rd(&et, 4) || !rd(&n, 8)) return -1;
            if (et == 8) {
                for (uint64_t j = 0; j < n; ++j)
                    if (!skip_str()) return -1;
            } else if (scalar(et) > 0) {
                f.seekg(static_cast<std::streamoff>(n * scalar(et)), std::ios::cur);
            } else {
                return -1;
            }
        } else if (scalar(t) > 0) {
            f.seekg(scalar(t), std::ios::cur);
        } else {
            return -1;
        }
        if (!f) return -1;
    }
    long long count = 0;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        uint32_t n_dims = 0, t = 0;
        uint64_t offset = 0;
        if (!skip_str() || !rd(&n_dims, 4) || n_dims > 8) return -1;
        f.seekg(static_cast<std::streamoff>(8 * n_dims), std::ios::cur);
        if (!rd(&t, 4) || !rd(&offset, 8)) return -1;
        count += t == type;
    }
    return count;
}

}  // namespace onebit
