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
// gguf_writer.h: builds small GGUF files in memory for tests.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct GgufWriter {
    struct Tensor {
        std::string name;
        std::vector<uint64_t> ne;
        uint32_t type;
        std::vector<uint8_t> bytes;
    };

    std::vector<uint8_t> kv;
    uint64_t n_kv = 0;
    std::vector<Tensor> tensors;
    uint64_t alignment = 32;

    template <typename T>
    static void put(std::vector<uint8_t>& out, T v) {
        uint8_t b[sizeof(T)];
        std::memcpy(b, &v, sizeof(T));
        out.insert(out.end(), b, b + sizeof(T));
    }
    static void put_str(std::vector<uint8_t>& out, const std::string& s) {
        put<uint64_t>(out, s.size());
        out.insert(out.end(), s.begin(), s.end());
    }

    void key(const std::string& k, uint32_t type) {
        put_str(kv, k);
        put<uint32_t>(kv, type);
        ++n_kv;
    }
    void u32(const std::string& k, uint32_t v) { key(k, 4); put(kv, v); }
    void i32(const std::string& k, int32_t v) { key(k, 5); put(kv, v); }
    void f32(const std::string& k, float v) { key(k, 6); put(kv, v); }
    void str(const std::string& k, const std::string& v) { key(k, 8); put_str(kv, v); }
    void str_array(const std::string& k, const std::vector<std::string>& v) {
        key(k, 9);
        put<uint32_t>(kv, 8);
        put<uint64_t>(kv, v.size());
        for (const auto& s : v) put_str(kv, s);
    }
    void tensor_f32(const std::string& name, std::vector<uint64_t> ne, const std::vector<float>& data) {
        Tensor t{name, std::move(ne), 0, {}};
        t.bytes.resize(data.size() * 4);
        std::memcpy(t.bytes.data(), data.data(), t.bytes.size());
        tensors.push_back(std::move(t));
    }

    std::vector<uint8_t> build() const {
        std::vector<uint8_t> out;
        put<uint32_t>(out, 0x46554747);
        put<uint32_t>(out, 3);
        put<uint64_t>(out, tensors.size());
        put<uint64_t>(out, n_kv);
        out.insert(out.end(), kv.begin(), kv.end());
        std::vector<uint64_t> offsets;
        uint64_t off = 0;
        for (const auto& t : tensors) {
            offsets.push_back(off);
            off += (t.bytes.size() + alignment - 1) / alignment * alignment;
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            put_str(out, t.name);
            put<uint32_t>(out, t.ne.size());
            for (auto d : t.ne) put<uint64_t>(out, d);
            put<uint32_t>(out, t.type);
            put<uint64_t>(out, offsets[i]);
        }
        while (out.size() % alignment) out.push_back(0);
        for (const auto& t : tensors) {
            out.insert(out.end(), t.bytes.begin(), t.bytes.end());
            while (out.size() % alignment) out.push_back(0);
        }
        return out;
    }

    static std::string write_temp(const std::vector<uint8_t>& bytes, const std::string& tag) {
        std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/onebit-test-" + tag + ".gguf";
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        return path;
    }
};
