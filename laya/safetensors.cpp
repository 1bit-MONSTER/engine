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

#include "safetensors.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include "file_map.h"

namespace onebit::laya {

namespace {

// Real IEEE754 half-precision float16 (5-bit exponent, bias 15). Matches the
// engine's npu path and the 1bit-MONSTER reader it was ported from.
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    const float sign = s ? -1.0f : 1.0f;
    if (e == 0) return sign * (float)m * 5.9604644775390625e-08f;  // subnormal: m * 2^-24
    if (e == 31) return m ? NAN : sign * INFINITY;
    return sign * (1.0f + (float)m / 1024.0f) * std::pow(2.0f, (float)((int)e - 15));
}

// bfloat16: bits are float32's truncated upper half.
inline float bf16_to_f32(uint16_t bf) {
    const uint32_t bits = (uint32_t)bf << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

bool Safetensors::open(const std::string& path) {
    err_.clear();
    const uint8_t* m = onebit::npu::map_file(path, size_, &err_);
    if (!m) return false;
    map_ = const_cast<uint8_t*>(m);

    uint64_t header = 0;
    if (size_ < 8) {
        err_ = path + ": truncated";
        return false;
    }
    std::memcpy(&header, map_, 8);
    if (8 + header > size_) {
        err_ = path + ": header past the end of the file";
        return false;
    }
    const uint64_t base = 8 + header;
    const auto table = nlohmann::json::parse(map_ + 8, map_ + 8 + header);
    for (const auto& [name, t] : table.items()) {
        if (!t.is_object() || !t.contains("data_offsets")) continue;  // __metadata__
        Tensor x;
        x.dtype = t.value("dtype", "");
        x.shape = t.at("shape").get<std::vector<int64_t>>();
        const auto off = t.at("data_offsets").get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[1] < off[0] || base + off[1] > size_) {
            err_ = path + ": bad data_offsets for " + name;
            return false;
        }
        x.data = map_ + base + off[0];
        x.bytes = off[1] - off[0];
        tensors_.emplace(name, std::move(x));
    }
    return true;
}

Safetensors::~Safetensors() {
    onebit::npu::unmap_file(map_, size_);
}

const Tensor* Safetensors::find(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool Safetensors::get_f32(const std::string& name, std::vector<float>& out) const {
    const Tensor* t = find(name);
    if (!t) return false;
    size_t elems = 1;
    for (const int64_t d : t->shape) elems *= size_t(d);

    out.clear();
    out.reserve(elems);
    const uint8_t* p = t->data;
    const std::string& dt = t->dtype;
    if (dt == "F32") {
        for (size_t i = 0; i < elems; ++i) {
            float v;
            std::memcpy(&v, p + i * 4, 4);
            out.push_back(v);
        }
    } else if (dt == "F16") {
        for (size_t i = 0; i < elems; ++i) {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            out.push_back(f16_to_f32(h));
        }
    } else if (dt == "BF16") {
        for (size_t i = 0; i < elems; ++i) {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            out.push_back(bf16_to_f32(h));
        }
    } else {
        return false;  // unsupported dtype
    }
    return true;
}

}  // namespace onebit::laya
