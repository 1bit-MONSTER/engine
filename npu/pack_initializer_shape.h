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
// pack_initializer_shape.h -- the SHAPE rule for a 1.8-generation initializer read out of external data
// (engine#1503, the graph-initializer reader). Header-only and dependency-free, so the rule is testable with no
// ONNX parser, no pack and no device -- which is the point: the resolution path must not be written until this rule can
// be shown to refuse the cases it cannot decide.
//
// Why the rule is the hard part and not the reading: those packs zero every `dims` field on the initializers, so all
// that survives is `external_data { offset, length }` against the sibling `.data` file -- a LENGTH, not a shape. A shape
// guessed wrongly imports cleanly and serves garbage, which is the failure mode this thread has recorded three times
// (#1874's unsigned zero points, #1888's wrong quantisation rule, #1993's untied head). So the rule refuses whenever the
// length is consistent with more than one shape, and names what it saw.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::npu {

// A resolved initializer shape. ndim is 1 for the norms and biases, 2 for a weight; d1 is unused at ndim == 1.
struct InitShape {
    bool ok = false;
    int ndim = 0;
    int64_t d0 = 0;
    int64_t d1 = 0;
    std::string why;  // set on refusal: what was measured, in numbers
};

// Names that are 1-D by convention in this genre. The refusal this reader fixes names exactly these two:
// `model.layers.N.input_layernorm.weight` and `model.layers.N.post_attention_layernorm.weight`.
inline bool is_1d_by_name(const std::string& name) {
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0) return true;
    return name.find("norm") != std::string::npos || name.find("Norm") != std::string::npos;
}

// Resolve one initializer's shape from its byte length and the model's known sizes (hidden, intermediate, vocab, ...).
//
//   * a length that is not a whole number of elements is refused (the file and the element type disagree);
//   * a 1-D name (norm/bias) takes [elements] ONLY when that length is one of the known sizes -- a norm is exactly one
//     of them, and a length that matches none is not a norm;
//   * a weight takes [k, elements/k] ONLY when exactly one known size divides it. Zero divisors or two are both a
//     refusal: two would mean two plausible weight shapes, which is the guess this rule exists to avoid.
inline InitShape resolve_init_shape(const std::string& name, int64_t length, int elem_bytes,
                                   const std::vector<int64_t>& known) {
    InitShape s;
    if (length <= 0 || elem_bytes <= 0) {
        s.why = "a non-positive length (" + std::to_string(length) + " B) or element size (" +
                std::to_string(elem_bytes) + " B)";
        return s;
    }
    if (length % elem_bytes != 0) {
        s.why = std::to_string(length) + " B is not a whole number of " + std::to_string(elem_bytes) + "-byte elements";
        return s;
    }
    const int64_t elems = length / elem_bytes;
    if (is_1d_by_name(name)) {
        for (const int64_t k : known)
            if (k > 0 && k == elems) {
                s.ok = true;
                s.ndim = 1;
                s.d0 = elems;
                return s;
            }
        s.why = "a norm/bias of " + std::to_string(elems) + " elements matches none of the model's declared sizes";
        return s;
    }
    int64_t found = 0;
    int ndiv = 0;
    for (const int64_t k : known)
        if (k > 0 && elems % k == 0) {
            ++ndiv;
            found = k;
        }
    if (ndiv == 0) {
        s.why = "a weight of " + std::to_string(elems) + " elements has no declared dimension that divides it";
        return s;
    }
    if (ndiv > 1) {
        std::string all;
        for (const int64_t k : known)
            if (k > 0 && elems % k == 0) all += (all.empty() ? "" : ", ") + std::to_string(k);
        s.why = "a weight of " + std::to_string(elems) + " elements is consistent with several declared dimensions (" +
                all + "), so the shape cannot be decided from the length";
        return s;
    }
    s.ok = true;
    s.ndim = 2;
    s.d0 = found;
    s.d1 = elems / found;
    return s;
}

}  // namespace onebit::npu
