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

#include "moe.h"

#include <nlohmann/json.hpp>

#include "pack.h"   // bf16_to_f32
#include "q4nx.h"   // q4nx::kRows/kCols: the chunk grid the geometry is expressed on

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace onebit::npu {

const char* const kExpertGate = "mlp.experts.gate_proj.weight";
const char* const kExpertUp = "mlp.experts.up_proj.weight";
const char* const kExpertDown = "mlp.experts.down_proj.weight";

namespace {

std::string layer_name(int layer, const std::string& suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

}  // namespace

Moe::Moe(const Model& model) : model_(model) {
    const std::string text = model.metadata("moe");
    if (text.empty())
        throw std::runtime_error("model directory has no __metadata__.moe: it is not a MoE "
                                 "container (docs/npu.md, \"Routed expert stacks\")");
    const auto meta = nlohmann::json::parse(text);
    dims_.experts = meta.at("experts").get<int>();
    dims_.used = meta.at("used").get<int>();
    dims_.norm_topk = meta.value("norm_topk_prob", true);
    if (dims_.experts <= 0 || dims_.used <= 0 || dims_.used > dims_.experts)
        throw std::runtime_error("__metadata__.moe: experts/used out of range");
    const auto& tensors = meta.at("tensors");
    if (tensors.empty()) throw std::runtime_error("__metadata__.moe: no routed expert tensors");

    // Every stack is checked against its own tensor now, so expert() cannot fail later.
    for (const auto& [name, t] : tensors.items()) {
        Stack s;
        s.experts = t.at("experts").get<int>();
        s.rows = t.at("rows").get<int>();
        s.cols = t.at("cols").get<int>();
        s.tiles = t.at("tiles").get<int>();
        s.bytes = t.at("bytes").get<uint64_t>();
        s.offsets = t.at("offsets").get<std::vector<uint64_t>>();
        if (s.experts != dims_.experts || s.tiles <= 0 || s.bytes == 0)
            throw std::runtime_error(name + ": metadata disagrees with __metadata__.moe.experts");
        if (s.bytes != uint64_t(s.tiles) * kTileBytes)
            throw std::runtime_error(name + ": bytes is not a whole number of chunks");
        // The GEOMETRY those three fields describe, checked against the tensor's own shape -- this is
        // where a kernel's K and N come from.  A transposed rows/cols pair is the plausible converter
        // mistake (ggml stores [cols, rows, experts], so keeping that field order swaps them) and it
        // changes no byte count, no offset and no coverage; on a real model the tile count does not
        // see it either, because gate/up 768x2048 and down 2048x768 are both 192 tiles (24x8 vs 64x3).
        // Only the shape can tell them apart, and a swapped contraction must never reach a kernel
        // (engine#1152).
        if (s.rows <= 0 || s.cols <= 0 || s.rows % q4nx::kRows != 0 || s.cols % q4nx::kCols != 0)
            throw std::runtime_error(name + ": metadata geometry " + std::to_string(s.rows) + "x" +
                                     std::to_string(s.cols) + " is not on the " +
                                     std::to_string(q4nx::kRows) + "x" +
                                     std::to_string(q4nx::kCols) + " chunk grid");
        const int want_tiles = (s.rows / q4nx::kRows) * (s.cols / q4nx::kCols);
        if (s.tiles != want_tiles)
            throw std::runtime_error(name + ": metadata tiles " + std::to_string(s.tiles) +
                                     " is not (rows/" + std::to_string(q4nx::kRows) + ")*(cols/" +
                                     std::to_string(q4nx::kCols) + ") = " + std::to_string(want_tiles) +
                                     " for the geometry it declares");
        if (int(s.offsets.size()) != s.experts)
            throw std::runtime_error(name + ": offsets does not list one entry per expert");
        const Tensor* tensor = model.find(name);
        if (!tensor) throw std::runtime_error(name + ": listed in __metadata__.moe but absent");
        const std::vector<int64_t> want_shape{int64_t(s.experts) * (s.rows / q4nx::kRows),
                                              s.cols / q4nx::kCols, int64_t(kTileBytes)};
        if (tensor->shape != want_shape)
            throw std::runtime_error(
                name + ": metadata geometry " + std::to_string(s.rows) + "x" +
                std::to_string(s.cols) + " does not describe this tensor (it would be a [" +
                std::to_string(want_shape[0]) + ", " + std::to_string(want_shape[1]) + ", " +
                std::to_string(want_shape[2]) + "] stack, the tensor is [" +
                std::to_string(tensor->shape.size() > 0 ? tensor->shape[0] : 0) + ", " +
                std::to_string(tensor->shape.size() > 1 ? tensor->shape[1] : 0) + ", " +
                std::to_string(tensor->shape.size() > 2 ? tensor->shape[2] : 0) +
                "]): rows/cols look transposed, and a swapped pair hands the kernel K and N exchanged "
                "(engine#1152)");
        // The offsets are measured from the data section, like every data_offsets in the
        // container, so expert 0's offset is the tensor's own start and the rest are relative
        // to it. Checking coverage here means a wrong list cannot reach a kernel.
        const uint64_t start = s.offsets.front();
        for (int e = 0; e < s.experts; ++e) {
            const uint64_t off = s.offsets[size_t(e)] - start;
            if (off + s.bytes > tensor->bytes)
                throw std::runtime_error(name + ": expert " + std::to_string(e) +
                                         " ends past the tensor");
            if (e > 0 && off != s.offsets[size_t(e) - 1] - start + s.bytes)
                throw std::runtime_error(name + ": expert " + std::to_string(e) +
                                         " is not contiguous with its predecessor");
        }
        if (s.offsets.back() - start + s.bytes != tensor->bytes)
            throw std::runtime_error(name + ": the expert list does not cover the tensor");
        s.tensor = tensor;
        stacks_[name] = std::move(s);
    }
}

ExpertChunks Moe::expert(int layer, const std::string& suffix, int e) const {
    const std::string name = layer_name(layer, suffix);
    const auto it = stacks_.find(name);
    if (it == stacks_.end()) throw std::runtime_error(name + ": no expert metadata");
    const Stack& s = it->second;
    if (e < 0 || e >= s.experts) throw std::runtime_error(name + ": expert out of range");
    ExpertChunks c;
    c.data = s.tensor->data + (s.offsets[size_t(e)] - s.offsets.front());
    c.bytes = s.bytes;
    c.rows = s.rows;
    c.cols = s.cols;
    c.tiles = s.tiles;
    return c;
}

void Moe::router_probs(const float* h, int layer, float* probs, float* logits_out) const {
    const Tensor& router = model_.tensor(router_name(layer));
    if (router.dtype != "BF16")
        throw std::runtime_error(router_name(layer) + ": the router must be bf16");
    if (router.shape.size() != 2 || router.shape[0] != dims_.experts ||
        int(router.shape[1]) != model_.dims().hidden)
        throw std::runtime_error(router_name(layer) + ": unexpected router shape");
    const int hidden = int(router.shape[1]);
    const int experts = int(router.shape[0]);
    if (router.bytes != uint64_t(experts) * uint64_t(hidden) * 2)
        throw std::runtime_error(router_name(layer) + ": size does not match its shape");
    const uint16_t* w = reinterpret_cast<const uint16_t*>(router.data);

    float max_logit = -INFINITY;
    for (int e = 0; e < experts; ++e) {
        float acc = 0.0f;
        for (int k = 0; k < hidden; ++k) acc += bf16_to_f32(w[size_t(e) * hidden + k]) * h[k];
        if (logits_out) logits_out[e] = acc;
        probs[e] = acc;
        max_logit = std::max(max_logit, acc);
    }
    double sum = 0.0;
    for (int e = 0; e < experts; ++e) {
        probs[e] = std::exp(probs[e] - max_logit);
        sum += probs[e];
    }
    for (int e = 0; e < experts; ++e) probs[e] = float(probs[e] / sum);
}

void Moe::router_topk(const float* h, int layer, int k, int* ids, float* weights) const {
    if (!h || !ids || !weights) throw std::runtime_error("router_topk: null argument");
    if (k < 1 || k > dims_.used)
        throw std::runtime_error("router_topk: k must be in [1, __metadata__.moe.used]");
    std::vector<float> p(size_t(dims_.experts));
    router_probs(h, layer, p.data());
    std::vector<int> order(size_t(dims_.experts));
    for (int e = 0; e < dims_.experts; ++e) order[size_t(e)] = e;
    // descending by probability, ties by the lower expert id: the order the reference takes
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return p[size_t(a)] > p[size_t(b)]; });
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += p[size_t(order[size_t(i)])];
    for (int i = 0; i < k; ++i) {
        ids[i] = order[size_t(i)];
        const float pi = p[size_t(ids[i])];
        weights[i] = dims_.norm_topk && sum > 0 ? float(pi / sum) : pi;
    }
}

}  // namespace onebit::npu
