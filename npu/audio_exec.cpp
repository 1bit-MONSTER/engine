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
// audio_exec.cpp -- the host f32 executor for an audio pack's graph (see audio_exec.h for the contract).

#include "audio_exec.h"
#include "audio_parallel.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "forward/include/audio_ops.h"
#include "file_map.h"

namespace onebit::npu::pack_import {
namespace {

struct Tensor {
    std::vector<int64_t> dims;
    std::vector<float> data;      // f32 payload
    std::vector<int64_t> i64;     // int64 payload (tokens, indices, shapes); exactly one of the two is used
    std::vector<uint8_t> b;       // bool payload (a mask)
    size_t count() const { return i64.empty() ? (b.empty() ? data.size() : b.size()) : i64.size(); }
};

size_t nelem(const std::vector<int64_t>& d) {
    size_t n = 1;
    for (int64_t v : d) n *= size_t(v < 0 ? 0 : v);
    return n;
}

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }

// numpy-style right-aligned broadcast of `small` against `big`; returns the result dims and, per output index, the source
// index. Only the two forms this graph uses: equal, or `small` has trailing dims of 1/size and fewer ranks.
struct Broadcast {
    std::vector<int64_t> dims;
    std::vector<size_t> stride_big, stride_small;   // 0 where the axis is broadcast
};

Broadcast plan_broadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    Broadcast p;
    const size_t r = std::max(a.size(), b.size());
    p.dims.assign(r, 1);
    p.stride_big.assign(r, 0);
    p.stride_small.assign(r, 0);
    for (size_t i = 0; i < r; ++i) {
        const int64_t da = (i < r - a.size()) ? 1 : a[i - (r - a.size())];
        const int64_t db = (i < r - b.size()) ? 1 : b[i - (r - b.size())];
        if (da != db && da != 1 && db != 1) fail("broadcast mismatch " + std::to_string(da) + " vs " + std::to_string(db));
        p.dims[i] = da == 1 ? db : da;
    }
    size_t sb = 1, ss = 1;
    for (size_t i = r; i-- > 0;) {
        const int64_t da = (i < r - a.size()) ? 1 : a[i - (r - a.size())];
        const int64_t db = (i < r - b.size()) ? 1 : b[i - (r - b.size())];
        p.stride_big[i] = da == 1 ? 0 : sb;
        p.stride_small[i] = db == 1 ? 0 : ss;
        sb *= size_t(da);
        ss *= size_t(db);
    }
    return p;
}

// Flat index into `src` for the right-aligned broadcast of `src` against an output of dims `od` at multi-index `idx`.
// numpys rule: a missing or unit axis is read as index 0.
size_t bcast_index(const std::vector<int64_t>& od, const std::vector<int64_t>& src, const std::vector<size_t>& idx) {
    size_t off = 0, stride = 1;
    for (size_t i = src.size(); i-- > 0;) {
        const size_t from_right = src.size() - i;            // 1 = last axis
        const size_t oi = od.size() - from_right;            // the output axis it aligns with
        const int64_t d = src[i];
        const size_t k = (oi < od.size() && d != 1) ? idx[oi] : 0;
        off += k * stride;
        stride *= size_t(d);
    }
    return off;
}

// Elements per parallel unit of the elementwise loops below: large enough that a unit is not dominated by the fork, small
// enough that a [1, 1500, 768] activation splits into ~70 units.
constexpr size_t kElemUnit = 16384;

// engine#2366: one row (the output's last axis) at a time, the row's source offsets computed once from its index and then
// stepped by the last-axis strides, rows split over threads. Each element is still the one float operation of the serial
// loop on the same two operands, so the result is bit-identical at any thread count (tests/audio_parallel_test.cpp).
template <typename Op>
void broadcast_rows(const Tensor& a, const Tensor& b, const Broadcast& p, Tensor* out, Op op) {
    const size_t rank = p.dims.size();
    const size_t width = rank == 0 ? 1 : size_t(p.dims[rank - 1]);
    const size_t rows = width == 0 ? 0 : out->data.size() / width;
    const size_t sa = rank == 0 ? 0 : p.stride_big[rank - 1], sb = rank == 0 ? 0 : p.stride_small[rank - 1];
    const size_t rows_per_unit = std::max<size_t>(1, kElemUnit / std::max<size_t>(1, width));
    const float* ad = a.data.data();
    const float* bd = b.data.data();
    float* od = out->data.data();
    parallel_units((rows + rows_per_unit - 1) / rows_per_unit, [&](size_t u) {
        const size_t r0 = u * rows_per_unit, r1 = std::min(rows, r0 + rows_per_unit);
        for (size_t row = r0; row < r1; ++row) {
            size_t ia = 0, ib = 0, rem = row;
            for (size_t d = rank > 0 ? rank - 1 : 0; d-- > 0;) {   // the row's index on every axis but the last
                const size_t i = rem % size_t(p.dims[d]);
                rem /= size_t(p.dims[d]);
                ia += i * p.stride_big[d];
                ib += i * p.stride_small[d];
            }
            float* o = od + row * width;
            // A last-axis stride of 0 (a scalar, a column) is one value for the whole row: read it once, which lets the
            // loop vectorise (scalar Mul on 1x1500x3072, one thread: 1.39 -> 0.6 ms). Same operation per element.
            if (sb == 0) {
                const float y = bd[ib];
                for (size_t j = 0; j < width; ++j) o[j] = op(ad[ia + j * sa], y);
            } else if (sa == 0) {
                const float x = ad[ia];
                for (size_t j = 0; j < width; ++j) o[j] = op(x, bd[ib + j * sb]);
            } else {
                for (size_t j = 0; j < width; ++j) o[j] = op(ad[ia + j * sa], bd[ib + j * sb]);
            }
        }
    });
}

// engine#2366: equal dims (residual adds, the GELU chain) and a scalar operand need no index arithmetic at all, so they
// take a flat loop (#2383: 66.76 -> 7.10 ms for Add/Mul/Div on 1x1500x3072). Every other broadcast takes the row loop
// above. Both split over the executor's threads in kElemUnit ranges or row blocks, and each element is the serial loop's
// one float operation on the same operands, so the output is bit-identical at any thread count.
template <typename Op>
void broadcast_apply(const Tensor& a, const Tensor& b, const Broadcast& p, Tensor* out, Op op) {
    const size_t n = out->data.size();
    const float* ad = a.data.data();
    const float* bd = b.data.data();
    float* od = out->data.data();
    const bool equal = a.dims == p.dims && b.dims == p.dims;
    if (equal || b.data.size() == 1) {
        parallel_units((n + kElemUnit - 1) / kElemUnit, [&](size_t u) {
            const size_t i1 = std::min(n, (u + 1) * kElemUnit);
            if (equal) {
                for (size_t i = u * kElemUnit; i < i1; ++i) od[i] = op(ad[i], bd[i]);
            } else {
                const float y = bd[0];
                for (size_t i = u * kElemUnit; i < i1; ++i) od[i] = op(ad[i], y);
            }
        });
        return;
    }
    broadcast_rows(a, b, p, out, op);
}

void broadcast_binary(const Tensor& a, const Tensor& b, const std::string& op, Tensor* out) {
    const Broadcast p = plan_broadcast(a.dims, b.dims);
    out->dims = p.dims;
    out->data.resize(nelem(p.dims));
    if (op == "Add") broadcast_apply(a, b, p, out, [](float x, float y) { return x + y; });
    else if (op == "Sub") broadcast_apply(a, b, p, out, [](float x, float y) { return x - y; });
    else if (op == "Mul") broadcast_apply(a, b, p, out, [](float x, float y) { return x * y; });
    else broadcast_apply(a, b, p, out, [](float x, float y) { return x / y; });
}

// y[i] = f(x[i]) over kElemUnit-sized ranges on the executor's threads (engine#2366); per element the serial arithmetic.
template <typename Fn>
void map_elements(const std::vector<float>& x, std::vector<float>* y, Fn f) {
    y->resize(x.size());
    const float* xd = x.data();
    float* yd = y->data();
    const size_t n = x.size();
    parallel_units((n + kElemUnit - 1) / kElemUnit, [&](size_t u) {
        const size_t i1 = std::min(n, (u + 1) * kElemUnit);
        for (size_t i = u * kElemUnit; i < i1; ++i) yd[i] = f(xd[i]);
    });
}

std::vector<int64_t> dims_of(const OnnxInit& t) { return t.dims; }

// Copies an initializer to f32, following external_data into the pack's side file when the bytes are not inline.
void load_init_f32(const OnnxGraphInfo& g, const OnnxInit& t, Tensor* dst) {
    const uint8_t* src = t.raw;
    size_t len = t.raw_len;
    size_t size = 0;
    std::string why;
    const uint8_t* mapped = nullptr;
    if (!t.location.empty()) {
        mapped = map_file(g.dir + "/" + t.location, size, &why);
        if (mapped == nullptr) fail("external_data " + t.location + ": " + why);
        const uint64_t off = t.offset, want = t.length;
        if (off > size || want > size - off) fail("external_data " + t.location + ": range past the end");
        src = mapped + off;
        len = size_t(want);
        if (len == 0) len = size_t(size - off);
    }
    if (src == nullptr) fail("initializer '" + t.name + "' has no data");
    const size_t n = nelem(t.dims);
    if (t.dtype == 7) {   // INT64 -- indices and shape operands; the payload stays integer
        dst->i64.assign(n, 0);
        if (len < n * 8) fail("int64 initializer '" + t.name + "' is short");
        std::memcpy(dst->i64.data(), src, n * 8);
        if (mapped != nullptr) unmap_file(mapped, size);
        return;
    }
    dst->data.assign(n, 0.0f);
    if (t.dtype == 1) {   // FLOAT
        if (len < n * 4) fail("initializer '" + t.name + "' is short: " + std::to_string(len) + " < " + std::to_string(n * 4));
        std::memcpy(dst->data.data(), src, n * 4);
    } else if (t.dtype == 10) {  // FLOAT16
        if (len < n * 2) fail("f16 initializer '" + t.name + "' is short");
        for (size_t i = 0; i < n; ++i) {
            uint16_t h = 0;
            std::memcpy(&h, src + i * 2, 2);
            const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
            uint32_t bits;
            if (exp == 0) {
                bits = sign | (man ? (uint32_t(127 - 15) << 23) + (man << 13) : 0);
            } else if (exp == 31) {
                bits = sign | 0x7f800000u | (man << 13);
            } else {
                bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
            }
            std::memcpy(&dst->data[i], &bits, 4);
        }
    } else {
        fail("initializer '" + t.name + "' has dtype " + std::to_string(t.dtype) + ", which this executor does not load");
    }
    if (mapped != nullptr) unmap_file(mapped, size);
}

void load_init_i64(const OnnxGraphInfo& g, const OnnxInit& t, std::vector<int64_t>* dst) {
    const uint8_t* src = t.raw;
    size_t len = t.raw_len;
    size_t size = 0;
    std::string why;
    const uint8_t* mapped = nullptr;
    if (!t.location.empty()) {
        mapped = map_file(g.dir + "/" + t.location, size, &why);
        if (mapped == nullptr) fail("external_data " + t.location + ": " + why);
        if (t.offset > size || t.length > size - t.offset) fail("external_data range past the end");
        src = mapped + t.offset;
        len = size_t(t.length);
    }
    if (t.dtype != 7) fail("'" + t.name + "' is not INT64");
    const size_t n = nelem(t.dims);
    if (src == nullptr || len < n * 8) fail("int64 initializer '" + t.name + "' is short");
    dst->assign(n, 0);
    std::memcpy(dst->data(), src, n * 8);
    if (mapped != nullptr) unmap_file(mapped, size);
}

}  // namespace

namespace {

bool run_graph_impl(const OnnxGraphInfo& g, std::map<std::string, Tensor> t, const std::string& out_name, AudioRunResult* out,
                    std::string* why, AudioGemm* dev = nullptr) {
    try {

        std::map<std::string, const OnnxInit*> inits;
        for (const auto& ti : g.init_tensors) inits[ti.name] = &ti;

        auto need = [&](const std::string& name) -> const Tensor& {
            const auto it = t.find(name);
            if (it == t.end()) fail("node input '" + name + "' was never produced");
            return it->second;
        };
        auto init_f32 = [&](const std::string& name) -> const Tensor& {
            auto it = t.find(name);
            if (it != t.end()) return it->second;
            const auto ii = inits.find(name);
            if (ii == inits.end()) fail("'" + name + "' is neither a tensor nor an initializer");
            Tensor v;
            v.dims = dims_of(*ii->second);
            load_init_f32(g, *ii->second, &v);
            return t.emplace(name, std::move(v)).first->second;
        };

        // A node operand is either a tensor an earlier node produced or an initializer (a weight, an index vector, a
        // scalar attribute operand). `any_t` is the one spelling of that rule.
        auto any_t = [&](const std::string& name) -> const Tensor& {
            return t.count(name) != 0 ? need(name) : init_f32(name);
        };

        // ONEBIT_AUDIO_OP_TIMES=1 (engine#1783): wall time per op type over this graph, printed once at the end, so a
        // host split ("Conv 3.74 s, Softmax 2.56 s") is read from the executor that runs the window rather than inferred.
        const char* ot = std::getenv("ONEBIT_AUDIO_OP_TIMES");
        const bool op_times = ot != nullptr && ot[0] != '\0' && ot[0] != '0';
        std::map<std::string, std::pair<double, long>> op_ms;   // op -> (ms, calls)

        // engine#1783: offer one product to the device. True when the device ran it into `c`; false when the host must, with
        // the reason counted. A device without a `gemm` is no device.
        auto offer = [&](const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                         std::vector<float>* c) -> bool {
            if (dev == nullptr) return false;
            std::string reason;
            if (dev->gemm && dev->gemm(key, A, M, K, B, N, c, &reason)) {
                if (c->size() != size_t(M) * size_t(N))
                    fail("device gemm for '" + key + "' wrote " + std::to_string(c->size()) + " floats, not " +
                         std::to_string(M) + " x " + std::to_string(N));
                ++dev->device;
                return true;
            }
            ++dev->host;
            ++dev->host_why[reason.empty() ? std::string("declined without a reason") : reason];
            return false;
        };

        // Every node ends here, including the products the device ran (they leave the loop early), so ONEBIT_AUDIO_OP_TIMES
        // counts a device-run MatMul or Conv under its op like a host one.
        // ONEBIT_AUDIO_NODE_TIMES=1 (engine#2366): one line per node, in graph order, with the output shape. The op totals
        // below collapse the 72 projections and the 24 attention products into one `MatMul` row, and apportioning that row by
        // node count is how a 0.39 s family was reported as 1.7 s. The executor already times every node to build the totals,
        // so this costs a string per node and nothing else.
        const char* nt = std::getenv("ONEBIT_AUDIO_NODE_TIMES");
        const bool node_times = nt != nullptr && nt[0] != '\0' && nt[0] != '0';
        std::vector<std::string> node_lines;
        auto finish_node = [&](const std::string& op, const std::string& name, Tensor& res,
                               std::chrono::steady_clock::time_point t0) {
            if (node_times) {
                std::string shape;
                for (int64_t d : res.dims) shape += (shape.empty() ? "" : "x") + std::to_string(d);
                char line[220];
                std::snprintf(line, sizeof line, "[audio-node] %4d %-18s %-22s %9.3f ms\n", int(node_lines.size()) + 1,
                              op.c_str(), shape.c_str(),
                              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                node_lines.emplace_back(line);
            }
            t[name] = std::move(res);
            if (op_times) {
                auto& e = op_ms[op];
                e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                e.second++;
            }
        };

        int node_index = 0;
        for (const auto& n : g.nodes) {
            ++node_index;
            const auto op_t0 = std::chrono::steady_clock::now();
            const auto where = [&]() { return "node " + std::to_string(node_index) + " (" + n.op + ")"; };
            if (n.outs.empty()) fail(where() + " has no output");
            const std::string& oname = n.outs[0];
            Tensor r;
            if (n.op == "MatMul") {
                const Tensor& a = need(n.ins[0]);
                const Tensor& b = t.count(n.ins[1]) != 0 ? need(n.ins[1]) : init_f32(n.ins[1]);
                if (a.dims.size() < 2 || b.dims.size() < 2) fail(where() + ": needs rank >= 2 on both sides");
                const size_t ra = a.dims.size(), rb = b.dims.size();
                const int64_t M = a.dims[ra - 2], K = a.dims.back(), K2 = b.dims[rb - 2], N = b.dims.back();
                if (K != K2) fail(where() + ": " + std::to_string(K) + " vs " + std::to_string(K2));
                // Attention's q.k^T and scores.v are 4-D: (batch, heads, M|K, K|N). The batch dims broadcast, the last two
                // are a plain GEMM, so one projection covers both the layer weights and the attention products.
                const std::vector<int64_t> ab(a.dims.begin(), a.dims.end() - 2), bb(b.dims.begin(), b.dims.end() - 2);
                const Broadcast bp = plan_broadcast(ab, bb);
                r.dims = bp.dims;
                r.dims.push_back(M);
                r.dims.push_back(N);
                // A projection (B a rank-2 weight) is one GEMM over every row of A, whatever A's leading axes are.
                if (dev != nullptr) {
                    if (rb == 2 && inits.count(n.ins[1]) != 0) {
                        std::vector<float> c;
                        if (offer(n.ins[1], a.data.data(), int64_t(a.data.size() / size_t(K)), K, b.data.data(), N, &c)) {
                            r.dims = a.dims;
                            r.dims.back() = N;
                            r.data = std::move(c);
                            finish_node(n.op, oname, r, op_t0);
                            continue;
                        }
                    } else {
                        ++dev->host;
                        ++dev->host_why["MatMul with an activation as B (attention product): not offered"];
                    }
                }
                r.data.assign(nelem(r.dims), 0.0f);
                // Rows are independent, so the work splits over (batch, row) without touching any element's arithmetic.
                const size_t batches = nelem(bp.dims);
                const size_t rows = size_t(std::max<int64_t>(M, 1));
                // engine#2366: one row per unit streams every `b[k]` row once PER output row -- M passes over the whole
                // weight matrix. Grouping four rows per unit shares that streamed row; measured on a whisper projection
                // (M=1500 K=768 N=768, best of 3, one process): 189.8 -> 138.7 ms, 9.3 -> 12.8 GFLOP/s, 1.35x, and the
                // operands and their order are untouched so the result is bit-identical (max|diff| 0).
                // The zero-row skip is kept, but only when all four rows are zero in that k: skipping one of four would
                // change nothing arithmetically, yet the check is on the group so the common case reads one weight row.
                constexpr size_t kRowGroup = 4;
                const size_t groups = (rows + kRowGroup - 1) / kRowGroup;
                parallel_units(batches * groups, [&](size_t u) {
                    const size_t bi = u / groups;
                    const size_t g = u % groups;
                    std::vector<size_t> idx(bp.dims.size(), 0);
                    size_t rem = bi;
                    for (size_t d = bp.dims.size(); d-- > 0;) {   // row-major unflatten of the batch index
                        const size_t dim = size_t(bp.dims[d] ? bp.dims[d] : 1);
                        idx[d] = rem % dim;
                        rem /= dim;
                    }
                    size_t oa = 0, ob = 0;
                    for (size_t d = 0; d < idx.size(); ++d) {
                        oa += idx[d] * bp.stride_big[d];
                        ob += idx[d] * bp.stride_small[d];
                    }
                    const float* ap = a.data.data() + oa * size_t(M * K);
                    const float* bpp = b.data.data() + ob * size_t(K * N);
                    const int64_t i0 = int64_t(g * kRowGroup), i1 = std::min<int64_t>(M, i0 + int64_t(kRowGroup));
                    float* r0 = r.data.data() + bi * size_t(M * N) + size_t(i0) * size_t(N);
                    if (i1 - i0 == 4) {
                        const float* a0 = ap + size_t(i0) * size_t(K);
                        for (int64_t k = 0; k < K; ++k) {
                            const float v0 = a0[size_t(k)], v1 = a0[size_t(K) + size_t(k)];
                            const float v2 = a0[size_t(2 * K) + size_t(k)], v3 = a0[size_t(3 * K) + size_t(k)];
                            if ((v0 == 0.0f) && (v1 == 0.0f) && (v2 == 0.0f) && (v3 == 0.0f)) continue;
                            const float* br = bpp + size_t(k) * size_t(N);
                            for (int64_t j = 0; j < N; ++j) {
                                const float bv = br[j];
                                r0[j] += v0 * bv;
                                r0[N + j] += v1 * bv;
                                r0[2 * N + j] += v2 * bv;
                                r0[3 * N + j] += v3 * bv;
                            }
                        }
                    } else {
                        for (int64_t i = i0; i < i1; ++i) {
                            float* rr = r0 + size_t(i - i0) * size_t(N);
                            for (int64_t k = 0; k < K; ++k) {
                                const float av = ap[size_t(i) * size_t(K) + size_t(k)];
                                if (av == 0.0f) continue;
                                const float* br = bpp + size_t(k) * size_t(N);
                                for (int64_t j = 0; j < N; ++j) rr[j] += av * br[j];
                            }
                        }
                    }
                });
            } else if (n.op == "Add" || n.op == "Mul" || n.op == "Div" || n.op == "Sub") {
                // An operand is either a tensor an earlier node produced or an initializer (a bias vector, the 1/sqrt(d)
                // scalar); init_f32() handles the second without a separate code path at every call site.
                auto operand = [&](const std::string& name) -> const Tensor& {
                    return t.count(name) != 0 ? need(name) : init_f32(name);
                };
                if (n.ins.size() < 2) fail(where() + ": needs two operands");
                broadcast_binary(operand(n.ins[0]), operand(n.ins[1]), n.op, &r);
            } else if (n.op == "Erf") {
                const Tensor& a = need(n.ins[0]);
                r.dims = a.dims;
                map_elements(a.data, &r.data, [](float v) { return std::erf(v); });
            } else if (n.op == "Gelu") {
                const Tensor& a = need(n.ins[0]);
                r.dims = a.dims;
                const auto ap = n.attrs.find("approximate");
                (void)ap;   // ONNX spells it as a string; the export here is 'none', which is what audio_ops::gelu_erf is
                map_elements(a.data, &r.data, [](float v) { return audio::gelu_erf(v); });
            } else if (n.op == "LayerNormalization") {
                const Tensor& a = need(n.ins[0]);
                const Tensor& scale = init_f32(n.ins[1]);
                const Tensor* bias = (n.ins.size() > 2 && !n.ins[2].empty()) ? &init_f32(n.ins[2]) : nullptr;
                const int64_t axis = n.attrs.count("axis") ? n.attrs.at("axis") : -1;
                if (axis != -1 && axis != int64_t(a.dims.size()) - 1) fail(where() + ": only the last axis is implemented");
                const int64_t nrm = a.dims.back();
                const float eps = n.attrs_float.count("epsilon") ? n.attrs_float.at("epsilon") : 1e-5f;
                if (int64_t(scale.data.size()) != nrm) fail(where() + ": scale is not the normalised width");
                r.dims = a.dims;
                r.data.resize(a.data.size());
                parallel_units(a.data.size() / size_t(nrm), [&](size_t row) {   // per row, as above (engine#1783)
                    audio::layer_norm(a.data.data() + row * size_t(nrm), scale.data.data(),
                                      bias != nullptr ? bias->data.data() : nullptr, int(nrm), eps,
                                      r.data.data() + row * size_t(nrm));
                });
            } else if (n.op == "Softmax") {
                const Tensor& a = need(n.ins[0]);
                const int64_t axis = n.attrs.count("axis") ? n.attrs.at("axis") : -1;
                if (axis != -1 && axis != int64_t(a.dims.size()) - 1) fail(where() + ": only the last axis is implemented");
                r = a;
                const int64_t nrm = a.dims.back();
                // engine#1783: rows are independent and each keeps its own serial arithmetic, so splitting them over
                // threads moves no bit (attention's 12 x 1500 rows of 1500 per layer were the serial term).
                parallel_units(r.data.size() / size_t(nrm), [&](size_t row) {
                    audio::softmax_row(r.data.data() + row * size_t(nrm), int(nrm));
                });
            } else if (n.op == "Transpose") {
                const Tensor& a = need(n.ins[0]);
                const auto it = n.attrs_list.find("perm");
                std::vector<int64_t> perm;
                if (it != n.attrs_list.end()) perm = it->second;
                else
                    for (size_t i = a.dims.size(); i-- > 0;) perm.push_back(int64_t(i));
                if (perm.size() != a.dims.size()) fail(where() + ": perm rank mismatch");
                r.dims.resize(perm.size());
                for (size_t i = 0; i < perm.size(); ++i) r.dims[i] = a.dims[size_t(perm[i])];
                r.data.resize(a.data.size());
                std::vector<size_t> in_stride(perm.size(), 1), out_stride(perm.size(), 1);
                for (size_t i = perm.size(); i-- > 1;) {
                    in_stride[i - 1] = in_stride[i] * size_t(a.dims[i]);
                    out_stride[i - 1] = out_stride[i] * size_t(r.dims[i]);
                }
                // engine#2366: the inner loop walked the output one element at a time and recomputed the source offset
                // with a dot product over all ranks (plus a carry) for every element. The output's LAST axis has a
                // constant source stride -- in_stride[perm[rank-1]] -- whatever the perm is, so it is one strided gather
                // per row: the offset is computed once, then stepped. Same reads, same writes, same order, so the result
                // is bit-identical; only the index arithmetic leaves the inner loop.
                const size_t rank = perm.size();
                const size_t width = rank == 0 ? 1 : size_t(r.dims[rank - 1]);
                const size_t rows_out = width == 0 ? 0 : r.data.size() / width;
                const size_t step = rank == 0 ? 0 : in_stride[size_t(perm[rank - 1])];
                parallel_units(rows_out, [&](size_t row) {
                    size_t src = 0, rem = row;
                    for (size_t d = rank; d-- > 1;) {
                        const size_t dim = size_t(r.dims[d - 1]);
                        src += (rem % dim) * in_stride[size_t(perm[d - 1])];
                        rem /= dim;
                    }
                    float* o = r.data.data() + row * width;
                    for (size_t j = 0; j < width; ++j) o[j] = a.data[src + j * step];
                });
            } else if (n.op == "Reshape") {
                const Tensor& a = need(n.ins[0]);
                std::vector<int64_t> want;
                auto ii = inits.find(n.ins[1]);
                if (ii == inits.end()) fail(where() + ": target shape is not an initializer");
                load_init_i64(g, *ii->second, &want);
                std::vector<int64_t> dims(want.size());
                int64_t infer = -1;
                size_t known = 1;
                for (size_t i = 0; i < want.size(); ++i) {
                    int64_t d = want[i];
                    if (d == 0) d = a.dims[i];
                    if (d == -1) {
                        if (infer >= 0) fail(where() + ": two -1");
                        infer = int64_t(i);
                        d = 1;
                    }
                    dims[i] = d;
                    known *= size_t(d);
                }
                if (infer >= 0) dims[size_t(infer)] = int64_t(a.data.size() / std::max<size_t>(known, 1));
                if (nelem(dims) != a.data.size()) fail(where() + ": reshape changes the element count");
                r.dims = dims;
                r.data = a.data;
            } else if (n.op == "Conv") {
                const Tensor& a = need(n.ins[0]);
                const Tensor& w = init_f32(n.ins[1]);
                const Tensor* bias = (n.ins.size() > 2 && !n.ins[2].empty()) ? &init_f32(n.ins[2]) : nullptr;
                const auto gs = n.attrs_list.find("strides");
                const auto gp = n.attrs_list.find("pads");
                const int stride = (gs != n.attrs_list.end() && !gs->second.empty()) ? int(gs->second[0]) : 1;
                const int pad = (gp != n.attrs_list.end() && !gp->second.empty()) ? int(gp->second[0]) : 0;
                if (a.dims.size() != 3 || w.dims.size() != 3) fail(where() + ": only 1-D temporal Conv on (N,C,T)");
                const int64_t cin = a.dims[1], T = a.dims[2];
                const int64_t cout = w.dims[0], k = w.dims[2];
                if (w.dims[1] != cin) fail(where() + ": kernel channels do not match the input");
                const int64_t tout = audio::conv1d_out_len(int(T), int(k), stride, pad);
                r.dims = {a.dims[0], cout, tout};
                r.data.resize(size_t(a.dims[0]) * size_t(cout) * size_t(tout));
                const auto ga = n.attrs.find("group");
                if (ga != n.attrs.end() && ga->second != 1) fail(where() + ": only group=1 is implemented");
                if (dev != nullptr) {
                    // The im2col GEMM: (tout, cin*k) x (cin*k, cout), then out[co][t] = bias[co] + C[t][co].
                    const int64_t kk = cin * k;
                    std::vector<float> bkn(size_t(kk) * size_t(cout));
                    audio::conv1d_weight_as_kn(w.data.data(), int(cout), int(cin), int(k), bkn.data());
                    std::vector<float> acol(size_t(tout) * size_t(kk));
                    bool all = true;
                    for (int64_t n0 = 0; n0 < a.dims[0] && all; ++n0) {
                        audio::conv1d_im2col(a.data.data() + size_t(n0) * size_t(cin) * size_t(T), int(cin), int(T), int(k),
                                             stride, pad, acol.data());
                        std::vector<float> c;
                        if (!offer(n.ins[1] + "#conv", acol.data(), tout, kk, bkn.data(), cout, &c)) {
                            all = false;
                            break;
                        }
                        float* o = r.data.data() + size_t(n0) * size_t(cout) * size_t(tout);
                        for (int64_t co = 0; co < cout; ++co) {
                            const float bv = bias != nullptr ? bias->data[size_t(co)] : 0.0f;
                            for (int64_t tt = 0; tt < tout; ++tt)
                                o[size_t(co) * size_t(tout) + size_t(tt)] = bv + c[size_t(tt) * size_t(cout) + size_t(co)];
                        }
                    }
                    if (all) {
                        finish_node(n.op, oname, r, op_t0);
                        continue;
                    }
                }
                // engine#1783: one output channel per unit. conv1d with cout = 1 on that channel's weights, bias and output
                // row runs exactly the loop the full call runs for it, so the result is bit-identical; the mel stem's two
                // Convs were a serial 3.2 G multiply-adds per 30 s window.
                const size_t units = size_t(a.dims[0]) * size_t(cout);
                parallel_units(units, [&](size_t u) {
                    const size_t n0 = u / size_t(cout), co = u % size_t(cout);
                    audio::conv1d(a.data.data() + n0 * size_t(cin) * size_t(T), int(cin), int(T),
                                  w.data.data() + co * size_t(cin) * size_t(k),
                                  bias != nullptr ? bias->data.data() + co : nullptr, 1, int(k), stride, pad,
                                  r.data.data() + (n0 * size_t(cout) + co) * size_t(tout));
                });
            } else if (n.op == "Gather") {
                // The token embedding: axis 0, indices are the int64 token window.
                const Tensor& a = any_t(n.ins[0]);
                const Tensor& ix = any_t(n.ins[1]);
                if (!ix.i64.empty()) {
                    const int64_t axis = n.attrs.count("axis") ? n.attrs.at("axis") : 0;
                    if (axis != 0 || a.dims.empty()) fail(where() + ": only axis 0 is implemented");
                    const int64_t rows = a.dims[0];
                    const size_t row = a.data.size() / size_t(rows);
                    r.dims = ix.dims;
                    for (size_t i = 1; i < a.dims.size(); ++i) r.dims.push_back(a.dims[i]);
                    r.data.resize(ix.i64.size() * row);
                    for (size_t i = 0; i < ix.i64.size(); ++i) {
                        const int64_t src = ix.i64[i];
                        if (src < 0 || src >= rows) fail(where() + ": index out of range");
                        std::memcpy(r.data.data() + i * row, a.data.data() + size_t(src) * row, row * 4);
                    }
                } else {
                    fail(where() + ": indices are not int64");
                }
            } else if (n.op == "Equal") {
                // Three forms in this family: the token window against <|endoftext|> (int64), and the causal-mask constant
                // against itself (float, exact equality -- which is the ONNX semantic and the whole point of the pattern).
                const Tensor& a = any_t(n.ins[0]);
                const Tensor& b = any_t(n.ins[1]);
                if (a.count() != b.count() && a.count() != 1 && b.count() != 1)
                    fail(where() + ": only scalar broadcast is implemented");
                r.dims = a.dims.size() >= b.dims.size() ? a.dims : b.dims;
                const size_t n_out = std::max(a.count(), b.count());
                r.b.assign(n_out, 0);
                for (size_t i = 0; i < n_out; ++i) {
                    const size_t ia = a.count() == 1 ? 0 : i, ib = b.count() == 1 ? 0 : i;
                    if (!a.i64.empty() && !b.i64.empty()) r.b[i] = a.i64[ia] == b.i64[ib] ? 1 : 0;
                    else if (!a.b.empty() && !b.b.empty()) r.b[i] = a.b[ia] == b.b[ib] ? 1 : 0;
                    else if (!a.data.empty() && !b.data.empty()) r.b[i] = a.data[ia] == b.data[ib] ? 1 : 0;
                    else fail(where() + ": operands are of different kinds");
                }
            } else if (n.op == "Not") {
                const Tensor& a = any_t(n.ins[0]);
                if (a.b.empty()) fail(where() + ": operand is not a mask");
                r.dims = a.dims;
                r.b.resize(a.b.size());
                for (size_t i = 0; i < a.b.size(); ++i) r.b[i] = a.b[i] ? 0 : 1;
            } else if (n.op == "Where") {
                // The causal-mask pattern in this family is Where(mask, -inf_scalar, 0_scalar): the *condition* carries the
                // shape, so broadcasting only the two value operands would collapse the result to one element.
                const Tensor& c = any_t(n.ins[0]);
                const Tensor& a = any_t(n.ins[1]);
                const Tensor& bb2 = any_t(n.ins[2]);
                if (c.b.empty()) fail(where() + ": condition is not a mask");
                std::vector<int64_t> vals = plan_broadcast(a.dims, bb2.dims).dims;
                r.dims = plan_broadcast(vals, c.dims).dims;
                const size_t total = nelem(r.dims);
                const bool use_i64 = !a.i64.empty();
                if (use_i64) r.i64.resize(total, 0);
                else r.data.resize(total, 0.0f);
                std::vector<size_t> idx(r.dims.size(), 0);
                for (size_t o = 0; o < total; ++o) {
                    const size_t ci = bcast_index(r.dims, c.dims, idx);
                    const size_t ia = bcast_index(r.dims, a.dims, idx);
                    const size_t ib = bcast_index(r.dims, bb2.dims, idx);
                    if (use_i64) r.i64[o] = c.b[ci] ? a.i64[ia] : bb2.i64[ib];
                    else r.data[o] = c.b[ci] ? a.data[ia] : bb2.data[ib];
                    for (size_t d = idx.size(); d-- > 0;) {
                        if (++idx[d] < size_t(r.dims[d])) break;
                        idx[d] = 0;
                    }
                }
            } else if (n.op == "Trilu") {
                // The causal mask. ONNX zeroes the masked triangle, which for the float form is the "large constant" the
                // following Equal/Where pair looks for; an int64 input is zeroed the same way.
                const Tensor& a = any_t(n.ins[0]);
                const int64_t k = n.attrs.count("k") ? n.attrs.at("k") : 0;
                const bool upper = n.attrs.count("upper") ? n.attrs.at("upper") != 0 : true;
                if (a.dims.size() < 2) fail(where() + ": needs a matrix");
                const int64_t R = a.dims[a.dims.size() - 2], C = a.dims.back();
                r.dims = a.dims;
                r.data = a.data;
                r.i64 = a.i64;
                const size_t mat = size_t(R * C);
                for (size_t base = 0; base + mat <= a.count(); base += mat) {
                    for (int64_t i = 0; i < R; ++i) {
                        for (int64_t j = 0; j < C; ++j) {
                            const bool keep = upper ? (j >= i + k) : (j <= i + k);
                            if (keep) continue;
                            if (!r.i64.empty()) r.i64[base + size_t(i * C + j)] = 0;
                            else r.data[base + size_t(i * C + j)] = 0.0f;
                        }
                    }
                }
            } else if (n.op == "Slice") {
                const Tensor& a = any_t(n.ins[0]);
                std::vector<int64_t> starts, ends, axes, steps;
                auto shape_op = [&](size_t which, std::vector<int64_t>* dst) {
                    auto ii = inits.find(n.ins[which]);
                    if (ii == inits.end()) fail(where() + ": slice bounds must be initializers");
                    load_init_i64(g, *ii->second, dst);
                };
                if (n.ins.size() < 3) fail(where() + ": needs starts/ends");
                shape_op(1, &starts);
                shape_op(2, &ends);
                if (n.ins.size() > 3) shape_op(3, &axes);
                if (n.ins.size() > 4) shape_op(4, &steps);
                if (axes.empty())
                    for (size_t i = 0; i < starts.size(); ++i) axes.push_back(int64_t(i));
                if (steps.empty())
                    for (size_t i = 0; i < starts.size(); ++i) steps.push_back(1);
                // Partial-rank slicing: an axis the node does not name is taken whole. Whisper's decoder slices one axis
                // at a time (the position embedding, the logits' last position), so a full-rank-only rule refuses it.
                std::vector<int64_t> from(a.dims.size(), 0), cnt = a.dims, step(a.dims.size(), 1);
                for (size_t i = 0; i < axes.size(); ++i) {
                    const size_t ax = size_t(axes[i]);
                    if (ax >= a.dims.size()) fail(where() + ": axis out of range");
                    int64_t st = starts[i] < 0 ? starts[i] + a.dims[ax] : starts[i];
                    int64_t en = ends[i] < 0 ? ends[i] + a.dims[ax] : ends[i];
                    st = std::max<int64_t>(0, std::min(st, a.dims[ax]));
                    en = std::max<int64_t>(st, std::min(en, a.dims[ax]));
                    from[ax] = st;
                    step[ax] = steps[i];
                    cnt[ax] = (en - st + steps[i] - 1) / steps[i];
                }
                for (size_t ax = 0; ax < step.size(); ++ax)
                    if (step[ax] != 1) fail(where() + ": only unit steps are implemented");
                r.dims = cnt;
                r.data.resize(nelem(cnt));
                r.i64.resize(a.i64.empty() ? 0 : nelem(cnt));
                std::vector<size_t> stride(a.dims.size(), 1);
                for (size_t i = a.dims.size(); i-- > 1;) stride[i - 1] = stride[i] * size_t(a.dims[i]);
                std::vector<int64_t> idx(cnt.size(), 0);
                for (size_t o = 0; o < nelem(cnt); ++o) {
                    size_t src = 0;
                    for (size_t d = 0; d < idx.size(); ++d) src += size_t(from[d] + idx[d]) * stride[d];
                    if (!a.i64.empty()) r.i64[o] = a.i64[src];
                    else r.data[o] = a.data[src];
                    for (size_t d = idx.size(); d-- > 0;) {
                        if (++idx[d] < cnt[d]) break;
                        idx[d] = 0;
                    }
                }
            } else {
                fail(where() + " is not implemented; the pack needs an op this executor does not have");
            }
            finish_node(n.op, oname, r, op_t0);
        }
        if (op_times) {
            std::vector<std::pair<double, std::string>> order;
            double total = 0.0;
            for (const auto& kv : op_ms) { order.push_back({kv.second.first, kv.first}); total += kv.second.first; }
            std::sort(order.rbegin(), order.rend());
            std::fprintf(stderr, "[audio-op] %zu nodes, %.1f ms, %zu thread(s)\n", g.nodes.size(), total, audio_thread_count());
            for (const auto& o : order)
                std::fprintf(stderr, "[audio-op] %-20s %6ld calls %10.1f ms\n", o.second.c_str(), op_ms[o.second].second, o.first);
        }
        if (node_times) {
            for (const auto& l : node_lines) std::fputs(l.c_str(), stderr);
            std::fprintf(stderr, "[audio-node] %zu node(s)\n", node_lines.size());
        }

        const std::string want = out_name.empty() ? (g.outputs.empty() ? std::string() : g.outputs[0]) : out_name;
        const auto it = t.find(want);
        if (it == t.end()) fail("output '" + want + "' was never produced");
        out->dims = it->second.dims;
        out->data = it->second.data;
        out->i64 = it->second.i64;
        out->b = it->second.b;
        return true;
    } catch (const std::exception& e) {
        if (why != nullptr) *why = e.what();
        return false;
    }
}

}  // namespace

bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames, const std::string& out_name,
                     AudioRunResult* out, std::string* why) {
    return run_audio_graph(g, mel, mel_bins, frames, out_name, out, why, nullptr);
}

bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames, const std::string& out_name,
                     AudioRunResult* out, std::string* why, AudioGemm* dev) {
    std::map<std::string, Tensor> seed;
    Tensor in;
    in.dims = {1, mel_bins, frames};
    in.data.assign(mel, mel + size_t(mel_bins) * size_t(frames));
    if (g.inputs.empty()) {
        if (why != nullptr) *why = "graph declares no input";
        return false;
    }
    seed[g.inputs[0]] = std::move(in);
    return run_graph_impl(g, std::move(seed), out_name, out, why, dev);
}

bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why) {
    if (g.inputs.size() < 1) {
        if (why != nullptr) *why = "graph declares no input";
        return false;
    }
    std::map<std::string, Tensor> seed;
    Tensor toks;
    toks.dims = {1, n_tokens};
    toks.i64.assign(tokens, tokens + size_t(n_tokens));
    seed[g.inputs[0]] = std::move(toks);
    if (g.inputs.size() > 1) {
        if (xa == nullptr || xa_dims.empty()) {
            if (why != nullptr) *why = "the graph has a second input (cross-attention) but none was supplied";
            return false;
        }
        Tensor x;
        x.dims = xa_dims;
        x.data.assign(xa, xa + nelem(xa_dims));
        seed[g.inputs[1]] = std::move(x);
    }
    return run_graph_impl(g, std::move(seed), out_name, out, why);
}

}  // namespace onebit::npu::pack_import
