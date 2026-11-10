// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// audio_exec.cpp -- the host f32 executor for an audio pack's graph (see audio_exec.h for the contract).

#include "audio_exec.h"

#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

#include "forward/include/audio_ops.h"
#include "file_map.h"

namespace onebit::npu::pack_import {
namespace {

struct Tensor {
    std::vector<int64_t> dims;
    std::vector<float> data;
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

void broadcast_binary(const Tensor& a, const Tensor& b, const std::string& op, Tensor* out) {
    const Broadcast p = plan_broadcast(a.dims, b.dims);
    out->dims = p.dims;
    out->data.resize(nelem(p.dims));
    std::vector<size_t> idx(p.dims.size(), 0);
    for (size_t o = 0; o < out->data.size(); ++o) {
        size_t ia = 0, ib = 0;
        for (size_t d = 0; d < idx.size(); ++d) {
            ia += idx[d] * p.stride_big[d];
            ib += idx[d] * p.stride_small[d];
        }
        const float x = a.data[ia], y = b.data[ib];
        out->data[o] = (op == "Add") ? x + y : (op == "Sub") ? x - y : (op == "Mul") ? x * y : x / y;
        for (size_t d = idx.size(); d-- > 0;) {
            if (++idx[d] < size_t(p.dims[d])) break;
            idx[d] = 0;
        }
    }
}

std::vector<int64_t> dims_of(const OnnxInit& t) { return t.dims; }

// Copies an initializer to f32, following external_data into the pack's side file when the bytes are not inline.
void load_init_f32(const OnnxGraphInfo& g, const OnnxInit& t, std::vector<float>* dst) {
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
    dst->assign(n, 0.0f);
    if (t.dtype == 1) {   // FLOAT
        if (len < n * 4) fail("initializer '" + t.name + "' is short: " + std::to_string(len) + " < " + std::to_string(n * 4));
        std::memcpy(dst->data(), src, n * 4);
    } else if (t.dtype == 7) {   // INT64 -- shape operands
        if (len < n * 8) fail("int64 initializer '" + t.name + "' is short");
        std::memcpy(dst->data(), src, n * 8);   // bit pattern; callers that need the integer read it via load_init_i64
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
            std::memcpy(&(*dst)[i], &bits, 4);
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

bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames, const std::string& out_name,
                     AudioRunResult* out, std::string* why) {
    try {
        std::map<std::string, Tensor> t;
        // The graph input: the mel window, in the layout the first Conv consumes.
        Tensor in;
        in.dims = {1, mel_bins, frames};
        in.data.assign(mel, mel + size_t(mel_bins) * size_t(frames));
        if (g.inputs.empty()) fail("graph declares no input");
        t[g.inputs[0]] = std::move(in);

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
            load_init_f32(g, *ii->second, &v.data);
            return t.emplace(name, std::move(v)).first->second;
        };

        int node_index = 0;
        for (const auto& n : g.nodes) {
            ++node_index;
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
                r.data.assign(nelem(r.dims), 0.0f);
                std::vector<size_t> idx(bp.dims.size(), 0);
                const size_t batches = nelem(bp.dims);
                for (size_t bi = 0; bi < batches; ++bi) {
                    size_t oa = 0, ob = 0;
                    for (size_t d = 0; d < idx.size(); ++d) {
                        oa += idx[d] * bp.stride_big[d];
                        ob += idx[d] * bp.stride_small[d];
                    }
                    const float* ap = a.data.data() + oa * size_t(M * K);
                    const float* bpp = b.data.data() + ob * size_t(K * N);
                    float* rp = r.data.data() + bi * size_t(M * N);
                    for (int64_t i = 0; i < M; ++i) {
                        float* rr = rp + size_t(i) * size_t(N);
                        for (int64_t k = 0; k < K; ++k) {
                            const float av = ap[size_t(i) * size_t(K) + size_t(k)];
                            if (av == 0.0f) continue;
                            const float* br = bpp + size_t(k) * size_t(N);
                            for (int64_t j = 0; j < N; ++j) rr[j] += av * br[j];
                        }
                    }
                    for (size_t d = idx.size(); d-- > 0;) {
                        if (++idx[d] < size_t(bp.dims[d])) break;
                        idx[d] = 0;
                    }
                }
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
                r.data.resize(a.data.size());
                for (size_t i = 0; i < a.data.size(); ++i) r.data[i] = std::erf(a.data[i]);
            } else if (n.op == "Gelu") {
                const Tensor& a = need(n.ins[0]);
                r.dims = a.dims;
                r.data.resize(a.data.size());
                const auto ap = n.attrs.find("approximate");
                (void)ap;   // ONNX spells it as a string; the export here is 'none', which is what audio_ops::gelu_erf is
                for (size_t i = 0; i < a.data.size(); ++i) r.data[i] = audio::gelu_erf(a.data[i]);
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
                for (size_t row = 0; row < a.data.size() / size_t(nrm); ++row) {
                    audio::layer_norm(a.data.data() + row * size_t(nrm), scale.data.data(),
                                      bias != nullptr ? bias->data.data() : nullptr, int(nrm), eps,
                                      r.data.data() + row * size_t(nrm));
                }
            } else if (n.op == "Softmax") {
                const Tensor& a = need(n.ins[0]);
                const int64_t axis = n.attrs.count("axis") ? n.attrs.at("axis") : -1;
                if (axis != -1 && axis != int64_t(a.dims.size()) - 1) fail(where() + ": only the last axis is implemented");
                r = a;
                const int64_t nrm = a.dims.back();
                for (size_t row = 0; row < r.data.size() / size_t(nrm); ++row)
                    audio::softmax_row(r.data.data() + row * size_t(nrm), int(nrm));
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
                std::vector<size_t> idx(perm.size(), 0);
                for (size_t o = 0; o < r.data.size(); ++o) {
                    size_t src = 0;
                    for (size_t i = 0; i < idx.size(); ++i) src += idx[i] * in_stride[size_t(perm[i])];
                    r.data[o] = a.data[src];
                    for (size_t d = idx.size(); d-- > 0;) {
                        if (++idx[d] < size_t(r.dims[d])) break;
                        idx[d] = 0;
                    }
                }
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
                for (int64_t n0 = 0; n0 < a.dims[0]; ++n0)
                    audio::conv1d(a.data.data() + size_t(n0) * size_t(cin) * size_t(T), int(cin), int(T), w.data.data(),
                                  bias != nullptr ? bias->data.data() : nullptr, int(cout), int(k), stride, pad,
                                  r.data.data() + size_t(n0) * size_t(cout) * size_t(tout));
            } else {
                fail(where() + " is not implemented; the pack needs an op this executor does not have");
            }
            t[oname] = std::move(r);
        }

        const std::string want = out_name.empty() ? (g.outputs.empty() ? std::string() : g.outputs[0]) : out_name;
        const auto it = t.find(want);
        if (it == t.end()) fail("output '" + want + "' was never produced");
        out->dims = it->second.dims;
        out->data = it->second.data;
        return true;
    } catch (const std::exception& e) {
        if (why != nullptr) *why = e.what();
        return false;
    }
}

}  // namespace onebit::npu::pack_import
