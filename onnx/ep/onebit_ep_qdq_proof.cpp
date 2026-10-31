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

// Proof for the A8W8 QDQ shape that Quark's ONNX-to-ONNX flow emits:
//
//     x_q --DequantizeLinear--> x_dq \
//                                     MatMul --> y --QuantizeLinear--> y_q
//     w_q --DequantizeLinear--> w_dq /
//
// Everything is built in memory through OrtModelEditorApi (no .onnx, no protobuf), the EP is loaded
// through RegisterExecutionProviderLibrary, and the two sessions run the same OrtModel so a numeric
// difference can only come from the device that executed the MatMul.
//
// Four cases, each with its own claim:
//
//   tensor   per-tensor scales (the shape Quark emits by default). The four nodes are claimed as ONE
//            fused node and the int8 output is identical to the CPU EP's (worst |d| = 0 LSB).
//   axis     per-axis scales: s_x[m] along the activation's row axis, s_w[n] along the weight's
//            output axis. Same gate. This is what SmoothQuant needs for activations and what Quark
//            emits for weights.
//   xaxis-K  a per-axis ACTIVATION scale along K, which cannot fold into an int8 x int8 epilogue
//            (it weights the products inside the sum). The EP must refuse the group: the case gates
//            on "not fused, still correct through the per-node path".
//   rss      a shape whose float intermediates are big enough to see: the CPU EP materialises x_dq,
//            w_dq and y as float32 (4 bytes per element) where the fused path keeps int8 (1 byte).
//            The peak RSS growth of the run is measured around the run itself.
//
// The trace the EP appends is read back per case, which is what says which kernel ran: the fused path
// logs `qdq rows=...` and never the float `matmul batch=...` line. ONEBIT_EP_QDQ=0 turns fusion off
// and the same binary runs the control: one claimed MatMul, the float kernel, the same int8 outputs.
//
// usage: onebit_ep_qdq_proof [path/to/libonebit_ep.so]

#include <onnxruntime_c_api.h>

#include <unistd.h>

#include <malloc.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

const OrtApi* api = nullptr;
const OrtModelEditorApi* editor = nullptr;

// How the graph's scales are shaped. kPerTensor is one value, the rest are 1-D vectors along the axis
// named in the comment.
enum ScaleMode {
    kPerTensor,
    kPerAxisRows,  // activation: one scale per row (DQ axis 0 on a rank-2 x)
    kPerAxisK,     // activation: one scale per K column (DQ axis 1) -- NOT foldable, must be refused
    kPerAxisCols,  // weight: one scale per output column (DQ axis 1 on [K, N])
};

// How the zero points are shaped: absent (symmetric, 0), a non-zero scalar, or one value per row /
// per output column -- the only per-axis forms the epilogue can fold.
enum ZpMode { kZpZero, kZpScalar, kZpAxis };

struct Case {
    const char* label;
    int64_t m, k, n;
    ScaleMode x_mode;
    ScaleMode w_mode;
    ZpMode x_zp;
    ZpMode w_zp;
    int32_t zp_y;  // output zero point; 0 unless the case is about asymmetry
    bool expect_fused;
    bool measure_rss;
    int64_t batch;  // 0 = rank-2 activation, 1 = rank-3 [1, m, k] like a real export
};

int64_t kM = 8;
int64_t kK = 16;
int64_t kN = 6;
// A real quantised model's activations are rank 3 ([batch, sequence, hidden]), not rank 2. 0 keeps the
// proof's flat case.
int64_t kBatch = 0;
ScaleMode kXMode = kPerTensor;
ScaleMode kWMode = kPerTensor;
ZpMode kXZp = kZpZero;
ZpMode kWZp = kZpZero;
int32_t kZpY = 0;

// x_q and w_q are int8 and quantised from a deterministic zero-mean source, so a scale per tensor (or
// per row/column) is enough to exercise the folding without hiding an error behind saturation.
std::vector<int8_t> xq, wq;
std::vector<int8_t> ref_out, ep_out;
std::vector<float> s_x_vec, s_w_vec;  // per-axis scale vectors, sized by the case
std::vector<int8_t> zp_x_vec, zp_w_vec;
float s_x = 0.0f;
float s_w = 0.0f;
float s_y = 0.0f;

int fail(const char* what, const OrtStatus* st) {
    std::fprintf(stderr, "onebit-ort-qdq: FAIL %s: %s\n", what,
                 st ? api->GetErrorMessage(st) : "-");
    if (st) api->ReleaseStatus(const_cast<OrtStatus*>(st));
    return 1;
}

// The activation scale and zero point to use for row m, the weight's for output column n, whichever
// form the case asked for.
float sx_of(int64_t m) { return kXMode == kPerAxisRows ? s_x_vec[(size_t)m] : s_x; }
float sw_of(int64_t n) { return kWMode == kPerAxisCols ? s_w_vec[(size_t)n] : s_w; }
// The zero-point value the graph carries: 0 when symmetric, a constant when the case asks for a
// scalar one, and the generated vector when the case is about a per-axis zero point.
int32_t zx_of(int64_t m) {
    switch (kXZp) {
        case kZpZero: return 0;
        case kZpScalar: return -40;
        case kZpAxis: return zp_x_vec[(size_t)m];
    }
    return 0;
}
int32_t zw_of(int64_t n) {
    switch (kWZp) {
        case kZpZero: return 0;
        case kZpScalar: return -25;
        case kZpAxis: return zp_w_vec[(size_t)n];
    }
    return 0;
}

// A per-axis scale forces the zero point to be a vector of the same length (ONNX rejects a scalar
// beside a per-axis scale), so the shape of the zero point follows the shape of the scale even when
// its value is 0.
bool x_zp_is_vector() { return kXMode != kPerTensor || kXZp == kZpAxis; }
bool w_zp_is_vector() { return kWMode != kPerTensor || kWZp == kZpAxis; }

int8_t q8_sat(int64_t v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

void fill_inputs() {
    xq.assign((size_t)(kM * kK), 0);
    wq.assign((size_t)(kK * kN), 0);
    ref_out.assign((size_t)(kM * kN), 0);
    ep_out.assign((size_t)(kM * kN), 0);
    auto uniform = [](unsigned* s) {
        *s = *s * 1664525u + 1013904223u;
        return ((double)(*s >> 8) / 8388608.0) - 1.0;
    };
    unsigned seed = 12345u;
    std::vector<float> xf((size_t)(kM * kK)), wf((size_t)(kK * kN));
    for (float& v : xf) {
        v = (float)uniform(&seed);
        if (v < 0) v = -v;
    }
    for (float& v : wf) {
        v = (float)uniform(&seed);
        if (v < 0) v = -v;
    }

    // Per-axis candidates: one scale per activation row, per K column, per weight column. amax/127 is
    // the convention the forward route uses; with a non-zero zero point the usable headroom is
    // (127 - zp), so the largest value lands on 127 and nothing saturates.
    std::vector<float> row_amax((size_t)kM, 0.0f), kcol_amax((size_t)kK, 0.0f),
        wcol_amax((size_t)kN, 0.0f);
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t p = 0; p < kK; ++p) {
            const float v = xf[(size_t)(m * kK + p)];
            if (v > row_amax[(size_t)m]) row_amax[(size_t)m] = v;
            if (v > kcol_amax[(size_t)p]) kcol_amax[(size_t)p] = v;
        }
    }
    for (int64_t p = 0; p < kK; ++p) {
        for (int64_t n = 0; n < kN; ++n) {
            const float v = wf[(size_t)(p * kN + n)];
            if (v > wcol_amax[(size_t)n]) wcol_amax[(size_t)n] = v;
        }
    }
    // Negative zero points, so the shifted values stay inside int8: headroom is 127 - zp.
    zp_x_vec.assign(x_zp_is_vector() ? (size_t)(kXMode == kPerAxisK ? kK : kM) : 0, 0);
    for (size_t i = 0; i < zp_x_vec.size(); ++i)
        zp_x_vec[i] = (int8_t)(kXZp == kZpAxis ? -(24 + (int32_t)(i % 8) * 4) : zx_of((int64_t)i));
    zp_w_vec.assign(w_zp_is_vector() ? (size_t)kN : 0, 0);
    for (size_t i = 0; i < zp_w_vec.size(); ++i)
        zp_w_vec[i] = (int8_t)(kWZp == kZpAxis ? -(8 + (int32_t)(i % 4) * 6) : zw_of((int64_t)i));

    auto to_scale = [](float amax, int32_t zp) {
        const int32_t headroom = 127 - zp;
        return amax > 0.0f ? amax / (float)(headroom > 0 ? headroom : 1) : 1.0f;
    };

    float xmax = 0.0f, wmax = 0.0f;
    for (int64_t m = 0; m < kM; ++m) xmax = row_amax[(size_t)m] > xmax ? row_amax[(size_t)m] : xmax;
    for (int64_t n = 0; n < kN; ++n) wmax = wcol_amax[(size_t)n] > wmax ? wcol_amax[(size_t)n] : wmax;
    s_x = to_scale(xmax, zx_of(0));
    s_w = to_scale(wmax, zw_of(0));
    s_x_vec.resize(kXMode == kPerAxisK ? (size_t)kK : (size_t)kM);
    for (size_t i = 0; i < s_x_vec.size(); ++i)
        s_x_vec[i] = to_scale(kXMode == kPerAxisK ? kcol_amax[i] : row_amax[i], zx_of((int64_t)i));
    s_w_vec.resize((size_t)kN);
    for (size_t n = 0; n < s_w_vec.size(); ++n) s_w_vec[n] = to_scale(wcol_amax[n], zw_of((int64_t)n));

    for (int64_t m = 0; m < kM; ++m) {
        const bool per_k = kXMode == kPerAxisK;
        for (int64_t p = 0; p < kK; ++p) {
            const float s = per_k ? s_x_vec[(size_t)p] : sx_of(m);
            // ONNX DequantizeLinear: x_dq = (x_q - zp) * s, so x_q = round(x/s) + zp.
            xq[(size_t)(m * kK + p)] =
                q8_sat(std::lround(xf[(size_t)(m * kK + p)] / s) + (per_k ? zp_x_vec[(size_t)p] : zx_of(m)));
        }
    }
    for (int64_t p = 0; p < kK; ++p)
        for (int64_t n = 0; n < kN; ++n)
            wq[(size_t)(p * kN + n)] =
                q8_sat(std::lround(wf[(size_t)(p * kN + n)] / sw_of(n)) + zw_of(n));

    // The output scale comes from the exact shifted product, so y_q uses a fraction of the int8 range
    // instead of clipping -- and leaves room for the output zero point.
    float ymax = 0.0f;
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t n = 0; n < kN; ++n) {
            int64_t acc = 0;
            for (int64_t p = 0; p < kK; ++p) {
                const int64_t zx = kXMode == kPerAxisK ? zp_x_vec[(size_t)p] : zx_of(m);
                acc += ((int64_t)xq[(size_t)(m * kK + p)] - zx) *
                       ((int64_t)wq[(size_t)(p * kN + n)] - zw_of(n));
            }
            const float y = std::fabs((float)acc * sx_of(m) * sw_of(n));
            if (y > ymax) ymax = y;
        }
    }
    s_y = ymax / 48.0f;
}

OrtStatus* make_type(const int64_t* dims, size_t rank, ONNXTensorElementDataType elem,
                     OrtTypeInfo** out) {
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = api->CreateTensorTypeAndShapeInfo(&tsi)) return st;
    OrtStatus* st = api->SetTensorElementType(tsi, elem);
    if (!st) st = api->SetDimensions(tsi, dims, rank);
    if (!st) st = editor->CreateTensorTypeInfo(tsi, out);
    api->ReleaseTensorTypeAndShapeInfo(tsi);
    return st;
}

// An initializer whose memory ORT owns, filled from a copy of `data`.
//
// Not CreateTensorWithDataAsOrtValue: that borrows the caller's memory, which then has to outlive the
// session. The scales and the zero point are ordinary locals and globals here, and borrowing them
// produced a garbage zero point in a no-optimization build as soon as the stack slot was reused --
// every value of the graph then differed from the CPU EP's by tens of LSB. ORT's own small-initializer
// path costs a memcpy and removes the whole class.
OrtStatus* add_initializer(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                           const void* data, size_t bytes, const int64_t* dims, size_t rank) {
    OrtAllocator* alloc = nullptr;
    if (OrtStatus* st = api->GetAllocatorWithDefaultOptions(&alloc)) return st;
    OrtValue* v = nullptr;
    OrtStatus* st = api->CreateTensorAsOrtValue(alloc, dims, rank, elem, &v);
    if (!st) {
        void* p = nullptr;
        st = api->GetTensorMutableData(v, &p);
        if (!st) std::memcpy(p, data, bytes);
    }
    if (!st) return editor->AddInitializerToGraph(graph, name, v, false);  // takes ownership
    if (v) api->ReleaseValue(v);
    return st;
}

// A rank-0 (scalar) initializer.
OrtStatus* add_scalar(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      const void* data, size_t bytes) {
    return add_initializer(graph, name, elem, data, bytes, nullptr, 0);
}

// A rank-`rank` initializer with explicit dims.
OrtStatus* add_tensor(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      const void* data, size_t bytes, const int64_t* dims, size_t rank) {
    return add_initializer(graph, name, elem, data, bytes, dims, rank);
}

OrtStatus* add_node_attr(OrtGraph* graph, const char* op, const char* name, const char** in,
                         size_t n_in, const char** out, size_t n_out, OrtOpAttr* attr) {
    OrtNode* node = nullptr;
    OrtOpAttr* attrs[1] = {attr};
    if (OrtStatus* st = editor->CreateNode(op, "", name, in, n_in, out, n_out, attr ? attrs : nullptr,
                                           attr ? 1 : 0, &node))
        return st;
    return editor->AddNodeToGraph(graph, node);  // takes ownership
}

OrtStatus* add_node(OrtGraph* graph, const char* op, const char* name, const char** in, size_t n_in,
                    const char** out, size_t n_out) {
    return add_node_attr(graph, op, name, in, n_in, out, n_out, nullptr);
}

// The `axis` attribute a per-axis DequantizeLinear needs.
OrtStatus* make_axis_attr(int64_t axis, OrtOpAttr** out) {
    return api->CreateOpAttr("axis", &axis, 1, ORT_OP_ATTR_INT, out);
}

OrtStatus* build_model(OrtModel** out) {
    const int64_t x_dims[3] = {kBatch ? kBatch : kM, kBatch ? kM : kK, kBatch ? kK : 0};
    const int64_t w_dims[2] = {kK, kN};
    const int64_t y_dims[3] = {kBatch ? kBatch : kM, kBatch ? kM : kN, kBatch ? kN : 0};
    const size_t x_rank = kBatch ? 3 : 2;

    OrtTypeInfo* x_type = nullptr;
    OrtTypeInfo* y_type = nullptr;
    if (OrtStatus* st = make_type(x_dims, x_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &x_type)) return st;
    if (OrtStatus* st = make_type(y_dims, x_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &y_type)) {
        api->ReleaseTypeInfo(x_type);
        return st;
    }
    OrtValueInfo* x_vi = nullptr;
    OrtValueInfo* y_vi = nullptr;
    OrtStatus* st = editor->CreateValueInfo("x_q", x_type, &x_vi);
    api->ReleaseTypeInfo(x_type);
    if (!st) st = editor->CreateValueInfo("y_q", y_type, &y_vi);
    api->ReleaseTypeInfo(y_type);
    if (st) return st;

    OrtGraph* graph = nullptr;
    if (!st) st = editor->CreateGraph(&graph);
    if (!st) st = editor->SetGraphInputs(graph, &x_vi, 1);    // graph takes ownership
    if (!st) st = editor->SetGraphOutputs(graph, &y_vi, 1);   // graph takes ownership

    const bool x_per_axis = kXMode != kPerTensor;
    const bool w_per_axis = kWMode != kPerTensor;
    const bool x_zp_vec = x_zp_is_vector();
    const bool w_zp_vec = w_zp_is_vector();
    const int32_t zx_scalar = zx_of(0);
    const int32_t zw_scalar = zw_of(0);

    // Scales and zero points: a scalar (rank 0) or one value per row/column, plus the output pair.
    if (!st) {
        if (!x_per_axis) {
            st = add_scalar(graph, "s_x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_x, sizeof(s_x));
        } else {
            const int64_t dims[1] = {(int64_t)s_x_vec.size()};
            st = add_tensor(graph, "s_x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, s_x_vec.data(),
                            s_x_vec.size() * sizeof(float), dims, 1);
        }
    }
    if (!st) {
        if (!w_per_axis) {
            st = add_scalar(graph, "s_w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_w, sizeof(s_w));
        } else {
            const int64_t dims[1] = {(int64_t)s_w_vec.size()};
            st = add_tensor(graph, "s_w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, s_w_vec.data(),
                            s_w_vec.size() * sizeof(float), dims, 1);
        }
    }
    if (!st) {
        if (!x_zp_vec) {
            const int8_t z = (int8_t)zx_scalar;
            st = add_scalar(graph, "zp_x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &z, sizeof(z));
        } else {
            const int64_t dims[1] = {(int64_t)zp_x_vec.size()};
            st = add_tensor(graph, "zp_x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, zp_x_vec.data(),
                            zp_x_vec.size(), dims, 1);
        }
    }
    if (!st) {
        if (!w_zp_vec) {
            const int8_t z = (int8_t)zw_scalar;
            st = add_scalar(graph, "zp_w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &z, sizeof(z));
        } else {
            const int64_t dims[1] = {(int64_t)zp_w_vec.size()};
            st = add_tensor(graph, "zp_w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, zp_w_vec.data(),
                            zp_w_vec.size(), dims, 1);
        }
    }
    if (!st) st = add_scalar(graph, "s_y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_y, sizeof(s_y));
    if (!st) st = add_scalar(graph, "zp_y", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &kZpY, sizeof(int8_t));
    if (!st) st = add_tensor(graph, "w_q", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, wq.data(), wq.size(),
                             w_dims, 2);
    if (st) {
        api->ReleaseGraph(graph);
        return st;
    }

    const char* dq_x_in[3] = {"x_q", "s_x", "zp_x"};
    const char* dq_w_in[3] = {"w_q", "s_w", "zp_w"};
    const char* mm_in[2] = {"x_dq", "w_dq"};
    const char* q_in[3] = {"y", "s_y", "zp_y"};
    const char* x_dq_out[1] = {"x_dq"};
    const char* w_dq_out[1] = {"w_dq"};
    const char* y_out[1] = {"y"};
    const char* yq_out[1] = {"y_q"};

    // A per-axis scale or zero point says which dimension it runs along; the default is 1. Both
    // operands of one DequantizeLinear share the node's axis, which is what ONNX requires.
    OrtOpAttr* x_axis = nullptr;
    OrtOpAttr* w_axis = nullptr;
    if (!st && x_zp_vec) st = make_axis_attr(kXMode == kPerAxisRows ? 0 : 1, &x_axis);
    if (!st && w_zp_vec) st = make_axis_attr(1, &w_axis);
    // Both DequantizeLinear nodes carry a zero point input in every case: ONNX requires a per-axis
    // zero point to match the scale's shape (both vectors, as generated here), and for a per-tensor
    // scale it is a scalar, which is what a symmetric graph writes as an explicit 0.
    if (!st)
        st = add_node_attr(graph, "DequantizeLinear", "dq_x", dq_x_in, 3, x_dq_out, 1, x_axis);
    if (!st)
        st = add_node_attr(graph, "DequantizeLinear", "dq_w", dq_w_in, 3, w_dq_out, 1, w_axis);
    if (!st) st = add_node(graph, "MatMul", "qdq_matmul", mm_in, 2, y_out, 1);
    if (!st) st = add_node(graph, "QuantizeLinear", "q_y", q_in, 3, yq_out, 1);
    // The OrtModelEditorApi takes the attribute into the node; releasing it here as well is a double
    // free (found the hard way: SIGSEGV in onnx::AttributeProto's destructor when the model is
    // released). One graph per process phase, so nothing leaks that matters.
    if (st) {
        api->ReleaseGraph(graph);
        return st;
    }

    const char* domains[1] = {""};
    const int opsets[1] = {21};
    OrtModel* model = nullptr;
    if (!st) st = editor->CreateModel(domains, opsets, 1, &model);
    if (!st) st = editor->AddGraphToModel(model, graph);  // takes ownership
    if (st) {
        api->ReleaseGraph(graph);
        return st;
    }
    *out = model;
    return nullptr;
}

OrtStatus* make_session(OrtEnv* env, OrtModel* model, bool with_ep, const char* ep_lib,
                        OrtSession** out) {
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = api->CreateSessionOptions(&so)) return st;
    // BASIC, not ALL: the QDQ-to-QLinearMatMul rewrite would replace the pattern under test with an
    // op our EP does not claim, and then there is nothing to observe.
    OrtStatus* st = api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC);
    // One thread: the RSS case measures the tensors, and MLAS's per-thread scratch buffers would be
    // tens of megabytes of noise next to a 6 MB difference.
    if (!st) st = api->SetIntraOpNumThreads(so, 1);
    if (!st && with_ep && ep_lib) st = api->RegisterExecutionProviderLibrary(env, "onebit_ep", ep_lib);
    const OrtEpDevice* const* devices = nullptr;
    size_t n = 0;
    if (!st) st = api->GetEpDevices(env, &devices, &n);
    std::vector<const OrtEpDevice*> ours;
    const OrtEpDevice* cpu = nullptr;
    for (size_t i = 0; !st && i < n; ++i) {
        const char* name = api->EpDevice_EpName(devices[i]);
        const char* vendor = api->EpDevice_EpVendor(devices[i]);
        std::printf("onebit-ort-qdq: device[%zu] ep=%s vendor=%s\n", i, name ? name : "?",
                    vendor ? vendor : "?");
        if (name && std::strcmp(name, "OneBitEp") == 0) {
            if (with_ep) ours.push_back(devices[i]);
        } else if (name && std::strcmp(name, "CPUExecutionProvider") == 0 && !cpu) {
            cpu = devices[i];
        }
    }
    if (!st && with_ep && ours.empty()) {
        st = api->CreateStatus(ORT_FAIL, "OneBitEp not among the registered EP devices");
    }
    if (!st && with_ep) {
        st = api->SessionOptionsAppendExecutionProvider_V2(so, env, ours.data(), ours.size(), nullptr,
                                                           nullptr, 0);
    }
    if (!st && cpu) {
        const OrtEpDevice* one[1] = {cpu};
        st = api->SessionOptionsAppendExecutionProvider_V2(so, env, one, 1, nullptr, nullptr, 0);
    }
    if (!st) st = editor->CreateSessionFromModel(env, model, so, out);
    api->ReleaseSessionOptions(so);
    return st;
}

int run_once(OrtSession* session, std::vector<int8_t>* out) {
    const int64_t x_dims[3] = {kBatch ? kBatch : kM, kBatch ? kM : kK, kBatch ? kK : 0};
    const size_t x_rank = kBatch ? 3 : 2;
    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi))
        return fail("cpu memory info", st);
    OrtValue* x = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(mi, xq.data(), xq.size(), x_dims, x_rank,
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &x);
    api->ReleaseMemoryInfo(mi);
    if (st) return fail("input tensor", st);

    const char* in_names[1] = {"x_q"};
    const char* out_names[1] = {"y_q"};
    OrtValue* y = nullptr;
    st = api->Run(session, nullptr, in_names, (const OrtValue* const*)&x, 1, out_names, 1, &y);
    api->ReleaseValue(x);
    if (st) return fail("Run", st);

    void* p = nullptr;
    st = api->GetTensorMutableData(y, &p);
    if (!st) std::memcpy(out->data(), p, out->size());
    api->ReleaseValue(y);
    if (st) return fail("read output", st);
    return 0;
}

std::string slurp(const char* path) {
    std::string s;
    if (FILE* f = std::fopen(path, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
        std::fclose(f);
    }
    return s;
}

// How many times `needle` appears in `hay`.
size_t count_substr(const std::string& hay, const char* needle) {
    size_t n = 0;
    const size_t len = std::strlen(needle);
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + len)) ++n;
    return n;
}

// One Vm* figure out of /proc/self/status, in kB. -1 when it is not there.
long proc_kb(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    const size_t klen = std::strlen(key);
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, klen) == 0) {
            kb = std::strtol(line + klen, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return kb;
}

// Writing 5 to clear_refs resets VmHWM, so the next run's peak is the run's own.
bool reset_peak() {
    FILE* f = std::fopen("/proc/self/clear_refs", "w");
    if (!f) return false;
    const bool ok = std::fputc('5', f) != EOF;
    std::fclose(f);
    return ok;
}

// Peak RSS grown by one run: the resident set just before it, then the peak over it. malloc_trim
// first, so the freed tensors of the warm-up run go back to the kernel instead of being handed to
// the run being measured -- otherwise whichever session ran second would look free.
long run_peak_kb(OrtSession* session, std::vector<int8_t>* out) {
    (void)run_once(session, out);  // warm: one-time costs are not the tensors under test
    malloc_trim(0);
    if (!reset_peak()) return -1;
    const long before = proc_kb("VmRSS:");
    if (run_once(session, out)) return -1;
    const long peak = proc_kb("VmHWM:");
    if (before < 0 || peak < 0) return -1;
    return peak - before;
}

// Worst absolute int8 difference between the reference and the EP's output, counting values that
// differ at all.
int compare(size_t* differing) {
    int worst = 0;
    *differing = 0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        const int d = std::abs((int)ref_out[i] - (int)ep_out[i]);
        if (d) ++(*differing);
        if (d > worst) worst = d;
    }
    return worst;
}

const char* zp_mode_name(ZpMode z) {
    switch (z) {
        case kZpZero: return "0";
        case kZpScalar: return "scalar";
        case kZpAxis: return "per-axis";
    }
    return "?";
}

const char* mode_name(ScaleMode m) {
    switch (m) {
        case kPerTensor: return "per-tensor";
        case kPerAxisRows: return "per-row";
        case kPerAxisK: return "per-K";
        case kPerAxisCols: return "per-column";
    }
    return "?";
}

// What a case found, for the verdict at the end.
struct Result {
    int worst = 0;
    size_t differing = 0;
    bool fused = false;      // the fused kernel ran (trace: a qdq line)
    bool float_mm = false;   // the float MatMul kernel ran (trace: a matmul line)
    long peak_cpu = -1;
    long peak_fused = -1;
    int rc = 0;
};

// One phase: build the graph for the case, run the CPU EP control session and the EP session on it,
// compare their int8 outputs, and read back the trace lines the EP wrote during the phase. Each phase
// gets its own OrtEnv because a plugin library is registered per env, once.
Result run_case(const Case& c, const char* ep_lib, const char* trace_path) {
    Result r;
    kM = c.m;
    kK = c.k;
    kN = c.n;
    kXMode = c.x_mode;
    kWMode = c.w_mode;
    kXZp = c.x_zp;
    kWZp = c.w_zp;
    kZpY = c.zp_y;
    kBatch = c.batch;
    fill_inputs();

    std::printf("onebit-ort-qdq: case %s %s m=%lld k=%lld n=%lld scales x=%s w=%s zp x=%s w=%s y=%d\n",
                c.label, kBatch ? "[1,seq,k] activation" : "[m,k] activation", (long long)kM,
                (long long)kK, (long long)kN, mode_name(kXMode), mode_name(kWMode),
                zp_mode_name(kXZp), zp_mode_name(kWZp), (int)kZpY);

    const size_t trace_mark = slurp(trace_path).size();
    OrtEnv* env = nullptr;
    OrtModel* model = nullptr;
    OrtSession* cpu_session = nullptr;
    OrtSession* ep_session = nullptr;
    if (OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "onebit-qdq", &env)) {
        r.rc = fail("create env", st);
    } else if (OrtStatus* st = build_model(&model)) {
        r.rc = fail("build model", st);
    } else if (OrtStatus* st = make_session(env, model, false, ep_lib, &cpu_session)) {
        r.rc = fail("cpu session", st);
    } else if (OrtStatus* st = make_session(env, model, true, ep_lib, &ep_session)) {
        r.rc = fail("ep session", st);
    } else if (c.measure_rss) {
        r.peak_cpu = run_peak_kb(cpu_session, &ref_out);
        r.peak_fused = run_peak_kb(ep_session, &ep_out);
        r.worst = compare(&r.differing);
    } else if (run_once(cpu_session, &ref_out) || run_once(ep_session, &ep_out)) {
        r.rc = 1;
    } else {
        r.worst = compare(&r.differing);
    }
    if (ep_session) api->ReleaseSession(ep_session);
    if (cpu_session) api->ReleaseSession(cpu_session);
    if (model) api->ReleaseModel(model);
    if (env) api->ReleaseEnv(env);
    if (r.rc) return r;

    // The trace slice this case wrote -- which of the two kernels ran for it, and only for it.
    const std::string trace = slurp(trace_path);
    const std::string slice = trace_mark < trace.size() ? trace.substr(trace_mark) : std::string();
    r.fused = slice.find("qdq rows=") != std::string::npos;
    r.float_mm = slice.find("matmul batch=") != std::string::npos;
    std::printf("onebit-ort-qdq: case %s: %zu of %zu int8 values differ, worst |d|=%d LSB, "
                "path=%s\n",
                c.label, r.differing, ref_out.size(), r.worst, r.fused ? "fused" : "per-node");
    if (std::getenv("ONEBIT_EP_VERBOSE")) {
        for (size_t pos = 0; pos < slice.size();) {
            size_t nl = slice.find('\n', pos);
            if (nl == std::string::npos) nl = slice.size();
            std::printf("  %s\n", slice.substr(pos, nl - pos).c_str());
            pos = nl + 1;
        }
    }
    if (c.measure_rss) {
        if (r.peak_cpu < 0 || r.peak_fused < 0) {
            std::printf("onebit-ort-qdq: case %s: peak RSS not measurable here (no VmHWM reset via "
                        "/proc/self/clear_refs), memory gate skipped\n",
                        c.label);
        } else {
            std::printf("onebit-ort-qdq: case %s: peak RSS growth: CPU EP +%ld kB, fused +%ld kB "
                        "(delta %ld kB)\n",
                        c.label, r.peak_cpu, r.peak_fused, r.peak_cpu - r.peak_fused);
        }
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [path/to/libonebit_ep.so]\n", argv[0]);
        return 2;
    }
    const char* ep_lib = argc == 2 ? argv[1] : nullptr;
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        std::fprintf(stderr, "onebit-ort-qdq: FAIL no OrtApi\n");
        return 1;
    }
    editor = api->GetModelEditorApi();
    if (!editor) {
        std::fprintf(stderr, "onebit-ort-qdq: FAIL this ONNX Runtime has no ModelEditorApi\n");
        return 1;
    }
    // The EP writes its trace to /tmp/onebit_ep_trace.<pid>.log; this process is the one that
    // dlopen()s the library, so the pid is the same.
    const std::string trace_path =
        std::string("/tmp/onebit_ep_trace.") + std::to_string(::getpid()) + ".log";
    std::remove(trace_path.c_str());
    setenv("ONEBIT_EP_TRACE", "1", 1);
    const char* qdq_env = std::getenv("ONEBIT_EP_QDQ");
    const bool fusion_off = qdq_env && *qdq_env == '0';
    if (fusion_off) std::printf("onebit-ort-qdq: control run: ONEBIT_EP_QDQ=0, fusion disabled\n");

    const Case cases[] = {
        {"tensor", 8, 16, 6, kPerTensor, kPerTensor, kZpZero, kZpZero, 0, true, false},
        {"axis", 8, 16, 6, kPerAxisRows, kPerAxisCols, kZpZero, kZpZero, 0, true, false},
        // Asymmetric: a non-zero zero point on both operands and on the output.
        {"asym", 8, 16, 6, kPerTensor, kPerTensor, kZpScalar, kZpScalar, -7, true, false},
        {"asym-axis", 8, 16, 6, kPerAxisRows, kPerAxisCols, kZpAxis, kZpAxis, -7, true, false},
        // What a Quark A8W8 export actually looks like: rank-3 activation, one scale and one zero point
        // per output column of the weight, everything else per-tensor.
        {"quark", 8, 64, 48, kPerTensor, kPerAxisCols, kZpZero, kZpZero, 0, true, false, 1},
        {"quark-asym", 8, 64, 48, kPerTensor, kPerAxisCols, kZpScalar, kZpAxis, -5, true, false, 1},
        // The same shape one of Qwen3-0.6B's own projections has: 5 tokens x [1024 -> 2048].
        {"quark-wide", 5, 1024, 2048, kPerTensor, kPerAxisCols, kZpZero, kZpZero, 0, true, false, 1},
        // The same, with enough values that a quantisation boundary is likely to be hit: 64x1024x2048
        // is 131072 int8 outputs, and the CPU EP reaches each one through a float accumulation where
        // the fused path rounds an exact int32 sum.
        {"quark-stress", 64, 1024, 2048, kPerTensor, kPerAxisCols, kZpZero, kZpZero, 0, true, false, 1},
        // A scale along K is not foldable: the group must stay on the per-node path.
        {"xaxis-K", 8, 16, 6, kPerAxisK, kPerTensor, kZpZero, kZpZero, 0, false, false},
        // x_dq is 1 MB of float32 and w_dq 4 MB, against 0.25 MB and 1 MB of int8.
        {"rss", 256, 1024, 1024, kPerTensor, kPerTensor, kZpZero, kZpZero, 0, true, true},
    };

    int rc = 0;
    int worst_tensor = 0, worst_axis = 0, worst_asym = 0, worst_asym_axis = 0, worst_refused = 0;
    int worst_quark = 0, worst_quark_asym = 0, worst_quark_wide = 0;
    int worst_quark_stress = 0;
    size_t differing_quark_stress = 0;
    long peak_cpu = -1, peak_fused = -1;
    size_t fused_cases = 0;
    Result refused;
    for (const Case& c : cases) {
        Result r = run_case(c, ep_lib, trace_path.c_str());
        if (r.rc) {
            rc = r.rc;
            break;
        }
        if (r.fused) ++fused_cases;
        // The fused path and the per-node path must never both have run for one case.
        if (r.fused && r.float_mm) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL case %s ran both the fused and the float "
                                 "MatMul kernel\n",
                         c.label);
            rc = 1;
            break;
        }
        const bool want_fused = c.expect_fused && !fusion_off;
        if (r.fused != want_fused) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL case %s: %s, expected %s\n", c.label,
                         r.fused ? "the group was fused" : "the group was NOT fused",
                         want_fused ? "the fused path" : "the per-node path");
            rc = 1;
            break;
        }
        if (std::strcmp(c.label, "tensor") == 0) {
            worst_tensor = r.worst;
        } else if (std::strcmp(c.label, "axis") == 0) {
            worst_axis = r.worst;
        } else if (std::strcmp(c.label, "asym") == 0) {
            worst_asym = r.worst;
        } else if (std::strcmp(c.label, "asym-axis") == 0) {
            worst_asym_axis = r.worst;
        } else if (std::strcmp(c.label, "quark") == 0) {
            worst_quark = r.worst;
        } else if (std::strcmp(c.label, "quark-asym") == 0) {
            worst_quark_asym = r.worst;
        } else if (std::strcmp(c.label, "quark-stress") == 0) {
            worst_quark_stress = r.worst;
            differing_quark_stress = r.differing;
        } else if (std::strcmp(c.label, "quark-wide") == 0) {
            worst_quark_wide = r.worst;
        } else if (std::strcmp(c.label, "xaxis-K") == 0) {
            worst_refused = r.worst;
            refused = r;
        } else {
            peak_cpu = r.peak_cpu;
            peak_fused = r.peak_fused;
        }
    }

    if (!rc) {
        // Per-tensor and per-axis must be bit-identical to the CPU EP; the refused case is gated one
        // LSB looser because it runs the float MatMul (MLAS accumulates in its own order).
        if (worst_tensor != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL per-tensor case differs by %d LSB\n",
                         worst_tensor);
            rc = 1;
        } else if (!fusion_off && worst_axis != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL per-axis case differs by %d LSB\n", worst_axis);
            rc = 1;
        } else if (!fusion_off && worst_asym != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL asymmetric case differs by %d LSB\n", worst_asym);
            rc = 1;
        } else if (!fusion_off && worst_asym_axis != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL asymmetric per-axis case differs by %d LSB\n",
                         worst_asym_axis);
            rc = 1;
        } else if (!fusion_off && worst_quark != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL Quark-shaped (rank-3, per-column weight "
                                 "scales) case differs by %d LSB\n",
                         worst_quark);
            rc = 1;
        } else if (!fusion_off && worst_quark_stress > 1) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL stress case differs by %d LSB (%zu values)\n",
                         worst_quark_stress, differing_quark_stress);
            rc = 1;
        } else if (!fusion_off && worst_quark_wide != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL Quark-wide (5x1024x2048, per-column weight "
                                 "scales) case differs by %d LSB\n",
                         worst_quark_wide);
            rc = 1;
        } else if (!fusion_off && worst_quark_asym != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL Quark-shaped asymmetric case differs by %d LSB\n",
                         worst_quark_asym);
            rc = 1;
        } else if (worst_refused > 1) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL the refused per-K-scale case differs by %d LSB\n",
                         worst_refused);
            rc = 1;
        } else if (!fusion_off) {
            if (fused_cases < 6) {
                std::fprintf(stderr, "onebit-ort-qdq: FAIL only %zu case(s) were fused\n", fused_cases);
                rc = 1;
            } else if (peak_cpu > 0 && peak_fused > 0 && peak_fused >= peak_cpu) {
                std::fprintf(stderr,
                             "onebit-ort-qdq: FAIL the fused run grew RSS by %ld kB, not less than "
                             "the CPU EP's %ld kB\n",
                             peak_fused, peak_cpu);
                rc = 1;
            } else {
                std::printf("onebit-ort-qdq: PASS OneBitEp fused the A8W8 QDQ group: worst |d| = "
                            "%d/%d/%d/%d LSB (per-tensor, per-axis, asymmetric, asymmetric per-axis), "
                            "per-K scale refused (worst |d|=%d), peak RSS -%ld kB\n",
                            worst_tensor, worst_axis, worst_asym, worst_asym_axis, worst_refused,
                            peak_cpu > 0 ? peak_cpu - peak_fused : 0L);
            }
        } else {
            // The control only has to prove the per-node path still works; the other numbers are
            // measurements, not gates.
            std::printf("onebit-ort-qdq: PASS OneBitEp ran the A8W8 QDQ MatMul unfused "
                        "(control: ONEBIT_EP_QDQ=0), per-tensor worst |d|=%d LSB, others %d/%d/%d/%d "
                        "(informational)\n",
                        worst_tensor, worst_axis, worst_asym, worst_asym_axis, worst_refused);
        }
    }
    (void)refused;
    return rc;
}
