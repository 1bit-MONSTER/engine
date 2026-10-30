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
// difference can only come from the device that executed the MatMul. The graph is symmetric int8
// (zero point 0) here on purpose: that is the form our kernel folds exactly. Per-axis scales are the
// natural next step and map onto the kernel's per-row (activation) and per-column (weight) scales —
// see docs/onnx-pack.md.
//
// usage: onebit_ep_qdq_proof [path/to/libonebit_ep.so]

#include <onnxruntime_c_api.h>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const OrtApi* api = nullptr;
const OrtModelEditorApi* editor = nullptr;

int64_t kM = 8;
int64_t kK = 16;
int64_t kN = 6;

// x_q and w_q are int8 and quantised from a deterministic zero-mean source, so a scalar scale is
// enough to exercise the folding without hiding an error behind saturation.
std::vector<int8_t> xq, wq;
std::vector<int8_t> ref_out, ep_out;
float s_x = 0.0f;
float s_w = 0.0f;
float s_y = 0.0f;

int fail(const char* what, const OrtStatus* st) {
    std::fprintf(stderr, "onebit-ort-qdq: FAIL %s: %s\n", what,
                 st ? api->GetErrorMessage(st) : "-");
    if (st) api->ReleaseStatus(const_cast<OrtStatus*>(st));
    return 1;
}

// The float source is deterministic (an LCG, no libm): a diff against the CPU EP must be
// reproducible, not a function of a random seed or a platform's sin().
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
    float xmax = 0.0f, wmax = 0.0f;
    for (float& v : xf) {
        v = (float)uniform(&seed);
        if (v < 0) v = -v;
        if (v > xmax) xmax = v;
    }
    for (float& v : wf) {
        v = (float)uniform(&seed);
        if (v < 0) v = -v;
        if (v > wmax) wmax = v;
    }
    s_x = xmax / 127.0f;
    s_w = wmax / 127.0f;
    for (size_t i = 0; i < xf.size(); ++i) xq[i] = (int8_t)std::lround(xf[i] / s_x);
    for (size_t i = 0; i < wf.size(); ++i) wq[i] = (int8_t)std::lround(wf[i] / s_w);
    // The output scale comes from the exact integer product, so y_q uses a fraction of the int8 range
    // instead of clipping: a saturated result would hide a scale-folding error rather than show it.
    int32_t zmax = 1;
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t n = 0; n < kN; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < kK; ++k) acc += (int32_t)xq[(size_t)(m * kK + k)] * wq[(size_t)(k * kN + n)];
            if (std::abs(acc) > zmax) zmax = std::abs(acc);
        }
    }
    s_y = s_x * s_w * (float)zmax / 64.0f;
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

// Adds a 1-D (rank 0 for scalars) initializer; the graph copies the data, so `data` only has to
// outlive this call.
OrtStatus* add_scalar(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      const void* data, size_t size) {
    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi)) return st;
    OrtValue* v = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(mi, const_cast<void*>(data), size, nullptr, 0,
                                                        elem, &v);
    api->ReleaseMemoryInfo(mi);
    if (!st) st = editor->AddInitializerToGraph(graph, name, v, false);  // takes ownership
    return st;
}

OrtStatus* add_matrix(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      void* data, size_t size, const int64_t* dims) {
    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi)) return st;
    OrtValue* v = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(mi, data, size, dims, 2, elem, &v);
    api->ReleaseMemoryInfo(mi);
    if (!st) st = editor->AddInitializerToGraph(graph, name, v, false);
    return st;
}

OrtStatus* add_node(OrtGraph* graph, const char* op, const char* name, const char** in, size_t n_in,
                    const char** out, size_t n_out) {
    OrtNode* node = nullptr;
    if (OrtStatus* st = editor->CreateNode(op, "", name, in, n_in, out, n_out, nullptr, 0, &node))
        return st;
    return editor->AddNodeToGraph(graph, node);  // takes ownership
}

OrtStatus* build_model(OrtModel** out) {
    const int64_t x_dims[2] = {kM, kK};
    const int64_t w_dims[2] = {kK, kN};
    const int64_t y_dims[2] = {kM, kN};

    OrtTypeInfo* x_type = nullptr;
    OrtTypeInfo* y_type = nullptr;
    if (OrtStatus* st = make_type(x_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &x_type)) return st;
    if (OrtStatus* st = make_type(y_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &y_type)) {
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

    const int8_t zp = 0;
    if (!st) st = add_scalar(graph, "zp", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &zp, sizeof(zp));
    if (!st) st = add_scalar(graph, "s_x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_x, sizeof(s_x));
    if (!st) st = add_scalar(graph, "s_w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_w, sizeof(s_w));
    if (!st) st = add_scalar(graph, "s_y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &s_y, sizeof(s_y));
    if (!st) st = add_matrix(graph, "w_q", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, wq.data(),
                             wq.size(), w_dims);
    if (st) {
        api->ReleaseGraph(graph);
        return st;
    }

    const char* dq_x_in[3] = {"x_q", "s_x", "zp"};
    const char* dq_w_in[3] = {"w_q", "s_w", "zp"};
    const char* mm_in[2] = {"x_dq", "w_dq"};
    const char* q_in[3] = {"y", "s_y", "zp"};
    const char* x_dq_out[1] = {"x_dq"};
    const char* w_dq_out[1] = {"w_dq"};
    const char* y_out[1] = {"y"};
    const char* yq_out[1] = {"y_q"};
    st = add_node(graph, "DequantizeLinear", "dq_x", dq_x_in, 3, x_dq_out, 1);
    if (!st) st = add_node(graph, "DequantizeLinear", "dq_w", dq_w_in, 3, w_dq_out, 1);
    if (!st) st = add_node(graph, "MatMul", "qdq_matmul", mm_in, 2, y_out, 1);
    if (!st) st = add_node(graph, "QuantizeLinear", "q_y", q_in, 3, yq_out, 1);
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
    const int64_t x_dims[2] = {kM, kK};
    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi))
        return fail("cpu memory info", st);
    OrtValue* x = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(mi, xq.data(), xq.size(), x_dims, 2,
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
    const std::string trace_path = std::string("/tmp/onebit_ep_trace.") + std::to_string(::getpid()) + ".log";
    std::remove(trace_path.c_str());
    setenv("ONEBIT_EP_TRACE", "1", 1);

    fill_inputs();
    std::printf("onebit ort-qdq: A8W8 QDQ MatMul, m=%lld k=%lld n=%lld s_x=%.6f s_w=%.6f s_y=%.6f\n",
                (long long)kM, (long long)kK, (long long)kN, (double)s_x, (double)s_w,
                (double)s_y);

    OrtEnv* env = nullptr;
    if (OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "onebit-qdq", &env))
        return fail("create env", st);
    OrtModel* model = nullptr;
    if (OrtStatus* st = build_model(&model)) {
        int rc = fail("build model", st);
        api->ReleaseEnv(env);
        return rc;
    }

    OrtSession* cpu_session = nullptr;
    OrtSession* ep_session = nullptr;
    int rc = 0;
    if (OrtStatus* st = make_session(env, model, false, ep_lib, &cpu_session)) {
        rc = fail("cpu session", st);
    } else if (OrtStatus* st = make_session(env, model, true, ep_lib, &ep_session)) {
        rc = fail("ep session", st);
    } else if (run_once(cpu_session, &ref_out) || run_once(ep_session, &ep_out)) {
        rc = 1;
    } else {
        int worst = 0;
        size_t differing = 0;
        for (size_t i = 0; i < ref_out.size(); ++i) {
            const int d = std::abs((int)ref_out[i] - (int)ep_out[i]);
            if (d) ++differing;
            if (d > worst) worst = d;
        }
        const std::string trace = slurp(trace_path.c_str());
        const bool took_matmul = trace.find("matmul") != std::string::npos;
        const bool took_dq = trace.find("DequantizeLinear") != std::string::npos ||
                             trace.find("dequantize") != std::string::npos;
        std::printf("onebit-ort-qdq: int8 outputs differ in %zu of %zu values, worst |d|=%d\n",
                    differing, ref_out.size(), worst);
        std::printf("onebit-ort-qdq: sample ref[0]=%d ep[0]=%d\n", (int)ref_out[0], (int)ep_out[0]);
        for (size_t pos = 0; pos < trace.size();) {
            size_t nl = trace.find('\n', pos);
            if (nl == std::string::npos) nl = trace.size();
            std::printf("  %s\n", trace.substr(pos, nl - pos).c_str());
            pos = nl + 1;
        }
        // int8 in, int8 out: one LSB is the quantisation step itself, so differences of one are the
        // two kernels summing in a different order, not a wrong result.
        if (!took_matmul) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL OneBitEp never ran the MatMul inside the QDQ pattern\n");
            rc = 1;
        } else if (took_dq) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL OneBitEp claimed a DequantizeLinear node\n");
            rc = 1;
        } else if (worst > 1) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL int8 outputs differ by %d\n", worst);
            rc = 1;
        } else {
            std::printf("onebit-ort-qdq: PASS OneBitEp ran the A8W8 QDQ MatMul, worst |d|=%d LSB%s\n",
                        worst, differing ? "" : " (exact)");
        }
    }

    if (ep_session) api->ReleaseSession(ep_session);
    if (cpu_session) api->ReleaseSession(cpu_session);
    api->ReleaseModel(model);
    api->ReleaseEnv(env);
    return rc;
}
