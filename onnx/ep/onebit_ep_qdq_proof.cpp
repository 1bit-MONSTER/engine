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
// Two phases, two claims:
//  1. the four nodes are claimed as ONE fused node and the int8 outputs are identical to the CPU EP's
//     (worst |d| = 0 LSB). The trace read back at the end is the evidence for which kernel ran: the
//     fused path logs `qdq rows=...` and never the float `matmul batch=...` line, so the float MatMuls
//     cannot be what produced these numbers. ONEBIT_EP_QDQ=0 turns fusion off and the same binary
//     runs the control: one claimed MatMul, the float kernel, the same int8 outputs.
//  2. a larger shape whose float intermediates are big enough to see: the CPU EP materialises x_dq,
//     w_dq and y as float32 (4 bytes per element) where the fused path keeps int8 (1 byte). The peak
//     RSS growth of each run is measured around the run itself, so the difference is the tensors and
//     not the two sessions' own graphs.
//
// usage: onebit_ep_qdq_proof [path/to/libonebit_ep.so]

#include <onnxruntime_c_api.h>

#include <unistd.h>

#include <malloc.h>

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

// The RSS phase's shape: x_dq is 1 MB of float32 and w_dq 4 MB, against 0.25 MB and 1 MB of int8.
constexpr int64_t kRssM = 256;
constexpr int64_t kRssK = 1024;
constexpr int64_t kRssN = 1024;

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

// An initializer whose memory ORT owns, filled from a copy of `data`.
//
// Not CreateTensorWithDataAsOrtValue: that borrows the caller's memory, which then has to outlive the
// session. The scales and the zero point are ordinary locals here, and borrowing them produced a
// garbage zero point in a no-optimization build as soon as the stack slot was reused -- every int8
// output then differed from the CPU EP's. ORT's own small-initializer path costs a memcpy and removes
// the whole class.
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

OrtStatus* add_scalar(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      const void* data, size_t size) {
    return add_initializer(graph, name, elem, data, size, nullptr, 0);
}

OrtStatus* add_matrix(OrtGraph* graph, const char* name, ONNXTensorElementDataType elem,
                      void* data, size_t size, const int64_t* dims) {
    return add_initializer(graph, name, elem, data, size, dims, 2);
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
    // One thread: the RSS phase measures the tensors, and MLAS's per-thread scratch buffers would be
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
int compare(const char* label, size_t* differing) {
    int worst = 0;
    *differing = 0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        const int d = std::abs((int)ref_out[i] - (int)ep_out[i]);
        if (d) ++(*differing);
        if (d > worst) worst = d;
    }
    std::printf("onebit-ort-qdq: %s m=%lld k=%lld n=%lld: %zu of %zu int8 values differ, "
                "worst |d|=%d LSB (ref[0]=%d ep[0]=%d)\n",
                label, (long long)kM, (long long)kK, (long long)kN, *differing, ref_out.size(),
                worst, (int)ref_out[0], (int)ep_out[0]);
    return worst;
}

}  // namespace

// One phase: build the graph for the current kM/kK/kN, run the CPU EP control session and the EP
// session on it, and compare their int8 outputs. `measure_rss` adds the peak-RSS growth of each run.
// Each phase gets its own OrtEnv because a plugin library is registered per env, once.
int run_phase(const char* label, const char* ep_lib, bool measure_rss, int* worst, size_t* differing,
              long* peak_cpu, long* peak_fused) {
    OrtEnv* env = nullptr;
    if (OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "onebit-qdq", &env))
        return fail("create env", st);
    OrtModel* model = nullptr;
    OrtSession* cpu_session = nullptr;
    OrtSession* ep_session = nullptr;
    int rc = 0;
    if (OrtStatus* st = build_model(&model)) {
        rc = fail("build model", st);
    } else if (OrtStatus* st = make_session(env, model, false, ep_lib, &cpu_session)) {
        rc = fail("cpu session", st);
    } else if (OrtStatus* st = make_session(env, model, true, ep_lib, &ep_session)) {
        rc = fail("ep session", st);
    } else if (measure_rss) {
        *peak_cpu = run_peak_kb(cpu_session, &ref_out);
        *peak_fused = run_peak_kb(ep_session, &ep_out);
        *worst = compare(label, differing);
    } else if (run_once(cpu_session, &ref_out) || run_once(ep_session, &ep_out)) {
        rc = 1;
    } else {
        *worst = compare(label, differing);
    }
    if (ep_session) api->ReleaseSession(ep_session);
    if (cpu_session) api->ReleaseSession(cpu_session);
    if (model) api->ReleaseModel(model);
    api->ReleaseEnv(env);
    return rc;
}

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

    // ---- phase 1: the proof graph, exactness against the CPU EP --------------------------------
    fill_inputs();
    std::printf("onebit ort-qdq: A8W8 QDQ MatMul, m=%lld k=%lld n=%lld s_x=%.6f s_w=%.6f s_y=%.6f\n",
                (long long)kM, (long long)kK, (long long)kN, (double)s_x, (double)s_w,
                (double)s_y);
    int worst_small = 0;
    size_t differing_small = 0;
    long peak_cpu = -1, peak_fused = -1;
    long unused_cpu = -1, unused_fused = -1;
    int rc = run_phase("phase 1", ep_lib, false, &worst_small, &differing_small, &unused_cpu,
                       &unused_fused);

    // ---- phase 2: what the float intermediates cost --------------------------------------------
    int worst_big = 0;
    size_t differing_big = 0;
    if (!rc) {
        kM = kRssM;
        kK = kRssK;
        kN = kRssN;
        fill_inputs();
        rc = run_phase("phase 2", ep_lib, true, &worst_big, &differing_big, &peak_cpu, &peak_fused);
        if (!rc) {
            if (peak_cpu < 0 || peak_fused < 0) {
                std::printf("onebit-ort-qdq: peak RSS: not measurable here (no VmHWM reset via "
                            "/proc/self/clear_refs), memory gate skipped\n");
            } else {
                std::printf("onebit-ort-qdq: peak RSS growth: CPU EP run +%ld kB, fused run +%ld kB "
                            "(delta %ld kB)\n",
                            peak_cpu, peak_fused, peak_cpu - peak_fused);
            }
        }
    }

    // ---- what actually ran, read back from the EP's own trace ----------------------------------
    const std::string trace = slurp(trace_path.c_str());
    for (size_t pos = 0; pos < trace.size();) {
        size_t nl = trace.find('\n', pos);
        if (nl == std::string::npos) nl = trace.size();
        std::printf("  %s\n", trace.substr(pos, nl - pos).c_str());
        pos = nl + 1;
    }
    const size_t fused_lines = count_substr(trace, "qdq rows=");
    const size_t float_lines = count_substr(trace, "matmul batch=");
    const size_t groups = count_substr(trace, "qdq_groups=1");
    const bool fused = fused_lines > 0;
    if (fused) {
        std::printf("onebit-ort-qdq: path: fused (%zu QDQ group(s) claimed as one node, "
                    "%zu int8 kernel runs, %zu float MatMul runs)\n",
                    groups, fused_lines, float_lines);
    } else {
        std::printf("onebit-ort-qdq: path: unfused (%zu float MatMul runs, no fused QDQ kernel)\n",
                    float_lines);
    }

    if (!rc) {
        if (worst_small != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL phase 1 differs by %d LSB from the CPU EP\n",
                         worst_small);
            rc = 1;
        } else if (fused && worst_big != 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL phase 2 differs by %d LSB from the CPU EP\n",
                         worst_big);
            rc = 1;
        } else if (fused) {
            // Fused: the float MatMul kernel must not have run at all, and the peak RSS growth of the
            // run must be smaller than the CPU EP's, because the float tensors never exist.
            if (float_lines != 0) {
                std::fprintf(stderr,
                             "onebit-ort-qdq: FAIL the float MatMul kernel ran inside the fused path\n");
                rc = 1;
            } else if (groups == 0) {
                std::fprintf(stderr, "onebit-ort-qdq: FAIL no QDQ group was claimed as one node\n");
                rc = 1;
            } else if (peak_cpu > 0 && peak_fused > 0 && peak_fused >= peak_cpu) {
                std::fprintf(stderr,
                             "onebit-ort-qdq: FAIL the fused run grew RSS by %ld kB, not less than "
                             "the CPU EP's %ld kB\n",
                             peak_fused, peak_cpu);
                rc = 1;
            } else {
                std::printf("onebit-ort-qdq: PASS OneBitEp fused the A8W8 QDQ group, worst |d|=%d LSB "
                            "(exact), %zu int8 kernel runs, 0 float MatMul runs, peak RSS -%ld kB\n",
                            worst_small, fused_lines, peak_cpu > 0 ? peak_cpu - peak_fused : 0L);
            }
        } else if (float_lines == 0) {
            std::fprintf(stderr, "onebit-ort-qdq: FAIL OneBitEp ran neither the group nor the MatMul\n");
            rc = 1;
        } else {
            // The control only has to prove the per-node path still works; its phase-2 number is a
            // measurement, not a gate. It is 1 LSB on a handful of values because the float path
            // accumulates (x_q*s_x)*(w_q*s_w) in float where the fused path sums exact int32 first.
            std::printf("onebit-ort-qdq: PASS OneBitEp ran the A8W8 QDQ MatMul unfused "
                        "(control: ONEBIT_EP_QDQ=0), phase 1 worst |d|=%d LSB%s, phase 2 worst |d|=%d "
                        "(informational)\n",
                        worst_small, differing_small ? "" : " (exact)", worst_big);
        }
    }
    return rc;
}
