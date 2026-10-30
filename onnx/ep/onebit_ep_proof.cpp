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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// limitations under the License.

// Proof for deliverable 2: ONNX Runtime runs a two-op graph with libonebit_ep.so registered FIRST
// and the CPU EP last, our EP takes the MatMul, and the numbers match the CPU EP run.
//
// Everything here is the real path -- the model is built through OrtModelEditorApi (no .onnx file,
// no protobuf, nothing generated), the EP is loaded through RegisterExecutionProviderLibrary exactly
// as a user would load it, and the ownership evidence is the trace the library writes from inside its
// own Compute callback: ORT cannot print that line unless our kernel is the one that ran.
//
// usage: onebit_ep_proof [path/to/libonebit_ep.so]

#include <onnxruntime_c_api.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const OrtApi* api = nullptr;
const OrtModelEditorApi* editor = nullptr;

int fail(const char* what, const OrtStatus* st) {
    std::fprintf(stderr, "onebit-ort-ep: FAIL %s: %s\n", what, st ? api->GetErrorMessage(st) : "-");
    if (st) api->ReleaseStatus(const_cast<OrtStatus*>(st));
    return 1;
}

// The graph under test: y = Relu(x @ W), x [M,K] float input, W [K,N] float initializer.
// Defaults are tiny so that what is under test is the EP ABI; ONEBIT_EP_PROOF_M/K/N size it to a
// real projection, which the NPU path needs because its kernel is built for one K and one N.
int64_t kM = 4;
int64_t kK = 8;
int64_t kN = 5;

// Deterministic on purpose: a diff against the CPU EP must be reproducible across runs, not a
// function of the random seed.
std::vector<float> x_data, w_data, ref_out, ep_out;

void read_dims() {
    auto get = [](const char* name, int64_t fallback) {
        const char* v = std::getenv(name);
        return (v && *v) ? (int64_t)std::atoll(v) : fallback;
    };
    kM = get("ONEBIT_EP_PROOF_M", kM);
    kK = get("ONEBIT_EP_PROOF_K", kK);
    kN = get("ONEBIT_EP_PROOF_N", kN);
    x_data.assign((size_t)(kM * kK), 0.0f);
    w_data.assign((size_t)(kK * kN), 0.0f);
    ref_out.assign((size_t)(kM * kN), 0.0f);
    ep_out.assign((size_t)(kM * kN), 0.0f);
}

void fill_inputs() {
    // Zero-mean and uncorrelated: a periodic pattern makes the K-long sum cancel, and then int8
    // noise measured against the (tiny) result looks like a kernel bug.
    uint32_t rng = 12345u;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return ((float)(rng >> 8) / 8388608.0f) - 1.0f;  // [-1, 1)
    };
    for (int64_t i = 0; i < kM * kK; ++i) x_data[(size_t)i] = next();
    for (int64_t i = 0; i < kK * kN; ++i) w_data[(size_t)i] = next() * 0.5f;
}

OrtStatus* make_tensor_type(const int64_t* dims, size_t rank, OrtTypeInfo** out) {
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = api->CreateTensorTypeAndShapeInfo(&tsi)) return st;
    OrtStatus* st = api->SetTensorElementType(tsi, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (!st) st = api->SetDimensions(tsi, dims, rank);
    if (!st) st = editor->CreateTensorTypeInfo(tsi, out);
    api->ReleaseTensorTypeAndShapeInfo(tsi);
    return st;
}

// Builds the model once; both sessions run the same OrtModel, so a numeric difference can only come
// from the EP that executed the node.
OrtStatus* build_model(OrtModel** out) {
    const int64_t x_dims[2] = {kM, kK};
    const int64_t w_dims[2] = {kK, kN};
    const int64_t y_dims[2] = {kM, kN};

    OrtTypeInfo* x_type = nullptr;
    OrtTypeInfo* y_type = nullptr;
    if (OrtStatus* st = make_tensor_type(x_dims, 2, &x_type)) return st;
    if (OrtStatus* st = make_tensor_type(y_dims, 2, &y_type)) {
        api->ReleaseTypeInfo(x_type);
        return st;
    }
    OrtValueInfo* x_vi = nullptr;
    OrtValueInfo* y_vi = nullptr;
    OrtStatus* st = editor->CreateValueInfo("x", x_type, &x_vi);
    api->ReleaseTypeInfo(x_type);
    if (!st) st = editor->CreateValueInfo("y", y_type, &y_vi);
    api->ReleaseTypeInfo(y_type);
    if (st) {
        api->ReleaseValueInfo(x_vi);
        return st;
    }

    // The initializer owns its data: AddInitializerToGraph(tensor, data_is_external=false) copies it,
    // so w_data only has to outlive the call.
    OrtMemoryInfo* mi = nullptr;
    OrtValue* w_val = nullptr;
    st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi);
    if (!st) {
        st = api->CreateTensorWithDataAsOrtValue(mi, w_data.data(), sizeof(float) * w_data.size(), w_dims, 2,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &w_val);
    }
    api->ReleaseMemoryInfo(mi);
    if (st) {
        api->ReleaseValueInfo(x_vi);
        api->ReleaseValueInfo(y_vi);
        return st;
    }

    OrtGraph* graph = nullptr;
    if (!st) st = editor->CreateGraph(&graph);
    if (!st) st = editor->SetGraphInputs(graph, &x_vi, 1);   // graph takes ownership
    if (!st) st = editor->SetGraphOutputs(graph, &y_vi, 1); // graph takes ownership
    if (!st) st = editor->AddInitializerToGraph(graph, "W", w_val, false);  // takes ownership
    if (st) {
        api->ReleaseGraph(graph);
        return st;
    }

    const char* in_names[2] = {"x", "W"};
    const char* mm_names[1] = {"mm"};
    const char* out_names[1] = {"y"};
    OrtNode* matmul = nullptr;
    OrtNode* relu = nullptr;
    st = editor->CreateNode("MatMul", "", "onebit_matmul", in_names, 2, mm_names, 1, nullptr, 0,
                            &matmul);
    if (!st) st = editor->AddNodeToGraph(graph, matmul);
    if (!st) st = editor->CreateNode("Relu", "", "onebit_relu", mm_names, 1, out_names, 1, nullptr,
                                     0, &relu);
    if (!st) st = editor->AddNodeToGraph(graph, relu);
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

OrtStatus* make_session(OrtEnv* env, const OrtModel* model, bool with_ep, const char* ep_lib,
                        OrtSession** out) {
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = api->CreateSessionOptions(&so)) return st;
    OrtStatus* st = api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC);

    // Registration order is strict priority with no tie-break, so our devices are appended first
    // and the CPU device last. ORT 1.23 has no legacy AppendExecutionProvider_CPU any more: the CPU
    // EP is itself an OrtEpDevice, so it is appended the same way ours is.
    if (!st) {
        if (with_ep && ep_lib) st = api->RegisterExecutionProviderLibrary(env, "onebit_ep", ep_lib);
        const OrtEpDevice* const* devices = nullptr;
        size_t num_devices = 0;
        if (!st) st = api->GetEpDevices(env, &devices, &num_devices);
        std::vector<const OrtEpDevice*> ours;
        const OrtEpDevice* cpu = nullptr;
        for (size_t i = 0; !st && i < num_devices; ++i) {
            const char* name = api->EpDevice_EpName(devices[i]);
            const char* vendor = api->EpDevice_EpVendor(devices[i]);
            std::printf("onebit-ort-ep: device[%zu] ep=%s vendor=%s\n", i, name ? name : "?",
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
            st = api->SessionOptionsAppendExecutionProvider_V2(so, env, ours.data(), ours.size(),
                                                              nullptr, nullptr, 0);
        }
        if (!st && cpu) {
            const OrtEpDevice* one[1] = {cpu};
            st = api->SessionOptionsAppendExecutionProvider_V2(so, env, one, 1, nullptr, nullptr, 0);
        }
    }

    if (!st) st = editor->CreateSessionFromModel(env, model, so, out);
    api->ReleaseSessionOptions(so);
    return st;
}

int run_once(OrtSession* session, float* out) {
    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi)) {
        return fail("cpu memory info", st);
    }
    const int64_t x_dims[2] = {kM, kK};
    OrtValue* x = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
        mi, x_data.data(), sizeof(float) * x_data.size(), x_dims, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &x);
    api->ReleaseMemoryInfo(mi);
    if (st) return fail("input tensor", st);

    const char* in_names[1] = {"x"};
    const char* out_names[1] = {"y"};
    OrtValue* y = nullptr;
    st = api->Run(session, nullptr, in_names, (const OrtValue* const*)&x, 1, out_names, 1, &y);
    api->ReleaseValue(x);
    if (st) return fail("Run", st);

    void* p = nullptr;
    st = api->GetTensorMutableData(y, &p);
    if (!st) std::memcpy(out, p, sizeof(float) * (size_t)(kM * kN));
    api->ReleaseValue(y);
    if (st) return fail("read output", st);
    return 0;
}

// Reads back the EP's own trace: one line per node the library executed.
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

// Worst absolute difference, and the reference's own magnitude to scale it by.
//
// Per-element relative error is the wrong metric here: this tensor has entries at 0.000000 (the data
// is zero-mean, so some outputs cancel), and dividing by max(1e-6, |ref|) turns a 1e-7 difference on
// one of them into 1e-1. Error is reported against the result's magnitude instead -- the same bound
// the probe asserts, and the one docs/onnx.md states.
struct Diff {
    float max_abs = 0.0f;
    float max_ref = 0.0f;
};

Diff abs_diff(const float* a, const float* b, size_t n) {
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        d.max_abs = std::max(d.max_abs, std::fabs(a[i] - b[i]));
        d.max_ref = std::max(d.max_ref, std::fabs(a[i]));
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const char* ep_lib = argc > 1 ? argv[1] : ONEBIT_EP_DEFAULT_PATH;
    // The library writes its trace to /tmp/onebit_ep_trace.<pid>.log (onebit_ep.cpp); we are the
    // process that loaded it, so the path is ours to compute.
    const std::string trace_path =
        std::string("/tmp/onebit_ep_trace.") + std::to_string(::getpid()) + ".log";
    std::remove(trace_path.c_str());
    setenv("ONEBIT_EP_TRACE", "1", 1);
    // Caller may preset ONEBIT_EP_OPS (e.g. "none") to run the negative control: the CPU EP then has
    // to take the MatMul and our library must not execute anything.
    const char* ops_env = std::getenv("ONEBIT_EP_OPS");
    setenv("ONEBIT_EP_OPS", ops_env ? ops_env : "MatMul", 0);
    const bool expect_ours = std::strstr(std::getenv("ONEBIT_EP_OPS"), "MatMul") != nullptr;

    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        std::fprintf(stderr, "onebit-ort-ep: FAIL no OrtApi for version %d\n", ORT_API_VERSION);
        return 1;
    }
    editor = api->GetModelEditorApi();
    if (!editor) {
        std::fprintf(stderr, "onebit-ort-ep: FAIL this ORT is a minimal build (no model editor API)\n");
        return 1;
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // line buffered: a crash must not hide the evidence
    std::printf("onebit-ort-ep: ortv=%s api=v%d ep=%s\n", OrtGetApiBase()->GetVersionString(),
                ORT_API_VERSION, ep_lib);

    read_dims();
    fill_inputs();

    OrtEnv* env = nullptr;
    if (OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onebit_ep_proof", &env)) {
        return fail("CreateEnv", st);
    }
    OrtModel* model = nullptr;
    if (OrtStatus* st = build_model(&model)) return fail("build model", st);

    int rc = 0;
    OrtSession* cpu_session = nullptr;
    OrtSession* ep_session = nullptr;
    if (OrtStatus* st = make_session(env, model, false, nullptr, &cpu_session)) {
        rc = fail("cpu session", st);
    } else if (int r = run_once(cpu_session, ref_out.data())) {
        rc = r;
    } else if (OrtStatus* st2 = make_session(env, model, true, ep_lib, &ep_session)) {
        rc = fail("session with OneBitEp", st2);
    } else if (int r = run_once(ep_session, ep_out.data())) {
        rc = r;
    }

    if (!rc) {
        // Ownership: the EP traces every node it executes, so an empty trace means ORT never
        // handed us the MatMul and the matching numbers are the CPU EP's, not ours.
        const std::string trace = slurp(trace_path.c_str());
        const bool ran = trace.find("matmul") != std::string::npos;
        // The device-backed build declines the graph when the NPU is not ours to take (another
        // process holds the device lock). That is the documented behaviour, so it is a skip.
        const bool declined = trace.find("npu_declined") != std::string::npos;
        const bool skip = expect_ours && !ran && declined;
        // npu=1 means the device did the multiply: int8 weights and activations, so the diff is
        // quantisation noise, not float rounding. The bound is the same one the probe asserts.
        const bool on_npu = trace.find("npu=1") != std::string::npos;
        const float tol = on_npu ? 2e-2f : 1e-5f;
        const bool claimed = trace.find("claimed=1") != std::string::npos;
        std::printf("onebit-ort-ep: trace=%s\n", trace.empty() ? "(empty)" : "");
        for (size_t pos = 0; pos < trace.size();) {
            size_t nl = trace.find('\n', pos);
            if (nl == std::string::npos) nl = trace.size();
            std::printf("  %s\n", trace.substr(pos, nl - pos).c_str());
            pos = nl + 1;
        }
        const Diff d = abs_diff(ref_out.data(), ep_out.data(), (size_t)(kM * kN));
        const float rel = d.max_ref > 0.0f ? d.max_abs / d.max_ref : d.max_abs;
        std::printf("onebit-ort-ep: backend=%s max_abs=%.3e of max|ref|=%.3f over %lld values "
                    "(rel %.3e, tol %.0e)\n",
                    on_npu ? "npu" : "host", (double)d.max_abs, (double)d.max_ref,
                    (long long)(kM * kN), (double)rel, (double)tol);
        std::printf("onebit-ort-ep: sample ref[0]=%.6f ep[0]=%.6f ref[last]=%.6f ep[last]=%.6f\n",
                    (double)ref_out[0], (double)ep_out[0], (double)ref_out[(size_t)(kM * kN) - 1],
                    (double)ep_out[(size_t)(kM * kN) - 1]);
        if (skip) {
            std::printf("SKIP OneBitEp declined the MatMul (the device is not available to this build)\n");
        } else if (ran != expect_ours) {
            std::fprintf(stderr, "onebit-ort-ep: FAIL expected OneBitEp %s the MatMul\n",
                         expect_ours ? "to take" : "to leave alone");
            rc = 1;
        } else if (expect_ours && !claimed) {
            std::fprintf(stderr, "onebit-ort-ep: FAIL OneBitEp ran but its capability pass claimed nothing\n");
            rc = 1;
        } else if (!(rel < tol)) {
            std::fprintf(stderr,
                         "onebit-ort-ep: FAIL error %.3e of the result magnitude exceeds %.0e\n",
                         (double)rel, (double)tol);
            rc = 1;
        } else {
            std::printf("onebit-ort-ep: PASS OneBitEp %s the MatMul on the %s, error %.3e of the "
                        "result magnitude\n",
                        expect_ours ? "ran" : "left alone", on_npu ? "npu" : "host", (double)rel);
        }
    }

    if (ep_session) api->ReleaseSession(ep_session);
    if (cpu_session) api->ReleaseSession(cpu_session);
    api->ReleaseModel(model);
    // ReleaseEnv unloads the plugin library (EpLibraryPlugin::Unload), so this is also the regression
    // test for the ReleaseEpFactory signature: ORT dereferences its return value as an OrtStatus.
    api->ReleaseEnv(env);
    return rc;
}
