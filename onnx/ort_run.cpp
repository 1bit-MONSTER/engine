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

// `1bit ort-run`: run a plain ONNX model through ONNX Runtime with our plugin EP registered first
// and the CPU EP last, so a model's matmuls land on whichever provider we taught to take them.
//
//   ort-run --model qwen3-0.6b.onnx [--ep ./libonebit_ep.so] --ids "15 23 40" -n 8
//           [--dump-logits DIR] [--threads N]
//
// The graph's interface is fixed to what onnx/export_qwen3.py emits: `input_ids` int64 [1,S],
// `attention_mask` int64 [1,S] (omit it with --no-mask) and a `logits` float32 [1,S,V] output. A
// decoding step is a full re-run over the growing sequence, because the exported graph carries no
// KV cache: the tokens/s printed here is that graph's rate, not the forward route's, and the two are
// compared only where both do the same work (see docs/onnx.md, "Qwen3-0.6B").

#include <fcntl.h>
#include <unistd.h>

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const OrtApi* api = nullptr;

int fail(const char* what, const OrtStatus* st) {
    std::fprintf(stderr, "onebit-ort-run: FAIL %s: %s\n", what, st ? api->GetErrorMessage(st) : "-");
    if (st) api->ReleaseStatus(const_cast<OrtStatus*>(st));
    return 1;
}

// One registration pass: every --ep library in the order given, then the CPU device last, because
// ORT offers each node to the providers in session order and the first one that returns a kernel
// keeps it. That ordering is the whole point of the two-engine story: our NPU EP is asked first for
// the matmul shapes it has an ELF for, a GPU EP (HIP/ROCM) is asked next, and the CPU EP mopped up.
OrtStatus* make_session(OrtEnv* env, const char* model_path, const std::vector<const char*>& eps,
                        int threads, OrtSession** out) {
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = api->CreateSessionOptions(&so)) return st;
    OrtStatus* st = api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    if (!st) st = api->SetIntraOpNumThreads(so, threads);
    std::vector<const OrtEpDevice*> order;                  // everything registered so far
    std::vector<std::vector<const OrtEpDevice*>> groups;    // one group per library, in order
    for (size_t i = 0; !st && i < eps.size(); ++i) {
        // Each library needs its own registration name, so the log can say which one a device came
        // from; ours keeps its own name when it is the only one, which is what the proof greps for.
        const std::string reg = eps.size() == 1 ? "onebit_ep" : "onebit_ep" + std::to_string(i + 1);
        st = api->RegisterExecutionProviderLibrary(env, reg.c_str(), eps[i]);
        const OrtEpDevice* const* devices = nullptr;
        size_t n = 0;
        if (!st) st = api->GetEpDevices(env, &devices, &n);
        std::vector<const OrtEpDevice*> mine;
        for (size_t d = 0; !st && d < n; ++d) {
            const char* name = api->EpDevice_EpName(devices[d]);
            if (!name || std::strcmp(name, "CPUExecutionProvider") == 0) continue;
            // A device belongs to this library if it was not in the list before the registration;
            // the roster only grows, so anything new after our call is ours.
            bool known = false;
            for (const OrtEpDevice* prev : order) known = known || prev == devices[d];
            if (!known) {
                order.push_back(devices[d]);
                mine.push_back(devices[d]);
            }
        }
        if (!st && mine.empty()) st = api->CreateStatus(ORT_FAIL, "an EP library registered no devices");
        groups.push_back(std::move(mine));
    }
    // One V2 call per library: the API refuses a single call that mixes devices of two EPs, and the
    // call order is the priority order, so ours is asked first, then any GPU EP, then the CPU EP.
    for (size_t g = 0; !st && g < groups.size(); ++g) {
        st = api->SessionOptionsAppendExecutionProvider_V2(so, env, groups[g].data(),
                                                          groups[g].size(), nullptr, nullptr, 0);
    }
    // The CPU EP is a device like any other and must come last; a V2 call carries devices of one EP.
    const OrtEpDevice* const* all = nullptr;
    size_t n = 0;
    if (!st) st = api->GetEpDevices(env, &all, &n);
    for (size_t i = 0; !st && i < n; ++i) {
        const char* name = api->EpDevice_EpName(all[i]);
        if (name && std::strcmp(name, "CPUExecutionProvider") == 0) {
            const OrtEpDevice* one[1] = {all[i]};
            st = api->SessionOptionsAppendExecutionProvider_V2(so, env, one, 1, nullptr, nullptr, 0);
        }
    }
    if (!st) st = api->CreateSession(env, model_path, so, out);
    api->ReleaseSessionOptions(so);
    return st;
}

struct Args {
    const char* model = nullptr;
    std::vector<const char*> eps;  // plugin EP libraries, in registration order
    std::vector<int64_t> ids;
    int steps = 8;
    const char* dump = nullptr;
    bool mask = true;
    int threads = 1;
};

bool parse_args(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool more = i + 1 < argc;
        if (arg == "--model" && more) a->model = argv[++i];
        else if (arg == "--ep" && more) a->eps.push_back(argv[++i]);
        else if ((arg == "--steps" || arg == "-n") && more) a->steps = std::atoi(argv[++i]);
        else if (arg == "--dump-logits" && more) a->dump = argv[++i];
        else if (arg == "--threads" && more) a->threads = std::atoi(argv[++i]);
        else if (arg == "--no-mask") a->mask = false;
        else if (arg == "--ids" && more) {
            const char* p = argv[++i];
            while (*p) {
                char* end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                a->ids.push_back(v);
                p = end;
                while (*p == ' ' || *p == ',') ++p;
            }
        } else {
            std::fprintf(stderr, "onebit-ort-run: unknown argument %s\n", arg.c_str());
            return false;
        }
    }
    if (!a->model || a->ids.empty()) {
        std::fprintf(stderr,
                     "usage: ort-run --model <file.onnx> [--ep <lib.so>]... --ids \"<id> <id>\"\n"
                     "               [-n N] [--dump-logits DIR] [--no-mask] [--threads N]\n"
                     "       --ep may be repeated: the EPs are registered in the order given and the\n"
                     "       CPU EP is always appended last (docs/onnx.md, \"Two engines\").\n");
        return false;
    }
    return true;
}

// The logits row of the last position: that is the distribution the next token comes from.
int run_step(OrtSession* session, OrtMemoryInfo* mi, const std::vector<int64_t>& ids, bool with_mask,
             OrtValue** out, std::vector<int64_t>* shape) {
    // mi is owned by the caller and outlives every tensor made from it: a tensor keeps a reference
    // to its memory info, and releasing the info first leaves that reference dangling.
    const int64_t dims[2] = {1, (int64_t)ids.size()};
    std::vector<int64_t> mask(ids.size(), 1);
    OrtValue* in_ids = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
        mi, (void*)ids.data(), sizeof(int64_t) * ids.size(), dims, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_ids);
    OrtValue* in_mask = nullptr;
    if (!st && with_mask) {
        st = api->CreateTensorWithDataAsOrtValue(mi, mask.data(), sizeof(int64_t) * mask.size(), dims,
                                                 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_mask);
    }
    if (st) {
        if (in_ids) api->ReleaseValue(in_ids);
        return fail("input tensors", st);
    }

    const char* names[2] = {"input_ids", "attention_mask"};
    const OrtValue* vals[2] = {in_ids, in_mask};
    const char* out_names[1] = {"logits"};
    st = api->Run(session, nullptr, names, vals, with_mask ? 2 : 1, out_names, 1, out);
    api->ReleaseValue(in_ids);
    if (in_mask) api->ReleaseValue(in_mask);
    if (st) return fail("Run", st);

    OrtTensorTypeAndShapeInfo* info = nullptr;
    st = api->GetTensorTypeAndShape(*out, &info);
    if (!st) {
        size_t rank = 0;
        st = api->GetDimensionsCount(info, &rank);
        shape->assign(rank, 0);
        if (!st) st = api->GetDimensions(info, shape->data(), rank);
    }
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (st) return fail("logits shape", st);
    return 0;
}

}  // namespace

int onebit_ort_run_main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, &a)) return 2;

    const OrtApiBase* base = OrtGetApiBase();
    api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        std::fprintf(stderr, "onebit-ort-run: FAIL the host serves no OrtApi v%d\n", ORT_API_VERSION);
        return 1;
    }

    OrtEnv* env = nullptr;
    if (OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onebit-ort-run", &env))
        return fail("create env", st);

    OrtMemoryInfo* mi = nullptr;
    if (OrtStatus* st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi)) {
        int rc = fail("cpu memory info", st);
        api->ReleaseEnv(env);
        return rc;
    }

    OrtSession* session = nullptr;
    if (OrtStatus* st = make_session(env, a.model, a.eps, a.threads, &session)) {
        int rc = fail("create session", st);
        api->ReleaseMemoryInfo(mi);
        api->ReleaseEnv(env);
        return rc;
    }

    std::printf("onebit ort-run\n  ort     : %s (api v%d)\n  model   : %s\n", base->GetVersionString(),
                ORT_API_VERSION, a.model);
    for (size_t i = 0; i < a.eps.size(); ++i)
        std::printf("  ep%-6zu: %s\n", i + 1, a.eps[i]);
    if (a.eps.empty()) std::printf("  ep      : (CPU only)\n");

    // Prefill, then greedy steps: each step re-runs the whole sequence (no KV cache in this graph),
    // so the loop time grows with the sequence. The printed rate is per decode step, measured over
    // the steps themselves, not over the prefill.
    auto t0 = std::chrono::steady_clock::now();
    OrtValue* logits = nullptr;
    std::vector<int64_t> shape;
    if (run_step(session, mi, a.ids, a.mask, &logits, &shape)) {
        api->ReleaseSession(session);
        api->ReleaseEnv(env);
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t vocab = shape.empty() ? 0 : shape.back();
    std::printf("  prefill : %zu tokens in %.1f ms (logits [%lld])\n", a.ids.size(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(), (long long)vocab);

    auto dump = [&](int step) {
        if (!a.dump) return;
        size_t count = 1;
        for (int64_t d : shape) count *= (size_t)d;
        if (count < (size_t)vocab) return;
        const float* p = nullptr;
        if (OrtStatus* st = api->GetTensorData(logits, (const void**)&p)) {
            api->ReleaseStatus(st);
            return;
        }
        char path[512];
        std::snprintf(path, sizeof(path), "%s/logits_%03d.bin", a.dump, step);
        // open-then-fdopen, the shape the rest of the engine uses (tools/moe_trace.cpp), with 0600:
        // a dumped logits row is model output, so it is not created world-readable.
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (FILE* f = fd >= 0 ? ::fdopen(fd, "wb") : nullptr) {
            std::fwrite(p + (count - (size_t)vocab), sizeof(float), (size_t)vocab, f);
            std::fclose(f);
        }
    };
    dump(-1);

    std::vector<int64_t> ids = a.ids;
    std::vector<int64_t> generated;
    int64_t best = 0;
    double ms = 0.0;
    for (int step = 0; step < a.steps; ++step) {
        const float* p = nullptr;
        if (OrtStatus* st = api->GetTensorData(logits, (const void**)&p)) {
            api->ReleaseValue(logits);
            int rc = fail("read logits", st);
            api->ReleaseSession(session);
            api->ReleaseMemoryInfo(mi);
            api->ReleaseEnv(env);
            return rc;
        }
        best = 0;
        const float* row = p + (size_t)vocab * (size_t)(ids.size() - 1);
        for (int64_t v = 1; v < vocab; ++v)
            if (row[v] > row[best]) best = v;
        generated.push_back(best);
        ids.push_back(best);
        api->ReleaseValue(logits);
        // Run() allocates the next logits value through this pointer; leaving the released one here
        // lets ORT inspect freed memory (it reported the input's int64 type as the output's).
        logits = nullptr;

        const auto s0 = std::chrono::steady_clock::now();
        if (run_step(session, mi, ids, a.mask, &logits, &shape)) break;
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        dump(step);
    }

    std::printf("  decode  : %zu tokens in %.1f ms (%.2f tok/s, %.2f ms/token)\n", generated.size(),
                ms, ms > 0.0 ? 1000.0 * (double)generated.size() / ms : 0.0,
                generated.empty() ? 0.0 : ms / (double)generated.size());
    std::printf("  ids     :");
    for (int64_t id : generated) std::printf(" %lld", (long long)id);
    std::printf("\n");

    if (logits) api->ReleaseValue(logits);
    api->ReleaseSession(session);
    api->ReleaseMemoryInfo(mi);
    api->ReleaseEnv(env);
    return 0;
}
