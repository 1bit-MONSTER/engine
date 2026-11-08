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
// Two kinds of graph:
//   * with --input, ANY graph: each input is a named raw tensor from a file, each output is written
//     where you ask (`--output NAME=FILE`) or into a directory (`--dump-all`), and nothing about
//     names, shapes or roles is assumed -- a vision model, an audio model, a one-op toy;
//   * without it, the Qwen3 LLM interface below (kept because the benchmarks are built on it).
//
// The LLM graph's interface is fixed to what onnx/export_qwen3.py emits: `input_ids` int64 [1,S],
// `attention_mask` int64 [1,S] (omit it with --no-mask) and a `logits` float32 [1,S,V] output. A
// decoding step is a full re-run over the growing sequence, because the exported graph carries no
// KV cache: the tokens/s printed here is that graph's rate, not the forward route's, and the two are
// compared only where both do the same work (see docs/onnx.md, "Qwen3-0.6B").

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cctype>
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

// "Could not find an implementation for ConvInteger(10) node ..." means the runtime has no kernel for
// the op at these input types; it says nothing about the model or an EP. Say that, say what the pin
// does carry for ONNX's integer ops (measured, docs/onnx.md "Integer-op exports"), and list the other
// exports next to the model so the reader knows what to try instead (engine #1436).
void explain_missing_kernel(const char* msg, const char* model_path) {
    static const char kNeedle[] = "Could not find an implementation for ";
    const char* at = msg ? std::strstr(msg, kNeedle) : nullptr;
    if (!at) return;
    std::string op(at + sizeof kNeedle - 1);
    op = op.substr(0, op.find('('));
    std::fprintf(stderr, "onebit-ort-run: the pinned ONNX Runtime has no CPU kernel for %s at this graph's input types: "
                         "a runtime coverage gap, not a model or EP failure\n", op.c_str());
    if (op == "ConvInteger")
        std::fprintf(stderr, "onebit-ort-run:   ConvInteger runs only uint8 x uint8; DynamicQuantizeLinear feeds uint8, "
                             "so an int8-weight export cannot load\n");
    else if (op == "MatMulInteger")
        std::fprintf(stderr, "onebit-ort-run:   MatMulInteger runs uint8 x uint8, uint8 x int8 and int8 x int8, not int8 x uint8\n");
    std::string dir(model_path), self(model_path);
    const size_t slash = dir.rfind('/');
    dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    self = slash == std::string::npos ? self : self.substr(slash + 1);
    std::vector<std::string> other;
    if (DIR* d = opendir(dir.c_str())) {
        while (const dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n != self && n.size() > 5 && n.compare(n.size() - 5, 5, ".onnx") == 0) other.push_back(n);
        }
        closedir(d);
    }
    std::sort(other.begin(), other.end());
    if (!other.empty()) {
        std::string list;
        for (const auto& n : other) list += (list.empty() ? "" : ", ") + n;
        std::fprintf(stderr, "onebit-ort-run:   other exports in %s: %s (not checked; a float or QDQ one uses no integer ops)\n",
                     dir.c_str(), list.c_str());
    }
}

// One registration pass: every --ep library in the order given, then the CPU device last, because
// ORT offers each node to the providers in session order and the first one that returns a kernel
// keeps it. That ordering is the whole point of the two-engine story: our NPU EP is asked first for
// the matmul shapes it has an ELF for, a GPU EP (HIP/ROCM) is asked next, and the CPU EP mopped up.
OrtStatus* make_session(OrtEnv* env, const char* model_path, const std::vector<const char*>& eps,
                        int threads, bool basic, const char* profile, OrtSession** out) {
    OrtSessionOptions* so = nullptr;
    if (OrtStatus* st = api->CreateSessionOptions(&so)) return st;
    // ALL is the right level for a float graph. A QDQ graph is the exception: ALL runs the graph
    // transformations that fold DequantizeLinear/QuantizeLinear back into float ops (and into
    // QLinearMatMul), which is exactly the pattern a quantised EP wants to claim, so --basic keeps the
    // graph as it was exported.
    OrtStatus* st = api->SetSessionGraphOptimizationLevel(so, basic ? ORT_ENABLE_BASIC : ORT_ENABLE_ALL);
    // ORT writes the trace when the session is destroyed, to <prefix>_<timestamp>.json.
    if (!st && profile) st = api->EnableProfiling(so, profile);
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

// One named input for a non-LLM graph: the tensor is read from a raw little-endian file, because
// that is what every preprocessing step in the world can write without a dependency (engine#1313's
// sibling problem, TGPNPU): a vision model's bytes are the caller's, not ours to guess.
struct InSpec {
    std::string name, file, dtype;
    std::vector<int64_t> shape;
};

struct Args {
    const char* model = nullptr;
    std::vector<const char*> eps;  // plugin EP libraries, in registration order
    std::vector<int64_t> ids;
    std::vector<InSpec> inputs;                        // --input name=file:dtype:1x3x640x640
    std::vector<std::pair<std::string, std::string>> outputs;  // --output name=file
    const char* dump_all = nullptr;                    // --dump-all DIR: every output + a sidecar
    int steps = 8;
    const char* dump = nullptr;
    bool mask = true;
    bool cached = false;
    int threads = 1;
    // AMD's benchmark protocol (Ryzen AI docs, npu_oga.html): -l prompt lengths, -g tokens to
    // generate, -w warmup runs, -r repeats. The defaults are theirs, so the block we print is
    // directly comparable with model_benchmark.exe output.
    bool bench = false;
    bool basic = false;  // ORT_ENABLE_BASIC: keep a QDQ graph's DequantizeLinear/QuantizeLinear nodes
    const char* profile = nullptr;  // ORT's own per-op JSON trace: decides what to optimise next
    std::vector<int64_t> lens;
    int gen = 128;
    int warmup = 1;
    int repeat = 5;
};

// Timing samples from one generation run: one TTFT, one entry per decode step, one per argmax.
struct RunStats {
    double ttft_ms = 0.0;
    double e2e_ms = 0.0;
    std::vector<double> token_ms;
    std::vector<double> sample_ms;
};

struct Stat {
    double avg = 0.0, p50 = 0.0, sd = 0.0;
    size_t n = 0;
};

Stat stat_of(std::vector<double> v) {
    Stat s;
    s.n = v.size();
    if (v.empty()) return s;
    for (double x : v) s.avg += x;
    s.avg /= (double)v.size();
    std::sort(v.begin(), v.end());
    s.p50 = v[v.size() / 2];
    double var = 0.0;
    for (double x : v) var += (x - s.avg) * (x - s.avg);
    s.sd = std::sqrt(var / (double)v.size());  // population: these are all the samples there are
    return s;
}

std::vector<int64_t> parse_lens(const char* p) {
    std::vector<int64_t> out;
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        out.push_back(v);
        p = end;
        while (*p == ' ' || *p == ',') ++p;
    }
    return out;
}

bool parse_args(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool more = i + 1 < argc;
        if (arg == "--model" && more) a->model = argv[++i];
        else if (arg == "--ep" && more) a->eps.push_back(argv[++i]);
        else if ((arg == "--steps" || arg == "-n") && more) a->steps = std::atoi(argv[++i]);
        else if (arg == "--dump-logits" && more) a->dump = argv[++i];
        else if (arg == "--basic") a->basic = true;
        else if (arg == "--threads" && more) a->threads = std::atoi(argv[++i]);
        else if (arg == "--profile" && more) a->profile = argv[++i];
        else if (arg == "--no-mask") a->mask = false;
        else if (arg == "--cached") a->cached = true;
        else if (arg == "--bench") a->bench = true;
        else if (arg == "--lens" && more) a->lens = parse_lens(argv[++i]);
        else if (arg == "--gen" && more) a->gen = std::atoi(argv[++i]);
        else if (arg == "--warmup" && more) a->warmup = std::atoi(argv[++i]);
        else if (arg == "--repeat" && more) a->repeat = std::atoi(argv[++i]);
        else if (arg == "--input" && more) {
            // name=file:dtype:SHAPE, SHAPE with x separators. The dtype and the shape are the
            // caller's declaration and are checked against the file's byte count before ORT sees
            // them: a silently short buffer is a wrong answer, not an error.
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            const size_t c1 = eq == std::string::npos ? std::string::npos : spec.find(':', eq + 1);
            const size_t c2 = c1 == std::string::npos ? std::string::npos : spec.rfind(':');
            if (c1 == std::string::npos || c2 == c1) {
                std::fprintf(stderr, "onebit-ort-run: --input wants NAME=FILE:DTYPE:SHAPE (%s)\n", spec.c_str());
                return false;
            }
            InSpec in;
            in.name = spec.substr(0, eq);
            in.file = spec.substr(eq + 1, c1 - eq - 1);
            in.dtype = spec.substr(c1 + 1, c2 - c1 - 1);
            std::string dims = spec.substr(c2 + 1);
            for (size_t at = 0; at <= dims.size();) {
                const size_t x = dims.find('x', at);
                const std::string tok = dims.substr(at, x == std::string::npos ? std::string::npos : x - at);
                if (tok.empty()) {
                    std::fprintf(stderr, "onebit-ort-run: --input %s: empty dimension (%s)\n", in.name.c_str(), spec.c_str());
                    return false;
                }
                in.shape.push_back(std::atoll(tok.c_str()));
                if (x == std::string::npos) break;
                at = x + 1;
            }
            a->inputs.push_back(std::move(in));
        }
        else if (arg == "--output" && more) {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
                std::fprintf(stderr, "onebit-ort-run: --output wants NAME=FILE (%s)\n", spec.c_str());
                return false;
            }
            a->outputs.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
        }
        else if (arg == "--dump-all" && more) a->dump_all = argv[++i];
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
    if (a->lens.empty()) a->lens = {2048, 1024, 512, 256, 128};  // AMD's default sweep
    if (!a->model || (a->ids.empty() && a->inputs.empty() && !a->bench)) {
        std::fprintf(stderr,
                     "usage: ort-run --model <file.onnx> [--ep <lib.so>]... --ids \"<id> <id>\"\n"
                     "               [-n N] [--dump-logits DIR] [--no-mask] [--threads N] [--cached]\n"
                     "               [--basic]   (keep a QDQ graph's DQ/Q nodes: no graph-transformation\n"
                     "                            folding, which is what a quantised EP claims)\n"
                     "       ort-run --model <cached.onnx> --cached --bench [--lens \"2048,1024,512,256,128\"]\n"
                     "               [--gen 128] [--warmup 1] [--repeat 5]\n"
                     "       --bench prints AMD's model_benchmark.exe block (avg/p50/stddev per stage) so\n"
                     "       the numbers sit next to their Ryzen AI runs; it needs a --cached graph.\n"
                     "       --ep may be repeated: the EPs are registered in the order given and the\n"
                     "       CPU EP is always appended last (docs/onnx.md, \"Two engines\").\n"
                     "       ort-run --model <any.onnx> [--ep <lib.so>] --input NAME=FILE:DTYPE:SHAPE ...\n"
                     "               [--output NAME=FILE]... [--dump-all DIR]\n"
                     "       Any graph: each --input is a raw little-endian tensor (float32, float16,\n"
                     "       uint8, int8, int32, int64), the declared shape must match the file's size\n"
                     "       or the run is refused by name, and outputs go where --output says them or\n"
                     "       into --dump-all with an outputs.txt sidecar (docs/onnx.md, \"Any graph\").\n");
        return false;
    }
    return true;
}

// A dumped logits row is model output, so it is not created world-readable (tools/moe_trace.cpp
// uses the same shape).
int dump_row(const float* p, size_t count, int64_t vocab, const char* dir, int step) {
    if (!dir || count < (size_t)vocab) return 0;
    ::mkdir(dir, 0700);  // EEXIST is the normal case; a missing dir would silently drop every row
    char path[512];
    std::snprintf(path, sizeof(path), "%s/logits_%03d.bin", dir, step);
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    FILE* f = fd >= 0 ? ::fdopen(fd, "wb") : nullptr;
    if (!f) return 0;
    std::fwrite(p + (count - (size_t)vocab), sizeof(float), (size_t)vocab, f);
    std::fclose(f);
    return 0;
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

// Cached decode: the graph holds the KV cache, so the host stops feeding the prefix. Interface is
// fixed to what onnx/export_qwen3.py --cached emits: `past` and `present` are each ONE stacked
// (2*n_layers, 1, kv_heads, past, head_dim) tensor, so the loop carries a single OrtValue.
int run_cached(OrtSession* session, OrtMemoryInfo* mi, const Args& a, RunStats* stats = nullptr) {
    const char* in_n[] = {"input_ids", "past"};  // no mask/positions: the graph derives both
    const char* out_n[] = {"logits", "present"};

    OrtTypeInfo* ti = nullptr;
    if (OrtStatus* st = api->SessionGetInputTypeInfo(session, 1, &ti)) return fail("past type", st);
    const OrtTensorTypeAndShapeInfo* si = nullptr;
    if (OrtStatus* st = api->CastTypeInfoToTensorInfo(ti, &si)) api->ReleaseStatus(st);
    int64_t stacked[5] = {0, 0, 0, 0, 0};
    size_t rank = 0;
    if (si && !api->GetDimensionsCount(si, &rank) && rank == 5) api->GetDimensions(si, stacked, 5);
    api->ReleaseTypeInfo(ti);
    if (rank != 5) {
        std::fprintf(stderr, "onebit-ort-run: FAIL `past` is not 5-D: not a cached graph\n");
        return 2;
    }
    const int64_t n_kv = stacked[2], head_dim = stacked[4];

    // OWN THE CACHE. ORT allocates a fresh `present` every step and its arena keeps the old ones:
    // a 2048-token prompt peaked at 6.5 GB RSS and 164 ms/token against 59 ms at 128, with 2.5M
    // minor faults doing the copying. Two preallocated stacks, bound as `past` and `present` and
    // swapped each step, leave the graph's own Concat as the only traffic and cap the footprint at
    // exactly two caches -- which is the difference between decode decaying with context and not.
    const int64_t stack = stacked[0];                   // 2 * n_layers
    const int64_t per_token = stack * n_kv * head_dim;   // float per cache position
    const int64_t cap = (int64_t)a.ids.size() + (int64_t)a.steps + 1;
    std::vector<float> arena((size_t)(2 * cap * per_token), 0.0f);
    float* buf[2] = {arena.data(), arena.data() + (size_t)(cap * per_token)};
    int src = 0;  // which buffer currently holds `past`
    int64_t past_len = 0;
    float dummy = 0.f;  // a zero-length cache has no bytes to point at
    OrtIoBinding* bind = nullptr;
    OrtAllocator* alloc = nullptr;
    if (OrtStatus* st = api->CreateIoBinding(session, &bind)) return fail("io binding", st);
    if (OrtStatus* st = api->GetAllocatorWithDefaultOptions(&alloc)) return fail("allocator", st);

    std::vector<int64_t> step_ids = a.ids;
    std::vector<int64_t> gen;
    std::vector<int64_t> shape;
    int64_t vocab = 0;
    double prefill_ms = 0.0, decode_ms = 0.0;
    int rc = 0;

    for (int step = -1; step < a.steps && !rc; ++step) {
        const int64_t seq = (int64_t)step_ids.size();
        const int64_t dims_ids[2] = {1, seq};

        OrtValue *v_ids = nullptr, *v_past = nullptr, *v_pres = nullptr;
        const int64_t dims_past[5] = {stack, 1, n_kv, past_len, head_dim};
        const int64_t dims_pres[5] = {stack, 1, n_kv, past_len + seq, head_dim};
        const size_t bytes_past = (size_t)(past_len * per_token) * sizeof(float);
        const size_t bytes_pres = (size_t)((past_len + seq) * per_token) * sizeof(float);
        OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
            mi, past_len ? (void*)buf[src] : (void*)&dummy, bytes_past, dims_past, 5,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_past);
        if (!st)
            st = api->CreateTensorWithDataAsOrtValue(mi, buf[1 - src], bytes_pres, dims_pres, 5,
                                                     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_pres);
        if (!st)
            st = api->CreateTensorWithDataAsOrtValue(
                mi, step_ids.data(), sizeof(int64_t) * step_ids.size(), dims_ids, 2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v_ids);
        if (!st) st = api->BindInput(bind, in_n[0], v_ids);
        if (!st) st = api->BindInput(bind, in_n[1], v_past);
        // `logits` is bound to the device, not to memory: it is read straight after the run and
        // preallocating it would mean 1.2 GB at a 2048-token prompt. Bound outputs arrive in bind
        // order, so outs[0] stays logits and outs[1] stays our `present`.
        if (!st) st = api->BindOutputToDevice(bind, out_n[0], mi);
        if (!st) st = api->BindOutput(bind, out_n[1], v_pres);
        OrtValue** outs = nullptr;
        size_t n_out = 0;
        double dt = 0.0;
        if (!st) {
            const auto s0 = std::chrono::steady_clock::now();
            st = api->RunWithBinding(session, nullptr, bind);
            const auto s1 = std::chrono::steady_clock::now();
            dt = std::chrono::duration<double, std::milli>(s1 - s0).count();
            if (!st) st = api->GetBoundOutputValues(bind, alloc, &outs, &n_out);
        }
        if (!st) {            if (!st) {
                if (step < 0) prefill_ms = dt;
                else {
                    decode_ms += dt;
                    if (stats) stats->token_ms.push_back(dt);
                }
                OrtTensorTypeAndShapeInfo* info = nullptr;
                if (OrtStatus* s = api->GetTensorTypeAndShape(outs[0], &info)) {
                    rc = fail("logits shape", s);
                } else {
                    size_t lrank = 0;
                    api->GetDimensionsCount(info, &lrank);
                    shape.assign(lrank, 0);
                    api->GetDimensions(info, shape.data(), lrank);
                    api->ReleaseTensorTypeAndShapeInfo(info);
                    vocab = shape.empty() ? 0 : shape.back();
                    size_t count = 1;
                    for (int64_t d : shape) count *= (size_t)d;
                    const float* p = nullptr;
                    if (OrtStatus* s = api->GetTensorData(outs[0], (const void**)&p)) {
                        rc = fail("read logits", s);
                    } else {
                        dump_row(p, count, vocab, a.dump, step);
                        // The last row of whatever the graph returned: a graph that slices its own
                        // output to the final position returns one row, a full-logits graph returns
                        // `seq`. Indexing by seq-1 read past the end of the sliced one.
                        const float* row = p + (size_t)count - (size_t)vocab;
                        const auto q0 = std::chrono::steady_clock::now();
                        int64_t best = 0;
                        for (int64_t id = 1; id < vocab; ++id)
                            if (row[id] > row[best]) best = id;
                        if (stats && step >= 0)
                            stats->sample_ms.push_back(
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - q0).count());
                        gen.push_back(best);
                        step_ids.assign(1, best);
                    }
                }
            }
        }
        for (size_t i = 0; i < n_out; ++i) api->ReleaseValue(outs[i]);
        if (outs) alloc->Free(alloc, outs);
        api->ReleaseValue(v_past);
        api->ReleaseValue(v_pres);
        api->ReleaseValue(v_ids);
        if (st) {
            rc = fail("Run", st);
        } else {
            src = 1 - src;  // the buffer just written becomes the next step's `past`
            past_len += seq;
        }
    }

    api->ReleaseIoBinding(bind);
    if (rc) return rc;

    if (stats) {  // bench mode: hand the samples back and print nothing
        stats->ttft_ms = prefill_ms;
        stats->e2e_ms = prefill_ms + decode_ms;
        return 0;
    }

    // gen[0] is the prefill's own argmax; the steps after it are the decode steps we time.
    const size_t decoded = gen.empty() ? 0 : gen.size() - 1;
    std::printf("  prefill : %zu tokens in %.1f ms (logits [%lld])\n", a.ids.size(), prefill_ms,
                (long long)vocab);
    std::printf("  kv      : %lld layers x %lld heads x %lld (%.1f KB/token)\n",
                (long long)(stacked[0] / 2), (long long)n_kv, (long long)head_dim,
                (double)stacked[0] * (double)n_kv * (double)head_dim * sizeof(float) / 1024.0);
    std::printf("  decode  : %zu tokens in %.1f ms (%.2f tok/s, %.2f ms/token)\n", decoded, decode_ms,
                decode_ms > 0.0 ? 1000.0 * (double)decoded / decode_ms : 0.0,
                decoded ? decode_ms / (double)decoded : 0.0);
    std::printf("  ids     :");
    for (size_t i = 0; i + 1 < gen.size(); ++i) std::printf(" %lld", (long long)gen[i]);
    std::printf("\n");
    return 0;
}

}  // namespace

long peak_rss_bytes() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = std::atol(line + 6);
            break;
        }
    std::fclose(f);
    return kb * 1024L;
}

// Element type of a declared dtype, or -1 for one this runner does not carry. Kept deliberately
// small: the tensors vision models take are float32 and uint8, and a dtype we cannot name is
// refused by name rather than coerced.
int elem_type(const std::string& d) {
    if (d == "float32" || d == "f32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (d == "float16" || d == "f16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (d == "uint8" || d == "u8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    if (d == "int8" || d == "i8") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    if (d == "int32" || d == "i32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    if (d == "int64" || d == "i64") return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return -1;
}

size_t elem_size(int t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        default: return 0;
    }
}

const char* type_name(int t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        default: return "other";
    }
}

// A filename that cannot escape the output directory: ONNX names carry `::`, `/` and `:`.
std::string safe_name(const std::string& n) {
    std::string out;
    for (char c : n) out += (std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
    return out.empty() ? std::string("tensor") : out;
}

// One run of an arbitrary graph: every declared input from a file, one Run(), every output either
// to the path named for it or into --dump-all's directory with a sidecar that says what it is. This
// is the whole of "the runner speaks any graph's interface" -- no inference about names, shapes or
// roles, so a vision model, an audio model and a matmul toy all take the same path.
int run_generic(OrtSession* session, OrtMemoryInfo* mi, const Args& a) {
    std::vector<std::vector<uint8_t>> storage;   // outlives every tensor made from it
    std::vector<OrtValue*> values;
    std::vector<const char*> names;
    storage.resize(a.inputs.size());
    int rc = 0;
    for (size_t i = 0; i < a.inputs.size() && !rc; ++i) {
        const InSpec& in = a.inputs[i];
        const int et = elem_type(in.dtype);
        if (et < 0) {
            std::fprintf(stderr, "onebit-ort-run: --input %s: dtype %s is not one this runner carries\n",
                         in.name.c_str(), in.dtype.c_str());
            rc = 2;
            break;
        }
        size_t elems = 1;
        for (int64_t d : in.shape) {
            if (d <= 0) {
                std::fprintf(stderr, "onebit-ort-run: --input %s: dimension %lld is not positive (a static shape is required)\n",
                             in.name.c_str(), (long long)d);
                rc = 2;
                break;
            }
            elems *= (size_t)d;
        }
        if (rc) break;
        const size_t want = elems * elem_size(et);
        FILE* f = std::fopen(in.file.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "onebit-ort-run: --input %s: cannot read %s\n", in.name.c_str(), in.file.c_str());
            rc = 2;
            break;
        }
        std::fseek(f, 0, SEEK_END);
        const long have = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (have != (long)want) {
            std::fprintf(stderr,
                         "onebit-ort-run: --input %s: %s holds %ld B, %s of %lld elements needs %zu B\n",
                         in.name.c_str(), in.file.c_str(), have, in.dtype.c_str(), (long long)elems, want);
            std::fclose(f);
            rc = 2;
            break;
        }
        storage[i].assign(want, 0);
        if (want && std::fread(storage[i].data(), 1, want, f) != want) {
            std::fprintf(stderr, "onebit-ort-run: --input %s: short read from %s\n", in.name.c_str(), in.file.c_str());
            std::fclose(f);
            rc = 2;
            break;
        }
        std::fclose(f);
        OrtValue* v = nullptr;
        if (OrtStatus* st = api->CreateTensorWithDataAsOrtValue(mi, storage[i].data(), want, in.shape.data(),
                                                              in.shape.size(), (ONNXTensorElementDataType)et, &v)) {
            rc = fail("input tensor", st);
            break;
        }
        values.push_back(v);
        names.push_back(in.name.c_str());
        std::printf("  in      : %s %s %zu B", in.name.c_str(), in.dtype.c_str(), want);
        for (int64_t d : in.shape) std::printf("x%lld", (long long)d);
        std::printf("  <- %s\n", in.file.c_str());
    }
    if (rc) {
        for (OrtValue* v : values) api->ReleaseValue(v);
        return rc;
    }

    // Every output, named. Passing a null name array here is not "all outputs" in this API -- it
    // aborted inside ORT (a terminate with no active exception, and the buffered log went with it).
    std::vector<std::string> out_owned;
    std::vector<const char*> out_names;
    std::vector<OrtValue*> outs;
    {
        OrtAllocator* alloc = nullptr;
        api->GetAllocatorWithDefaultOptions(&alloc);
        size_t n_out = 0;
        if (OrtStatus* st = api->SessionGetOutputCount(session, &n_out)) {
            rc = fail("output count", st);
        } else {
            for (size_t i = 0; i < n_out && !rc; ++i) {
                char* nm = nullptr;
                if (OrtStatus* st = api->SessionGetOutputName(session, i, alloc, &nm)) {
                    rc = fail("output name", st);
                } else {
                    out_owned.emplace_back(nm);
                    alloc->Free(alloc, nm);
                }
            }
        }
        if (!rc && n_out) {
            for (const auto& nm : out_owned) out_names.push_back(nm.c_str());
            outs.assign(n_out, nullptr);
            const auto t0 = std::chrono::steady_clock::now();
            if (OrtStatus* st = api->Run(session, nullptr, names.data(), values.data(), values.size(),
                                         out_names.data(), out_names.size(), outs.data()))
                rc = fail("Run", st);
            else
                std::printf("  run     : %.2f ms, %zu output(s)\n",
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                            outs.size());
        }
    }
    for (OrtValue* v : values) api->ReleaseValue(v);
    if (rc) {
        for (OrtValue* v : outs) if (v) api->ReleaseValue(v);
        return rc;
    }

    if (a.dump_all) ::mkdir(a.dump_all, 0700);   // EEXIST is the normal case
    FILE* side = a.dump_all ? std::fopen((std::string(a.dump_all) + "/outputs.txt").c_str(), "a") : nullptr;
    for (size_t i = 0; i < outs.size(); ++i) {
        OrtAllocator* alloc = nullptr;
        char* gname = nullptr;
        api->GetAllocatorWithDefaultOptions(&alloc);
        if (OrtStatus* st = api->SessionGetOutputName(session, i, alloc, &gname)) {
            rc = fail("output name", st);
            break;
        }
        OrtTensorTypeAndShapeInfo* info = nullptr;
        size_t rank = 0;
        std::vector<int64_t> dims;
        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        if (OrtStatus* st = api->GetTensorTypeAndShape(outs[i], &info)) {
            rc = fail("output type", st);
            break;
        }
        api->GetDimensionsCount(info, &rank);
        dims.assign(rank, 0);
        api->GetDimensions(info, dims.data(), rank);
        api->GetTensorElementType(info, &et);
        api->ReleaseTensorTypeAndShapeInfo(info);

        // The write path: an explicit --output, else a file in --dump-all's directory.
        std::string path;
        for (const auto& o : a.outputs) if (o.first == gname) path = o.second;
        if (path.empty() && !a.dump_all) continue;
        if (path.empty()) path = std::string(a.dump_all) + "/" + safe_name(gname) + ".bin";
        const void* data = nullptr;
        if (OrtStatus* st = api->GetTensorData(outs[i], &data)) {
            rc = fail("output data", st);
            break;
        }
        size_t elems = 1;
        for (int64_t d : dims) elems *= (size_t)(d > 0 ? d : 1);
        const size_t bytes = elems * elem_size(et);
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "onebit-ort-run: cannot write %s\n", path.c_str());
            rc = 2;
            break;
        }
        if (bytes) std::fwrite(data, 1, bytes, f);
        std::fclose(f);
        std::printf("  out     : %s %s %zu B", gname, type_name(et), bytes);
        for (int64_t d : dims) std::printf("x%lld", (long long)d);
        std::printf("  -> %s\n", path.c_str());
        if (side) {
            std::fprintf(side, "%s %s", gname, type_name(et));
            for (int64_t d : dims) std::fprintf(side, "x%lld", (long long)d);
            std::fprintf(side, " %zu %s\n", bytes, path.c_str());
        }
        if (alloc && gname) alloc->Free(alloc, gname);
    }
    if (side) std::fclose(side);
    for (OrtValue* v : outs) if (v) api->ReleaseValue(v);
    return rc;
}

// AMD's model_benchmark.exe block: same stages, same statistics, same defaults (-l/-g/-w/-r), so the
// two outputs are read side by side without translation. Each repeat restarts from a zero-length
// cache, which is their repeat loop resetting the OGA generator.
int bench_amd(OrtSession* session, OrtMemoryInfo* mi, const Args& a) {
    std::printf("-----------------------------\n");
    for (size_t li = 0; li < a.lens.size(); ++li) {
        const int64_t L = a.lens[li];
        Args one = a;
        one.steps = a.gen > 0 ? a.gen : 1;
        one.dump = nullptr;
        one.ids.assign((size_t)L, 0);
        // Deterministic filler that exists in every model of AMD's set (Phi-3's vocab is 32064).
        // They feed prompts.txt; for timing only the token count matters.
        for (int64_t i = 0; i < L; ++i) one.ids[(size_t)i] = 100 + (i * 7919) % 30000;

        std::vector<double> ttft_us, tok_us, sample_us, e2e_ms;
        for (int i = 0; i < a.warmup + a.repeat; ++i) {
            RunStats s;
            if (const int rc = run_cached(session, mi, one, &s)) return rc;
            if (i < a.warmup) continue;
            ttft_us.push_back(s.ttft_ms * 1000.0);
            e2e_ms.push_back(s.e2e_ms);
            for (double ms : s.token_ms) tok_us.push_back(ms * 1000.0);
            for (double ms : s.sample_ms) sample_us.push_back(ms * 1000.0);
        }

        const Stat t = stat_of(ttft_us), g = stat_of(tok_us), q = stat_of(sample_us);
        const Stat e = stat_of(e2e_ms);
        std::printf("Prompt\n  Number of Tokens: %lld\n", (long long)L);
        std::printf("Batch size: 1, prompt tokens: %lld, tokens to generate: %d\n", (long long)L,
                    one.steps);
        std::printf("  Prompt processing (time to first token):\n");
        std::printf("      avg (us): %.0f  avg (tokens/s): %.3f  p50 (us): %.0f  stddev (us): %.0f"
                    "  n: %zu * %lld token(s)\n",
                    t.avg, t.avg > 0.0 ? 1e6 * (double)L / t.avg : 0.0, t.p50, t.sd, t.n,
                    (long long)L);
        std::printf("  Token generation:\n");
        std::printf("      avg (us): %.1f  avg (tokens/s): %.4f  p50 (us): %.1f  stddev (us): %.1f"
                    "  n: %zu * 1 token(s)\n",
                    g.avg, g.avg > 0.0 ? 1e6 / g.avg : 0.0, g.p50, g.sd, g.n);
        std::printf("  Token sampling:\n");
        std::printf("      avg (us): %.3f  avg (tokens/s): %.2f  p50 (us): %.3f  stddev (us): %.3f"
                    "  n: %zu * 1 token(s)\n",
                    q.avg, q.avg > 0.0 ? 1e6 / q.avg : 0.0, q.p50, q.sd, q.n);
        std::printf("  E2E generation (entire generation loop):\n");
        std::printf("      avg (ms): %.2f  p50 (ms): %.2f  stddev (ms): %.2f  n: %zu\n", e.avg,
                    e.p50, e.sd, e.n);
        std::printf("-----------------------------\n");
    }
    std::printf("Peak working set size (bytes): %ld\n", peak_rss_bytes());
    return 0;
}

int onebit_ort_run_main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // an abort must not swallow the trace that explains it
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
    if (OrtStatus* st = make_session(env, a.model, a.eps, a.threads, a.basic, a.profile, &session)) {
        const std::string why = api->GetErrorMessage(st);
        int rc = fail("create session", st);
        explain_missing_kernel(why.c_str(), a.model);
        api->ReleaseMemoryInfo(mi);
        api->ReleaseEnv(env);
        return rc;
    }

    std::printf("onebit ort-run\n  ort     : %s (api v%d)\n  model   : %s\n", base->GetVersionString(),
                ORT_API_VERSION, a.model);
    for (size_t i = 0; i < a.eps.size(); ++i)
        std::printf("  ep%-6zu: %s\n", i + 1, a.eps[i]);
    if (a.eps.empty()) std::printf("  ep      : (CPU only)\n");

    // Any graph with declared inputs takes the generic path: nothing below this line assumes the
    // Qwen3 interface, and nothing above it assumed the caller had one.
    if (!a.inputs.empty()) {
        const int rc = run_generic(session, mi, a);
        api->ReleaseSession(session);
        api->ReleaseMemoryInfo(mi);
        api->ReleaseEnv(env);
        return rc;
    }

    if (a.bench) {
        if (!a.cached) {
            std::fprintf(stderr,
                         "onebit-ort-run: --bench needs a --cached graph (export_qwen3.py --cached)\n");
            api->ReleaseSession(session);
            api->ReleaseMemoryInfo(mi);
            api->ReleaseEnv(env);
            return 2;
        }
        const int rc = bench_amd(session, mi, a);
        api->ReleaseSession(session);
        api->ReleaseMemoryInfo(mi);
        api->ReleaseEnv(env);
        return rc;
    }

    if (a.cached) {
        const int rc = run_cached(session, mi, a);
        api->ReleaseSession(session);
        api->ReleaseMemoryInfo(mi);
        api->ReleaseEnv(env);
        return rc;
    }

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
        const float* p = nullptr;
        if (count < (size_t)vocab || api->GetTensorData(logits, (const void**)&p)) return;
        dump_row(p, count, vocab, a.dump, step);
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
