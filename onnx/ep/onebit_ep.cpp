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

// libonebit_ep.so -- our ONNX Runtime plugin execution provider.
//
// ONNX Runtime 1.23.2 is linked by the host, never by us; this library only implements the
// OrtEpFactory / OrtEp / OrtNodeComputeInfo structs from onnxruntime_ep_c_api.h. ORT dlopens it
// through RegisterExecutionProviderLibrary() and calls CreateEpFactories().
//
// Delivered in two steps on purpose. Here (deliverable 2) the claimed nodes are executed on the
// host by onebit_ep::gemm_f32, so every claim the EP makes is a *correctness* claim: if the
// partitioning is wrong, or the allocator hands back memory ORT cannot use, or the data transfer
// is a lie, the numbers move and the proof binary says so. Deliverable 3 swaps that kernel for the
// NPU without touching anything else in this file.
//
// Contract rules this file exists to obey (from the OrtEpApi headers, each one a crash if broken):
//   * the allocator must report OrtDeviceAllocator -- OrtArenaAllocator is refused by name;
//   * its allocations must be 256-byte aligned -- ORT's kernels use aligned vector loads;
//   * CreateDataTransfer must return a non-null implementation -- ORT calls it unconditionally;
//   * claims go through EpGraphSupportInfo_AddNodesToFuse, never AddSingleNode (a plugin EP has no
//     kernel registry, so AddSingleNode fails at session init with "Failed to find kernel");
//   * an appended EP's OrtEpDevice handles die with UnregisterExecutionProviderLibrary.

// onnxruntime_c_api.h already includes onnxruntime_ep_c_api.h, and that header has no
// include guard (it is meant to be included exactly once).
#include <onnxruntime_c_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "onebit_ep_kernel.h"
#ifdef ONEBIT_EP_NPU
#include "../../npu/ort/onebit_npu_gemm.h"  // has_kernel(): which (K, N) shapes this build can run
#endif

// ORT_EXPORT is empty on Linux (the header only defines it for macOS), so with -fvisibility=hidden
// the two entry points ORT dlopens would not be exported at all. Say it explicitly.
#if defined(_WIN32)
#define ONEBIT_EP_API __declspec(dllexport)
#elif defined(__GNUC__)
#define ONEBIT_EP_API __attribute__((visibility("default")))
#else
#define ONEBIT_EP_API
#endif

namespace {

const OrtApi* g_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;

constexpr const char* kEpName = "OneBitEp";
constexpr const char* kEpVendor = "1bit";
constexpr const char* kEpVersion = "0.1.0";
// '1','b' -- nothing else claims it, and a real vendor id is not ours to invent.
constexpr uint32_t kEpVendorId = 0x3162u;

// ONEBIT_EP_TRACE=1 makes the EP append one line per executed node to kTracePrefix<pid>.log. The
// proof binary reads that file back: it, not a log line, is the evidence that ORT handed us the
// node. The path is fixed (no environment-supplied path, no world-writable mode) because a plugin
// that opens whatever path an env var names is a path-injection hole in a .so other processes load.
constexpr const char* kTracePrefix = "/tmp/onebit_ep_trace.";
FILE* g_trace = nullptr;

std::string trace_path() { return std::string(kTracePrefix) + std::to_string(::getpid()) + ".log"; }

void tracef(const char* fmt, ...) {
    if (!g_trace) {
        const char* on = std::getenv("ONEBIT_EP_TRACE");
        if (!on || !*on || *on == '0') return;
        const int fd = ::open(trace_path().c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd < 0) return;
        g_trace = ::fdopen(fd, "a");
        if (!g_trace) {
            ::close(fd);
            return;
        }
        std::setvbuf(g_trace, nullptr, _IOLBF, 0);
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_trace, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_trace);
    std::fflush(g_trace);
}

bool verbose() {
    const char* v = std::getenv("ONEBIT_EP_VERBOSE");
    return v && *v && *v != '0';
}

// Ops we claim, comma separated. MatMul is the whole of deliverable 2; the rest of the LLM set is
// added when it is actually implemented, not before.
bool claims_op(const char* op) {
    if (!op) return false;
    const char* list = std::getenv("ONEBIT_EP_OPS");
    if (!list || !*list) list = "MatMul";
    size_t left = std::strlen(op);
    for (const char* p = list; *p;) {
        const char* comma = std::strchr(p, ',');
        size_t n = comma ? (size_t)(comma - p) : std::strlen(p);
        if (n == left && std::strncmp(p, op, n) == 0) return true;
        if (!comma) break;
        p = comma + 1;
    }
    return false;
}

// ---------------------------------------------------------------------------- allocator (G-ON7)

// One allocator per OrtMemoryInfo the host asks for: OrtAllocator::Info must describe the device
// the memory really is on, so a single shared instance would lie as soon as there are two.
struct EpAllocator {
    OrtAllocator self;
    const OrtMemoryInfo* mi;
    size_t allocs;
    size_t frees;
    size_t bytes;
};

constexpr size_t kMaxAllocators = 8;
EpAllocator g_allocators[kMaxAllocators];
size_t g_num_allocators = 0;

void* ORT_API_CALL A_Alloc(OrtAllocator* self, size_t size) noexcept {
    // 256 bytes, not malloc's 16: ORT's kernels (MLAS) use aligned vector loads on tensor memory and
    // fault on anything less. This is the single most common way a plugin EP dies.
    size_t rounded = ((size ? size : 1) + 255) & ~(size_t)255;
    void* p = nullptr;
    if (posix_memalign(&p, 256, rounded) != 0) p = nullptr;
    if (auto* a = reinterpret_cast<EpAllocator*>(self); a && p) {
        ++a->allocs;
        a->bytes += size;
    }
    return p;
}

void ORT_API_CALL A_Free(OrtAllocator* self, void* p) noexcept {
    if (auto* a = reinterpret_cast<EpAllocator*>(self); a && p) ++a->frees;
    std::free(p);
}

const OrtMemoryInfo* ORT_API_CALL A_Info(const OrtAllocator* self) noexcept {
    return reinterpret_cast<const EpAllocator*>(self)->mi;
}

OrtAllocator* new_allocator(const OrtMemoryInfo* mi) {
    if (g_num_allocators >= kMaxAllocators) return nullptr;
    EpAllocator* a = &g_allocators[g_num_allocators++];
    std::memset(a, 0, sizeof(*a));
    a->mi = mi;
    a->self.version = ORT_API_VERSION;
    a->self.Alloc = A_Alloc;
    a->self.Free = A_Free;
    a->self.Info = A_Info;
    return &a->self;
}

// ------------------------------------------------------------------------- data transfer (must exist)

// Every device this EP registers is host memory, so a copy is a memcpy. It is real code rather than
// a "cannot copy" stub because deliverable 3's device memory makes this the CPU<->NPU bridge, and
// because CanCopy answering false for CPU->CPU makes ORT create its own copies behind our back.
bool ORT_API_CALL DT_CanCopy(const OrtDataTransferImpl*, const OrtMemoryDevice* src,
                             const OrtMemoryDevice* dst) noexcept {
    if (!src || !dst) return false;
    return g_ep_api->MemoryDevice_GetDeviceType(src) == OrtMemoryInfoDeviceType_CPU &&
           g_ep_api->MemoryDevice_GetDeviceType(dst) == OrtMemoryInfoDeviceType_CPU;
}

OrtStatus* ORT_API_CALL DT_CopyTensors(OrtDataTransferImpl*, const OrtValue** src_tensors,
                                       OrtValue** dst_tensors, OrtSyncStream**, size_t num_tensors) noexcept {
    for (size_t i = 0; i < num_tensors; ++i) {
        size_t bytes = 0;
        if (OrtStatus* st = g_api->GetTensorSizeInBytes(src_tensors[i], &bytes)) return st;
        void* src = nullptr;
        void* dst = nullptr;
        if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(src_tensors[i]), &src))
            return st;
        if (OrtStatus* st = g_api->GetTensorMutableData(dst_tensors[i], &dst)) return st;
        std::memcpy(dst, src, bytes);
        tracef("copy bytes=%zu", bytes);
    }
    return nullptr;
}

void ORT_API_CALL DT_Release(OrtDataTransferImpl*) noexcept {}

const OrtDataTransferImpl g_transfer = {
    /* ort_version_supported */ ORT_API_VERSION,
    /* Release */ DT_Release,
    /* CanCopy */ DT_CanCopy,
    /* CopyTensors */ DT_CopyTensors,
};

// ------------------------------------------------------------------------------ node compute info

OrtStatus* ORT_API_CALL C_CreateState(OrtNodeComputeInfo*, OrtNodeComputeContext*, void** state) noexcept {
    *state = nullptr;
    return nullptr;
}

void ORT_API_CALL C_ReleaseState(OrtNodeComputeInfo*, void*) noexcept {}

// Reads a rank-2/rank-3 float tensor's dims. Anything else is refused rather than guessed at: a
// wrong shape here corrupts the graph silently, which is worse than a session that fails to load.
OrtStatus* tensor_dims(const OrtValue* v, int64_t dims[3], size_t* rank) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* st = g_api->GetTensorTypeAndShape(v, &info)) return st;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    OrtStatus* st = g_api->GetTensorElementType(info, &type);
    if (!st && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        st = g_api->CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "onebit_ep: only float32 tensors are supported yet");
    }
    size_t ndim = 0;
    if (!st) st = g_api->GetDimensionsCount(info, &ndim);
    if (!st && (ndim < 2 || ndim > 3)) {
        st = g_api->CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "onebit_ep: only rank-2 and rank-3 MatMul are supported yet");
    }
    if (!st) st = g_api->GetDimensions(info, dims, ndim);
    if (!st) *rank = ndim;
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    return st;
}

OrtStatus* ORT_API_CALL C_Compute(OrtNodeComputeInfo*, void*, OrtKernelContext* ctx) noexcept {
    const OrtValue* av = nullptr;
    const OrtValue* bv = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, 0, &av)) return st;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, 1, &bv)) return st;

    int64_t ad[3] = {0, 0, 0}, bd[3] = {0, 0, 0};
    size_t arank = 0, brank = 0;
    if (OrtStatus* st = tensor_dims(av, ad, &arank)) return st;
    if (OrtStatus* st = tensor_dims(bv, bd, &brank)) return st;

    // A: [batch..., M, K], B: [batch..., K, N] with at most one of the two batched. The batch dims
    // of A and B must agree when both are present; B rank-2 is the broadcast case (one weight).
    const int64_t m = ad[arank - 2], k = ad[arank - 1];
    const int64_t b_k = bd[brank - 2], n = bd[brank - 1];
    if (k != b_k) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: MatMul inner dimensions disagree");
    }
    int64_t batch = 1;
    int64_t out_dims[3] = {0, 0, 0};
    size_t orank = 0;
    if (arank == 3) {
        batch = ad[0];
        out_dims[orank++] = ad[0];
    }
    if (brank == 3) {
        if (arank == 3 && bd[0] != ad[0]) {
            return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "onebit_ep: MatMul batch dimensions disagree");
        }
        if (arank == 2) {
            batch = bd[0];
            out_dims[orank++] = bd[0];
        }
    }
    out_dims[orank++] = m;
    out_dims[orank++] = n;

    OrtValue* out = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetOutput(ctx, 0, out_dims, orank, &out)) return st;

    const float* a = nullptr;
    const float* b = nullptr;
    float* c = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(av), (void**)&a)) return st;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(bv), (void**)&b)) return st;
    if (OrtStatus* st = g_api->GetTensorMutableData(out, (void**)&c)) return st;

    const int64_t a_batch_stride = (arank == 3) ? m * k : 0;
    const int64_t b_batch_stride = (brank == 3) ? k * n : 0;
    const int64_t c_batch_stride = m * n;
    bool on_npu = true;
    for (int64_t bi = 0; bi < batch && on_npu; ++bi)
        on_npu = onebit_ep::gemm_f32(a + bi * a_batch_stride, b + bi * b_batch_stride,
                                     c + bi * c_batch_stride, m, k, n);
    // A slice that could not run on the device leaves its C untouched, so redo all of them.
    // gemm_f32 falls back to the host loop itself once the device refuses.
    if (!on_npu) {
        for (int64_t bi = 0; bi < batch; ++bi)
            (void)onebit_ep::gemm_f32(a + bi * a_batch_stride, b + bi * b_batch_stride,
                                      c + bi * c_batch_stride, m, k, n);
    }

    tracef("matmul batch=%lld m=%lld k=%lld n=%lld npu=%d", (long long)batch, (long long)m,
           (long long)k, (long long)n, on_npu ? 1 : 0);
    if (verbose()) {
        std::fprintf(stderr, "onebit_ep: matmul %lldx%lldx%lld -> %p\n", (long long)m, (long long)k,
                     (long long)n, (void*)c);
    }
    return nullptr;
}

const OrtNodeComputeInfo g_nci = {
    /* ort_version_supported */ ORT_API_VERSION,
    /* CreateState */ C_CreateState,
    /* Compute */ C_Compute,
    /* ReleaseState */ C_ReleaseState,
};

// ----------------------------------------------------------------------------------------- the EP

const char* ORT_API_CALL E_GetName(const OrtEp*) noexcept { return kEpName; }

// The rank and element type of one node input, read from the graph's type information. Symbolic
// dimensions are fine: only the rank and the type decide what C_Compute can run.
bool input_rank_type(const OrtNode* node, size_t index, ONNXTensorElementDataType* elem,
                     size_t* rank) {
    size_t n = 0;
    if (OrtStatus* st = g_api->Node_GetNumInputs(node, &n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (index >= n) return false;
    std::vector<const OrtValueInfo*> inputs(n, nullptr);
    if (OrtStatus* st = g_api->Node_GetInputs(node, inputs.data(), n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    const OrtTypeInfo* ti = nullptr;
    if (OrtStatus* st = g_api->GetValueInfoTypeInfo(inputs[index], &ti)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = g_api->CastTypeInfoToTensorInfo(ti, &tsi)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!tsi) return false;
    OrtStatus* e_st = g_api->GetTensorElementType(tsi, elem);
    OrtStatus* r_st = g_api->GetDimensionsCount(tsi, rank);
    if (e_st) g_api->ReleaseStatus(e_st);
    if (r_st) g_api->ReleaseStatus(r_st);
    return e_st == nullptr && r_st == nullptr;
}

// The node input's dimensions, with -1 for every symbolic one.
bool input_dims(const OrtNode* node, size_t index, std::vector<int64_t>* dims) {
    size_t n = 0;
    if (OrtStatus* st = g_api->Node_GetNumInputs(node, &n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (index >= n) return false;
    std::vector<const OrtValueInfo*> inputs(n, nullptr);
    if (OrtStatus* st = g_api->Node_GetInputs(node, inputs.data(), n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    const OrtTypeInfo* ti = nullptr;
    if (OrtStatus* st = g_api->GetValueInfoTypeInfo(inputs[index], &ti)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = g_api->CastTypeInfoToTensorInfo(ti, &tsi)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!tsi) return false;
    size_t rank = 0;
    if (OrtStatus* st = g_api->GetDimensionsCount(tsi, &rank)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    std::vector<int64_t> raw(rank, -1);
    OrtStatus* st = g_api->GetDimensions(tsi, raw.data(), rank);
    if (st) {
        g_api->ReleaseStatus(st);
        return false;
    }
    *dims = std::move(raw);
    return true;
}

// MatMul's K and N when both are static (A is [..., M, K], B is [K, N] or [..., K, N]).
bool static_kn(const OrtNode* node, int64_t* k, int64_t* n) {
    std::vector<int64_t> a, b;
    if (!input_dims(node, 0, &a) || !input_dims(node, 1, &b)) return false;
    if (a.size() < 2 || b.size() < 2) return false;
    const int64_t ak = a.back(), bn = b.back();
    if (ak <= 0 || bn <= 0) return false;
    *k = ak;
    *n = bn;
    return true;
}

// Claim a node only if it is one we can actually compute. Claiming by op name alone is the classic
// plugin-EP footgun: a rank-4 attention MatMul (or a float16 one) would be routed here and then fail
// at compute time, so a graph that used to fall back to the CPU EP now refuses to run.
bool can_compute(const OrtNode* node, const char* op) {
    if (!claims_op(op)) return false;
    if (std::strcmp(op, "MatMul") == 0) {
        for (size_t i = 0; i < 2; ++i) {
            ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t rank = 0;
            if (!input_rank_type(node, i, &elem, &rank)) return false;
            if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
            if (rank != 2 && rank != 3) return false;
        }
#ifdef ONEBIT_EP_NPU
        // With the device backend built in, take only the shapes we have an ELF for. The rest stay
        // with the next EP in the order (a GPU EP, then the CPU EP), which is much faster than our
        // host kernel and is the whole point of registering several engines in one session.
        int64_t k = 0, n = 0;
        if (!static_kn(node, &k, &n)) return false;
        if (!onebit::npu::has_kernel(k, n)) return false;
        // And only when the device is actually ours: if another process holds the lock (the owner's
        // serve, or a bench), claiming would run the shape on our host loop at a fraction of the
        // CPU EP's speed. Declining leaves the graph on the next EP in the order.
        std::string why;
        if (!onebit::npu::available(&why)) {
            tracef("npu_declined why=%s", why.c_str());
            return false;
        }
#endif
    }
    return true;
}

OrtStatus* ORT_API_CALL E_GetCapability(OrtEp*, const OrtGraph* graph,
                                        OrtEpGraphSupportInfo* support_info) noexcept {
    size_t n = 0;
    if (OrtStatus* st = g_api->Graph_GetNumNodes(graph, &n)) return st;
    if (n == 0) return nullptr;
    const OrtNode** nodes = (const OrtNode**)std::calloc(n, sizeof(OrtNode*));
    if (!nodes) return g_api->CreateStatus(ORT_FAIL, "onebit_ep: out of memory");
    OrtStatus* st = g_api->Graph_GetNodes(graph, nodes, n);
    size_t claimed = 0;
    for (size_t i = 0; !st && i < n; ++i) {
        const char* op = nullptr;
        const char* domain = nullptr;
        // Neither call can fail for a node the support query handed us, but the API marks them
        // warn_unused_result, so the status is released rather than dropped.
        if (OrtStatus* s = g_api->Node_GetOperatorType(nodes[i], &op)) g_api->ReleaseStatus(s);
        if (OrtStatus* s = g_api->Node_GetDomain(nodes[i], &domain)) g_api->ReleaseStatus(s);
        const bool onnx_domain = !domain || !*domain || std::strcmp(domain, "ai.onnx") == 0;
        if (onnx_domain && can_compute(nodes[i], op)) {
            st = g_ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, &nodes[i], 1, nullptr);
            ++claimed;
        }
    }
    std::free(nodes);
    tracef("capability nodes=%zu claimed=%zu", n, claimed);
    if (verbose()) std::fprintf(stderr, "onebit_ep: GetCapability %zu nodes -> %zu claimed\n", n, claimed);
    return st;
}

OrtStatus* ORT_API_CALL E_Compile(OrtEp*, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                  size_t count, OrtNodeComputeInfo** node_compute_infos,
                                  OrtNode** ep_context_nodes) noexcept {
    (void)graphs;
    // Compile is per claimed node and is where an EP would see the shapes to bake in. We keep the
    // shapes for the log only; the kernel reads them again at run time, because claiming a node does
    // not promise its input shapes stay fixed across sessions.
    for (size_t i = 0; i < count; ++i) {
        if (verbose() && fused_nodes) {
            const char* op = nullptr;
            if (!g_api->Node_GetOperatorType(fused_nodes[i], &op)) {
                std::fprintf(stderr, "onebit_ep: Compile fused[%zu] %s\n", i, op ? op : "?");
            }
        }
        tracef("compile node=%p", (const void*)fused_nodes[i]);
        node_compute_infos[i] = (OrtNodeComputeInfo*)&g_nci;
        if (ep_context_nodes) ep_context_nodes[i] = nullptr;
    }
    return nullptr;
}

void ORT_API_CALL E_ReleaseNodeComputeInfos(OrtEp*, OrtNodeComputeInfo** infos, size_t count) noexcept {
    (void)infos;
    (void)count;  // g_nci is static; nothing to free
}

OrtStatus* ORT_API_CALL E_GetPreferredDataLayout(OrtEp*, OrtEpDataLayout* out) noexcept {
    *out = OrtEpDataLayout_NCHW;
    return nullptr;
}

OrtStatus* ORT_API_CALL E_ShouldConvertDataLayoutForOp(OrtEp*, const char*, const char*,
                                                       OrtEpDataLayout, int* should_convert) noexcept {
    *should_convert = 0;
    return nullptr;
}

OrtStatus* ORT_API_CALL E_SetDynamicOptions(OrtEp*, const char* const*, const char* const*,
                                            size_t) noexcept {
    return nullptr;
}

OrtStatus* ORT_API_CALL E_OnRunStart(OrtEp*, const OrtRunOptions*) noexcept { return nullptr; }

OrtStatus* ORT_API_CALL E_OnRunEnd(OrtEp*, const OrtRunOptions*, bool) noexcept { return nullptr; }

OrtStatus* ORT_API_CALL E_CreateAllocator(OrtEp*, const OrtMemoryInfo* mi, OrtAllocator** out) noexcept {
    *out = new_allocator(mi);
    return *out ? nullptr : g_api->CreateStatus(ORT_FAIL, "onebit_ep: allocator table full");
}

OrtStatus* ORT_API_CALL E_CreateSyncStreamForDevice(OrtEp*, const OrtMemoryDevice*,
                                                    OrtSyncStreamImpl** out) noexcept {
    // Nothing here is asynchronous yet, so the honest answer is "no stream" rather than a stream
    // object that lies about ordering. Deliverable 3 needs a real one for the NPU queue.
    *out = nullptr;
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "onebit_ep: no sync streams yet");
}

const char* ORT_API_CALL E_GetCompiledModelCompatibilityInfo(OrtEp*, const OrtGraph*) noexcept {
    return nullptr;  // no .ort artefacts produced
}

const OrtEp g_ep = {
    /* ort_version_supported */ ORT_API_VERSION,
    /* GetName */ E_GetName,
    /* GetCapability */ E_GetCapability,
    /* Compile */ E_Compile,
    /* ReleaseNodeComputeInfos */ E_ReleaseNodeComputeInfos,
    /* GetPreferredDataLayout */ E_GetPreferredDataLayout,
    /* ShouldConvertDataLayoutForOp */ E_ShouldConvertDataLayoutForOp,
    /* SetDynamicOptions */ E_SetDynamicOptions,
    /* OnRunStart */ E_OnRunStart,
    /* OnRunEnd */ E_OnRunEnd,
    /* CreateAllocator */ E_CreateAllocator,
    /* CreateSyncStreamForDevice */ E_CreateSyncStreamForDevice,
    /* GetCompiledModelCompatibilityInfo */ E_GetCompiledModelCompatibilityInfo,
};

// ------------------------------------------------------------------------------------- the factory

const char* ORT_API_CALL F_GetName(const OrtEpFactory*) noexcept { return kEpName; }
const char* ORT_API_CALL F_GetVendor(const OrtEpFactory*) noexcept { return kEpVendor; }
const char* ORT_API_CALL F_GetVersion(const OrtEpFactory*) noexcept { return kEpVersion; }
uint32_t ORT_API_CALL F_GetVendorId(const OrtEpFactory*) noexcept { return kEpVendorId; }
bool ORT_API_CALL F_IsStreamAware(const OrtEpFactory*) noexcept { return false; }

OrtStatus* ORT_API_CALL F_GetSupportedDevices(OrtEpFactory* f,
                                              const OrtHardwareDevice* const* devices,
                                              size_t num_devices, OrtEpDevice** ep_devices,
                                              size_t max_ep_devices, size_t* num_ep_devices) noexcept {
    size_t n = 0;
    // Empty key/value pairs, never NULL: the host reads these back when it builds a session entry.
    OrtKeyValuePairs* md = nullptr;
    g_api->CreateKeyValuePairs(&md);  // returns void in 1.23
    if (!md) return g_api->CreateStatus(ORT_FAIL, "onebit_ep: no metadata pairs");
    for (size_t i = 0; i < num_devices && n < max_ep_devices; ++i) {
        OrtStatus* st = g_ep_api->CreateEpDevice(f, devices[i], md, md, &ep_devices[n]);
        if (st) {
            g_api->ReleaseKeyValuePairs(md);
            return st;
        }
        // Declare the memory this EP hands back. OrtDeviceAllocator, not OrtArenaAllocator: ORT
        // refuses an EP that declares an arena ("reserved for ONNX Runtime internal usage only").
        OrtMemoryInfo* mi = nullptr;
        if (OrtStatus* st2 = g_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi)) {
            g_api->ReleaseKeyValuePairs(md);
            return st2;
        }
        // Not released on purpose: the device keeps this OrtMemoryInfo (the host reads it back when
        // it builds a session entry, and OrtEp::CreateAllocator is handed the same pointer).
        st = g_ep_api->EpDevice_AddAllocatorInfo(ep_devices[n], mi);
        if (st) {
            g_api->ReleaseKeyValuePairs(md);
            return st;
        }
        ++n;
    }
    g_api->ReleaseKeyValuePairs(md);
    *num_ep_devices = n;
    tracef("supported_devices given=%zu max=%zu claimed=%zu", num_devices, max_ep_devices, n);
    return nullptr;
}

OrtStatus* ORT_API_CALL F_ValidateCompatibility(OrtEpFactory*, const OrtHardwareDevice* const*,
                                                size_t, const char*,
                                                OrtCompiledModelCompatibility* out) noexcept {
    *out = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
}

OrtStatus* ORT_API_CALL F_CreateAllocator(OrtEpFactory*, const OrtMemoryInfo* mi,
                                          const OrtKeyValuePairs*, OrtAllocator** out) noexcept {
    *out = new_allocator(mi);
    return *out ? nullptr : g_api->CreateStatus(ORT_FAIL, "onebit_ep: allocator table full");
}

void ORT_API_CALL F_ReleaseAllocator(OrtEpFactory*, OrtAllocator*) noexcept {}

OrtStatus* ORT_API_CALL F_CreateDataTransfer(OrtEpFactory*, OrtDataTransferImpl** out) noexcept {
    *out = (OrtDataTransferImpl*)&g_transfer;
    return nullptr;
}

OrtStatus* ORT_API_CALL F_CreateSyncStreamForDevice(OrtEpFactory*, const OrtMemoryDevice*,
                                                    const OrtKeyValuePairs*,
                                                    OrtSyncStreamImpl** out) noexcept {
    *out = nullptr;
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "onebit_ep: no sync streams yet");
}

OrtStatus* ORT_API_CALL F_CreateEp(OrtEpFactory*, const OrtHardwareDevice* const*,
                                   const OrtKeyValuePairs* const*, size_t num_devices,
                                   const OrtSessionOptions*, const OrtLogger*, OrtEp** ep) noexcept {
    (void)num_devices;
    *ep = (OrtEp*)&g_ep;
    return nullptr;
}

void ORT_API_CALL F_ReleaseEp(OrtEpFactory*, OrtEp* ep) noexcept { (void)ep; }

const OrtEpFactory g_factory = {
    /* ort_version_supported */ ORT_API_VERSION,
    /* GetName */ F_GetName,
    /* GetVendor */ F_GetVendor,
    /* GetSupportedDevices */ F_GetSupportedDevices,
    /* CreateEp */ F_CreateEp,
    /* ReleaseEp */ F_ReleaseEp,
    /* GetVendorId */ F_GetVendorId,
    /* GetVersion */ F_GetVersion,
    /* ValidateCompiledModelCompatibilityInfo */ F_ValidateCompatibility,
    /* CreateAllocator */ F_CreateAllocator,
    /* ReleaseAllocator */ F_ReleaseAllocator,
    /* CreateDataTransfer */ F_CreateDataTransfer,
    /* IsStreamAware */ F_IsStreamAware,
    /* CreateSyncStreamForDevice */ F_CreateSyncStreamForDevice,
};

}  // namespace

// ------------------------------------------------------------------------------------- entry points

extern "C" ONEBIT_EP_API OrtStatus* ORT_API_CALL CreateEpFactories(const char* registered_name,
                                                               const OrtApiBase* ort_api_base,
                                                               const OrtLogger*,
                                                               OrtEpFactory** factories,
                                                               size_t max_factories,
                                                               size_t* num_factories) noexcept {
    // GetApi returns NULL for an ABI version this library cannot speak; returning NULL rather than
    // dereferencing is what ORT expects (a non-null OrtStatus* would be reported as a load error).
    g_api = ort_api_base ? ort_api_base->GetApi(ORT_API_VERSION) : nullptr;
    if (!g_api) return nullptr;
    g_ep_api = g_api->GetEpApi();
    if (!g_ep_api) return g_api->CreateStatus(ORT_FAIL, "onebit_ep: OrtEpApi unavailable");
    if (max_factories < 1) return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: no room");
    factories[0] = (OrtEpFactory*)&g_factory;
    *num_factories = 1;
    tracef("create_ep_factories name=%s max=%zu", registered_name ? registered_name : "(null)",
           max_factories);
    return nullptr;
}

// Must return OrtStatus*: ORT's EpLibraryPlugin::Unload() wraps the result in ToStatusAndRelease and
// dereferences it. Returning void made ORT read whatever was in RAX as an OrtStatus and segfault on
// ReleaseEnv (found by bisecting /tmp/envtest against the ORT 1.23.2 source).
extern "C" ONEBIT_EP_API OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory*) noexcept { return nullptr; }
