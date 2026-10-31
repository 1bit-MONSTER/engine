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
#include <map>
#include <mutex>
#include <memory>
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
    bool used;  // the host holds this slot; F_ReleaseAllocator frees it again
};

// The host creates one allocator per session and memory info it needs, and releases it when the
// session goes away, so these are recycled rather than accumulated: a process that opens several
// sessions (a bench, a model plus a warm-up session) would otherwise run out. Eight live at once is
// already more than a single-device session ever asks for.
constexpr size_t kMaxAllocators = 8;
EpAllocator g_allocators[kMaxAllocators];

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
    for (size_t i = 0; i < kMaxAllocators; ++i) {
        EpAllocator* a = &g_allocators[i];
        if (a->used) continue;
        std::memset(a, 0, sizeof(*a));
        a->used = true;
        a->mi = mi;
        a->self.version = ORT_API_VERSION;
        a->self.Alloc = A_Alloc;
        a->self.Free = A_Free;
        a->self.Info = A_Info;
        return &a->self;
    }
    return nullptr;
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

// Reads a rank-2/rank-3 tensor's dims, refusing any other element type. Anything else is refused
// rather than guessed at: a wrong shape or type here corrupts the graph silently, which is worse
// than a session that fails to load.
OrtStatus* tensor_dims(const OrtValue* v, ONNXTensorElementDataType want, int64_t dims[3],
                       size_t* rank) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* st = g_api->GetTensorTypeAndShape(v, &info)) return st;
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    OrtStatus* st = g_api->GetTensorElementType(info, &type);
    if (!st && type != want) {
        st = g_api->CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "onebit_ep: tensor element type not supported by this kernel");
    }
    size_t ndim = 0;
    if (!st) st = g_api->GetDimensionsCount(info, &ndim);
    if (!st && (ndim < 2 || ndim > 3)) {
        st = g_api->CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "onebit_ep: only rank-2 and rank-3 tensors are supported yet");
    }
    if (!st) st = g_api->GetDimensions(info, dims, ndim);
    if (!st) *rank = ndim;
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    return st;
}

// ------------------------------------------------------------- graph introspection helpers

// Everything below reads nodes and values out of the graph's type information. Each helper swallows
// the OrtStatus instead of propagating it: a claim decision is a boolean, and a value with no type
// information is simply one this EP cannot run.
bool node_inputs(const OrtNode* node, std::vector<const OrtValueInfo*>* out) {
    size_t n = 0;
    if (OrtStatus* st = g_api->Node_GetNumInputs(node, &n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    out->assign(n ? n : 1, nullptr);
    if (OrtStatus* st = g_api->Node_GetInputs(node, out->data(), n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    out->resize(n);
    return true;
}

bool node_outputs(const OrtNode* node, std::vector<const OrtValueInfo*>* out) {
    size_t n = 0;
    if (OrtStatus* st = g_api->Node_GetNumOutputs(node, &n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    out->assign(n ? n : 1, nullptr);
    if (OrtStatus* st = g_api->Node_GetOutputs(node, out->data(), n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    out->resize(n);
    return true;
}

std::string node_op(const OrtNode* node) {
    const char* op = nullptr;
    if (OrtStatus* st = g_api->Node_GetOperatorType(node, &op)) {
        g_api->ReleaseStatus(st);
        return {};
    }
    return op ? op : "";
}

std::string node_name(const OrtNode* node) {
    const char* name = nullptr;
    if (OrtStatus* st = g_api->Node_GetName(node, &name)) {
        g_api->ReleaseStatus(st);
        return {};
    }
    return name ? name : "";
}

bool onnx_domain(const OrtNode* node) {
    const char* domain = nullptr;
    if (OrtStatus* st = g_api->Node_GetDomain(node, &domain)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    return !domain || !*domain || std::strcmp(domain, "ai.onnx") == 0;
}

std::string value_name(const OrtValueInfo* v) {
    const char* name = nullptr;
    if (OrtStatus* st = g_api->GetValueInfoName(v, &name)) {
        g_api->ReleaseStatus(st);
        return {};
    }
    return name ? name : "";
}

bool value_rank_type(const OrtValueInfo* v, ONNXTensorElementDataType* elem, size_t* rank) {
    const OrtTypeInfo* ti = nullptr;
    if (OrtStatus* st = g_api->GetValueInfoTypeInfo(v, &ti)) {
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

// The value's dimensions, with -1 for every symbolic one.
bool value_dims(const OrtValueInfo* v, std::vector<int64_t>* dims) {
    const OrtTypeInfo* ti = nullptr;
    if (OrtStatus* st = g_api->GetValueInfoTypeInfo(v, &ti)) {
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
    if (OrtStatus* st = g_api->GetDimensions(tsi, raw.data(), rank)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    *dims = std::move(raw);
    return true;
}

// A value's constant data when it is an initializer. The OrtValue belongs to the graph (it dies with
// it), so this is a look, never a cache: callers read what they need and keep nothing.
const OrtValue* initializer(const OrtValueInfo* v) {
    const OrtValue* value = nullptr;
    if (OrtStatus* st = g_api->ValueInfo_GetInitializerValue(v, &value)) {
        g_api->ReleaseStatus(st);
        return nullptr;
    }
    return value;
}

// A per-tensor scale: a float initializer holding exactly one number.
bool float_scalar_init(const OrtValueInfo* v) {
    ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    if (!value_rank_type(v, &elem, &rank)) return false;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank != 0) return false;
    return initializer(v) != nullptr;
}

// How many values a zero point contributes, or -1 when it cannot be folded. A zero point folds when
// it is a scalar, or a 1-D vector along the same axis its operand's epilogue term uses (the rows for
// the activation, the output columns for the weight) -- the axes the scales use too, which is also
// what ONNX requires of a per-axis zero point. A zero point along K would weight the products inside
// the sum (one column-weighted vector of the weight, which no kernel here precomputes), so a graph
// with one keeps the per-node path.
int64_t zp_count(const OrtValueInfo* v, int64_t axis, int64_t required_axis, int64_t expected_len) {
    ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    if (!value_rank_type(v, &elem, &rank)) return -1;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) return -1;
    if (!initializer(v)) return -1;
    if (rank == 0) return 1;
    if (rank != 1) return -1;
    std::vector<int64_t> dims;
    if (!value_dims(v, &dims) || dims.size() != 1 || dims[0] <= 0) return -1;
    if (dims[0] == 1) return 1;
    if (axis != required_axis) return -1;
    if (expected_len > 0 && dims[0] != expected_len) return -1;
    return dims[0];
}

// The elements of an int8 initializer, or nullptr. Used at Compile to decide whether a zero point is
// worth folding at all and, if so, to compute the weight's column sums.
const int8_t* int8_init_data(const OrtValueInfo* v, size_t* count) {
    const OrtValue* value = initializer(v);
    if (!value) return nullptr;
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = g_api->GetTensorTypeAndShape(value, &tsi)) {
        g_api->ReleaseStatus(st);
        return nullptr;
    }
    size_t n = 0;
    OrtStatus* c_st = g_api->GetTensorShapeElementCount(tsi, &n);
    g_api->ReleaseTensorTypeAndShapeInfo(tsi);
    if (c_st) {
        g_api->ReleaseStatus(c_st);
        return nullptr;
    }
    const int8_t* data = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(value), (void**)&data)) {
        g_api->ReleaseStatus(st);
        return nullptr;
    }
    *count = n;
    return data;
}

// How many values a scale input holds: 1 for a scalar, the axis length for a per-axis vector.
int64_t scale_count(const OrtValueInfo* v) {
    std::vector<int64_t> dims;
    if (!value_dims(v, &dims)) return 0;
    if (dims.empty()) return 1;
    if (dims.size() == 1 && dims[0] > 0) return dims[0];
    return 0;
}

// How many values a zero point input holds. Unlike a scale it can be absent, which is 1 (zero).
int64_t zp_element_count(const OrtValueInfo* v) {
    const int64_t n = scale_count(v);
    return n > 0 ? n : 1;
}

// The producer of a value and the output index it comes from, or nullptr.
const OrtNode* producer(const OrtValueInfo* v, size_t* out_index) {
    const OrtNode* node = nullptr;
    size_t index = 0;
    if (OrtStatus* st = g_api->ValueInfo_GetValueProducer(v, &node, &index)) {
        g_api->ReleaseStatus(st);
        return nullptr;
    }
    if (out_index) *out_index = index;
    return node;
}

// Number of nodes that consume this value. A group we fuse must be the only consumer of the values
// inside it, otherwise fusing removes a value someone else still reads.
bool sole_consumer(const OrtValueInfo* v) {
    size_t n = 0;
    if (OrtStatus* st = g_api->ValueInfo_GetValueNumConsumers(v, &n)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    return n == 1;
}

OrtStatus* ORT_API_CALL C_Compute(OrtNodeComputeInfo*, void*, OrtKernelContext* ctx) noexcept {
    const OrtValue* av = nullptr;
    const OrtValue* bv = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, 0, &av)) return st;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, 1, &bv)) return st;

    int64_t ad[3] = {0, 0, 0}, bd[3] = {0, 0, 0};
    size_t arank = 0, brank = 0;
    if (OrtStatus* st = tensor_dims(av, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ad, &arank)) return st;
    if (OrtStatus* st = tensor_dims(bv, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, bd, &brank)) return st;

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

// --------------------------------------------------- the fused A8W8 QDQ group (task 1)

// One fused DequantizeLinear -> MatMul -> QuantizeLinear group, as the EP needs it at run time: two
// int8 tensors and three scales. The float tensors the graph dequantises never exist, so there is
// nothing else to carry.
//
// The initializers stay inputs of the fused node (no drop_constant_initializers), so the weight and
// the scales are read out of the kernel context by index on every run: ORT owns that memory, it is
// stable across runs, and the alternative -- copying the weight at Compile -- doubles it for nothing.
struct QdqPlan {
    size_t x_index;   // int8 activations [..., M, K]
    size_t w_index;   // int8 weights [K, N], a constant initializer
    size_t sx_index;  // float: activation scale, 1 value or one per row
    size_t sw_index;  // float: weight scale, 1 value or one per output column
    size_t sy_index;  // float scalar: output scale
    int64_t k;
    int64_t n;
    int64_t sx_count;  // 1 = per-tensor, m = per-row
    int64_t sw_count;  // 1 = per-tensor, n = per-column
    // Zero points, same convention. Absent ones are (size_t)-1 with a count of 1 and a value of 0.
    size_t zx_index;
    size_t zw_index;
    size_t zy_index;
    int64_t zx_count;
    int64_t zw_count;
    // sum_k w_q[k][n], the `-zp_x * Sum_k w_q` term of the asymmetric epilogue. Computed on the first
    // run that needs it and kept: recomputing per run would add O(K*N) to every decode step, and a
    // symmetric graph never asks for it at all. The mutex is for the first run of two sessions that
    // share a plan object; after that the vector is read-only.
    std::mutex colsum_mu;
    std::vector<int32_t> colsum_w;
    const int8_t* colsum_source = nullptr;  // the weight these sums were taken from
};

// Compile runs per claimed node and the EP is a process-wide singleton, so the plans outlive the
// session that registered them -- the same lifetime the static g_nci has always had.
std::map<std::string, QdqPlan*> g_plans;  // fused node name -> plan
std::vector<std::unique_ptr<QdqPlan>> g_plan_storage;

OrtStatus* ORT_API_CALL QC_CreateState(OrtNodeComputeInfo*, OrtNodeComputeContext* context,
                                       void** state) noexcept {
    const char* name = g_ep_api->NodeComputeContext_NodeName(context);
    auto it = name ? g_plans.find(name) : g_plans.end();
    if (it == g_plans.end()) {
        *state = nullptr;
        return g_api->CreateStatus(ORT_FAIL,
                                   "onebit_ep: a fused QDQ node has no plan (Compile did not see it)");
    }
    *state = it->second;
    return nullptr;
}

// Reads a scale input as a pointer plus its number of values. ORT owns the memory (the scale stays an
// input of the fused node), so this is a look, not a copy.
bool input_scale(const OrtKernelContext* ctx, size_t index, const float** data, int64_t* count) {
    const OrtValue* v = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, index, &v)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!v) return false;
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = g_api->GetTensorTypeAndShape(v, &tsi)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    size_t n = 0;
    OrtStatus* c_st = g_api->GetTensorShapeElementCount(tsi, &n);
    g_api->ReleaseTensorTypeAndShapeInfo(tsi);
    if (c_st) {
        g_api->ReleaseStatus(c_st);
        return false;
    }
    const float* p = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(v), (void**)&p)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    *data = p;
    *count = (int64_t)n;
    return true;
}

// Reads an int8 input (a zero point) as a pointer plus its number of values.
bool input_int8(const OrtKernelContext* ctx, size_t index, const int8_t** data, int64_t* count) {
    const OrtValue* v = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, index, &v)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!v) return false;
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (OrtStatus* st = g_api->GetTensorTypeAndShape(v, &tsi)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    size_t n = 0;
    OrtStatus* c_st = g_api->GetTensorShapeElementCount(tsi, &n);
    g_api->ReleaseTensorTypeAndShapeInfo(tsi);
    if (c_st) {
        g_api->ReleaseStatus(c_st);
        return false;
    }
    const int8_t* p = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(v), (void**)&p)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    *data = p;
    *count = (int64_t)n;
    return true;
}

// Reads a float scalar input at run time.
bool input_scalar(const OrtKernelContext* ctx, size_t index, float* out) {
    const OrtValue* v = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(ctx, index, &v)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!v) return false;
    const float* p = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(v), (void**)&p)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    *out = *p;
    return true;
}

// sum_k w_q[k][n] for this plan's weight, computed once per (plan, weight) and reused. Only called for
// a non-zero activation zero point: that is the term that makes it free rather than a second pass.
void ensure_colsum(QdqPlan* plan, const int8_t* w) {
    std::lock_guard<std::mutex> lock(plan->colsum_mu);
    if (plan->colsum_source == w && plan->colsum_w.size() == (size_t)plan->n) return;
    plan->colsum_w.assign((size_t)plan->n, 0);
    for (int64_t p = 0; p < plan->k; ++p)
        for (int64_t j = 0; j < plan->n; ++j)
            plan->colsum_w[(size_t)j] += w[p * plan->n + j];
    plan->colsum_source = w;
    tracef("qdq_colsum k=%lld n=%lld", (long long)plan->k, (long long)plan->n);
}

OrtStatus* ORT_API_CALL QC_Compute(OrtNodeComputeInfo*, void* state, OrtKernelContext* context) noexcept {
    QdqPlan* plan = (QdqPlan*)state;
    const OrtValue* av = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(context, plan->x_index, &av)) return st;
    int64_t ad[3] = {0, 0, 0};
    size_t arank = 0;
    if (OrtStatus* st = tensor_dims(av, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ad, &arank)) return st;
    // Any number of leading dims is a batch here: the multiply only ever sees [rows, K], which is
    // what a rank-3 activation ([batch, sequence, K]) from a real QDQ graph is.
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < arank; ++i) rows *= ad[i];
    if (ad[arank - 1] != plan->k) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: fused QDQ activation width changed since Compile");
    }
    int64_t out_dims[3] = {0, 0, 0};
    for (size_t i = 0; i + 1 < arank; ++i) out_dims[i] = ad[i];
    out_dims[arank - 1] = plan->n;
    OrtValue* out = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetOutput(context, 0, out_dims, arank, &out)) return st;

    const OrtValue* wv = nullptr;
    if (OrtStatus* st = g_api->KernelContext_GetInput(context, plan->w_index, &wv)) return st;
    const int8_t* a = nullptr;
    const int8_t* w = nullptr;
    int8_t* y = nullptr;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(av), (void**)&a)) return st;
    if (OrtStatus* st = g_api->GetTensorMutableData(const_cast<OrtValue*>(wv), (void**)&w)) return st;
    if (OrtStatus* st = g_api->GetTensorMutableData(out, (void**)&y)) return st;
    const float* s_x = nullptr;
    const float* s_w = nullptr;
    int64_t s_x_count = 0, s_w_count = 0;
    float s_y = 1.0f;
    if (!input_scale(context, plan->sx_index, &s_x, &s_x_count) ||
        !input_scale(context, plan->sw_index, &s_w, &s_w_count) ||
        !input_scalar(context, plan->sy_index, &s_y)) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: fused QDQ scales are unreadable");
    }
    if (s_y == 0.0f) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: fused QDQ output scale is zero");
    }
    // A per-axis scale has to cover the rows (activation) or the columns (weight) exactly. If the
    // shape moved under us, fail instead of reading past the end of the vector.
    if (s_x_count != 1 && s_x_count != rows) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: fused QDQ activation scale does not cover the rows");
    }
    if (s_w_count != 1 && s_w_count != plan->n) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: fused QDQ weight scale does not cover the columns");
    }

    onebit_ep::Q8Epilogue epilogue;
    epilogue.s_x_row = s_x_count == 1 ? nullptr : s_x;
    epilogue.s_x = s_x_count == 1 ? s_x[0] : 1.0f;
    epilogue.s_w_col = s_w_count == 1 ? nullptr : s_w;
    epilogue.s_w = s_w_count == 1 ? s_w[0] : 1.0f;
    epilogue.inv_s_y = 1.0f / s_y;

    // Zero points. Read from the kernel context like the scales, and all-zero ones are dropped here so
    // the kernel's symmetric path stays the one that runs for a symmetric graph.
    const int8_t* zx = nullptr;
    const int8_t* zw = nullptr;
    const int8_t* zy = nullptr;
    int64_t zx_count = 1, zw_count = 1, zy_count = 1;
    if (plan->zx_index != (size_t)-1 && !input_int8(context, plan->zx_index, &zx, &zx_count)) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: fused QDQ zp_x is unreadable");
    }
    if (plan->zw_index != (size_t)-1 && !input_int8(context, plan->zw_index, &zw, &zw_count)) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: fused QDQ zp_w is unreadable");
    }
    if (plan->zy_index != (size_t)-1 && !input_int8(context, plan->zy_index, &zy, &zy_count)) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "onebit_ep: fused QDQ zp_y is unreadable");
    }
    if (zx_count != 1 && zx_count != rows) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: fused QDQ activation zero point does not cover the rows");
    }
    if (zw_count != 1 && zw_count != plan->n) {
        return g_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                   "onebit_ep: fused QDQ weight zero point does not cover the columns");
    }
    if (zy) epilogue.zp_y = zy[0];
    bool zx_nonzero = false;
    if (zx) {
        for (int64_t i = 0; i < zx_count; ++i) zx_nonzero = zx_nonzero || zx[i] != 0;
        if (zx_count == 1) epilogue.zp_x = zx[0];
        else if (zx_nonzero) epilogue.zp_x_row = zx;
    }
    if (zw) {
        bool nonzero = false;
        for (int64_t i = 0; i < zw_count; ++i) nonzero = nonzero || zw[i] != 0;
        if (zw_count == 1) epilogue.zp_w = zw[0];
        else if (nonzero) epilogue.zp_w_col = zw;
    }
    if (epilogue.zp_x != 0 || epilogue.zp_x_row) {
        ensure_colsum(plan, w);
        epilogue.colsum_w = plan->colsum_w.data();
    }

    onebit_ep::gemm_q8(a, w, y, rows, plan->k, plan->n, epilogue);
    // The int8-in / int8-out line, not the float MatMul one: the proof reads this trace back to
    // prove which of the two ran.
    tracef("qdq rows=%lld k=%lld n=%lld sx=%lld sw=%lld zp_x=%lld zp_w=%lld zp_y=%d",
           (long long)rows, (long long)plan->k, (long long)plan->n, (long long)s_x_count,
           (long long)s_w_count, (long long)zx_count, (long long)zw_count, (int)epilogue.zp_y);
    if (verbose()) {
        std::fprintf(stderr,
                     "onebit_ep: qdq %lldx%lldx%lld int8 -> int8 (sx=%.9g%s, sw=%.9g%s, zp_x=%d%s, "
                     "zp_w=%d%s, zp_y=%d)\n",
                     (long long)rows, (long long)plan->k, (long long)plan->n,
                     s_x_count == 1 ? s_x[0] : 0.0, s_x_count == 1 ? "" : " per row",
                     s_w_count == 1 ? s_w[0] : 0.0, s_w_count == 1 ? "" : " per column",
                     epilogue.zp_x, epilogue.zp_x_row ? " per row" : "", epilogue.zp_w,
                     epilogue.zp_w_col ? " per column" : "", (int)epilogue.zp_y);
    }
    return nullptr;
}

const OrtNodeComputeInfo g_qdq_nci = {
    /* ort_version_supported */ ORT_API_VERSION,
    /* CreateState */ QC_CreateState,
    /* Compute */ QC_Compute,
    /* ReleaseState */ C_ReleaseState,
};

// ----------------------------------------------------------------------------------------- the EP

const char* ORT_API_CALL E_GetName(const OrtEp*) noexcept { return kEpName; }

// The rank and element type of one node input.
bool input_rank_type(const OrtNode* node, size_t index, ONNXTensorElementDataType* elem,
                     size_t* rank) {
    std::vector<const OrtValueInfo*> inputs;
    if (!node_inputs(node, &inputs) || index >= inputs.size()) return false;
    return value_rank_type(inputs[index], elem, rank);
}

// The node input's dimensions, with -1 for every symbolic one.
bool input_dims(const OrtNode* node, size_t index, std::vector<int64_t>* dims) {
    std::vector<const OrtValueInfo*> inputs;
    if (!node_inputs(node, &inputs) || index >= inputs.size()) return false;
    return value_dims(inputs[index], dims);
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

// The DequantizeLinear `axis` attribute, which says which dimension a per-axis scale runs along.
// Default 1 (the opset 13+ default), and ORT hands the default back for an unset optional attribute.
int64_t node_axis(const OrtNode* node) {
    const OrtOpAttr* attr = nullptr;
    if (OrtStatus* st = g_api->Node_GetAttributeByName(node, "axis", &attr)) {
        g_api->ReleaseStatus(st);
        return 1;
    }
    if (!attr) return 1;
    int64_t axis = 1;
    size_t out = 0;
    if (OrtStatus* st = g_api->ReadOpAttr(attr, ORT_OP_ATTR_INT, &axis, sizeof(axis), &out)) {
        g_api->ReleaseStatus(st);
        return 1;
    }
    return axis;
}

// How many values a DequantizeLinear scale contributes, or 0 when the shape cannot be folded into the
// kernel's epilogue. 1 means per-tensor (a scalar, or a one-element vector). A per-axis scale is
// folded only along `required_axis`: for the weight that is the output axis (so `s_w[n]` multiplies
// output column n), for the activation the row axis (so `s_x[m]` scales output row m). Anything else --
// notably an activation scale along K -- weights the individual products inside the sum and would
// need a different kernel, so the graph keeps the per-node path instead.
int64_t foldable_scale_len(const OrtValueInfo* scale, int64_t axis, int64_t required_axis,
                           int64_t expected_len) {
    ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    if (!value_rank_type(scale, &elem, &rank)) return 0;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return 0;
    if (!initializer(scale)) return 0;
    if (rank == 0) return 1;
    if (rank != 1) return 0;
    std::vector<int64_t> dims;
    if (!value_dims(scale, &dims) || dims.size() != 1 || dims[0] <= 0) return 0;
    if (dims[0] == 1) return 1;
    if (axis != required_axis) return 0;
    if (expected_len > 0 && dims[0] != expected_len) return 0;
    return dims[0];
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

// --------------------------------------------------------------- the A8W8 QDQ pattern

// What GetCapability learned about one group, so Compile can turn the fused node back into a
// QdqPlan. The fused node keeps the group's values as inputs under their original names, and the
// three float scalars are indistinguishable by type and rank -- the names are the only honest way to
// say which one is the activation scale, the weight scale and the output scale.
struct QdqDef {
    std::string x_name, w_name, sx_name, sw_name, sy_name;
    std::string zx_name, zw_name, zy_name;  // empty when the group has no such zero point
};

std::map<std::string, QdqDef> g_qdq_defs;  // group output (y_q) value name -> roles

// Fusion can be switched off (ONEBIT_EP_QDQ=0) so the proof can measure the same graph both ways.
bool qdq_fusion_enabled() {
    const char* v = std::getenv("ONEBIT_EP_QDQ");
    return !(v && *v == '0');
}

// Matches the A8W8 QDQ group Quark's ONNX-to-ONNX flow emits (docs/onnx-pack.md):
//
//   x_q --DQ(s_x, 0)--> x_dq \
//                              MatMul --> y --Q(s_y, 0)--> y_q
//   w_q --DQ(s_w, 0)--> w_dq /
//
// Symmetric and per-tensor only. Per-axis scales need the kernel to take s_x[m] and s_w[n] and a
// non-zero zero point needs the correction terms in the epilogue (tasks 2 and 3 of the brief), so a
// graph with either keeps the per-node path: slower, still correct.
//
// `group` comes back as the four nodes to fuse into one, in graph order: dq_x, dq_w, matmul, q_y.
bool match_qdq(const OrtNode* mm, const OrtNode* group[4], QdqDef* def) {
    std::vector<const OrtValueInfo*> mm_in, mm_out;
    if (!node_inputs(mm, &mm_in) || mm_in.size() != 2) return false;
    if (!node_outputs(mm, &mm_out) || mm_out.size() != 1) return false;

    // Both MatMul inputs are a DequantizeLinear output, and this group is its only consumer.
    const OrtNode* dq[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        size_t out_index = 0;
        dq[i] = producer(mm_in[(size_t)i], &out_index);
        if (!dq[i] || out_index != 0) return false;
        if (node_op(dq[i]) != "DequantizeLinear" || !onnx_domain(dq[i])) return false;
        if (!sole_consumer(mm_in[(size_t)i])) return false;
    }

    std::vector<const OrtValueInfo*> xin, win;
    if (!node_inputs(dq[0], &xin) || xin.size() < 2 || xin.size() > 3) return false;
    if (!node_inputs(dq[1], &win) || win.size() < 2 || win.size() > 3) return false;

    ONNXTensorElementDataType elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t rank = 0;
    // Activation: int8, rank 2 or 3, zero point zero (or absent).
    if (!value_rank_type(xin[0], &elem, &rank)) return false;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) return false;
    if (rank != 2 && rank != 3) return false;
    const size_t xrank = rank;
    if (xin.size() == 3 && zp_count(xin[2], node_axis(dq[0]), xrank == 2 ? 0 : -1, -1) < 0)
        return false;

    // Weight: a constant int8 [K, N], zero point zero (or absent).
    if (!value_rank_type(win[0], &elem, &rank)) return false;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || rank != 2) return false;
    if (!initializer(win[0])) return false;
    if (win.size() == 3 && zp_count(win[2], node_axis(dq[1]), 1, -1) < 0) return false;
    std::vector<int64_t> xd, wd;
    if (!value_dims(xin[0], &xd) || !value_dims(win[0], &wd)) return false;
    if (wd.size() != 2 || wd[0] <= 0 || wd[1] <= 0) return false;
    if (xd.empty() || (xd.back() > 0 && xd.back() != wd[0])) return false;

    // Scales: per-tensor, or per-axis along a dimension the epilogue can fold. The activation has to
    // be rank 2 for a per-row scale (a rank-3 activation's rows are batch*sequence, which no exporter
    // writes as one axis), and the weight's per-axis scale has to run along N.
    if (!foldable_scale_len(xin[1], node_axis(dq[0]), xrank == 2 ? 0 : -1, xd.front())) return false;
    if (!foldable_scale_len(win[1], node_axis(dq[1]), 1, wd[1])) return false;

    // The output side: the MatMul has exactly one consumer, a QuantizeLinear of that same value.
    size_t consumers = 0;
    if (OrtStatus* st = g_api->ValueInfo_GetValueNumConsumers(mm_out[0], &consumers)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (consumers != 1) return false;
    const OrtNode* q = nullptr;
    int64_t q_input = -1;
    if (OrtStatus* st = g_api->ValueInfo_GetValueConsumers(mm_out[0], &q, &q_input, 1)) {
        g_api->ReleaseStatus(st);
        return false;
    }
    if (!q || q_input != 0) return false;
    if (node_op(q) != "QuantizeLinear" || !onnx_domain(q)) return false;
    std::vector<const OrtValueInfo*> qin, qout;
    if (!node_inputs(q, &qin) || qin.size() < 2 || qin.size() > 3) return false;
    if (qin[0] != mm_out[0]) return false;
    if (!node_outputs(q, &qout) || qout.size() != 1) return false;
    if (!float_scalar_init(qin[1])) return false;
    // The output zero point is added after the round, so it is always foldable -- but only as a
    // scalar: a per-axis output zero point would need the axis of the output, and nothing emits one.
    if (qin.size() == 3) {
        size_t zy_count = 0;
        if (!value_rank_type(qin[2], &elem, &rank)) return false;
        if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || rank > 1) return false;
        if (!int8_init_data(qin[2], &zy_count) || zy_count != 1) return false;
    }
    if (!value_rank_type(qout[0], &elem, &rank)) return false;
    if (elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || rank != xd.size()) return false;

    group[0] = dq[0];
    group[1] = dq[1];
    group[2] = mm;
    group[3] = q;
    def->x_name = value_name(xin[0]);
    def->w_name = value_name(win[0]);
    def->sx_name = value_name(xin[1]);
    def->sw_name = value_name(win[1]);
    def->sy_name = value_name(qin[1]);
    if (xin.size() == 3) def->zx_name = value_name(xin[2]);
    if (win.size() == 3) def->zw_name = value_name(win[2]);
    if (qin.size() == 3) def->zy_name = value_name(qin[2]);
    return !def->x_name.empty() && !def->w_name.empty() && !def->sx_name.empty() &&
           !def->sw_name.empty() && !def->sy_name.empty();
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
    size_t fused_groups = 0;
    for (size_t i = 0; !st && i < n; ++i) {
        const char* op = nullptr;
        const char* domain = nullptr;
        // Neither call can fail for a node the support query handed us, but the API marks them
        // warn_unused_result, so the status is released rather than dropped.
        if (OrtStatus* s = g_api->Node_GetOperatorType(nodes[i], &op)) g_api->ReleaseStatus(s);
        if (OrtStatus* s = g_api->Node_GetDomain(nodes[i], &domain)) g_api->ReleaseStatus(s);
        const bool onnx_domain = !domain || !*domain || std::strcmp(domain, "ai.onnx") == 0;
        if (onnx_domain && claims_op("MatMul") && qdq_fusion_enabled()) {
            // A QDQ group is claimed whole: fusing it is what keeps the dequantised float tensors
            // from ever existing. Only nodes in this group are named, and the group's output value
            // is the key Compile finds it by again.
            const OrtNode* group[4] = {nullptr, nullptr, nullptr, nullptr};
            QdqDef def;
            if (match_qdq(nodes[i], group, &def)) {
                std::vector<const OrtValueInfo*> outs;
                if (node_outputs(group[3], &outs) && outs.size() == 1) {
                    st = g_ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, group, 4, nullptr);
                    g_qdq_defs[value_name(outs[0])] = def;
                    ++fused_groups;
                    continue;
                }
            }
        }
        if (onnx_domain && can_compute(nodes[i], op)) {
            st = g_ep_api->EpGraphSupportInfo_AddNodesToFuse(support_info, &nodes[i], 1, nullptr);
            ++claimed;
        }
    }
    std::free(nodes);
    tracef("capability nodes=%zu claimed=%zu qdq_groups=%zu", n, claimed, fused_groups);
    if (verbose())
        std::fprintf(stderr, "onebit_ep: GetCapability %zu nodes -> %zu claimed, %zu QDQ groups fused\n",
                     n, claimed, fused_groups);
    return st;
}

// Turns a fused node back into the plan its compute needs. Returns false in `is_qdq` for a node that
// was claimed on its own (the float MatMul path); a QDQ group whose input roles cannot be found is an
// error, not a silent fallback -- computing it with the float kernel would read int8 as float.
OrtStatus* qdq_plan_for(const OrtNode* node, bool* is_qdq) {
    *is_qdq = false;
    if (!node) return nullptr;
    std::vector<const OrtValueInfo*> outs;
    if (!node_outputs(node, &outs) || outs.size() != 1) return nullptr;
    auto it = g_qdq_defs.find(value_name(outs[0]));
    if (it == g_qdq_defs.end()) return nullptr;
    const QdqDef& def = it->second;
    *is_qdq = true;

    auto plan = std::make_unique<QdqPlan>();
    plan->x_index = plan->w_index = plan->sx_index = plan->sw_index = plan->sy_index =
        plan->zx_index = plan->zw_index = plan->zy_index = (size_t)-1;
    plan->zx_count = plan->zw_count = 1;
    std::vector<const OrtValueInfo*> ins;
    if (!node_inputs(node, &ins)) {
        return g_api->CreateStatus(ORT_FAIL, "onebit_ep: fused QDQ node has unreadable inputs");
    }
    for (size_t i = 0; i < ins.size(); ++i) {
        const std::string name = value_name(ins[i]);
        if (name == def.x_name) {
            plan->x_index = i;
        } else if (name == def.w_name) {
            plan->w_index = i;
            std::vector<int64_t> wd;
            if (!value_dims(ins[i], &wd) || wd.size() != 2 || wd[0] <= 0 || wd[1] <= 0) {
                return g_api->CreateStatus(ORT_FAIL,
                                           "onebit_ep: fused QDQ weight has no static [K, N]");
            }
            plan->k = wd[0];
            plan->n = wd[1];
        } else if (name == def.sx_name) {
            plan->sx_index = i;
            plan->sx_count = scale_count(ins[i]);
        } else if (name == def.sw_name) {
            plan->sw_index = i;
            plan->sw_count = scale_count(ins[i]);
        } else if (name == def.sy_name) {
            plan->sy_index = i;
        } else if (!def.zx_name.empty() && name == def.zx_name) {
            plan->zx_index = i;
            plan->zx_count = zp_element_count(ins[i]);
        } else if (!def.zw_name.empty() && name == def.zw_name) {
            plan->zw_index = i;
            plan->zw_count = zp_element_count(ins[i]);
        } else if (!def.zy_name.empty() && name == def.zy_name) {
            plan->zy_index = i;
        }
    }
    if (plan->x_index == (size_t)-1 || plan->w_index == (size_t)-1 ||
        plan->sx_index == (size_t)-1 || plan->sw_index == (size_t)-1 ||
        plan->sy_index == (size_t)-1) {
        return g_api->CreateStatus(
            ORT_FAIL, "onebit_ep: fused QDQ node does not carry the values GetCapability matched");
    }
    if (plan->sx_count <= 0 || plan->sw_count <= 0 ||
        (plan->sw_count != 1 && plan->sw_count != plan->n)) {
        return g_api->CreateStatus(ORT_FAIL, "onebit_ep: fused QDQ scales do not fit the weight");
    }
    // Read what the log needs before the plan is moved into its storage: after std::move, `plan` is
    // null and dereferencing it is the kind of bug that only shows up in an optimized build.
    const std::string fname = node_name(node);
    const int64_t plan_k = plan->k, plan_n = plan->n;
    g_plans[fname] = plan.get();
    g_plan_storage.push_back(std::move(plan));
    tracef("qdq_plan node=%s k=%lld n=%lld", fname.c_str(), (long long)plan_k, (long long)plan_n);
    return nullptr;
}

OrtStatus* ORT_API_CALL E_Compile(OrtEp*, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                  size_t count, OrtNodeComputeInfo** node_compute_infos,
                                  OrtNode** ep_context_nodes) noexcept {
    (void)graphs;
    // Compile is per claimed node and is where an EP would see the shapes to bake in. We keep the
    // shapes for the log only; the kernel reads them again at run time, because claiming a node does
    // not promise its input shapes stay fixed across sessions.
    for (size_t i = 0; i < count; ++i) {
        bool is_qdq = false;
        if (OrtStatus* st = qdq_plan_for(fused_nodes ? fused_nodes[i] : nullptr, &is_qdq)) return st;
        if (verbose()) {
            const char* op = nullptr;
            if (!g_api->Node_GetOperatorType(fused_nodes[i], &op)) {
                std::fprintf(stderr, "onebit_ep: Compile fused[%zu] %s (%s)\n", i, op ? op : "?",
                             is_qdq ? "QDQ group" : "node");
            }
        }
        tracef("compile node=%p qdq=%d", (const void*)fused_nodes[i], is_qdq ? 1 : 0);
        node_compute_infos[i] = (OrtNodeComputeInfo*)(is_qdq ? &g_qdq_nci : &g_nci);
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

void ORT_API_CALL F_ReleaseAllocator(OrtEpFactory*, OrtAllocator* self) noexcept {
    // The host is done with this allocator; hand the slot back. The struct itself stays put, so a
    // stray late Alloc() call still lands on a valid object.
    if (auto* a = reinterpret_cast<EpAllocator*>(self); a && a->used) {
        a->used = false;
        a->mi = nullptr;
    }
}

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
