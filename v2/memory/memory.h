#pragma once

// [MEM] Memory manager (design D5): eviction-safe memory by construction.
//
// Residency classes:
//   R0 device-local     amdgpu VRAM (the BIOS carve-out), TTM-managed
//   R1 driver-owned GTT BOs the kernel driver allocates in GTT (not imported user pages)
//   R2 pinned user pages (hipHostRegister, userptr, SVM, SVA): FORBIDDEN on the execution path,
//      the #315 eviction class (kswapd/kcompactd invalidate them and KFD evicts every queue)
//   R3 pageable         CPU only (tokenizer, HTTP, and the CPU reference executor)
//
// The API has no call that takes a caller's pointer: there is nothing to register, import or map
// from user memory, so R2 cannot be reached by construction; asking for R2 by class is refused
// with the reason. GPU and NPU executors may only receive R0/R1 buffers. Everything is allocated
// at load; freeze() ends the load, and any allocation after it is refused (serving never
// allocates).
//
// One engine process holds as many models as fit (owner 2026-10-12: "we hold all the models we
// can"): one MemoryManager per process, every buffer carries the model that owns it, and a model is
// loaded inside begin_load(owner) ... end_load(owner) and removed with release_owner(owner). The
// freeze is per model: allocations for a model are refused once its load has ended.
//
// The GPU allocator backend is a stub in this slice: which of R1 or R0 is eviction-free is the P1(b)
// memory soak's answer, and the backend is wired after it reports.

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::v2 {

enum class Residency : unsigned char { R0_DeviceLocal, R1_DriverGtt, R2_PinnedUser, R3_Pageable };
enum class Device : unsigned char { Cpu, Gpu, Npu };
const char* residency_name(Residency r);
const char* device_name(Device d);

// D5's policy: "" when `dev` may execute from memory of class `r`, else the reason it may not.
std::string residency_policy(Device dev, Residency r);

struct PolicyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Buffer {
    uint64_t id = 0;
    void* host = nullptr;      // CPU mapping (R3 always; R0/R1 when host-mapped)
    uint64_t bytes = 0;
    Residency cls = Residency::R3_Pageable;
    Device dev = Device::Cpu;
    std::string tag;
    std::string owner;         // the model it belongs to ("" = engine-wide)
    explicit operator bool() const { return id != 0; }
};

class AllocatorBackend {
public:
    virtual ~AllocatorBackend() = default;
    virtual const char* name() const = 0;
    virtual Device device() const = 0;
    virtual bool supports(Residency r) const = 0;
    // Returns the host mapping (nullptr when not host-mapped); throws on failure.
    virtual void* allocate(uint64_t bytes, Residency r) = 0;
    virtual void release(void* p, uint64_t bytes, Residency r) = 0;
};

// R3 pageable host memory, zero-filled, 64-byte aligned: the CPU reference executor's backend.
class HostBackend final : public AllocatorBackend {
public:
    const char* name() const override { return "host"; }
    Device device() const override { return Device::Cpu; }
    bool supports(Residency r) const override { return r == Residency::R3_Pageable; }
    void* allocate(uint64_t bytes, Residency r) override;
    void release(void* p, uint64_t bytes, Residency r) override;
};

// The GPU backend until P1(b) reports: knows its classes, allocates nothing.
class GpuBackendStub final : public AllocatorBackend {
public:
    const char* name() const override { return "gpu-stub"; }
    Device device() const override { return Device::Gpu; }
    bool supports(Residency r) const override { return r == Residency::R0_DeviceLocal || r == Residency::R1_DriverGtt; }
    void* allocate(uint64_t bytes, Residency r) override;
    void release(void*, uint64_t, Residency) override {}
};

struct LedgerRow {
    uint64_t id = 0;
    std::string tag;
    std::string owner;
    Device dev = Device::Cpu;
    Residency cls = Residency::R3_Pageable;
    uint64_t bytes = 0;
};

class MemoryManager {
public:
    void add_backend(std::unique_ptr<AllocatorBackend> b);
    // Checks the policy, picks the backend for the device, records the allocation. Throws
    // PolicyError for a refused class or an allocation after freeze().
    Buffer allocate(Device dev, uint64_t bytes, Residency r, std::string tag, const std::string& owner = "");
    void release(Buffer& b);
    // A model's load window. allocate() for an owner is refused outside it.
    void begin_load(const std::string& owner);
    void end_load(const std::string& owner);
    bool loading(const std::string& owner) const;
    // Frees every buffer of a model (unload); returns the bytes released.
    uint64_t release_owner(const std::string& owner);
    // Engine-wide freeze (single-model tools): no allocation at all afterwards.
    void freeze() { frozen_ = true; }
    bool frozen() const { return frozen_; }
    uint64_t used(Device dev) const;
    uint64_t used(Residency r) const;
    uint64_t used_by(const std::string& owner) const;
    std::vector<std::string> owners() const;
    std::vector<LedgerRow> ledger() const;
    std::string report() const;
    ~MemoryManager();

private:
    struct Live {
        LedgerRow row;
        void* host = nullptr;
        AllocatorBackend* backend = nullptr;
    };
    mutable std::mutex mu_;
    std::vector<std::unique_ptr<AllocatorBackend>> backends_;
    std::vector<Live> live_;
    uint64_t next_id_ = 1;
    bool frozen_ = false;
    std::vector<std::string> loading_;
};

}  // namespace onebit::v2
