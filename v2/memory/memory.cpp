#include "memory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace onebit::v2 {

const char* residency_name(Residency r) {
    switch (r) {
        case Residency::R0_DeviceLocal: return "R0 device-local";
        case Residency::R1_DriverGtt: return "R1 driver-owned GTT";
        case Residency::R2_PinnedUser: return "R2 pinned user pages";
        case Residency::R3_Pageable: return "R3 pageable";
    }
    return "?";
}

const char* device_name(Device d) {
    switch (d) {
        case Device::Cpu: return "cpu";
        case Device::Gpu: return "gpu";
        case Device::Npu: return "npu";
    }
    return "?";
}

std::string residency_policy(Device dev, Residency r) {
    if (r == Residency::R2_PinnedUser)
        return "R2 (userptr / SVM / SVA / hipHostRegister) is forbidden on the execution path: reclaim and compaction "
               "invalidate those ranges and KFD evicts every queue of the process (engine #315, design D5)";
    if (dev != Device::Cpu && r == Residency::R3_Pageable)
        return std::string("R3 pageable memory is CPU-only; the ") + device_name(dev) + " executes from R0 or R1 buffers";
    return "";
}

void* HostBackend::allocate(uint64_t bytes, Residency r) {
    if (r != Residency::R3_Pageable) throw PolicyError("host backend allocates R3 only");
    const size_t n = size_t((bytes + 63) / 64 * 64);
    void* p = std::aligned_alloc(64, n ? n : 64);
    if (!p) throw std::bad_alloc();
    std::memset(p, 0, n);
    return p;
}

void HostBackend::release(void* p, uint64_t, Residency) { std::free(p); }

void* GpuBackendStub::allocate(uint64_t bytes, Residency r) {
    throw std::runtime_error("GPU allocation of " + std::to_string(bytes) + " bytes (" + residency_name(r) +
                             ") is not wired in this slice: the HRX allocator backend lands after the P1(b) memory soak "
                             "picks R1 or R0 (docs/v2.md)");
}

void MemoryManager::add_backend(std::unique_ptr<AllocatorBackend> b) {
    std::lock_guard lk(mu_);
    backends_.push_back(std::move(b));
}

Buffer MemoryManager::allocate(Device dev, uint64_t bytes, Residency r, std::string tag, const std::string& owner) {
    if (const std::string why = residency_policy(dev, r); !why.empty()) throw PolicyError(tag + ": " + why);
    std::lock_guard lk(mu_);
    if (frozen_) throw PolicyError(tag + ": allocation after load (" + std::to_string(bytes) + " bytes); V2 allocates everything at load");
    if (!owner.empty() && std::find(loading_.begin(), loading_.end(), owner) == loading_.end())
        throw PolicyError(tag + ": allocation for model '" + owner + "' outside its load (" + std::to_string(bytes) +
                          " bytes); V2 allocates everything at load");
    AllocatorBackend* be = nullptr;
    for (const auto& b : backends_)
        if (b->device() == dev && b->supports(r)) {
            be = b.get();
            break;
        }
    if (!be) throw PolicyError(tag + ": no " + device_name(dev) + " backend for " + residency_name(r));
    void* p = be->allocate(bytes, r);
    Live l;
    l.row = {next_id_++, tag, owner, dev, r, bytes};
    l.host = p;
    l.backend = be;
    live_.push_back(l);
    Buffer out;
    out.id = l.row.id;
    out.host = p;
    out.bytes = bytes;
    out.cls = r;
    out.dev = dev;
    out.tag = std::move(tag);
    out.owner = owner;
    return out;
}

void MemoryManager::begin_load(const std::string& owner) {
    std::lock_guard lk(mu_);
    if (owner.empty()) throw PolicyError("a model load needs a name");
    for (const auto& l : live_)
        if (l.row.owner == owner) throw PolicyError("model '" + owner + "' is already loaded");
    if (std::find(loading_.begin(), loading_.end(), owner) != loading_.end()) throw PolicyError("model '" + owner + "' is already loading");
    loading_.push_back(owner);
}

void MemoryManager::end_load(const std::string& owner) {
    std::lock_guard lk(mu_);
    loading_.erase(std::remove(loading_.begin(), loading_.end(), owner), loading_.end());
}

bool MemoryManager::loading(const std::string& owner) const {
    std::lock_guard lk(mu_);
    return std::find(loading_.begin(), loading_.end(), owner) != loading_.end();
}

uint64_t MemoryManager::release_owner(const std::string& owner) {
    std::lock_guard lk(mu_);
    uint64_t n = 0;
    for (size_t i = 0; i < live_.size();) {
        if (live_[i].row.owner == owner) {
            live_[i].backend->release(live_[i].host, live_[i].row.bytes, live_[i].row.cls);
            n += live_[i].row.bytes;
            live_.erase(live_.begin() + long(i));
        } else {
            ++i;
        }
    }
    loading_.erase(std::remove(loading_.begin(), loading_.end(), owner), loading_.end());
    return n;
}

uint64_t MemoryManager::used_by(const std::string& owner) const {
    std::lock_guard lk(mu_);
    uint64_t n = 0;
    for (const auto& l : live_)
        if (l.row.owner == owner) n += l.row.bytes;
    return n;
}

std::vector<std::string> MemoryManager::owners() const {
    std::lock_guard lk(mu_);
    std::vector<std::string> out;
    for (const auto& l : live_)
        if (!l.row.owner.empty() && std::find(out.begin(), out.end(), l.row.owner) == out.end()) out.push_back(l.row.owner);
    return out;
}

void MemoryManager::release(Buffer& b) {
    std::lock_guard lk(mu_);
    for (size_t i = 0; i < live_.size(); ++i)
        if (live_[i].row.id == b.id) {
            live_[i].backend->release(live_[i].host, live_[i].row.bytes, live_[i].row.cls);
            live_.erase(live_.begin() + long(i));
            b = Buffer{};
            return;
        }
}

uint64_t MemoryManager::used(Device dev) const {
    std::lock_guard lk(mu_);
    uint64_t n = 0;
    for (const auto& l : live_)
        if (l.row.dev == dev) n += l.row.bytes;
    return n;
}

uint64_t MemoryManager::used(Residency r) const {
    std::lock_guard lk(mu_);
    uint64_t n = 0;
    for (const auto& l : live_)
        if (l.row.cls == r) n += l.row.bytes;
    return n;
}

std::vector<LedgerRow> MemoryManager::ledger() const {
    std::lock_guard lk(mu_);
    std::vector<LedgerRow> out;
    for (const auto& l : live_) out.push_back(l.row);
    return out;
}

std::string MemoryManager::report() const {
    const auto rows = ledger();
    uint64_t total = 0;
    for (const auto& r : rows) total += r.bytes;
    char b[200];
    std::snprintf(b, sizeof b, "memory ledger: %zu buffers, %.2f MiB%s\n", rows.size(), double(total) / double(1 << 20),
                  frozen_ ? " (frozen: load done)" : "");
    std::string s = b;
    for (const auto& r : rows) {
        std::snprintf(b, sizeof b, "  %-16s %-28s %-4s %-22s %10.2f MiB\n", r.owner.empty() ? "-" : r.owner.c_str(), r.tag.c_str(),
                      device_name(r.dev), residency_name(r.cls),
                      double(r.bytes) / double(1 << 20));
        s += b;
    }
    return s;
}

MemoryManager::~MemoryManager() {
    for (auto& l : live_) l.backend->release(l.host, l.row.bytes, l.row.cls);
}

}  // namespace onebit::v2
