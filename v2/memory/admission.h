#pragma once

// [MEM] Admission against honest free memory (design D5, gap G1). HRX reports free == total, so
// V2 never asks the runtime: it reads what the box has (MemAvailable from /proc/meminfo; the amdgpu
// GTT pool's total and used from sysfs, the same DRAM on Strix Halo) and compares it with an exact
// budget built at load:
//
//   weights + KV/state at (max context x slots) + max scratch over all planned graphs
//   + code objects + margin (2 GiB)
//
// A load that does not fit is refused before anything is allocated, with the numbers. The readings
// and the min(GTT free, MemAvailable) rule are the method of serve's admission (engine PR #398,
// app/mem_admit.h / app/sysinfo.h, after tools/mem-budget/mem_budget.py); `root` prefixes every path
// so tests stand up a fake /proc and /sys.

#include "memory.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::v2 {

constexpr uint64_t kMiB = uint64_t(1) << 20, kGiB = uint64_t(1) << 30;

struct BoxMemory {
    bool has_meminfo = false;
    uint64_t mem_total = 0, mem_available = 0;
    bool has_gtt = false;
    uint64_t gtt_total = 0, gtt_used = 0;
    uint64_t vram_total = 0, vram_used = 0;
    std::string card;
};

BoxMemory read_box_memory(const std::string& root = "");

struct Budget {
    uint64_t weights = 0, kv_state = 0, scratch = 0, code = 0, margin = 2 * kGiB;
    uint64_t total() const { return weights + kv_state + scratch + code + margin; }
};

struct Verdict {
    bool checked = false;   // the box reported its memory
    bool fits = true;
    uint64_t need = 0, free = 0;
    std::string box;        // "GTT 20.1 of 120.0 GiB free, MemAvailable 40.1 GiB"
    std::vector<std::string> warnings;
};

Verdict admit(const Budget& b, const BoxMemory& m, Device dev);

// Human-readable budget report (one line per item) and a JSON object for /metrics and the CLI.
std::string budget_report(const Budget& b, const Verdict& v, Device dev);
std::string budget_json(const Budget& b, const Verdict& v, Device dev);

}  // namespace onebit::v2
