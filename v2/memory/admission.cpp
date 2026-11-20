#include "admission.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#include <nlohmann/json.hpp>

namespace onebit::v2 {

namespace {

std::optional<uint64_t> read_u64(const std::filesystem::path& p) {
    std::ifstream f(p);
    unsigned long long v = 0;
    if (!(f >> v)) return std::nullopt;
    return uint64_t(v);
}

std::string gib(uint64_t b) {
    char s[32];
    std::snprintf(s, sizeof s, "%.2f", double(b) / double(kGiB));
    return s;
}

}  // namespace

BoxMemory read_box_memory(const std::string& root) {
    namespace fs = std::filesystem;
    BoxMemory m;
    {
        std::ifstream f(root + "/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            std::string key;
            unsigned long long kb = 0;
            if (!(in >> key >> kb)) continue;
            if (key == "MemTotal:") {
                m.mem_total = uint64_t(kb) << 10;
                m.has_meminfo = true;
            } else if (key == "MemAvailable:") {
                m.mem_available = uint64_t(kb) << 10;
            }
        }
    }
    std::error_code ec;
    std::string best;
    for (const auto& c : fs::directory_iterator(root + "/sys/class/drm", ec)) {
        const std::string name = c.path().filename().string();
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;  // not card0-DP-1
        if (!fs::exists(c.path() / "device" / "mem_info_gtt_total")) continue;
        if (best.empty() || name < best) best = name;
    }
    if (!best.empty()) {
        const fs::path d = fs::path(root + "/sys/class/drm") / best / "device";
        const auto total = read_u64(d / "mem_info_gtt_total"), used = read_u64(d / "mem_info_gtt_used");
        if (total && used) {
            m.has_gtt = true;
            m.gtt_total = *total;
            m.gtt_used = *used;
            m.card = best;
        }
        m.vram_total = read_u64(d / "mem_info_vram_total").value_or(0);
        m.vram_used = read_u64(d / "mem_info_vram_used").value_or(0);
    }
    return m;
}

Verdict admit(const Budget& b, const BoxMemory& m, Device dev) {
    Verdict v;
    v.need = b.total();
    if (!m.has_meminfo) return v;   // not Linux / no /proc: cannot judge, do not refuse
    v.checked = true;
    v.free = m.mem_available;
    v.box = "MemAvailable " + gib(m.mem_available) + " GiB";
    if (dev == Device::Gpu) {
        if (m.has_gtt) {
            const uint64_t gtt_free = m.gtt_total > m.gtt_used ? m.gtt_total - m.gtt_used : 0;
            v.free = std::min(v.free, gtt_free);
            v.box = "GTT " + gib(gtt_free) + " of " + gib(m.gtt_total) + " GiB free, " + v.box;
        } else {
            v.warnings.push_back("no amdgpu GTT reading in sysfs; judged against MemAvailable only");
        }
    }
    v.fits = v.need <= v.free;
    if (v.fits && dev == Device::Gpu && m.has_gtt && m.gtt_used + v.need >= 95 * kGiB)
        v.warnings.push_back("GTT use after this load (~" + gib(m.gtt_used + v.need) +
                             " GiB) is in the zone where engine #315's faults were seen (>= 95 GiB)");
    return v;
}

std::string budget_report(const Budget& b, const Verdict& v, Device dev) {
    std::string s = std::string("memory budget (") + device_name(dev) + "):\n";
    s += "  weights     " + gib(b.weights) + " GiB\n";
    s += "  KV/state    " + gib(b.kv_state) + " GiB\n";
    s += "  scratch     " + gib(b.scratch) + " GiB (max over planned graphs)\n";
    s += "  code        " + gib(b.code) + " GiB\n";
    s += "  margin      " + gib(b.margin) + " GiB\n";
    s += "  total need  " + gib(b.total()) + " GiB\n";
    if (!v.checked) s += "  box         not checked (no /proc/meminfo)\n";
    else s += "  box         " + v.box + " -> " + (v.fits ? "fits" : "DOES NOT FIT") + "\n";
    for (const auto& w : v.warnings) s += "  warning     " + w + "\n";
    return s;
}

std::string budget_json(const Budget& b, const Verdict& v, Device dev) {
    nlohmann::json j = {{"device", device_name(dev)},  {"weights", b.weights}, {"kv_state", b.kv_state},
                        {"scratch", b.scratch},        {"code", b.code},       {"margin", b.margin},
                        {"total", b.total()},          {"checked", v.checked}, {"fits", v.fits},
                        {"free", v.free},              {"box", v.box},         {"warnings", v.warnings}};
    return j.dump();
}

}  // namespace onebit::v2
