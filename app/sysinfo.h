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

// What the box says about itself, from procfs and sysfs (Linux; elsewhere every reading is
// absent): the CPU's Tctl (k10temp), RAM (/proc/meminfo) and the amdgpu GTT pool
// (mem_info_gtt_*). On Strix Halo the GPU's memory is GTT carved from the same DRAM, so a model
// needs room in both. `root` prefixes every path, so a test can stand up a fake /proc and /sys.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace onebit::sysinfo {

inline std::optional<uint64_t> read_u64(const std::filesystem::path& p) {
    std::ifstream f(p);
    unsigned long long v = 0;
    if (!(f >> v)) return std::nullopt;
    return uint64_t(v);
}

inline std::string read_line(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string s;
    std::getline(f, s);
    return s;
}

// Tctl in degrees C (k10temp's "Tctl" input), or nothing on a box without one.
inline std::optional<double> tctl_celsius(const std::string& root = "") {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& h : fs::directory_iterator(root + "/sys/class/hwmon", ec)) {
        if (read_line(h.path() / "name") != "k10temp") continue;
        for (int i = 1; i <= 16; i++) {
            const std::string t = "temp" + std::to_string(i);
            if (read_line(h.path() / (t + "_label")) != "Tctl") continue;
            if (auto v = read_u64(h.path() / (t + "_input"))) return double(*v) / 1000.0;
        }
    }
    return std::nullopt;
}

struct Memory {
    bool has_meminfo = false;
    uint64_t mem_total = 0, mem_available = 0;   // bytes
    bool has_gtt = false;
    uint64_t gtt_total = 0, gtt_used = 0;        // bytes, the first amdgpu card that reports GTT
    std::string card;                            // e.g. "card0"
};

inline Memory memory(const std::string& root = "") {
    namespace fs = std::filesystem;
    Memory m;
    {
        std::ifstream f(root + "/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            std::string key;
            unsigned long long kb = 0;
            if (!(in >> key >> kb)) continue;
            if (key == "MemTotal:") { m.mem_total = uint64_t(kb) << 10; m.has_meminfo = true; }
            else if (key == "MemAvailable:") m.mem_available = uint64_t(kb) << 10;
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
        auto total = read_u64(d / "mem_info_gtt_total");
        auto used = read_u64(d / "mem_info_gtt_used");
        if (total && used) {
            m.has_gtt = true;
            m.gtt_total = *total;
            m.gtt_used = *used;
            m.card = best;
        }
    }
    return m;
}

}  // namespace onebit::sysinfo
