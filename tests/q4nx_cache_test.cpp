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
#pragma once
// engine#1956: the Q4NX container cache had no eviction -- 104 G in 29 entries on one box, and it is regenerable scratch, not
// user data. These arms drive the sweep on a synthetic cache: the bound is respected, the least-recently-used goes first, the
// entry just written is never removed, and an entry younger than the age window survives even when the bound is exceeded
// (a serve may be reading it). No pack, no serve, no device.
#include "q4nx_cache.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; }
}

static void make_entry(const fs::path& root, const std::string& name, int mb, int age_minutes) {
    const fs::path d = root / name;
    fs::create_directories(d);
    std::ofstream f(d / "model.q4nx", std::ios::binary);
    f << std::string(size_t(mb) * 1024 * 1024, 'x');
    f.close();
    std::error_code ec;
    fs::last_write_time(d, fs::file_time_type::clock::now() - std::chrono::minutes(age_minutes), ec);
}

int main() {
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / ("q4nx-cache-test-" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    // (a) under the bound: nothing is removed
    make_entry(root, "old-a", 4, 300);
    make_entry(root, "old-b", 4, 240);
    Q4nxCacheSweep r = q4nx_cache_evict(root.string(), "", 64LL << 20);
    check(r.removed == 0 && fs::exists(root / "old-a") && fs::exists(root / "old-b"),
          "under the bound nothing is removed (removed " + std::to_string(r.removed) + ")");

    // (b) over the bound: the LRU goes first, and only enough to fit
    make_entry(root, "old-c", 4, 120);           // 12 MiB total, bound 6 MiB
    r = q4nx_cache_evict(root.string(), "", 6LL << 20);
    check(r.removed == 2, "over the bound the oldest entries go (removed " + std::to_string(r.removed) + ")");
    check(!fs::exists(root / "old-a") && !fs::exists(root / "old-b"), "and they are the two oldest");
    check(fs::exists(root / "old-c"), "while the newest survives");
    check(r.bytes_after <= (6LL << 20), "and the total is within the bound (" + std::to_string(r.bytes_after) + ")");

    // (c) the entry just written is never removed, even when it alone exceeds the bound
    fs::remove_all(root, ec);
    make_entry(root, "fresh", 8, 500);
    make_entry(root, "older", 2, 400);
    r = q4nx_cache_evict(root.string(), (root / "fresh").string(), 1LL << 20);
    check(fs::exists(root / "fresh"), "the entry just written survives even though it alone exceeds the bound");
    check(!fs::exists(root / "older"), "and the sweep still removed what it could");

    // (d) an entry younger than the age window survives: a serve may be reading it
    fs::remove_all(root, ec);
    make_entry(root, "just-written", 8, 0);
    make_entry(root, "stale", 8, 600);
    r = q4nx_cache_evict(root.string(), "", 1LL << 20, /*min_age_seconds=*/600);
    check(fs::exists(root / "just-written"), "an entry inside the age window is not removed");
    check(!fs::exists(root / "stale"), "an entry outside it is");
    check(r.reported, "and the sweep says so once rather than silently");

    // (e) max_bytes <= 0 disables the sweep -- the old unbounded behaviour, deliberately reachable
    fs::remove_all(root, ec);
    make_entry(root, "keepme-1", 4, 600);
    make_entry(root, "keepme-2", 4, 600);
    r = q4nx_cache_evict(root.string(), "", 0);
    check(r.removed == 0 && fs::exists(root / "keepme-1") && fs::exists(root / "keepme-2"),
          "a non-positive bound disables the sweep");

    fs::remove_all(root, ec);
    std::printf("q4nx cache: %d arm(s) over the sweep (bound, LRU order, keep, age window, disabled)\n",
                failures ? 0 : 12);
    return failures ? 1 : 0;
}
