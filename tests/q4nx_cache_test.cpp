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
#include "tmp_root.h"   // engine#2029: scratch goes on disk, not in tmpfs

namespace fs = std::filesystem;
static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; }
}

// argv for run_q4nx_cache(): the vector must outlive the call, so callers keep it named.
static std::vector<char*> mkargv(std::vector<std::string>& a) {
    std::vector<char*> v;
    for (auto& s : a) v.push_back(const_cast<char*>(s.c_str()));
    return v;
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
    const fs::path root = ::onebit::test::tmp_root() / ("q4nx-cache-test-" + std::to_string(::getpid()));
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

    // (f) a hit is a use: the older entry, touched, outlives a younger one nobody opened. Without the touch the order is
    // by creation and the entry served every day goes first.
    fs::remove_all(root, ec);
    make_entry(root, "served-daily", 4, 7 * 24 * 60);
    make_entry(root, "one-off", 4, 24 * 60);
    q4nx_cache_touch((root / "served-daily").string());
    r = q4nx_cache_evict(root.string(), "", 6LL << 20, /*min_age_seconds=*/0);
    check(r.removed == 1, "one entry has to go (removed " + std::to_string(r.removed) + ")");
    check(fs::exists(root / "served-daily"), "the entry that was just used survives");
    check(!fs::exists(root / "one-off"), "and the unused one is the one removed");
    // ... and a touched entry is inside the age window, so a serve that just opened it is covered like a new one
    make_entry(root, "one-off", 4, 24 * 60);
    make_entry(root, "opened-now", 4, 7 * 24 * 60);
    fs::remove_all(root / "served-daily", ec);
    q4nx_cache_touch((root / "opened-now").string());
    r = q4nx_cache_evict(root.string(), "", 1LL << 20);
    check(fs::exists(root / "opened-now") && !fs::exists(root / "one-off"),
          "a touched entry is protected by the age window");
    q4nx_cache_touch((root / "absent").string());
    check(!fs::exists(root / "absent"), "touching a missing entry creates nothing");

    // (g) the bound from ONEBIT_Q4NX_CACHE_MAX_GIB's text: never wraps, and a typo is not "unbounded"
    const int64_t gib = 1LL << 30, top = (INT64_MAX >> 30) << 30;
    check(q4nx_cache_bound_bytes(nullptr) == 60 * gib && q4nx_cache_bound_bytes("") == 60 * gib, "unset is the default");
    check(q4nx_cache_bound_bytes("100") == 100 * gib, "a plain number is that many GiB");
    check(q4nx_cache_bound_bytes("0") == 0 && q4nx_cache_bound_bytes("-5") == 0, "zero or negative disables the sweep");
    check(q4nx_cache_bound_bytes("8589934592") == top, "2^33 GiB clamps instead of overflowing");
    check(q4nx_cache_bound_bytes("17179869184") == top, "2^34 GiB is not a bound of 0");
    check(q4nx_cache_bound_bytes("17179869185") == top, "2^34+1 GiB is not a bound of 1 GiB");
    check(q4nx_cache_bound_bytes("99999999999999999999999") == top, "a number past long long clamps too");
    check(q4nx_cache_bound_bytes("sixty") == 60 * gib, "a value that is not a number is the default, not unbounded");
    check(q4nx_cache_bound_bytes("40G") == 40 * gib, "a number with a suffix keeps its number");
    check(q4nx_cache_bound_bytes("7", 9) == 7 * gib && q4nx_cache_bound_bytes(nullptr, 9) == 9 * gib, "the default is the caller's");

    // engine#1788 / disk pressure: the sweep as a COMMAND. Until now a serve start was the only trigger, so nothing
    // swept while the NPU was off -- the cache sat at 70.9 GiB against its 60 GiB bound with / at 149 G.
    {
        const fs::path cr = root / "cli";
        make_entry(cr, "big-a", 8, 500);
        make_entry(cr, "big-b", 8, 400);
        const std::string root_arg = "--root", root_val = cr.string();
        {
            // dry-run over the bound: reports and removes nothing
            std::vector<std::string> a = {"evict", root_arg, root_val, "--max-gib", "0", "--dry-run"};
            std::vector<char*> av = mkargv(a);
            check(::run_q4nx_cache(int(av.size()), av.data()) == 0,
                  "q4nx-cache evict: --max-gib 0 is 'sweep is off', exit 0");
            check(fs::exists(cr / "big-a") && fs::exists(cr / "big-b"), "q4nx-cache evict: nothing removed with the sweep off");
        }
        {
            // a real sweep with min-age 0 and a 0 GiB bound still cannot remove the `keep` entry
            std::vector<std::string> a = {"evict", root_arg, root_val, "--max-mib", "6", "--min-age", "0",
                                          "--keep", "big-a"};
            std::vector<char*> av = mkargv(a);
            ::run_q4nx_cache(int(av.size()), av.data());
            check(fs::exists(cr / "big-a"), "q4nx-cache evict: the --keep entry is never removed");
            check(!fs::exists(cr / "big-b"), "q4nx-cache evict: the least-recently-used entry goes first");
        }
        {
            std::vector<std::string> a = {"evict", "--nonsense"};
            std::vector<char*> av = mkargv(a);
            check(::run_q4nx_cache(int(av.size()), av.data()) == 2,
                  "q4nx-cache evict: an unknown argument is usage, exit 2");
        }
        fs::remove_all(cr, ec);
    }
    fs::remove_all(root, ec);
    std::printf("q4nx cache: %d arm(s) over the sweep (bound, LRU order, keep, age window, disabled, touch on a hit, bound text)\n",
                failures ? 0 : 26);
    return failures ? 1 : 0;
}
