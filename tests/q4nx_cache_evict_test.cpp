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

// The Q4NX cache cap's planner (engine#1956), device-free and cache-free: the policy is a pure function of
// (entry name, size, mtime), a cap, and the entry that was just written, so it is tested here rather than by
// filling a disk. The four cases below are the ones a real cache produces: under the cap, over it, an entry
// larger than the whole cap, and two entries written in the same second.
#include "q4nx_cache_evict.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using onebit::Q4nxCacheEntry;
using onebit::q4nx_cache_evict_plan;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    std::printf("%s   %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

int main() {
    const uint64_t G = 1024ull * 1024 * 1024;

    {   // under the cap: nothing is evicted, so a box serving one model never pays
        const std::vector<Q4nxCacheEntry> e = {{"a", 3 * G, 100}, {"b", 3 * G, 200}, {"c", 3 * G, 300}};
        const auto v = q4nx_cache_evict_plan(e, 40 * G, "c");
        check(v.empty(), "under the cap: nothing is evicted");
    }
    {   // over the cap: oldest first, and only until it fits
        const std::vector<Q4nxCacheEntry> e = {
            {"old", 3 * G, 100}, {"mid", 3 * G, 200}, {"keep", 3 * G, 300}};
        const auto v = q4nx_cache_evict_plan(e, 4 * G, "keep");
        check(v.size() == 2 && v[0] == "old" && v[1] == "mid",
              "over the cap: oldest first, only until it fits (" + (v.empty() ? std::string("none") : v[0]) + "…)");
        check(std::find(v.begin(), v.end(), "keep") == v.end(),
              "over the cap: the entry just written is never a victim");
    }
    {   // one entry larger than the whole cap: it stays, the rest go -- never the caller's own work
        const std::vector<Q4nxCacheEntry> e = {{"huge", 50 * G, 300}, {"old", 3 * G, 100}};
        const auto v = q4nx_cache_evict_plan(e, 40 * G, "huge");
        check(v.size() == 1 && v[0] == "old",
              "an entry larger than the cap: kept, and everything else goes (the importer does not delete what it "
              "just imported)");
    }
    {   // same second: the tie-break is by name, so the choice is reproducible rather than filesystem order
        const std::vector<Q4nxCacheEntry> e = {{"b", 3 * G, 100}, {"a", 3 * G, 100}, {"keep", 3 * G, 100}};
        const auto v = q4nx_cache_evict_plan(e, 4 * G, "keep");
        check(v.size() == 2 && v[0] == "a" && v[1] == "b",
              "same mtime: evicted by name, deterministically");
    }
    {   // the cap counted over ALL entries including the kept one -- a keep-only cache is not "free"
        const std::vector<Q4nxCacheEntry> e = {{"keep", 39 * G, 300}, {"x", 2 * G, 100}};
        const auto v = q4nx_cache_evict_plan(e, 40 * G, "keep");
        check(v.size() == 1 && v[0] == "x", "the kept entry counts toward the cap, so the overflow is still paid");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all q4nx cache eviction checks passed\n");
    return 0;
}
