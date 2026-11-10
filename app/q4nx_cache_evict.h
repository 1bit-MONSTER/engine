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

// engine#1956: the Q4NX container cache is a bounded product store, not an unbounded log.
//
// `~/.cache/1bit/q4nx` holds one repacked model directory per GGUF stem (~3.5 G each) so a second serve
// skips a ~70 s import. Nothing ever removed an old one: the only `remove_all` calls on that path rebuild
// ONE entry (app/serve.cpp:3006/3037/3094/3101), so on 2026-11-10 it held **104 G across 29 entries** while
// the root filesystem had 119 G free -- ~34 more imports from a full disk.
//
// An age rule would be wrong here even though #1797's 6 h sweep is right for the `.fconst` scratch: those
// files are scratch deleted on success, these are PRODUCTS whose whole value is being hit later. So the
// policy is a cap with **eviction on write**, oldest first: the lane that writes the entry that overflows
// pays for it, and never a lane that is merely serving. The planner is here, pure and header-only, so the
// policy can be tested without a cache, a converter or an NPU.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace onebit {

struct Q4nxCacheEntry {
    std::string name;      // the directory name under the cache root (the GGUF stem)
    uint64_t    bytes = 0;  // its size on disk, recursively
    int64_t     mtime = 0;  // its newest write, as seconds; older entries go first
};

// Which entries to remove so the cache stays within `cap_bytes`, newest kept, `keep` never removed.
//
// `keep` is the entry the caller has JUST written. It is excluded from the candidates by design: removing
// it would re-cost the import that just paid for it, so an entry larger than the whole cap stays and the
// rest go (best effort) rather than the caller deleting its own work.
inline std::vector<std::string> q4nx_cache_evict_plan(const std::vector<Q4nxCacheEntry>& entries,
                                                     uint64_t cap_bytes, const std::string& keep) {
    uint64_t total = 0;
    std::vector<const Q4nxCacheEntry*> candidates;
    candidates.reserve(entries.size());
    for (const Q4nxCacheEntry& e : entries) {
        total += e.bytes;
        if (e.name != keep) candidates.push_back(&e);
    }
    // Oldest first, then by name so two entries written in the same second evict deterministically.
    std::sort(candidates.begin(), candidates.end(), [](const Q4nxCacheEntry* a, const Q4nxCacheEntry* b) {
        if (a->mtime != b->mtime) return a->mtime < b->mtime;
        return a->name < b->name;
    });
    std::vector<std::string> out;
    for (const Q4nxCacheEntry* e : candidates) {
        if (total <= cap_bytes) break;
        out.push_back(e->name);
        total -= e->bytes;
    }
    return out;
}

}  // namespace onebit
