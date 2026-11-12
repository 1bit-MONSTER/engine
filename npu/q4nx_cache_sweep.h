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

#include <cstdint>
#include <string>
#include <vector>

// engine#2263: the sweep lives here, below both of its callers, so the conversion that WRITES cache entries
// (npu/q4nx_device.cpp, device_form()) can bound the cache too. It used to sit in app/, where only serve start and the
// `1bit q4nx-cache` command could reach it, and npu/ cannot include app/. Built as its own small library
// (onebit_q4nx_cache) that `onebit`, onebit_npu and q4nx_cache_test link, so each binary carries one copy.
// The Q4NX container cache (engine#1956): app/serve.cpp repacks a GGUF into `$ONEBIT_Q4NX_CACHE` (default
// ~/.cache/1bit/q4nx) and reuses the result, and until this existed nothing ever removed an old entry. Measured on strixhalo
// 2026-11-10: 104 G in 29 entries, out of a 118 G cache, on a box whose root filesystem was down to 119 G -- which is why two
// lanes' build gates closed the same day. This is the same defect as #1770 -> #1797 one layer down (the .fconst scratch that
// only ever accumulated), with one difference that matters: this cache is SERVED FROM, so an entry may be in use.
struct Q4nxCacheSweep {
    int removed = 0;
    int64_t bytes_before = 0, bytes_after = 0;
    bool reported = false;   // true when a line was printed (i.e. something was removed)
};

// Remove least-recently-used entries under `root` until the total is at most `max_bytes`.
//
// Not evicted, ever: `keep` (the entry just written), and any entry whose mtime is within `min_age_seconds` -- a serve may be
// reading it right now, and this sweep cannot see other processes' open files. That is a heuristic, not a guarantee: a lane
// serving an entry older than the age window could still have it removed underneath it, which costs that lane a re-import and
// not a crash (the entry is regenerable from its GGUF). `max_bytes <= 0` disables the sweep entirely, which is what a lane that
// wants the old unbounded behaviour sets.
//
// Prints ONE line when it removes anything (never one per sweep, never one per entry): a lane that loses a 3.5 G container to
// the bound should see why in its log instead of re-importing into the same squeeze.
Q4nxCacheSweep q4nx_cache_evict(const std::string& root, const std::string& keep, int64_t max_bytes,
                               int64_t min_age_seconds = 600);

// engine#1788 follow-up, from USING the command: `--dry-run` reported totals only, so an operator could see "3 entries, 12 GiB"
// but not WHICH entries -- which is the question before deleting another lane's imports. This is the same selection
// q4nx_cache_evict() makes (LRU first, `keep` and the age window respected, stop at max_bytes), returned instead of applied,
// so the two cannot drift: evict() is now this list plus a remove loop.
struct Q4nxCacheCandidate {
    std::string name;    // the entry directory's name, as it appears under the root
    int64_t bytes = 0;
};
std::vector<Q4nxCacheCandidate> q4nx_cache_candidates(const std::string& root, const std::string& keep, int64_t max_bytes,
                                                      int64_t min_age_seconds = 600);

// Directory bytes under `path` (0 when it does not exist).
int64_t q4nx_dir_bytes(const std::string& path);

// The byte bound for ONEBIT_Q4NX_CACHE_MAX_GIB's value (`text` may be null). Unset, empty or not a number gives the default
// (`default_gib`); zero or negative gives 0, which q4nx_cache_evict() reads as "no sweep". The result never exceeds
// INT64_MAX: the plain `gib << 30` this replaces overflowed from 2^33 GiB up, so a very large number written to mean "no
// limit" could come out as a 1 GiB bound, and a typo ("sixty") silently turned the sweep off.
int64_t q4nx_cache_bound_bytes(const char* text, int64_t default_gib = 60);

// Mark `entry` (a directory in the cache) as used now. The sweep orders by the entry directory's mtime, which the filesystem
// only moves when a file is created or removed inside it -- so without this the order is by CREATION, and a container
// imported last week and served every day is evicted before yesterday's one-off. Call it on every cache hit. It also puts a
// just-opened entry inside the sweep's age window, which is what makes "a serve may be reading it" true for a reused entry
// and not only for a new one. Failure is ignored (a read-only cache still serves).
void q4nx_cache_touch(const std::string& entry);
