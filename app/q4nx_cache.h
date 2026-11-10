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
#pragma once

#include <cstdint>
#include <string>

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

// Directory bytes under `path` (0 when it does not exist).
int64_t q4nx_dir_bytes(const std::string& path);
