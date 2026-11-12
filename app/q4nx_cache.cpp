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
#include "q4nx_cache.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

int run_q4nx_cache(int argc, char** argv) {
    const auto usage = [] {
        std::fprintf(stderr,
                     "usage: 1bit q4nx-cache evict [--root DIR] [--max-gib N | --max-mib N] [--keep NAME] [--min-age S]\n"
                     "                              [--dry-run]\n"
                     "  Sweeps the container cache the way a serve does on start: least-recently-used first, never the\n"
                     "  entry named by --keep, never one written within --min-age seconds (a serve may be reading it).\n"
                     "  --root defaults to ONEBIT_Q4NX_CACHE_DIR, then $XDG_CACHE_HOME/1bit/q4nx, then ~/.cache/1bit/q4nx.\n"
                     "  --max-gib defaults to ONEBIT_Q4NX_CACHE_MAX_GIB(), then 60; 0 disables the sweep.\n"
                     "  --dry-run prints what it would remove and removes nothing.\n");
    };
    if (argc < 1 || std::string(argv[0]) != "evict") { usage(); return 2; }
    std::string root, keep;
    int64_t max_gib = -1, max_mib = -1, min_age = 600;
    bool dry = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "q4nx-cache: %s needs a value\n", what); return nullptr; }
            return argv[++i];
        };
        if (a == "--root") { const char* v = next("--root"); if (!v) return 2; root = v; }
        else if (a == "--keep") { const char* v = next("--keep"); if (!v) return 2; keep = v; }
        else if (a == "--max-gib") { const char* v = next("--max-gib"); if (!v) return 2; max_gib = std::atoll(v); }
        else if (a == "--max-mib") { const char* v = next("--max-mib"); if (!v) return 2; max_mib = std::atoll(v); }
        else if (a == "--min-age") { const char* v = next("--min-age"); if (!v) return 2; min_age = std::atoll(v); }
        else if (a == "--dry-run") dry = true;
        else { std::fprintf(stderr, "q4nx-cache: unknown argument %s\n", a.c_str()); usage(); return 2; }
    }
    if (root.empty()) {
        const char* env = std::getenv("ONEBIT_Q4NX_CACHE_DIR");
        if (env && *env) {
            root = env;
        } else {
            const char* xdg = std::getenv("XDG_CACHE_HOME");
            const char* home = std::getenv("HOME");
            root = (xdg && *xdg) ? std::string(xdg) : std::string(home ? home : ".") + "/.cache";
            root += "/1bit/q4nx";
        }
    }
    const int64_t max_bytes = (max_mib >= 0)   ? max_mib * (int64_t(1) << 20)
                              : (max_gib >= 0) ? max_gib * (int64_t(1) << 30)
                                               : q4nx_cache_bound_bytes(std::getenv("ONEBIT_Q4NX_CACHE_MAX_GIB"));
    const int64_t before = q4nx_dir_bytes(root);
    if (max_bytes <= 0) {
        std::printf("q4nx-cache: %s is %s, the sweep is off (bound 0)\n", root.c_str(), q4nx_cache_size_text(before).c_str());
        return 0;
    }
    const auto size = [](int64_t b, char* out, size_t n) { std::snprintf(out, n, "%s", q4nx_cache_size_text(b).c_str()); };
    if (dry) {
        const std::vector<Q4nxCacheCandidate> would = q4nx_cache_candidates(root, keep, max_bytes, min_age);
        int64_t would_free = 0;
        for (const auto& c : would) would_free += c.bytes;
        char b1[32], b2[32], b3[32];
        size(before, b1, sizeof b1);
        size(max_bytes, b2, sizeof b2);
        size(would_free, b3, sizeof b3);
        std::printf("q4nx-cache: %s is %s against a %s bound; --dry-run removes nothing\n", root.c_str(), b1, b2);
        if (would.empty()) {
            std::printf("q4nx-cache: nothing would be removed (bound met, or every entry is inside the %lld s age window)\n",
                        (long long)min_age);
            return before > max_bytes ? 1 : 0;
        }
        std::printf("q4nx-cache: would remove %zu entr%s, %s, oldest first:\n", would.size(),
                    would.size() == 1 ? "y" : "ies", b3);
        for (const auto& c : would) std::printf("  %8.1f MiB  %s\n", double(c.bytes) / (1 << 20), c.name.c_str());
        return 1;   // over the bound: the dry run is telling you there IS work
    }
    const Q4nxCacheSweep s = q4nx_cache_evict(root, keep, max_bytes, min_age);
    // A 6 MiB bound printed as "0.0 GiB" reads like "the sweep is off", which is a different state -- so small sizes
    // are reported in MiB. This line is the whole output; it has to be unambiguous.
    char before_s[32], after_s[32], bound_s[32];
    size(s.bytes_before, before_s, sizeof before_s);
    size(s.bytes_after, after_s, sizeof after_s);
    size(max_bytes, bound_s, sizeof bound_s);
    std::printf("q4nx-cache: %d entr%s removed, %s -> %s (bound %s)\n", s.removed, s.removed == 1 ? "y" : "ies",
                before_s, after_s, bound_s);
    if (s.bytes_after > max_bytes) {
        std::fprintf(stderr,
                     "q4nx-cache: still over the bound -- every remaining entry was written within %lld s, so a serve may be "
                     "reading it; re-run with a larger --min-age once the box is idle\n",
                     (long long)min_age);
        return 1;
    }
    return 0;
}
