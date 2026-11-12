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

int64_t q4nx_dir_bytes(const std::string& path) {
    std::error_code ec;
    if (!fs::is_directory(path, ec)) return 0;
    int64_t total = 0;
    for (auto it = fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code fec;
        if (it->is_regular_file(fec)) total += int64_t(fs::file_size(it->path(), fec));
        if (fec) continue;
    }
    return total;
}

Q4nxCacheSweep q4nx_cache_evict(const std::string& root, const std::string& keep, int64_t max_bytes,
                                int64_t min_age_seconds) {
    Q4nxCacheSweep r;
    std::error_code ec;
    if (max_bytes <= 0 || !fs::is_directory(root, ec)) return r;
    struct Ent { std::string path; int64_t bytes; fs::file_time_type mtime; };
    std::vector<Ent> all;
    for (auto it = fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code tec;
        if (!it->is_directory(tec)) continue;
        Ent e;
        e.path = it->path().string();
        if (!keep.empty() && (e.path == keep || fs::path(e.path).filename() == fs::path(keep).filename())) continue;  // never the new one
        e.bytes = q4nx_dir_bytes(e.path);
        e.mtime = fs::last_write_time(it->path(), tec);
        if (tec) continue;
        r.bytes_before += e.bytes;
        all.push_back(std::move(e));
    }
    if (all.empty()) return r;
    r.bytes_after = r.bytes_before;
    const auto now = fs::file_time_type::clock::now();
    const auto min_age = std::chrono::seconds(min_age_seconds);
    std::vector<Ent*> candidates;      // LRU first; the age window protects an entry a serve may be reading
    for (auto& e : all)
        if (now - e.mtime > min_age) candidates.push_back(&e);
    std::sort(candidates.begin(), candidates.end(), [](const Ent* a, const Ent* b) { return a->mtime < b->mtime; });
    for (Ent* e : candidates) {
        if (r.bytes_after <= max_bytes) break;
        std::error_code rec;
        fs::remove_all(e->path, rec);
        if (rec) continue;
        r.bytes_after -= e->bytes;
        ++r.removed;
    }
    if (r.removed > 0) {
        r.reported = true;
        std::fprintf(stderr,
                     "note: q4nx cache: removed %d entr%s, %.1f GiB -> %.1f GiB (bound %.1f GiB, %s); entries younger than "
                     "%lld s and the one just written are kept (engine#1956)\n",
                     r.removed, r.removed == 1 ? "y" : "ies", double(r.bytes_before) / (1 << 30),
                     double(r.bytes_after) / (1 << 30), double(max_bytes) / (1 << 30), root.c_str(),
                     (long long)min_age_seconds);
    }
    return r;
}

int64_t q4nx_cache_bound_bytes(const char* text, int64_t default_gib) {
    const int64_t max_gib = INT64_MAX >> 30;      // the largest GiB count whose byte count still fits
    long long gib = default_gib;
    if (text && *text) {
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(text, &end, 10);
        if (end != text) gib = v;                 // a number (ERANGE saturates, which the clamp below handles)
    }
    if (gib <= 0) return 0;
    if (gib > max_gib) gib = max_gib;
    return int64_t(gib) << 30;
}

void q4nx_cache_touch(const std::string& entry) {
    std::error_code ec;
    if (fs::is_directory(entry, ec)) fs::last_write_time(entry, fs::file_time_type::clock::now(), ec);
}


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
        std::printf("q4nx-cache: %s is %.1f GiB, the sweep is off (bound 0)\n", root.c_str(), double(before) / (1 << 30));
        return 0;
    }
    if (dry) {
        std::printf("q4nx-cache: %s is %.1f GiB against a %.1f GiB bound; --dry-run removes nothing\n", root.c_str(),
                    double(before) / (1 << 30), double(max_bytes) / (1 << 30));
        return before > max_bytes ? 1 : 0;
    }
    const Q4nxCacheSweep s = q4nx_cache_evict(root, keep, max_bytes, min_age);
    // A 6 MiB bound printed as "0.0 GiB" reads like "the sweep is off", which is a different state -- so small sizes
    // are reported in MiB. This line is the whole output; it has to be unambiguous.
    const auto size = [](int64_t b, char* out, size_t n) {
        if (b >= (int64_t(1) << 30)) std::snprintf(out, n, "%.1f GiB", double(b) / (1 << 30));
        else std::snprintf(out, n, "%.1f MiB", double(b) / (1 << 20));
    };
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
