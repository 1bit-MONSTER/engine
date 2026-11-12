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
#include "q4nx_cache_sweep.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

std::string q4nx_cache_size_text(int64_t bytes) {
    char out[32];
    if (bytes >= (int64_t(1) << 30)) {
        std::snprintf(out, sizeof out, "%.1f GiB", double(bytes) / (1 << 30));
        return out;
    }
    std::snprintf(out, sizeof out, "%.1f MiB", double(bytes) / (1 << 20));
    // Within half a MiB of a GiB the MiB reading rounds to "1024.0", which is a value in the unit it has just left -- the same
    // kind of misleading exchange this formatter exists to stop. Print the GiB it is about to become.
    if (std::string(out) == "1024.0 MiB") std::snprintf(out, sizeof out, "%.1f GiB", double(bytes) / (1 << 30));
    return out;
}

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

std::vector<Q4nxCacheCandidate> q4nx_cache_candidates(const std::string& root, const std::string& keep,
                                                      int64_t max_bytes, int64_t min_age_seconds) {
    std::vector<Q4nxCacheCandidate> out;
    std::error_code ec;
    if (max_bytes <= 0 || !fs::is_directory(root, ec)) return out;
    struct Ent { std::string path; int64_t bytes; fs::file_time_type mtime; };
    std::vector<Ent> all;
    for (auto it = fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        Ent e;
        e.path = it->path().string();
        std::error_code tec;
        if (!it->is_directory(tec)) continue;
        e.bytes = q4nx_dir_bytes(e.path);
        if (!keep.empty() && (e.path == keep || fs::path(e.path).filename() == fs::path(keep).filename())) continue;
        e.mtime = fs::last_write_time(it->path(), tec);
        if (tec) continue;
        all.push_back(std::move(e));
    }
    if (all.empty()) return out;
    const auto now = fs::file_time_type::clock::now();
    const auto min_age = std::chrono::seconds(min_age_seconds);
    std::vector<Ent*> cands;
    for (auto& e : all)
        if (now - e.mtime > min_age) cands.push_back(&e);
    std::sort(cands.begin(), cands.end(), [](const Ent* a, const Ent* b) { return a->mtime < b->mtime; });
    int64_t after = 0;
    for (const auto& e : all) after += e.bytes;
    for (Ent* e : cands) {
        if (after <= max_bytes) break;
        out.push_back({fs::path(e->path).filename().string(), e->bytes});
        after -= e->bytes;
    }
    return out;
}

Q4nxCacheSweep q4nx_cache_evict(const std::string& root, const std::string& keep, int64_t max_bytes,
                                int64_t min_age_seconds) {
    Q4nxCacheSweep r;
    r.bytes_before = q4nx_dir_bytes(root);
    r.bytes_after = r.bytes_before;
    // One selection, shared with `--dry-run`: the list is what this function removes, in this order.
    const std::vector<Q4nxCacheCandidate> cands = q4nx_cache_candidates(root, keep, max_bytes, min_age_seconds);
    for (const auto& cand : cands) {
        std::error_code rec;
        fs::remove_all(fs::path(root) / cand.name, rec);
        if (rec) continue;
        r.bytes_after -= cand.bytes;
        ++r.removed;
    }
    if (r.removed > 0) {
        r.reported = true;
        std::fprintf(stderr,
                     "note: q4nx cache: removed %d entr%s, %s -> %s (bound %s, %s); entries younger than "
                     "%lld s and the one just written are kept (engine#1956)\n",
                     r.removed, r.removed == 1 ? "y" : "ies", q4nx_cache_size_text(r.bytes_before).c_str(),
                     q4nx_cache_size_text(r.bytes_after).c_str(), q4nx_cache_size_text(max_bytes).c_str(), root.c_str(),
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
