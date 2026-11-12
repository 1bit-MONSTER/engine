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

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

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

namespace {
struct HeadGroup { std::string path; int64_t bytes = 0; fs::file_time_type mtime{}; std::vector<std::string> files; };

// The key a head-image file belongs to: "headimg-<16 hex>" from "headimg-<16 hex>-t<tile>.q4hi" (or its ".tmp.<pid>" temp).
// Any other name is its own key, so a file this sweep does not recognise is still counted and still evictable.
std::string head_image_key(const std::string& name) {
    static const std::string pre = "headimg-";
    const size_t n = pre.size() + 16;
    if (name.size() > n && name.compare(0, pre.size(), pre) == 0 && name[n] == '-') return name.substr(0, n);
    return name;
}

// The groups of one head-images directory: bytes summed over a key's files, age = its newest file (a hit marks its tile used,
// npu/forward/src/q4nx_head_tile_cache.cpp), so a key a serve is still loading from is inside the age window.
std::vector<HeadGroup> head_image_groups(const fs::path& dir) {
    std::vector<HeadGroup> out;
    std::vector<std::string> keys;
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        const int64_t b = int64_t(fs::file_size(it->path(), fec));
        if (fec) continue;
        const auto t = fs::last_write_time(it->path(), fec);
        if (fec) continue;
        const std::string key = head_image_key(it->path().filename().string());
        size_t i = 0;
        while (i < keys.size() && keys[i] != key) ++i;
        if (i == keys.size()) {
            keys.push_back(key);
            HeadGroup g;
            g.path = (dir / key).string();
            g.mtime = t;
            out.push_back(std::move(g));
        }
        HeadGroup& g = out[i];
        g.bytes += b;
        if (t > g.mtime) g.mtime = t;
        g.files.push_back(it->path().string());
    }
    return out;
}
}  // namespace

std::vector<Q4nxCacheCandidate> q4nx_cache_candidates(const std::string& root, const std::string& keep,
                                                      int64_t max_bytes, int64_t min_age_seconds) {
    std::vector<Q4nxCacheCandidate> out;
    std::error_code ec;
    if (max_bytes <= 0 || !fs::is_directory(root, ec)) return out;
    struct Ent { std::string path; int64_t bytes; fs::file_time_type mtime; std::vector<std::string> files; };
    std::vector<Ent> all;
    for (auto it = fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        Ent e;
        e.path = it->path().string();
        std::error_code tec;
        if (!it->is_directory(tec)) continue;
        if (it->path().filename() == kQ4nxHeadImagesDir) {
            // engine#2263: one entry per head-image key, not one for the directory. As one entry the sweep could only
            // keep every pack's head or remove all of them (16 G in 36 keys on strixhalo 11-12, most unread for a day).
            for (auto& g : head_image_groups(it->path())) all.push_back({g.path, g.bytes, g.mtime, std::move(g.files)});
            continue;
        }
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
        // A head-image key is named by its path under the root ("head-images/headimg-<digest>"), a directory entry by
        // its own name, so `--dry-run` and the sweep's line say which one it is.
        Q4nxCacheCandidate c;
        c.name = fs::path(e->path).lexically_relative(root).string();
        if (c.name.empty() || c.name == ".") c.name = fs::path(e->path).filename().string();
        c.bytes = e->bytes;
        c.files = e->files;
        out.push_back(std::move(c));
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
        if (!cand.files.empty()) {
            // A head-image key: its tile files (and any temp a crashed writer left under the same key), not the directory.
            bool any = false;
            for (const auto& f : cand.files) {
                std::error_code fec;
                if (fs::remove(f, fec)) any = true;
            }
            if (!any) continue;
        } else {
            fs::remove_all(fs::path(root) / cand.name, rec);
            if (rec) continue;
        }
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

std::string q4nx_cache_default_root() {
    if (const char* c = std::getenv("ONEBIT_Q4NX_CACHE"); c && *c) return c;
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/1bit/q4nx";
#ifndef _WIN32
    if (const struct passwd* pw = ::getpwuid(::geteuid()); pw && pw->pw_dir && *pw->pw_dir)
        return std::string(pw->pw_dir) + "/.cache/1bit/q4nx";
#endif
    return std::string();
}

Q4nxCacheSweep q4nx_cache_sweep_after_load() {
    const std::string root = q4nx_cache_default_root();
    if (root.empty()) return Q4nxCacheSweep{};
    return q4nx_cache_evict(root, std::string(), q4nx_cache_bound_bytes(std::getenv("ONEBIT_Q4NX_CACHE_MAX_GIB")));
}
