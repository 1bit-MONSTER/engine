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
//
// tools/whisper_npu_wcache.h -- the device route's packed weights, kept on disk between runs (engine#1783).
//
// The first window of a one-shot transcription packs every encoder weight for the device (per-column int8 scales, the
// blocked tile layout) and, before that, loads every f32 weight from the pack. A later run of the same model with the
// same kernel set produces the same bytes, so this file keeps them: the next run reads the images straight into the
// device buffers, NpuAudioGemm marks the weights resident, and the executor never loads their f32 copies.
//
// Exactness: an image is the bytes the target's own pack wrote, read back from the device buffer after the pack, plus the
// per-column scales the dequant reads. Nothing else enters a launch, so the encoder output is the same byte for byte
// (the encode-only "output fnv" checks it).
//
// Identity: a file serves only the run whose identity string matches the one stored in it. The string holds:
//  * the format version (kWcacheVersion) and the launch rows;
//  * the encoder graph file itself: its path, size and a CONTENT hash of every byte, which covers the weights the
//    pack reads, because whisper's exported graph holds them inline;
//  * every regular file of the kernel directory, by name, size and content hash;
//  * name, size, mtime and ctime of every other file in the graph's directory (external weight data and the like).
// A content hash, not metadata alone: `cp -a`, `rsync -a`, `touch -r` or an unpacked tarball replace a file with new
// bytes at the same name, size and mtime (engine#3036 review; llama.cpp#449 is the same class). ctime is in the listing
// because a copy cannot preserve it. The hash is a word-wise multiply-xorshift over fixed 8 MiB chunks, combined in
// order, so it does not depend on the thread count (about 40 ms for whisper-small's 354 MB graph on 16 threads).
// Two graphs in one directory therefore never share a file, and a file whose identity does not match is refused, never
// restored. A file is written to "<name>.tmp.<pid>" with mode 0600, never through a symlink, and renamed into place, so
// a reader never sees a partial one. Each write keeps the kKeep most recently written cache files in the directory and
// removes the rest, so a kernel-set or model change does not leave the directory growing.
//
// Where: $ONEBIT_WHISPER_NPU_WCACHE when it names a directory, "0" turns it off; otherwise
// $XDG_CACHE_HOME/onebit/npu-wcache, else $HOME/.cache/onebit/npu-wcache.

#pragma once

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>

#include <thread>

#include "audio_npu_gemm.h"
#include "file_create.h"

namespace whisper_npu {

inline constexpr char kWcacheMagic[8] = {'O', 'N', 'E', 'B', 'I', 'T', 'W', 'C'};
// Bump when anything changes the bytes a given model + kernel set packs to: the packed layout, the packing arithmetic
// (scales, rounding), or what an entry stores. The identity cannot see code changes, only files.
inline constexpr uint32_t kWcacheVersion = 2;
inline constexpr size_t kKeep = 4;   // cache files kept per directory

// The cache directory, or "" when turned off or there is no home to put it in.
inline std::string wcache_dir() {
    if (const char* e = std::getenv("ONEBIT_WHISPER_NPU_WCACHE")) {
        if (e[0] == '0' && e[1] == 0) return {};
        if (e[0] != 0) return e;
    }
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x != nullptr && x[0] == '/') return std::string(x) + "/onebit/npu-wcache";
    if (const char* h = std::getenv("HOME"); h != nullptr && h[0] == '/') return std::string(h) + "/.cache/onebit/npu-wcache";
    return {};
}

inline bool mkdirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i < path.size() && path[i] != '/') continue;
        const std::string p = path.substr(0, i);
        if (::mkdir(p.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }
    return true;
}

// A content hash of `path`: 64-bit multiply-xorshift over each fixed 8 MiB chunk, the chunks combined in order. The
// result does not depend on `threads`. "" when the file cannot be read.
inline std::string content_hash(const std::string& path, unsigned threads = 0) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    struct stat st {};
    if (::fstat(fd, &st) != 0) { ::close(fd); return {}; }
    const size_t n = size_t(st.st_size);
    const uint8_t* p = nullptr;
    void* m = n > 0 ? ::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
    ::close(fd);
    if (n > 0 && m == MAP_FAILED) return {};
    p = static_cast<const uint8_t*>(m);
    constexpr size_t kChunk = size_t(8) << 20;
    const size_t chunks = (n + kChunk - 1) / kChunk;
    std::vector<uint64_t> part(chunks, 0);
    auto hash_chunk = [&](size_t c) {
        const size_t a = c * kChunk, b = std::min(n, a + kChunk);
        uint64_t h = 0x9e3779b97f4a7c15ULL ^ uint64_t(c);
        size_t i = a;
        for (; i + 8 <= b; i += 8) {
            uint64_t w;
            std::memcpy(&w, p + i, 8);
            h ^= w;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
        }
        for (; i < b; ++i) { h ^= p[i]; h *= 0xc4ceb9fe1a85ec53ULL; }
        part[c] = h ^ (uint64_t(b - a) << 1);
    };
    if (threads == 0) threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads && t < chunks; ++t)
        pool.emplace_back([&, t] { for (size_t c = t; c < chunks; c += threads) hash_chunk(c); });
    for (auto& th : pool) th.join();
    if (n > 0) ::munmap(m, n);
    uint64_t h = 1469598103934665603ULL ^ uint64_t(n);
    for (uint64_t x : part) { h ^= x; h *= 1099511628211ULL; h ^= h >> 29; }
    char out[24];
    std::snprintf(out, sizeof out, "%016llx", (unsigned long long)h);
    return out;
}

// name, size, mtime and ctime of every regular file in `dir`, sorted by name; with `content`, a content hash in place of
// the times
inline std::string dir_identity(const std::string& dir, bool content = false) {
    std::vector<std::string> lines;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            struct stat st {};
            const std::string f = dir + "/" + e->d_name;
            if (::stat(f.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            std::string l = std::string(e->d_name) + " " + std::to_string((long long)st.st_size) + " ";
            if (content) {
                l += content_hash(f);
            } else {
                l += std::to_string((long long)st.st_mtim.tv_sec) + "." + std::to_string((long long)st.st_mtim.tv_nsec) + " " +
                     std::to_string((long long)st.st_ctim.tv_sec) + "." + std::to_string((long long)st.st_ctim.tv_nsec);
            }
            lines.push_back(l);
        }
        closedir(d);
    }
    std::sort(lines.begin(), lines.end());
    std::string out = dir + "\n";
    for (const auto& l : lines) out += l + "\n";
    return out;
}

inline std::string wcache_identity(const std::string& encoder_graph, const std::string& kdir, int rows) {
    const size_t slash = encoder_graph.find_last_of('/');
    const std::string gdir = slash == std::string::npos ? "." : encoder_graph.substr(0, slash);
    struct stat st {};
    const long long gsize = ::stat(encoder_graph.c_str(), &st) == 0 ? (long long)st.st_size : -1;
    const std::string gh = content_hash(encoder_graph);
    return "onebit-npu-wcache v" + std::to_string(kWcacheVersion) + " rows " + std::to_string(rows) + "\ngraph " +
           encoder_graph + " " + std::to_string(gsize) + " " + (gh.empty() ? std::string("unreadable") : gh) +
           "\nkernels " + dir_identity(kdir, /*content=*/true) + "graph dir " + dir_identity(gdir);
}

// Keeps the kKeep most recently modified *.wc files of `dir`, removing the rest.
inline void wcache_prune(const std::string& dir) {
    std::vector<std::pair<long long, std::string>> files;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string nm = e->d_name;
            if (nm.size() < 4 || nm.compare(nm.size() - 3, 3, ".wc") != 0) continue;
            struct stat st {};
            const std::string f = dir + "/" + nm;
            if (::lstat(f.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            files.push_back({(long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec, f});
        }
        closedir(d);
    }
    std::sort(files.rbegin(), files.rend());
    for (size_t i = kKeep; i < files.size(); ++i) std::remove(files[i].second.c_str());
}

inline std::string wcache_path(const std::string& dir, const std::string& identity) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : identity) h = (h ^ c) * 1099511628211ULL;
    char name[40];
    std::snprintf(name, sizeof name, "/%016llx.wc", (unsigned long long)h);
    return dir + name;
}

template <class T>
inline bool rd(std::FILE* f, T* v) { return std::fread(v, sizeof(T), 1, f) == 1; }
template <class T>
inline bool wr(std::FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }

// Restores every weight the file holds into `gm`. Returns the number restored (0 when there is no matching file); stops at
// the first entry that does not fit, and those weights are then packed as usual.
inline int wcache_load(const std::string& path, const std::string& identity, onebit::audio::NpuAudioGemm& gm, std::string* why) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return 0;
    int n = 0;
    char magic[8];
    uint32_t ver = 0, idlen = 0, count = 0;
    std::string id;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kWcacheMagic, 8) != 0 || !rd(f, &ver) || ver != kWcacheVersion ||
        !rd(f, &idlen) || idlen > (1u << 20)) {
        if (why) *why = path + ": not a weight cache of this version";
        std::fclose(f);
        return 0;
    }
    id.resize(idlen);
    if (std::fread(id.data(), 1, idlen, f) != idlen || id != identity || !rd(f, &count)) {
        if (why) *why = path + ": written for another model, kernel set or row count";
        std::fclose(f);
        return 0;
    }
    std::vector<float> sc;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t klen = 0, ns = 0;
        int32_t role = 0, k = 0, nn = 0;
        uint64_t bytes = 0;
        std::string key;
        if (!rd(f, &klen) || klen > 4096) break;
        key.resize(klen);
        if (std::fread(key.data(), 1, klen, f) != klen || !rd(f, &role) || !rd(f, &k) || !rd(f, &nn) || !rd(f, &bytes) ||
            !rd(f, &ns) || ns > (1u << 24))
            break;
        sc.resize(ns);
        if (std::fread(sc.data(), sizeof(float), ns, f) != ns) break;
        std::string w;
        const bool ok = gm.import_weight(
            key, role, k, nn, size_t(bytes), [&](void* dst) { return std::fread(dst, 1, size_t(bytes), f) == size_t(bytes); }, sc,
            &w);
        if (!ok) {
            if (why) *why = w;
            break;
        }
        ++n;
    }
    std::fclose(f);
    return n;
}

// Writes every weight `gm` holds. False + why on an I/O failure (the run itself is unaffected).
inline bool wcache_save(const std::string& dir, const std::string& path, const std::string& identity,
                        const onebit::audio::NpuAudioGemm& gm, int* written, std::string* why) {
    if (!mkdirs(dir)) {
        if (why) *why = "cannot create " + dir;
        return false;
    }
    const std::string tmp = path + ".tmp." + std::to_string((long long)::getpid());
    std::FILE* f = onebit_fopen_private_wb(tmp.c_str());
    if (f == nullptr) {
        if (why) *why = "cannot write " + tmp;
        return false;
    }
    bool ok = std::fwrite(kWcacheMagic, 1, 8, f) == 8 && wr(f, kWcacheVersion) && wr(f, uint32_t(identity.size())) &&
              std::fwrite(identity.data(), 1, identity.size(), f) == identity.size();
    const long count_at = std::ftell(f);
    ok = ok && wr(f, uint32_t(0));
    uint32_t count = 0;
    gm.export_weights([&](const std::string& key, int role, int k, int n, const std::vector<uint8_t>& img,
                          const std::vector<float>& sc) {
        if (!ok) return;
        ok = wr(f, uint32_t(key.size())) && std::fwrite(key.data(), 1, key.size(), f) == key.size() && wr(f, int32_t(role)) &&
             wr(f, int32_t(k)) && wr(f, int32_t(n)) && wr(f, uint64_t(img.size())) && wr(f, uint32_t(sc.size())) &&
             std::fwrite(sc.data(), sizeof(float), sc.size(), f) == sc.size() &&
             std::fwrite(img.data(), 1, img.size(), f) == img.size();
        ++count;
    });
    ok = ok && std::fseek(f, count_at, SEEK_SET) == 0 && wr(f, count);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        if (why) *why = "write failed: " + path;
        return false;
    }
    if (written) *written = int(count);
    wcache_prune(dir);
    return true;
}

}  // namespace whisper_npu
