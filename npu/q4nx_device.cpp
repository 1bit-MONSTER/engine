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
#include "q4nx_device.h"

#include "file_map.h"
#include "model.h"
#include "q4nx.h"
#include "q4nx_cache_sweep.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <pwd.h>
#include <unistd.h>
#endif

namespace onebit::npu::q4nx {
namespace fs = std::filesystem;
using nlohmann::json;

std::string cache_root() {
    if (const char* c = std::getenv("ONEBIT_Q4NX_CACHE"); c && *c) return c;
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/1bit/q4nx";
#if !defined(_WIN32)
    // Same rule as app/serve.cpp's GGUF repack cache: never fall back to a world-writable /tmp
    // path a cached directory could have been planted in; ask the user database instead.
    if (const passwd* pw = ::getpwuid(::geteuid()); pw && pw->pw_dir && *pw->pw_dir)
        return std::string(pw->pw_dir) + "/.cache/1bit/q4nx";
#endif
    throw std::runtime_error("no home directory for the Q4NX cache: set HOME or ONEBIT_Q4NX_CACHE");
}

namespace {

// FNV-1a: the cache name must be the same across builds and runs (std::hash is not guaranteed).
uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex16(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf, 16);
}

// model.q4nx's header (npu/model.cpp's contract): the tensor table, the data base offset.
json read_table(const std::string& path, uint64_t& base) {
    std::error_code ec;
    const uint64_t size = uint64_t(fs::file_size(path, ec));
    if (ec) throw std::runtime_error("cannot stat " + path + ": " + ec.message());
    if (size < 8) throw std::runtime_error(path + ": truncated (shorter than the header length)");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n > size - 8) throw std::runtime_error(path + ": header past the end of the file");
    std::string header(size_t(n), '\0');
    f.read(header.data(), std::streamsize(n));
    if (!f) throw std::runtime_error(path + ": truncated header");
    base = 8 + n;
    try {
        const json t = json::parse(header);
        if (!t.is_object()) throw std::runtime_error("not a tensor table");
        return t;
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": unreadable tensor table: " + e.what());
    }
}

void write_table(const std::string& path, const json& table, const std::vector<uint8_t>& blob) {
    std::string header = table.dump();
    header.append((64 - header.size() % 64) % 64, ' ');
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint64_t n = header.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(header.data(), std::streamsize(header.size()));
    f.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
    if (!f) throw std::runtime_error("short write into " + path);
}

// What the device-form conversion changes: an I8 tensor whose last dimension is the imported
// fp32 q4_1f chunk (npu/q4nx.h, engine#1185).
bool is_q4_1f(const json& t) {
    if (!t.is_object() || t.value("dtype", std::string()) != "I8") return false;
    const auto it = t.find("shape");
    if (it == t.end() || !it->is_array() || it->empty()) return false;
    return it->back().is_number_integer() && it->back().get<int64_t>() == int64_t(kTileBytesF32);
}

uint64_t tiles_of(const json& shape) {
    uint64_t n = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        if (!shape[i].is_number_integer() || shape[i].get<int64_t>() <= 0)
            throw std::runtime_error("q4_1f tensor: unusable shape");
        const uint64_t v = uint64_t(shape[i].get<int64_t>());
        if (n > UINT64_MAX / v) throw std::runtime_error("q4_1f tensor: tile count overflows");
        n *= v;
    }
    return n;
}

long long mtime_of(const std::string& path) {
    std::error_code ec;
    const auto t = fs::last_write_time(path, ec);
    if (ec) throw std::runtime_error("cannot stat " + path + ": " + ec.message());
    return static_cast<long long>(t.time_since_epoch().count());
}

bool stamp_matches(const std::string& dir, const std::string& source, uint64_t bytes,
                    long long mtime) {
    std::ifstream f(dir + "/device_form.json");
    if (!f) return false;
    json s;
    try {
        f >> s;
    } catch (const std::exception&) {
        return false;
    }
    return s.value("format", 0) == 1 && s.value("source", std::string()) == source &&
           s.value("model_bytes", int64_t(-1)) == int64_t(bytes) &&
           s.value("model_mtime", int64_t(-1)) == mtime;
}

}  // namespace

DeviceForm device_form(const std::string& model_path, const std::string& cache_dir) {
    const fs::path source_file(model_path);
    const std::string model_dir =
        source_file.parent_path().empty() ? std::string(".") : source_file.parent_path().string();
    DeviceForm out;
    out.dir = model_dir;
    out.model = model_path;

    uint64_t base = 0;
    const json table = read_table(out.model, base);
    bool needed = false;
    for (const auto& [name, t] : table.items())
        if (name != "__metadata__" && is_q4_1f(t)) needed = true;
    if (!needed) return out;  // already the device's form: nothing written
    // Resolved here, not by the caller: a GGUF-derived container must not need HOME or the cache.
    const std::string root = cache_dir.empty() ? cache_root() : cache_dir;

    std::error_code ec;
    const fs::path canon = fs::weakly_canonical(source_file, ec);
    const std::string source = ec ? model_path : canon.string();
    uint64_t file_bytes = 0;
    {
        const uint64_t n = uint64_t(fs::file_size(out.model, ec));
        if (ec) throw std::runtime_error("cannot stat " + out.model + ": " + ec.message());
        file_bytes = n;
    }
    const long long mtime = mtime_of(out.model);
    const std::string tag = source_file.parent_path().filename().string();
    const std::string dir =
        (fs::path(root) / (tag + "-q4nx1-" + hex16(fnv1a(source)))).string();
    const std::string model = dir + "/model.q4nx";
    if (fs::exists(model) && stamp_matches(dir, source, file_bytes, mtime)) {
        out.dir = dir;
        out.model = model;
        out.converted = true;
        out.reused = true;
        // A reuse is a use. The cache's sweep (q4nx_cache_sweep.h) orders entries by this directory's mtime, which nothing
        // else moves after the conversion is written; the stamp compares the SOURCE's mtime, so this does not disturb it.
        q4nx_cache_touch(dir);
        std::fprintf(stderr, "1bit q4nx: reusing the device-form conversion of %s at %s\n",
                     model_path.c_str(), dir.c_str());
        return out;
    }
    fs::remove_all(dir, ec);

    // Build beside the target and rename, so a reader never sees a half-written model.q4nx.
#if !defined(_WIN32)
    const std::string tmp = dir + ".tmp." + std::to_string(::getpid());
#else
    const std::string tmp = dir + ".tmp";
#endif
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);
    // A build that throws (a malformed source tensor, a failed write) must not leave a partial
    // cache directory behind for the next load to trip over.
    struct CleanTmp {
        std::string p;
        bool armed = true;
        ~CleanTmp() {
            if (!armed) return;
            std::error_code e;
            fs::remove_all(p, e);
        }
    } clean{tmp};

    size_t size = 0;
    std::string why;
    const uint8_t* map = map_file(out.model, size, &why);
    if (!map) throw std::runtime_error(why);

    // The mapped copy is released on every path, including a malformed source tensor's throw.
    struct Unmap {
        const uint8_t* p;
        size_t n;
        ~Unmap() { unmap_file(p, n); }
    } unmap{map, size};

    json out_table = json::object();
    json meta = json::object();
    if (const auto it = table.find("__metadata__"); it != table.end() && it->is_object())
        meta = *it;
    const std::string source_layout = meta.value("layout", std::string());
    meta["layout"] = "q4nx-1";
    meta["parameters"] = "bf16 scale/zero per 32-column group (the device form of an imported fp32 "
                         "container: q4nx-2 repacked through q4nx::as_q4_1(), engine#1185)";
    meta["device_form"] = {{"source_layout", source_layout},
                           {"tile_bytes", int64_t(kTileBytes)},
                           {"note", "a cache artifact; the container on disk is unchanged"}};
    out_table["__metadata__"] = meta;

    std::vector<uint8_t> blob;
    long long tensors = 0, tiles = 0;
    uint64_t off = 0;
    for (const auto& [name, t] : table.items()) {
        if (name == "__metadata__") continue;
        if (!t.is_object() || !t.contains("data_offsets") || !t.contains("shape")) {
            out_table[name] = t;  // not a tensor entry: carry it over untouched
            continue;
        }
        const auto o = t.at("data_offsets").get<std::vector<uint64_t>>();
        if (o.size() != 2 || o[1] < o[0] || o[1] > size - base)
            throw std::runtime_error(out.model + ": bad data_offsets for " + name);
        const uint64_t len = o[1] - o[0];
        if (is_q4_1f(t)) {
            const auto shape = t.at("shape").get<std::vector<int64_t>>();
            const uint64_t n = tiles_of(t.at("shape"));
            if (len != n * uint64_t(kTileBytesF32))
                throw std::runtime_error(out.model + ": " + name + " declares " +
                                         std::to_string(len) + " B for " + std::to_string(n) +
                                         " q4_1f tiles");
            Tensor x;
            x.dtype = "I8";
            x.shape = shape;
            x.data = map + base + o[0];
            x.bytes = len;
            x.avail = size - base - o[0];
            const std::vector<uint8_t> conv = as_q4_1(x);
            if (conv.size() != n * uint64_t(kTileBytes))
                throw std::runtime_error(out.model + ": " + name + " repacked to " +
                                         std::to_string(conv.size()) + " B");
            blob.insert(blob.end(), conv.begin(), conv.end());
            std::vector<int64_t> dev_shape = shape;
            dev_shape.back() = int64_t(kTileBytes);
            out_table[name] = {{"dtype", "I8"},
                               {"shape", dev_shape},
                               {"data_offsets", {off, off + uint64_t(conv.size())}}};
            off += uint64_t(conv.size());
            ++tensors;
            tiles += static_cast<long long>(n);
        } else {
            blob.insert(blob.end(), map + base + o[0], map + base + o[1]);
            out_table[name] = {{"dtype", t.value("dtype", std::string())},
                               {"shape", t.at("shape")},
                               {"data_offsets", {off, off + len}}};
            off += len;
        }
    }
    unmap_file(map, size);
    unmap.p = nullptr;

    write_table(tmp + "/model.q4nx", out_table, blob);
    // config.json so the cache directory is a directory a Model (and tools/check_q4nx_dir.py) can
    // open on its own; the forward still reads the rest of the model directory from the source.
    if (fs::exists(fs::path(model_dir) / "config.json")) {
        fs::copy_file(fs::path(model_dir) / "config.json", fs::path(tmp) / "config.json",
                      fs::copy_options::overwrite_existing, ec);
        if (ec) throw std::runtime_error("cannot copy config.json: " + ec.message());
    }
    {
        std::ofstream f(tmp + "/device_form.json", std::ios::trunc);
        f << json{{"format", 1},
                  {"source", source},
                  {"model_bytes", int64_t(file_bytes)},
                  {"model_mtime", mtime},
                  {"tensors", tensors},
                  {"tiles", tiles},
                  {"tile_bytes", int64_t(kTileBytes)}}
                 .dump();
        if (!f) throw std::runtime_error("cannot write " + tmp + "/device_form.json");
    }
    fs::create_directories(root, ec);
    fs::rename(tmp, dir, ec);
    if (ec) {
        // Another serve won the race (or the cache is read-only): keep whatever is there only if
        // it is the same conversion, otherwise say why instead of serving something else.
        if (!(fs::exists(model) && stamp_matches(dir, source, file_bytes, mtime)))
            throw std::runtime_error("cannot publish the device-form conversion at " + dir + ": " +
                                     ec.message());
        out.reused = true;
    } else {
        clean.armed = false;  // renamed into place: there is no temporary directory left
        // engine#2263: this is a WRITER of the cache, so it bounds the cache like serve start does (app/serve.cpp). Before
        // this only serve start and `1bit q4nx-cache evict` ever swept, so a box that converted through the forward route
        // and never started a serve grew past the bound with no sweep at all. Same bound (ONEBIT_Q4NX_CACHE_MAX_GIB, 60 GiB
        // by default, 0 = no sweep), same age window, and never the entry just written.
        q4nx_cache_evict(root, dir, q4nx_cache_bound_bytes(std::getenv("ONEBIT_Q4NX_CACHE_MAX_GIB")));
    }
    out.dir = dir;
    out.model = model;
    out.converted = true;
    out.tensors = tensors;
    out.tiles = tiles;
    std::fprintf(stderr,
                 "1bit q4nx: %s is a q4nx-2 container (fp32 scale/zero, engine#1185); repacked "
                 "%lld I8 "
                 "tensor(s), %lld tile(s) to the device's 5,120-byte q4_1 at %s\n",
                 model_path.c_str(), tensors, tiles, dir.c_str());
    return out;
}

}  // namespace onebit::npu::q4nx
