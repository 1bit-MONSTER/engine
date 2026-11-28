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

// The Loom GPU route (docs/loom-route.md): `1bit serve` serves a .gguf on the GPU through `loom-serve`
// (AMD's Loom programs on the HRX runtime) when config/loom-routes.json has an entry for that model, and
// through the llama.cpp fork's llama-server (the frozen fallback) when it has none. A model gets an entry
// only after it passed the Loom-vs-fallback gate (speed and accuracy, docs/loom-route.md "Gate"), so the
// shipped table starts empty and serve behaves exactly as before.
//
// Everything here is a pure function of its inputs (the table text, a reply body, a byte stream), so
// tests/loom_route_test.cpp checks it without a GPU or a server:
//   - LoomRoutes::parse / match_loom_route: the table, and which entry (if any) a model file matches;
//   - loom_ready: what loom-serve's GET /v1/status says about the card serve registered;
//   - loom_request: a llama-server-shaped request made acceptable to loom-serve, which rejects every
//     field it does not implement (HTTP 400) instead of ignoring it;
//   - loom_reply / LoomSse: loom-serve's replies made to look like llama-server's where clients read
//     them (the `model` name, a `timings` object Lemonade's telemetry reads, usage in the final chunk).
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace onebit::loom {

using json = nlohmann::json;

// ---------------------------------------------------------------------------------------------------
// SHA-256 over a stream (a GGUF is gigabytes; app/hrx_runtime.h's helper reads its input whole).

class Sha256 {
public:
    void update(const uint8_t* p, size_t n) {
        len_ += n;
        while (n > 0) {
            const size_t take = std::min(n, size_t(64) - fill_);
            std::copy(p, p + take, buf_ + fill_);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == 64) { block(buf_); fill_ = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = len_ * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill_ != 56) update(&zero, 1);
        uint8_t tail[8];
        for (int i = 0; i < 8; i++) tail[i] = uint8_t(bits >> (8 * (7 - i)));
        update(tail, 8);
        char out[65];
        for (int i = 0; i < 8; i++) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* m) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = uint32_t(m[4 * i]) << 24 | uint32_t(m[4 * i + 1]) << 16 | uint32_t(m[4 * i + 2]) << 8 | m[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }
    uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf_[64] = {};
    size_t fill_ = 0;
    uint64_t len_ = 0;
};

// sha256 of a whole file, read in 4 MiB pieces; "" when it cannot be read.
inline std::string sha256_of_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    Sha256 s;
    std::vector<char> buf(size_t(4) << 20);
    while (f) {
        f.read(buf.data(), std::streamsize(buf.size()));
        const std::streamsize n = f.gcount();
        if (n > 0) s.update(reinterpret_cast<const uint8_t*>(buf.data()), size_t(n));
    }
    if (f.bad()) return "";
    return s.hex();
}

// engine#3619: the verified sha256 of a checkpoint, cached so that a second load of the same unchanged file does not
// hash it again (17.66 GB took 124-170 s on every load). One file per checkpoint in `dir`, named <st_dev>-<st_ino>,
// holding "1 <size> <mtime_ns> <sha256> <unix time of the hash>". A cached digest is used only when the file's device,
// inode, size and mtime_ns (of the file itself, after symlinks) all equal the record's, so a rewrite, a touch, a
// rename-over or a copy is hashed again.
// The cache must not become a way to skip the hash: a record is trusted only when it and its directory are owned by
// this process's user and neither is group/other-writable, and neither is a symlink; anything else is ignored (hashed
// as before, not written). Writes are atomic (a temp file in the same directory, fsync, rename) and only happen when
// the file's identity is the same after hashing as before. `dir` empty = no cache.
struct FileIdentity {
    uint64_t dev = 0, ino = 0, size = 0;
    int64_t mtime_ns = 0;
    bool operator==(const FileIdentity& o) const {
        return dev == o.dev && ino == o.ino && size == o.size && mtime_ns == o.mtime_ns;
    }
};
inline std::optional<FileIdentity> file_identity(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return std::nullopt;
    return FileIdentity{uint64_t(st.st_dev), uint64_t(st.st_ino), uint64_t(st.st_size),
                        int64_t(st.st_mtim.tv_sec) * 1000000000 + int64_t(st.st_mtim.tv_nsec)};
}
// owned by us, a directory (or a regular file), no group/other write, not a symlink
inline bool sha_cache_entry_trusted(const std::string& path, bool want_dir) {
    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) return false;
    if (want_dir ? !S_ISDIR(st.st_mode) : !S_ISREG(st.st_mode)) return false;
    return st.st_uid == ::geteuid() && (st.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}
// ONEBIT_LOOM_SHA_CACHE > $XDG_STATE_HOME/1bit/loom-route-sha > $HOME/.local/state/1bit/loom-route-sha > "" (no cache)
inline std::string default_sha_cache_dir(const char* env, const char* xdg_state, const char* home) {
    if (env) return env;   // set but empty = caching off
    if (xdg_state && *xdg_state) return (std::filesystem::path(xdg_state) / "1bit" / "loom-route-sha").string();
    if (home && *home) return (std::filesystem::path(home) / ".local" / "state" / "1bit" / "loom-route-sha").string();
    return "";
}
struct CachedSha {
    std::string sha;        // "" when the file cannot be read
    bool cached = false;    // true: from a trusted record that matches the file's identity
    int64_t verified = 0;   // unix time of the hash the record holds (cached only)
    std::string note;       // why the cache was not used or not written, when that is worth a log line
};
inline CachedSha sha256_of_file_cached(const std::string& path, const std::string& dir,
                                       const std::function<std::string(const std::string&)>& hash = sha256_of_file) {
    CachedSha out;
    const std::optional<FileIdentity> id = file_identity(path);
    if (!id || dir.empty()) { out.sha = hash(path); return out; }
    const std::string rec = (std::filesystem::path(dir) / (std::to_string(id->dev) + "-" + std::to_string(id->ino))).string();
    const bool dir_ok = sha_cache_entry_trusted(dir, true);
    if (dir_ok && sha_cache_entry_trusted(rec, false)) {
        std::ifstream in(rec, std::ios::binary);
        int ver = 0;
        uint64_t size = 0;
        int64_t mtime = 0, when = 0;
        std::string sha;
        if (in >> ver >> size >> mtime >> sha >> when && ver == 1 && size == id->size && mtime == id->mtime_ns &&
            sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string::npos) {
            out.sha = sha;
            out.cached = true;
            out.verified = when;
            return out;
        }
    }
    out.sha = hash(path);
    if (out.sha.empty()) return out;
    if (!dir_ok) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) {
            std::filesystem::create_directories(dir, ec);
            if (!ec) ::chmod(dir.c_str(), 0700);
        }
        if (!sha_cache_entry_trusted(dir, true)) {
            out.note = "sha256 cache " + dir + " is not a directory of this user without group/other write: not cached";
            return out;
        }
    }
    if (const std::optional<FileIdentity> after = file_identity(path); !after || !(*after == *id)) {
        out.note = "the file changed while it was hashed: not cached";
        return out;
    }
    std::string tmp = rec + ".XXXXXX";
    const int fd = ::mkstemp(tmp.data());
    if (fd < 0) { out.note = "cannot write the sha256 cache in " + dir; return out; }
    const std::string line = "1 " + std::to_string(id->size) + " " + std::to_string(id->mtime_ns) + " " + out.sha + " " +
                             std::to_string(int64_t(::time(nullptr))) + "\n";
    bool ok = ::fchmod(fd, 0600) == 0 && ::write(fd, line.data(), line.size()) == ssize_t(line.size()) && ::fsync(fd) == 0;
    ok = ::close(fd) == 0 && ok;
    if (!ok || ::rename(tmp.c_str(), rec.c_str()) != 0) {
        ::unlink(tmp.c_str());
        out.note = "cannot write the sha256 cache in " + dir;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------
// The table (config/loom-routes.json, compiled in; `serve --loom-routes FILE` replaces it).

struct LoomRoute {
    std::string id;                    // a name for logs ("qwen3-0.6b-q4_k_m")
    std::vector<std::string> models;   // match.model: served ids / file names (serve's model id or the file name)
    std::string sha256;                // match.gguf_sha256: the exact checkpoint the gate measured
    uint64_t size = 0;                 // match.gguf_size: its size in bytes (checked first, so a miss costs no hash)
    std::string card;                  // the fetched Loom package: a model.json, or the directory holding it
    std::string preset;                // --preset=NAME, for a card with several presets
    std::string device;                // loom-serve --device URI (default amdgpu://)
    std::vector<std::string> args;     // more loom-serve flags, as written (--memory_bytes=..., ...)
    std::vector<std::string> env;      // NAME=VALUE for loom-serve's environment (a value the user set wins)
    std::string gate;                  // where the gate's result is recorded (an issue link); required
};

struct LoomRoutes {
    std::vector<LoomRoute> routes;

    bool parse(const std::string& text, std::string& err) {
        routes.clear();
        json j;
        try {
            j = json::parse(text);
        } catch (const std::exception& e) {
            err = std::string("not JSON: ") + e.what();
            return false;
        }
        if (!j.is_object() || !j.contains("routes") || !j["routes"].is_array()) {
            err = "expected an object with a \"routes\" array";
            return false;
        }
        static const char* kKeys[] = {"id", "match", "card", "preset", "device", "args", "env", "gate", "why", "measured"};
        static const char* kMatch[] = {"model", "gguf_sha256", "gguf_size"};
        for (size_t i = 0; i < j["routes"].size(); ++i) {
            const json& r = j["routes"][i];
            const std::string at = "routes[" + std::to_string(i) + "]";
            if (!r.is_object()) { err = at + ": expected an object"; return false; }
            for (const auto& [k, v] : r.items()) {
                (void)v;
                if (std::find(std::begin(kKeys), std::end(kKeys), k) == std::end(kKeys)) {
                    err = at + ": unknown key \"" + k + "\"";
                    return false;
                }
            }
            LoomRoute out;
            auto str = [&](const char* key, std::string& dst, bool required) {
                if (!r.contains(key)) {
                    if (required) err = at + ": \"" + key + "\" is required";
                    return !required;
                }
                if (!r[key].is_string() || r[key].get<std::string>().empty()) {
                    err = at + ": \"" + key + "\" must be a non-empty string";
                    return false;
                }
                dst = r[key].get<std::string>();
                return true;
            };
            auto strings = [&](const json& v, const std::string& name, std::vector<std::string>& dst) {
                if (!v.is_array()) { err = at + ": \"" + name + "\" must be a list of strings"; return false; }
                for (const auto& s : v) {
                    if (!s.is_string()) { err = at + ": \"" + name + "\" must be a list of strings"; return false; }
                    dst.push_back(s.get<std::string>());
                }
                return true;
            };
            if (!str("id", out.id, true) || !str("card", out.card, true) || !str("gate", out.gate, true) ||
                !str("preset", out.preset, true) || !str("device", out.device, false))
                return false;
            if (r.contains("args") && !strings(r["args"], "args", out.args)) return false;
            if (r.contains("env")) {
                if (!r["env"].is_object()) { err = at + ": \"env\" must be an object of strings"; return false; }
                for (const auto& [k, v] : r["env"].items()) {
                    if (!v.is_string()) { err = at + ": \"env\" must be an object of strings"; return false; }
                    out.env.push_back(k + "=" + v.get<std::string>());
                }
            }
            if (!r.contains("match") || !r["match"].is_object()) { err = at + ": \"match\" is required (an object)"; return false; }
            const json& m = r["match"];
            for (const auto& [k, v] : m.items()) {
                (void)v;
                if (std::find(std::begin(kMatch), std::end(kMatch), k) == std::end(kMatch)) {
                    err = at + ".match: unknown key \"" + k + "\"";
                    return false;
                }
            }
            if (m.contains("model") && !strings(m["model"], "match.model", out.models)) return false;
            if (m.contains("gguf_sha256")) {
                if (!m["gguf_sha256"].is_string()) { err = at + ".match.gguf_sha256 must be a string"; return false; }
                out.sha256 = m["gguf_sha256"].get<std::string>();
                std::transform(out.sha256.begin(), out.sha256.end(), out.sha256.begin(),
                               [](unsigned char c) { return char(std::tolower(c)); });
                if (out.sha256.size() != 64 || out.sha256.find_first_not_of("0123456789abcdef") != std::string::npos) {
                    err = at + ".match.gguf_sha256 is not a sha256 (64 hex)";
                    return false;
                }
                if (!m.contains("gguf_size") || !m["gguf_size"].is_number_unsigned() || m["gguf_size"].get<uint64_t>() == 0) {
                    err = at + ".match.gguf_sha256 needs gguf_size (bytes): the size is compared first, so a "
                               "model that is not this checkpoint is never hashed";
                    return false;
                }
                out.size = m["gguf_size"].get<uint64_t>();
            } else if (m.contains("gguf_size")) {
                err = at + ".match.gguf_size without gguf_sha256: a size alone does not name a checkpoint";
                return false;
            }
            if (out.models.empty() && out.sha256.empty()) {
                err = at + ".match names nothing: give \"model\" and/or \"gguf_sha256\" + \"gguf_size\"";
                return false;
            }
            routes.push_back(std::move(out));
        }
        return true;
    }
};

// The entry a model matches, or none. Every condition an entry names must hold: its model names (any
// one of them equals one of `names`) and its checkpoint (the file's size, then its sha256). `sha_of`
// is called at most once, and only for a file whose size equals an entry's, so an empty table or a
// table of other checkpoints never reads the model.
inline const LoomRoute* match_loom_route(const LoomRoutes& t, const std::string& model_path,
                                         const std::vector<std::string>& names,
                                         const std::function<std::string(const std::string&)>& sha_of,
                                         std::string* sha_out = nullptr) {
    std::optional<std::string> sha;
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(model_path, ec);
    for (const LoomRoute& r : t.routes) {
        if (!r.models.empty()) {
            bool named = false;
            for (const auto& m : r.models)
                for (const auto& n : names) named = named || (!n.empty() && m == n);
            if (!named) continue;
        }
        if (!r.sha256.empty()) {
            if (ec || size != r.size) continue;
            if (!sha) sha = sha_of(model_path);
            if (*sha != r.sha256) continue;
            if (sha_out) *sha_out = *sha;
        }
        return &r;
    }
    return nullptr;
}

// The card file an entry names: absolute as written, else under `cards_dir`; a directory means its model.json.
// The card path is checked before anything opens it (CodeQL cpp/path-injection: the cards directory
// can come from ONEBIT_LOOM_CARDS): a relative card may not climb out of the cards directory with
// "..", and the result must be a regular file named model.json. "" when the path is refused.
inline std::string loom_card_path(const LoomRoute& r, const std::string& cards_dir) {
    namespace fs = std::filesystem;
    fs::path card = fs::path(r.card).lexically_normal();
    if (card.empty()) return "";
    if (card.is_relative())
        for (const auto& part : card)
            if (part == "..") return "";
    fs::path p = card;
    if (p.is_relative() && !cards_dir.empty()) p = (fs::path(cards_dir) / p).lexically_normal();
    std::error_code ec;
    if (fs::is_directory(p, ec)) p /= "model.json";
    if (p.filename() != "model.json" || !fs::is_regular_file(p, ec)) return "";
    return p.string();
}

// The model id a card registers under (model.id): what a request's `model` must name. "" when the
// card cannot be read; `why` then says what is wrong.
inline std::string loom_card_model_id(const std::string& card, std::string& why) {
    std::ifstream in(card, std::ios::binary);
    if (!in) { why = "cannot read " + card; return ""; }
    try {
        const json j = json::parse(in);
        if (j.contains("model") && j["model"].is_object() && j["model"].contains("id") && j["model"]["id"].is_string() &&
            !j["model"]["id"].get<std::string>().empty())
            return j["model"]["id"].get<std::string>();
        why = card + " has no model.id";
    } catch (const std::exception& e) {
        why = card + " is not JSON: " + e.what();
    }
    return "";
}

// engine#3350: a card assembled on the appliance's data partition (1bit-loom-cards) names its weights in model.json
// (`parameter/<file>`, origin `@checkpoint/<file>`) and links them into Lemonade's Hugging Face cache - but which
// snapshot revision a pull lands in is only known after the pull (the catalogue pulls the repo's head: the Inkling card's
// lock says ea0c79b2, the head moved to 800c946d with the same files). The model serve was given IS that checkpoint (the
// route matched its sha256), so each such parameter/ entry that is a link, or missing, is pointed at the file of the same
// name beside the model. A regular file (a card fetched with --materialize=copy) is never touched, and an entry whose
// sibling is not there is left as it is. Returns how many links were (re)made; `why` names a failure.
//
// A sibling whose size differs from the card's own model.json.lock entry for that origin is NOT linked, and `refused`
// is set: the checkpoint repo changed under the card. Measured 23 Nov: unsloth/Inkling-2-Mini-GGUF added
// dspark-Inkling-2-Mini-Q8_0_Old.gguf "a copy of the current DSpark drafter (padded-vocab layout)", i.e. the drafter at
// the card's name is about to be replaced, and a catalogue pull takes the repo's head. The route's sha256 match covers
// the main model only; loom-fetch would check the lock, but the appliance links instead of fetching. Size only (no
// hashing at serve start): a different layout is a different size.
// What a person does about a companion origin's missing weights (engine#3562 review): on 1bit OS no command pulls a
// companion repository - Lemonade's pull fetches only the checkpoint - and 1bit-loom-companions (engine#3561) fetches
// it, pinned and sha256-checked, within 5 minutes of the model's pull; elsewhere the repository is pulled by hand.
// Pure: the caller says whether that tool is on this machine.
inline std::string companion_fix_hint(const std::string& repo, bool fetcher_on_box) {
    if (repo.empty()) return std::string();       // nothing to name, nothing to say (either branch)
    if (fetcher_on_box)                           // review cl-pragent: say WHICH companion in both branches
        return ": " + repo + " is fetched by this box (1bit-loom-companions, every 5 minutes once the model is pulled) - "
               "`1bit-loom-companions --status` says where it is, `--once` fetches it now";
    return ": pull " + repo;
}

inline int loom_link_card_parameters(const std::string& card_model_json, const std::string& model_path, std::string& why,
                                     bool* refused = nullptr) {
    namespace fs = std::filesystem;
    std::ifstream in(card_model_json, std::ios::binary);
    if (!in) { why = "cannot read " + card_model_json; return 0; }
    json j;
    try { j = json::parse(in); } catch (const std::exception& e) { why = card_model_json + " is not JSON: " + e.what(); return 0; }
    // engine#3528: the model's directory is made ABSOLUTE first - a relative `-m models/x.gguf` used to write the link
    // target relative to the cards directory ("-> models/x.gguf"), which dangles there
    std::error_code abs_ec;
    const fs::path model_abs = fs::absolute(fs::path(model_path), abs_ec);
    const fs::path card_dir = fs::path(card_model_json).parent_path(),
                   model_dir = (abs_ec ? fs::path(model_path) : model_abs).parent_path();
    std::map<std::string, std::uintmax_t> lock_size;   // origin -> size, from the card's own lock (absent lock: no check)
    std::map<std::string, std::string> lock_sha;        // origin -> sha256, same lock
    std::map<std::string, std::string> lock_rev;        // origin NAME (e.g. "mxfp8") -> the revision the lock pinned
    {
        std::ifstream lin(card_dir / "model.json.lock", std::ios::binary);
        if (lin) {
            try {
                const json lk = json::parse(lin);
                if (lk.contains("origins") && lk["origins"].is_object())   // origin name -> the revision the lock pinned
                    for (const auto& ov : lk["origins"].items()) {
                        const json& o = ov.value();
                        const json* rv = nullptr;
                        if (o.is_object() && o.contains("resolved") && o["resolved"].is_object() && o["resolved"].contains("revision"))
                            rv = &o["resolved"]["revision"];
                        else if (o.is_object() && o.contains("requested") && o["requested"].is_object() && o["requested"].contains("revision"))
                            rv = &o["requested"]["revision"];
                        if (rv && rv->is_string()) lock_rev[ov.key()] = rv->get<std::string>();
                    }
                if (lk.contains("files") && lk["files"].is_array())
                    for (const auto& f : lk["files"])
                        if (f.is_object() && f.contains("origin") && f["origin"].is_string() && f.contains("size") &&
                            f["size"].is_number_unsigned())
                        {
                            lock_size[f["origin"].get<std::string>()] = f["size"].get<std::uintmax_t>();
                            if (f.contains("sha256") && f["sha256"].is_string())
                                lock_sha[f["origin"].get<std::string>()] = f["sha256"].get<std::string>();
                        }
            } catch (const std::exception&) {
            }
        }
    }
    int made = 0;
    std::vector<const json*> stack = {&j};
    while (!stack.empty()) {
        const json* o = stack.back();
        stack.pop_back();
        if (o->is_array()) { for (const auto& v : *o) stack.push_back(&v); continue; }
        if (!o->is_object()) continue;
        for (const auto& kv : o->items()) stack.push_back(&kv.value());
        if (!o->contains("path") || !o->contains("origin") || !(*o)["path"].is_string() || !(*o)["origin"].is_string()) continue;
        const std::string origin = (*o)["origin"].get<std::string>(), rel = (*o)["path"].get<std::string>();
        const std::string pre = "@checkpoint/";
        if (origin.compare(0, pre.size(), pre) != 0) continue;
        const fs::path name = fs::path(origin.substr(pre.size())).filename(), link = (card_dir / rel).lexically_normal();
        bool climbs = false;
        for (const auto& part : fs::path(rel)) climbs = climbs || part == "..";
        if (name.empty() || climbs) continue;
        // the link must stay inside the card: an ABSOLUTE path replaces card_dir in `card_dir / rel` (std::filesystem), and
        // serve runs as root, so a card naming "/etc/resolv.conf" would have it deleted and re-pointed; a symlinked
        // directory inside the card would lead out the same way. A card is downloaded content, so neither is trusted.
        const fs::path relp(rel);
        if (relp.is_absolute() || relp.has_root_name() || relp.has_root_directory()) continue;
        {
            std::error_code cec;
            const fs::path base = fs::weakly_canonical(card_dir, cec), parent = fs::weakly_canonical(link.parent_path(), cec);
            if (cec) continue;
            const fs::path inside = parent.lexically_relative(base);
            if (inside.empty() || *inside.begin() == "..") continue;
        }
        std::error_code ec;
        fs::path target = model_dir / name;
        if (const auto it = lock_size.find(origin); it != lock_size.end()) {
            const std::uintmax_t want = it->second;
            auto fits = [&](const fs::path& p) {
                std::error_code e;
                return fs::is_regular_file(p, e) && fs::file_size(p, e) == want && !e;
            };
            // a link that already resolves to a file of the lock's size is kept: a normal start hashes nothing
            if (fs::is_symlink(fs::symlink_status(link, ec)) && fits(fs::canonical(link, ec)) && !ec) continue;
            if (!fits(target)) {
                // engine#3394 + orchestrator 23 Nov (option b): the file under the card's name is missing or another size
                // (unsloth keeps the old DSpark drafter as dspark-Inkling-2-Mini-Q8_0_Old.gguf and the Lemonade entry
                // pulls that name), so the lock's sha256 picks the file: every same-size file beside the model is hashed
                // once, when the link is made
                fs::path found;
                if (const auto sh = lock_sha.find(origin); sh != lock_sha.end())
                    for (const auto& de : fs::directory_iterator(model_dir, ec))
                        if (fits(de.path()) && sha256_of_file(de.path().string()) == sh->second) { found = de.path(); break; }
                if (found.empty()) {
                    // refuse only when a file of the card's name IS there with other bytes (the checkpoint changed under
                    // the card). A file that is simply absent is left as before: a card may list variants its preset never
                    // loads (loom-fun's Inkling r9700 card also names a UD-Q8_K_XL file), and loom-serve reports a missing
                    // file of the selected variant itself, by name, at registration.
                    if (fs::exists(target, ec)) {
                        why += std::string(why.empty() ? "" : "; ") + "no file beside the model has the size (" +
                               std::to_string(want) + ") and sha256 the card's lock gives for " + origin +
                               ": the checkpoint changed under the card, not linked";
                        if (refused) *refused = true;
                    }
                    continue;
                }
                target = found;
            }
        } else if (!fs::is_regular_file(target, ec)) {
            continue;
        }
        const fs::file_status st = fs::symlink_status(link, ec);
        if (fs::exists(st) && !fs::is_symlink(st)) continue;
        if (fs::is_symlink(st) && fs::read_symlink(link, ec) == target) continue;
        fs::create_directories(link.parent_path(), ec);
        fs::remove(link, ec);
        fs::create_symlink(target, link, ec);
        if (ec) {
            // a link the card needs and cannot get (a non-root shell and a root-owned cards dir, engine#3545 review): the
            // card is not usable as it stands, so serve takes the llama.cpp route instead of starting Loom on a stale link
            why += std::string(why.empty() ? "" : "; ") + "cannot link " + link.string() + " -> " + target.string() + ": " +
                   ec.message();
            if (refused) *refused = true;
            continue;
        }
        ++made;
    }
    // engine#3553 review: a card can bind weights from OTHER named origins than the checkpoint (the native Inkling card's
    // `@mxfp8/` experts, pulled from their own repo). serve does not relink those - boot or the catalogue pull links them -
    // but a missing one used to go unnoticed: Loom started and loom-serve failed at load, naming no file. In the variant(s)
    // that serve this model (a `@checkpoint/` file of the served file's name), every other named origin's parameter must
    // resolve to a regular file of the lock's size; otherwise the card is refused and the reason names the origin, the
    // size and the repository to pull. Unused variants are not checked (engine#3406), nor is a card no variant of which
    // names the served file.
    if (j.contains("variants") && j["variants"].is_object()) {
        const std::string served = fs::path(model_path).filename().string();
        auto files_of = [](const json& v, auto&& visit) {
            std::vector<const json*> st = {&v};
            while (!st.empty()) {
                const json* x = st.back();
                st.pop_back();
                if (x->is_array()) { for (const auto& e : *x) st.push_back(&e); continue; }
                if (!x->is_object()) continue;
                for (const auto& kv : x->items()) st.push_back(&kv.value());
                if (x->contains("path") && x->contains("origin") && (*x)["path"].is_string() && (*x)["origin"].is_string())
                    visit((*x)["path"].get<std::string>(), (*x)["origin"].get<std::string>());
            }
        };
        const std::string ck = "@checkpoint/";
        for (const auto& var : j["variants"].items()) {
            bool serves = false;
            files_of(var.value(), [&](const std::string&, const std::string& origin) {
                if (origin.compare(0, ck.size(), ck) == 0 && fs::path(origin.substr(ck.size())).filename().string() == served)
                    serves = true;
            });
            if (!serves) continue;
            files_of(var.value(), [&](const std::string& rel, const std::string& origin) {
                if (origin.empty() || origin[0] != '@' || origin.compare(0, ck.size(), ck) == 0) return;
                const auto slash = origin.find('/');
                if (slash == std::string::npos) return;
                std::error_code fec;
                const fs::path f = (card_dir / rel).lexically_normal();
                const auto want = lock_size.find(origin);
                auto sized_ok = [&](const fs::path& p) {
                    std::error_code e;
                    return fs::is_regular_file(p, e) && !e && (want == lock_size.end() || fs::file_size(p, e) == want->second) && !e;
                };
                const bool present = fs::is_regular_file(f, fec) && !fec;
                if (sized_ok(f)) return;
                std::string repo;
                const std::string oname = origin.substr(1, slash - 1);
                if (j.contains("origins") && j["origins"].is_object() && j["origins"].contains(oname) &&
                    j["origins"][oname].is_object() && j["origins"][oname].contains("repository") &&
                    j["origins"][oname]["repository"].is_string())
                    repo = j["origins"][oname]["repository"].get<std::string>();
                // engine#3553 review (cl-hf): boot links a named origin into the HF cache snapshot of the LOCK's revision, but
                // a pull by `main` lands in the snapshot of the repo's head (a README edit moves it). When the served model is
                // in an HF cache (<hub>/models--o--r/snapshots/<rev>/<file>), look for the origin's file in every snapshot of
                // its repo there: the lock's size picks it, its sha256 breaks a tie; the link is (re)made inside the card,
                // with the same containment rules as a checkpoint link.
                const fs::path name = fs::path(origin.substr(slash + 1)).filename(), relp(rel);
                bool contained = !name.empty() && !relp.is_absolute() && !relp.has_root_name() && !relp.has_root_directory();
                for (const auto& part : relp) contained = contained && part != "..";
                if (contained) {
                    std::error_code cec;
                    const fs::path base = fs::weakly_canonical(card_dir, cec), parent = fs::weakly_canonical(f.parent_path(), cec);
                    const fs::path inside = cec ? fs::path() : parent.lexically_relative(base);
                    contained = !cec && !inside.empty() && *inside.begin() != "..";
                }
                const fs::path snap_dir = model_dir.parent_path(), repo_dir = snap_dir.parent_path(), hub = repo_dir.parent_path();
                if (contained && !repo.empty() && snap_dir.filename() == "snapshots" &&
                    repo_dir.filename().string().rfind("models--", 0) == 0) {
                    std::string cache = repo;
                    for (std::string::size_type at = cache.find('/'); at != std::string::npos; at = cache.find('/', at + 2))
                        cache.replace(at, 1, "--");
                    // review (cl-release): size alone is a weak identity (a re-quantised companion keeps its shapes), so the
                    // lock's OWN revision snapshot is taken on its size; a file in ANY OTHER snapshot (a pull by `main`) only
                    // when its sha256 equals the lock's - one candidate or several - and never when the lock has no sha256
                    const fs::path snaps = hub / ("models--" + cache) / "snapshots";
                    fs::path pick;
                    // review of #3566 (cl-220): the lock's revision is joined into a path, so it must be ONE path component
                    // (an HF revision is a commit sha or a ref name): a revision such as "../../x" would link a same-size
                    // file from outside the HF cache. A revision that is not one component takes the sha256-only path below.
                    const auto one_component = [](const std::string& r) {
                        return !r.empty() && r != "." && r != ".." && r.find_first_of(std::string("/\\\0", 3)) == std::string::npos;
                    };
                    if (const auto rv = lock_rev.find(oname);
                        rv != lock_rev.end() && one_component(rv->second) && sized_ok(snaps / rv->second / name))
                        pick = snaps / rv->second / name;
                    else if (const auto sh = lock_sha.find(origin); sh != lock_sha.end()) {
                        std::error_code dec;
                        for (fs::directory_iterator it(snaps, dec), end; !dec && it != end; it.increment(dec))
                            if (sized_ok(it->path() / name) && sha256_of_file((it->path() / name).string()) == sh->second) {
                                pick = it->path() / name;
                                break;
                            }
                    }
                    if (!pick.empty()) {
                        std::error_code lec;
                        const fs::file_status st = fs::symlink_status(f, lec);
                        if (!(fs::exists(st) && !fs::is_symlink(st))) {   // a real file in parameter/ is never replaced
                            fs::create_directories(f.parent_path(), lec);
                            fs::remove(f, lec);
                            fs::create_symlink(fs::canonical(pick, lec), f, lec);
                            if (!lec) { ++made; return; }
                        }
                    }
                }
                why += std::string(why.empty() ? "" : "; ") + origin +
                       (want != lock_size.end() ? " (" + std::to_string(want->second) + " B)" : std::string()) +
                       (present ? " has another size than the card's lock" : " is not linked at " + f.string()) +
                       companion_fix_hint(repo, fs::exists("/usr/local/bin/1bit-loom-companions", fec));
                if (refused) *refused = true;
            });
        }
    }
    return made;
}

// ---------------------------------------------------------------------------------------------------
// Readiness: loom-serve answers /healthz and /v1/models as soon as it listens, with the card still
// preparing; GET /v1/status names every registration and its state (registered, preparing, ready,
// retiring, failed, release_failed). A failed card leaves the server running, so serve has to read
// the failure there instead of waiting out its load timeout.

enum class LoomReady { kWaiting, kReady, kFailed };

inline LoomReady loom_ready(const std::string& status_body, const std::string& model_id, std::string& why) {
    json j;
    try {
        j = json::parse(status_body);
    } catch (const std::exception&) {
        return LoomReady::kWaiting;
    }
    if (!j.is_object() || !j.contains("data") || !j["data"].is_array()) return LoomReady::kWaiting;
    for (const json& m : j["data"]) {
        if (!m.is_object() || m.value("id", "") != model_id || !m.contains("loom") || !m["loom"].is_object()) continue;
        const json& l = m["loom"];
        const std::string state = l.value("registration", "");
        if (state == "ready") return LoomReady::kReady;
        if (state == "failed" || state == "release_failed") {
            why = "loom-serve: " + model_id + " " + state;
            if (l.contains("error") && l["error"].is_object() && l["error"].contains("message") && l["error"]["message"].is_string())
                why += ": " + l["error"]["message"].get<std::string>().substr(0, 400);
            return LoomReady::kFailed;
        }
    }
    return LoomReady::kWaiting;
}

// ---------------------------------------------------------------------------------------------------
// Requests. What loom-serve does with a field it does not interpret changed between releases:
// hrx-loom 0.1.0 answers 400 for any field outside a fixed set (model, messages, stream, stream_options,
// max_tokens, max_completion_tokens, temperature, top_p, top_k, seed, tools, tool_choice, store,
// enable_thinking, reasoning_effort, preserve_thinking, chat_template_kwargs, model_options,
// media_io_kwargs); hrx-loom 0.2.0 accepts every field and hands it to the model program, which
// ignores what it does not consume -- `stop`, the penalties and `response_format` included. Under
// 0.1.0 an unknown field is an error, under 0.2.0 it is silently without effect; serve handles both
// the same way. A client written for llama-server sends three kinds of extra field:
//   - llama-server's own serving hints, which have no meaning on another server: dropped;
//   - llama-server's spelling of a field loom-serve has (n_predict): renamed;
//   - an OpenAI sampling/output field loom-serve does not implement: dropped when it holds the value
//     that means "off" (a client that sends every field with its default), refused with a 400 that
//     names it otherwise -- serving `stop` or a penalty as if it applied (0.2.0 accepts it, and a card
//     whose program does not consume it ignores it) would be a wrong answer.

inline const std::vector<std::string>& loom_dropped_hints() {
    static const std::vector<std::string> k = {"cache_prompt", "id_slot", "timings_per_token", "return_progress",
                                               "speculative.n_max", "speculative.n_min", "speculative.p_min"};
    return k;
}

struct LoomRequest {
    bool ok = true;
    std::string error;                // why serve refused it (HTTP 400), when !ok
    std::vector<std::string> dropped; // the fields removed, for the request line
    bool stream = false;
    bool client_usage = false;        // the client asked for stream_options.include_usage itself
};

inline LoomRequest loom_request(json& body, const std::string& loom_model) {
    LoomRequest r;
    if (!body.is_object()) {
        r.ok = false;
        r.error = "request body is not a JSON object";
        return r;
    }
    for (const auto& k : loom_dropped_hints())
        if (body.contains(k)) { body.erase(k); r.dropped.push_back(k); }
    if (body.contains("n_predict")) {
        const json v = body["n_predict"];
        body.erase("n_predict");
        if (!body.contains("max_tokens") && !body.contains("max_completion_tokens") && v.is_number_integer() &&
            v.get<long long>() > 0)
            body["max_tokens"] = v;
        r.dropped.push_back("n_predict");
    }
    // llama-server reads a negative output limit as "until the context is full"; loom-serve takes 1..16384 or its
    // deployment default, which is the nearest meaning
    for (const char* k : {"max_tokens", "max_completion_tokens"})
        if (body.contains(k) && body[k].is_number() && body[k].get<double>() < 0) { body.erase(k); r.dropped.push_back(k); }
    // OpenAI fields loom-serve does not implement, with the value that means "not used"
    auto neutral = [](const std::string& k, const json& v) {
        if (v.is_null()) return true;
        if (k == "n") return v.is_number() && v.get<double>() == 1;
        if (k == "presence_penalty" || k == "frequency_penalty" || k == "min_p")
            return v.is_number() && v.get<double>() == 0;
        if (k == "repeat_penalty" || k == "typical_p") return v.is_number() && v.get<double>() == 1;
        if (k == "logprobs") return v.is_boolean() && !v.get<bool>();
        if (k == "stop") return (v.is_array() && v.empty()) || (v.is_string() && v.get<std::string>().empty());
        if (k == "response_format") return v.is_object() && v.value("type", "") == "text";
        if (k == "user" || k == "parallel_tool_calls") return true;   // client metadata / a hint with no effect here
        return false;
    };
    for (const char* k : {"n", "presence_penalty", "frequency_penalty", "repeat_penalty", "min_p", "typical_p", "logprobs",
                          "top_logprobs", "stop", "response_format", "logit_bias", "grammar", "json_schema", "user",
                          "parallel_tool_calls"}) {
        if (!body.contains(k)) continue;
        if (neutral(k, body[k])) {
            body.erase(k);
            r.dropped.push_back(k);
            continue;
        }
        r.ok = false;
        r.error = std::string("\"") + k + "\" is not supported on the Loom route (loom-serve does not implement it); "
                  "drop it, or serve this model with --no-loom (docs/loom-route.md)";
        return r;
    }
    body["model"] = loom_model;
    r.stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
    if (r.stream) {
        // usage arrives in a final chunk only when asked for; serve always asks, so the reply can carry it
        // (and the timings) the way llama-server's last chunk does
        r.client_usage = body.contains("stream_options") && body["stream_options"].is_object() &&
                         body["stream_options"].value("include_usage", false);
        if (!body.contains("stream_options") || !body["stream_options"].is_object()) body["stream_options"] = json::object();
        body["stream_options"]["include_usage"] = true;
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------
// Replies. loom-serve reports no `timings`; Lemonade's telemetry reads prompt_n, predicted_n,
// prompt_ms, predicted_per_second and cache_n from it (streaming_proxy.cpp). serve fills the object
// from the reply's usage and from its own clock, and says so in `measured_by`: prompt_ms is the time
// from serve sending the request to the first generated text (it includes the HTTP hop), predicted_ms
// the time from there to the last chunk. A non-streamed reply has no first-token time, so it carries
// the counts only.

constexpr const char* kLoomMeasuredBy = "1bit serve: counts from loom-serve usage, times from serve's own clock";

inline json loom_timings(const json& usage, double first_ms, double end_ms) {
    json t = json::object();
    if (!usage.is_object()) return t;
    const long long prompt = usage.value("prompt_tokens", 0LL), completion = usage.value("completion_tokens", 0LL);
    long long cached = 0;
    if (usage.contains("prompt_tokens_details") && usage["prompt_tokens_details"].is_object())
        cached = usage["prompt_tokens_details"].value("cached_tokens", 0LL);
    t["prompt_n"] = prompt - cached;
    t["cache_n"] = cached;
    t["predicted_n"] = completion;
    if (first_ms >= 0 && end_ms >= first_ms) {
        t["prompt_ms"] = first_ms;
        t["predicted_ms"] = end_ms - first_ms;
        if (first_ms > 0) t["prompt_per_second"] = double(prompt - cached) * 1000.0 / first_ms;
        // the first token comes out of the prompt pass; the rest are decode steps
        if (completion > 1 && end_ms > first_ms) t["predicted_per_second"] = double(completion - 1) * 1000.0 / (end_ms - first_ms);
    }
    t["measured_by"] = kLoomMeasuredBy;
    return t;
}

// A non-streamed reply: the served id in `model`, the timings beside usage.
inline void loom_reply(json& out, const std::string& our_id) {
    if (!out.is_object()) return;
    if (out.contains("model")) out["model"] = our_id;
    if (out.contains("usage") && out["usage"].is_object() && !out.contains("timings"))
        out["timings"] = loom_timings(out["usage"], -1, -1);
}

// A streamed reply, as bytes in and bytes out. Each SSE event is rewritten: `model` becomes the served
// id; the chunk that carries finish_reason is held until the usage chunk serve asked for arrives, and
// leaves with usage and timings in it (llama-server's shape: one final chunk with both). A client that
// asked for include_usage itself gets the usage chunk as OpenAI sends it, with timings added. A reply
// that is not SSE (loom-serve answered an error as plain JSON) becomes one SSE error event and [DONE].
class LoomSse {
public:
    LoomSse(std::string our_id, bool client_usage, std::function<double()> now_ms)
        : id_(std::move(our_id)), client_usage_(client_usage), now_(std::move(now_ms)) {}

    std::string feed(const char* data, size_t n) {
        buf_.append(data, n);
        std::string out;
        if (!decided_) {
            const size_t first = buf_.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return out;
            raw_ = buf_[first] == '{';
            decided_ = true;
        }
        if (raw_) return out;   // a plain JSON body: kept whole for finish()
        for (size_t end; (end = event_end()) != std::string::npos;) {
            const size_t sep = buf_.compare(end, 4, "\r\n\r\n") == 0 ? 4 : 2;
            std::string ev = buf_.substr(0, end);
            buf_.erase(0, end + sep);
            out += event(ev);
        }
        return out;
    }

    std::string finish() {
        std::string out;
        if (raw_) {
            json err;
            try {
                err = json::parse(buf_);
            } catch (const std::exception&) {
                err = json{{"error", {{"message", buf_.substr(0, 400)}, {"type", "loom_output_error"}}}};
            }
            buf_.clear();
            return "data: " + err.dump() + "\n\ndata: [DONE]\n\n";
        }
        if (!buf_.empty()) { out += event(buf_); buf_.clear(); }
        if (held_) { out += emit(*held_); held_.reset(); }
        return out;
    }

    bool saw_done() const { return done_; }

private:
    size_t event_end() const {
        const size_t a = buf_.find("\n\n"), b = buf_.find("\r\n\r\n");
        return std::min(a, b);
    }
    static std::string emit(const json& j) { return "data: " + j.dump() + "\n\n"; }

    std::string event(const std::string& ev) {
        // the data lines of one event (usually one); other fields (event:, id:, comments) pass through
        std::string data, rest;
        size_t pos = 0;
        while (pos <= ev.size()) {
            size_t nl = ev.find('\n', pos);
            if (nl == std::string::npos) nl = ev.size();
            std::string line = ev.substr(pos, nl - pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data:", 0) == 0) {
                std::string v = line.substr(5);
                if (!v.empty() && v[0] == ' ') v.erase(0, 1);
                data += (data.empty() ? "" : "\n") + v;
            } else if (!line.empty()) {
                rest += line + "\n";
            }
            pos = nl + 1;
        }
        if (data.empty()) return rest.empty() ? "" : rest + "\n";
        if (data == "[DONE]") {
            done_ = true;
            std::string out;
            if (held_) { out += emit(*held_); held_.reset(); }
            return out + rest + "data: [DONE]\n\n";
        }
        json j;
        try {
            j = json::parse(data);
        } catch (const std::exception&) {
            return rest + "data: " + data + "\n\n";
        }
        if (!j.is_object()) return rest + "data: " + data + "\n\n";
        if (j.contains("model")) j["model"] = id_;
        const bool has_choices = j.contains("choices") && j["choices"].is_array() && !j["choices"].empty();
        if (has_choices && first_ms_ < 0) {
            const json& c = j["choices"][0];
            if (c.contains("delta") && c["delta"].is_object()) {
                const json& d = c["delta"];
                for (const char* k : {"content", "reasoning_content"})
                    if (d.contains(k) && d[k].is_string() && !d[k].get<std::string>().empty()) first_ms_ = now_();
                if (d.contains("tool_calls")) first_ms_ = now_();
            }
        }
        const bool usage_only = j.contains("usage") && j["usage"].is_object() && !has_choices;
        if (usage_only) {
            const json t = loom_timings(j["usage"], first_ms_, now_());
            if (client_usage_) {
                j["timings"] = t;
                std::string out;
                if (held_) { out += emit(*held_); held_.reset(); }
                return out + rest + emit(j);
            }
            if (held_) {
                json h = *held_;
                held_.reset();
                h["usage"] = j["usage"];
                h["timings"] = t;
                return rest + emit(h);
            }
            return rest;   // a usage chunk the client did not ask for, with no final chunk to carry it
        }
        if (has_choices && !client_usage_ && j["choices"][0].contains("finish_reason") &&
            !j["choices"][0]["finish_reason"].is_null()) {
            std::string out;
            if (held_) out += emit(*held_);
            held_ = j;
            return out + rest;
        }
        return rest + emit(j);
    }

    std::string id_;
    bool client_usage_;
    std::function<double()> now_;
    std::string buf_;
    bool decided_ = false, raw_ = false, done_ = false;
    double first_ms_ = -1;
    std::optional<json> held_;
};

// What the FIRST BYTES of an upstream reply say about its shape, decided without guessing. loom-serve answers a
// refused request as one small JSON object and a streamed answer as SSE, whose first line is `data:` (possibly after
// `event:`/`id:`/`:` lines). Deciding from a PARTIAL line is how a fragmented read turns a stream into a "refusal":
// `"da"` is a prefix of `data:` but matches none of the markers, so a two-way classifier has to pick one and gets it
// wrong -- and the cost is the exact bug the status pass exists to fix, since the answer is then handed to the client
// as an unreadable non-stream body (engine#3254 review). `NeedMore` is the third answer: the bytes are a prefix of a
// marker and the caller must wait for the rest of the line.
enum class StreamShape { Sse, NotSse, NeedMore };

inline bool loom_is_sse_marker_prefix(const std::string& piece) {
    for (const char* marker : {"data:", "event:", "id:", ":"})
        if (std::string(marker).compare(0, piece.size(), piece) == 0) return true;
    return false;
}

inline StreamShape loom_stream_shape(const std::string& bytes) {
    size_t pos = bytes.find_first_not_of(" \t\r\n");
    while (pos != std::string::npos) {
        const size_t eol = bytes.find('\n', pos);
        const bool complete = eol != std::string::npos;
        const std::string line = bytes.substr(pos, complete ? eol - pos : std::string::npos);
        if (line.rfind("data:", 0) == 0) return StreamShape::Sse;
        const bool other = line.rfind("event:", 0) == 0 || line.rfind("id:", 0) == 0 || line.rfind(":", 0) == 0;
        if (!complete) return other || loom_is_sse_marker_prefix(line) ? StreamShape::NeedMore : StreamShape::NotSse;
        if (!other) return StreamShape::NotSse;
        pos = bytes.find_first_not_of(" \t\r\n", eol);
    }
    // whitespace only so far: nothing has been said yet, so nothing can be decided
    return StreamShape::NeedMore;
}

// engine#3528: where serve looks when no flag and no environment says. A shell on the 1bit OS appliance has neither
// (10lemonade exports them for lemond only), so `1bit serve -m <model>` from a shell took the llama.cpp route for a
// model that has a Loom route. Pure: the caller passes what exists.
//   cards:  --loom-cards > ONEBIT_LOOM_CARDS > /data/loom-cards when it is a directory (the appliance's, assembled at
//           boot by 1bit-loom-cards) > $XDG_CACHE_HOME/1bit/loom-cards > $HOME/.cache/1bit/loom-cards > "".
//           /opt/1bit-package/loom-cards is NOT used: it is the read-only payload copy, without the parameter/ links.
//   server: --loom-serve > ONEBIT_LOOM_SERVE > <dir of this binary>/../loom/loom-serve when it exists (the payload's
//           layout: /opt/1bit-package/1bit/1bit and /opt/1bit-package/loom/loom-serve) > "loom-serve" (PATH).
inline std::string default_cards_dir(const std::string& flag, const char* env, bool data_cards_is_dir, const char* xdg,
                                     const char* home) {
    if (!flag.empty()) return flag;
    if (env && *env) return env;
    if (data_cards_is_dir) return "/data/loom-cards";
    if (xdg && *xdg) return (std::filesystem::path(xdg) / "1bit" / "loom-cards").string();
    if (home && *home) return (std::filesystem::path(home) / ".cache" / "1bit" / "loom-cards").string();
    return "";
}
inline std::string default_loom_serve_path(const std::string& flag, const char* env, const std::string& beside) {
    if (!flag.empty()) return flag;
    if (env && *env) return env;
    if (!beside.empty()) return beside;
    return "loom-serve";
}

}  // namespace onebit::loom
