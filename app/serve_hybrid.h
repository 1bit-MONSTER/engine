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

// `1bit serve --device hybrid` (docs/hybrid.md): the prompt's KV cache is computed by an external
// prefill provider, and llama-server decodes from it on HRX. The provider is a program named by
// ONEBIT_HYBRID_PREFILL (or --prefill-provider); the engine's public tree knows only its contract:
//
//   <provider> --tokens <ids file> --out <state file> --model <model> [--elfs <dir>]
//
//   ids file    the prompt's token ids, one decimal id per line, in order (what llama-server's
//               /tokenize returned for the templated prompt)
//   state file  written by the provider: a llama.cpp sequence-state file for the prompt (the format
//               llama_state_seq_save_file writes: the ids, then the KV cells at positions 0..n-1 of
//               sequence 0), which llama-server restores into its slot with /slots/0?action=restore
//   stdout      one JSON line: {"n_tokens": N, "argmax": id, "prefill_ms": x} (prefill_ms optional)
//   exit 0      on success; anything else, or a missing state file, falls back to a plain request
//
// The provider also sees ONEBIT_HYBRID_BACKEND_URL=http://127.0.0.1:<port>, the decoding
// llama-server's own address (a test provider makes its state file through it; a real one has no
// use for it). ONEBIT_HYBRID_ELFS, when set, is passed on as --elfs.
//
// Per request: the chat template and tokenizer are the backend's own (/apply-template, /tokenize),
// so the provider sees exactly the ids the backend would have prefilled; the provider runs; the
// state file is restored into slot 0; the request is forwarded with cache_prompt on and id_slot 0,
// so the backend finds its prompt in the slot, re-evaluates the last token for its logits (llama-server
// always does), and decodes. One slot: hybrid requests are serialized here. Any failure on the way
// is said on stderr once per request and the request is forwarded as it came (the backend prefills
// it itself): the device is never unusable because the provider is.

#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hybrid {

using json = nlohmann::json;

struct Outcome {
    bool handed_over = false;   // the slot holds the provider's KV; the body carries id_slot/cache_prompt
    size_t n_tokens = 0;
    double prefill_ms = 0;      // the provider's own figure when it gave one, else wall time
    double restore_ms = 0;      // file route: the restore; shared route: the adopt
    double share_ms = 0;        // shared route: export + fd handoff, before the provider runs
    bool decoder_cannot_share = false;  // the export/adopt handshake failed before the provider ran, so the
                                        // state-file route is still available and correct
    std::string why;            // when not handed over: the reason, for one stderr line
};

// The provider's stdout (its last line) and exit code, with a timeout in seconds (SIGTERM, then SIGKILL).
struct Run {
    int code = -1;
    std::string out;
    bool hard_kill = false;  // the grace window after SIGTERM expired and the provider had to be killed
    bool signalled = false;  // it was SIGTERMed at its deadline and still exited 0: the work landed, late
};

inline Run run_provider(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int timeout_s) {
    Run r;
#if defined(_WIN32)
    (void) argv; (void) env_extra; (void) timeout_s;
    r.code = -1;
    return r;
#else
    int out[2] = {-1, -1};
    if (::pipe2(out, O_CLOEXEC) != 0) return r;
    std::vector<char*> args;
    for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    std::vector<std::string> env_store;
    for (char** e = environ; *e; ++e) env_store.emplace_back(*e);
    for (const auto& e : env_extra) env_store.push_back(e);
    std::vector<char*> envp;
    for (auto& e : env_store) envp.push_back(e.data());
    envp.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid < 0) { ::close(out[0]); ::close(out[1]); return r; }
    if (pid == 0) {
        ::dup2(out[1], 1);
        ::execvpe(args[0], args.data(), envp.data());
        ::_exit(127);
    }
    ::close(out[1]);
    // Read stdout to its end (the provider's exit closes it), then reap, with the timeout on the whole. The stop is
    // two-stage: SIGTERM at the deadline, SIGKILL only after a grace window of its own. The grace is not politeness:
    // an NPU provider tears down by releasing its BOs and destroying the hardware context, and killing it inside
    // that window is the path that crashed this box on 2026-11-04 01:48 (~/briefs/NPU-LANE-RULE-20261104.md
    // section 4 asks for >= 20 s). ONEBIT_HYBRID_PREFILL_GRACE_S sets the window (default 60 s); 0 asks for the
    // old immediate kill. The clock is checked on every pass, not only when the provider is silent: a provider
    // that prints forever is exactly the one that must not be able to outlive its budget.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    int grace_s = 60;
    if (const char* g = std::getenv("ONEBIT_HYBRID_PREFILL_GRACE_S"); g && *g) grace_s = std::max(0, std::atoi(g));
    bool killed = false;
    std::chrono::steady_clock::time_point term_at{};
    char buf[4096];
    for (;;) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(out[0], &fds);
        timeval tv{1, 0};
        const int sel = ::select(out[0] + 1, &fds, nullptr, nullptr, &tv);
        if (sel > 0) {
            const ssize_t n = ::read(out[0], buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            r.out.append(buf, size_t(n));
        }
        const auto now = std::chrono::steady_clock::now();
        if (!killed && now > deadline) {
            ::kill(pid, SIGTERM);
            term_at = now;
            killed = true;
        } else if (killed && (grace_s == 0 || now > term_at + std::chrono::seconds(grace_s))) {
            ::kill(pid, SIGKILL);
            r.hard_kill = true;
            break;
        }
    }
    ::close(out[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) break;
    // The deadline starts a teardown, it is not a verdict on the work: a provider that took its SIGTERM and then
    // finished (its state file or its region written, exit 0) has done what the handoff needed, and discarding it
    // would charge the request for the provider's prefill *and* the backend's re-processing of the whole prompt.
    // What lands is still validated by the callers (n_tokens in its JSON line, n_restored in the restore reply), so
    // a provider that exits 0 with half its work done falls back there rather than being trusted here.
    const bool exited_ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    r.signalled = killed && exited_ok;
    r.code = killed ? (exited_ok ? 0 : -2) : (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return r;
#endif
}

#if !defined(_WIN32)

// Target 1, cross-process: the decoder exports each of its KV buffers as a dma-buf and hands the fds over a unix
// socket (Linux cannot reopen a dma-buf from /proc/<pid>/fd, measured). The engine is the middle of that: it
// listens, receives {fd, offset, size, layout} per region, and passes the fds on to the provider as inherited
// descriptors, so the provider maps the decoder's own KV memory instead of writing a state file.
struct KvShare {
    std::string header;      // the decoder's JSON: region descriptors + the context params it built its cache with
    std::vector<int> fds;    // the regions, in the decoder's buffer order

    void close_all() {
        for (int fd : fds) {
            if (fd >= 0) ::close(fd);
        }
        fds.clear();
    }
    ~KvShare() { close_all(); }
};

// bind + listen on the slot's socket path; -1 on failure (why set)
inline int kvshare_listen(const std::string& path, std::string* why) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { *why = std::string("socket: ") + std::strerror(errno); return -1; }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) { *why = "socket path too long"; ::close(fd); return -1; }
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    ::unlink(path.c_str());
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        *why = "bind " + path + ": " + std::strerror(errno);
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 1) != 0) { *why = std::string("listen: ") + std::strerror(errno); ::close(fd); return -1; }
    return fd;
}

// one message: [uint32 length][JSON header], the region fds riding along as SCM_RIGHTS
inline bool kvshare_recv(int lfd, KvShare* out, std::string* why) {
    const int fd = ::accept(lfd, nullptr, nullptr);
    if (fd < 0) { *why = std::string("accept: ") + std::strerror(errno); return false; }

    char buf[4 + 65536];
    std::vector<char> control(CMSG_SPACE(64 * sizeof(int)));
    size_t got = 0;
    bool have_control = false;
    for (;;) {
        if (got >= 4) {
            uint32_t len = 0;
            std::memcpy(&len, buf, 4);
            if (len > sizeof(buf) - 4) { *why = "the decoder's header is too large"; ::close(fd); return false; }
            if (got >= 4 + (size_t) len) break;
        }
        ssize_t n;
        if (!have_control) {
            msghdr msg{};
            iovec iov{ buf, sizeof(buf) };
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control.data();
            msg.msg_controllen = control.size();
            n = ::recvmsg(fd, &msg, 0);
            if (n > 0) {
                have_control = true;
                for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
                    if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) continue;
                    const size_t n_fds = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                    for (size_t i = 0; i < n_fds; ++i) {
                        int region_fd = -1;
                        std::memcpy(&region_fd, CMSG_DATA(c) + i * sizeof(int), sizeof(int));
                        ::fcntl(region_fd, F_SETFD, 0); // the provider inherits them through exec
                        out->fds.push_back(region_fd);
                    }
                }
            }
        } else {
            n = ::recv(fd, buf + got, sizeof(buf) - got, 0);
        }
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { *why = "the decoder closed the socket before sending the regions"; ::close(fd); return false; }
        got += (size_t) n;
    }
    ::close(fd);
    uint32_t len = 0;
    std::memcpy(&len, buf, 4);
    out->header.assign(buf + 4, len);
    return !out->fds.empty();
}

#endif

class Prefill {
public:    // provider: the program; elfs: passed as --elfs when not empty; slot_dir: the backend's
    // --slot-save-path (the state files go there, by bare name, as llama-server requires)
    Prefill(std::string provider, std::string elfs, std::string slot_dir, std::string model)
        : provider_(std::move(provider)), elfs_(std::move(elfs)), slot_dir_(std::move(slot_dir)), model_(std::move(model)) {
        if (const char* t = std::getenv("ONEBIT_HYBRID_PREFILL_TIMEOUT_S"); t && *t) timeout_s_ = std::max(1, std::atoi(t));
        // target 1: the decoder exports its KV region (ONEBIT_KV_SHARE, which this process's children inherit) and
        // the provider prefills into it instead of writing a state file
        if (const char* v = std::getenv("ONEBIT_KV_SHARE"); v && *v) shared_ = true;
    }

    const std::string& provider() const { return provider_; }

    // Serializes hybrid requests: one slot, one handoff at a time, held through the decode by the caller.
    std::mutex& slot_mutex() { return slot_mu_; }

    // The prompt as the backend sees it: the templated chat (/apply-template) or the completion's prompt.
    // Empty when the request has no text prompt (an array prompt, images), in which case there is no handoff.
    static std::string prompt_text(int port, const std::string& path, const json& body, std::string* why) {
        if (path == "/v1/chat/completions") {
            httplib::Client c("127.0.0.1", port);
            c.set_read_timeout(60);
            auto r = c.Post("/apply-template", body.dump(), "application/json");
            if (!r || r->status != 200) { *why = "apply-template answered " + std::to_string(r ? r->status : -1); return ""; }
            try {
                return json::parse(r->body).value("prompt", std::string());
            } catch (const std::exception&) {
                *why = "apply-template reply is not JSON";
                return "";
            }
        }
        if (body.contains("prompt") && body["prompt"].is_string()) return body["prompt"].get<std::string>();
        *why = "no text prompt";
        return "";
    }

    static std::vector<int32_t> tokenize(int port, const std::string& prompt, std::string* why) {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(60);
        const json req = {{"content", prompt}, {"add_special", true}, {"parse_special", true}};
        auto r = c.Post("/tokenize", req.dump(), "application/json");
        if (!r || r->status != 200) { *why = "tokenize answered " + std::to_string(r ? r->status : -1); return {}; }
        try {
            return json::parse(r->body).at("tokens").get<std::vector<int32_t>>();
        } catch (const std::exception&) {
            *why = "tokenize reply has no tokens";
            return {};
        }
    }

    // The whole handoff for one request. On success body carries id_slot 0 and cache_prompt true.
    Outcome prepare(int port, const std::string& path, json& body) {
        Outcome o;
        const std::string prompt = prompt_text(port, path, body, &o.why);
        if (prompt.empty()) { if (o.why.empty()) o.why = "empty prompt"; return o; }
        const std::vector<int32_t> ids = tokenize(port, prompt, &o.why);
        if (ids.size() < 2) { if (o.why.empty()) o.why = "prompt shorter than 2 tokens"; return o; }
        o.n_tokens = ids.size();
        // engine#941: hand over one token less than the prompt. A hybrid model's recurrent memory keeps its whole
        // prefix in one cell, so an exact-state handoff of the *entire* prompt leaves the decoder nothing it is
        // allowed to evaluate - the server's cache path insists on at least one token ([TAG_PROMPT_LOGITS]) and
        // trims it out of the memory, which a recurrent state cannot do. Leaving the last prompt token to the
        // decoder is one legitimate decode step from state that is already in place, and it covers a sliding
        // window too. Without this the decoder re-processes the whole prompt and the handoff buys nothing.
        const std::vector<int32_t> head(ids.begin(), ids.end() - 1);
        if (shared_) {
            Outcome shared = prepare_shared(port, head, o);
            if (shared.handed_over) {
                body["id_slot"] = 0;
                body["cache_prompt"] = true;
            }
            // A decoder that cannot export its cache (a params or model mismatch, or a decoder pinned before
            // the recurrent-state arm: docs/hybrid.md) leaves the shared route with nothing to hand over.
            // Falling back to the state file keeps the prompt off the decoder's own prefill path, which is
            // the whole reason this route exists; only the handshake failures fall back, never a provider
            // that already wrote into the region or an adopt that failed after it.
            if (shared.handed_over || !shared.decoder_cannot_share) return shared;
            std::fprintf(stderr, "1bit serve: hybrid: %s; this decoder cannot share its KV, so the state-file route is used\n",
                         shared.why.c_str());
            o = Outcome{};
            o.n_tokens = ids.size();
        }
        const unsigned n = ++seq_;
        const std::string stem = "hybrid-" + std::to_string(n);
        const std::filesystem::path tokens = std::filesystem::path(slot_dir_) / (stem + ".tokens");
        const std::filesystem::path state = std::filesystem::path(slot_dir_) / (stem + ".bin");
        {
            std::ofstream f(tokens);
            for (int32_t id : head) f << id << '\n';
            if (!f) { o.why = "cannot write " + tokens.string(); return o; }
        }
        std::vector<std::string> argv = {provider_, "--tokens", tokens.string(), "--out", state.string(), "--model", model_};
        if (!elfs_.empty()) { argv.push_back("--elfs"); argv.push_back(elfs_); }
        const auto t0 = std::chrono::steady_clock::now();
        const Run run = run_provider(argv, {"ONEBIT_HYBRID_BACKEND_URL=http://127.0.0.1:" + std::to_string(port)}, timeout_s_);
        o.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // said only when it happens, and only on a fallback: a provider that finished after its deadline is a
        // slow provider, not a broken one, and the operator needs to be able to tell the two apart
        const std::string late = run.signalled ? " (it was signalled at its " + std::to_string(timeout_s_) + " s deadline and still exited 0)" : "";
        std::error_code ec;
        std::filesystem::remove(tokens, ec);
        if (run.code != 0) {
            o.why = run.code == -2 ? "provider timed out after " + std::to_string(timeout_s_) + " s" +
                                         (run.hard_kill ? " and had to be killed" : " and stopped on SIGTERM")
                                   : "provider exited " + std::to_string(run.code);
            std::filesystem::remove(state, ec);
            return o;
        }
        if (!std::filesystem::exists(state)) { o.why = "provider wrote no state file" + late; return o; }
        // the provider's own figures, from its last stdout line
        try {
            std::string last = run.out;
            while (!last.empty() && (last.back() == '\n' || last.back() == '\r')) last.pop_back();
            if (const size_t nl = last.rfind('\n'); nl != std::string::npos) last = last.substr(nl + 1);
            const json j = json::parse(last);
            if (j.contains("prefill_ms") && j["prefill_ms"].is_number()) o.prefill_ms = j["prefill_ms"].get<double>();
            if (j.contains("n_tokens") && j["n_tokens"].is_number_integer() && j["n_tokens"].get<size_t>() != head.size()) {
                o.why = "provider prefilled " + std::to_string(j["n_tokens"].get<size_t>()) + " of " + std::to_string(head.size()) + " tokens" + late;
                std::filesystem::remove(state, ec);
                return o;
            }
        } catch (const std::exception&) {
            // no JSON line: the file is the result
        }
        // restore into slot 0 (the backend reads slot_dir/<filename>; its reply counts the cells as n_restored)
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(600);
        const auto t1 = std::chrono::steady_clock::now();
        auto r = c.Post("/slots/0?action=restore", json{{"filename", stem + ".bin"}, {"leave_one_token", true}}.dump(), "application/json");
        o.restore_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::filesystem::remove(state, ec);
        if (!r || r->status != 200) {
            o.why = "restore answered " + std::to_string(r ? r->status : -1) + (r ? ": " + r->body.substr(0, 200) : "");
            return o;
        }
        try {
            const json j = json::parse(r->body);
            const size_t restored = j.value("n_restored", size_t(0));
            if (restored != head.size()) {
                o.why = "restore holds " + std::to_string(restored) + " of " + std::to_string(head.size()) + " tokens";
                return o;
            }
            if (j.contains("timings") && j["timings"].contains("restore_ms")) o.restore_ms = j["timings"]["restore_ms"].get<double>();
        } catch (const std::exception&) {
            o.why = "restore reply is not JSON";
            return o;
        }
        body["id_slot"] = 0;
        body["cache_prompt"] = true;
        o.handed_over = true;
        return o;
    }

private:
#if !defined(_WIN32)
    // Target 1: the provider prefills into the decoder's own KV memory. No state file is written or read; the
    // only things that cross a boundary are the region fds and a few hundred bytes of JSON.
    Outcome prepare_shared(int port, const std::vector<int32_t>& ids, Outcome& o) {
        const unsigned n = ++seq_;
        const std::string stem = "kvshare-" + std::to_string(n);
        const std::filesystem::path tokens = std::filesystem::path(slot_dir_) / (stem + ".tokens");
        const std::filesystem::path layout = std::filesystem::path(slot_dir_) / (stem + ".json");
        const std::filesystem::path sockpath = std::filesystem::path(slot_dir_) / (stem + ".sock");
        {
            std::ofstream f(tokens);
            for (int32_t id : ids) f << id << '\n';
            if (!f) { o.why = "cannot write " + tokens.string(); return o; }
        }

        // 1. listen, so the decoder has somewhere to hand the fds; 2. it exports and connects
        std::string why;
        const int lfd = kvshare_listen(sockpath.string(), &why);
        if (lfd < 0) { o.decoder_cannot_share = true; o.why = "kv share: " + why; return o; }
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(600);
        const auto t0 = std::chrono::steady_clock::now();
        auto r = c.Post("/slots/0?action=share", json{{"socket", sockpath.string()}}.dump(), "application/json");
        if (!r || r->status != 200) {
            ::close(lfd);
            std::error_code ec;
            std::filesystem::remove(tokens, ec);
            std::filesystem::remove(sockpath, ec);
            o.decoder_cannot_share = true;
            o.why = "share answered " + std::to_string(r ? r->status : -1) + (r ? ": " + r->body.substr(0, 200) : "");
            return o;
        }
        // the decoder numbers every export; adopt has to quote it or it refuses the cells
        uint64_t share_id = 0;
        try {
            share_id = json::parse(r->body).at("share_id").get<uint64_t>();
        } catch (const std::exception&) {
            ::close(lfd);
            std::error_code ec;
            std::filesystem::remove(tokens, ec);
            std::filesystem::remove(sockpath, ec);
            o.decoder_cannot_share = true;
            o.why = "share reply has no share_id";
            return o;
        }
        KvShare share;
        const bool got = kvshare_recv(lfd, &share, &why);
        ::close(lfd);
        std::error_code ec;
        std::filesystem::remove(sockpath, ec);
        o.share_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!got) {
            std::filesystem::remove(tokens, ec);
            o.why = "kv share: " + (why.empty() ? std::string("no regions from the decoder") : why);
            return o;
        }
        {
            std::ofstream f(layout);
            f << share.header;
            if (!f) { std::filesystem::remove(tokens, ec); o.why = "cannot write " + layout.string(); return o; }
        }

        // 3. the provider maps those fds as its own KV tensors and prefills the prompt into them
        std::string fds_arg;
        for (size_t i = 0; i < share.fds.size(); ++i) {
            fds_arg += (i ? "," : "") + std::to_string(share.fds[i]);
        }
        std::vector<std::string> argv = {provider_, "--tokens", tokens.string(), "--model", model_,
                                         "--layout", layout.string(), "--fds", fds_arg};
        if (!elfs_.empty()) { argv.push_back("--elfs"); argv.push_back(elfs_); }
        const auto t1 = std::chrono::steady_clock::now();
        const Run run = run_provider(argv, {"ONEBIT_HYBRID_BACKEND_URL=http://127.0.0.1:" + std::to_string(port)}, timeout_s_);
        o.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        const std::string late = run.signalled ? " (it was signalled at its " + std::to_string(timeout_s_) + " s deadline and still exited 0)" : "";
        std::filesystem::remove(tokens, ec);
        std::filesystem::remove(layout, ec);
        if (run.code != 0) {
            o.why = run.code == -2 ? "provider timed out after " + std::to_string(timeout_s_) + " s" +
                                         (run.hard_kill ? " and had to be killed" : " and stopped on SIGTERM")
                                   : "provider exited " + std::to_string(run.code);
            return o;
        }

        // 4. the decoder takes the cells over: no file, no copy (llama_kv_cells_adopt)
        json adopt = {{"tokens", ids}, {"share_id", share_id}, {"leave_one_token", true}};
        const auto t2 = std::chrono::steady_clock::now();
        auto ra = c.Post("/slots/0?action=adopt", adopt.dump(), "application/json");
        o.restore_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
        if (!ra || ra->status != 200) {
            o.why = "adopt answered " + std::to_string(ra ? ra->status : -1) + (ra ? ": " + ra->body.substr(0, 200) : "") + late;
            return o;
        }
        try {
            const json j = json::parse(ra->body);
            const size_t adopted = j.value("n_adopted", size_t(0));
            if (adopted != ids.size()) {
                o.why = "adopt took " + std::to_string(adopted) + " of " + std::to_string(ids.size()) + " cells" + late;
                return o;
            }
            if (j.contains("timings") && j["timings"].contains("adopt_ms")) o.restore_ms = j["timings"]["adopt_ms"].get<double>();
        } catch (const std::exception&) {
            o.why = "adopt reply is not JSON";
            return o;
        }
        o.handed_over = true;
        return o;
    }
#else
    Outcome prepare_shared(int, const std::vector<int32_t>&, Outcome& o) {
        o.why = "the shared KV handoff needs unix sockets (POSIX only)";
        return o;
    }
#endif

private:
    std::string provider_, elfs_, slot_dir_, model_;
    bool shared_ = false;       // target 1: fds instead of a state file
    int timeout_s_ = 600;
    std::atomic<unsigned> seq_{0};
    std::mutex slot_mu_;
};

}  // namespace hybrid
