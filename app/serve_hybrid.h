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
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hybrid {

using json = nlohmann::json;

struct Outcome {
    bool handed_over = false;   // the slot holds the provider's KV; the body carries id_slot/cache_prompt
    size_t n_tokens = 0;
    double prefill_ms = 0;      // the provider's own figure when it gave one, else wall time
    double restore_ms = 0;
    std::string why;            // when not handed over: the reason, for one stderr line
};

// The provider's stdout (its last line) and exit code, with a timeout in seconds (SIGTERM, then SIGKILL).
struct Run {
    int code = -1;
    std::string out;
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
    // read stdout to its end (the provider's exit closes it), then reap, with the timeout on the whole
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    bool killed = false;
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
            continue;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            ::kill(pid, killed ? SIGKILL : SIGTERM);
            if (killed) break;
            killed = true;
        }
    }
    ::close(out[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) break;
    r.code = killed ? -2 : (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return r;
#endif
}

class Prefill {
public:
    // provider: the program; elfs: passed as --elfs when not empty; slot_dir: the backend's
    // --slot-save-path (the state files go there, by bare name, as llama-server requires)
    Prefill(std::string provider, std::string elfs, std::string slot_dir, std::string model)
        : provider_(std::move(provider)), elfs_(std::move(elfs)), slot_dir_(std::move(slot_dir)), model_(std::move(model)) {
        if (const char* t = std::getenv("ONEBIT_HYBRID_PREFILL_TIMEOUT_S"); t && *t) timeout_s_ = std::max(1, std::atoi(t));
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
        const unsigned n = ++seq_;
        const std::string stem = "hybrid-" + std::to_string(n);
        const std::filesystem::path tokens = std::filesystem::path(slot_dir_) / (stem + ".tokens");
        const std::filesystem::path state = std::filesystem::path(slot_dir_) / (stem + ".bin");
        {
            std::ofstream f(tokens);
            for (int32_t id : ids) f << id << '\n';
            if (!f) { o.why = "cannot write " + tokens.string(); return o; }
        }
        std::vector<std::string> argv = {provider_, "--tokens", tokens.string(), "--out", state.string(), "--model", model_};
        if (!elfs_.empty()) { argv.push_back("--elfs"); argv.push_back(elfs_); }
        const auto t0 = std::chrono::steady_clock::now();
        const Run run = run_provider(argv, {"ONEBIT_HYBRID_BACKEND_URL=http://127.0.0.1:" + std::to_string(port)}, timeout_s_);
        o.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::error_code ec;
        std::filesystem::remove(tokens, ec);
        if (run.code != 0) {
            o.why = run.code == -2 ? "provider timed out after " + std::to_string(timeout_s_) + " s"
                                   : "provider exited " + std::to_string(run.code);
            std::filesystem::remove(state, ec);
            return o;
        }
        if (!std::filesystem::exists(state)) { o.why = "provider wrote no state file"; return o; }
        // the provider's own figures, from its last stdout line
        try {
            std::string last = run.out;
            while (!last.empty() && (last.back() == '\n' || last.back() == '\r')) last.pop_back();
            if (const size_t nl = last.rfind('\n'); nl != std::string::npos) last = last.substr(nl + 1);
            const json j = json::parse(last);
            if (j.contains("prefill_ms") && j["prefill_ms"].is_number()) o.prefill_ms = j["prefill_ms"].get<double>();
            if (j.contains("n_tokens") && j["n_tokens"].is_number_integer() && j["n_tokens"].get<size_t>() != ids.size()) {
                o.why = "provider prefilled " + std::to_string(j["n_tokens"].get<size_t>()) + " of " + std::to_string(ids.size()) + " tokens";
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
        auto r = c.Post("/slots/0?action=restore", json{{"filename", stem + ".bin"}}.dump(), "application/json");
        o.restore_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::filesystem::remove(state, ec);
        if (!r || r->status != 200) {
            o.why = "restore answered " + std::to_string(r ? r->status : -1) + (r ? ": " + r->body.substr(0, 200) : "");
            return o;
        }
        try {
            const json j = json::parse(r->body);
            const size_t restored = j.value("n_restored", size_t(0));
            if (restored != ids.size()) {
                o.why = "restore holds " + std::to_string(restored) + " of " + std::to_string(ids.size()) + " tokens";
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
    std::string provider_, elfs_, slot_dir_, model_;
    int timeout_s_ = 600;
    std::atomic<unsigned> seq_{0};
    std::mutex slot_mu_;
};

}  // namespace hybrid
