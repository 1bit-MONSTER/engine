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

// `1bit forward-serve -m <model.q4nx> --kernels <elf dir> -p <port>`: one
// repacked Q4NX model on the model-generic NPU forward, behind OpenAI
// endpoints. It is the serve route for `1bit serve -m <model.gguf> --device
// npu` after the GGUF has been repacked (serve.cpp).
//
//   GET  /health, /v1/health    200 once the model is on the device
//   GET  /v1/models             the one model
//   POST /v1/chat/completions   ChatML prompt, greedy decoding, optional SSE stream
//   POST /v1/completions        raw prompt
//
// Requests run one at a time: the forward holds one KV cache, and each request
// resets it from position 0.
#include "forward_serve.h"

#include "../npu/forward/include/q4nx_forward.h"
#include "tokenizer.h"
#include <xrt/xrt_device.h>

#include <httplib.h>
#include "http_guard.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <sys/file.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onebit {

namespace {

using json = nlohmann::json;

struct ForwardEngine {
    std::string id;
    std::unique_ptr<npu::Tokenizer> tok;
    int  bos_id  = -1;     // config.json bos_token_id (the chat template may open with it)
    bool add_bos = false;  // config.json add_bos_token: the rendered chat prompt needs BOS
    std::unique_ptr<Q4nxNpuForward> fwd;
    int eos = -1;
    int endoftext = -1;
    int max_context = 0;   // config.json max_position_embeddings: prompt plus answer
    std::mutex mu;
};

std::string chatml(const json& messages, bool thinking = true) {
    std::string p;
    for (const auto& m : messages) {
        std::string content;
        const auto& c = m.at("content");
        if (c.is_string()) {
            content = c;
        } else if (c.is_array()) {
            for (const auto& part : c)
                if (part.value("type", "") == "text") content += part.value("text", "");
        }
        p += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + content + "<|im_end|>\n";
    }
    p += "<|im_start|>assistant\n";
    (void)thinking;  // the <think> block is a Qwen3-lane idiom; the model-generic
    // route serves arbitrary archs, whose tokenizers do not know it.
    return p;
}

size_t complete_utf8(const std::string& s) {
    size_t lead = s.size();
    while (lead > 0 && s.size() - lead < 4 && (static_cast<unsigned char>(s[lead - 1]) & 0xC0) == 0x80) --lead;
    if (lead == 0) return s.size();
    const unsigned char c = static_cast<unsigned char>(s[lead - 1]);
    const size_t need = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    return s.size() - (lead - 1) >= need ? s.size() : lead - 1;
}

std::string now_id(const char* prefix) {
    static std::atomic<unsigned> n{0};
    return std::string(prefix) + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(n++);
}

int json_int(const std::string& path, const char* key, int def) {
    std::ifstream f(path);
    if (!f) return def;
    json j;
    f >> j;
    if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
    return def;
}

void handle_generate(ForwardEngine& e, const httplib::Request& req, httplib::Response& res, bool chat) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
        return;
    }
    std::string prompt;
    try {
        bool thinking = true;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
            thinking = body["chat_template_kwargs"].value("enable_thinking", true);
        prompt = chat ? chatml(body.at("messages"), thinking) : body.at("prompt").get<std::string>();
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", std::string("bad request: ") + ex.what()}}}}.dump(), "application/json");
        return;
    }
    const int max_tokens = body.value("max_completion_tokens", body.value("max_tokens", 256));
    // A prompt this long cannot fit the context whatever it tokenizes to; refuse it before
    // spending time on it.
    if (prompt.size() > size_t(e.max_context) * 16) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"prompt longer than the context"}})", "application/json");
        return;
    }
    const bool stream = body.value("stream", false);
    const std::string id = now_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = long(std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

    auto run = [&e, &req, prompt, max_tokens, chat](const std::function<bool(const std::string&)>& on_text) {
        // The tokenizer is const and keeps no per-call state, so it runs before the lock.
        std::vector<int> ids = e.tok->encode(prompt);
        // A chat template that opens with {{- bos_token }} (MiniCPM5 and other
        // llama-class models) needs the BOS id: the ChatML prompt built above never
        // carries it, and those models derail into their special tokens without it.
        if (chat && e.add_bos && e.bos_id >= 0) ids.insert(ids.begin(), e.bos_id);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        // The KV cache grows with every position: the prompt and the answer together stay
        // inside the model's context.
        const int room = e.max_context - int(ids.size());
        if (room < 1) throw std::runtime_error("prompt longer than the context");
        const int limit = std::clamp(max_tokens, 0, room);
        std::lock_guard<std::mutex> lock(e.mu);

        auto argmax = [](const std::vector<float>& v) {
            int a = 0;
            for (size_t i = 1; i < v.size(); i++)
                if (v[i] > v[a]) a = (int)i;
            return a;
        };

        std::vector<float> logits;
        // Prefill in batched (M>1) passes of at most the kernel's row width: each
        // projection's weight stream is then read once per chunk instead of once
        // per token, which is where prefill time goes (the per-op kernels are
        // M=128 wide, so a per-token pass leaves ~99% of the array idle).
        // step_batch falls back to per-token step() for the architectures it does
        // not model, so this is behavior-preserving.
        {
            const int chunk_max = std::max(1, e.fwd->max_batch_rows());
            for (int off = 0; off < (int)ids.size(); off += chunk_max) {
                const int take = std::min(chunk_max, (int)ids.size() - off);
                std::vector<int> chunk(ids.begin() + off, ids.begin() + off + take);
                if (!e.fwd->step_batch(chunk, off, logits))
                    throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
            }
        }

        std::string pending;
        bool stopped_at_eos = false;
        for (int n = 0; n < limit; n++) {
            if (req.is_connection_closed()) break;  // the client is gone: free the NPU
            const int next = argmax(logits);
            if (next == e.eos || (e.endoftext >= 0 && next == e.endoftext)) { stopped_at_eos = true; break; }
            pending += e.tok->decode(next);
            const size_t cut = complete_utf8(pending);
            if (cut > 0) {
                const std::string out = pending.substr(0, cut);
                pending.erase(0, cut);
                if (!on_text(out)) break;
            }
            if (!e.fwd->step(next, (int)ids.size() + n, logits))
                throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
        }
        (void)stopped_at_eos;
    };

    auto wrap = [&](const std::string& content, const std::string& finish) {
        json out;
        out["id"] = id;
        out["object"] = object;
        out["created"] = created;
        out["model"] = e.id;
        if (chat) {
            out["choices"] = json::array({{{"index", 0},
                                           {"message", {{"role", "assistant"}, {"content", content}}},
                                           {"finish_reason", finish}}});
        } else {
            out["choices"] = json::array({{{"index", 0}, {"text", content}, {"finish_reason", finish}}});
        }
        return out.dump();
    };

    try {
        if (!stream) {
            std::string content;
            run([&](const std::string& t) {
                content += t;
                return true;
            });
            res.set_content(wrap(content, "stop"), "application/json");
        } else {
            // Build the SSE frames while the handler still owns its locals.  The
            // content provider below runs AFTER this handler returns, so it must
            // not capture anything by reference (id/created/object die here) --
            // that was a dangling-reference crash (`std::system_error`).  Run the
            // forward here too, so it is not invoked from httplib's write path.
            json chunk_tpl;
            chunk_tpl["id"] = id;
            chunk_tpl["object"] = "chat.completion.chunk";
            chunk_tpl["created"] = created;
            chunk_tpl["model"] = e.id;
            auto chunks = std::make_shared<std::vector<std::string>>();
            run([&](const std::string& t) {
                json chunk = chunk_tpl;
                if (chat)
                    chunk["choices"] = json::array({{{"index", 0},
                                                     {"delta", {{"content", t}}},
                                                     {"finish_reason", nullptr}}});
                else
                    chunk["choices"] = json::array({{{"index", 0}, {"text", t}, {"finish_reason", nullptr}}});
                chunks->push_back("data: " + chunk.dump() + "\n\n");
                return true;
            });
            auto cursor = std::make_shared<size_t>(0);
            res.set_chunked_content_provider("text/event-stream",
                [chunks, cursor](size_t, httplib::DataSink& sink) {
                    while (*cursor < chunks->size()) {
                        const std::string& d = (*chunks)[*cursor];
                        if (!sink.write(d.data(), d.size())) return false;
                        ++*cursor;
                    }
                    const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                });
        }
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
    }
}

}  // namespace

int run_forward_serve(int argc, char** argv) {
    std::string model, kernels, host = "127.0.0.1", alias;
    int port = 8000;
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--kernels") kernels = next();
        else if (a == "-p" || a == "--port") port = std::stoi(next());
        else if (a == "--host") host = next();
        else if (a == "--alias") alias = next();
        else throw std::runtime_error("unknown option " + a);
    }
    if (model.empty()) throw std::runtime_error("usage: 1bit forward-serve -m <model.q4nx> --kernels <dir> [-p PORT]");
    if (!std::filesystem::exists(model)) throw std::runtime_error(model + " does not exist");

    const std::string dir = std::filesystem::path(model).parent_path().string();
    const std::string cfg = dir + "/config.json";
    const std::string tok_json = dir + "/tokenizer.json";

    ForwardEngine e;
    e.id = alias.empty() ? std::filesystem::path(dir).filename().string() : alias;
    e.tok = std::make_unique<npu::Tokenizer>(tok_json);
    e.bos_id  = json_int(cfg, "bos_token_id", -1);
    e.add_bos = json_int(cfg, "add_bos_token", 0) != 0;
    e.eos = json_int(cfg, "eos_token_id", -1);
    e.endoftext = e.tok->token_id("<|endoftext|>");
    // q4nx_config_int finds the key at any depth (a text_config's too).
    e.max_context = q4nx_config_int(cfg.c_str(), "max_position_embeddings", 0);
    if (e.max_context <= 0) e.max_context = 32768;

    e.fwd = std::make_unique<Q4nxNpuForward>();

    // Serialise NPU access across processes.  The device exposes a single hardware
    // context, so a second concurrent `1bit serve --device npu` fails inside
    // DRM_IOCTL_AMDXDNA_CREATE_HWCTX with EINVAL and the whole serve exits 1 --
    // which is what happened when several models were served at once.  The
    // per-process ELF runner (npu_engine_universal) has always taken this lock;
    // the serve route did not, so it could not be serialised against it or
    // against a sibling serve.  Held for the life of the process (fd not closed),
    // exactly like the runner.
    if (!std::getenv("NPU_NO_DEVICE_LOCK")) {
        // flock needs only a read fd, so the file stays 0644 and other users can
        // still open it; O_NOFOLLOW refuses a symlink planted in /tmp.
        const char* lk = "/tmp/1bit-npu-device.lock";
        int lfd = ::open(lk, O_CREAT | O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0644);
        if (lfd >= 0) {
            if (::flock(lfd, LOCK_EX | LOCK_NB) != 0) {
                std::fprintf(stderr, "[npu] waiting for the exclusive device lock (%s)...\n", lk);
                if (::flock(lfd, LOCK_EX) != 0)
                    std::fprintf(stderr, "[npu] device lock unavailable; continuing unlocked\n");
                else
                    std::fprintf(stderr, "[npu] device lock acquired\n");
            }
        }
    }

    xrt::device dev(0);
    if (!e.fwd->init(dev, model.c_str(), kernels, /*use_elf=*/true))
        throw std::runtime_error(std::string("forward init failed: ") + e.fwd->last_error());

    std::atomic<bool> ready{false};
    httplib::Server srv;
    install_request_guard(srv, host);  // Host / Origin (app/http_guard.h)
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& r) {
        r.status = ready ? 200 : 503;
        r.set_content(ready ? R"({"status":"ok"})" : R"({"status":"loading"})", "application/json");
    });
    srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& r) {
        r.status = ready ? 200 : 503;
        r.set_content(ready ? R"({"status":"ok"})" : R"({"status":"loading"})", "application/json");
    });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& r) {
        r.set_content(json{{"object", "list"},
                           {"data", json::array({{{"id", e.id}, {"object", "model"}, {"owned_by", "1bit"}}})}}.dump(),
                      "application/json");
    });
    auto post = [&](const httplib::Request& q, httplib::Response& r, bool chat) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        handle_generate(e, q, r, chat);
    };
    srv.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { post(q, r, true); });
    srv.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) { post(q, r, false); });

    if (!srv.bind_to_port(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    std::thread listener([&] { srv.listen_after_bind(); });
    struct Stop {
        httplib::Server& s;
        std::thread& t;
        ~Stop() { s.stop(); if (t.joinable()) t.join(); }
    } stop{srv, listener};

    ready = true;
    std::fprintf(stderr, "1bit forward-serve: %s ready on http://%s:%d (%s)\n",
                 e.id.c_str(), host.c_str(), port, e.fwd->backend().c_str());

    while (true) std::this_thread::sleep_for(std::chrono::seconds(3600));
    return 0;
}

}  // namespace onebit
