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
//   POST /v1/chat/completions   ChatML prompt, sampling controls, optional SSE stream
//   POST /v1/completions        raw prompt
//
// Decoding is greedy unless the request asks for more: temperature, top_k, top_p and seed
// sample, stop cuts the answer, and grammar (GBNF) or response_format {"type":
// "json_object"} constrain it.  See app/npu_sampling.h, app/npu_grammar.h and
// docs/npu.md "Sampling and structured output".
//
// Requests run one at a time: the forward holds one KV cache, and each request
// resets it from position 0.
#include "forward_serve.h"

#include "../npu/forward/include/q4nx_forward.h"
#include "npu_device_lock.h"
#include "npu_grammar.h"
#include "npu_sampling.h"
#include "tokenizer.h"
#include "rt.h"  // the own NPU runtime (ONEBIT_NPU_RT_DIR)

#include <httplib.h>
#include "http_guard.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onebit {

namespace {

using json = nlohmann::json;

struct ForwardEngine {
    std::string id;
    int lock_fd = -1;  // app/npu_device_lock.h: the device lock, held for the process's life
    std::unique_ptr<npu::Tokenizer> tok;
    int  bos_id  = -1;     // config.json bos_token_id (the chat template may open with it)
    bool add_bos = false;  // config.json add_bos_token: the rendered chat prompt needs BOS
    std::unique_ptr<Q4nxNpuForward> fwd;
    int eos = -1;
    int endoftext = -1;
    int max_context = 0;   // config.json max_position_embeddings: prompt plus answer
    // decode() of every vocabulary id, built on the first constrained request: the GBNF
    // mask tests each token's text against the grammar state, per token.
    std::vector<std::string> tok_text;
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
    // Decode-time controls.  Every field is optional and absent means the greedy decode
    // this route has always done, so existing callers see no change; a field the route
    // cannot honour is a 400 rather than a silently ignored key.
    npu::Options opts;
    std::unique_ptr<npu::Grammar> grammar;
    try {
        opts = npu::options_from_body(body);
        if (opts.json_object || !opts.grammar.empty())
            grammar = std::make_unique<npu::Grammar>(opts.json_object ? npu::json_grammar() : opts.grammar);
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
        return;
    }
    const npu::Grammar* gram = grammar.get();
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

    // Returns the finish_reason the request earned: "stop" for an end-of-sequence token,
    // a stop sequence or a grammar that allows nothing more, "length" when the token
    // budget ran out.
    auto run = [&e, &req, prompt, max_tokens, chat, &opts, gram](const std::function<bool(const std::string&)>& on_text) {
        std::string finish = "stop";
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

        // Sampler state: one generator per request, seeded so a run can be reproduced.
        std::mt19937_64 rng(npu::resolve_seed(opts.sample.seed));
        npu::Grammar::State grammar_state;
        std::vector<char> allowed;
        if (gram) {
            grammar_state = gram->start();
            // One decoded piece per vocabulary id: the mask below tests those against the
            // grammar state.  A megabyte of small strings, built once per process.
            if (e.tok_text.size() < logits.size()) {
                e.tok_text.clear();
                e.tok_text.reserve(logits.size());
                for (size_t i = 0; i < logits.size(); i++) e.tok_text.push_back(e.tok->decode(int(i)));
            }
        }
        // allowed[i] says token i can be appended without leaving the grammar.  The first
        // byte test rejects almost every token cheaply (a JSON object at the start accepts
        // a handful of first bytes out of 256), so only the survivors pay a full feed().
        auto refresh_allowed = [&]() {
            allowed.assign(logits.size(), 0);
            const npu::ByteMask first = gram->next_bytes(grammar_state);
            for (size_t i = 0; i < logits.size(); i++) {
                const std::string& t = e.tok_text[i];
                if (t.empty()) {
                    allowed[i] = 1;  // an empty piece cannot leave the grammar
                } else if (first.has(static_cast<unsigned char>(t[0]))) {
                    npu::Grammar::State probe = grammar_state;
                    allowed[i] = gram->feed(probe, t) ? 1 : 0;
                }
            }
        };

        std::string pending;  // decoded, not yet sent: a stop sequence may still match in it
        bool stopped_at_eos = false, stopped_at_stop = false;
        int produced = 0;
        for (int n = 0; n < limit; n++) {
            if (req.is_connection_closed()) break;  // the client is gone: free the NPU
            if (gram) refresh_allowed();
            const int next = npu::sample_token(logits, opts.sample, rng, gram ? &allowed : nullptr);
            // -1: the grammar allows no token here, so the answer ends where it is.  EOS is
            // not masked by the grammar on purpose: a constrained answer is still allowed to
            // stop early (the client validates it), which beats a runaway inside the mask.
            if (next < 0) { stopped_at_eos = true; break; }
            if (next == e.eos || (e.endoftext >= 0 && next == e.endoftext)) { stopped_at_eos = true; break; }
            if (gram) gram->feed(grammar_state, e.tok_text[next]);
            pending += e.tok->decode(next);
            ++produced;
            const size_t stop_at = npu::find_stop(pending, opts.stop);
            if (stop_at != std::string::npos) {
                if (stop_at > 0) on_text(pending.substr(0, stop_at));
                stopped_at_stop = true;
                break;
            }
            // Send what no stop sequence can still match, and keep the tail: the next
            // token may complete a match that starts in it.
            const size_t cut = complete_utf8(pending);
            const size_t hold = npu::stop_holdback(opts.stop);
            size_t emit = cut > hold ? cut - hold : 0;
            while (emit > 0 && (static_cast<unsigned char>(pending[emit]) & 0xC0) == 0x80) --emit;  // keep whole characters
            if (emit > 0) {
                const std::string out = pending.substr(0, emit);
                pending.erase(0, emit);
                if (!on_text(out)) break;
            }
            if (!e.fwd->step(next, (int)ids.size() + n, logits))
                throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
        }
        if (!stopped_at_stop) {
            const size_t cut = complete_utf8(pending);
            if (cut > 0) on_text(pending.substr(0, cut));
        }
        if (produced >= limit && !stopped_at_eos && !stopped_at_stop) finish = "length";
        return finish;
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
            const std::string finish = run([&](const std::string& t) {
                content += t;
                return true;
            });
            res.set_content(wrap(content, finish), "application/json");
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
    int port = 8000, streams = 1;
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
        else if (a == "--npu-streams") streams = std::stoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    if (model.empty())
        throw std::runtime_error("usage: 1bit forward-serve -m <model.q4nx> --kernels <dir> [-p PORT] [--npu-streams N]");
    if (streams < 1 || streams > kNpuMaxStreams)
        throw std::runtime_error("--npu-streams takes 1.." + std::to_string(kNpuMaxStreams) +
                                 " (the device's hwctx limit), not " + std::to_string(streams));
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

    // Who may have the NPU (app/npu_device_lock.h): one process by default, N when the
    // operator asks with --npu-streams. It is a policy, not a hardware limit -- the
    // driver's npu4 sets hwctx_limit = 16 -- and the default stays exclusive because a
    // second stream takes the same DMA engine and its own copy of the weights: two
    // engines on two models measured 34.1 % / 70.4 % of solo, 0.41x aggregate
    // (docs/npu.md "Two streams on one NPU").
    // Held for the life of the process (fd not closed), exactly like the per-process
    // ELF runner (npu_engine_universal), which takes the same exclusive lock.
    if (!std::getenv("NPU_NO_DEVICE_LOCK"))
        e.lock_fd = take_npu_device_lock(streams);

    // Offline mode (npu/forward/include/q4nx_forward.h, init_host): with
    // NPU_INFER_HOST_GEMM=1 the forward runs the model with host GEMMs and never opens
    // the device, which is how the sampling and grammar path is exercised without an
    // NPU.  Open the device only when the forward is going to use it.
    const bool host_gemm = std::getenv("NPU_INFER_HOST_GEMM") != nullptr;
    std::unique_ptr<xrt::device> dev;
    if (!host_gemm) dev = std::make_unique<xrt::device>(0);
    const bool ok = dev ? e.fwd->init(*dev, model.c_str(), kernels) : e.fwd->init_host(model.c_str(), kernels);
    if (!ok) throw std::runtime_error(std::string("forward init failed: ") + e.fwd->last_error());

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
    std::fprintf(stderr, "1bit forward-serve: %s ready on http://%s:%d (%s)", e.id.c_str(), host.c_str(), port,
                 e.fwd->backend().c_str());
    if (streams > 1) std::fprintf(stderr, " [npu-streams=%d]", streams);
    std::fprintf(stderr, "\n");

    while (true) std::this_thread::sleep_for(std::chrono::seconds(3600));
    return 0;
}

}  // namespace onebit
