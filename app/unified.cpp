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

// `1bit unified -m <model dir> -p <port>`: one native model on the NPU fast lane,
// behind the OpenAI endpoints Lemonade's `onebit` backend forwards to
// (third_party/lemonade/src/cpp/server/backends/onebit/onebit_server.cpp):
//
//   GET  /health, /v1/health    200 once the model is on the device
//   GET  /v1/models             the one model
//   POST /v1/chat/completions   ChatML prompt, greedy decoding, optional SSE stream
//   POST /v1/completions        raw prompt
//
// Requests run one at a time: the lane holds one KV cache, and each request
// rewrites it from position 0.
#include "unified.h"

#include "generate.h"
#include "model.h"
#include "tokenizer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace onebit {

namespace {

using json = nlohmann::json;

struct Engine {
    std::string id;
    std::unique_ptr<npu::Model> model;
    std::unique_ptr<npu::Tokenizer> tok;
    std::unique_ptr<npu::Lane> lane;
    std::mutex mu;
};

// The chat scaffold for the model families the lane serves. Hugging Face ships
// the template as Jinja; ChatML is what Qwen2 and Qwen3 render it to for plain
// text messages, with generation starting after "<|im_start|>assistant\n".
std::string chatml(const json& messages) {
    std::string p;
    for (const auto& m : messages) {
        std::string content;
        const auto& c = m.at("content");
        if (c.is_string()) {
            content = c;
        } else if (c.is_array()) {  // [{type: text, text: ...}, ...]
            for (const auto& part : c)
                if (part.value("type", "") == "text") content += part.value("text", "");
        }
        p += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + content + "<|im_end|>\n";
    }
    return p + "<|im_start|>assistant\n";
}

// The longest prefix of `s` that does not end inside a UTF-8 sequence.
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

void handle_generate(Engine& e, const httplib::Request& req, httplib::Response& res, bool chat) {
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
        prompt = chat ? chatml(body.at("messages")) : body.at("prompt").get<std::string>();
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", std::string("bad request: ") + ex.what()}}}}.dump(), "application/json");
        return;
    }
    npu::GenerateOptions opt;
    opt.max_tokens = body.value("max_completion_tokens", body.value("max_tokens", 1024));
    opt.repetition_penalty = body.value("repetition_penalty", 1.1f);
    const bool stream = body.value("stream", false);
    const std::string id = now_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = long(std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

    auto run = [&e, prompt, opt](const std::function<bool(const std::string&)>& on_text) mutable {
        std::lock_guard<std::mutex> lock(e.mu);
        const auto ids = e.tok->encode(prompt);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        const int room = npu::Lane::kMaxContext - int(ids.size());
        if (room < 1) throw std::runtime_error("prompt longer than the context");
        opt.max_tokens = std::min(opt.max_tokens, room);
        std::string pending;
        opt.on_token = [&](int t) {
            pending += e.tok->decode(t);
            const size_t n = complete_utf8(pending);
            if (n == 0) return true;
            const std::string out = pending.substr(0, n);
            pending.erase(0, n);
            return on_text(out);
        };
        auto r = npu::generate(*e.lane, *e.model, ids, opt);
        if (!pending.empty()) on_text(pending);
        return std::make_pair(r, ids.size());
    };
    // By value: for a stream, httplib calls the content provider after this
    // handler has returned.
    auto chunk = [id, object, created, model = e.id, chat](const std::string& text, const json& finish) {
        json c{{"id", id}, {"object", object}, {"created", created}, {"model", model}};
        c["choices"] = json::array({chat ? json{{"index", 0}, {"delta", {{"content", text}}}, {"finish_reason", finish}}
                                         : json{{"index", 0}, {"text", text}, {"finish_reason", finish}}});
        return "data: " + c.dump() + "\n\n";
    };

    if (stream) {
        res.set_chunked_content_provider("text/event-stream", [run, chunk](size_t, httplib::DataSink& sink) mutable {
            try {
                auto [r, n_prompt] = run([&](const std::string& t) {
                    const std::string c = chunk(t, nullptr);
                    return sink.write(c.data(), c.size());
                });
                const std::string end = chunk("", r.stopped_at_eos ? "stop" : "length") + "data: [DONE]\n\n";
                sink.write(end.data(), end.size());
            } catch (const std::exception& ex) {
                const std::string err = "data: " + json{{"error", {{"message", ex.what()}}}}.dump() + "\n\n";
                sink.write(err.data(), err.size());
            }
            sink.done();
            return true;
        });
        return;
    }
    try {
        std::string text;
        auto [r, n_prompt] = run([&](const std::string& t) {
            text += t;
            return true;
        });
        const size_t completion = r.tokens.size() - (r.stopped_at_eos ? 1 : 0);
        json out{{"id", id}, {"object", object}, {"created", created}, {"model", e.id}};
        const char* finish = r.stopped_at_eos ? "stop" : "length";
        out["choices"] = json::array({chat ? json{{"index", 0},
                                                  {"message", {{"role", "assistant"}, {"content", text}}},
                                                  {"finish_reason", finish}}
                                           : json{{"index", 0}, {"text", text}, {"finish_reason", finish}}});
        out["usage"] = {{"prompt_tokens", n_prompt},
                        {"completion_tokens", completion},
                        {"total_tokens", n_prompt + completion}};
        out["timings"] = {{"prompt_ms", r.prefill_ms},
                          {"predicted_ms", r.decode_ms},
                          {"predicted_per_second", r.tokens.empty() ? 0.0 : 1000.0 * double(r.tokens.size()) / r.decode_ms}};
        res.set_content(out.dump(), "application/json");
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
    }
}

}  // namespace

std::string npu_kernel_dir(const std::string& model_dir) { return model_dir + "/npu"; }

bool is_npu_model_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    for (const char* f : {"model.q4nx", "config.json", "tokenizer.json", "npu/layer_ctx1.elf", "npu/layer_ctx2.elf",
                          "npu/layer_ctx17.elf", "npu/lmhead.elf", "npu/layer.pdi"})
        if (!fs::exists(fs::path(dir) / f)) return false;
    return true;
}

int run_unified(int argc, char** argv) {
    std::string model_dir, host = "127.0.0.1";
    int port = 8000;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_dir = next();
        else if (a == "-p" || a == "--port") port = std::stoi(next());
        else if (a == "--host") host = next();
        else if (a == "--help" || a == "-h") {
            std::printf("usage: 1bit unified -m <model dir> [-p 8000] [--host 127.0.0.1]\n"
                        "  <model dir> holds model.q4nx, config.json, tokenizer.json and npu/ (the lane's kernels)\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (model_dir.empty()) throw std::runtime_error("-m <model dir> is required");
    // Lemonade hands over the checkpoint path, which may name the container file.
    if (std::filesystem::is_regular_file(model_dir)) model_dir = std::filesystem::path(model_dir).parent_path().string();

    Engine e;
    e.id = std::filesystem::path(model_dir).filename().string();
    httplib::Server srv;
    std::atomic<bool> ready{false};
    srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = ready ? 200 : 503;
        res.set_content(json{{"status", ready ? "ok" : "loading"}, {"model", e.id}, {"device", "npu"}}.dump(),
                        "application/json");
    });
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = ready ? 200 : 503;
        res.set_content(json{{"status", ready ? "ok" : "loading"}, {"model", e.id}, {"device", "npu"}}.dump(),
                        "application/json");
    });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({{{"id", e.id}, {"object", "model"}, {"owned_by", "1bit"}}})}}.dump(),
                        "application/json");
    });
    srv.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) { r.status = 503; return; }
        handle_generate(e, q, r, true);
    });
    srv.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) { r.status = 503; return; }
        handle_generate(e, q, r, false);
    });
    if (!srv.bind_to_port(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    // Serve /v1/health (503 until ready) while the model loads.
    std::thread listener([&] { srv.listen_after_bind(); });
    struct Stop {
        httplib::Server& s;
        std::thread& t;
        ~Stop() {
            if (t.joinable()) {
                s.stop();
                t.join();
            }
        }
    } stop{srv, listener};

    std::fprintf(stderr, "1bit unified: loading %s on the NPU\n", model_dir.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    e.model = std::make_unique<npu::Model>(model_dir);
    const std::string type = e.model->dims().model_type;
    if (type != "qwen3" && type != "qwen2")
        throw std::runtime_error("model_type '" + type + "': the fast lane serves the Qwen2/Qwen3 layer kernel only");
    e.tok = std::make_unique<npu::Tokenizer>(model_dir + "/tokenizer.json");
    e.lane = std::make_unique<npu::Lane>(*e.model, npu_kernel_dir(model_dir));
    ready = true;
    std::fprintf(stderr, "1bit unified: %s ready in %.0f ms on %s:%d\n", e.id.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), host.c_str(),
                 port);
    listener.join();
    return 0;
}

}  // namespace onebit
