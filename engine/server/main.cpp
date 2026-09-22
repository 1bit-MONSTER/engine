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
// 1bit-server: llama-server-compatible OpenAI HTTP server.
//
// Speaks the subset of llama-server's interface that Lemonade drives: the
// launch flags it passes, GET /health (503 while loading), /v1/models,
// /v1/chat/completions and /v1/completions (streaming and not), with both
// OpenAI `usage` and llama-server `timings` in the responses.
#include <httplib.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "chat_template.h"
#include "generator.h"
#include "openai.h"

using namespace onebit;
using json = openai::json;

namespace {

constexpr const char* kVersion = "0.0.1";

struct Args {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    uint32_t ctx_size = 0;
    size_t threads = 0;
    std::string device = "auto";
    std::string alias;
};

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit-server -m MODEL.gguf [options]\n"
                 "  -m, --model PATH       GGUF model (required)\n"
                 "  -c, --ctx-size N       context length (default: min(training context, 4096))\n"
                 "      --host HOST        listen address (default 127.0.0.1)\n"
                 "      --port N           listen port (default 8080)\n"
                 "      --device NAME      auto | cpu (default auto)\n"
                 "  -t, --threads N        CPU threads (default: all)\n"
                 "  -a, --alias NAME       model name reported by the API (default: file name)\n"
                 "  -np, --parallel N      concurrent sequences; only 1 is supported\n"
                 "      --jinja            accepted: the GGUF chat template is always used\n"
                 "      --metrics          accepted: no /metrics endpoint yet\n"
                 "  -h, --help, --version\n");
}

std::optional<Args> parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s needs a value\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };
    auto to_int = [](const char* s, long long lo, long long hi, long long& out) {
        char* end = nullptr;
        out = std::strtoll(s, &end, 10);
        return end && *end == '\0' && out >= lo && out <= hi;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        long long n = 0;
        const char* v = nullptr;
        if (f == "-h" || f == "--help") {
            usage(stdout);
            std::exit(0);
        } else if (f == "--version") {
            std::printf("1bit-server %s\n", kVersion);
            std::exit(0);
        } else if (f == "-m" || f == "--model") {
            if (!(v = need(i))) return std::nullopt;
            a.model = v;
        } else if (f == "-c" || f == "--ctx-size") {
            if (!(v = need(i)) || !to_int(v, 0, UINT32_MAX, n)) return std::fprintf(stderr, "error: bad --ctx-size\n"), std::nullopt;
            a.ctx_size = uint32_t(n);
        } else if (f == "--host") {
            if (!(v = need(i))) return std::nullopt;
            a.host = v;
        } else if (f == "--port") {
            if (!(v = need(i)) || !to_int(v, 0, 65535, n)) return std::fprintf(stderr, "error: bad --port\n"), std::nullopt;
            a.port = int(n);
        } else if (f == "--device" || f == "-dev") {
            if (!(v = need(i))) return std::nullopt;
            a.device = v;
        } else if (f == "-t" || f == "--threads") {
            if (!(v = need(i)) || !to_int(v, 0, 4096, n)) return std::fprintf(stderr, "error: bad --threads\n"), std::nullopt;
            a.threads = size_t(n);
        } else if (f == "-a" || f == "--alias") {
            if (!(v = need(i))) return std::nullopt;
            a.alias = v;
        } else if (f == "-np" || f == "--parallel") {
            if (!(v = need(i)) || !to_int(v, 1, 1, n)) return std::fprintf(stderr, "error: only --parallel 1 is supported\n"), std::nullopt;
        } else if (f == "--jinja" || f == "--metrics") {
            // Accepted for llama-server compatibility; see usage().
        } else {
            std::fprintf(stderr, "error: unknown argument %s\n", f.c_str());
            usage(stderr);
            return std::nullopt;
        }
    }
    if (a.model.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        return std::nullopt;
    }
    std::string dev = a.device;
    for (char& c : dev) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (dev != "auto" && dev != "cpu") {
        std::fprintf(stderr, "error: device '%s' is not available in this build (available: auto, cpu)\n", a.device.c_str());
        return std::nullopt;
    }
    if (a.alias.empty()) a.alias = std::filesystem::path(a.model).stem().string();
    return a;
}

std::string random_id(const char* prefix) {
    static std::mutex mu;
    static std::mt19937_64 rng{std::random_device{}()};
    std::lock_guard lock(mu);
    return std::format("{}{:016x}", prefix, rng());
}

bool sse(httplib::DataSink& sink, const json& j) {
    const std::string s = "data: " + j.dump() + "\n\n";
    return sink.is_writable() && sink.write(s.data(), s.size());
}

// Everything that exists once the model has loaded.
struct Engine {
    std::unique_ptr<Generator> gen;
    std::unique_ptr<ChatTemplate> tmpl;
};

}  // namespace

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (!args) return 2;

    enum { kLoading, kReady, kFailed };
    std::atomic<int> state{kLoading};
    std::string load_error;
    Engine engine;

    httplib::Server svr;
    auto send_json = [](httplib::Response& res, int status, const json& j) {
        res.status = status;
        res.set_content(j.dump(), "application/json");
    };
    auto ready_or_503 = [&](httplib::Response& res) {
        const int s = state.load();
        if (s == kReady) return true;
        if (s == kLoading) send_json(res, 503, openai::error(503, "Loading model", "unavailable_error"));
        else send_json(res, 500, openai::error(500, "model failed to load: " + load_error, "server_error"));
        return false;
    };

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        if (ready_or_503(res)) send_json(res, 200, {{"status", "ok"}});
    });

    auto models = [&](const httplib::Request&, httplib::Response& res) {
        json m = {{"id", args->alias}, {"object", "model"}, {"created", int64_t(std::time(nullptr))}, {"owned_by", "1bit"}};
        send_json(res, 200, {{"object", "list"}, {"data", json::array({m})}});
    };
    svr.Get("/v1/models", models);
    svr.Get("/models", models);

    // Shared by both endpoints: parse, build the prompt, run, respond.
    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        if (!ready_or_503(res)) return;
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) return send_json(res, 400, openai::error(400, "request body is not valid JSON", "invalid_request_error"));
        auto parsed = chat ? openai::parse_chat(body) : openai::parse_completion(body);
        if (!parsed) return send_json(res, 400, openai::error(400, parsed.error(), "invalid_request_error"));
        openai::Request r = std::move(*parsed);

        std::string prompt = r.prompt;
        if (chat) {
            auto rendered = engine.tmpl->render(r.messages, r.tools, /*add_generation_prompt=*/true, r.template_kwargs);
            if (!rendered) return send_json(res, 400, openai::error(400, rendered.error(), "invalid_request_error"));
            prompt = std::move(*rendered);
            r.gen.split_reasoning = r.split_reasoning;
            r.gen.reasoning_open = prompt.ends_with("<think>\n") || prompt.ends_with("<think>");
        }
        auto tokens = engine.gen->tokenizer().encode(prompt, /*parse_special=*/true);
        if (!tokens) return send_json(res, 400, openai::error(400, tokens.error(), "invalid_request_error"));
        if (tokens->empty()) return send_json(res, 400, openai::error(400, "prompt is empty", "invalid_request_error"));
        if (tokens->size() >= engine.gen->n_ctx())
            return send_json(res, 400, openai::error(400, std::format("the prompt is {} tokens, which exceeds the context size of {}",
                                                                      tokens->size(), engine.gen->n_ctx()),
                                                     "exceed_context_size_error"));
        r.gen.prompt = std::move(*tokens);
        const openai::Meta meta{random_id(chat ? "chatcmpl-" : "cmpl-"), args->alias, int64_t(std::time(nullptr))};

        auto log = [](const GenerateResult& g) {
            std::fprintf(stderr, "request: prompt %d (cached %d) in %.0f ms, generated %d in %.0f ms (%.2f tok/s), %s\n",
                         g.prompt_tokens, g.cache_n, g.prompt_ms, g.predicted_n, g.predicted_ms,
                         g.predicted_ms > 0 ? 1000.0 * g.predicted_n / g.predicted_ms : 0.0, g.finish_reason.c_str());
        };

        if (!r.stream) {
            auto g = engine.gen->generate(r.gen, {});
            if (!g) return send_json(res, 500, openai::error(500, g.error(), "server_error"));
            log(*g);
            return send_json(res, 200, chat ? openai::chat_response(meta, *g) : openai::completion_response(meta, *g));
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream", [&, r = std::move(r), meta, chat, log](size_t, httplib::DataSink& sink) {
                if (chat && !sse(sink, openai::chat_chunk(meta, {{"role", "assistant"}, {"content", nullptr}}, nullptr)))
                    return false;
                auto g = engine.gen->generate(r.gen, [&](const TextDelta& d) {
                    if (!chat) return sse(sink, openai::completion_chunk(meta, d.content, nullptr));
                    json delta = json::object();
                    if (!d.reasoning.empty()) delta["reasoning_content"] = d.reasoning;
                    if (!d.content.empty()) delta["content"] = d.content;
                    return sse(sink, openai::chat_chunk(meta, delta, nullptr));
                });
                if (!g) {
                    // In-stream error: an error object and no [DONE], as OpenAI does.
                    sse(sink, openai::error(500, g.error(), "server_error"));
                    sink.done();
                    return true;
                }
                log(*g);
                json last = chat ? openai::chat_chunk(meta, json::object(), g->finish_reason)
                                 : openai::completion_chunk(meta, "", g->finish_reason);
                last["timings"] = openai::timings(*g);
                sse(sink, last);
                if (r.include_usage) {
                    json u = chat ? openai::chat_chunk(meta, json::object(), nullptr)
                                  : openai::completion_chunk(meta, "", nullptr);
                    u["choices"] = json::array();
                    u["usage"] = openai::usage(*g);
                    sse(sink, u);
                }
                const std::string done = "data: [DONE]\n\n";
                if (sink.is_writable()) sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };
    for (const char* path : {"/v1/chat/completions", "/chat/completions"})
        svr.Post(path, [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, true); });
    for (const char* path : {"/v1/completions", "/completions"})
        svr.Post(path, [&](const httplib::Request& q, httplib::Response& s) { handle(q, s, false); });

    svr.set_error_handler([&](const httplib::Request&, httplib::Response& res) {
        if (res.body.empty()) send_json(res, res.status, openai::error(res.status, "not found", "not_found_error"));
    });

    if (!svr.bind_to_port(args->host, args->port)) {
        std::fprintf(stderr, "error: cannot listen on %s:%d\n", args->host.c_str(), args->port);
        return 1;
    }
    std::fprintf(stderr, "1bit-server %s listening on http://%s:%d (loading %s)\n", kVersion, args->host.c_str(),
                 args->port, args->model.c_str());

    std::thread loader([&] {
        auto t0 = std::chrono::steady_clock::now();
        auto fail = [&](const std::string& e) {
            load_error = e;
            std::fprintf(stderr, "error: %s\n", e.c_str());
            state = kFailed;
        };
        auto file = GgufFile::open(args->model);
        if (!file) return fail(file.error());
        auto tok = Tokenizer::from_gguf(*file);
        if (!tok) return fail("tokenizer: " + tok.error());
        auto tmpl = ChatTemplate::from_gguf(*file);
        if (!tmpl) return fail(tmpl.error());
        auto model = CpuModel::load(args->model, args->threads, args->ctx_size);
        if (!model) return fail(model.error());
        engine.tmpl = std::make_unique<ChatTemplate>(std::move(*tmpl));
        engine.gen = std::make_unique<Generator>(std::move(*model), std::move(*tok));
        state = kReady;
        std::fprintf(stderr, "model ready in %.1f s: %s on cpu, context %u\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), args->alias.c_str(),
                     engine.gen->n_ctx());
    });
    loader.detach();

    return svr.listen_after_bind() ? 0 : 1;
}
