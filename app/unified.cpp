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

// `1bit unified -m <model dir> -p <port>`: one native model on the NPU, behind OpenAI
// endpoints. It is `1bit serve`'s NPU route (serve.cpp). The model decides the path
// (npu_model_kind): Qwen2/Qwen3 dense models run on the fast lane (docs/npu.md),
// Qwen3.6-35B-A3B on the lax decode from full ELFs (docs/npu-lax.md).
//
//   GET  /health, /v1/health    200 once the model is on the device
//   GET  /v1/models             the one model
//   POST /v1/chat/completions   ChatML prompt, greedy decoding, optional SSE stream
//   POST /v1/completions        raw prompt
//
// Requests run one at a time: the device holds one cache. The fast lane rewrites it from
// position 0 per request. The lax decode continues it when a request extends the last
// one's tokens exactly, or restores a snapshot of the DeltaNet state taken at an earlier
// prompt's prefix (--snapshots, npu/lax_turns.h): a chat follow-up re-feeds only the turns
// after the one its history last diverged in. It starts over otherwise.
#include "unified.h"

#include "generate.h"
#include "lax.h"
#include "model.h"
#include "tokenizer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

namespace onebit {

namespace {

using json = nlohmann::json;

struct Engine {
    std::string id;
    std::unique_ptr<npu::Model> model;
    std::unique_ptr<npu::Tokenizer> tok;
    std::unique_ptr<npu::Lane> lane;  // the fast lane, or
    std::unique_ptr<npu::lax::Config> lax_cfg;
    std::unique_ptr<npu::lax::Decoder> lax;  // the lax decode and its conversation
    std::unique_ptr<npu::lax::Session> session;
    std::vector<int> lax_stop;
    int max_context = npu::Lane::kMaxContext;
    float default_penalty = 1.1f;  // greedy 0.6B models loop without it; the 35B does not need it
    bool open_think = false;       // Qwen3.6's template opens "<think>\n" when thinking is on
    std::mutex mu;

    // reused: prompt tokens already on the device (the lax decode's conversation); cache:
    // where from ("live", "snapshot"), empty when none.
    npu::GenerateResult generate(const std::vector<int>& ids, const npu::GenerateOptions& opt, size_t& reused,
                                 std::string& cache) {
        reused = 0;
        cache.clear();
        if (lane) return npu::generate(*lane, *model, ids, opt);
        npu::lax::GenerateOptions lo;
        lo.max_tokens = opt.max_tokens;
        lo.repetition_penalty = opt.repetition_penalty;
        lo.penalty_window = opt.penalty_window;
        lo.stop = lax_stop;
        lo.on_token = opt.on_token;
        const auto r = session->generate(ids, lo);
        npu::GenerateResult out;
        out.tokens = r.tokens;
        out.prefill_ms = r.prefill_ms;
        out.decode_ms = r.decode_ms;
        out.stopped_at_eos = r.stopped_at_eos;
        reused = r.reused;
        if (r.source != npu::lax::Plan::Scratch) cache = r.source == npu::lax::Plan::Live ? "live" : "snapshot";
        if (r.source == npu::lax::Plan::Snapshot || r.snapshots_taken)
            std::fprintf(stderr, "1bit unified: %zu of %zu prompt tokens from %s (restore %.1f ms), %d snapshot%s taken "
                                 "(%.1f ms), %zu kept (%.0f MB)\n",
                         r.reused, ids.size(), cache.empty() ? "nothing" : cache.c_str(), r.restore_ms, r.snapshots_taken,
                         r.snapshots_taken == 1 ? "" : "s", r.snapshot_ms, session->snapshots().size(),
                         double(session->snapshots().bytes()) / 1e6);
        return out;
    }
};

// The chat scaffold for the model families the lane serves. Hugging Face ships
// the template as Jinja; ChatML is what Qwen2 and Qwen3 render it to for plain
// text messages, with generation starting after "<|im_start|>assistant\n".
// open_think: the template opens the think block itself when thinking is on (Qwen3.6's
// does; Qwen3's leaves it to the model).
std::string chatml(const json& messages, bool thinking = true, bool open_think = false) {
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
    p += "<|im_start|>assistant\n";
    // chat_template_kwargs.enable_thinking = false: what Qwen3's own template
    // emits, an empty think block, so the model answers directly.
    if (!thinking) p += "<think>\n\n</think>\n\n";
    else if (open_think) p += "<think>\n";
    return p;
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
        bool thinking = true;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
            thinking = body["chat_template_kwargs"].value("enable_thinking", true);
        prompt = chat ? chatml(body.at("messages"), thinking, e.open_think) : body.at("prompt").get<std::string>();
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", std::string("bad request: ") + ex.what()}}}}.dump(), "application/json");
        return;
    }
    npu::GenerateOptions opt;
    opt.max_tokens = body.value("max_completion_tokens", body.value("max_tokens", 1024));
    opt.repetition_penalty = body.value("repetition_penalty", e.default_penalty);
    const bool stream = body.value("stream", false);
    // stream_options.include_usage: a last chunk with no choices carries the usage, as OpenAI's.
    const bool stream_usage = stream && body.contains("stream_options") && body["stream_options"].is_object() &&
                              body["stream_options"].value("include_usage", false);
    const std::string id = now_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = long(std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

    auto run = [&e, prompt, opt](const std::function<bool(const std::string&)>& on_text) mutable {
        std::lock_guard<std::mutex> lock(e.mu);
        const auto ids = e.tok->encode(prompt);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        const int room = e.max_context - int(ids.size());
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
        size_t reused = 0;
        std::string cache;
        auto r = e.generate(ids, opt, reused, cache);
        if (!pending.empty()) on_text(pending);
        return std::make_tuple(r, ids.size(), reused, cache);
    };
    // By value: for a stream, httplib calls the content provider after this
    // handler has returned.
    auto chunk = [id, object, created, model = e.id, chat](const std::string& text, const json& finish) {
        json c{{"id", id}, {"object", object}, {"created", created}, {"model", model}};
        c["choices"] = json::array({chat ? json{{"index", 0}, {"delta", {{"content", text}}}, {"finish_reason", finish}}
                                         : json{{"index", 0}, {"text", text}, {"finish_reason", finish}}});
        return "data: " + c.dump() + "\n\n";
    };

    auto usage = [](const npu::GenerateResult& r, size_t n_prompt, size_t n_cached) {
        const size_t completion = r.tokens.size() - (r.stopped_at_eos ? 1 : 0);
        return json{{"prompt_tokens", n_prompt},
                    {"completion_tokens", completion},
                    {"total_tokens", n_prompt + completion},
                    {"prompt_tokens_details", {{"cached_tokens", n_cached}}}};
    };
    auto timings = [](const npu::GenerateResult& r, const std::string& cache) {
        json t{{"prompt_ms", r.prefill_ms},
               {"predicted_ms", r.decode_ms},
               {"predicted_per_second", r.tokens.empty() ? 0.0 : 1000.0 * double(r.tokens.size()) / r.decode_ms}};
        if (!cache.empty()) t["cache"] = cache;
        return t;
    };

    if (stream) {
        res.set_chunked_content_provider("text/event-stream", [run, chunk, usage, timings, stream_usage, id, object, created,
                                                               model = e.id](size_t, httplib::DataSink& sink) mutable {
            try {
                auto [r, n_prompt, n_cached, cache] = run([&](const std::string& t) {
                    const std::string c = chunk(t, nullptr);
                    return sink.write(c.data(), c.size());
                });
                std::string end = chunk("", r.stopped_at_eos ? "stop" : "length");
                if (stream_usage) {
                    const json u{{"id", id},       {"object", object},
                                 {"created", created}, {"model", model},
                                 {"choices", json::array()}, {"usage", usage(r, n_prompt, n_cached)},
                                 {"timings", timings(r, cache)}};
                    end += "data: " + u.dump() + "\n\n";
                }
                end += "data: [DONE]\n\n";
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
        auto [r, n_prompt, n_cached, cache] = run([&](const std::string& t) {
            text += t;
            return true;
        });
        json out{{"id", id}, {"object", object}, {"created", created}, {"model", e.id}};
        const char* finish = r.stopped_at_eos ? "stop" : "length";
        out["choices"] = json::array({chat ? json{{"index", 0},
                                                  {"message", {{"role", "assistant"}, {"content", text}}},
                                                  {"finish_reason", finish}}
                                           : json{{"index", 0}, {"text", text}, {"finish_reason", finish}}});
        out["usage"] = usage(r, n_prompt, n_cached);
        out["timings"] = timings(r, cache);
        res.set_content(out.dump(), "application/json");
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
    }
}

}  // namespace

std::string npu_kernel_dir(const std::string& model_dir) { return model_dir + "/npu"; }

namespace {

std::string model_type(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) return "";
    try {
        const auto j = json::parse(f);
        return j.value("model_type", "");
    } catch (const std::exception&) {
        return "";
    }
}

bool has_lax_kernels(const std::string& k) {
    namespace fs = std::filesystem;
    for (const char* kind : {"lax_l", "lax_a", "ln", "lm_head_q8"})
        if (!fs::exists(fs::path(k) / kind / "insts.elf")) return false;
    return true;
}

}  // namespace

std::string lax_kernel_dir(const std::string& dir, const std::string& lax_kernels) {
    if (!lax_kernels.empty()) return lax_kernels;
    if (has_lax_kernels(dir + "/npu/lax")) return dir + "/npu/lax";
    if (const char* env = std::getenv("ONEBIT_NPU_LAX_KERNELS"); env && *env) return env;
    return "";
}

NpuModel npu_model_kind(const std::string& dir, const std::string& lax_kernels) {
    namespace fs = std::filesystem;
    for (const char* f : {"model.q4nx", "config.json", "tokenizer.json"})
        if (!fs::exists(fs::path(dir) / f)) return NpuModel::None;
    if (model_type(dir) == "qwen3_5_moe") {
        const std::string k = lax_kernel_dir(dir, lax_kernels);
        return !k.empty() && has_lax_kernels(k) ? NpuModel::Lax : NpuModel::None;
    }
    for (const char* f : {"npu/layer_ctx1.elf", "npu/layer_ctx2.elf", "npu/layer_ctx17.elf", "npu/lmhead.elf", "npu/layer.pdi"})
        if (!fs::exists(fs::path(dir) / f)) return NpuModel::None;
    return NpuModel::Lane;
}

bool is_npu_model_dir(const std::string& dir, const std::string& lax_kernels) {
    return npu_model_kind(dir, lax_kernels) != NpuModel::None;
}

int run_unified(int argc, char** argv) {
    std::string model_dir, host = "127.0.0.1", alias, kernels, transport = "elf";
    int port = 8000, snapshots = 4;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_dir = next();
        else if (a == "-p" || a == "--port") port = std::stoi(next());
        else if (a == "--host") host = next();
        else if (a == "--alias") alias = next();
        else if (a == "--kernels") kernels = next();
        else if (a == "--transport") transport = next();
        else if (a == "--snapshots") snapshots = std::stoi(next());
        else if (a == "--help" || a == "-h") {
            std::printf("usage: 1bit unified -m <model dir> [-p 8000] [--host 127.0.0.1] [--alias NAME]\n"
                        "                    [--kernels DIR] [--transport elf|classic] [--snapshots 4]\n"
                        "  <model dir> holds model.q4nx, config.json, tokenizer.json, and for the fast lane\n"
                        "              npu/ (its kernels); Qwen3.6-35B-A3B runs on the lax kernels in --kernels,\n"
                        "              else <model dir>/npu/lax, else $ONEBIT_NPU_LAX_KERNELS (docs/npu-lax.md)\n"
                        "  --transport the lax kernels as full ELFs (default) or classic xclbin, for A/B\n"
                        "  --snapshots DeltaNet state snapshots the lax decode keeps in host memory for chat\n"
                        "              follow-ups (70 MB each; 0: only exact extensions reuse the cache)\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (model_dir.empty()) throw std::runtime_error("-m <model dir> is required");
    if (snapshots < 0) throw std::runtime_error("--snapshots must be 0 or more");
    // Lemonade hands over the checkpoint path, which may name the container file.
    if (std::filesystem::is_regular_file(model_dir)) model_dir = std::filesystem::path(model_dir).parent_path().string();

    Engine e;
    e.id = alias.empty() ? std::filesystem::path(model_dir).filename().string() : alias;
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
    const NpuModel kind = npu_model_kind(model_dir, kernels);
    if (kind == NpuModel::None)
        throw std::runtime_error(model_dir + " is not an NPU model directory (docs/serve.md)");
    e.model = std::make_unique<npu::Model>(model_dir);
    e.tok = std::make_unique<npu::Tokenizer>(model_dir + "/tokenizer.json");
    if (kind == NpuModel::Lax) {
        const std::string k = lax_kernel_dir(model_dir, kernels);
        e.lax_cfg = std::make_unique<npu::lax::Config>(npu::lax::Config::from_model(*e.model, model_dir));
        e.lax = std::make_unique<npu::lax::Decoder>(*e.model, *e.lax_cfg, k, npu::lax::parse_transport(transport));
        for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
            const auto ids = e.tok->encode(s);
            if (ids.size() == 1) e.lax_stop.push_back(ids[0]);
        }
        // Snapshots before each prompt's last <|im_start|>: a follow-up repeats every message
        // before the answer it re-renders.
        npu::lax::SessionOptions so;
        so.snapshots = size_t(snapshots);
        if (const auto ids = e.tok->encode("<|im_start|>"); ids.size() == 1) so.turn_tokens = ids;
        // Rows past the tokenizer are padding.
        e.session = std::make_unique<npu::lax::Session>(*e.lax, e.tok->size(), std::move(so));
        e.max_context = npu::lax::kMaxContext;
        e.default_penalty = 1.0f;
        e.open_think = true;
        std::fprintf(stderr, "1bit unified: Qwen3.6-35B-A3B on the lax decode (%s); up to %d state snapshots of %.0f MB\n",
                     e.lax->kernels().c_str(), snapshots, double(e.lax->state_bytes()) / 1e6);
    } else {
        const std::string type = e.model->dims().model_type;
        if (type != "qwen3" && type != "qwen2")
            throw std::runtime_error("model_type '" + type + "': the fast lane serves the Qwen2/Qwen3 layer kernel only");
        e.lane = std::make_unique<npu::Lane>(*e.model, npu_kernel_dir(model_dir));
    }
    ready = true;
    std::fprintf(stderr, "1bit unified: %s ready in %.0f ms on %s:%d\n", e.id.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), host.c_str(),
                 port);
    listener.join();
    return 0;
}

}  // namespace onebit
