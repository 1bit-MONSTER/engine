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

#include "embed_serve.h"

#include "embed_model.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <thread>

#ifdef ONEBIT_NPU
#include "embed_npu.h"
#include "../npu/forward/include/q4nx_forward.h"
#include "npu_device_lock.h"
#include "tokenizer.h"
#include "rt.h"
#include <cstdlib>
#include <fstream>
#include <mutex>
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace onebit::embed_serve {

bool container(const std::string& path) {
    std::error_code ec;
    return fs::is_directory(path, ec) && fs::is_regular_file(fs::path(path) / "model.embed", ec);
}

bool forward_encoder(const std::string& path) {
#ifdef ONEBIT_NPU
    std::error_code ec;
    const fs::path d(path);
    if (!fs::is_directory(d, ec) || !fs::is_regular_file(d / "model.q4nx", ec)) return false;
    std::ifstream f(d / "config.json");
    if (!f) return false;
    try {
        const json c = json::parse(f);
        return c.value("use_bidirectional_attention", false);
    } catch (const std::exception&) {
        return false;
    }
#else
    (void)path;
    return false;
#endif
}

struct Backend::Impl {
    embed::Model model;
#ifdef ONEBIT_NPU
    embed::NpuGemm npu;
    bool npu_installed = false;
    // engine#1786: the q4nx encoder kind. One forward, so requests are serialised.
    std::unique_ptr<xrt::device> dev;
    std::unique_ptr<Q4nxNpuForward> fwd;
    std::unique_ptr<onebit::npu::Tokenizer> tok;
    int bos = -1, eos = -1, cap = 0, lock_fd = -1;
    std::mutex mu;
#endif
};

bool Backend::load(const std::string& dir, const std::string& device, std::string& why) {
    if (device != "npu" && device != "cpu") {
        why = "an embedding container runs on --device npu or cpu, not '" + device + "'";
        return false;
    }
    impl_ = std::make_shared<Impl>();
#ifdef ONEBIT_NPU
    if (forward_encoder(dir)) {
        Impl& m = *impl_;
        const fs::path d(dir);
        // The kernel directory the chat route resolves too: ONEBIT_NPU_KERNELS, else <dir>/kernels, else the build's.
        std::string kernels;
        if (const char* e = std::getenv("ONEBIT_NPU_KERNELS")) kernels = e;
        if (kernels.empty() && fs::is_directory(d / "kernels")) kernels = (d / "kernels").string();
#ifdef ONEBIT_NPU_KERNELS_DIR
        if (kernels.empty()) kernels = ONEBIT_NPU_KERNELS_DIR;
#endif
        const bool host = device == "cpu" || std::getenv("NPU_INFER_HOST_GEMM") != nullptr;
        if (host) setenv("NPU_INFER_HOST_GEMM", "1", 1);   // the forward reads it: --device cpu is host GEMM
        if (!host && kernels.empty()) {
            why = "no NPU kernel (full ELF) directory resolves for this encoder: set ONEBIT_NPU_KERNELS or put them in " +
                  (d / "kernels").string();
            return false;
        }
        m.fwd = std::make_unique<Q4nxNpuForward>();
        const std::string model = (d / "model.q4nx").string();
        if (!host) {
            if (npu_lock_taken(true, std::getenv("NPU_NO_DEVICE_LOCK") != nullptr)) m.lock_fd = take_npu_device_lock(1);
            m.dev = std::make_unique<xrt::device>(0);
        }
        const bool ok = m.dev ? m.fwd->init(*m.dev, model.c_str(), kernels) : m.fwd->init_host(model.c_str(), kernels);
        if (!ok) { why = std::string("encoder init failed: ") + m.fwd->last_error(); return false; }
        if (!m.fwd->noncausal()) { why = "config.json declares use_bidirectional_attention, but the forward is causal"; return false; }
        try {
            m.tok = std::make_unique<onebit::npu::Tokenizer>((d / "tokenizer.json").string());
        } catch (const std::exception& e) {
            why = std::string("tokenizer.json: ") + e.what();
            return false;
        }
        // The framing the model was trained with: tokenizer_config.json's add_bos_token / add_eos_token (both true
        // on Embedding-Gemma-300M-NPU2, `<bos> text <eos>`, the reference tokenizer's ids exactly).
        std::ifstream tc(d / "tokenizer_config.json");
        json t = json::object(), c = json::object();
        try { if (tc) t = json::parse(tc); } catch (const std::exception&) {}
        std::ifstream cf(d / "config.json");
        try { if (cf) c = json::parse(cf); } catch (const std::exception&) {}
        auto token_of = [&](const char* key) {
            if (!t.contains(key) || !t[key].is_string()) return -1;
            return m.tok->token_id(t[key].get<std::string>());
        };
        if (t.value("add_bos_token", false)) m.bos = token_of("bos_token");
        if (t.value("add_eos_token", false)) m.eos = token_of("eos_token");
        const int win = c.value("sliding_window", 0);
        m.cap = win > 0 ? win / 2 : c.value("max_position_embeddings", 512);
        std::vector<int> probe;
        if (m.bos >= 0) probe.push_back(m.bos);
        if (m.eos >= 0) probe.push_back(m.eos);
        if (probe.empty()) probe.push_back(0);
        std::vector<float> v;
        if (!m.fwd->embed_pooled(probe, v)) { why = std::string("encoder warm-up failed: ") + m.fwd->last_error(); return false; }
        device_ = host ? "cpu" : "npu";
        dim_ = (int)v.size();
        id_ = d.filename().string();
        return true;
    }
#endif
    if (!impl_->model.load(dir, why)) return false;
    device_ = device;
    dim_ = impl_->model.config().out_dim;
    id_ = fs::path(dir).filename().string();
#ifdef ONEBIT_NPU
    if (device == "npu") {
        // Pack every weight block before the first request: the server answers from a worker pool, and a
        // lazily filled cache shared by those threads is a data race (npu/embed/embed_npu.h says why).
        std::string nwhy;
        if (!impl_->npu.prepare(impl_->model, nwhy)) {
            why = nwhy;
            return false;
        }
        impl_->model.set_gemm(impl_->npu.hook());
        impl_->npu_installed = true;
    }
#else
    if (device == "npu") {
        why = "this build has no NPU lane (ONEBIT_NPU=OFF): --device cpu serves the same vectors on the host";
        return false;
    }
#endif
    return true;
}

int Backend::max_tokens() const {
#ifdef ONEBIT_NPU
    if (impl_ && impl_->fwd) return impl_->cap;
#endif
    return impl_ ? impl_->model.config().max_tokens : 0;
}

std::string Backend::health_json(bool ready) const {
    return json{{"status", ready ? "ok" : "loading"},
                {"model", id_},
                {"device", device_},
                {"dim", dim_},
                {"max_input_tokens", max_tokens()},
                {"modalities", json::array({"text"})}}
        .dump();
}

std::string Backend::models_json() const {
    return json{{"object", "list"},
                {"data", json::array({{{"id", id_}, {"object", "model"}, {"owned_by", "1bit"},
                                       {"device", device_}, {"role", "embedding"}}})}}
        .dump();
}

std::string Backend::embeddings(const std::string& body, int& status) {
    auto fail = [&](int code, const std::string& message) {
        status = code;
        return json{{"error", {{"message", message}, {"type", "invalid_request_error"}}}}.dump();
    };
    json req;
    try {
        req = json::parse(body);
    } catch (const std::exception& e) {
        return fail(400, std::string("body is not JSON: ") + e.what());
    }
    if (!req.contains("input")) return fail(400, "input is required");
    // OpenAI accepts a string, an array of strings, or an array of token-id arrays. Only the two text forms
    // are served: a token-id array would need the caller's ids to be this model's ids, and answering one
    // silently with tokenised text is the kind of guess this route should not make.
    std::vector<std::string> texts;
    const auto& in = req["input"];
    if (in.is_string()) {
        texts.push_back(in.get<std::string>());
    } else if (in.is_array()) {
        for (const auto& e : in) {
            if (!e.is_string()) return fail(400, "input[] must be strings (token-id arrays are not served)");
            texts.push_back(e.get<std::string>());
        }
        if (texts.empty()) return fail(400, "input is an empty array");
    } else {
        return fail(400, "input must be a string or an array of strings");
    }

    json data = json::array();
    long tokens = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        std::string why;
#ifdef ONEBIT_NPU
        if (impl_->fwd) {
            Impl& m = *impl_;
            std::vector<int> ids;
            if (m.bos >= 0) ids.push_back(m.bos);
            for (int t : m.tok->encode(texts[i])) ids.push_back(t);
            if (m.eos >= 0) ids.push_back(m.eos);
            tokens += static_cast<long>(ids.size());
            if ((int)ids.size() > m.cap)
                return fail(400, "input[" + std::to_string(i) + "] is " + std::to_string(ids.size()) +
                                     " tokens; this encoder takes at most " + std::to_string(m.cap));
            std::vector<float> v;
            {
                std::lock_guard<std::mutex> g(m.mu);
                if (!m.fwd->embed_pooled(ids, v)) why = m.fwd->last_error();
            }
            if (v.empty()) return fail(500, "embedding of input[" + std::to_string(i) + "] failed: " + why);
            data.push_back({{"object", "embedding"}, {"index", static_cast<int>(i)}, {"embedding", v}});
            continue;
        }
#endif
        const auto ids = impl_->model.tokenize(texts[i]);
        tokens += static_cast<long>(ids.size());
        const auto v = impl_->model.encode_ids(ids, why);
        if (v.empty()) return fail(500, "embedding of input[" + std::to_string(i) + "] failed: " + why);
        data.push_back({{"object", "embedding"}, {"index", static_cast<int>(i)}, {"embedding", v}});
    }
    ++requests_;
    status = 200;
    return json{{"object", "list"},
                {"data", std::move(data)},
                {"model", id_},
                {"usage", {{"prompt_tokens", tokens}, {"total_tokens", tokens}}}}
        .dump();
}

int serve_only(const std::string& model_dir, const std::string& host, int port, const std::string& device,
               const std::string& alias) {
    Backend backend;
    std::string why;
    if (!backend.load(model_dir, device, why)) {
        std::fprintf(stderr, "1bit serve: %s: %s\n", model_dir.c_str(), why.c_str());
        return 2;
    }
    const std::string id = alias.empty() ? backend.id() : alias;
    std::atomic<bool> ready{true};
    httplib::Server srv;
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& r) {
        r.set_content(backend.health_json(ready), "application/json");
    });
    srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& r) {
        r.set_content(backend.health_json(ready), "application/json");
    });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& r) {
        std::string body = backend.models_json();
        if (id != backend.id()) {  // --alias renames the served model in the list as well as in the reply
            json j = json::parse(body);
            j["data"][0]["id"] = id;
            body = j.dump();
        }
        r.set_content(body, "application/json");
    });
    auto handler = [&](const httplib::Request& q, httplib::Response& r) {
        int status = 200;
        const auto t0 = std::chrono::steady_clock::now();
        r.set_content(backend.embeddings(q.body, status), "application/json");
        r.status = status;
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "1bit serve: POST %s - %d (%.1f ms, %s)\n", q.path.c_str(), status, ms,
                     backend.device().c_str());
    };
    srv.Post("/v1/embeddings", handler);
    srv.Post("/embeddings", handler);
    // bind_to_port() answers bool; the port comes back from bind_to_any_port(), which is the same bind_internal(host, 0)
    // with the chosen port returned. Same idiom as app/serve.cpp:2023 and :2694 - a caller that asked for port 0 has no
    // other way to learn which one it got.
    const int bound_port = port == 0 ? srv.bind_to_any_port(host) : (srv.bind_to_port(host, port) ? port : -1);
    if (bound_port < 0) {
        std::fprintf(stderr, "1bit serve: cannot listen on %s:%d\n", host.c_str(), port);
        return 2;
    }
    // listen_after_bind() on its own thread, the same shape the rest of serve uses: the blocking listen()
    // was observed to answer one request and then leave the next connection unanswered (alternating), which
    // is a liveness bug no client should have to retry around.
    std::atomic<bool> stop{false};
    std::thread listener([&] { srv.listen_after_bind(); });
    std::fprintf(stderr, "1bit serve: %s on %s embedding (%s), in-process: /v1/embeddings, dim %d, %d-token cap\n",
                 id.c_str(), backend.device().c_str(), model_dir.c_str(), backend.dim(), backend.max_tokens());
    // engine#1779: the readiness line every other serve route prints, and the one `serve_port_from_log()`
    // (tests/serve_port.sh) reads back. The in-process embedding route returned before either of the other two
    // "ready on" sites (serve.cpp:2050, :2765), so a caller that asked for --port 0 could not discover the port it was
    // listening on - which is what engine #1485 fixed for the other routes and left true only for them.
    std::fprintf(stderr, "1bit serve: ready on http://%s:%d\n", host.c_str(), bound_port);
    listener.join();
    return 0;
}

}  // namespace onebit::embed_serve
