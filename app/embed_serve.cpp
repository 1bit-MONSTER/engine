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
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace onebit::embed_serve {

bool container(const std::string& path) {
    std::error_code ec;
    return fs::is_directory(path, ec) && fs::is_regular_file(fs::path(path) / "model.embed", ec);
}

struct Backend::Impl {
    embed::Model model;
#ifdef ONEBIT_NPU
    embed::NpuGemm npu;
    bool npu_installed = false;
#endif
};

bool Backend::load(const std::string& dir, const std::string& device, std::string& why) {
    if (device != "npu" && device != "cpu") {
        why = "an embedding container runs on --device npu or cpu, not '" + device + "'";
        return false;
    }
    impl_ = std::make_shared<Impl>();
    if (!impl_->model.load(dir, why)) return false;
    device_ = device;
    dim_ = impl_->model.config().out_dim;
    id_ = fs::path(dir).filename().string();
#ifdef ONEBIT_NPU
    if (device == "npu") {
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

int Backend::max_tokens() const { return impl_ ? impl_->model.config().max_tokens : 0; }

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
    if (!srv.bind_to_port(host, port)) {
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
    listener.join();
    return 0;
}

}  // namespace onebit::embed_serve
