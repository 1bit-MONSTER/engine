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
//
// zinc_server.cpp: GGUF models through ZINC (local delta, see UPSTREAM.md).
#include "lemon/backends/zinc/zinc_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/zinc/zinc.h"
#include "lemon/utils/process_manager.h"

#include <lemon/utils/aixlog.hpp>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lemon {
namespace backends {

namespace {

// Checkpoints are GGUF files, so model management (variant resolution, GGUF
// metadata, cache validation) is llamacpp's; only the runtime differs.
class ZincOps : public BackendOps {
public:
    void populate_metadata(ModelInfo& info, const BackendOpsContext& ctx) const override {
        llamacpp::ops()->populate_metadata(info, ctx);
    }
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& ctx) const override {
        return llamacpp::ops()->resolve_checkpoint_path(info, ctx);
    }
    std::string find_imported_checkpoint(const std::string& import_dir) const override {
        return llamacpp::ops()->find_imported_checkpoint(import_dir);
    }
    std::string validate_registration_checkpoint(const std::string& checkpoint) const override {
        return llamacpp::ops()->validate_registration_checkpoint(checkpoint);
    }
    std::string validate_checkpoint_file(const std::string& resolved_path) const override {
        return llamacpp::ops()->validate_checkpoint_file(resolved_path);
    }
};

// zinc accepts only its own model id, and omitting `model` selects the one model
// it serves.
json without_model(json request) {
    if (request.is_object()) request.erase("model");
    return request;
}

// Report the caller's model name, not zinc's id for the GGUF.
json restore_model_name(json response, const json& request) {
    if (response.is_object() && response.contains("model") && request.contains("model")) {
        response["model"] = request["model"];
    }
    return response;
}

std::vector<std::string> split_args(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string a; in >> a;) out.push_back(a);
    return out;
}

}  // namespace

void ZincServer::load(const std::string& model_name,
                      const ModelInfo& model_info,
                      const RecipeOptions& options,
                      bool do_not_upgrade) {
    (void)do_not_upgrade;
    LOG(INFO, "ZINC") << "Loading model: " << model_name << std::endl;

    const std::string gguf = model_info.resolved_path();
    if (gguf.empty()) {
        throw std::runtime_error("ZINC model '" + model_name + "' has no resolved GGUF");
    }

    const std::string server = zinc::resolve_server_binary(options.get_option("zinc_bin"));
    const int backend_port = choose_port();
    const int ctx_size = options.get_option("ctx_size");

    // No --prompt: zinc starts its HTTP server with the model preloaded, so
    // /health means "ready to serve".
    std::vector<std::string> argv = {"-m", gguf, "-p", std::to_string(backend_port)};
    if (ctx_size > 0) {
        argv.push_back("-c");
        argv.push_back(std::to_string(ctx_size));
    }
    for (auto& a : split_args(options.get_option("zinc_args"))) argv.push_back(std::move(a));

    // zinc's Vulkan kernels are tuned for RADV's cooperative-matrix path; keep a
    // value the user already set.
    std::vector<std::pair<std::string, std::string>> env;
    if (const char* radv = std::getenv("RADV_PERFTEST"); !radv || !*radv) {
        env.emplace_back("RADV_PERFTEST", "coop_matrix");
    }

    const bool inherit_output = (log_level_ == "info") || is_debug();
    set_process_handle(
        utils::ProcessManager::start_process(server, argv, "", inherit_output, true, env),
        server,
        argv);

    if (!wait_for_ready("/health")) {
        const ProcessHandle handle = consume_process_handle_for_cleanup();
        if (has_process_handle(handle)) {
            utils::ProcessManager::stop_process(handle);
        }
        throw std::runtime_error("ZINC failed to start for '" + model_name + "' (" + server + ")");
    }
    LOG(DEBUG, "ZINC") << "Model loaded on port " << get_backend_port() << std::endl;
}

json ZincServer::chat_completion(const json& request) {
    return restore_model_name(LlamaCppServer::chat_completion(without_model(request)), request);
}

json ZincServer::completion(const json& request) {
    return restore_model_name(LlamaCppServer::completion(without_model(request)), request);
}

void ZincServer::forward_streaming_request(const std::string& endpoint,
                                           const std::string& request_body,
                                           httplib::DataSink& sink,
                                           bool sse,
                                           long timeout_seconds,
                                           TelemetryCallback telemetry_callback) {
    std::string body = request_body;
    try {
        body = without_model(json::parse(request_body)).dump();
    } catch (const json::exception&) {
        // not JSON: forward unchanged
    }
    LlamaCppServer::forward_streaming_request(endpoint, body, sink, sse, timeout_seconds,
                                              telemetry_callback);
}

namespace zinc {

std::string resolve_server_binary(const std::string& zinc_bin_option) {
    if (!zinc_bin_option.empty()) return zinc_bin_option;
    if (const char* env = std::getenv("LEMONADE_ZINC_SERVER"); env && *env) return env;
    return "zinc";
}

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<ZincServer>(ctx);
}

// Not a Lemonade download: the binary comes from zinc_bin / $LEMONADE_ZINC_SERVER / PATH.
const BackendSpec* spec() { return nullptr; }

const BackendOps* ops() { return single_ops<ZincOps>(); }

}  // namespace zinc

}  // namespace backends
}  // namespace lemon
