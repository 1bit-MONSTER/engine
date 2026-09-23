#pragma once
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
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/model_manager.h"

#include <memory>
#include <string>

namespace lemon {
namespace backends {

// ZINC as an executor. Reuses LlamaCppServer's OpenAI forwarding and overrides
// load() (spawn `zinc -m <gguf> -p <port> -c <ctx>`), plus the request paths:
// zinc serves one model and rejects any `model` id but its own, so requests go
// out without `model` and responses carry Lemonade's name back.
class ZincServer : public LlamaCppServer {
public:
    using LlamaCppServer::LlamaCppServer;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    json chat_completion(const json& request) override;
    json completion(const json& request) override;

    void forward_streaming_request(const std::string& endpoint,
                                   const std::string& request_body,
                                   httplib::DataSink& sink,
                                   bool sse = true,
                                   long timeout_seconds = 0,
                                   TelemetryCallback telemetry_callback = nullptr) override;
};

namespace zinc {

// The zinc binary: the zinc_bin option, else $LEMONADE_ZINC_SERVER, else `zinc`
// (resolved on PATH).
std::string resolve_server_binary(const std::string& zinc_bin_option);

std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<ZincServer>(); }

}  // namespace zinc

}  // namespace backends
}  // namespace lemon
