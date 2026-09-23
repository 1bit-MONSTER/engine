#pragma once
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/model_manager.h"

#include <memory>
#include <string>

namespace lemon {
namespace backends {

// lemon-mlx-engine as an executor. Reuses LlamaCppServer's OpenAI forwarding and
// overrides load() (spawn `<server> <checkpoint> --port <p>`), plus the request
// paths that must carry the checkpoint in `model`: the MLX server selects its
// model by Hugging Face id, not by Lemonade's model name.
class MlxServer : public LlamaCppServer {
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

namespace mlx {

// The server binary: the mlx_bin option, else $LEMONADE_MLX_SERVER, else
// `lemon-mlx-server` (resolved on PATH).
std::string resolve_server_binary(const std::string& mlx_bin_option);

std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<MlxServer>(); }

}  // namespace mlx

}  // namespace backends
}  // namespace lemon
