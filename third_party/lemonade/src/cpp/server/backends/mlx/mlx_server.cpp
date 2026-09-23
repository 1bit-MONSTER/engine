// mlx_server.cpp — MLX models on Apple Silicon through lemon-mlx-engine (local delta, see UPSTREAM.md).
#include "lemon/backends/mlx/mlx_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/mlx/mlx.h"
#include "lemon/utils/process_manager.h"

#include <lemon/utils/aixlog.hpp>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

namespace {

class MlxOps : public BackendOps {
public:
    std::string resolve_checkpoint_path(const ModelInfo&,
                                        const CheckpointResolveContext& ctx) const override {
        // The checkpoint is a Hugging Face id; the MLX server resolves and caches it.
        return ctx.checkpoint;
    }

    bool is_downloaded(const ModelInfo&, const BackendOpsContext&) const override {
        return true;  // the MLX server downloads on first load (self_manages_downloads)
    }
};

// Report the caller's model name, not the checkpoint the MLX server was given.
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

void MlxServer::load(const std::string& model_name,
                     const ModelInfo& model_info,
                     const RecipeOptions& options,
                     bool do_not_upgrade) {
    (void)do_not_upgrade;
    LOG(INFO, "MLX") << "Loading model: " << model_name << std::endl;

    const std::string checkpoint = model_info.checkpoint();
    if (checkpoint.empty()) {
        throw std::runtime_error("MLX model '" + model_name + "' has no checkpoint (Hugging Face id)");
    }

    const std::string server = mlx::resolve_server_binary(options.get_option("mlx_bin"));
    const int backend_port = choose_port();

    // The positional model argument preloads it, so /health means "ready to serve".
    std::vector<std::string> argv = {checkpoint, "--port", std::to_string(backend_port)};
    for (auto& a : split_args(options.get_option("mlx_args"))) argv.push_back(std::move(a));

    const bool inherit_output = (log_level_ == "info") || is_debug();
    set_process_handle(
        utils::ProcessManager::start_process(server, argv, "", inherit_output, true, {}),
        server,
        argv);

    if (!wait_for_ready("/health")) {
        const ProcessHandle handle = consume_process_handle_for_cleanup();
        if (has_process_handle(handle)) {
            utils::ProcessManager::stop_process(handle);
        }
        throw std::runtime_error("MLX server failed to start for '" + model_name + "' (" + server + ")");
    }
    LOG(DEBUG, "MLX") << "Model loaded on port " << get_backend_port() << std::endl;
}

json MlxServer::chat_completion(const json& request) {
    json r = request;
    r["model"] = checkpoint_;
    return restore_model_name(LlamaCppServer::chat_completion(r), request);
}

json MlxServer::completion(const json& request) {
    json r = request;
    r["model"] = checkpoint_;
    return restore_model_name(LlamaCppServer::completion(r), request);
}

void MlxServer::forward_streaming_request(const std::string& endpoint,
                                          const std::string& request_body,
                                          httplib::DataSink& sink,
                                          bool sse,
                                          long timeout_seconds,
                                          TelemetryCallback telemetry_callback) {
    std::string body = request_body;
    try {
        json r = json::parse(request_body);
        r["model"] = checkpoint_;
        body = r.dump();
    } catch (const json::exception&) {
        // not JSON: forward unchanged
    }
    LlamaCppServer::forward_streaming_request(endpoint, body, sink, sse, timeout_seconds,
                                              telemetry_callback);
}

namespace mlx {

std::string resolve_server_binary(const std::string& mlx_bin_option) {
    if (!mlx_bin_option.empty()) return mlx_bin_option;
    if (const char* env = std::getenv("LEMONADE_MLX_SERVER"); env && *env) return env;
    return "lemon-mlx-server";
}

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<MlxServer>(ctx);
}

// Not a Lemonade download: the server comes from mlx_bin / $LEMONADE_MLX_SERVER / PATH.
const BackendSpec* spec() { return nullptr; }

const BackendOps* ops() { return single_ops<MlxOps>(); }

}  // namespace mlx

}  // namespace backends
}  // namespace lemon
