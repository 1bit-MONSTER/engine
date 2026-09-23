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
// 1bit: the engine binary. Subcommands:
//
//   1bit lemonade [lemond options]   Lemonade's server core, embedded in-process,
//                                    with the engine's `onebit` backend.
//   1bit unified -m <model dir>      one native model on the NPU fast lane behind
//                                    OpenAI endpoints; what the onebit backend spawns.
//   1bit npu-run [options]           token ids in, token ids out, on the NPU fast
//                                    lane (docs/npu.md); for checks and benchmarks.
//
// Ported from 1bit-MONSTER tools/onebin.cpp and tools/unified_server.cpp
// (run_embedded_lemonade). See docs/PORTING.md for what lands next.
#include <lemon/backends/onebit/onebit_server.h>
#include <lemon/cli_parser.h>
#include <lemon/config_file.h>
#include <lemon/logging_config.h>
#include <lemon/runtime_config.h>
#include <lemon/server.h>
#include <lemon/utils/path_utils.h>

#include <nlohmann/json.hpp>

#ifdef ONEBIT_NPU
#include "generate.h"
#include "model.h"
#include "unified.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char* kVersion = "0.0.1";

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit <command> [options]\n"
                 "\n"
                 "commands:\n"
                 "  lemonade [lemond options]   run Lemonade's server with the engine behind it\n"
#ifdef ONEBIT_NPU
                 "  unified -m <model dir>      serve one NPU model (OpenAI endpoints)\n"
                 "  npu-run [options]           generate on the NPU fast lane (npu-run --help)\n"
#endif
                 "  version                     print the version\n"
                 "  help                        show this help\n");
}

#ifdef ONEBIT_HRX_SERVER
// Points Lemonade at this build's llama-server (HRX + Vulkan in one binary,
// docs/hrx.md): the llamacpp-hrx recipe runs on HRX0 (AMD's ggml-hrx), and the
// llamacpp recipe runs GGUF on Vulkan0, the faster device for standard quants. The
// config arrives merged with Lemonade's defaults, so only default values
// ("builtin", "auto", unset) are replaced; anything else the user set wins.
void use_engine_hrx_build(nlohmann::json& config) {
    auto set_default = [&](const char* section, const char* key, const char* default_value, const char* value) {
        auto& sec = config[section];
        if (!sec.is_object()) sec = nlohmann::json::object();
        if (!sec.contains(key) || sec[key] == "" || sec[key] == default_value) sec[key] = value;
    };
    set_default("hrx", "hrx_bin", "builtin", ONEBIT_HRX_SERVER);
    set_default("hrx", "device", "", "HRX0");
    set_default("llamacpp", "vulkan_bin", "builtin", ONEBIT_HRX_SERVER);
    set_default("llamacpp", "backend", "auto", "vulkan");
    set_default("llamacpp", "device", "", "Vulkan0");
    // HRX dlopens the HSA runtime; the distro's rejects gfx1151's PM4-emulation
    // probe and HRX registers no device. llama-server inherits this environment,
    // so point it at TheRock's unless the user already chose one.
    setenv("IREE_HAL_AMDGPU_LIBHSA_PATH", ONEBIT_HRX_LIBHSA, /*overwrite=*/0);
}
#endif

#ifdef ONEBIT_ZINC_SERVER
// Points Lemonade's zinc recipe at this build's zinc (third_party/zinc,
// docs/zinc.md) unless the user already chose a binary.
void use_engine_zinc_build(nlohmann::json& config) {
    auto& sec = config["zinc"];
    if (!sec.is_object()) sec = nlohmann::json::object();
    if (!sec.contains("zinc_bin") || sec["zinc_bin"] == "") sec["zinc_bin"] = ONEBIT_ZINC_SERVER;
}
#endif

// Native models for the `onebit` backend: every NPU model directory
// (docs/npu.md) directly under the roots in ONEBIT_MODEL_ROOTS (colon-separated,
// default ~/models). Lemonade serves each through `1bit unified -m <dir>`. The
// architecture registry (docs/PORTING.md step 5) replaces this scan.
std::vector<lemon::ModelInfo> native_models() {
    std::vector<lemon::ModelInfo> out;
#ifdef ONEBIT_NPU
    std::string roots;
    if (const char* r = std::getenv("ONEBIT_MODEL_ROOTS"); r && *r) roots = r;
    else if (const char* h = std::getenv("HOME"); h && *h) roots = std::string(h) + "/models";
    std::istringstream in(roots);
    for (std::string root; std::getline(in, root, ':');) {
        std::error_code ec;
        for (const auto& d : std::filesystem::directory_iterator(root, ec)) {
            if (!d.is_directory() || !onebit::is_npu_model_dir(d.path().string())) continue;
            lemon::ModelInfo mi;
            mi.model_name = d.path().filename().string();
            mi.recipe = "onebit";
            mi.checkpoints["main"] = d.path().string();
            mi.resolved_paths["main"] = d.path().string();
            mi.downloaded = true;
            mi.source = "engine";
            mi.labels = {"chat", "npu"};
            out.push_back(std::move(mi));
        }
    }
    std::fprintf(stderr, "1bit: %zu NPU model(s) registered with the onebit backend (roots: %s)\n", out.size(),
                 roots.c_str());
#endif
    return out;
}

// Hands argv to Lemonade's own CLI and runs its server in this process. The
// engine's native models reach Lemonade through the `onebit` backend
// (third_party/lemonade/src/cpp/include/lemon/backends/onebit/onebit.h); they
// are injected here before the server is constructed, so ModelManager's
// dynamic discovery sees them on its first cache build.
int run_lemonade(int argc, char** argv) {
    lemon::CLIParser parser;
    parser.parse(argc, argv);
    if (!parser.should_continue()) return parser.get_exit_code();
    auto cli_config = parser.get_config();

    lemon::utils::set_cache_dir(cli_config.cache_dir);
    auto config_json = lemon::ConfigFile::load(cli_config.cache_dir);

#ifdef ONEBIT_HRX_SERVER
    use_engine_hrx_build(config_json);
#endif
#ifdef ONEBIT_ZINC_SERVER
    use_engine_zinc_build(config_json);
#endif

    lemon::backends::onebit::set_onebit_models(native_models());

    if (cli_config.port != -1) config_json["port"] = cli_config.port;
    if (!cli_config.host.empty()) config_json["host"] = cli_config.host;
    auto config = std::make_shared<lemon::RuntimeConfig>(config_json);
    lemon::RuntimeConfig::set_global(config.get());
    lemon::configure_application_logging(config->log_level(), lemon::LoggingMode::direct_server);

    lemon::Server server(config, cli_config.cache_dir, cli_config.config_dir);
    server.run();
    return 0;
}

#ifdef ONEBIT_NPU
// Token ids in, token ids out. Prints one id per line on stdout and the timing
// on stderr; --dump-logits writes each step's logits as float32
// (decode_logits_idx<N>.bin, N = decode step), for comparison with a reference.
int run_npu(int argc, char** argv) {
    std::string model_dir, kernel_dir, ids, dump;
    onebit::npu::GenerateOptions opt;
    opt.max_tokens = 24;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model_dir = next();
        else if (a == "--kernels") kernel_dir = next();
        else if (a == "--ids") ids = next();
        else if (a == "-n") opt.max_tokens = std::stoi(next());
        else if (a == "--repetition-penalty") opt.repetition_penalty = std::stof(next());
        else if (a == "--no-eos") opt.stop_at_eos = false;
        else if (a == "--dump-logits") dump = next();
        else if (a == "--help" || a == "-h") {
            std::printf("usage: 1bit npu-run --model <dir> --kernels <dir> --ids \"<id> <id> ...\"\n"
                        "                    [-n 24] [--repetition-penalty 1.1] [--no-eos] [--dump-logits <dir>]\n"
                        "  --model    directory with model.q4nx and config.json\n"
                        "  --kernels  directory with layer_ctx{1,2,17}.elf, lmhead.elf and layer.pdi\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (model_dir.empty() || kernel_dir.empty() || ids.empty()) throw std::runtime_error("--model, --kernels and --ids are required");
    std::vector<int> prompt;
    std::istringstream in(ids);
    for (int t; in >> t;) prompt.push_back(t);

    const onebit::npu::Model model(model_dir);
    auto t0 = std::chrono::steady_clock::now();
    onebit::npu::Lane lane(model, kernel_dir);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> lg;
    if (!dump.empty())
        opt.on_logits = [&](int step, onebit::npu::Lane& l) {
            l.logits(lg);
            std::ofstream(dump + "/decode_logits_idx" + std::to_string(step) + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * sizeof(float)));
        };
    const auto r = onebit::npu::generate(lane, model, prompt, opt);
    for (int t : r.tokens) std::printf("%d\n", t);
    std::fprintf(stderr, "load %.0f ms | prefill %zu tokens %.1f ms | decode %zu tokens %.1f ms/token (%.1f tok/s)%s\n",
                 load_ms, prompt.size(), r.prefill_ms, r.tokens.size(), r.decode_ms / double(r.tokens.size()),
                 1000.0 * double(r.tokens.size()) / r.decode_ms, r.stopped_at_eos ? " | stopped at EOS" : "");
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "lemonade") {
        // Lemonade's parser sees "1bit" followed by the lemond options.
        std::vector<char*> args{argv[0]};
        for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
        return run_lemonade(static_cast<int>(args.size()), args.data());
    }
#ifdef ONEBIT_NPU
    if (cmd == "unified") {
        try {
            return onebit::run_unified(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit unified: %s\n", e.what());
            return 1;
        }
    }
    if (cmd == "npu-run") {
        try {
            return run_npu(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit npu-run: %s\n", e.what());
            return 1;
        }
    }
#endif
    if (cmd == "version" || cmd == "--version") {
        std::printf("1bit %s\n", kVersion);
        return 0;
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        usage(stdout);
        return 0;
    }
    std::fprintf(stderr, "1bit: unknown command '%s'\n\n", cmd.c_str());
    usage(stderr);
    return 2;
}
