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

#include <cstdio>
#include <cstring>
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
                 "  version                     print the version\n"
                 "  help                        show this help\n");
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

    // Native NPU artifacts are registered by the NPU engine port (docs/PORTING.md
    // step 3); until then the onebit backend lists none.
    lemon::backends::onebit::set_onebit_models({});

    if (cli_config.port != -1) config_json["port"] = cli_config.port;
    if (!cli_config.host.empty()) config_json["host"] = cli_config.host;
    auto config = std::make_shared<lemon::RuntimeConfig>(config_json);
    lemon::RuntimeConfig::set_global(config.get());
    lemon::configure_application_logging(config->log_level(), lemon::LoggingMode::direct_server);

    lemon::Server server(config, cli_config.cache_dir, cli_config.config_dir);
    server.run();
    return 0;
}

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
