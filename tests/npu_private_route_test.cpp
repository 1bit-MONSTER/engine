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

// npu_private_route_test: the private-route registry (npu/private_route.h). With nothing
// registered, a Qwen3.6-35B-A3B directory's model_type gets the "not part of this build"
// message and other types none; a registered route and command are found (a second
// registration replaces the first); --npu-opt KEY=VALUE parses.
#include "../npu/private_route.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

using namespace onebit::npu;

int fails = 0;

void check(const char* name, bool ok) {
    fails += !ok;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

bool throws(const std::string& kv) {
    PrivateOptions o;
    try {
        parse_private_option(kv, o);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("npu_private_route_test." + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::ofstream(dir / "config.json") << R"({"model_type": "qwen3_5_moe", "num_hidden_layers": 40})";
    check("model_type_of reads config.json", model_type_of(dir.string()) == "qwen3_5_moe");
    check("model_type_of: no config.json", model_type_of((dir / "absent").string()).empty());

    const std::string missing = private_route_missing("qwen3_5_moe");
    std::printf("     %s\n", missing.c_str());
    check("35B without the add-on: names the model",
          missing.find("Qwen3.6-35B-A3B NPU route is not part of this build") != std::string::npos);
    check("35B without the add-on: names the option", missing.find("-DONEBIT_NPU_PRIVATE=") != std::string::npos);
    check("dense Qwen3: no message", private_route_missing("qwen3").empty());
    check("no routes, no commands", !find_private_route("qwen3_5_moe") && private_commands().empty());

    register_private_route({"first", "qwen3_5_moe", {}, {}});
    register_private_route({"second", "qwen3_5_moe",
                            [](const std::string&, const PrivateOptions& o) { return o.count("k") ? "" : "no k"; }, {}});
    const PrivateRoute* r = find_private_route("qwen3_5_moe");
    check("a registered route is found", r != nullptr);
    check("the later registration replaces the earlier", r && r->name == "second");
    check("route availability sees the options", r && r->unavailable("", {{"k", "v"}}).empty() &&
                                                     r->unavailable("", {}) == "no k");
    check("with the route: no message", private_route_missing("qwen3_5_moe").empty());

    register_private_command({"x", "x help", [](int argc, char**) { return argc; }});
    const PrivateCommand* c = find_private_command("x");
    check("a registered command is found and runs", c && c->run(3, nullptr) == 3 && private_commands().size() == 1);
    check("an unknown command is not", !find_private_command("y"));

    PrivateOptions o;
    parse_private_option("dir=/a=b", o);
    parse_private_option("n=", o);
    check("KEY=VALUE splits at the first '='", o["dir"] == "/a=b" && o.count("n") && o["n"].empty());
    check("no '=' or an empty key throws", throws("dir") && throws("=x"));

    fs::remove_all(dir);
    return fails;
}
