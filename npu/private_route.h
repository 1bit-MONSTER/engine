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

// Private NPU routes: models the engine runs on the NPU through a closed-source add-on
// (docs/npu.md, "Private routes"). A build configured with -DONEBIT_NPU_PRIVATE=<checkout>
// links the add-on, and `1bit` calls its register_private_addon() at startup; the add-on
// registers a route per model_type it serves, and optionally `1bit` subcommands. Without
// the add-on nothing is registered, and a model directory of one of those types gets
// private_route_missing()'s message.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace onebit::npu {

class Model;
class Tokenizer;

// One request, as `1bit unified` hands it to a route.
struct PrivateRequest {
    int max_tokens = 256;
    float repetition_penalty = 1.0f;
    int penalty_window = 64;
    std::function<bool(int token)> on_token;  // each generated token; false stops
};

struct PrivateResult {
    std::vector<int> tokens;
    double prefill_ms = 0, decode_ms = 0;
    bool stopped_at_eos = false;
    size_t reused = 0;   // prompt tokens already on the device
    std::string cache;   // where the reused tokens came from ("live", ...); empty when none
};

// A model loaded on a private route. `1bit unified` runs one request at a time.
class PrivateSession {
public:
    virtual ~PrivateSession() = default;
    virtual PrivateResult generate(const std::vector<int>& prompt, const PrivateRequest& req) = 0;
    virtual int max_context() const = 0;
    virtual float default_repetition_penalty() const { return 1.0f; }
    // The model's chat template opens "<think>\n" itself when thinking is on.
    virtual bool opens_think() const { return false; }
    // One line for the load log.
    virtual std::string describe() const { return ""; }
};

// `1bit serve --npu-opt KEY=VALUE` (repeatable), passed through to the route.
using PrivateOptions = std::map<std::string, std::string>;

struct PrivateRoute {
    std::string name;        // for logs
    std::string model_type;  // the config.json model_type it serves
    // Why this route cannot serve dir with opts (e.g. a missing file); empty if it can.
    std::function<std::string(const std::string& dir, const PrivateOptions& opts)> unavailable;
    std::function<std::unique_ptr<PrivateSession>(const Model&, const Tokenizer&, const std::string& dir,
                                                  const PrivateOptions& opts)>
        open;
};

struct PrivateCommand {
    std::string name;  // `1bit <name> ...`
    std::string help;  // one line for `1bit --help`
    std::function<int(int argc, char** argv)> run;  // argv holds the options after the name
};

// A later registration for the same model_type (or command name) replaces the earlier one.
void register_private_route(PrivateRoute route);
void register_private_command(PrivateCommand command);
const PrivateRoute* find_private_route(const std::string& model_type);
const PrivateCommand* find_private_command(const std::string& name);
const std::vector<PrivateCommand>& private_commands();

// config.json's model_type in dir; empty if there is none.
std::string model_type_of(const std::string& dir);
// The model a private route exists for, by model_type ("Qwen3.6-35B-A3B" for qwen3_5_moe);
// empty if model_type has none.
std::string private_model_name(const std::string& model_type);
// Why a directory of this model_type cannot run here: empty if a route is registered for
// it or it is not a private route's type; otherwise that the route is not in this build.
std::string private_route_missing(const std::string& model_type);
// "KEY=VALUE" -> opts[KEY] = VALUE; throws without '=' or with an empty key.
void parse_private_option(const std::string& kv, PrivateOptions& opts);

// Defined by the add-on; `1bit` calls it once at startup in -DONEBIT_NPU_PRIVATE builds.
void register_private_addon();

}  // namespace onebit::npu
