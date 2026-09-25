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

#include "private_route.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace onebit::npu {

namespace {

std::vector<PrivateRoute>& routes() {
    static std::vector<PrivateRoute> r;
    return r;
}

std::vector<PrivateCommand>& commands() {
    static std::vector<PrivateCommand> c;
    return c;
}

// The private routes that exist, by model_type, so a build without them can say so.
const std::map<std::string, std::string>& known() {
    static const std::map<std::string, std::string> k = {{"qwen3_5_moe", "Qwen3.6-35B-A3B"}};
    return k;
}

}  // namespace

void register_private_route(PrivateRoute route) {
    for (auto& r : routes())
        if (r.model_type == route.model_type) {
            r = std::move(route);
            return;
        }
    routes().push_back(std::move(route));
}

void register_private_command(PrivateCommand command) {
    for (auto& c : commands())
        if (c.name == command.name) {
            c = std::move(command);
            return;
        }
    commands().push_back(std::move(command));
}

const PrivateRoute* find_private_route(const std::string& model_type) {
    for (const auto& r : routes())
        if (r.model_type == model_type) return &r;
    return nullptr;
}

const PrivateRoute* route_for(const std::string& dir, const PrivateOptions& opts, std::string* why) {
    why->clear();
    const PrivateRoute* r = find_private_route(model_type_of(dir));
    if (!r) return nullptr;
    const std::string no = r->unavailable ? r->unavailable(dir, opts) : "";
    if (no.empty()) return r;
    if (!r->optional) *why = no;
    return nullptr;
}

const PrivateCommand* find_private_command(const std::string& name) {
    for (const auto& c : commands())
        if (c.name == name) return &c;
    return nullptr;
}

const std::vector<PrivateCommand>& private_commands() { return commands(); }

std::string model_type_of(const std::string& dir) {
    std::ifstream f(dir + "/config.json");
    if (!f) return "";
    try {
        return nlohmann::json::parse(f).value("model_type", "");
    } catch (const std::exception&) {
        return "";
    }
}

std::string private_model_name(const std::string& model_type) {
    const auto it = known().find(model_type);
    return it == known().end() ? "" : it->second;
}

std::string private_route_missing(const std::string& model_type) {
    if (find_private_route(model_type)) return "";
    const std::string name = private_model_name(model_type);
    if (name.empty()) return "";
    return "the " + name + " NPU route is not part of this build; build with "
           "-DONEBIT_NPU_PRIVATE=<npu-kernels checkout> (docs/npu.md, \"Private routes\")";
}

void parse_private_option(const std::string& kv, PrivateOptions& opts) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos || eq == 0) throw std::runtime_error("--npu-opt takes KEY=VALUE, not '" + kv + "'");
    opts[kv.substr(0, eq)] = kv.substr(eq + 1);
}

}  // namespace onebit::npu
