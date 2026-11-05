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

#include "route_serve.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "laya_checkpoint.h"
#include "policy.h"
#include "route.h"
#include "scorer.h"

namespace onebit {
namespace {

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// One line out, flushed: the client reads a decision per line and must not wait for the next one.
void write_line(const nlohmann::ordered_json& obj) {
    std::cout << obj.dump() << '\n' << std::flush;
}

}  // namespace

std::string route_json_line(const std::string& cls, float confidence, const std::string& device,
                            double decision_ms) {
    return nlohmann::ordered_json{{"class", cls},
                                  {"confidence", confidence},
                                  {"device", device},
                                  {"decision_ms", decision_ms}}
        .dump();
}

int run_route_serve(const std::string& laya_model_arg, const std::vector<std::string>& devices,
                    const std::string& policy_path) {
    // Same resolution chain as the one-shot command and `serve --laya` (app/laya_checkpoint.h).
    const std::string laya_model = laya_checkpoint(laya_model_arg);

    const laya::RoutePolicy policy = laya::load_route_policy(policy_path);
    const auto t0 = std::chrono::steady_clock::now();
    laya::Scorer scorer;
    if (!scorer.load(laya_model)) {
        std::fprintf(stderr, "1bit route --serve: %s\n", scorer.error().c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    // Without this line a client cannot tell a slow scorer load from a slow decision, and the
    // first request's decision_ms would look like routing cost when it is warmup.
    write_line({{"ready", true}, {"load_ms", ms_between(t0, t1)}});

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        nlohmann::json req;
        try {
            req = nlohmann::json::parse(line);
        } catch (const std::exception& e) {
            write_line({{"error", std::string("bad json: ") + e.what()}});
            continue;
        }
        if (!req.is_object() || !req.contains("state") || !req["state"].is_string()) {
            write_line({{"error", std::string("each line must be {\"state\": \"<request text>\"}")}});
            continue;
        }
        const std::string state = req["state"].get<std::string>();
        const auto t2 = std::chrono::steady_clock::now();
        const laya::RequestClass cls = laya::classify_request(scorer, state);
        if (cls.label.empty()) {  // the scorer failed; answer this line and keep serving
            write_line({{"error", scorer.error()}});
            continue;
        }
        const std::string device = policy.pick(cls.label, cls.confidence, devices);
        const auto t3 = std::chrono::steady_clock::now();
        std::cout << route_json_line(cls.label, cls.confidence, device, ms_between(t2, t3)) << '\n' << std::flush;
    }
    return 0;  // EOF on stdin: the client is gone
}

}  // namespace onebit
