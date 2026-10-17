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

// tests/pm_routes_test.cpp: the `--pm-router laya` class -> expert table and the --pm-router
// values (docs/pm.md, "Routing with Laya"). No GPU, no Laya checkpoint, no Lemonade: this is
// app/pm.cpp's config and the mapping it feeds the engine's router.

#include "pm.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

using onebit::pm::Config;
using onebit::pm::Expert;
using onebit::pm::load_config;
using onebit::pm::parse_router;
using onebit::pm::route_expert;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static bool throws(const std::string& file) {
    try {
        load_config(file, "http://127.0.0.1:8000");
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static std::string write_file(const char* name, const std::string& text) {
    std::ofstream f(name);
    f << text;
    f.close();
    return name;
}

int main() {
    // The table is shipped in config/pm-experts.json, i.e. compiled into the binary.
    const Config c = load_config("", "http://127.0.0.1:8000");
    check(c.router == "model", "the default router is the model, as before --pm-router");
    check(c.routes.size() == 4, "the built-in table covers every Laya class (code, prose, short, long_doc)");
    check(c.routes.count("code") == 1 && c.routes.at("code") == "code", "built-in: code -> code");
    check(c.routes.count("long_doc") == 1 && c.routes.at("long_doc") == "code-heavy", "built-in: long_doc -> code-heavy");
    check(c.routes.count("prose") == 1 && c.routes.at("prose") == "reasoning", "built-in: prose -> reasoning");
    check(c.routes.count("short") == 1 && c.routes.at("short").empty(), "built-in: short is left to the PM");
    check(c.routes_min_confidence > 0.0f && c.routes_min_confidence < 1.0f, "the built-in floor is a fraction");

    // route_expert: what the engine asks before the PM's first round.
    check(route_expert(c, "code", 0.9f) == c.find("code"), "a confident code class picks the code expert");
    check(route_expert(c, "prose", 0.9f) == c.find("reasoning"), "a confident prose class picks the reasoning expert");
    check(route_expert(c, "long_doc", 0.9f) == c.find("code-heavy"), "long_doc picks the code-heavy expert");
    check(route_expert(c, "short", 1.0f) == nullptr, "short has no expert: the PM answers itself");
    check(route_expert(c, "no_such_class", 1.0f) == nullptr, "an unrouted class leaves the answer to the PM");
    check(route_expert(c, "code", c.routes_min_confidence) == c.find("code"), "at the floor the class still routes");
    check(route_expert(c, "code", c.routes_min_confidence - 0.01f) == nullptr, "below the floor the PM answers");

    // --pm-router values.
    check(parse_router("") == "model", "--pm-router unset is the model router");
    check(parse_router("model") == "model", "--pm-router model");
    check(parse_router("laya") == "laya", "--pm-router laya");
    bool threw = false;
    try {
        parse_router("bogus");
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("--pm-router takes model or laya") != std::string::npos;
    }
    check(threw, "--pm-router takes model or laya and nothing else");

    // The table is data: another file replaces it, and the floor moves with it.
    const std::string good = write_file("pm_routes_good.json", R"JSON({
      "pm": {"routes": {"code": "worker", "short": ""}, "routes_min_confidence": 0.5},
      "experts": [{"expert": "worker", "id": "some-coder-1bit", "domain": "code", "use_when": "code", "max_tokens": 128}]
    })JSON");
    const Config g = load_config(good, "http://127.0.0.1:8000");
    check(g.routes.size() == 2 && g.routes.at("code") == "worker", "a custom file's table replaces the built-in one");
    check(g.routes_min_confidence == 0.5f, "a custom file's confidence floor is read");
    check(route_expert(g, "code", 0.6f) == g.find("worker"), "the custom route picks its expert");
    check(route_expert(g, "code", 0.4f) == nullptr, "the custom floor is applied");
    check(route_expert(g, "prose", 0.9f) == nullptr, "a class the custom table leaves out stays with the PM");
    check(g.router == "model", "an experts file does not choose the router: the flag does");

    // A route that names an expert the file does not list is a file error, not a silent no-op.
    check(throws(write_file("pm_routes_unknown.json", R"JSON({
      "pm": {"routes": {"code": "nope"}},
      "experts": [{"expert": "code", "id": "x"}]
    })JSON")), "a route naming an unlisted expert is refused");
    check(throws(write_file("pm_routes_type.json", R"JSON({"pm": {"routes": {"code": 7}}, "experts": []})JSON")),
          "a non-string route is refused");
    check(throws(write_file("pm_routes_obj.json", R"JSON({"pm": {"routes": []}, "experts": []})JSON")),
          "routes that are not an object are refused");
    check(throws(write_file("pm_routes_conf.json", R"JSON({"pm": {"routes_min_confidence": 2}, "experts": []})JSON")),
          "a confidence floor outside 0..1 is refused");

    std::printf("%s\n", failures == 0 ? "pm_routes: all checks passed" : "pm_routes: FAILURES");
    return failures == 0 ? 0 : 1;
}
