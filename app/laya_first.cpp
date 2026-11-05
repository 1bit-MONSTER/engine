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

#include "laya_first.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "laya_checkpoint.h"     // laya_home(): the same ${XDG_DATA_HOME:-~/.local/share}/1bit root
#include "laya_first_default.h"  // kDefaultLayaFirstJson, from config/laya-first.json

namespace onebit {
namespace {

using json = nlohmann::ordered_json;

// The classes Laya answers, in the order the policy declares its rules (docs/laya-first.md §4).
const std::vector<std::string>& classes() {
    static const std::vector<std::string> order = {"short", "prose", "code", "long_doc"};
    return order;
}

const std::vector<std::string>& devices() {
    static const std::vector<std::string> known = {"hrx", "npu", "hybrid", "auto"};
    return known;
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    for (const auto& item : v)
        if (item == s) return true;
    return false;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The table: --laya-first FILE, else the installed <laya home dir>/laya-first.json, else the copy
// compiled in from config/laya-first.json (the same pattern as config/recipes.json and
// config/pm-experts.json).
std::string load_table(const std::string& given) {
    namespace fs = std::filesystem;
    if (!given.empty()) return read_file(given);
    const std::string installed = laya_home("") + "/laya-first.json";
    if (fs::exists(installed)) return read_file(installed);
    return kDefaultLayaFirstJson;
}

// Everything the table promises must be checkable here rather than discovered by a user whose
// request landed on a class with a typo'd device.
void validate(const json& table) {
    if (!table.is_object()) throw std::runtime_error("the table must be a JSON object");
    if (table.value("version", "") != "1")
        throw std::runtime_error("version must be \"1\"");
    if (!table.contains("classes") || !table["classes"].is_object() || table["classes"].empty())
        throw std::runtime_error("classes must be a non-empty object");
    for (const auto& name : classes()) {
        if (!table["classes"].contains(name))
            throw std::runtime_error("classes is missing '" + name + "' (Laya names exactly these)");
    }
    for (auto it = table["classes"].begin(); it != table["classes"].end(); ++it) {
        const std::string name = it.key();
        const json& cls = it.value();
        if (!contains(classes(), name))
            throw std::runtime_error("classes.'" + name + "' is not a Laya class (short, prose, code, long_doc)");
        if (!cls.is_object()) throw std::runtime_error("classes.'" + name + "' must be an object");
        for (const char* key : {"model", "hf_repo", "device"}) {
            if (!cls.contains(key) || !cls[key].is_string() || cls[key].get<std::string>().empty())
                throw std::runtime_error("classes.'" + name + "'.model/hf_repo/device are required strings");
        }
        const std::string device = cls["device"].get<std::string>();
        if (!contains(devices(), device))
            throw std::runtime_error("classes.'" + name + "'.device '" + device +
                                     "' is not one of hrx, npu, hybrid, auto");
        if (!cls.contains("ctx") || !cls["ctx"].is_number_integer() || cls["ctx"].get<long>() <= 0)
            throw std::runtime_error("classes.'" + name + "'.ctx must be a positive integer");
        if (cls.contains("on_busy")) {
            const json& busy = cls["on_busy"];
            if (!busy.is_object() || !busy.contains("device") || !busy["device"].is_string())
                throw std::runtime_error("classes.'" + name + "'.on_busy.device is required");
            if (!contains(devices(), busy["device"].get<std::string>()))
                throw std::runtime_error("classes.'" + name + "'.on_busy.device is not one of hrx, npu, hybrid, auto");
        }
    }
    if (table.contains("scorer")) {
        const json& scorer = table["scorer"];
        if (!scorer.is_object() || !scorer.contains("hf_repo") || !scorer.contains("file"))
            throw std::runtime_error("scorer must name hf_repo and file");
    }
    const std::string default_model = table.value("default_model", "");
    if (default_model.empty())
        throw std::runtime_error("default_model is required (the model a cold, unroutable request uses)");
    bool known = false;
    for (auto it = table["classes"].begin(); it != table["classes"].end(); ++it)
        if (it.value().value("model", "") == default_model) known = true;
    if (!known)
        throw std::runtime_error("default_model '" + default_model + "' is not one of the classes' models");
}

// The policy Lemonade already understands: one rule per class, `route_to` that class's model.
// `min_score: 0` is deliberate — Laya's calibrated confidence for `code` is about 0.44, so the
// default band (0.5) would drop the very decisions this table exists to route. A class is a class
// whatever Laya's confidence in it; an unroutable request still falls open to default_model.
json desugar(const json& table) {
    json components = json::array();
    json rules = json::array();
    for (const auto& name : classes()) {
        const json& cls = table["classes"][name];
        const std::string model = cls["model"].get<std::string>();
        components.push_back(model);
        rules.push_back(json{{"id", "laya-" + name},
                             {"match", json{{"classifier", "laya"}, {"label", name}, {"min_score", 0.0}}},
                             {"route_to", model}});
    }
    // The `laya` classifier entry itself: labels are the four classes, which is also what makes a
    // typo in a rule's `label` a parse error rather than a rule that never matches.
    json classifier = json{{"id", "laya"}, {"type", "laya"}, {"labels", classes()}};

    return json{{"version", "1"},
                {"model_name", "1bit.Laya-First"},
                {"recipe", "collection.router"},
                {"components", components},
                {"routing",
                 json{{"candidates", components},
                      {"default_model", table["default_model"]},
                      {"classifiers", json::array({classifier})},
                      {"rules", rules}}}};
}

void usage(std::FILE* out) {
    std::fprintf(out,
                 "usage: 1bit laya-first [--laya-first FILE] [--policy FILE] [--check] [--json]\n"
                 "  Reads the Laya-first routing table (config/laya-first.json, compiled in;\n"
                 "  --laya-first FILE replaces it, else ${XDG_DATA_HOME:-~/.local/share}/1bit/laya-first.json\n"
                 "  when it exists) and prints the collection.router policy it desugars into:\n"
                 "  one rule per class (short, prose, code, long_doc), route_to that class's model.\n"
                 "  --policy FILE  write the policy there instead of stdout\n"
                 "  --check        validate the table only; exit 0 when it is usable\n");
}

}  // namespace

int run_laya_first(int argc, char** argv) {
    std::string table_path, policy_out;
    bool check = false;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--laya-first") table_path = next();
        else if (a == "--policy") policy_out = next();
        else if (a == "--check") check = true;
        else if (a == "--json") continue;  // the output is JSON; accepted for symmetry with `route`
        else if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }

    const json table = json::parse(load_table(table_path), nullptr, /*allow_exceptions=*/false);
    if (table.is_discarded()) throw std::runtime_error("the table is not valid JSON");
    validate(table);
    if (check) {
        std::fprintf(stderr, "1bit laya-first: %zu classes, default %s\n", classes().size(),
                     table["default_model"].get<std::string>().c_str());
        return 0;
    }

    const std::string policy = desugar(table).dump(2) + "\n";
    if (policy_out.empty()) {
        std::fwrite(policy.data(), 1, policy.size(), stdout);
        return 0;
    }
    std::ofstream out(policy_out, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + policy_out);
    out << policy;
    if (!out) throw std::runtime_error("cannot write " + policy_out);
    std::fprintf(stderr, "1bit laya-first: policy written to %s\n", policy_out.c_str());
    return 0;
}

}  // namespace onebit
