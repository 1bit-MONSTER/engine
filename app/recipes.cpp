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

#include "recipes.h"

#include "recipes_default.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>

namespace onebit {

namespace {

bool any_of(const std::vector<std::string>& allowed, const std::string& v) {
    return allowed.empty() || std::find(allowed.begin(), allowed.end(), v) != allowed.end();
}

// A token that starts a flag: "-x" or "--xy", not a negative number.
bool is_flag(const std::string& t) {
    return t.size() > 1 && t[0] == '-' && !std::isdigit(static_cast<unsigned char>(t[1])) && t[1] != '.';
}

std::vector<std::string> strings(const nlohmann::json& j, const char* key, const std::string& id) {
    if (!j.is_array()) throw std::runtime_error("recipe " + id + ": " + key + " is not a list");
    std::vector<std::string> out;
    for (const auto& v : j) {
        if (!v.is_string()) throw std::runtime_error("recipe " + id + ": " + key + " holds a non-string");
        out.push_back(v.get<std::string>());
    }
    return out;
}

}  // namespace

bool Recipe::matches(const RecipeFacts& f) const {
    if (!any_of(device, f.device) || !any_of(architecture, f.architecture) || !any_of(drafter, f.drafter))
        return false;
    if (moe >= 0 && (moe == 1) != f.moe) return false;
    return true;
}

bool Recipes::parse(const std::string& json_text, std::string& err) {
    list.clear();
    try {
        const auto doc = nlohmann::json::parse(json_text);
        if (!doc.is_object() || !doc.contains("recipes") || !doc["recipes"].is_array())
            throw std::runtime_error("expected an object with a \"recipes\" list");
        std::set<std::string> ids;
        for (const auto& r : doc["recipes"]) {
            if (!r.is_object() || !r.contains("id") || !r["id"].is_string())
                throw std::runtime_error("every recipe needs a string \"id\"");
            Recipe rec;
            rec.id = r["id"].get<std::string>();
            if (!ids.insert(rec.id).second) throw std::runtime_error("recipe " + rec.id + " is listed twice");
            for (const auto& [key, v] : r.items()) {
                if (key == "id") continue;
                else if (key == "why" || key == "measured" || key == "source") {
                    if (!v.is_string()) throw std::runtime_error("recipe " + rec.id + ": " + key + " is not a string");
                    (key == "why" ? rec.why : key == "measured" ? rec.measured : rec.source) = v.get<std::string>();
                } else if (key == "args") {
                    rec.args = strings(v, "args", rec.id);
                } else if (key == "env") {
                    if (!v.is_object()) throw std::runtime_error("recipe " + rec.id + ": env is not an object");
                    for (const auto& [name, value] : v.items()) {
                        if (!value.is_string()) throw std::runtime_error("recipe " + rec.id + ": env " + name + " is not a string");
                        rec.env.emplace_back(name, value.get<std::string>());
                    }
                } else if (key == "match") {
                    if (!v.is_object()) throw std::runtime_error("recipe " + rec.id + ": match is not an object");
                    for (const auto& [mk, mv] : v.items()) {
                        if (mk == "device") rec.device = strings(mv, "match.device", rec.id);
                        else if (mk == "architecture") rec.architecture = strings(mv, "match.architecture", rec.id);
                        else if (mk == "drafter") rec.drafter = strings(mv, "match.drafter", rec.id);
                        else if (mk == "moe") {
                            if (!mv.is_boolean()) throw std::runtime_error("recipe " + rec.id + ": match." + mk + " is not true/false");
                            rec.moe = mv.get<bool>() ? 1 : 0;
                        } else {
                            throw std::runtime_error("recipe " + rec.id + ": unknown match key \"" + mk + "\"");
                        }
                    }
                } else {
                    throw std::runtime_error("recipe " + rec.id + ": unknown key \"" + key + "\"");
                }
            }
            if (rec.args.empty() && rec.env.empty()) throw std::runtime_error("recipe " + rec.id + " adds nothing");
            if (!rec.args.empty() && !is_flag(rec.args.front()))
                throw std::runtime_error("recipe " + rec.id + ": args must start with a flag");
            if (rec.measured.empty()) throw std::runtime_error("recipe " + rec.id + " has no \"measured\"");
            list.push_back(std::move(rec));
        }
    } catch (const std::exception& e) {
        err = e.what();
        list.clear();
        return false;
    }
    return true;
}

std::vector<std::string> Recipes::apply(const RecipeFacts& f, std::vector<std::string>& argv,
                                        std::vector<std::string>& env) const {
    std::vector<std::string> applied;
    for (const auto& r : list) {
        if (!r.matches(f)) continue;
        std::string added;
        for (size_t i = 0; i < r.args.size();) {
            size_t j = i + 1;
            while (j < r.args.size() && !is_flag(r.args[j])) ++j;
            if (std::find(argv.begin(), argv.end(), r.args[i]) == argv.end()) {
                for (size_t k = i; k < j; ++k) {
                    argv.push_back(r.args[k]);
                    added += (added.empty() ? "" : " ") + r.args[k];
                }
            }
            i = j;
        }
        for (const auto& [name, value] : r.env) {
            const bool set = std::getenv(name.c_str()) != nullptr ||
                             std::any_of(env.begin(), env.end(), [&](const std::string& e) { return e.rfind(name + "=", 0) == 0; });
            if (set) continue;
            env.push_back(name + "=" + value);
            added += (added.empty() ? "" : " ") + name + "=" + value;
        }
        if (!added.empty()) applied.push_back(r.id + ": " + added);
    }
    return applied;
}

const char* default_recipes_json() { return kDefaultRecipesJson; }

}  // namespace onebit
