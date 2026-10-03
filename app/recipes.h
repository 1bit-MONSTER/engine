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

// app/recipes.h — tuned llama-server settings per model and route, as data (docs/recipes.md).
// config/recipes.json is compiled in as the default; `serve --recipes FILE` replaces it and
// `serve --no-recipes` turns recipes off. A recipe names what it matches, the flags and
// environment it adds, why, and the measurement behind it. It never overrides a flag already on
// the backend's command line or a variable already in the environment.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace onebit {

// What a recipe can match on, for one backend launch.
struct RecipeFacts {
    std::string device;             // hrx, cpu
    std::string architecture;       // general.architecture of the -m file
    bool moe = false;               // <architecture>.expert_count > 0
    bool hadamard_q4_0 = false;     // stamped by tools/hadamard_q4_0.py
    std::string drafter = "none";   // none, mtp, dflash
};

struct Recipe {
    std::string id, why, measured, source;
    std::vector<std::string> device, architecture, drafter;  // empty: any
    int moe = -1, hadamard_q4_0 = -1;                         // -1: any, else 0 / 1
    std::vector<std::string> args;                            // llama-server flags and values
    std::vector<std::pair<std::string, std::string>> env;     // backend environment

    bool matches(const RecipeFacts& f) const;
};

struct Recipes {
    std::vector<Recipe> list;

    // Parses recipe JSON; on failure returns false and leaves the reason in `err`.
    bool parse(const std::string& json_text, std::string& err);

    // Adds each matching recipe's flags (a flag with the values after it) unless `argv` already
    // has that flag, and its variables unless `env` or the process environment already sets
    // them. Returns "id: what it added" for every recipe that added something.
    std::vector<std::string> apply(const RecipeFacts& f, std::vector<std::string>& argv,
                                   std::vector<std::string>& env) const;
};

// config/recipes.json as built into this binary.
const char* default_recipes_json();

}  // namespace onebit
