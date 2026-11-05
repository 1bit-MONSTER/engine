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

#include "laya_checkpoint.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

namespace onebit {

std::string laya_home(const std::string& given) {
    std::string dir = given;
    if (dir.empty()) {
        if (const char* e = std::getenv("ONEBIT_LAYA_MODEL"); e && *e) dir = e;
        else if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) dir = std::string(x) + "/1bit/laya";
        else if (const char* h = std::getenv("HOME"); h && *h) dir = std::string(h) + "/.local/share/1bit/laya";
    }
    return dir;
}

std::string laya_checkpoint(const std::string& given) {
    namespace fs = std::filesystem;
    const std::string dir = laya_home(given);
    if (dir.empty() || !fs::is_directory(dir))
        throw std::runtime_error("laya: no checkpoint at " + (dir.empty() ? std::string("~/.local/share/1bit/laya") : dir) +
                                 " (install it with scripts/fetch-laya.sh, or pass --laya-model DIR)");
    if (fs::exists(fs::path(dir) / "typed-decisions" / "model.safetensors")) return (fs::path(dir) / "typed-decisions").string();
    return dir;
}

}  // namespace onebit
