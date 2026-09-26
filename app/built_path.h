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
#pragma once

#include <filesystem>
#include <string>
#include <system_error>

// A backend the build compiled in by its absolute path in the build tree
// (ONEBIT_BUILD_DIR/<rel>). A packaged 1bit (scripts/package-linux.sh) keeps that relative
// layout beside the executable, so when the build tree is gone, <rel> is found there.
inline std::string built_path(const char* compiled) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(compiled, ec)) return compiled;
#if defined(ONEBIT_BUILD_DIR) && defined(__linux__)
    const fs::path rel = fs::path(compiled).lexically_relative(ONEBIT_BUILD_DIR);
    if (!rel.empty() && *rel.begin() != "..") {
        const fs::path here = fs::read_symlink("/proc/self/exe", ec).parent_path();
        if (!ec && fs::exists(here / rel, ec)) return (here / rel).string();
    }
#endif
    return compiled;
}
