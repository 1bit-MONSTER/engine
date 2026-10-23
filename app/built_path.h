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

// A backend the build compiled in, named RELATIVE to the build dir (CMake writes
// file(RELATIVE_PATH ...) for every such define). scripts/package-linux.sh keeps that layout
// beside the executable, and a 1bit run from the build tree finds it the same way, because the
// binary lives in the build dir: either way the path resolves from the executable's own
// directory. Nothing machine-specific is compiled in, so a shipped binary that has no build tree
// on disk names no build-machine path (scripts/weekly.sh package_checks asserts this).
inline std::string built_path(const char* compiled) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p = compiled;
#if defined(__linux__)
    if (!p.is_absolute()) {
        const fs::path here = fs::read_symlink("/proc/self/exe", ec).parent_path();
        if (!ec && fs::exists(here / p, ec)) return (here / p).string();
    }
#endif
    return std::string(compiled);   // an absolute (older build) or a path with no file beside us
}
