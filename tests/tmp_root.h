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

// tmp_root.h -- where a test puts its scratch.
//
// /tmp on this box is **tmpfs**, i.e. RAM, and swap has been full for hours (owner directive, 2026-11-10 21:15Z). A test
// that writes containers there -- `npu_pack_import_test` builds multi-hundred-MB packs, and every crafted 1.9 import
// writes a canonical `.fconst` copy -- spends memory that a GPU/NPU lane needs, and a leaked directory stays spent until
// something deletes it.
//
// So scratch goes on disk under `~/.cache/scratch-tests/tmp` unless `ONEBIT_TEST_TMP` names somewhere else; `temp_directory_path()`
// is the last resort (no HOME, unwritable cache), never the default.
#pragma once

#include <cstdlib>
#include <filesystem>

namespace onebit::test {

inline std::filesystem::path tmp_root() {
    if (const char* e = std::getenv("ONEBIT_TEST_TMP"); e && *e) return std::filesystem::path(e);
    if (const char* h = std::getenv("HOME"); h && *h) {
        std::error_code ec;
        const std::filesystem::path p = std::filesystem::path(h) / ".cache" / "scratch-tests" / "tmp";
        std::filesystem::create_directories(p, ec);
        // An existing directory is what we want; a create that failed because it is already there is fine too.
        if (std::filesystem::is_directory(p, ec) && !ec) return p;
    }
    return std::filesystem::temp_directory_path();
}

}  // namespace onebit::test
