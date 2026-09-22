# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third-party dependencies, each pinned to a release (or commit) and its
# sha256. All are header-only and MIT-licensed; see NOTICE.
include(FetchContent)

FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

# SOURCE_SUBDIR points at a directory that does not exist, so FetchContent
# only downloads these and does not run their own CMake projects.
FetchContent_Declare(cpp_httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.57.1.tar.gz
    URL_HASH SHA256=5c9e56b6638eb415dc451ac57531e84931fcb50e261ca3dd74eec6b484fb4712
    SOURCE_SUBDIR _none)
FetchContent_Declare(minja
    URL https://github.com/google/minja/archive/021c2293c187789ef13d56c6cfd89c9b134fd80f.tar.gz
    URL_HASH SHA256=dc3ddd37497b79a4cd35fd41550e22b3b0139439de8be6eab3d2d024fed47bb4
    SOURCE_SUBDIR _none)

FetchContent_MakeAvailable(nlohmann_json cpp_httplib minja)

add_library(onebit_httplib INTERFACE)
target_include_directories(onebit_httplib SYSTEM INTERFACE ${cpp_httplib_SOURCE_DIR})
add_library(onebit_minja INTERFACE)
target_include_directories(onebit_minja SYSTEM INTERFACE ${minja_SOURCE_DIR}/include)
target_link_libraries(onebit_minja INTERFACE nlohmann_json::nlohmann_json)
