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
#
# Hugging Face tokenizers (docs/tokenizers.md): builds hf_tokenizers/ (our C ABI
# over third_party/tokenizers) with cargo, --locked, into the build tree, and
# exposes it as the onebit_hf_tokenizers target. rustup honours
# hf_tokenizers/rust-toolchain.toml, which pins the Rust release.
find_program(CARGO cargo REQUIRED)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/tokenizers/tokenizers/Cargo.toml")
    message(FATAL_ERROR "ONEBIT_HF_TOKENIZERS needs third_party/tokenizers: git submodule update --init third_party/tokenizers")
endif()
set(_hf_target_dir "${CMAKE_BINARY_DIR}/hf_tokenizers")
set(_hf_lib "${_hf_target_dir}/release/${CMAKE_STATIC_LIBRARY_PREFIX}onebit_hf_tokenizers${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE _hf_sources CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/hf_tokenizers/src/*.rs")
add_custom_command(
    OUTPUT "${_hf_lib}"
    COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${_hf_target_dir}
            ${CARGO} build --release --locked --manifest-path ${CMAKE_SOURCE_DIR}/hf_tokenizers/Cargo.toml
    DEPENDS ${_hf_sources} ${CMAKE_SOURCE_DIR}/hf_tokenizers/Cargo.toml ${CMAKE_SOURCE_DIR}/hf_tokenizers/Cargo.lock
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/hf_tokenizers
    COMMENT "cargo build hf_tokenizers (Hugging Face tokenizers)"
    USES_TERMINAL)
add_custom_target(hf_tokenizers_build DEPENDS "${_hf_lib}")
add_library(onebit_hf_tokenizers STATIC IMPORTED GLOBAL)
set_target_properties(onebit_hf_tokenizers PROPERTIES IMPORTED_LOCATION "${_hf_lib}")
target_include_directories(onebit_hf_tokenizers INTERFACE "${CMAKE_SOURCE_DIR}/hf_tokenizers/include")
target_link_libraries(onebit_hf_tokenizers INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
add_dependencies(onebit_hf_tokenizers hf_tokenizers_build)
