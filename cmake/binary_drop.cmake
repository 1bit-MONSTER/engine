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
# The public binary drop (docs/releases.md, "The public binary drop").
#
# A source build outside the org has no private HIP kernels and no private NPU runtime source, so it
# cannot compile those parts itself. The drop carries them already compiled, for one exact engine
# pin, under the 1bit Binary License. ONEBIT_BINARY_DROP names it: a directory (unpacked, or the
# tarball's own directory) or the tarball's URL.
#
# Everything is verified before use and a mismatch is fatal:
#   URL    the sha256 pinned in config/binary-drop.config for this engine commit (or
#          -DONEBIT_BINARY_DROP_SHA256 when the pin file has no line yet) is the expected hash of
#          the download; file(DOWNLOAD) refuses a body that does not match.
#   path   the drop's MANIFEST lists a sha256 per file and the engine commit it was built from;
#          a file that does not match, or a drop built from another commit, stops the configure.
#
# On success this sets, for the two consumers:
#   ONEBIT_BINARY_DROP_DIR          the unpacked drop root
#   ONEBIT_BINARY_DROP_HRX_BIN      <root>/hrx/llama/bin      (cmake/hrx.cmake)
#   ONEBIT_BINARY_DROP_NPU_LIB      <root>/npu/libnpu_rt_api.a (the NPU block in CMakeLists.txt)
#   ONEBIT_BINARY_DROP_NPU_INCLUDE  <root>/npu/include
# and leaves them unset otherwise, so a build without a drop behaves exactly as before.

set(ONEBIT_BINARY_DROP "" CACHE STRING
    "Path or URL of the public binary drop to link instead of compiling the private parts (docs/releases.md)")
set(ONEBIT_BINARY_DROP_SHA256 "" CACHE STRING
    "sha256 of the drop tarball, for a URL whose engine pin has no line in config/binary-drop.config")
mark_as_advanced(ONEBIT_BINARY_DROP_SHA256)

set(ONEBIT_BINARY_DROP_DIR "")
set(ONEBIT_BINARY_DROP_HRX_BIN "")
set(ONEBIT_BINARY_DROP_NPU_LIB "")
set(ONEBIT_BINARY_DROP_NPU_INCLUDE "")

if(NOT ONEBIT_BINARY_DROP)
    return()
endif()

# The engine commit this build is from: the drop is built for exactly one.
set(_drop_engine_sha "")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
                        OUTPUT_VARIABLE _drop_engine_sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
endif()

# config/binary-drop.config: the url pattern and one sha256 per published engine commit.
set(_drop_url_pattern "")
set(_drop_pinned_sha "")
if(EXISTS "${CMAKE_SOURCE_DIR}/config/binary-drop.config")
    file(STRINGS "${CMAKE_SOURCE_DIR}/config/binary-drop.config" _drop_cfg)
    foreach(_line IN LISTS _drop_cfg)
        if(_line MATCHES "^url_pattern=(.+)$")
            set(_drop_url_pattern "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^drop=([0-9a-f]{40})[ \t]+([^ \t]+)[ \t]+([0-9a-f]{64})[ \t]*$")
            if(_drop_engine_sha STREQUAL CMAKE_MATCH_1)
                set(_drop_pinned_sha "${CMAKE_MATCH_3}")
                set(_drop_pinned_version "${CMAKE_MATCH_2}")
            endif()
        endif()
    endforeach()
endif()

set(_drop_verify_manifest FALSE)
if(IS_DIRECTORY "${ONEBIT_BINARY_DROP}")
    set(ONEBIT_BINARY_DROP_DIR "${ONEBIT_BINARY_DROP}")
    set(_drop_verify_manifest TRUE)
elseif(ONEBIT_BINARY_DROP MATCHES "^[a-zA-Z][a-zA-Z0-9+.-]*://")
    # A URL: download it, with the pinned sha256 as the expected hash.
    if(NOT _drop_pinned_sha)
        if(ONEBIT_BINARY_DROP_SHA256)
            set(_drop_pinned_sha "${ONEBIT_BINARY_DROP_SHA256}")
            message(STATUS "ONEBIT_BINARY_DROP: pin file has no line for this engine commit; using ONEBIT_BINARY_DROP_SHA256")
        else()
            message(FATAL_ERROR
                "ONEBIT_BINARY_DROP is a URL but config/binary-drop.config pins no drop for engine commit "
                "'${_drop_engine_sha}'. Add its line (drop=<sha> <version> <sha256>), or pass "
                "-DONEBIT_BINARY_DROP_SHA256=<sha256> (docs/releases.md, \"The public binary drop\").")
        endif()
    endif()
    set(_drop_dir "${CMAKE_BINARY_DIR}/_binary_drop")
    set(_drop_tgz "${_drop_dir}/binary-drop.tar.zst")
    file(MAKE_DIRECTORY "${_drop_dir}")
    message(STATUS "ONEBIT_BINARY_DROP: downloading ${ONEBIT_BINARY_DROP}")
    file(DOWNLOAD "${ONEBIT_BINARY_DROP}" "${_drop_tgz}"
         EXPECTED_HASH SHA256=${_drop_pinned_sha}
         TLS_VERIFY ON STATUS _drop_dl_status)
    list(GET _drop_dl_status 0 _drop_dl_code)
    if(NOT _drop_dl_code EQUAL 0)
        list(GET _drop_dl_status 1 _drop_dl_msg)
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: download failed (${_drop_dl_msg}); the pinned sha256 "
                            "was ${_drop_pinned_sha}. Refusing to build against an unverified drop.")
    endif()
    file(MAKE_DIRECTORY "${_drop_dir}/root")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf "${_drop_tgz}"
                    WORKING_DIRECTORY "${_drop_dir}/root"
                    RESULT_VARIABLE _drop_untar)
    if(NOT _drop_untar EQUAL 0)
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: cannot unpack ${_drop_tgz}")
    endif()
    file(GLOB _drop_roots "${_drop_dir}/root/binary-drop-*")
    list(LENGTH _drop_roots _drop_n)
    if(NOT _drop_n EQUAL 1)
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: the tarball does not hold one binary-drop-* directory")
    endif()
    list(GET _drop_roots 0 ONEBIT_BINARY_DROP_DIR)
    set(_drop_verify_manifest TRUE)
else()
    message(FATAL_ERROR "ONEBIT_BINARY_DROP='${ONEBIT_BINARY_DROP}' is neither a directory nor a URL")
endif()

# Verify: the MANIFEST lists a sha256 per file and the engine commit it was built from.
if(_drop_verify_manifest)
    if(NOT EXISTS "${ONEBIT_BINARY_DROP_DIR}/MANIFEST")
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: ${ONEBIT_BINARY_DROP_DIR} has no MANIFEST; not a binary drop")
    endif()
    file(STRINGS "${ONEBIT_BINARY_DROP_DIR}/MANIFEST" _drop_lines)
    set(_drop_manifest_sha "")
    set(_drop_checked 0)
    foreach(_line IN LISTS _drop_lines)
        if(_line MATCHES "^engine_sha=([0-9a-f]{40})$")
            set(_drop_manifest_sha "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^([0-9a-f]{64})  ([0-9]+)  (.+)$")
            set(_drop_hash "${CMAKE_MATCH_1}")
            set(_drop_file "${CMAKE_MATCH_3}")
            file(SHA256 "${ONEBIT_BINARY_DROP_DIR}/${_drop_file}" _drop_actual)
            if(NOT _drop_actual STREQUAL _drop_hash)
                message(FATAL_ERROR "ONEBIT_BINARY_DROP: ${_drop_file} does not match its MANIFEST sha256 "
                                    "(expected ${_drop_hash}, got ${_drop_actual}). Refusing to link it.")
            endif()
            math(EXPR _drop_checked "${_drop_checked} + 1")
        endif()
    endforeach()
    if(_drop_checked EQUAL 0)
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: the MANIFEST lists no files")
    endif()
    if(_drop_engine_sha AND _drop_manifest_sha AND NOT _drop_manifest_sha STREQUAL _drop_engine_sha)
        message(FATAL_ERROR "ONEBIT_BINARY_DROP: ${ONEBIT_BINARY_DROP_DIR} was built for engine commit "
                            "${_drop_manifest_sha}, but this checkout is ${_drop_engine_sha}. The drop is per "
                            "engine pin; build the matching one (docs/releases.md).")
    endif()
    message(STATUS "ONEBIT_BINARY_DROP: ${ONEBIT_BINARY_DROP_DIR}: ${_drop_checked} files verified against MANIFEST")
endif()

if(EXISTS "${ONEBIT_BINARY_DROP_DIR}/hrx/llama/bin/llama-server")
    set(ONEBIT_BINARY_DROP_HRX_BIN "${ONEBIT_BINARY_DROP_DIR}/hrx/llama/bin")
endif()
if(EXISTS "${ONEBIT_BINARY_DROP_DIR}/npu/libnpu_rt_api.a")
    set(ONEBIT_BINARY_DROP_NPU_LIB "${ONEBIT_BINARY_DROP_DIR}/npu/libnpu_rt_api.a")
    set(ONEBIT_BINARY_DROP_NPU_INCLUDE "${ONEBIT_BINARY_DROP_DIR}/npu/include")
endif()
