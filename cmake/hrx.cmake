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

# HRX + Vulkan in one llama.cpp build (docs/hrx.md). Three pinned sources,
# each cloned from its submodule into the build tree and patched there, so
# the submodules stay clean:
#
#   hrx_runtime  ROCm/hrx 0bc22fb + patches/hrx/0001..0003: libhrx, libloomc,
#                the Loom AMDGPU binding, assembled into ${ONEBIT_HRX_PREFIX}
#   loom_link    ROCm/hrx-system 6743075f + patches/hrx/0002: the loom-link the
#                fork's kernel catalog is linked with (the only version that
#                still has --mode=selective; its artifacts are byte-identical
#                to the working build's)
#   llama_hrx    bong-water-water-bong/llama.cpp 1bit-engine/hrx-vulkan with
#                GGML_HRX2 and GGML_VULKAN: one llama-server exposing HRX20
#                and Vulkan0
include(ExternalProject)
find_package(Git REQUIRED)

set(ONEBIT_HRX_TOOLCHAIN "/opt/rocm-therock" CACHE PATH "ROCm/TheRock root whose amdclang builds the HRX runtime")
set(ONEBIT_HRX_PREFIX "${CMAKE_BINARY_DIR}/hrx/prefix")
set(_hrx_patches "${CMAKE_SOURCE_DIR}/patches/hrx")

foreach(_sub hrx hrx-system llama.cpp)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/${_sub}/.git")
        message(FATAL_ERROR "ONEBIT_HRX needs third_party/${_sub}: git submodule update --init third_party/${_sub}")
    endif()
endforeach()

# Clone a submodule into the build tree at its pinned commit, then apply patches.
function(_hrx_fetch_command out submodule)
    set(_src "${CMAKE_SOURCE_DIR}/third_party/${submodule}")
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${_src}" rev-parse HEAD
                    OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set(_cmd ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} clone --quiet --shared --no-checkout "${_src}" <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> checkout --quiet ${_rev})
    foreach(_patch ${ARGN})
        list(APPEND _cmd COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> apply "${_hrx_patches}/${_patch}")
    endforeach()
    set(${out} ${_cmd} PARENT_SCOPE)
endfunction()

# The HRX build configuration of the working Strix Halo setup, verbatim.
set(_hrx_flags
    -DCMAKE_C_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
    -DCMAKE_CXX_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++
    -DCMAKE_ASM_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
    -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    -DHRX_ENABLE_IPO=OFF -DHRX_INSTALL_TESTS=OFF
    -DIREE_ALLOCATOR_SYSTEM=libc -DIREE_BUILD_BENCHMARKS=OFF -DIREE_BUILD_DOCS=OFF
    -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_TESTS=OFF
    -DIREE_DEPENDENCY_MODE=pinned -DIREE_DEV_MODE=OFF -DIREE_ENABLE_ASSERTIONS=OFF
    -DIREE_ENABLE_LIBBACKTRACE=OFF -DIREE_ENABLE_LLD=OFF -DIREE_ENABLE_POSITION_INDEPENDENT_CODE=ON
    -DIREE_ENABLE_RUNTIME_TRACING=OFF -DIREE_ENABLE_WERROR_FLAG=OFF
    -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt -DIREE_HAL_AMDGPU_LIBHSA_STATIC=OFF
    "-DIREE_HAL_AMDGPU_TARGETS=gfx9-generic|gfx90a|gfx9-4-generic|gfx10-1-generic|gfx10-3-generic|gfx11-generic|gfx12-generic"
    -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_TASK=ON
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_VULKAN=OFF -DIREE_HAL_DRIVER_WEBGPU=OFF
    -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=ON -DIREE_VISIBILITY_HIDDEN=ON
    -DLIBHRX_BUILD_CTS=OFF -DLIBHRX_BUILD_CUDA_BINDING=OFF -DLIBHRX_BUILD_HIP_BINDING=ON
    -DLIBHRX_BUILD_PASSTHROUGH=ON
    -DLOOM_BUILD=ON -DLOOM_EMIT_AMDGPU=OFF -DLOOM_EMIT_LLVMIR=OFF -DLOOM_EMIT_SPIRV=OFF -DLOOM_EMIT_WASM=OFF
    -DLOOM_EXECUTE_DEFAULTS=ON -DLOOM_EXECUTE_IREE_HAL=ON -DLOOM_IMPORT_MLIR=OFF -DLOOM_IMPORT_TILELANG=OFF
    -DLOOM_TARGET_AMDGPU=ON -DLOOM_TARGET_AMDGPU_TARGETS=loom_defaults -DLOOM_TARGET_ARCH_AMDGPU=OFF
    -DLOOM_TARGET_ARCH_LLVMIR=OFF -DLOOM_TARGET_ARCH_SPIRV=OFF -DLOOM_TARGET_ARCH_WASM=OFF
    -DLOOM_TARGET_ARCH_X86=OFF -DLOOM_TARGET_DEFAULTS=ON -DLOOM_TARGET_LLVMIR=ON -DLOOM_TARGET_SPIRV=ON
    -DLOOM_TARGET_WASM=OFF -DLOOM_TARGET_X86=ON)

_hrx_fetch_command(_fetch_runtime hrx
    0001-libhrx-skip-install-export.patch
    0002-loomc-drop-export-restriction.patch
    0003-amdgpu-tolerate-missing-pm4-emulation-probe.patch)
ExternalProject_Add(hrx_runtime
    PREFIX ${CMAKE_BINARY_DIR}/hrx/runtime
    DOWNLOAD_COMMAND ${_fetch_runtime}
    LIST_SEPARATOR |
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS ${_hrx_flags} -DLIBHRX_BUILD=ON
    # Only what llama.cpp's ggml-hrx2 links; the full tree has unrelated targets.
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target
        libhrx_src_libhrx_hrx loomc_shared loom_binding_c_target_amdgpu_amdgpu.objects
    INSTALL_COMMAND ${CMAKE_COMMAND} -DSRC=<SOURCE_DIR> -DBIN=<BINARY_DIR> -DPREFIX=${ONEBIT_HRX_PREFIX}
        -DAR=${CMAKE_AR} -DCONFIG_IN=${CMAKE_SOURCE_DIR}/cmake/hrx-config.cmake.in
        -P ${CMAKE_SOURCE_DIR}/cmake/hrx_assemble_prefix.cmake
    USES_TERMINAL_BUILD ON)

_hrx_fetch_command(_fetch_system hrx-system 0002-loomc-drop-export-restriction.patch)
ExternalProject_Add(loom_link
    PREFIX ${CMAKE_BINARY_DIR}/hrx/loom-link
    DOWNLOAD_COMMAND ${_fetch_system}
    LIST_SEPARATOR |
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS ${_hrx_flags} -DLIBHRX_BUILD=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target loom_tools_loom-link_loom-link
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/loom/src/loom/tools/loom-link/loom-link
        ${ONEBIT_HRX_PREFIX}/bin/loom-link
    USES_TERMINAL_BUILD ON)

set(ONEBIT_HRX_SERVER "${CMAKE_BINARY_DIR}/hrx/llama/bin/llama-server")
ExternalProject_Add(llama_hrx
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
    BINARY_DIR ${CMAKE_BINARY_DIR}/hrx/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -Dhrx_DIR=${ONEBIT_HRX_PREFIX}/cmake/hrx
        -DGGML_HRX2=ON -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
        -DGGML_HRX2_LOOM_LINK_EXECUTABLE=${ONEBIT_HRX_PREFIX}/bin/loom-link
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    DEPENDS hrx_runtime loom_link
    USES_TERMINAL_BUILD ON)
