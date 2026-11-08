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

# The HRX llama.cpp build (docs/hrx.md). Two pinned sources, the pair AMD's integration repo
# (ROCm/ggml-staging-automation) builds and tests together:
#
#   third_party/llama.cpp   1bit-MONSTER/llama.cpp 1bit/hrx-vulkan-patched: AMD's
#                           hrx-graph-develop-v2 (ggml-hrx) on ggml-org llama.cpp, plus ours
#   third_party/hrx-system  1bit-MONSTER/hrx-system 1bit/main: ROCm/hrx-system (libhrx, loomc and
#                           the Loom tools) at AMD's pin, plus our Loom/libhrx commits
#
# llama.cpp's ggml-hrx builds hrx-system itself when given HRX_SOURCE_DIR, with
# the compilers of this build, so the only inputs are the two trees and TheRock.
# IREE_ROCM_PATH is deliberately not passed: it switches hrx-system to "package"
# mode, which requires TheRock's own aqlprofile-sdk headers (absent from our
# /opt/rocm-therock); unset, hrx-system fetches its pinned HSA/AQL headers.
# .github/workflows/bump-hrx.yml moves the llama.cpp pin when AMD moves its pair; hrx-system is
# rebased onto AMD's new commit by hand (our commits are few).
# Only HRX0 and the CPU are built: no Vulkan, ROCm/HIP or CUDA backend (RFC #213 stage 3). The
# branch name still says "vulkan" (AMD's pair carried both); the engine no longer builds it.
include(ExternalProject)

# The toolchain (docs/hrx.md, "The toolchain"; engine#713): the pinned copy under the tree first,
# because the build must need nothing outside the repository (scripts/fetch-rocm-sdk.sh fills it at
# the pin config/hrx-toolchain.config names), then the box's TheRock, then a distro ROCm.
# The tree copy is resolved from the pin itself, version and target, not from a glob: a glob takes
# whatever sorts first, so after a version bump the old SDK would win silently (pi-prs's review of
# #771). -DONEBIT_HRX_TOOLCHAIN=<dir>, or ROCM_PATH through the build scripts, overrides all three.
if(NOT ONEBIT_HRX_TOOLCHAIN)
    set(_hrx_toolchain_pinned "")
    set(_hrx_pin "${CMAKE_SOURCE_DIR}/config/hrx-toolchain.config")
    if(EXISTS "${_hrx_pin}")
        file(STRINGS "${_hrx_pin}" _hrx_ver REGEX "^version=")
        file(STRINGS "${_hrx_pin}" _hrx_gfx REGEX "^gfx=")
        list(LENGTH _hrx_ver _hrx_ver_n)
        list(LENGTH _hrx_gfx _hrx_gfx_n)
        if(_hrx_ver_n GREATER 0 AND _hrx_gfx_n GREATER 0)
            list(GET _hrx_ver 0 _hrx_ver)
            list(GET _hrx_gfx 0 _hrx_gfx)
            string(REPLACE "version=" "" _hrx_ver "${_hrx_ver}")
            string(REPLACE "gfx=" "" _hrx_gfx "${_hrx_gfx}")
            set(_hrx_pinned_dir "${CMAKE_SOURCE_DIR}/third_party/toolchain/rocm-sdk-${_hrx_ver}-${_hrx_gfx}")
            if(EXISTS "${_hrx_pinned_dir}/bin/amdclang")
                set(_hrx_toolchain_pinned "${_hrx_pinned_dir}")
            endif()
        endif()
    endif()
    if(_hrx_toolchain_pinned)
        set(_hrx_toolchain_default "${_hrx_toolchain_pinned}")
    elseif(EXISTS "/opt/rocm-therock")
        set(_hrx_toolchain_default "/opt/rocm-therock")
    else()
        set(_hrx_toolchain_default "/opt/rocm")
    endif()
    set(ONEBIT_HRX_TOOLCHAIN "${_hrx_toolchain_default}" CACHE PATH "ROCm/TheRock root: its amdclang builds llama.cpp and HRX")
endif()
message(STATUS "ONEBIT_HRX: toolchain ${ONEBIT_HRX_TOOLCHAIN}")

# The HRX binaries carry strings from this tree, from the add-on checkout (-DONEBIT_GPU_PRIVATE), and from
# the BUILD tree: ggml-hrx builds its deps as a nested ExternalProject that fetches Loom into
# <build>/hrx/llama/ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build/loom, which is where libloomc.so.0's
# __FILE__ strings come from (engine #1461: 7 of the payload's 146 build-machine paths, and mapping only
# the source tree left every one of them absolute). All three are rewritten so the shipped libraries name
# no build-machine path. -ffile-prefix-map covers debug info as well, which the Release -g needs.
set(ONEBIT_HRX_PATH_MAP_FLAGS "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    CACHE STRING "Path mapping for the HRX llama.cpp build")
if(ONEBIT_GPU_PRIVATE)
    string(APPEND ONEBIT_HRX_PATH_MAP_FLAGS " -ffile-prefix-map=${ONEBIT_GPU_PRIVATE}=addon")
endif()
string(APPEND ONEBIT_HRX_PATH_MAP_FLAGS " -ffile-prefix-map=${CMAKE_BINARY_DIR}=build")
# Flags the HRX llama.cpp build needs from this toolchain, kept apart from the path maps above.
set(ONEBIT_HRX_EXTRA_FLAGS "")

# A source build outside the org has no private HIP kernels to compile, so it links the prebuilt
# HRX tree from the public binary drop instead (docs/releases.md, "The public binary drop"): the
# pinned llama-server and its libraries, libggml-hrx.so with those kernels compiled in. Nothing is
# built here in that case, so neither TheRock nor the llama.cpp/hrx-system submodules are needed;
# HRX still dlopens TheRock's HSA runtime on the machine that runs it.
if(ONEBIT_BINARY_DROP_HRX_BIN AND EXISTS "${ONEBIT_BINARY_DROP_HRX_BIN}/llama-server")
    if(ONEBIT_GPU_PRIVATE)
        message(FATAL_ERROR "ONEBIT_HRX: set either ONEBIT_GPU_PRIVATE (compile the private kernels here) "
                            "or ONEBIT_BINARY_DROP (use them prebuilt), not both (docs/releases.md)")
    endif()
    if(NOT ONEBIT_HRX_LIBHSA)
        file(GLOB_RECURSE _drop_libhsa "${ONEBIT_HRX_TOOLCHAIN}/*/libhsa-runtime64.so.1")
        set(_drop_libhsa_first "")
        if(_drop_libhsa)
            list(SORT _drop_libhsa)
            list(GET _drop_libhsa 0 _drop_libhsa_first)
        endif()
        if(_drop_libhsa_first)
            # A cache value, not a compile definition: CMakeLists.txt compiles nothing from it (engine#1453 - baking a
            # build-machine path put /home/<builder>/.../libhsa-runtime64.so.1 in the c33 tarball and failed its own
            # package gate), but os/mini/mkimage.sh reads it to know which runtime to ship. ONEBIT_HRX_LIBHSA_BAKE is
            # the opt-in that does go into the binary.
            set(ONEBIT_HRX_LIBHSA "${_drop_libhsa_first}" CACHE FILEPATH
                "TheRock HSA runtime HRX loads (IREE_HAL_AMDGPU_LIBHSA_PATH); build metadata, not baked in" FORCE)
        else()
            set(ONEBIT_HRX_LIBHSA "" CACHE FILEPATH
                "TheRock HSA runtime HRX loads (IREE_HAL_AMDGPU_LIBHSA_PATH)" FORCE)
            message(STATUS "ONEBIT_HRX: no libhsa-runtime64.so.1 under ${ONEBIT_HRX_TOOLCHAIN}; the build "
                           "proceeds, but HRX needs TheRock's on the machine that runs it (pass "
                           "-DONEBIT_HRX_LIBHSA=<path> to bake it in)")
        endif()
    endif()
    set(ONEBIT_HRX_SERVER "${ONEBIT_BINARY_DROP_HRX_BIN}/llama-server" CACHE FILEPATH
        "HRX llama-server 1bit runs" FORCE)
    set(ONEBIT_HRX_ALLOW_NO_SVM 1)
    message(STATUS "ONEBIT_HRX: prebuilt from the binary drop: ${ONEBIT_HRX_SERVER} (HSA runtime: ${ONEBIT_HRX_LIBHSA})")
    return()
endif()

foreach(_sub hrx-system llama.cpp)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/${_sub}/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_HRX needs third_party/${_sub}: git submodule update --init third_party/${_sub}")
    endif()
endforeach()

# HRX dlopens the HSA runtime. The distro's libhsa-runtime64 rejects the
# HSA_AMD_AGENT_INFO_PM4_EMULATION probe on gfx1151, so HRX registers no device;
# TheRock's answers it. `1bit lemonade` passes this path to llama-server as
# IREE_HAL_AMDGPU_LIBHSA_PATH (docs/hrx.md).
file(GLOB_RECURSE _hrx_libhsa "${ONEBIT_HRX_TOOLCHAIN}/*/libhsa-runtime64.so.1")
list(SORT _hrx_libhsa)
list(GET _hrx_libhsa 0 _hrx_libhsa_first)
if(NOT _hrx_libhsa_first)
    message(FATAL_ERROR "ONEBIT_HRX: no libhsa-runtime64.so.1 under ONEBIT_HRX_TOOLCHAIN=${ONEBIT_HRX_TOOLCHAIN}")
endif()
set(ONEBIT_HRX_LIBHSA "${_hrx_libhsa_first}" CACHE FILEPATH "TheRock HSA runtime HRX loads (IREE_HAL_AMDGPU_LIBHSA_PATH)")
message(STATUS "ONEBIT_HRX: HSA runtime ${ONEBIT_HRX_LIBHSA}")

# The device library (the amdgcn bitcode hipcc links for --offload-arch) sits inside the SDK's pip
# tree, not under <root>/lib (the 2026-10-31 rebuild): without this flag every HIP step dies with
# "cannot find ROCm device library: provide its path via '--rocm-path' or '--rocm-device-lib-path'".
# Read from the toolchain itself, so the build needs no HIP_DEVICE_LIB_PATH / ROCM_PATH / HIP_PATH,
# which is what engine#713 is for. A toolchain without one builds on hipcc's own default.
set(ONEBIT_HRX_DEVICE_LIB "" CACHE FILEPATH "gfx device library (amdgcn bitcode) hipcc links against")
set(_hrx_sdk_core "")
if(NOT ONEBIT_HRX_DEVICE_LIB)
    file(GLOB _hrx_sdk_cores "${ONEBIT_HRX_TOOLCHAIN}/lib/python3*/site-packages/_rocm_sdk_core")
    if(_hrx_sdk_cores)
        list(GET _hrx_sdk_cores 0 _hrx_sdk_core)
        if(EXISTS "${_hrx_sdk_core}/lib/llvm/amdgcn/bitcode")
            set(ONEBIT_HRX_DEVICE_LIB "${_hrx_sdk_core}/lib/llvm/amdgcn/bitcode" CACHE FILEPATH
                "gfx device library (amdgcn bitcode) hipcc links against" FORCE)
        endif()
    endif()
endif()
# It goes to the HIP kernel compile only (ONEBIT_HRX_HIP_FLAGS below), never into CMAKE_C/CXX_FLAGS:
# those reach every host object of the hrx-system deps build, which compiles with -Werror, and a host C
# file has no device code to link, so clang rejects the flag as unused (-Wunused-command-line-argument).
# A cold HRX build failed on it here, and so did hrx-build whenever the deps rebuilt (runs 42524820160, 42526921955).
if(ONEBIT_HRX_DEVICE_LIB)
    message(STATUS "ONEBIT_HRX: device library ${ONEBIT_HRX_DEVICE_LIB}")
else()
    message(STATUS "ONEBIT_HRX: no device library under ${ONEBIT_HRX_TOOLCHAIN}; hipcc's default is used")
endif()
# The private HIP kernels are compiled by a separate amdclang++ invocation (ggml-hrx-hip.cmake), not by
# hipcc, and it does not inherit the flags above: without GGML_HRX_HIP_FLAGS the device library has to come
# from HIP_DEVICE_LIB_PATH in the environment. Pass the compiler from this toolchain (its own find_program
# would otherwise take /opt/rocm-therock/bin before $ROCM_PATH) and the same device library in the flags.
set(_hrx_hip_flags "-O3")
if(ONEBIT_HRX_DEVICE_LIB)
    string(APPEND _hrx_hip_flags " --rocm-device-lib-path=${ONEBIT_HRX_DEVICE_LIB}")
endif()
set(ONEBIT_HRX_HIP_COMPILER "${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++" CACHE FILEPATH
    "amdclang++ that compiles the HRX HIP kernels")
set(ONEBIT_HRX_HIP_FLAGS "${_hrx_hip_flags}" CACHE STRING "Extra flags for the HRX HIP kernels")
set(_hrx_toolchain_env)
if(_hrx_sdk_core)
    set(_hrx_toolchain_env ${CMAKE_COMMAND} -E env "ROCM_PATH=${_hrx_sdk_core}" "HIP_PATH=${_hrx_sdk_core}")
endif()

# Private GPU kernels (docs/hrx.md, "Private HIP kernels"): a checkout of the private add-on
# repository whose addons/hip-hrx is built into the HRX llama.cpp (its GGML_HRX_HIP_ADDON_DIR hook).
set(_hrx_gpu_private_args)
if(ONEBIT_GPU_PRIVATE)
    if(NOT IS_DIRECTORY "${ONEBIT_GPU_PRIVATE}/addons/hip-hrx/kernels")
        message(FATAL_ERROR "ONEBIT_GPU_PRIVATE=${ONEBIT_GPU_PRIVATE} has no addons/hip-hrx/kernels")
    endif()
    set(_hrx_hip_cmake "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-hrx/hip/ggml-hrx-hip.cmake")
    set(_hrx_addon_hook)
    if(EXISTS "${_hrx_hip_cmake}")
        file(STRINGS "${_hrx_hip_cmake}" _hrx_addon_hook REGEX "GGML_HRX_HIP_ADDON_DIR")
    endif()
    if(NOT _hrx_addon_hook)
        message(FATAL_ERROR "ONEBIT_GPU_PRIVATE: third_party/llama.cpp has no GGML_HRX_HIP_ADDON_DIR hook (needs the HIP plumbing)")
    endif()
    list(APPEND _hrx_gpu_private_args "-DGGML_HRX_HIP_ADDON_DIR=${ONEBIT_GPU_PRIVATE}/addons/hip-hrx")
    message(STATUS "ONEBIT_HRX: private HIP kernels from ${ONEBIT_GPU_PRIVATE}/addons/hip-hrx")
endif()

# engine#315 candidate 1 (docs/hrx.md, "CWSR save-area placement"): serve moves the CWSR save area
# into GTT only when the hrx-system it builds has the HRX_ALLOW_NO_SVM gate; without it, ROCr's
# HSA_USE_SVM=0 leaves HRX no device.
set(_hrx_no_svm_gate "${CMAKE_SOURCE_DIR}/third_party/hrx-system/runtime/src/iree/hal/drivers/amdgpu/util/no_svm_gate.c")
set(ONEBIT_HRX_ALLOW_NO_SVM 0)
if(EXISTS "${_hrx_no_svm_gate}")
    file(STRINGS "${_hrx_no_svm_gate}" _hrx_no_svm_hook REGEX "HRX_ALLOW_NO_SVM")
    if(_hrx_no_svm_hook)
        set(ONEBIT_HRX_ALLOW_NO_SVM 1)
    endif()
endif()
message(STATUS "ONEBIT_HRX: HRX_ALLOW_NO_SVM gate in hrx-system: ${ONEBIT_HRX_ALLOW_NO_SVM}")

set(ONEBIT_HRX_SERVER "${CMAKE_BINARY_DIR}/hrx/llama/bin/llama-server")
ExternalProject_Add(llama_hrx
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
    BINARY_DIR ${CMAKE_BINARY_DIR}/hrx/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
        -DCMAKE_CXX_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++
        -DHRX_SOURCE_DIR=${CMAKE_SOURCE_DIR}/third_party/hrx-system
        -DGGML_HRX=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DGGML_HRX_HIP_COMPILER=${ONEBIT_HRX_HIP_COMPILER}
        "-DGGML_HRX_HIP_FLAGS=${ONEBIT_HRX_HIP_FLAGS}"
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
        "-DCMAKE_C_FLAGS=${ONEBIT_HRX_PATH_MAP_FLAGS}${ONEBIT_HRX_EXTRA_FLAGS}"
        "-DCMAKE_CXX_FLAGS=${ONEBIT_HRX_PATH_MAP_FLAGS}${ONEBIT_HRX_EXTRA_FLAGS}"
        ${_hrx_gpu_private_args}
    BUILD_COMMAND ${_hrx_toolchain_env} ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench llama-perplexity
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
