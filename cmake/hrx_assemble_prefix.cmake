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

# Assembles the HRX prefix llama.cpp's ggml-hrx2 builds against:
#   lib/      libhrx.so*, libloomc.so*, libloom_binding_c_loomc.so, libloomc_amdgpu.a
#   include/  hrx_runtime*.h and loomc/
#   cmake/hrx/hrx-config.cmake  (hrx::hrx, loom::binding::c::loomc, ...::target::amdgpu)
# The runtime's own install/export rules are disabled by patches/hrx/0001-0002,
# so the shared libraries are used from the build tree through their RPATHs.
# Usage: cmake -DSRC= -DBIN= -DPREFIX= -DAR= -DCONFIG_IN= -P hrx_assemble_prefix.cmake
foreach(v SRC BIN PREFIX AR CONFIG_IN)
    if(NOT DEFINED ${v})
        message(FATAL_ERROR "hrx_assemble_prefix: ${v} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${PREFIX}/lib" "${PREFIX}/include" "${PREFIX}/cmake")
file(MAKE_DIRECTORY "${PREFIX}/lib" "${PREFIX}/include/loomc" "${PREFIX}/cmake/hrx")

file(GLOB _libs
    "${BIN}/libhrx/src/libhrx/libhrx.so*"
    "${BIN}/loom/binding/c/libloomc.so*"
    "${BIN}/loom/binding/c/libloom_binding_c_loomc.so")
if(NOT _libs)
    message(FATAL_ERROR "hrx_assemble_prefix: no libraries found under ${BIN}")
endif()
file(COPY ${_libs} DESTINATION "${PREFIX}/lib" FOLLOW_SYMLINK_CHAIN)

# The Loom AMDGPU binding is a single object library; ggml-hrx2 links it as an archive.
set(_amdgpu_obj "${BIN}/loom/binding/c/target/amdgpu/CMakeFiles/loom_binding_c_target_amdgpu_amdgpu.objects.dir/amdgpu.c.o")
execute_process(COMMAND "${AR}" rcs "${PREFIX}/lib/libloomc_amdgpu.a" "${_amdgpu_obj}" COMMAND_ERROR_IS_FATAL ANY)

file(COPY "${SRC}/libhrx/include/hrx_runtime.h" "${SRC}/libhrx/include/hrx_runtime_cxx.h"
     DESTINATION "${PREFIX}/include")
file(COPY "${SRC}/loom/binding/c/include/loomc/" DESTINATION "${PREFIX}/include/loomc")
configure_file("${CONFIG_IN}" "${PREFIX}/cmake/hrx/hrx-config.cmake" @ONLY)
message(STATUS "HRX prefix assembled at ${PREFIX}")
