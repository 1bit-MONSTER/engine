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
# OSS Scanner build (github.com/anthropics/oss-scanner): the engine as CI builds it, on a plain
# x86-64 host with no GPU and no NPU. The scanner builds this with the repository checkout as the
# build context, then removes network access and audits what is in /src. This mirrors
# .github/workflows/ci.yml: GCC 15 (the engine is C++26), the CI target list, ctest.
FROM ubuntu:24.04
# CC/CXX: the tests that configure a scratch CMake tree (tests/npu_runtime_option.sh) pick up the
# compilers from the environment; curl and jq are what the serve tests drive the API with.
ENV DEBIAN_FRONTEND=noninteractive CC=gcc-15 CXX=g++-15
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates software-properties-common gnupg \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y --no-install-recommends \
      g++-15 gcc-15 make ninja-build git pkg-config curl jq python3 python3-pip python3-yaml \
      libcurl4-openssl-dev libzstd-dev libssl-dev libdrm-dev libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*
# CMake from PyPI: the engine is C++26 and Ubuntu 24.04's CMake 3.28 does not know that dialect
# (CI's runner image carries a newer CMake). numpy and gguf are for the Python tests, which read
# GGUF files and build Q4NX / ternary / draft-vocabulary fixtures.
RUN python3 -m pip install --no-cache-dir --break-system-packages "cmake>=3.31" numpy gguf

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Build as CI does, with debug info and frame pointers for the audit. A failing step fails the build.
RUN cmake -S /src -B /src/build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_CXX_COMPILER=g++-15 \
      -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer" \
      -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
    && cmake --build /src/build -j"$(nproc)" --target onebit \
      npu_full_elf_test npu_pack_test npu_q4nx_test npu_private_route_test \
      tokenizer_test think_split_test gguf_meta_test laya_encoder_test \
      http_guard_test moe_gguf_index_test laya_bounds_test \
      npu_model_bounds_test npu_forward_loader_test tokenizer_linear_test

# The cheap tests, so the image is known to work; a failing test does not abort the build.
RUN cd /src && (python3 tests/gguf_to_q4nx_test.py && python3 tests/ternary_to_q4_0_test.py \
      && python3 tests/mtp_draft_vocab_test.py || echo "WARNING: a Python test reported failures") \
    && (ctest --test-dir /src/build --output-on-failure --timeout 300 || echo "WARNING: ctest reported failures")
