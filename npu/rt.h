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

// The NPU runtime the engine's NPU code compiles against (docs/npu.md, "The NPU runtime").
//
// Our own runtime, private and XRT-compatible, behind our own public interface npu_rt_api.h: the
// engine sees only that header. The runtime comes either as a prebuilt static library
// (ONEBIT_NPU_RT_LIB) or as its private sources (ONEBIT_NPU_RT_DIR), compiled into 1bit. Its classes
// carry XRT's names, so the alias below lets the engine's sources keep the xrt:: spelling. AMD's XRT
// itself is not supported. This repository holds no runtime code: only these headers and the CMake
// options.
#pragma once

#if !defined(ONEBIT_NPU_RUNTIME_OWN)
#error "the NPU lane builds only against the own NPU runtime (ONEBIT_NPU_RT_LIB or ONEBIT_NPU_RT_DIR, docs/npu.md)"
#endif
#include "npu_rt_api.h"
namespace xrt = npu_rt_api;
// XRT's global names the engine's code uses unqualified.
using npu_rt_api::ert_cmd_state;
using npu_rt_api::ERT_CMD_STATE_COMPLETED;
using npu_rt_api::ERT_CMD_STATE_TIMEOUT;
using npu_rt_api::XCL_BO_SYNC_BO_TO_DEVICE;
using npu_rt_api::XCL_BO_SYNC_BO_FROM_DEVICE;
