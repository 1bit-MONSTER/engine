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

// The NPU runtime the engine's NPU code compiles against (docs/npu.md, "The runtime").
//
// -DONEBIT_NPU_RUNTIME=xrt (the default): XRT, from ONEBIT_XRT_ROOT.
// -DONEBIT_NPU_RUNTIME=own: an XRT-compatible runtime from a private add-on directory
//   (ONEBIT_NPU_RT_DIR, holding npu_rt.h and npu_rt.cpp). Its classes mirror XRT's names, so the
//   alias below lets the same sources build against it unchanged. This repository holds no
//   runtime code: only this header and the CMake option.
#pragma once

#if defined(ONEBIT_NPU_RUNTIME_OWN)
#define NPU_RT_XRT_COMPAT 1
#include <npu_rt.h>
namespace xrt = npu_rt;
#else
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#endif
