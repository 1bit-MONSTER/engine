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
// Our own XRT-compatible runtime, from the private directory ONEBIT_NPU_RT_DIR (npu_rt.h and
// npu_rt.cpp), compiled into 1bit. Its classes mirror XRT's names, so the alias below lets the
// engine's sources keep the xrt:: spelling. AMD's XRT itself is not supported. This repository
// holds no runtime code: only this header and the CMake option.
#pragma once

#if !defined(ONEBIT_NPU_RUNTIME_OWN)
#error "the NPU lane builds only against the own NPU runtime (-DONEBIT_NPU_RT_DIR, docs/npu.md)"
#endif
#define NPU_RT_XRT_COMPAT 1
#include <npu_rt.h>
namespace xrt = npu_rt;
