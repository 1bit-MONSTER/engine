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
#pragma once

#include "q4nx_cache_sweep.h"   // the sweep itself (npu/, engine#2263)

// engine#1788 / disk pressure 11-12: the sweep as a COMMAND, because until now the only trigger was a serve starting
// (`app/serve.cpp` calls q4nx_cache_evict() on its way up). With the NPU off nothing serves, so nothing sweeps and the
// cache grows past its own bound -- measured at 72.5 G against the 60 GiB default on this box while / was at 149 G.
// `1bit q4nx-cache evict [--root DIR] [--max-gib N] [--keep NAME] [--min-age S] [--dry-run]`.
// Exit: 0 swept (or nothing to do), 1 nothing removed but over the bound (age window), 2 bad usage.
int run_q4nx_cache(int argc, char** argv);
