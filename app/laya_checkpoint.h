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

// Where the Laya checkpoint lives, in one place. `serve --laya` and `1bit route` (and so the
// resident `--serve` daemon a caller keeps warm) must answer that question identically: two copies
// of this chain would drift, and the drift shows up as "the router routes but serve --laya cannot
// find a checkpoint".
#pragma once

#include <string>

namespace onebit {

// The Laya checkpoint directory for an explicit --laya-model DIR, or `given` empty to search:
// $ONEBIT_LAYA_MODEL, else ${XDG_DATA_HOME:-~/.local/share}/1bit/laya (where
// scripts/fetch-laya.sh installs it). Empty only when no candidate could be named at all.
std::string laya_home(const std::string& given);

// laya_home(), checked: a directory holding the pinned repo's typed-decisions/ checkpoint resolves
// to that subdirectory (it classifies best, 95.5% against 92.0% for the root checkpoint), any other
// directory is used as given. Throws std::runtime_error naming the path and how to install it.
std::string laya_checkpoint(const std::string& given);

}  // namespace onebit
