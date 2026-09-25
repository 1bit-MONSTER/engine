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
// A whole file mapped read-only (the Q4NX model, Laya's safetensors): mmap on POSIX, a file
// mapping on Windows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace onebit::npu {

// The file's bytes, or nullptr with *why ("cannot open <path>" / "cannot map <path>").
// size is the file's size. The mapping outlives the file handle.
const uint8_t* map_file(const std::string& path, size_t& size, std::string* why);
void unmap_file(const uint8_t* data, size_t size);

}  // namespace onebit::npu
