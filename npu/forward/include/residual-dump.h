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
//
// Writing NPU_INFER_DUMP_DIR's per-layer residual stream, and saying so when it does not work (engine#1726).
//
// The dump exists so a per-layer cosine against a reference trace can localise a divergence to an op. Its first version
// wrote with a bare `if (FILE* fp = fopen(...)) { fwrite(...); fclose(fp); }`, so a missing directory, an unwritable one or
// a full disk produced no file, no error, and a run that still printed that it was dumping -- and a missing
// `layer_<l>.f32` then reads as "that layer never ran" to whoever analyses the dump. That is the misleading-instrument
// class, not a tidiness problem.
//
// Three calls, three ways to fail, and the third is the one people forget: `fwrite` can be short (a full disk), and a
// buffered write of the final block is where it actually surfaces, in `fclose`. The helper reports the first failure and
// stays non-fatal: a diagnostic must not fail a valid run, so the caller decides what to do with it.
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

#include "file_create.h"   // the dump is created 0600, never through a symlink

namespace onebit::npu {

// Writes `count` floats to `path`. Returns false and fills `err` (path + strerror) when the file cannot be created, the
// write is short, or the close fails. `err` is left untouched on success.
inline bool write_residual_dump(const char * path, const float * data, size_t count, std::string * err) {
    FILE * fp = onebit_fopen_private_wb(path);
    if (fp == nullptr) {
        if (err != nullptr) {
            *err = std::string(path) + ": " + std::strerror(errno);
        }
        return false;
    }
    const size_t got = std::fwrite(data, sizeof(float), count, fp);
    // errno has to be captured before fclose(), which may set its own.
    const int write_errno = errno;
    const bool closed      = std::fclose(fp) == 0;
    const int close_errno  = errno;
    if (got != count) {
        if (err != nullptr) {
            *err = std::string(path) + ": wrote " + std::to_string(got) + " of " + std::to_string(count) + " floats: " +
                   std::strerror(write_errno != 0 ? write_errno : (close_errno != 0 ? close_errno : ENOSPC));
        }
        return false;
    }
    if (!closed) {
        if (err != nullptr) {
            *err = std::string(path) + ": close failed: " + std::strerror(close_errno);
        }
        return false;
    }
    return true;
}

}  // namespace onebit::npu
