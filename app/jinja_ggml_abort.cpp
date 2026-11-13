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

// engine#2138: the one ggml symbol the fork's chat-template sources reach for.
//
// From llama.cpp #327 on, common/jinja/value.cpp works through common_json (common/json.cpp), and
// json.cpp routes nlohmann's JSON_ASSERT to GGML_ASSERT, which calls ggml_abort(). The 1bit binary
// does not link ggml (the HRX and Vulkan servers are separate processes), so onebit_jinja carries
// this definition: the same message ggml prints (file:line: text) on stderr, then abort(). It has
// no abort callback and no backtrace, because nothing in the binary sets one.
#include "ggml.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fflush(stdout);
    char message[2048];
    const int offset = std::snprintf(message, sizeof message, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    if (offset >= 0 && size_t(offset) < sizeof message) std::vsnprintf(message + offset, sizeof message - size_t(offset), fmt, args);
    va_end(args);
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}
