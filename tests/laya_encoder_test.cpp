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

// The encoder registry (laya/encoder.h): what a private add-on registers is found by name,
// replaced by a later registration, and hands back the encoder its factory builds.
#include "encoder.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}
}  // namespace

int main() {
    using namespace onebit::laya;
    check("nothing registered: npu not found", find_encoder("npu") == nullptr);
    register_encoder("npu", [](const std::string& dir, std::string* err) -> EncoderFn {
        if (dir.empty()) {
            *err = "no checkpoint";
            return {};
        }
        return [](float* h, int rows, int n) {
            for (int i = 0; i < rows * 1024; i++) h[i] += 1.f;
            return n <= rows;
        };
    });
    const EncoderFactory* f = find_encoder("npu");
    check("registered: npu found", f != nullptr);
    std::string err;
    check("factory failure: empty encoder and a reason", f && !(*f)("", &err) && err == "no checkpoint");
    EncoderFn fn = f ? (*f)("ckpt", &err) : EncoderFn{};
    float h[2 * 1024] = {};
    check("the encoder runs", fn && fn(h, 2, 1) && h[0] == 1.f && h[2047] == 1.f);
    check("it may decline", fn && !fn(h, 1, 2));
    register_encoder("npu", [](const std::string&, std::string*) -> EncoderFn { return {}; });
    check("a later registration replaces it", !(*find_encoder("npu"))("ckpt", &err));
    check("names", encoder_names() == std::vector<std::string>{"npu"});
    return failures ? 1 : 0;
}
