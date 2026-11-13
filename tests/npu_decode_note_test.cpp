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
// tests/npu_decode_note_test.cpp -- the "say it out loud" decision of app/npu_decode_note.h (engine#2102).
//
// The arms that matter are the silent ones: a warning that fires on a healthy serve is worse than no warning, because it
// teaches an operator to ignore the line that exists to be noticed. Every arm here is a hand-computed case; no device.

#include <cstdio>
#include <string>

#include "npu_decode_note.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

}  // namespace

int main() {
    using onebit::npu::decode_family_note;

    // 1. the measured case: 0 of 4 roles, which is 1.9 tok/s where the fused route does 88-101
    {
        const std::string note = decode_family_note(0, 4, "/kernels/i8");
        check(!note.empty(), "0 of 4 roles: a note is produced");
        check(note.find("0 of 4 role(s)") != std::string::npos, "... it names the counts");
        check(note.find("/kernels/i8") != std::string::npos, "... and the directory the operator pointed at");
        check(note.find("engine#2102") != std::string::npos, "... and the issue, so the message can be followed");
        check(note.find("fused route") != std::string::npos, "... and what the healthy route is called");
    }

    // 2. THE SILENT CASES -- a warning on a healthy serve is worse than none
    check(decode_family_note(4, 4, "/k").empty(), "4 of 4 roles (the fused route): silent");
    check(decode_family_note(5, 4, "/k").empty(), "more families than roles (a wide set too): silent");
    check(decode_family_note(0, 0, "/k").empty(), "0 of 0 (no roles at all, a different fault): silent");
    check(decode_family_note(3, 0, "/k").empty(), "a role count of zero is not this issue's business: silent");

    // 3. partial families are still the slow path, and the count says how slow to expect
    {
        const std::string note = decode_family_note(2, 4, "/k");
        check(!note.empty(), "2 of 4 roles: a partial family still warns (the other two run prefill width)");
        check(note.find("2 of 4") != std::string::npos, "... naming the partial count");
    }

    // 3b. the cases that are not missing kernels (engine#2102 review, all three reproduced on 692ae19)
    check(decode_family_note(0, 0, "/k").empty(), "host-GEMM arm (design_roles() is 0 there): silent");
    check(decode_family_note(8, 9, "/k", false, true).empty(), "healthy t2s container, 8 of 9 (docs/npu.md): silent");
    check(!decode_family_note(0, 9, "/k", false, true).empty(), "a ternary container with no decode kernel at all: warns");
    {
        const std::string off = decode_family_note(0, 4, "/k", true, false);
        check(off.find("NPU_DECODE_M16=0") != std::string::npos, "NPU_DECODE_M16=0: names the switch");
        check(off.find("of 4 role(s) have") == std::string::npos, "... and does not call the kernels missing");
        check(!decode_family_note(4, 4, "/k", true, false).empty(), "... even when every twin is on disk");
    }
    // 3c. the way out is the fused route: the twins alone measured ~2.7 tok/s against 88-101 (#2102)
    check(decode_family_note(0, 4, "/k").find("Ship the") == std::string::npos, "the note does not promise speed from the _m16 designs");

    // 4. no directory: the message must not read as a path
    {
        const std::string note = decode_family_note(0, 4, "");
        check(note.find("(no kernel directory)") != std::string::npos, "an empty kernel directory is named as absent, not as ''");
    }

    if (failures != 0) {
        std::printf("npu decode note: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("npu decode note: all cases ok\n");
    return 0;
}
