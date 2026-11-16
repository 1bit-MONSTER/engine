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

// npu_elf_twin_test -- engine#2599's decision, pinned without a device: open_prefill() runs a role at 16 rows only when
// its prefill ELF is byte-identical to the _m16 twin of the SAME stem.  Every arm of that decision:
//   1. the twin path is "<dir>/<stem>_m16.elf";
//   2. identical bytes -> twin (the bonsai2-27b-NPU case);
//   3. one byte different, same size -> not a twin (a real M=128 build of equal size);
//   4. a different size -> not a twin;
//   5. no twin on disk, or an empty file -> not a twin (nothing changes);
//   6. two ROLES of one shape (G and U) are identical by design, and the rule never compares them: G's twin is G_m16.
#include "target/xdna/elf_twin.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++fails; } } while (0)

static void put(const std::string& p, const std::string& bytes) { std::ofstream(p, std::ios::binary) << bytes; }

int main() {
    char tmpl[] = "/tmp/npu-elf-twin.XXXXXX";
    const char* tmp = std::getenv("TMPDIR");
    std::string base = tmp && *tmp ? std::string(tmp) + "/npu-elf-twin.XXXXXX" : std::string(tmpl);
    std::vector<char> buf(base.begin(), base.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) { std::printf("mkdtemp failed\n"); return 1; }
    const std::string d(buf.data());
    using onebit::npu::m16_twin_path;
    using onebit::npu::same_artifact;

    CHECK(m16_twin_path(d, "full_i8_U_K5120_N17408.elf") == d + "/full_i8_U_K5120_N17408_m16.elf", "1. twin path");
    CHECK(m16_twin_path(d, "noext") == d + "/noext_m16.elf", "1. twin path without .elf");

    const std::string big(200000, 'x');   // spans several 64 KiB compare blocks
    put(d + "/U.elf", big);
    put(d + "/U_m16.elf", big);
    CHECK(same_artifact(d + "/U.elf", m16_twin_path(d, "U.elf")), "2. identical bytes are a twin");

    std::string last = big; last.back() = 'y';
    put(d + "/D.elf", big);
    put(d + "/D_m16.elf", last);
    CHECK(!same_artifact(d + "/D.elf", m16_twin_path(d, "D.elf")), "3. a last-byte difference is not a twin");
    std::string mid = big; mid[70000] = 'z';
    put(d + "/O_m16.elf", mid);
    put(d + "/O.elf", big);
    CHECK(!same_artifact(d + "/O.elf", m16_twin_path(d, "O.elf")), "3. a difference in the second block is not a twin");

    put(d + "/Z.elf", big);
    put(d + "/Z_m16.elf", big + "x");
    CHECK(!same_artifact(d + "/Z.elf", m16_twin_path(d, "Z.elf")), "4. a size difference is not a twin");

    put(d + "/Q.elf", big);
    CHECK(!same_artifact(d + "/Q.elf", m16_twin_path(d, "Q.elf")), "5. no twin on disk is not a twin");
    put(d + "/E.elf", "");
    put(d + "/E_m16.elf", "");
    CHECK(!same_artifact(d + "/E.elf", m16_twin_path(d, "E.elf")), "5. two empty files are not a twin");

    // 6. G and U: same shape, same bytes, no _m16 for G here -> G is not reported as a twin of U.
    put(d + "/G.elf", big);
    CHECK(!same_artifact(d + "/G.elf", m16_twin_path(d, "G.elf")), "6. G identical to U is not G's twin");

    for (const char* f : {"U", "U_m16", "D", "D_m16", "O", "O_m16", "Z", "Z_m16", "Q", "E", "E_m16", "G"})
        std::remove((d + "/" + f + ".elf").c_str());
    rmdir(d.c_str());
    if (fails) { std::printf("npu_elf_twin_test: %d failure(s)\n", fails); return 1; }
    std::printf("npu_elf_twin_test: OK\n");
    return 0;
}
