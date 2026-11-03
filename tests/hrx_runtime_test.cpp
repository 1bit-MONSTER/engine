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

// engine#713 criterion 4 (app/hrx_runtime.h, docs/hrx.md "The HSA runtime"): the identity of the
// HSA runtime HRX will load - sha256 of a buffer, of a file, and the line that names both. Known
// FIPS 180-4 vectors, so the digest HRX prints can be checked against `sha256sum`. No GPU.
#include "hrx_runtime.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace onebit;

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static std::string hex_of(const std::string& s) {
    return sha256_hex(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

int main() {
    // FIPS 180-4 / NIST vectors: the empty message, "abc", and the 56-byte two-block case.
    expect(hex_of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "sha256(\"\")");
    expect(hex_of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "sha256(\"abc\")");
    expect(hex_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
           "sha256(the 56-byte NIST message)");

    // A file hashes the same as its bytes, and an unreadable path says so instead of hashing air.
    const std::string path =
        std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/hrx_runtime_test.bin";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "abc";
    }
    expect(sha256_file(path) == hex_of("abc"), "sha256_file: a real file hashes its bytes");
    std::remove(path.c_str());
    expect(sha256_file(path).empty(), "sha256_file: a missing file is \"\", not a digest");
    expect(sha256_file("/nonexistent/rocm/lib/libhsa-runtime64.so.1").empty(),
           "sha256_file: a missing runtime is \"\"");

    // The line HRX prints, and the unreadable variant that keeps HRX's intent visible.
    const std::string sha = "21a6682452e159cfcdc6bdc84676f6b7f9c1236f3a2957a97a2e7170889cdfa3";
    const std::string line = hrx_runtime_line("/opt/rocm-therock/lib/libhsa-runtime64.so.1", sha);
    expect(line == "HRX: HSA runtime /opt/rocm-therock/lib/libhsa-runtime64.so.1 sha256 " + sha,
           "hrx_runtime_line: path and sha256, nothing else (got \"" + line + "\")");
    const std::string unreadable = hrx_runtime_line("/gone/libhsa-runtime64.so.1", "");
    expect(unreadable ==
               "HRX: HSA runtime /gone/libhsa-runtime64.so.1 (unreadable: no sha256, HRX may fail to open it)",
           "hrx_runtime_line: an unreadable runtime says so (got \"" + unreadable + "\")");

    if (failures) return 1;
    std::printf("hrx_runtime: all checks passed\n");
    return 0;
}
