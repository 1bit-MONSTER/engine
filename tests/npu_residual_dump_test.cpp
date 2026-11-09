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

// Host test for NPU_INFER_DUMP_DIR's residual write (engine#1726). No device, no NPU headers: the helper is pure stdio.
//
// The defect this pins is silence, not incorrectness: the first version wrote with a bare
// `if (FILE* fp = fopen(...)) { fwrite(...); fclose(fp); }`, so an unwritable directory or a full disk produced no file,
// no error, and a run that still printed that it was dumping. A missing layer_<l>.f32 then reads as "that layer never
// ran" to whoever compares it against a reference trace.

#include "residual-dump.h"

#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

int main() {
    const std::vector<float> data = { 1.0f, -2.0f, 3.5f, 0.0f };

    // 1. a writable path: success, and the bytes really land (a dump nobody can read is the same silence).
    const std::string good = "/tmp/npu-residual-dump-test.f32";
    std::string err = "untouched";
    check(onebit::npu::write_residual_dump(good.c_str(), data.data(), data.size(), &err), "a writable path succeeds");
    check(err == "untouched", "and leaves the error string untouched");
    if (FILE * fp = std::fopen(good.c_str(), "rb")) {
        std::vector<float> back(data.size(), 0.0f);
        const size_t n = std::fread(back.data(), sizeof(float), back.size(), fp);
        std::fclose(fp);
        check(n == data.size() && back == data, "and the floats read back byte for byte");
    } else {
        check(false, "the written file can be read back");
    }
    std::remove(good.c_str());

    // 2. a directory that cannot be written: the failure has to be REPORTED, with the path and a reason, instead of
    //    handing back a run that believes it produced a dump.
    err.clear();
    check(!onebit::npu::write_residual_dump("/nonexistent-dir-for-1726/layer_00.f32", data.data(), data.size(), &err),
          "a missing directory is a failure, not a silent no-op");
    check(err.find("/nonexistent-dir-for-1726/layer_00.f32") != std::string::npos,
          "and the error names the path it could not write");
    check(err.find(':') != std::string::npos && err.size() > std::string("/nonexistent-dir-for-1726/layer_00.f32: ").size(),
          "and carries a reason after the path");
    std::printf("     (reason: %s)\n", err.c_str());

    // 3. a short write: /dev/full accepts the open and fails the write, which is the full-disk case, and it is the arm a
    //    version that only checks fopen would pass.
    err.clear();
    if (FILE * probe = std::fopen("/dev/full", "wb")) {  // skips cleanly on a host without it
        std::fclose(probe);
        check(!onebit::npu::write_residual_dump("/dev/full", data.data(), data.size(), &err),
              "a full disk is a failure even though fopen succeeded");
        check(err.find("/dev/full") != std::string::npos, "and the short-write error names the path");
        std::printf("     (reason: %s)\n", err.c_str());
    } else {
        std::printf("ok   (skipped the /dev/full arm: this host has no /dev/full)\n");
    }

    // 4. zero floats is not an error: an empty write still creates the file, and a caller must not see a failure for a
    //    layer with no rows.
    err.clear();
    check(onebit::npu::write_residual_dump(good.c_str(), data.data(), 0, &err), "an empty write succeeds");
    std::remove(good.c_str());

    if (failures != 0) {
        std::printf("test-npu-residual-dump: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-npu-residual-dump: all checks passed\n");
    return 0;
}
