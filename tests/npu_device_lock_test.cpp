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

// npu_device_lock_test: app/npu_device_lock.h -- the exclusive default of engine#612
// and the shared mode --npu-streams >= 2 opts into, on a lock file in a temp dir so
// no NPU and no /tmp/1bit-npu-device.lock is touched. Header only: runs in CI.
#include "npu_device_lock.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

int main() {
    std::string dir = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/npu-lock-XXXXXX";
    if (!::mkdtemp(dir.data())) {
        std::printf("FAIL mkdtemp %s\n", dir.c_str());
        return 1;
    }
    const std::string path = dir + "/device.lock";
    const char* lk = path.c_str();

    // The mode a stream count implies: the default (and no flag at all) is exclusive.
    expect(onebit::npu_lock_mode_for(0) == onebit::NpuLockMode::Exclusive, "streams=0 is exclusive");
    expect(onebit::npu_lock_mode_for(1) == onebit::NpuLockMode::Exclusive, "streams=1 is exclusive");
    expect(onebit::npu_lock_mode_for(2) == onebit::NpuLockMode::Shared, "streams=2 is shared");
    expect(onebit::npu_lock_mode_for(onebit::kNpuMaxStreams) == onebit::NpuLockMode::Shared, "16 is shared");

    // Exclusive: one holder at a time, and it keeps the shared streams out.
    int ex = onebit::take_npu_device_lock(1, lk, false);
    expect(ex >= 0, "exclusive on a free device");
    expect(onebit::take_npu_device_lock(1, lk, false) < 0, "a second exclusive waits");
    expect(onebit::take_npu_device_lock(2, lk, false) < 0, "a shared stream waits for an exclusive holder");
    ::close(ex);

    // Shared: the opted-in streams co-exist; an exclusive holder still waits for them.
    int s1 = onebit::take_npu_device_lock(2, lk, false);
    int s2 = onebit::take_npu_device_lock(2, lk, false);
    int s3 = onebit::take_npu_device_lock(3, lk, false);
    expect(s1 >= 0 && s2 >= 0 && s3 >= 0, "three shared streams co-exist (hwctx_limit is 16)");
    expect(onebit::take_npu_device_lock(1, lk, false) < 0, "an exclusive holder waits for the streams");
    ::close(s2);
    expect(onebit::take_npu_device_lock(1, lk, false) < 0, "an exclusive still waits for the last two streams");
    ::close(s1);
    ::close(s3);
    ex = onebit::take_npu_device_lock(1, lk, false);
    expect(ex >= 0, "exclusive takes the device once every stream is gone");
    ::close(ex);

    expect(std::string(onebit::kNpuDeviceLock) == "/tmp/1bit-npu-device.lock", "the lock path is the fixed one");

    // No lock file is not a crash: the caller keeps running unlocked.
    expect(onebit::take_npu_device_lock(1, "/nonexistent-dir/npu.lock", false) < 0, "a missing lock file is -1");
    ::unlink(lk);
    ::rmdir(dir.c_str());
    std::printf("%s\n", failures ? "FAILED" : "npu_device_lock: ok");
    return failures ? 1 : 0;
}
