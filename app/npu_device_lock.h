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

// The NPU device lock (engine#612): who may have the XDNA 2 NPU.
//
// One process owns the device by default. The driver does not stop a second one --
// amdxdna's npu4 sets hwctx_limit = 16 (drivers/accel/amdxdna/npu4_regs.c) and only
// rejects a context past that -- but the two streams share one DMA engine and each
// stream allocates its own copy of the weights, so a second serving process makes
// the first slower and doubles the footprint. Measured on Qwen3-0.6B: 65.7 % of
// solo per stream, 1.31x aggregate (docs/npu.md, "Two streams on one NPU").
//
// So the lock is a policy, and the policy is exclusive unless the operator asks:
//   streams <= 1  LOCK_EX  today's behaviour, and still what a bare run takes
//   streams >= 2  LOCK_SH  the opted-in streams co-exist; any exclusive holder --
//                          another build's `1bit serve`, the per-process ELF runner
//                          (npu_engine_universal) -- still keeps them out
// A shared holder and an exclusive holder never overlap, in either order, because
// they use one file and one flock(2). It is advisory: only processes that take it
// are serialised. NPU_NO_DEVICE_LOCK=1 skips it entirely (unsupported escape hatch).
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace onebit {

// The path the NPU runner and every `1bit` NPU process serialise on.
// ONEBIT_NPU_DEVICE_LOCK points the engine at another lock file (a second box, a test);
// anything else that must interlock with it has to be told the same path.
inline constexpr const char* kNpuDeviceLock = "/tmp/1bit-npu-device.lock";

inline const char* npu_device_lock_path() {
    const char* env = std::getenv("ONEBIT_NPU_DEVICE_LOCK");
    return env && *env ? env : kNpuDeviceLock;
}

// How many hardware contexts the device advertises (npu4_regs.c hwctx_limit), which
// is the ceiling --npu-streams may ask for. Past it the driver refuses the context
// and the whole serve exits.
inline constexpr int kNpuMaxStreams = 16;

enum class NpuLockMode { Exclusive, Shared };

// The mode a stream count asks for. 1 (or a 0 that no flag set) is exclusive.
inline NpuLockMode npu_lock_mode_for(int streams) {
    return streams >= 2 ? NpuLockMode::Shared : NpuLockMode::Exclusive;
}

// Take the device lock and return its fd, which must stay open for the life of the
// process (closing it, or exiting, releases the lock). Returns -1 when the lock
// could not be taken: the caller keeps running unlocked, as it always has.
// wait = false is for a caller that wants to know rather than queue (and the test).
inline int take_npu_device_lock(int streams, const char* path = nullptr, bool wait = true) {
    if (!path) path = npu_device_lock_path();
    const bool shared = npu_lock_mode_for(streams) == NpuLockMode::Shared;
    const int op = shared ? LOCK_SH : LOCK_EX;
    // flock needs only a read fd, so the file stays 0644 and other users can still
    // open it; O_NOFOLLOW refuses a symlink planted in /tmp.
    int fd = ::open(path, O_CREAT | O_RDONLY | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    if (::flock(fd, op | LOCK_NB) != 0) {
        if (!wait) {
            ::close(fd);
            return -1;
        }
        std::fprintf(stderr, "[npu] waiting for the %s device lock (%s)...\n", shared ? "shared" : "exclusive", path);
        if (::flock(fd, op) != 0) {
            std::fprintf(stderr, "[npu] device lock unavailable: %s\n", std::strerror(errno));
            ::close(fd);
            return -1;
        }
    }
    if (wait)  // the serve says which mode it took; a caller that only asked stays quiet
        std::fprintf(stderr, "[npu] %s device lock acquired%s\n", shared ? "shared" : "exclusive",
                     shared ? " (one of --npu-streams)" : "");
    return fd;
}

}  // namespace onebit
