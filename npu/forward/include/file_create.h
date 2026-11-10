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

// file_create.h -- create a file for writing with a stated mode, never world-writable, never through a symlink.
//
// fopen(path, "w") creates with 0666 & ~umask: world-writable under a permissive umask, and it follows a symlink that
// someone else planted at a predictable name. CodeQL reports every such site (cpp/world-writable-file-creation), and a
// PR that only touches the surrounding code inherits the alert as "new in code changed by this pull request" -- which
// is how #1830 sat red for a scan it did not cause. These two openers are what the env-gated dumps and the head-image
// cache use instead. The mode is a constant inside each function, so the scan can see it.
//   onebit_fopen_private_wb  0600: dumps and probes (an activation or weight dump is nobody else's business)
//   onebit_fopen_shared_wb   0644: a cache file another reader of the same tree may open
// Both truncate an existing file, refuse a symlink at the final component (O_NOFOLLOW), and leave errno from the
// failing call, so a caller's strerror(errno) still names the open's reason. Plain C, so .c and .cpp sources share it.
#pragma once

#include <errno.h>
#include <stdio.h>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
static inline FILE* onebit_fdopen_wb_(int fd) {
    if (fd < 0) return NULL;
    FILE* f = fdopen(fd, "wb");
    if (!f) {
        const int e = errno;
        close(fd);
        errno = e;
    }
    return f;
}
#endif

static inline FILE* onebit_fopen_private_wb(const char* path) {
#if defined(_WIN32)
    return fopen(path, "wb");
#else
    return onebit_fdopen_wb_(open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR));
#endif
}

static inline FILE* onebit_fopen_shared_wb(const char* path) {
#if defined(_WIN32)
    return fopen(path, "wb");
#else
    return onebit_fdopen_wb_(
        open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH));
#endif
}
