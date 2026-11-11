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
//   onebit_fopen_shared_wb   0644: the head-image cache, which a later serve reads back. A cache file only; anything
//                            that holds activations, weights or a request is a dump and takes the 0600 one.
//   onebit_fopen_private_ab  0600, appending: a dump log several runs add to (ort_run's outputs.txt)
// The two `wb` openers truncate an existing file (the `ab` one appends); all three refuse a symlink at the final component (O_NOFOLLOW), and leave errno from the
// failing call, so a caller's strerror(errno) still names the open's reason. O_CREAT's mode applies only when the file
// is created, so a regular file that already exists -- one an older build wrote 0666 -- has every bit beyond the stated
// mode cleared with fchmod() as well. That only narrows: a stricter umask still wins (a 077 umask leaves the cache 0600),
// and a device or pseudo-file (/dev/full, /proc/...) keeps its own mode. Pinned by tests/file_create_test.c,
// and tools/ci/file_create_check.py refuses a new fopen(..., "w"/"a") anywhere in the tree.
// Plain C, so .c and .cpp sources share it.
#pragma once

#include <errno.h>
#include <stdio.h>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
static inline FILE* onebit_fdopen_(int fd, mode_t mode, const char* how) {
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        const int e = errno;
        close(fd);
        errno = e;
        return NULL;
    }
    // Only ever narrows: bits beyond the stated mode are cleared, and none is added, so a stricter umask still wins.
    if (S_ISREG(st.st_mode) && ((st.st_mode & 07777) & ~mode) != 0 && fchmod(fd, (st.st_mode & 07777) & mode) != 0) {
        const int e = errno;
        close(fd);
        errno = e;
        return NULL;
    }
    FILE* f = fdopen(fd, how);
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
    return fopen(path, "wb");  // file-create-ok: Windows has no POSIX mode; the CodeQL finding is POSIX-only
#else
    return onebit_fdopen_(open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR),
                          S_IRUSR | S_IWUSR, "wb");
#endif
}

static inline FILE* onebit_fopen_shared_wb(const char* path) {
#if defined(_WIN32)
    return fopen(path, "wb");  // file-create-ok: Windows has no POSIX mode; the CodeQL finding is POSIX-only
#else
    return onebit_fdopen_(
        open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH),
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, "wb");
#endif
}

static inline FILE* onebit_fopen_private_ab(const char* path) {
#if defined(_WIN32)
    return fopen(path, "ab");  // file-create-ok: Windows has no POSIX mode; the CodeQL finding is POSIX-only
#else
    return onebit_fdopen_(open(path, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC, S_IRUSR | S_IWUSR),
                          S_IRUSR | S_IWUSR, "ab");
#endif
}
