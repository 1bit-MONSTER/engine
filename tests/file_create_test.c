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

// file_create_test.c -- npu/forward/include/file_create.h: the modes the two openers promise, and the symlink refusal.
//
// The fix this pins is invisible without an arm: every call site only swapped fopen() for an opener, and a later
// "simplification" back to fopen() would pass every other test. So, with umask 0 (the case where fopen gives 0666):
//   [1] a new file from the private opener is 0600, and from the shared opener 0644;
//   [2] a file that already exists wider than the stated mode -- 0664, what fopen() left under a 002 umask; the test
//       does not make a world-writable file itself, which CodeQL would rightly flag -- is narrowed to 0600 / 0644 when
//       rewritten, and its old contents are truncated;
//   [3] a stricter umask still wins: under 077 the shared opener's new file is 0600, not widened to 0644;
//   [4] a symlink at the path is refused with ELOOP, and the file it points at is untouched;
//   [5] a failure leaves the open's errno (a missing directory reads ENOENT), so a caller's strerror still says why;
//   [6] a device keeps its own mode: /dev/null opens, and is not chmod-ed;
//   [7] the append opener creates 0600, keeps what is there, and narrows an existing 0664 log.
// Scratch goes under ONEBIT_TEST_TMP, else ~/.cache/scratch-tests/tmp (tests/tmp_root.h's rule), and is removed.
#include "file_create.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
static void check(int ok, const char* what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

static int mode_of(const char* p) {
    struct stat st;
    return stat(p, &st) == 0 ? (int)(st.st_mode & 07777) : -1;
}

static long size_of(const char* p) {
    struct stat st;
    return stat(p, &st) == 0 ? (long)st.st_size : -1;
}

static int write_one(FILE* f, const char* s) {
    if (!f) return 0;
    const int ok = fputs(s, f) >= 0;
    return fclose(f) == 0 && ok;
}

int main(void) {
    char root[1024];
    const char* e = getenv("ONEBIT_TEST_TMP");
    const char* h = getenv("HOME");
    if (e && *e) snprintf(root, sizeof root, "%s/file_create_test.XXXXXX", e);
    else if (h && *h) {
        char base[900];
        snprintf(base, sizeof base, "%s/.cache/scratch-tests/tmp", h);
        char mk[1000];
        snprintf(mk, sizeof mk, "%s/.cache", h); mkdir(mk, 0700);
        snprintf(mk, sizeof mk, "%s/.cache/scratch-tests", h); mkdir(mk, 0700);
        mkdir(base, 0700);
        snprintf(root, sizeof root, "%s/file_create_test.XXXXXX", base);
    } else snprintf(root, sizeof root, "/tmp/file_create_test.XXXXXX");
    if (!mkdtemp(root)) {
        printf("FAIL  mkdtemp(%s): %s\n", root, strerror(errno));
        return 1;
    }
    char a[1200], b[1200], c[1200], l[1200], t[1200], n[1200];
    snprintf(a, sizeof a, "%s/private", root);
    snprintf(b, sizeof b, "%s/shared", root);
    snprintf(c, sizeof c, "%s/strict", root);
    snprintf(l, sizeof l, "%s/link", root);
    snprintf(t, sizeof t, "%s/target", root);
    snprintf(n, sizeof n, "%s/no-such-dir/x", root);

    umask(0);
    check(write_one(onebit_fopen_private_wb(a), "x") && mode_of(a) == 0600, "[1] a new private file is 0600 under umask 0");
    check(write_one(onebit_fopen_shared_wb(b), "x") && mode_of(b) == 0644, "[1] a new shared file is 0644 under umask 0");

    chmod(a, 0664);
    chmod(b, 0664);
    check(mode_of(a) == 0664 && mode_of(b) == 0664, "[2] the two files start wider than their openers allow (0664)");
    check(write_one(onebit_fopen_private_wb(a), "") && mode_of(a) == 0600 && size_of(a) == 0,
          "[2] an existing 0664 file is narrowed to 0600 and truncated");
    check(write_one(onebit_fopen_shared_wb(b), "") && mode_of(b) == 0644, "[2] an existing 0664 cache file is narrowed to 0644");

    umask(077);
    check(write_one(onebit_fopen_shared_wb(c), "x") && mode_of(c) == 0600,
          "[3] under umask 077 the shared opener's new file is 0600 (narrow-only, never widened)");
    umask(0);

    {
        FILE* f = onebit_fopen_private_wb(t);   // the symlink's target, created 0600 and expected to stay untouched
        if (f) { fputs("keep", f); fclose(f); }
    }
    check(symlink(t, l) == 0, "a symlink to a target file");
    errno = 0;
    FILE* f = onebit_fopen_private_wb(l);
    const int e1 = errno;
    check(f == NULL && e1 == ELOOP, "[4] the private opener refuses a symlink with ELOOP");
    if (f) fclose(f);
    errno = 0;
    f = onebit_fopen_shared_wb(l);
    const int e2 = errno;
    check(f == NULL && e2 == ELOOP, "[4] the shared opener refuses a symlink with ELOOP");
    if (f) fclose(f);
    check(size_of(t) == 4 && mode_of(t) == 0600, "[4] the symlink's target is untouched (size and mode)");

    errno = 0;
    f = onebit_fopen_private_wb(n);
    check(f == NULL && errno == ENOENT, "[5] a missing directory fails with ENOENT, for the caller's strerror");

    struct stat before, after;
    const int had = stat("/dev/null", &before) == 0;
    f = onebit_fopen_private_wb("/dev/null");
    check(f != NULL, "[6] a device opens");
    if (f) fclose(f);
    check(!had || (stat("/dev/null", &after) == 0 && (after.st_mode & 07777) == (before.st_mode & 07777)),
          "[6] and keeps its own mode (it is not chmod-ed to 0600)");

    char g[1200];
    snprintf(g, sizeof g, "%s/log", root);
    check(write_one(onebit_fopen_private_ab(g), "ab") && mode_of(g) == 0600, "[7] the append opener creates 0600");
    chmod(g, 0664);
    check(write_one(onebit_fopen_private_ab(g), "cd") && size_of(g) == 4 && mode_of(g) == 0600,
          "[7] a second append keeps the first line and narrows a 0664 log to 0600");

    unlink(a); unlink(b); unlink(c); unlink(l); unlink(t); unlink(g);
    rmdir(root);
    printf("%s (%d failed)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
