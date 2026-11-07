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

// npu_device_lock_test: app/npu_device_lock.h -- the exclusive default of engine#612, the shared
// mode --npu-streams >= 2 opts into, and (engine#1241) the holder diagnostic on a live lock: the
// /proc/locks parse, the wait naming the holder after ~2 s, and ONEBIT_NPU_LOCK_WAIT_MAX failing
// fast with it named instead of blocking. Header only and device-free: a forked child holds the
// flock on a temp file, so no NPU and no /tmp/1bit-npu-device.lock is touched. Runs in CI.
#include "npu_device_lock.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Capture stderr for the span of one call: the diagnostic under test is what an operator sees, so
// the test reads the same stream rather than an internal string.
struct StderrCapture {
    int fd = -1;
    int saved = -1;
    std::string path;

    explicit StderrCapture(const std::string& dir) {
        char tmpl[256];
        std::snprintf(tmpl, sizeof tmpl, "%s/npu-lock-err-XXXXXX", dir.c_str());
        fd = ::mkstemp(tmpl);
        if (fd < 0) return;
        path = tmpl;
        std::fflush(stderr);
        saved = ::dup(STDERR_FILENO);
        if (saved >= 0) ::dup2(fd, STDERR_FILENO);
    }
    std::string finish() {
        std::string out;
        if (saved >= 0) {
            std::fflush(stderr);
            ::dup2(saved, STDERR_FILENO);
            ::close(saved);
            saved = -1;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (!path.empty()) {
            out = onebit::npu_read_small_file(path.c_str());
            ::unlink(path.c_str());
            path.clear();
        }
        return out;
    }
    ~StderrCapture() { finish(); }
};

// Fork a child that takes the exclusive flock on `path` and holds it until a byte arrives on the
// returned pipe write end. The child acknowledges only after flock, so the parent never races it.
static pid_t spawn_holder(const char* path, int* release_fd) {
    int ack[2] = {-1, -1};
    int rel[2] = {-1, -1};
    if (::pipe(ack) != 0) return -1;
    if (::pipe(rel) != 0) {
        ::close(ack[0]);
        ::close(ack[1]);
        return -1;
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(ack[0]);
        ::close(ack[1]);
        ::close(rel[0]);
        ::close(rel[1]);
        return -1;
    }
    if (pid == 0) {
        ::close(ack[0]);
        ::close(rel[1]);
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) ::_exit(90);
        if (::flock(fd, LOCK_EX) != 0) ::_exit(91);
        const char ok = 'y';
        if (::write(ack[1], &ok, 1) != 1) ::_exit(92);
        char b = 0;
        const ssize_t released = ::read(rel[0], &b, 1);  // hold the lock until the parent releases us
        (void)released;
        ::_exit(0);  // exit closes the fd, which releases the flock
    }
    ::close(ack[1]);
    ::close(rel[0]);
    char b = 0;
    const bool acked = ::read(ack[0], &b, 1) == 1;
    ::close(ack[0]);
    if (!acked) {
        ::waitpid(pid, nullptr, 0);
        ::close(rel[1]);
        return -1;
    }
    *release_fd = rel[1];
    return pid;
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

    // Whether the lock is taken at all (engine#1353): a run that never opens the device takes none,
    // because the lock is exclusive for the process's life and a host-GEMM serve that held it would
    // block every lane that does use the array (measured: 4 h 52 m, two other lanes' work waiting).
    expect(onebit::npu_lock_taken(true, false), "a run that opens the device takes the lock");
    expect(!onebit::npu_lock_taken(false, false), "a run that never opens the device (host GEMM) takes none");
    expect(!onebit::npu_lock_taken(true, true), "NPU_NO_DEVICE_LOCK=1 still skips it for a run that does open the device");
    expect(!onebit::npu_lock_taken(false, true), "and for one that does not");

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

    // --- engine#1241: naming the holder -----------------------------------------------------------
    // The /proc/locks parse: only an unblocked FLOCK on our device:inode is a holder. A POSIX lock on
    // the same file, another inode, another device, or a blocked waiter ("->", waiting not holding)
    // is not; a mandatory (MSNFS) FLOCK on it is; and the decimal device form an older kernel printed
    // still matches.
    {
        // Device 0:255 (0x00:0xff) and inode 1234 are "ours"; the file's other inode, another minor,
        // another device, and a POSIX lock on our file must not match. A mandatory (MSNFS) FLOCK does.
        const std::string proc =
            "1: POSIX  ADVISORY  WRITE 1111 00:ff:1234 0 EOF\n"
            "2: FLOCK  ADVISORY  WRITE 4242 00:ff:1234 0 EOF\n"
            "3: FLOCK  ADVISORY  WRITE 4343 00:ff:9999 0 EOF\n"
            "4: FLOCK  ADVISORY  WRITE 4444 00:30:1234 0 EOF\n"
            "5: FLOCK  ADVISORY  WRITE 4545 103:02:1234 0 EOF\n"
            "6: FLOCK  MSNFS     READ  4646 00:ff:1234 0 EOF\n"
            "7: -> FLOCK  ADVISORY  WRITE 4747 00:ff:1234 0 EOF\n"
            "8: FLOCK  ADVISORY  READ  4848 0:255:1234 0 EOF\n"
            "9: FLOCK  ADVISORY  WRITE 4949 0:200:1234 0 EOF\n";
        expect(onebit::npu_lock_holders_in(proc, 0, 0xff, 1234) == std::vector<long>({4242, 4646, 4848}),
               "FLOCK holders of this device:inode only, not a waiter/POSIX/other file/device");
        expect(onebit::npu_lock_holders_in(proc, 0, 0xff, 7777).empty(), "a different inode names nobody");
        expect(onebit::npu_lock_holders_in(proc, 0, 0xff, 1234).size() == 3,
               "the decimal device form (0:255) is the same device as the hex form (00:ff)");
        expect(onebit::npu_lock_holders_in("", 0, 0x2f, 1234).empty(), "an empty /proc/locks names nobody");
        expect(onebit::npu_lock_holder_line("/nonexistent-npu-lock", 5).empty(),
               "no lock file is no diagnostic, not a failure");
    }

    // ONEBIT_NPU_LOCK_WAIT_MAX: only a positive number bounds the wait; everything else keeps the
    // unlimited wait that engine#612 shipped.
    {
        ::unsetenv("ONEBIT_NPU_LOCK_WAIT_MAX");
        expect(onebit::npu_lock_wait_max_s() < 0, "unset ONEBIT_NPU_LOCK_WAIT_MAX waits forever");
        ::setenv("ONEBIT_NPU_LOCK_WAIT_MAX", "0", 1);
        expect(onebit::npu_lock_wait_max_s() < 0, "0 waits forever");
        ::setenv("ONEBIT_NPU_LOCK_WAIT_MAX", "nonsense", 1);
        expect(onebit::npu_lock_wait_max_s() < 0, "a non-number waits forever");
        ::setenv("ONEBIT_NPU_LOCK_WAIT_MAX", "1x", 1);
        expect(onebit::npu_lock_wait_max_s() < 0, "trailing junk is not silently a number");
        ::setenv("ONEBIT_NPU_LOCK_WAIT_MAX", "1", 1);
        expect(std::abs(onebit::npu_lock_wait_max_s() - 1.0) < 1e-9, "ONEBIT_NPU_LOCK_WAIT_MAX=1 is one second");
        ::unsetenv("ONEBIT_NPU_LOCK_WAIT_MAX");
    }

    // A live holder, for real: a forked child takes the exclusive flock on our temp file, so the
    // checks below are the issue's -- it is named, a bounded wait fails fast with it named, and the
    // plain wait names it after ~2 s.
    int release_fd = -1;
    const pid_t holder = spawn_holder(lk, &release_fd);
    expect(holder > 0, "forked a live holder");
    if (holder > 0) {
        const std::string text = onebit::npu_lock_holders_text(lk);
        expect(text.find("pid " + std::to_string(holder)) != std::string::npos,
               "the live holder's pid is named: " + text);
        expect(text.find("npu_device_lock_test") != std::string::npos, "and its cmdline: " + text);
        const std::string line = onebit::npu_lock_holder_line(lk, 41);
        expect(line.find("[npu] device lock held by pid " + std::to_string(holder)) != std::string::npos,
               "one line in the issue's shape: " + line);
        expect(line.find(" for 41 s") != std::string::npos, "and how long we have waited: " + line);

        // Cure 3: ONEBIT_NPU_LOCK_WAIT_MAX=1 fails in about a second instead of blocking, naming the
        // holder and the variable. The holder does not release until the test says so, so returning
        // at all proves the wait was bounded.
        ::setenv("ONEBIT_NPU_LOCK_WAIT_MAX", "1", 1);
        bool timed_out = false;
        std::string why, err;
        const double t0 = now_s();
        {
            StderrCapture cap(dir);
            try {
                (void)onebit::take_npu_device_lock(1, lk);
            } catch (const std::exception& ex) {
                timed_out = true;
                why = ex.what();
            }
            err = cap.finish();
        }
        const double waited = now_s() - t0;
        ::unsetenv("ONEBIT_NPU_LOCK_WAIT_MAX");
        expect(timed_out, "ONEBIT_NPU_LOCK_WAIT_MAX=1 fails instead of blocking on a live holder");
        expect(waited >= 0.9 && waited < 3.0,
               "and fails in about a second, took " + std::to_string(int(waited * 1000)) + " ms");
        expect(why.find("pid " + std::to_string(holder)) != std::string::npos,
               "the failure names the holder's pid: " + why);
        expect(why.find("ONEBIT_NPU_LOCK_WAIT_MAX") != std::string::npos, "and the variable that bounded the wait");
        expect(err.find("pid " + std::to_string(holder)) != std::string::npos,
               "the holder line is on stderr as well as in the exception: " + err);

        // Cure 1: the plain blocking wait names the live holder after ~2 s, unprompted. A forked
        // releaser frees the holder at 3 s, so the wait also proves it still takes the lock when the
        // holder goes -- the diagnostic changed nothing about who gets the device.
        const pid_t releaser = ::fork();
        if (releaser == 0) {
            struct timespec ts{3, 0};
            ::nanosleep(&ts, nullptr);
            const char b = 'r';
            if (::write(release_fd, &b, 1) != 1) ::_exit(93);
            ::_exit(0);
        }
        int acquired = -1;
        std::string wait_err;
        {
            StderrCapture cap(dir);
            acquired = onebit::take_npu_device_lock(1, lk);
            wait_err = cap.finish();
        }
        if (releaser > 0) ::waitpid(releaser, nullptr, 0);
        expect(acquired >= 0, "the blocking wait takes the lock once the holder goes");
        if (acquired >= 0) ::close(acquired);
        expect(wait_err.find("[npu] waiting for the exclusive device lock") != std::string::npos,
               "the wait says it is waiting: " + wait_err);
        expect(wait_err.find("held by pid " + std::to_string(holder)) != std::string::npos,
               "and names the live holder after ~2 s without being asked: " + wait_err);
        expect(wait_err.find("npu_device_lock_test") != std::string::npos, "with the holder's cmdline: " + wait_err);
        if (release_fd >= 0) ::close(release_fd);
        ::waitpid(holder, nullptr, 0);
    }

    ::unlink(lk);
    ::rmdir(dir.c_str());
    std::printf("%s\n", failures ? "FAILED" : "npu_device_lock: ok");
    return failures ? 1 : 0;
}
