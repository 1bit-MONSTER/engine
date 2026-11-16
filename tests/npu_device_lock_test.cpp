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

#include <algorithm>
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

// Fork a child that opens `path` and holds the fd until a byte arrives on the returned pipe write end. Same
// acknowledgement discipline as spawn_holder, no flock: this is the co-tenant that has the device open but never took
// our lock, which is the client engine#2689 item 2 has to be able to name.
static pid_t spawn_fd_holder(const std::string& path, int* release_fd) {
    int ack[2] = {-1, -1};
    int rel[2] = {-1, -1};
    if (::pipe(ack) != 0 || ::pipe(rel) != 0) return -1;
    const pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        ::close(ack[0]);
        ::close(rel[1]);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) ::_exit(90);
        const char ok = 'y';
        if (::write(ack[1], &ok, 1) != 1) ::_exit(92);
        char b = 0;
        const ssize_t released = ::read(rel[0], &b, 1);
        (void)released;
        ::_exit(0);
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

    // --- engine#2689 item 2: naming the co-tenant that never took the lock -----------------------------------------
    // The fd scan reads the link target, so the fixture is a fake /proc whose <pid>/fd/<n> symlinks point at the same
    // kind of names a real one holds: the node, the directory itself, a second node, a node whose path is gone
    // (" (deleted)"), a look-alike (accelx0), an unrelated fd, a pid directory with no fd, and a non-pid directory.
    {
        const std::string prefix = dir + "/dev/accel";
        const std::string fake = dir + "/proc";
        auto link = [](const std::string& from, const std::string& to) {
            return ::symlink(to.c_str(), from.c_str()) == 0;
        };
        ::mkdir(fake.c_str(), 0700);
        for (const std::string& d : {fake + "/4242", fake + "/4343", fake + "/4444", fake + "/4646",
                                     fake + "/notanumber", fake + "/5000"})
            ::mkdir(d.c_str(), 0700);
        for (const std::string& d : {fake + "/4242/fd", fake + "/4343/fd", fake + "/4444/fd", fake + "/4646/fd",
                                     fake + "/notanumber/fd"})
            ::mkdir(d.c_str(), 0700);
        bool made = link(fake + "/4242/fd/3", prefix + "/accel0");         // ours
        made &= link(fake + "/4242/fd/4", "/dev/null");                     // an unrelated fd on the same pid
        made &= link(fake + "/4343/fd/0", prefix + "/accel1 (deleted)");    // a second node whose path is gone
        made &= link(fake + "/4444/fd/1", prefix);                           // the directory itself
        made &= link(fake + "/4646/fd/9", prefix + "x0");                    // a look-alike, not below the prefix
        made &= link(fake + "/notanumber/fd/3", prefix + "/accel0");        // not a pid
        expect(made, "the fake /proc symlinks are in place");
        expect(onebit::npu_device_users_in(fake, prefix) == std::vector<long>({4242, 4343, 4444}),
               "the fd scan names the pids with a node below the prefix, and only those");
        expect(onebit::npu_device_users_in(fake, prefix, 4242) == std::vector<long>({4343, 4444}),
               "self is left out: naming the process that just failed is the engine#2434 bug");
        expect(onebit::npu_device_users_in(fake, "").empty(), "an empty prefix names nobody");
        expect(onebit::npu_device_users_in("/nonexistent-proc", prefix).empty(), "a missing proc root names nobody");
        expect(onebit::npu_fd_target_is(prefix + "/accel0", prefix), "a child of the prefix is a match");
        expect(onebit::npu_fd_target_is(prefix, prefix), "the prefix itself is a match");
        expect(!onebit::npu_fd_target_is(prefix + "x0", prefix), "a look-alike is not a match");
        expect(!onebit::npu_fd_target_is(prefix + "/accel0", ""), "an empty prefix matches nothing");
    }

    // A real /proc, a live co-tenant: a forked child opens a file that stands in for the node (the scan reads the link
    // target, so a plain file exercises the same path as a device here) and the scan must name it. No NPU is touched.
    {
        const std::string prefix = dir + "/dev/accel";
        ::mkdir((dir + "/dev").c_str(), 0700);
        ::mkdir(prefix.c_str(), 0700);
        const std::string node = prefix + "/accel0";
        ::close(::open(node.c_str(), O_CREAT | O_WRONLY, 0600));   // a plain file stands in for the device node
        int release_fd = -1;
        const pid_t holder = spawn_fd_holder(node, &release_fd);
        expect(holder > 0, "forked a live fd holder");
        if (holder > 0) {
            const std::vector<long> pids = onebit::npu_device_users_in("/proc", prefix, static_cast<long>(::getpid()));
            expect(std::find(pids.begin(), pids.end(), static_cast<long>(holder)) != pids.end(),
                   "a live holder is named by a scan of the real /proc");
            const std::string text = onebit::npu_device_users_text(prefix);
            expect(text.find("pid " + std::to_string(holder)) != std::string::npos, "and rendered: " + text);
            expect(text.find("npu_device_lock_test") != std::string::npos, "with its cmdline: " + text);
            // The note an operator reads when a load is refused: the lock file exists and nobody holds it, so the
            // fd scan is the only thing that can name the co-tenant -- and it must.
            const std::string note = onebit::npu_lock_refusal_note(lk, prefix);
            expect(note.find("pid " + std::to_string(holder)) != std::string::npos,
                   "the refusal note names the client that never took the lock: " + note);
            expect(note.find("no client holds the device lock") != std::string::npos,
                   "and says the lock was not the thing to look at: " + note);
            const char go = 'g';
            (void)!::write(release_fd, &go, 1);
            ::close(release_fd);
            ::waitpid(holder, nullptr, 0);
        }
        expect(onebit::npu_device_users_text(prefix).empty(), "nobody holds it once the child is gone");
        // And with nobody holding the device the note says nothing at all, rather than inventing a holder.
        expect(onebit::npu_lock_refusal_note(lk, prefix).empty(), "no lock holder and no device user is no note");
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
        // engine#2434: the refusal NOTE must not name this process. A serve takes the lock before it opens a device
        // context, so when its own init is refused /proc/locks reports the only holder - itself - and the note told the
        // operator to wait for, or stop, the process that had just failed. Measured with a FastFlowLM co-tenant that does
        // not take our lock: the note named the failing serve's own pid.
        {
            const std::string note = onebit::npu_lock_refusal_note(lk);
            expect(note.find("pid " + std::to_string(holder)) != std::string::npos,
                   "the note names the real holder: " + note);
        }
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

    // engine#2441 review: the BOUNDARY, with crafted pids. A forked child can never share a prefix with our own pid on
    // demand, so the arm that matters here is the pure classifier: "pid 1234" must not match "pid 12345".
    expect(onebit::npu_holder_item_is_self("pid 1234", 1234), "an entry with no cmdline is ours");
    expect(onebit::npu_holder_item_is_self("pid 1234 (1bit serve)", 1234), "an entry with a cmdline is ours");
    expect(!onebit::npu_holder_item_is_self("pid 12345 (1bit serve)", 1234),
           "a pid that merely STARTS with ours is not ours (the prefix bug the review caught)");
    expect(!onebit::npu_holder_item_is_self("pid 123 (1bit serve)", 1234), "a pid that is a prefix OF ours is not ours");
    expect(!onebit::npu_holder_item_is_self("pid 0 (kernel)", 1234), "an unrelated pid is not ours");
    expect(!onebit::npu_holder_item_is_self("pid 1234x (odd)", 1234), "our pid followed by a non-delimiter is not ours");

    // engine#2434: the SELF case, which is the one the note got wrong - and it has to run AFTER the forked child above
    // releases, because a blocking flock here would deadlock against the child's own hold (it did: the test hung until the
    // timeout, which is how this placement was found). A serve takes this lock before it opens a device context, so when
    // its own init is refused /proc/locks reports the only holder - itself - and the old note named the process that had
    // just failed. The child-holder checks above cannot see this: there an exclusion that did nothing would still pass.
    {
        const int self_fd = ::open(lk, O_RDONLY | O_CLOEXEC);
        const bool took = self_fd >= 0 && ::flock(self_fd, LOCK_EX | LOCK_NB) == 0;
        expect(took, "this process can take the lock once the child released it");
        if (took) {
            const std::string mine = "pid " + std::to_string(static_cast<long>(::getpid()));
            const std::string held = onebit::npu_lock_holders_text(lk);
            expect(held.find(mine) != std::string::npos, "the raw holder text does name this process: " + held);
            const std::string note = onebit::npu_lock_refusal_note(lk);
            expect(note.find(mine) == std::string::npos, "but the refusal note does not: " + note);
            expect(note.find("this process holds the device lock and is still refused") != std::string::npos,
                   "and it says the true thing instead of advising us to stop ourselves: " + note);
            (void)::flock(self_fd, LOCK_UN);
        }
        if (self_fd >= 0) ::close(self_fd);
    }
        ::waitpid(holder, nullptr, 0);
    }

    ::unlink(lk);
    ::rmdir(dir.c_str());
    std::printf("%s\n", failures ? "FAILED" : "npu_device_lock: ok");
    return failures ? 1 : 0;
}
