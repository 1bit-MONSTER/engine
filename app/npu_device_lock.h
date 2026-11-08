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
// One process owns the device by default. hwctx_limit = 16 (drivers/accel/amdxdna/
// npu5_regs.c:74; npu4_regs.c:174 says the same) is not the reason: that counts the contexts of one client, while the
// limit on a second client is the array partition -- 8 columns, and the kernel set this
// engine ships asks for all 8, so a second client has no columns to be given. Before
// third_party/patches/amdxdna/0003 the solver shared the tiles anyway and took the array
// down for every client (engine#1084, measured 2026-11-05); with it, the second client
// is refused with -ENODEV / code 102. And even where a kernel set leaves columns free,
// the two streams share one DMA engine and each allocates its own copy of the weights,
// so a second serving process makes the first slower and doubles the footprint: two
// engines on two models measured 34.1 % and 70.4 % of solo, 0.41x aggregate
// (docs/npu.md, "Two streams on one NPU").
//
// So the lock is a policy, and the policy is exclusive unless the operator asks:
//   streams <= 1  LOCK_EX  today's behaviour, and still what a bare run takes
//   streams >= 2  LOCK_SH  the opted-in streams co-exist; any exclusive holder --
//                          another build's `1bit serve`, the per-process ELF runner
//                          (npu_engine_universal) -- still keeps them out
// A shared holder and an exclusive holder never overlap, in either order, because
// they use one file and one flock(2). It is advisory: only processes that take it
// are serialised. NPU_NO_DEVICE_LOCK=1 skips it entirely (unsupported escape hatch).
//
// engine#1241: the wait used to say nothing, so a *live* orphan holding this file looked like a wedged device. flock(2)
// dies with its holder, so there is no stale lock to reap -- the failure mode is a process that is still alive, and the
// fix is a diagnostic, not a reaper. After ~2 s of a blocking wait this header reads /proc/locks for the FLOCK on this
// file, names the holder pid(s) with their truncated cmdline, and repeats that line at most once a minute. The same
// line is available for the NPU's "device error (code 102): No such device" path: a held array answers exactly like a
// dead one, and "No such device" otherwise sends the operator looking at the device, the ELF, or the container.
// ONEBIT_NPU_LOCK_WAIT_MAX=<seconds> bounds the wait (unset = wait forever, the behaviour since #612); on timeout it
// names the holder and throws, so a gate can fail fast instead of spending a device ticket. Nothing here signals, reaps
// or kills the holder.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vector>

namespace onebit {

// The path the NPU runner and every `1bit` NPU process serialise on. A fixed path, not an
// environment variable: flock(2) interlocking only works if everyone names the same file, and a
// settable lock path is one more thing an operator can point somewhere unsafe (CodeQL cpp/path-injection).
inline constexpr const char* kNpuDeviceLock = "/tmp/1bit-npu-device.lock";

// How many hardware contexts the device advertises (npu4_regs.c hwctx_limit), which
// is the ceiling --npu-streams may ask for. Past it the driver refuses the context
// and the whole serve exits.
inline constexpr int kNpuMaxStreams = 16;

enum class NpuLockMode { Exclusive, Shared };

// The mode a stream count asks for. 1 (or a 0 that no flag set) is exclusive.
inline NpuLockMode npu_lock_mode_for(int streams) {
    return streams >= 2 ? NpuLockMode::Shared : NpuLockMode::Exclusive;
}

// Whether a run takes the lock at all (engine#1353). A run that never opens the device takes no
// lock: NPU_INFER_HOST_GEMM=1 is exactly that run (app/forward_serve.cpp initialises the forward
// with init_host and makes no xrt::device), and the lock is exclusive for the process's life, so an
// idle host-GEMM serve blocks every lane that does use the array. Measured 2026-11-06: one orphaned
// host-GEMM serve held it 4 h 52 m with 0 /dev/accel* handles and no CPU, and during that window a
// second lane's serve and two device tickets waited on it. NPU_NO_DEVICE_LOCK=1 remains the
// operator's escape hatch for the runs that do open the device.
inline bool npu_lock_taken(bool opens_device, bool escape_hatch) {
    return opens_device && !escape_hatch;
}

// --- engine#1241: naming the holder --------------------------------------------------------------
// The wait prints the holder line after this many seconds, and not again for this long. Both are
// diagnostic only: they never decide whether the lock is taken or who takes it.
inline constexpr double kNpuLockHolderAfterS = 2.0;
inline constexpr double kNpuLockHolderRepeatS = 60.0;
// How often a waiting process retries flock(LOCK_NB): the same lock, taken as soon as it is free,
// so contention costs at most this much extra latency. Replaces the bare blocking flock so the wait
// can talk and can honour ONEBIT_NPU_LOCK_WAIT_MAX.
inline constexpr double kNpuLockPollS = 0.1;

// A small /proc file read whole, or "" on any error: every caller here degrades to no diagnostic
// rather than failing the lock because /proc is unavailable (a container without it, a masked
// procfs, a process that exits between the locks scan and the cmdline read).
inline std::string npu_read_small_file(const char* path, size_t max = 256 * 1024) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return {};
    std::string out;
    char buf[512];
    while (out.size() < max) {
        const size_t n = std::fread(buf, 1, sizeof buf, f);
        out.append(buf, n);
        if (n < sizeof buf) break;
    }
    std::fclose(f);
    return out;
}

// A whole token parsed in `base`, false when the token is not entirely that number.
inline bool npu_parse_uint(const std::string& token, int base, unsigned long long* out) {
    if (token.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(token.c_str(), &end, base);
    if (errno != 0 || end != token.c_str() + token.size()) return false;
    *out = value;
    return true;
}

// A positive pid out of its /proc/locks column (a POSIX/OFD lock can report -1).
inline bool npu_parse_pid(const std::string& token, long* out) {
    if (token.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(token.c_str(), &end, 10);
    if (errno != 0 || end != token.c_str() + token.size() || value <= 0) return false;
    *out = value;
    return true;
}

// One /proc/locks line, split on the spaces the kernel uses; empty tokens are dropped.
inline std::vector<std::string> npu_split_spaces(const std::string& line) {
    std::vector<std::string> tok;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        size_t j = i;
        while (j < line.size() && line[j] != ' ' && line[j] != '\t') ++j;
        if (j > i) tok.push_back(line.substr(i, j - i));
        i = j;
    }
    return tok;
}

// Whether one /proc/locks line is the FLOCK on (dev_major, dev_minor, inode), and if so the pid
// holding it. The format (fs/locks.c) is:
//   <id>: FLOCK  ADVISORY  WRITE <pid> <major>:<minor>:<inode> <start> <end>
// The kernel prints major/minor in hex; kernels before that printed them decimal, so both readings
// are accepted. The inode is decimal. A blocked waiter's line is "<id>: -> FLOCK ..." -- it is
// waiting for the holder, not holding, so it is not a holder.
inline bool npu_proc_lock_holder(const std::string& line, unsigned dev_major, unsigned dev_minor,
                                 unsigned long long inode, long* pid) {
    if (pid == nullptr) return false;
    const std::vector<std::string> tok = npu_split_spaces(line);
    size_t t = 1;  // tok[0] is "<id>:"
    if (t < tok.size() && tok[t] == "->") return false;
    if (t >= tok.size() || tok[t] != "FLOCK") return false;
    // The "<major>:<minor>:<inode>" token has exactly two colons; the pid is the token before it.
    for (size_t d = t + 1; d < tok.size(); ++d) {
        if (std::count(tok[d].begin(), tok[d].end(), ':') != 2) continue;
        long holder = 0;
        if (!npu_parse_pid(tok[d - 1], &holder)) continue;
        const size_t c1 = tok[d].find(':');
        const size_t c2 = tok[d].find(':', c1 + 1);
        const std::string maj_s = tok[d].substr(0, c1);
        const std::string min_s = tok[d].substr(c1 + 1, c2 - c1 - 1);
        const std::string ino_s = tok[d].substr(c2 + 1);
        unsigned long long ino = 0;
        if (!npu_parse_uint(ino_s, 10, &ino) || ino != inode) continue;
        unsigned long long maj = 0, min = 0;
        bool match = npu_parse_uint(maj_s, 16, &maj) && npu_parse_uint(min_s, 16, &min) &&
                     maj == dev_major && min == dev_minor;
        if (!match)  // a kernel that printed the device in decimal
            match = npu_parse_uint(maj_s, 10, &maj) && npu_parse_uint(min_s, 10, &min) &&
                    maj == dev_major && min == dev_minor;
        if (!match) continue;
        *pid = holder;
        return true;
    }
    return false;
}

// The pids holding a FLOCK on (dev_major, dev_minor, inode) in one read of /proc/locks. Empty when
// none matches (or the content is not /proc/locks).
inline std::vector<long> npu_lock_holders_in(const std::string& proc_locks, unsigned dev_major,
                                             unsigned dev_minor, unsigned long long inode) {
    std::vector<long> pids;
    size_t pos = 0;
    while (pos < proc_locks.size()) {
        size_t eol = proc_locks.find('\n', pos);
        if (eol == std::string::npos) eol = proc_locks.size();
        long pid = 0;
        if (npu_proc_lock_holder(proc_locks.substr(pos, eol - pos), dev_major, dev_minor, inode, &pid) &&
            std::find(pids.begin(), pids.end(), pid) == pids.end())
            pids.push_back(pid);
        pos = eol + 1;
    }
    return pids;
}

// /proc/<pid>/cmdline with its NUL separators as spaces, truncated for one log line. Falls back to
// /proc/<pid>/comm in brackets when the process has no cmdline (a kernel thread) or it went away
// between the scan and the read. "" when the pid is gone and has no comm.
inline std::string npu_holder_cmdline(long pid, size_t max = 96) {
    if (pid <= 0) return {};
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%ld/cmdline", pid);
    std::string out = npu_read_small_file(path, 4096);
    for (char& c : out)
        if (c == '\0') c = ' ';
    while (!out.empty() && out.back() == ' ') out.pop_back();
    if (out.empty()) {
        std::snprintf(path, sizeof path, "/proc/%ld/comm", pid);
        out = npu_read_small_file(path, 256);
        while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
        if (!out.empty()) out = "[" + out + "]";
    }
    if (out.size() > max) {
        out.resize(max);
        out += "...";
    }
    return out;
}

// "pid 2689658 (1bit serve --device npu ...)" for every FLOCK holder of `path`, comma separated;
// "" when nobody holds it, the file does not exist, or /proc/locks is unreadable.
inline std::string npu_lock_holders_text(const char* path = kNpuDeviceLock) {
    struct stat st;
    if (::stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return {};
    const std::string locks = npu_read_small_file("/proc/locks");
    if (locks.empty()) return {};
    const std::vector<long> pids = npu_lock_holders_in(
        locks, static_cast<unsigned>(major(st.st_dev)), static_cast<unsigned>(minor(st.st_dev)),
        static_cast<unsigned long long>(st.st_ino));
    std::string out;
    for (size_t i = 0; i < pids.size(); ++i) {
        if (i) out += ", ";
        out += "pid " + std::to_string(pids[i]);
        const std::string cmd = npu_holder_cmdline(pids[i]);
        if (!cmd.empty()) out += " (" + cmd + ")";
    }
    return out;
}

// The one line engine#1241 asks for, or "" when nobody holds `path`. `held_s >= 0` adds "for N s", the
// time this process has spent waiting: a lower bound on the hold, because the holder had it before we
// started waiting.
inline std::string npu_lock_holder_line(const char* path = kNpuDeviceLock, long held_s = -1) {
    const std::string text = npu_lock_holders_text(path);
    if (text.empty()) return {};
    std::string line = "[npu] device lock held by " + text;
    if (held_s >= 0) line += " for " + std::to_string(held_s) + " s";
    return line;
}

// engine#1241 cure 2: the line an NPU failure path prints when the runtime said "No such device" but
// another pid owns the array through the device lock -- the state that made a held array read as a
// broken device, ELF or container. "" when nobody holds it, i.e. the device really is the thing to look at.
inline std::string npu_lock_refusal_note(const char* path = kNpuDeviceLock) {
    const std::string text = npu_lock_holders_text(path);
    if (text.empty()) return {};
    return "[npu] the device lock is held by " + text +
           ": 'No such device' here is another client owning the array, not a dead device/ELF (wait for it, or stop "
           "it if it is yours)";
}

// ONEBIT_NPU_LOCK_WAIT_MAX: an optional ceiling in seconds on the blocking wait. Unset, empty, zero,
// negative or not a number keeps the unlimited wait -- the default since engine#612 and the right
// shape for ordinary contention on a shared box.
inline double npu_lock_wait_max_s() {
    const char* s = std::getenv("ONEBIT_NPU_LOCK_WAIT_MAX");
    if (!s || !*s) return -1.0;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s) return -1.0;
    while (*end == ' ' || *end == '\t') ++end;  // tolerate a trailing space from a shell export
    if (*end != '\0' || !(v > 0)) return -1.0;
    return v;
}

// Take the device lock and return its fd, which must stay open for the life of the
// process (closing it, or exiting, releases the lock). Returns -1 when the lock
// could not be taken and the caller may keep running unlocked, as it always has.
// The one exception is ONEBIT_NPU_LOCK_WAIT_MAX: an operator who set it asked to fail rather than
// run without the device, so a timeout prints the holder and throws std::runtime_error.
// wait = false is for a caller that wants to know rather than queue (and the test).
inline int take_npu_device_lock(int streams, const char* path = kNpuDeviceLock, bool wait = true) {
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
        // engine#1241: retry LOCK_NB instead of a bare blocking flock. Same file, same lock, same
        // exclusivity -- but the wait can now say who holds it after ~2 s (cure 1) and can stop on
        // ONEBIT_NPU_LOCK_WAIT_MAX (cure 3). A holder is never signalled: flock dies with it.
        const auto started = std::chrono::steady_clock::now();
        const double max_s = npu_lock_wait_max_s();
        double said_at = -1.0;
        for (;;) {
            const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            // The timeout is checked first so a short bound names the holder exactly once (the periodic
            // line below would otherwise repeat it at a 2 s timeout).
            if (max_s > 0 && waited >= max_s) {
                const std::string text = npu_lock_holders_text(path);
                if (!text.empty())
                    std::fprintf(stderr, "[npu] device lock held by %s for %ld s\n", text.c_str(),
                                 static_cast<long>(waited));
                const char* raw = std::getenv("ONEBIT_NPU_LOCK_WAIT_MAX");
                std::string why = "[npu] device lock wait ended after " + std::to_string(static_cast<long>(waited)) +
                                  " s (ONEBIT_NPU_LOCK_WAIT_MAX=" + (raw ? raw : "") + ")";
                if (!text.empty()) why += ": the lock is held by " + text;
                why += "; unset or raise ONEBIT_NPU_LOCK_WAIT_MAX to keep waiting (the holder is another process, "
                       "do not kill it unless it is yours)";
                ::close(fd);
                throw std::runtime_error(why);
            }
            if (waited >= kNpuLockHolderAfterS && (said_at < 0.0 || waited - said_at >= kNpuLockHolderRepeatS)) {
                said_at = waited;
                std::string line = npu_lock_holder_line(path, static_cast<long>(waited));
                if (line.empty())
                    line = "[npu] the device lock is held but /proc/locks does not name the holder (different pid "
                           "namespace, or procfs unavailable)";
                std::fprintf(stderr, "%s\n", line.c_str());
            }
            double nap = kNpuLockPollS;
            if (max_s > 0) nap = std::min(nap, max_s - waited);
            if (nap > 0) {
                struct timespec ts;
                ts.tv_sec = static_cast<time_t>(nap);
                ts.tv_nsec = static_cast<long>((nap - static_cast<double>(ts.tv_sec)) * 1e9);
                if (ts.tv_nsec > 999999999L) ts.tv_nsec = 999999999L;
                ::nanosleep(&ts, nullptr);  // EINTR is fine: the deadline and the lock are re-checked
            }
            if (::flock(fd, op | LOCK_NB) == 0) break;
            const int err = errno;
            if (err != EWOULDBLOCK && err != EINTR) {
                std::fprintf(stderr, "[npu] device lock unavailable: %s\n", std::strerror(err));
                ::close(fd);
                return -1;
            }
        }
    }
    if (wait)  // the serve says which mode it took; a caller that only asked stays quiet
        std::fprintf(stderr, "[npu] %s device lock acquired%s\n", shared ? "shared" : "exclusive",
                     shared ? " (one of --npu-streams)" : "");
    return fd;
}

}  // namespace onebit
