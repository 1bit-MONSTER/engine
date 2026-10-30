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
//
// runner.cpp -- the arena's on-device scorer.  Loads ONE full ELF, feeds it the
// frozen int8 harness of a task, checks the int32 output BYTE-EXACT against the
// expected tensor, then times `iters` launches and prints one RESULT line.
//
// Why int32 and no tolerance: int8 x int8 -> int32 accumulation is exact, so a
// single differing element is a bug in the submission (layout, stride, tile
// order, accumulation width), never rounding.  There is nothing to tune.
//
//   runner --elf E --task DIR --seed N [--iters 5] [--m 16] [--dump FILE]
//
// RESULT line (one line, machine readable):
//   RESULT task=<name> seed=<n> K=.. N=.. M=.. verdict=PASS|FAIL mismatches=<n>
//          bytesB=<B> med_ms=<x> min_ms=<y> gbsB=<z> gbs_total=<w> iters=<n>
#include "rt.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::vector<char> slurp(const std::string& p, size_t want) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "runner: cannot open %s\n", p.c_str()); std::exit(3); }
    std::vector<char> v((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (want && v.size() != want) {
        std::fprintf(stderr, "runner: %s is %zu bytes, expected %zu\n", p.c_str(), v.size(), want);
        std::exit(3);
    }
    return v;
}

int main(int argc, char** argv) {
    std::string elf, task, dump, wfile;
    int seed = 1, iters = 5, M = 16, syncA = 0, postUs = 0, preUs = 0, syncC = 0, noCheck = 0;
    size_t wBytesArg = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--elf") elf = next();
        else if (a == "--task") task = next();
        else if (a == "--seed") seed = std::atoi(next().c_str());
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--m") M = std::atoi(next().c_str());
        else if (a == "--dump") dump = next();
        else if (a == "--w") wfile = next();          // a submission's own B buffer
        else if (a == "--w-bytes") wBytesArg = (size_t)std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--sync-a") syncA = 1;
        else if (a == "--sync-c") syncC = 1;
        else if (a == "--no-check") noCheck = 1;   // DMA-ceiling probe: time only, no compare
        else if (a == "--post-us") postUs = std::atoi(next().c_str());
        else if (a == "--pre-us") preUs = std::atoi(next().c_str());
        else { std::fprintf(stderr, "runner: unknown arg %s\n", a.c_str()); return 2; }
    }
    if (elf.empty() || task.empty()) {
        std::fprintf(stderr, "usage: runner --elf E --task DIR --seed N [--iters 5] [--m 16]\n");
        return 2;
    }
    if (iters < 2) iters = 2;

    // ---- geometry from the task's meta (K, N; M is the activation rows) ----
    std::string h = task + "/harness/";
    int K = 0, N = 0;
    {
        std::ifstream f(h + "geometry.txt");
        if (!f) { std::fprintf(stderr, "runner: no %sgeometry.txt\n", h.c_str()); return 3; }
        std::string k, v;
        while (f >> k >> v) {
            if (k == "K") K = std::atoi(v.c_str());
            if (k == "N") N = std::atoi(v.c_str());
        }
    }
    if (K <= 0 || N <= 0) { std::fprintf(stderr, "runner: bad geometry\n"); return 3; }

    const size_t aBytes = (size_t)M * (size_t)K;
    // bBytes is the buffer the kernel streams: the harness's packed int8 B, unless
    // the submission declares its own layout (then the oracle passed --w and
    // --w-bytes).  The expected C never changes: it is A @ B for the *semantic*
    // matrix, so a submission may reorder or repack the weights but not the maths.
    size_t bBytes = (size_t)K * (size_t)N;
    const size_t cBytes = (size_t)M * (size_t)N * 4;
    std::vector<char> A = slurp(h + "seed" + std::to_string(seed) + "_A.bin", aBytes);
    std::vector<char> B = wfile.empty()
                              ? slurp(h + "seed" + std::to_string(seed) + "_B.bin", bBytes)
                              : slurp(wfile, 0);   // any size: the layout declares it
    if (!wfile.empty()) {
        bBytes = B.size();
        if (wBytesArg && wBytesArg != bBytes) {
            std::fprintf(stderr, "runner: %s is %zu bytes, LAYOUT says %zu\n", wfile.c_str(),
                         bBytes, wBytesArg);
            return 3;
        }
    }
    std::vector<char> E = slurp(h + "seed" + std::to_string(seed) + "_expected.bin", cBytes);

    // ---- device: one ELF, one context, three BOs ----
    xrt::device dev(0);
    xrt::elf e(elf);
    xrt::hw_context hc(dev, e);
    xrt::ext::kernel k(hc, "main:seq");
    xrt::bo bA(xrt::ext::bo(dev, aBytes));
    xrt::bo bW(xrt::ext::bo(dev, bBytes));
    xrt::bo bC(xrt::ext::bo(dev, cBytes));
    std::memcpy(bA.map(), A.data(), aBytes);
    std::memcpy(bW.map(), B.data(), bBytes);
    std::memset(bC.map(), 0, cBytes);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (syncC) { bC.sync(XCL_BO_SYNC_BO_TO_DEVICE); bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE); }

    std::vector<double> t;
    int lastState = 0;
    int readOK = 0;
    // The launch+verify pair is retried as a whole: the runtime occasionally hands
    // back a short result BO (zeros for a few percent of the tile) for a kernel
    // that is correct -- see the note above the check.  Re-launching is the honest
    // mitigation: the *timing* that is scored comes from a launch whose output was
    // verified twice, so a submission cannot win by being wrong.
    for (int attempt = 0; attempt < 3; attempt++) {
        std::vector<double> tTry;
        for (int i = 0; i < iters; i++) {
            if (syncA) bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            if (preUs) usleep(preUs);
            const auto t0 = clk::now();
            xrt::run r = k(bA, bW, bC);
            const int st = (int)r.wait();
            if (postUs) usleep(postUs);
            const auto t1 = clk::now();
            if (st != 4) { lastState = st; std::fprintf(stderr, "runner: ERT state=%d on iter %d\n", st, i); }
            if (i >= 1) tTry.push_back(ms_between(t0, t1));   // drop the first (cold) launch
        }
        t = tTry;
        // A cheap pre-check on this attempt's result, so a short read costs one
        // relaunch instead of the full 6-read/12 ms retry loop below.
        bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        {
            const int32_t* g0 = (const int32_t*)bC.map();
            const int32_t* e0 = (const int32_t*)E.data();
            size_t bad = 0;
            for (size_t i = 0; i < (size_t)M * (size_t)N && bad < 64; i++)
                if (g0[i] != e0[i]) bad++;
            if (!bad) { readOK = 1; break; }           // clean: go and time this one
            t.clear();                                  // not clean: this timing is not scored
            std::fprintf(stderr, "runner: attempt %d read back short, relaunching\n", attempt + 1);
        }
    }
    if (!readOK) t.clear();
    // ---- byte-exact check of the whole C tile ----
    // The readback of a result BO is the one point where this runtime is not
    // trustworthy: npu/forward/engine-src/npu_engine_i8ctx_inc.h records the
    // same finding ("bC is identical immediately before each launch, and yet
    // the dequantized bA -- the host's reading of that same bC -- differs ...
    // what the host gets back from it is not", RESULTS-coverage-multifamily
    // sections 68/82: the BOCHK sweep exonerated every host input).  A single
    // sync therefore returns zeros for a few percent of the tile now and then,
    // on a kernel that is perfectly correct.  The arena will not charge that to
    // a submission, so a PASS needs TWO CONSECUTIVE reads that both match: a
    // stale or half-transferred read cannot pass, and a kernel that is wrong in
    // the device memory itself never passes however often it is re-read.
    const int32_t* want = (const int32_t*)E.data();
    const size_t n = (size_t)M * (size_t)N;
    size_t mismatches = 0, firstIdx = 0;
    int32_t firstGot = 0, firstWant = 0, worst = 0;
    int cleanReads = 0, reads = 0, persistentBad = 0;
    for (; reads < 6; reads++) {
        bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const int32_t* got = (const int32_t*)bC.map();
        size_t mis = 0;
        int32_t w = 0;
        for (size_t i = 0; i < n; i++) {
            if (got[i] != want[i]) {
                if (!mis) { firstIdx = i; firstGot = got[i]; firstWant = want[i]; }
                const int32_t d = got[i] > want[i] ? got[i] - want[i] : want[i] - got[i];
                if (d > w) w = d;
                mis++;
            }
        }
        if (mis == 0) { cleanReads++; if (cleanReads >= 2) { mismatches = 0; break; } }
        else { cleanReads = 0; mismatches = mis; worst = w; persistentBad++; }
        usleep(2000);
    }
    if (cleanReads < 2) mismatches = (mismatches ? mismatches : 1);   // never report a clean 0
    const int32_t* got = (const int32_t*)bC.map();
    if (noCheck) mismatches = 0;      // the probe kernel computes nothing by design
    if (!dump.empty()) {
        std::ofstream o(dump, std::ios::binary);
        o.write((const char*)got, (std::streamsize)cBytes);
    }

    std::vector<double> s = t;
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2], mn = s.front();
    const double gbsB = bBytes / 1e9 / (med / 1000.0);
    const double gbsAll = (aBytes + bBytes + cBytes) / 1e9 / (med / 1000.0);
    std::string name = task;
    {
        const size_t p = name.find_last_of('/');
        if (p != std::string::npos) name = name.substr(p + 1);
    }
    std::printf("RESULT task=%s seed=%d K=%d N=%d M=%d verdict=%s mismatches=%zu "
                "bytesB=%zu med_ms=%.3f min_ms=%.3f gbsB=%.2f gbs_total=%.2f iters=%zu state=%d",
                name.c_str(), seed, K, N, M,
                noCheck ? "NOCHECK" : (mismatches == 0 ? "PASS" : "FAIL"), mismatches,
                bBytes, med, mn, gbsB, gbsAll, t.size(), lastState);
    if (mismatches)
        std::printf(" first_idx=%zu got=%d want=%d worst_absdiff=%d", firstIdx, firstGot,
                    firstWant, worst);
    std::printf(" reads=%d clean_reads=%d", reads + 1, cleanReads);
    std::printf("\n");
    std::fflush(stdout);
    return mismatches == 0 ? 0 : 1;
}
