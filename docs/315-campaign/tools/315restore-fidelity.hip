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
// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// engine#315 - the minimal consumer experiment the runtime proposal asks for
// (docs/315-campaign/evidence/315-runtime-restore-fidelity-proposal.md, "Suggested next step").
//
// Question: can a consumer observe different contents on a mapped host-visible range after
// its OWN queue is evicted and restored, with no model, no regime and no neighbours?
//
// Shape: one process holds a hipHostRegister'd pattern buffer (4096 bytes, one page, filled
// with a known counter). A device kernel writes a second known pattern into it and the host
// reads it back after hipDeviceSynchronize. An evict/restore forcer runs INSIDE this process
// (evictonly's mprotect churn on a separate registered range) so the eviction is of this
// process's own queues, exactly as the campaign rig attributes it per-arm.
//
// Verdicts, fixed before the run:
//   t_invalid  : a read-back that does not match the pattern the device was told to write
//                -> the fidelity gap is reproduced at the smallest possible scale
//   t_clean    : every read-back matches
// Either way the counts are printed, plus the observed Q_EVICT/Q_RESTORE if the log is grepped.
//
// Usage: 315restore-fidelity <iters> <churn_us> [--no-forcer]
//   iters      number of write/read-back rounds (default 200000)
//   churn_us   forcer mprotect period in microseconds (default 500, the rig's value)
//   --no-forcer  control: identical loop with no eviction (the negative control the rig needs)

#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <pthread.h>
#include <atomic>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(2); } } while (0)

static std::atomic<bool> g_stop{false};

// The forcer: identical in effect to ~/wt/evictonly.hip, in-thread so the eviction targets
// THIS process's queues (a separate process would evict its own, which is the confound the
// proposal warns about).
static void* forcer(void* arg) {
    const int period_us = *(int*)arg;
    const size_t bytes = 16 << 20;
    char* b = (char*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) return nullptr;
    memset(b, 1, bytes);
    if (hipHostRegister(b, bytes, hipHostRegisterDefault) != hipSuccess) return nullptr;
    for (long c = 0; !g_stop.load(); ++c) {
        mprotect(b, bytes, PROT_READ);
        mprotect(b, bytes, PROT_READ | PROT_WRITE);
        b[(c * 4096) % bytes] ^= 1;
        usleep(period_us);
    }
    hipHostUnregister(b);
    munmap(b, bytes);
    return nullptr;
}

__global__ void fill(unsigned* p, unsigned v) { p[threadIdx.x] = v; }

int main(int argc, char** argv) {
    long iters = argc > 1 ? atol(argv[1]) : 200000;
    int churn_us = argc > 2 ? atoi(argv[2]) : 500;
    bool use_forcer = true;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--no-forcer") == 0) use_forcer = false;

    const int NW = 1024;              // 4096 bytes, one page
    void* host = nullptr;
    CHECK(hipHostMalloc(&host, NW * sizeof(unsigned), hipHostMallocDefault));
    unsigned* hp = (unsigned*)host;

    unsigned* d = nullptr;
    CHECK(hipMalloc(&d, 4));
    CHECK(hipMemset(d, 0, 4));
    CHECK(hipDeviceSynchronize());

    pthread_t th;
    if (use_forcer) pthread_create(&th, nullptr, forcer, &churn_us);
    usleep(200000);   // let the first eviction land before the loop starts

    long bad = 0, clean = 0;
    long first_bad_iter = -1;
    unsigned first_bad_got = 0, first_bad_want = 0;

    for (long i = 0; i < iters; ++i) {
        const unsigned want = (unsigned)(0xA5A50000u ^ (unsigned)i);
        fill<<<1, NW>>>((unsigned*)host, want);     // device writes the pattern into the mapping
        CHECK(hipDeviceSynchronize());
        unsigned got = hp[0];                        // host reads the mapping back
        // cheap full check on the last element too (the write covers all NW)
        unsigned got_last = hp[NW - 1];
        if (got == want && got_last == want) {
            ++clean;
        } else {
            ++bad;
            if (first_bad_iter < 0) {
                first_bad_iter = i;
                first_bad_got = got;
                first_bad_want = want;
            }
            if (bad <= 5)
                printf("MISMATCH iter=%ld want=0x%08x got=0x%08x got_last=0x%08x\n",
                       i, want, got, got_last);
        }
    }

    g_stop.store(true);
    if (use_forcer) pthread_join(th, nullptr);

    printf("no_forcer=%d iters=%ld churn_us=%d clean=%ld invalid=%ld\n",
           use_forcer ? 0 : 1, iters, use_forcer ? churn_us : 0, clean, bad);
    if (first_bad_iter >= 0)
        printf("first_invalid iter=%ld want=0x%08x got=0x%08x\n",
               first_bad_iter, first_bad_want, first_bad_got);
    printf("VERDICT %s\n", bad ? "t_invalid" : "t_clean");

    CHECK(hipHostFree(host));
    CHECK(hipFree(d));
    return 0;
}
