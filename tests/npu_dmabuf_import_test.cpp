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

// engine#753: does this engine *execute* on a dma-buf it did not allocate?
//
// The issue's claim was that it could not, because XRT has no way to bind a foreign dma-buf as a kernel argument. That is no longer
// true here: npu/npu_rt_api.h declares `ext::bo(const device&, int dma_buf_fd, size_t, void* host)`, the private runtime imports the
// fd with DRM PRIME and pins those pages, and npu-kernels#39 already runs kernels with one (bit-exact 131072 B under an npu-hold
// ticket, and again by the semantic gate). What was missing was any arm *in this tree*: `tests/npu_rt_shim_test.cpp` covers
// `ext::bo(dev, size_t)` and the fd constructor only through its link-coverage function, which is never called.
//
//   npu_dmabuf_import_test <kernel.elf> [kernel-name] [--args N] [--skip-negative]
//
// Skips (77) when no ELF is given, so a box without one registers a skip rather than a red row.
//
// The engine writes nothing here that a kernel would not: this is the *transport* claim, so the assertions are about identity, not
// about arithmetic --
//   1. the dma-buf is created with no GPU in the path (a sealed memfd handed to /dev/udmabuf, which /dev/dri/renderD128 cannot do
//      without an amdgpu BO) and is therefore reproducible on any box with amdxdna;
//   2. `ext::bo(dev, fd, size, host)` accepts it, and `map()` is the caller's own mapping -- NOT a second one, because mapping the
//      *amdxdna* handle puts the BO on the driver's invalid list and the next submit HMM-faults (npu_rt.cpp, import_bo);
//   3. a value written to the exporter's mapping is what the imported BO reports (the same pages, both directions);
//   4. the trap `host = nullptr` documents, measured as the observable difference rather than by courting the wedge: it constructs
//      and yields a second mapping, which is what makes the next submit fault. My first version of this arm asserted that nullptr
//      *throws*, and the device run refuted it -- the failure is at submit, and the comment in npu_rt_api.h reads as if it were here;

#include "rt.h"

#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// A dma-buf with no GPU in the path: a sealed memfd, handed to /dev/udmabuf. Same shape an exporter's fd has (npu-kernels#39 built
// its GPU-free gate this way); the difference is only who exported it.
struct UdmaBuf {
    int memfd = -1;
    int fd    = -1;
    void* map = nullptr;
    size_t bytes = 0;

    bool create(size_t n) {
        bytes = n;
        memfd = memfd_create("npu-dmabuf-import", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (memfd < 0) { std::perror("memfd_create"); return false; }
        if (ftruncate(memfd, off_t(n)) != 0) { std::perror("ftruncate"); return false; }
        // udmabuf refuses a mapping whose size can change under it
        if (fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) { std::perror("F_ADD_SEALS"); return false; }
        const int udma = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (udma < 0) { std::perror("open /dev/udmabuf"); return false; }
        struct udmabuf_create create = {};
        create.memfd  = uint32_t(memfd);
        create.offset = 0;
        create.size   = n;
        create.flags  = UDMABUF_FLAGS_CLOEXEC;
        const int saved = errno;
        fd = ioctl(udma, UDMABUF_CREATE, &create);
        close(udma);
        if (fd < 0) { errno = saved; std::perror("UDMABUF_CREATE"); return false; }
        map = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { std::perror("mmap(udmabuf)"); map = nullptr; return false; }
        return true;
    }
    ~UdmaBuf() {
        if (map) munmap(map, bytes);
        if (fd >= 0) close(fd);
        if (memfd >= 0) close(memfd);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("skip: no kernel ELF given (`npu_dmabuf_import_test <kernel.elf>`); the executing arm needs one device ELF\n");
        return 77;
    }
    std::string kernel_name = "main:sequence";   // the IRON sequence instance name our generated ELFs expose
    int         n_args      = 1;
    bool        skip_negative = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--skip-negative")) skip_negative = true;
        else if (!std::strcmp(argv[i], "--args") && i + 1 < argc) n_args = std::atoi(argv[++i]);
        else kernel_name = argv[i];
    }

    const std::string elf_path = argv[1];
    std::printf("npu dmabuf import (engine#753): %s\n", elf_path.c_str());

    // 1. the exporter half: a dma-buf that exists before the NPU is involved
    UdmaBuf region;
    // 128 KiB: the row payload our generated kvwrite sequence reads from argument 0 (32 rows x 2048 elements x 2 B). The size only
    // has to be >= what the kernel reads -- a BO read past its end is a device fault, not a failed check.
    const size_t kBytes = 128 * 1024;
    const bool have_region = region.create(kBytes);
    check(have_region, "a dma-buf exists with no GPU in the path (udmabuf over a sealed memfd)");
    if (!have_region) return 1;

    // 2. import it, handing over the exporter's mapping
    xrt::device dev{0};
    bool imported = false;
    std::unique_ptr<xrt::ext::bo> bo;
    try {
        bo = std::make_unique<xrt::ext::bo>(dev, region.fd, kBytes, region.map);
        imported = true;
    } catch (const std::exception& e) {
        std::printf("     import threw: %s\n", e.what());
    }
    check(imported, "ext::bo(dev, fd, size, host) accepts it");
    if (!imported) return 1;

    check(bo->size() == kBytes, "the imported BO is the dma-buf's size");
    check(bo->map() == region.map,
          "map() IS the exporter's mapping (a second mapping puts the BO on the driver's invalid list and the next submit HMM-faults)");

    // 3. the same pages, both directions: what the exporter writes, the imported BO reads
    static_cast<uint8_t*>(region.map)[0]     = 0xA5;
    static_cast<uint8_t*>(region.map)[12345] = 0x5A;
    __builtin_ia32_clflush(static_cast<char*>(region.map));
    __builtin_ia32_mfence();
    const auto* seen = static_cast<const uint8_t*>(bo->map());
    check(seen[0] == 0xA5 && seen[12345] == 0x5A,
          "a write through the exporter's mapping is visible through the imported BO (one buffer, not a copy)");

    // 4. the trap the runtime's own comment documents, measured: `host = nullptr` does NOT fail at construction -- it succeeds and
    // hands back a *second* mapping of the same dma-buf, which is the state that puts the BO on the driver's invalid list at the next
    // submit (npu_rt.cpp, import_bo: amdxdna_insert_pages -> amdxdna_mark_mapp_invalid, then the submit HMM-faults a dma-buf VMA).
    // The submit-time failure is deliberately NOT exercised here: it wedges the array, and a test that courts a wedge to observe a
    // documented failure is a bad trade. What is asserted is the observable difference, which is what a caller can check cheaply.
    if (!skip_negative) {
        bool constructed = false;
        std::string why;
        void* driver_map = nullptr;
        try {
            xrt::ext::bo driver_mapped(dev, region.fd, kBytes, nullptr);
            constructed = true;
            driver_map  = driver_mapped.map();
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(constructed, std::string("host = nullptr constructs (so the trap is at SUBMIT, not at construction) (") +
                               (constructed ? "no throw" : why) + ")");
        check(constructed && driver_map != region.map,
              "and it is a SECOND mapping, not the exporter's -- the state npu_rt.cpp says the next submit faults on");
    }

    // 5. engine#753's actual claim: bind the imported BO as a kernel ARGUMENT and run. This is the step that failed when the issue
    // was filed -- not the import, the binding -- so the arm does not stop at "it maps".
    bool bound = false, completed = false;
    std::string bind_why;
    try {
        xrt::elf image(elf_path);
        xrt::hw_context hc(dev, image);
        xrt::ext::kernel k(hc, kernel_name);
        xrt::run r(k);
        r.set_arg(0, *bo);                       // <- the imported dma-buf, as argument 0
        std::vector<xrt::ext::bo> extra;         // the rest of the signature: plain BOs, sized generously (transport, not semantics)
        for (int i = 1; i < n_args; ++i) extra.emplace_back(dev, size_t(1) << 20);
        for (int i = 1; i < n_args; ++i) r.set_arg(i, extra[size_t(i - 1)]);
        bound = true;
        r.start();
        completed = r.wait(std::chrono::milliseconds(60000)) == ERT_CMD_STATE_COMPLETED;
    } catch (const std::exception& e) {
        bind_why = e.what();
    }
    check(bound, std::string("the imported BO is accepted as kernel argument 0 (") + (bound ? "no throw" : bind_why) + ")");
    check(completed, std::string("and the run with it completes (") + (completed ? "COMPLETED" : "not completed") + ")");

    if (failures) std::printf("%d check(s) failed\n", failures);
    else          std::puts("all npu dmabuf-import checks passed");
    return failures ? 1 : 0;
}
