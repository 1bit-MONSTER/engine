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
#include "file_map.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace onebit::npu {

const uint8_t* map_file(const std::string& path, size_t& size, std::string* why) {
    size = 0;
#ifdef _WIN32
    const HANDLE f = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        if (why) *why = "cannot open " + path;
        return nullptr;
    }
    LARGE_INTEGER sz{};
    ::GetFileSizeEx(f, &sz);
    const HANDLE m = sz.QuadPart ? ::CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    void* p = m ? ::MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (m) ::CloseHandle(m);
    ::CloseHandle(f);
    if (!p) {
        if (why) *why = "cannot map " + path;
        return nullptr;
    }
    size = size_t(sz.QuadPart);
    return static_cast<const uint8_t*>(p);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (why) *why = "cannot open " + path;
        return nullptr;
    }
    struct stat st{};
    ::fstat(fd, &st);
    void* p = st.st_size ? ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    ::close(fd);
    if (p == MAP_FAILED) {
        if (why) *why = "cannot map " + path;
        return nullptr;
    }
    size = size_t(st.st_size);
    return static_cast<const uint8_t*>(p);
#endif
}

void unmap_file(const uint8_t* data, size_t size) {
    if (!data) return;
#ifdef _WIN32
    (void)size;
    ::UnmapViewOfFile(data);
#else
    ::munmap(const_cast<uint8_t*>(data), size);
#endif
}

}  // namespace onebit::npu
