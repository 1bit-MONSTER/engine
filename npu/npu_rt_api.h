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

// The NPU runtime's public interface: the calls the engine's NPU code makes, and nothing else.
//
// The runtime itself is private (docs/npu.md, "The NPU runtime"). This header is ours and public;
// every class is an opaque handle onto an implementation the runtime library defines, so a build
// links a prebuilt runtime library (ONEBIT_NPU_RT_LIB) or compiles the private sources
// (ONEBIT_NPU_RT_DIR) against this same declaration. The names follow XRT's, so npu/rt.h can alias
// the namespace to xrt:: and the engine's sources read as XRT code. Handles copy the way XRT's do:
// a copy refers to the same device, buffer, context, kernel or run.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace npu_rt_api {

// What a run's wait() and state() report (XRT's ert_cmd_state numbering).
enum ert_cmd_state : int {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_QUEUED = 2,
    ERT_CMD_STATE_RUNNING = 3,
    ERT_CMD_STATE_COMPLETED = 4,
    ERT_CMD_STATE_ERROR = 5,
    ERT_CMD_STATE_ABORT = 6,
    ERT_CMD_STATE_SUBMITTED = 7,
    ERT_CMD_STATE_TIMEOUT = 8,
    ERT_CMD_STATE_NORESPONSE = 9,
    ERT_CMD_STATE_SKERROR = 10,
    ERT_CMD_STATE_SKCRASHED = 11,
};

// bo::sync directions (XRT's values).
constexpr int XCL_BO_SYNC_BO_TO_DEVICE = 0;
constexpr int XCL_BO_SYNC_BO_FROM_DEVICE = 1;

namespace detail {
struct access;  // defined only inside the runtime library
}

// The NPU device.
class device {
public:
    explicit device(unsigned index = 0);
    device(const device&);
    device& operator=(const device&);
    ~device();

private:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};

// A buffer the host and the NPU share.
class bo {
public:
    bo();
    // A sub-buffer: [offset, offset + size) of parent.
    bo(const bo& parent, size_t size, size_t offset);
    bo(const bo&);
    bo& operator=(const bo&);
    ~bo();

    void* map() const;
    template <typename T> T map() const { return static_cast<T>(map()); }
    void sync(int direction);
    void sync(int direction, size_t size, size_t offset);
    size_t size() const;
    explicit operator bool() const { return static_cast<bool>(m); }

protected:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};

namespace ext {
// A buffer the NPU reads and writes, allocated for a device.
class bo : public npu_rt_api::bo {
public:
    bo() = default;
    bo(const device& d, size_t size);
};
}  // namespace ext

// A configuration UID, as text.
class uuid {
public:
    uuid() = default;
    explicit uuid(std::string s) : s_(std::move(s)) {}
    std::string to_string() const { return s_; }

private:
    std::string s_;
};

// An NPU ELF, from a file or from memory.
class elf {
public:
    elf();
    explicit elf(const std::string& path);
    elf(const void* data, size_t size);
    elf(const elf&);
    elf& operator=(const elf&);
    ~elf();

    bool is_full_elf() const;
    uuid get_cfg_uuid() const;
    explicit operator bool() const { return static_cast<bool>(m); }

private:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};

// A hardware context made from an ELF; add_config adds another ELF's kernels to it.
class hw_context {
public:
    hw_context();
    hw_context(const device& d, const elf& e);
    hw_context(const hw_context&);
    hw_context& operator=(const hw_context&);
    ~hw_context();

    void add_config(const elf& e);

private:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};

class run;

namespace ext {
// A kernel of the context's ELFs, by name.
class kernel {
public:
    kernel();
    kernel(const hw_context& hc, const std::string& name);
    kernel(const kernel&);
    kernel& operator=(const kernel&);
    ~kernel();

    // Buffer arguments in order; returns a started run.
    template <typename... Args>
    run operator()(Args&&... args) const;

private:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};
}  // namespace ext

// One run of a kernel.
class run {
public:
    run();
    explicit run(const ext::kernel& k);
    run(const run&);
    run& operator=(const run&);
    ~run();

    void set_arg(int index, const bo& b);
    void start();
    // 0 ms waits without a timeout. Returns the run's state when the wait ends.
    ert_cmd_state wait(const std::chrono::milliseconds& timeout = std::chrono::milliseconds{0}) const;
    ert_cmd_state state() const;
    explicit operator bool() const { return static_cast<bool>(m); }

private:
    friend struct detail::access;
    struct impl;
    std::shared_ptr<impl> m;
};

// Several runs on one context, submitted together and waited on once.
class runlist {
public:
    explicit runlist(const hw_context& hc);
    runlist(const runlist&) = delete;
    runlist& operator=(const runlist&) = delete;
    ~runlist();

    void add(const run& r);
    void execute();
    // Throws std::runtime_error if a run did not complete.
    std::cv_status wait(const std::chrono::milliseconds& timeout = std::chrono::milliseconds{0}) const;
    void reset();

private:
    struct impl;
    std::unique_ptr<impl> m;
};

template <typename... Args>
run ext::kernel::operator()(Args&&... args) const {
    static_assert((std::is_base_of_v<npu_rt_api::bo, std::decay_t<Args>> && ...),
                  "kernel arguments are buffers");
    run r(*this);
    int i = 0;
    (r.set_arg(i++, static_cast<const npu_rt_api::bo&>(args)), ...);
    r.start();
    return r;
}

}  // namespace npu_rt_api
