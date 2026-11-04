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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A link-time stand-in for the private NPU runtime, used ONLY by the CI check that compiles and links
// npu/ort/onebit_npu_gemm.cpp (engine #874, the ONNX EP's ONEBIT_EP_NPU leg).
//
// Why this file exists: that leg is built only when ONEBIT_EP_NPU=ON, which needs ONNX Runtime *and*
// the private runtime (ONEBIT_NPU_RT_DIR, docs/npu.md), so CI compiled neither it nor anything it
// needs. #874 was two defects in one week-green build: a call to q4nx_elf_arg_sizes() with no
// declaration (a compile error) and the file defining it missing from the EP's source list (a link
// error). The check target in CMakeLists.txt compiles the EP's own source list against this stub, so
// both halves turn red in the `build` job.
//
// It is compiled against the public interface header npu_rt_api.h, which is the point: a signature
// change there breaks this file at compile time rather than on someone's device. Everything that
// would open, submit to or wait on the device throws, so a build that ever *ran* against it fails
// immediately instead of silently doing nothing; the accessors that only describe a handle
// (map/size/address/is_full_elf) return empty values, which is what an empty handle has.
// It is never linked into libonebit_ep.so, `1bit`, or any probe.

#include "npu_rt_api.h"

#include <chrono>
#include <stdexcept>
#include <string>

namespace npu_rt_api {

namespace {
[[noreturn]] void stub_called(const char* what) {
    throw std::runtime_error(std::string("npu_rt_api CI stub: ") + what +
                             " was called, but this build links the CI stub, not the NPU runtime");
}
}  // namespace

// The handle bodies the runtime library owns (npu/rt.h: "defined only inside the runtime library").
struct device::impl {};
struct bo::impl {};
struct elf::impl {};
struct hw_context::impl {};
struct run::impl {};
struct runlist::impl {};
namespace ext {
struct kernel::impl {};
}  // namespace ext

// --- device
device::device(unsigned) { stub_called("device::device"); }
device::device(const device&) { stub_called("device::device(copy)"); }
device& device::operator=(const device&) { stub_called("device::operator="); }
device::~device() = default;

// --- bo
bo::bo() = default;
bo::bo(const bo&, size_t, size_t) { stub_called("bo::bo(parent, size, offset)"); }
bo::bo(const bo&) { stub_called("bo::bo(copy)"); }
bo& bo::operator=(const bo&) { stub_called("bo::operator="); }
bo::~bo() = default;
void* bo::map() const { return nullptr; }
void bo::sync(int) { stub_called("bo::sync"); }
void bo::sync(int, size_t, size_t) { stub_called("bo::sync(size, offset)"); }
size_t bo::size() const { return 0; }
uint64_t bo::address() const { return 0; }

// --- ext::bo
ext::bo::bo(const device&, size_t) { stub_called("ext::bo::bo"); }
ext::bo::bo(const device&, int, size_t, void*) { stub_called("ext::bo::bo(dma-buf)"); }

// --- elf
elf::elf() = default;
elf::elf(const std::string&) { stub_called("elf::elf(path)"); }
elf::elf(const void*, size_t) { stub_called("elf::elf(data, size)"); }
elf::elf(const elf&) { stub_called("elf::elf(copy)"); }
elf& elf::operator=(const elf&) { stub_called("elf::operator="); }
elf::~elf() = default;
bool elf::is_full_elf() const { return false; }
uuid elf::get_cfg_uuid() const { return uuid{}; }

// --- hw_context
hw_context::hw_context() = default;
hw_context::hw_context(const device&, const elf&) { stub_called("hw_context::hw_context"); }
hw_context::hw_context(const hw_context&) { stub_called("hw_context::hw_context(copy)"); }
hw_context& hw_context::operator=(const hw_context&) { stub_called("hw_context::operator="); }
hw_context::~hw_context() = default;
void hw_context::add_config(const elf&) { stub_called("hw_context::add_config"); }

// --- ext::kernel
ext::kernel::kernel() = default;
ext::kernel::kernel(const hw_context&, const std::string&) { stub_called("ext::kernel::kernel"); }
ext::kernel::kernel(const kernel&) { stub_called("ext::kernel::kernel(copy)"); }
ext::kernel& ext::kernel::operator=(const kernel&) { stub_called("ext::kernel::operator="); }
ext::kernel::~kernel() = default;

// --- run
run::run() = default;
run::run(const ext::kernel&) { stub_called("run::run"); }
run::run(const run&) { stub_called("run::run(copy)"); }
run& run::operator=(const run&) { stub_called("run::operator="); }
run::~run() = default;
void run::set_arg(int, const bo&) { stub_called("run::set_arg"); }
void run::start() { stub_called("run::start"); }
ert_cmd_state run::wait(const std::chrono::milliseconds&) const { return ERT_CMD_STATE_COMPLETED; }
ert_cmd_state run::state() const { return ERT_CMD_STATE_COMPLETED; }

// --- runlist
runlist::runlist(const hw_context&) { stub_called("runlist::runlist"); }
runlist::~runlist() = default;
void runlist::add(const run&) { stub_called("runlist::add"); }
void runlist::execute() { stub_called("runlist::execute"); }
std::cv_status runlist::wait(const std::chrono::milliseconds&) const { return std::cv_status::no_timeout; }
void runlist::reset() { stub_called("runlist::reset"); }

}  // namespace npu_rt_api
