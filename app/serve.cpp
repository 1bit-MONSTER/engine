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

// `1bit serve -m <model> [--port 8000] [--device auto|npu|hrx|hybrid|cpu|ds4] [--mtp HEAD]`
//                        [--npu-opt KEY=VALUE ...] [--prefill-provider PATH]
//
// The engine's one front door (docs/serve.md): one model per process behind an
// OpenAI-compatible API. It is how the engine runs inside Lemonade: Lemonade's
// onebit backend runs `1bit serve` the way it runs llama-server:
//
//   GET  /health, /v1/health     503 while the model loads, then 200 while the backend answers its own
//                                (503 when it has exited, stopped answering, or the watchdog fired)
//   GET  /v1/models              the one model
//   POST /v1/chat/completions    (stream or not)
//   POST /v1/completions
//   GET  /metrics                serve's own Prometheus metrics, then the backend's (app/serve_metrics.h)
//
// Which device runs the model follows from the model and --device:
//   an NPU model directory (model.q4nx + npu/)  -> the NPU fast lane, in process
//   a model a private NPU route serves          -> that route, in process, in builds with
//                                                  -DONEBIT_NPU_PRIVATE (docs/npu.md)
//   a .gguf, --device hrx                       -> the HRX build's llama-server on HRX0
//   a .gguf, --device cpu                       -> the same llama-server with no GPU layers
//   a .gguf, --device hybrid                    -> the HRX llama-server decodes; the prompt's KV comes
//                                                  from an external prefill provider (the NPU engine,
//                                                  ONEBIT_HYBRID_PREFILL; app/serve_hybrid.h, docs/hybrid.md)
//   a .1bp (1BP v5 package), --device auto|ds4 -> the same ds4-server (our DwarfStar fork reads 1BP)
//   a DwarfStar .gguf, --device ds4             -> this build's DwarfStar ds4-server (DeepSeek
//                                                  V4/V4.1 Flash, GLM 5.x, Qwen3.8-Flash-Next in
//                                                  DwarfStar's own GGUFs; docs/dwarfstar.md)
//   --laya with --device auto                   -> Laya classifies each conversation and the route
//                                                  policy picks the device (RFC #186, docs/laya.md)
//   --pm [--pm-experts FILE] [--pm-router model|laya] [--lemonade-url U] [--pm-spawn-url URL]
//                                              -> the model is the Project Manager: each chat
//                                                  completion runs a tool loop in which it answers,
//                                                  delegates to an expert model Lemonade runs, or
//                                                  asks --pm-spawn-url for an agent worker
//                                                  (app/pm.cpp, docs/pm.md). --pm-router laya asks
//                                                  the Laya router which expert the request fits,
//                                                  before the PM's first round
//   --mtp <head.gguf> on hrx or cpu             -> multi-token prediction: the model's MTP
//                                                  head drafts, the model verifies (docs/serve.md)
//   --dflash <draft.gguf> on hrx or cpu         -> a DFlash block-diffusion draft model
//                                                  drafts a block per step (docs/serve.md)
//   --mmproj <mmproj.gguf> on hrx or cpu        -> images: the vision encoder runs beside the
//                                                  model, and chat messages may carry image parts
// For a .gguf, the engine starts that server as a private child on a loopback
// port and forwards the OpenAI routes to it, streaming included. `auto` picks
// HRX for a .gguf where the build has it, else the CPU (auto_gguf_device). The engine has
// no Vulkan, ROCm, Metal/MLX, ZINC or CUDA route: on the GPU it is HRX with HIP kernels.
// What HRX does not run, serve declines with a clear message, and Lemonade serves it with
// its own backends.
#include "serve.h"

#include "built_path.h"

#include "gguf_meta.h"
#include "hrx_archs.h"
#include "hrx_cwsr.h"
#include "hrx_strict.h"
#include "mem_admit.h"
#include "pm.h"
#include "recipes.h"
#include "serve_metrics.h"
#include "serve_watchdog.h"
#include "serve_hybrid.h"
#include "sysinfo.h"

#ifdef ONEBIT_MOE
#include "gguf_index.h"
#endif

#ifdef ONEBIT_NPU
#include "forward_serve.h"
#include "unified.h"
#endif
#ifdef ONEBIT_LAYA
#include "encoder.h"
#include "policy.h"
#include "route.h"
#include "scorer.h"
#endif

#include <httplib.h>
#include "http_guard.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
// httplib.h brought in winsock2.h and windows.h
#include <process.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
#include <pwd.h>
extern char** environ;
#endif

namespace onebit {

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Options {
    std::string model, host = "127.0.0.1", device = "auto", alias;
    int port = 8000, ctx_size = 0;
    std::string flash_attn;  // -fa on/off forwarded to the child llama-server
    std::string llama_server, hrx_libhsa, ds4;
    bool ssd_streaming = false;  // --device ds4: stream routed experts from the SSD
    std::string mtp;
    bool no_mtp = false;  // --no-mtp (or ONEBIT_NO_MTP=1): no drafting from the GGUF's own MTP head (own_mtp_head)
    std::string dflash;  // DFlash block-diffusion draft model for the target
    int moe_slots = 0;   // --moe-slots N: stream routed experts from the model file, N held in RAM (-1: auto)
    int moe_prefetch = 0;   // --moe-prefetch N: gate-ahead prefetch N MoE layers ahead (0: off)
    std::string moe_subst;  // --moe-subst R: a resident expert scoring at least R x a missing one takes its place
    std::string mmproj;  // a vision (or audio) projector: image parts in chat messages
    int mtp_max = 0;
    std::string mtp_p_min;
    int parallel = 0;
    std::string embed, rerank;   // RAG: an embedding model and a reranker, served beside the chat model
    std::string role;            // "embedding" or "reranking": the model itself is served in that role
    std::vector<std::string> npu_opts;   // --npu-opt KEY=VALUE, for a private NPU route
    std::string prefill_provider;        // --prefill-provider PATH (or ONEBIT_HYBRID_PREFILL): --device hybrid's prefill (docs/hybrid.md)
    std::string laya_model;              // --laya-model DIR: route each request by the Laya scorer (docs/laya.md)
    bool laya_auto = false;              // --laya: the same with the installed checkpoint (laya_checkpoint())
    std::string route_policy;            // --route-policy FILE: request class -> device (default: built in)
    std::string recipes;                 // --recipes FILE: tuned backend settings (default: built in, docs/recipes.md)
    bool no_recipes = false;             // --no-recipes: the backend gets serve's own flags only
    // --precise (or ONEBIT_PRECISE=1): every request re-prefills its whole conversation, so a follow-up turn
    // takes the same path a cold run of that conversation takes and comes out bitwise identical to it
    bool precise = false;
    bool pm = false;                     // --pm: the model is the Project Manager, delegating to experts (docs/pm.md)
    std::string pm_experts;              // --pm-experts FILE (default: config/pm-experts.json, built in)
    std::string pm_router;               // --pm-router model|laya: who picks the expert (default: model)
    std::string pm_spawn_url;            // --pm-spawn-url URL: the executor that starts agent workers (off by default)
    std::string lemonade_url = "http://127.0.0.1:8000";  // --lemonade-url: the Lemonade that runs the experts
    bool mem_check = true;               // --no-mem-check (or ONEBIT_MEM_CHECK=0): skip the admission check
    bool mem_check_only = false;         // --mem-check: print the estimate and exit (0 fits, 3 does not)
    // strict placement on HRX (app/hrx_strict.h): the default refuses a graph with a node off HRX;
    // --strict-device adds the every-graph report, --no-strict-device / ONEBIT_STRICT_DEVICE=0 lets HRX fall back
    StrictDevice strict = StrictDevice::Default;
    bool placement_report = false;       // --placement-report: ask the backend for its off-HRX op counts
    bool print_backend = false;          // --print-backend: print the llama-server command and exit (Lemonade one hop)
    bool print_backend_env = false;      // --print-backend-env: only its environment, NAME=VALUE lines (lemonade-hrx)
    WatchdogConfig watchdog;             // --watchdog SECS, --watchdog-prompt-tps N (app/serve_watchdog.h)
    CwsrPlacement cwsr_placement = CwsrPlacement::Auto;  // --hrx-cwsr-placement (app/hrx_cwsr.h, engine#315)
};

// HRX dlopens the HSA runtime, and a distro libhsa rejects gfx1151's
// PM4-emulation probe, so HRX registers no device (docs/hrx.md). Use, in order:
// --hrx-libhsa, the build's TheRock copy, or the first one under /opt/rocm-therock.
std::string hrx_libhsa(const std::string& option) {
    if (!option.empty()) return option;
#ifdef ONEBIT_HRX_LIBHSA
    if (fs::exists(ONEBIT_HRX_LIBHSA)) return ONEBIT_HRX_LIBHSA;
#endif
    std::error_code ec;
    for (fs::recursive_directory_iterator it("/opt/rocm-therock", fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it.depth() > 6) { it.disable_recursion_pending(); continue; }
        if (it->path().filename() == "libhsa-runtime64.so.1") return it->path().string();
    }
    return "";
}

// engine#315 candidate 1 (app/hrx_cwsr.h): the variables that put the CWSR save area in GTT, for
// every HRX backend's environment. Decided once per serve, on the first HRX backend, with one log line.
const std::vector<std::string>& hrx_cwsr_env(const Options& o) {
#if defined(ONEBIT_HRX_ALLOW_NO_SVM) && ONEBIT_HRX_ALLOW_NO_SVM
    constexpr bool gate = true;
#else
    constexpr bool gate = false;
#endif
    static const CwsrDecision d = [&] {
        CwsrDecision r = decide_cwsr_placement(o.cwsr_placement, read_cwsr_enable(), gate,
                                               [](const char* k) { return std::getenv(k); });
        std::fprintf(stderr, "1bit serve: %s\n", r.log.c_str());
        return r;
    }();
    return d.env;
}

// The llama-server --device hrx and cpu run: the HRX build's (HRX0 and the CPU, no Vulkan),
// else one on PATH.
std::string default_llama_server() {
    if (const char* e = std::getenv("ONEBIT_LLAMA_SERVER"); e && *e) return e;
#ifdef ONEBIT_HRX_SERVER
    return built_path(ONEBIT_HRX_SERVER);
#else
    return "llama-server";
#endif
}

// The devices this engine no longer has (RFC #213 stage 3, 2026-10-01: HRX + NPU only, zero
// Vulkan, zero ROCm; the core strip, 2026-10-04: no ZINC). Asked for by name, serve says so
// instead of failing somewhere later.
void refuse_removed_device(const std::string& device) {
    if (device == "vulkan" || device == "rocm")
        throw std::runtime_error("--device " + device + " was removed: the engine's GPU route is HRX (--device hrx, or auto), "
                                 "and it builds no Vulkan or ROCm llama.cpp (RFC #213). Lemonade's own llamacpp backend "
                                 "serves Vulkan and ROCm (docs/lemonade.md)");
    if (device == "zinc")
        throw std::runtime_error("--device zinc was removed: the engine no longer builds ZINC (its GPU route is HRX, "
                                 "--device hrx, or auto). Lemonade's own llamacpp backends serve other GPUs (docs/lemonade.md)");
    if (device == "mlx")
        throw std::runtime_error("--device mlx was removed: the engine has no MLX or Metal route (its GPU route is HRX). "
                                 "Lemonade's own MLX backend serves Apple Silicon (docs/lemonade.md)");
    if (device == "onnx")
        throw std::runtime_error("--device onnx was removed: the engine ships no ONNX Runtime (its NPU route is its own "
                                 "engine, --device npu). Lemonade's own backends serve ONNX models where they exist "
                                 "(ryzenai-llm on Windows, onnxruntime on the CPU; docs/lemonade.md)");
}

// An "H32" file: a Q4_0 whose matmul weights were rotated by a 32-point Walsh-Hadamard transform
// and stamped onebit.hadamard_q4_0 = 32 by the engine's former tools/hadamard_q4_0.py. The format
// is dropped (owner, 2026-10-03: slower and less accurate on HRX than UD-Q4_K_XL), so serve refuses
// it by name. Not to be confused with PrismML's own rotation (prism_hadamard below), which stays.
bool hadamard_q4_0(const std::string& model) {
    return model.size() > 5 && model.compare(model.size() - 5, 5, ".gguf") == 0 &&
           gguf_int(model, "onebit.hadamard_q4_0") == 32;
}

// A ternary model written as exact Q4_0 by tools/ternary_to_q4_0.py (each 128-value group of trits
// with one scale). HRX0 decodes it from a 2-bit repack of those weights (llama.cpp #54).
bool ternary_q4_0(const std::string& model) {
    return model.size() > 5 && model.compare(model.size() - 5, 5, ".gguf") == 0 &&
           gguf_int(model, "onebit.ternary_q4_0") == 128;
}

// A PrismML model (Ternary Bonsai): prism.hadamard.* says its weights are stored rotated by a
// Walsh-Hadamard transform. Our llama.cpp (the HRX build, src/llama-hadamard.cpp) rotates the
// activations to match; upstream llama.cpp would load the file and answer garbage.
bool prism_hadamard(const std::string& model) {
    return model.size() > 5 && model.compare(model.size() - 5, 5, ".gguf") == 0 &&
           gguf_int(model, "prism.hadamard.version") > 0;
}

// PrismML's own ternary types (general.file_type PQ2_0 / PTQ1_0 in its fork): only the HRX
// build reads them (llama.cpp #62); tools/ternary_to_q4_0.py writes the same weights as Q4_0
// for any other route.
bool prism_ternary_types(const std::string& model) {
    const long long ft = gguf_int(model, "general.file_type");
    return ft == 128 || ft == 129 || ft == 141 || ft == 142 || ft == 143;
}

// A 1BP package (1bit-MONSTER's model format; our DwarfStar fork reads version 5, its
// docs/1BP.md): its first four bytes are "1BP\0".
bool onebp_file(const std::string& model) {
    std::ifstream f(model, std::ios::binary);
    char magic[4] = {};
    return f.read(magic, 4) && std::memcmp(magic, "1BP\0", 4) == 0;
}

// DwarfStar's ds4-server (third_party/ds4, MIT), built by scripts/build-ds4.sh
std::string default_ds4() {
    if (const char* e = std::getenv("ONEBIT_DS4"); e && *e) return e;
#ifdef ONEBIT_DS4_SERVER
    return built_path(ONEBIT_DS4_SERVER);
#else
    return "ds4-server";
#endif
}

// SIGTERM / SIGINT: stop serving and take the backend down with us.
std::atomic<bool> g_stop{false};
extern "C" void on_stop_signal(int) { g_stop = true; }

int free_port() {
#ifdef _WIN32
    // Winsock is started by httplib.h (its WSInit)
    const SOCKET s = ::socket(AF_INET, SOCK_STREAM, 0);
    const bool bad_socket = s == INVALID_SOCKET;
    auto close_socket = [&] { ::closesocket(s); };
    int len = sizeof(sockaddr_in);
#else
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    const bool bad_socket = s < 0;
    auto close_socket = [&] { ::close(s); };
    socklen_t len = sizeof(sockaddr_in);
#endif
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bad_socket || ::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
        ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len) != 0) {
        if (!bad_socket) close_socket();
        throw std::runtime_error("cannot find a free loopback port");
    }
    const int port = ntohs(a.sin_port);
    close_socket();
    return port;
}

#if defined(__linux__)
// Who holds a loopback port: the backend serve started (or a process it started), or someone
// else. free_port() releases its port before the child binds it, so another local process could
// bind it first and answer the readiness probe; serve must not send it client requests.
enum class PortOwner { kOurs, kOther, kUnknown };

PortOwner port_owner(pid_t child, int port) {
    // the LISTEN sockets on the port (any address), by inode
    std::vector<std::string> inodes;
    bool readable = false;
    for (const char* table : {"/proc/net/tcp", "/proc/net/tcp6"}) {
        std::ifstream f(table);
        if (!f) continue;
        readable = true;
        std::string line;
        std::getline(f, line);  // header
        while (std::getline(f, line)) {
            // sl local_address rem_address st tx:rx tr:when retrnsmt uid timeout inode
            std::istringstream in(line);
            std::string sl, local, remote, st, txrx, trwhen, retr, uid, timeout, inode;
            if (!(in >> sl >> local >> remote >> st >> txrx >> trwhen >> retr >> uid >> timeout >> inode)) continue;
            const size_t colon = local.rfind(':');
            if (st != "0A" || colon == std::string::npos) continue;  // 0A: LISTEN
            if (std::strtol(local.c_str() + colon + 1, nullptr, 16) == port) inodes.push_back("socket:[" + inode + "]");
        }
    }
    if (!readable || inodes.empty()) return PortOwner::kUnknown;
    // the child and its descendants (a backend may be a wrapper that starts the real server)
    std::unordered_map<pid_t, pid_t> parent;
    std::error_code ec;
    for (const auto& ent : fs::directory_iterator("/proc", ec)) {
        const std::string name = ent.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) continue;
        std::ifstream st(ent.path() / "stat");
        std::string stat;
        std::getline(st, stat);
        const size_t close = stat.rfind(')');  // the command name may hold spaces and parentheses
        if (close == std::string::npos) continue;
        std::istringstream rest(stat.substr(close + 1));
        std::string state;
        long ppid = 0;
        if (rest >> state >> ppid) parent[pid_t(std::stol(name))] = pid_t(ppid);
    }
    for (const auto& [pid, _] : parent) {
        bool ours = false;
        for (pid_t p = pid, hops = 0; p > 1 && hops < 64; p = parent.count(p) ? parent[p] : 0, ++hops)
            if (p == child) { ours = true; break; }
        if (!ours) continue;
        const fs::path fds = fs::path("/proc") / std::to_string(pid) / "fd";
        for (const auto& fd : fs::directory_iterator(fds, ec)) {
            std::error_code lec;
            const std::string target = fs::read_symlink(fd.path(), lec).string();
            if (!lec && std::find(inodes.begin(), inodes.end(), target) != inodes.end()) return PortOwner::kOurs;
        }
    }
    return PortOwner::kOther;
}
#endif

// A child OpenAI-compatible server on a loopback port.
// The child's environment: ours, plus env_extra where a variable is not set (a value the user
// set wins), plus env_override in place of whatever was set (the few a route must own).
void merge_env(std::vector<std::string>& env_store, const std::vector<std::string>& env_extra,
               const std::vector<std::string>& env_override) {
    for (const auto& e : env_extra) {
        const std::string key = e.substr(0, e.find('=') + 1);
        bool present = false;
        for (const auto& have : env_store) present = present || have.rfind(key, 0) == 0;
        if (!present) env_store.push_back(e);
    }
    for (const auto& e : env_override) {
        const std::string key = e.substr(0, e.find('=') + 1);
        std::erase_if(env_store, [&](const std::string& have) { return have.rfind(key, 0) == 0; });
        env_store.push_back(e);
    }
}

// ONEBIT_BACKEND_STOP_GRACE_S: how long Child::stop waits after SIGTERM before SIGKILL (default 30 s).
int stop_grace_s() {
    if (const char* e = std::getenv("ONEBIT_BACKEND_STOP_GRACE_S"); e && *e) {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end && *end == '\0' && v >= 0 && v <= 86400) return int(v);
        std::fprintf(stderr, "1bit serve: ONEBIT_BACKEND_STOP_GRACE_S=%s ignored (whole seconds, 0..86400); using 30\n", e);
    }
    return 30;
}

#ifdef _WIN32
// Windows: CreateProcess into a job object that kills its processes when its last handle
// closes, so the backend dies with `1bit serve` for any reason (Linux: PR_SET_PDEATHSIG).
class Child {
public:
    Child(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int port,
          std::string ready_path = "/health", std::function<void(const std::string&)> on_line = nullptr,
          const std::vector<std::string>& env_override = {})
        : port_(port), ready_path_(std::move(ready_path)) {
        (void)on_line;  // the backend's stderr is its own on Windows
        // the command line, each argument quoted the way CommandLineToArgvW reads it back
        std::string cmd;
        for (const auto& a : argv) {
            if (!cmd.empty()) cmd += ' ';
            cmd += '"';
            size_t bs = 0;
            for (char ch : a) {
                if (ch == '\\') { bs++; continue; }
                if (ch == '"') cmd.append(bs * 2 + 1, '\\');
                else cmd.append(bs, '\\');
                bs = 0;
                cmd += ch;
            }
            cmd.append(bs * 2, '\\');
            cmd += '"';
        }
        // our environment plus env_extra (a value the user set wins), as a double-NUL block
        std::vector<std::string> env_store;
        if (char* block = ::GetEnvironmentStringsA()) {
            for (char* e = block; *e; e += std::strlen(e) + 1) env_store.emplace_back(e);
            ::FreeEnvironmentStringsA(block);
        }
        merge_env(env_store, env_extra, env_override);
        std::string env_block;
        for (const auto& e : env_store) { env_block += e; env_block += '\0'; }
        env_block += '\0';
        job_ = ::CreateJobObjectA(nullptr, nullptr);
        if (job_) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            ::SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof li);
        }
        STARTUPINFOA si{};
        si.cb = sizeof si;
        PROCESS_INFORMATION pi{};
        if (!::CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, env_block.data(), nullptr,
                              &si, &pi))
            throw std::runtime_error("cannot start " + argv[0] + " (error " + std::to_string(::GetLastError()) + ")");
        if (job_) ::AssignProcessToJobObject(job_, pi.hProcess);   // before it runs: its children join too
        ::ResumeThread(pi.hThread);
        ::CloseHandle(pi.hThread);
        proc_ = pi.hProcess;
    }
    ~Child() {
        stop();
        if (job_) ::CloseHandle(job_);
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;

    void stop() {
        if (!proc_) return;
        ::TerminateProcess(proc_, 1);   // no SIGTERM on Windows; llama-server keeps no state to flush
        ::WaitForSingleObject(proc_, 5000);
        ::CloseHandle(proc_);
        proc_ = nullptr;
    }

    bool alive() const { return proc_ && ::WaitForSingleObject(proc_, 0) == WAIT_TIMEOUT; }

    bool kill_now(std::chrono::seconds wait) {
        if (!proc_) return true;
        ::TerminateProcess(proc_, 1);
        const bool gone = ::WaitForSingleObject(proc_, DWORD(wait.count() * 1000)) == WAIT_OBJECT_0;
        if (gone) { ::CloseHandle(proc_); proc_ = nullptr; }
        return gone;
    }
    int pid() const { return proc_ ? int(::GetProcessId(proc_)) : -1; }

#else
class Child {
public:
    // on_line, where given, sees every complete line the backend writes to its stderr (Linux)
    Child(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int port,
          std::string ready_path = "/health", std::function<void(const std::string&)> on_line = nullptr,
          const std::vector<std::string>& env_override = {})
        : port_(port), ready_path_(std::move(ready_path)) {
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        std::vector<std::string> env_store;
        for (char** e = environ; *e; ++e) env_store.emplace_back(*e);
        merge_env(env_store, env_extra, env_override);
        std::vector<char*> envp;
        for (auto& e : env_store) envp.push_back(e.data());
        envp.push_back(nullptr);
#if defined(__linux__)
        // The backend's stderr comes through serve (passed on unchanged), so serve keeps its last
        // lines for a watchdog report and can read what it says (docs/serve.md).
        int err[2] = {-1, -1};
        const bool tee = ::pipe2(err, O_CLOEXEC) == 0;
        const pid_t parent = ::getpid();
        pid_ = ::fork();
        if (pid_ < 0) throw std::runtime_error("cannot fork for " + argv[0] + ": " + std::strerror(errno));
        if (pid_ == 0) {
            // The kernel ends the child when `1bit serve` dies for any reason,
            // SIGKILL included, so a backend is never left running on its own.
            ::prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (::getppid() != parent) ::_exit(127);  // parent already gone
            if (tee) ::dup2(err[1], 2);
            ::execvpe(args[0], args.data(), envp.data());
            ::_exit(127);
        }
        if (tee) {
            ::close(err[1]);
            // detached: a backend stuck in the kernel keeps the pipe open, and serve must still exit
            std::thread([fd = err[0], tail = tail_, on_line = std::move(on_line)] {
                char buf[4096];
                for (;;) {
                    const ssize_t n = ::read(fd, buf, sizeof buf);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) break;
                    for (ssize_t done = 0; done < n;) {
                        const ssize_t w = ::write(2, buf + done, size_t(n - done));
                        if (w < 0 && errno == EINTR) continue;
                        if (w <= 0) break;
                        done += w;
                    }
                    for (const std::string& line : tail->feed(buf, size_t(n)))
                        if (on_line) on_line(line);
                }
                ::close(fd);
            }).detach();
        }
#else
        // No parent-death signal outside Linux: the SIGTERM/SIGINT handler
        // below still takes the child down with a normal stop.
        if (::posix_spawnp(&pid_, args[0], nullptr, nullptr, args.data(), envp.data()) != 0)
            throw std::runtime_error("cannot start " + argv[0] + ": " + std::strerror(errno));
#endif
    }
    ~Child() { stop(); }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;

    // SIGTERM to the child itself (it shares our process group, so no group signal: that would
    // reach serve and every other backend), then a wait before SIGKILL: a backend that holds an
    // NPU hardware context must get to close it itself.
    // Killing such a process outright, with a live context after an earlier firmware timeout,
    // took the kernel down on amdxdna 0.17.0 (2026-10-11, list_del corruption in
    // amdxdna_client_cleanup). The wait is ONEBIT_BACKEND_STOP_GRACE_S (30 s by default: a
    // backend busy in a long dispatch); the escalation stays, a hung backend for ever is worse.
    // A child that already exited (reaped by alive()) needs nothing: no signal, no wait.
    void stop() {
        if (pid_ <= 0 || reap(WNOHANG)) return;
        ::kill(pid_, SIGTERM);
        const int grace = stop_grace_s();
        for (int i = 0; i < grace * 10; i++) {
            if (reap(WNOHANG)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::fprintf(stderr, "1bit serve: backend pid %d ignored SIGTERM for %d s; sending SIGKILL "
                             "(on amdxdna, killing a process that holds an NPU context can take the kernel down; "
                             "ONEBIT_BACKEND_STOP_GRACE_S sets this wait)\n", int(pid_), grace);
        ::kill(pid_, SIGKILL);
        reap(0);
    }

    bool alive() const { return pid_ > 0 && !reap(WNOHANG); }

    // how the child ended ("exit 3", "signal 6"), "" while it runs or before it was reaped
    std::string exit_text() const {
        if (!exited_) return "";
        if (WIFEXITED(status_)) return "exit " + std::to_string(WEXITSTATUS(status_));
        if (WIFSIGNALED(status_)) return "signal " + std::to_string(WTERMSIG(status_));
        return "gone";
    }

    // The watchdog's stop: SIGKILL at once (a wedged GPU process does not act on SIGTERM), then
    // up to `wait` for it to go. False when it is still there (stuck in the kernel).
    bool kill_now(std::chrono::seconds wait) {
        if (reap(WNOHANG)) return true;
        ::kill(pid_, SIGKILL);
        const auto end = std::chrono::steady_clock::now() + wait;
        while (std::chrono::steady_clock::now() < end) {
            if (reap(WNOHANG)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return false;
    }
    int pid() const { return int(pid_); }
#endif
    // the backend's last stderr lines (empty where serve does not pass its stderr through)
    std::vector<std::string> last_lines(size_t n) const { return tail_->last(n); }
    const std::string& ready_path() const { return ready_path_; }

    // Polls the child's ready path (/health; /v1/models for servers without one, which answer
    // only once the model is loaded) until it answers 200, it exits, or time runs out.
    bool wait_ready(std::chrono::seconds timeout, const std::atomic<bool>& stop) const {
        httplib::Client c("127.0.0.1", port_);
        c.set_connection_timeout(1);
        const auto end = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < end) {
            if (!alive() || stop) return false;
            if (auto r = c.Get(ready_path_); r && r->status == 200) {
#if defined(__linux__)
                if (port_owner(pid_, port_) == PortOwner::kOther) {
                    std::fprintf(stderr, "1bit serve: port %d answers, but not from the backend serve started (pid %d); "
                                         "refusing to use it\n", port_, int(pid_));
                    return false;
                }
#endif
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        return false;
    }

    int port() const { return port_; }

private:
#ifdef _WIN32
    HANDLE proc_ = nullptr, job_ = nullptr;
#else
    // Every waitpid on the child goes through here, so alive() and stop() agree: once the child
    // is reaped (options WNOHANG to poll, 0 to wait) its status is kept and pid_ is -1. True once
    // the child is gone. ECHILD (reaped elsewhere) counts as gone with no status.
    bool reap(int options) const {
        if (pid_ <= 0) return true;
        int status = 0;
        pid_t r;
        while ((r = ::waitpid(pid_, &status, options)) < 0 && errno == EINTR) {}
        if (r == 0) return false;
        if (r == pid_) { exited_ = true; status_ = status; }
        pid_ = -1;
        return true;
    }
    mutable pid_t pid_ = -1;      // the running child; -1 once reaped (or never started)
    mutable bool exited_ = false; // reaped with a status (status_)
    mutable int status_ = 0;
#endif
    int port_;
    std::string ready_path_;
    std::shared_ptr<LineTail> tail_ = std::make_shared<LineTail>();
};

std::string model_id(const Options& o) {
    if (!o.alias.empty()) return o.alias;
    fs::path p(o.model);
    if (!p.has_filename()) p = p.parent_path();
    // Only a .gguf loses its extension: "Qwen3-0.6B-4bit" is a name, not a stem.
    return p.extension() == ".gguf" || p.extension() == ".1bp" ? p.stem().string() : p.filename().string();
}

// Forwards an OpenAI POST to the child: the model id becomes the child's (a
// child that rejects ids it did not load gets none), and replies carry ours.
// Counts a request against a backend from the moment it is routed until its reply
// (streamed or not) is complete; the exit summary reports these counts.
struct InFlight {
    explicit InFlight(std::atomic<int>* n) : n_(n) { if (n_) ++*n_; }
    ~InFlight() { if (n_) --*n_; }
    InFlight(const InFlight&) = delete;
    InFlight& operator=(const InFlight&) = delete;
private:
    std::atomic<int>* n_;
};

// whether a request body asks for a streamed reply
bool wants_stream(const std::string& body) {
    try {
        return json::parse(body).value("stream", false);
    } catch (const std::exception&) {
        return false;
    }
}

// watch, where given, tracks the request for the progress watchdog (app/serve_watchdog.h).
// body_override, where given, is the JSON body to send in place of the request's (hybrid: id_slot and
// cache_prompt added); on_first_byte, where given, runs when the first byte of the reply arrives.
void forward(int port, std::shared_ptr<InFlight> held, int draft_max, const std::string& our_id, bool drop_model,
             const std::string& set_model, bool precise, const httplib::Request& req, httplib::Response& res,
             ProgressWatch* watch = nullptr, const json* body_override = nullptr,
             std::function<void()> on_first_byte = nullptr) {
    json body;
    try {
        body = body_override ? *body_override : json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
        return;
    }
    if (drop_model) body.erase("model");
    // --precise: a follow-up turn differs from a cold run of the same conversation because its new tokens form a
    // different batch than the cold prefill does (different kernels, different reduction order), and a near-tie
    // then flips. The only way to make the two bitwise identical is to take the cold path: re-prefill the whole
    // conversation instead of continuing from the KV cache. That is what this costs - the TTFT of the whole
    // conversation on every turn - which is why it is off by default.
    if (precise) body["cache_prompt"] = false;
    // MTP by load: a request's own draft length wins; otherwise the router's (see post)
    if (draft_max >= 0 && !body.contains("speculative.n_max")) body["speculative.n_max"] = draft_max;
    if (!set_model.empty()) body["model"] = set_model;
    const bool stream = body.value("stream", false);
    const std::string payload = body.dump();
    const std::string path = req.path;
    if (!stream) {
        // nothing arrives until the whole answer does: the backend's probe is the only progress
        const auto tracked = watch ? watch->begin(port, "POST " + path, req.body.size(), true) : nullptr;
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(3600);
        // The backend learns of a client that hung up only when serve closes its own connection:
        // wait for the reply on another thread and close it as soon as the client is gone.
        auto pending = std::async(std::launch::async, [&] { return c.Post(path, payload, "application/json"); });
        bool closed = false;
        while (pending.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
            if (!closed && req.is_connection_closed()) {
                closed = true;
                c.stop();
            }
        auto r = pending.get();
        if (on_first_byte) on_first_byte();
        if (!r) {
            res.status = 502;
            res.set_content(R"({"error":{"message":"backend did not answer"}})", "application/json");
            return;
        }
        res.status = r->status;
        try {
            json out = json::parse(r->body);
            if (out.is_object() && out.contains("model")) out["model"] = our_id;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception&) {
            res.set_content(r->body, r->get_header_value("Content-Type"));
        }
        return;
    }
    // Streaming: relay the child's SSE bytes as they arrive.
    const size_t prompt_bytes = req.body.size();
    res.set_chunked_content_provider("text/event-stream", [port, path, payload, held, watch, prompt_bytes, on_first_byte](size_t, httplib::DataSink& sink) {
        const auto tracked = watch ? watch->begin(port, "POST " + path + " (streamed)", prompt_bytes, false) : nullptr;
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(3600);
        bool first = true;
        c.Post(path, httplib::Headers{}, payload, "application/json",
               [&sink, &tracked, &first, &on_first_byte](const char* data, size_t n) {
                   if (first && on_first_byte) on_first_byte();
                   first = false;
                   if (tracked) ProgressWatch::progress(*tracked, n);
                   return sink.write(data, n);
               });
        sink.done();
        return true;
    });
}

// llama-server's own routes that are not chat or completions (/slots, /props, /metrics):
// method, query string and body pass through unchanged; the reply comes back as it is.
void relay(int port, const httplib::Request& req, httplib::Response& res) {
    std::string path = req.path;
    if (!req.params.empty()) {
        std::string query;
        for (const auto& [k, v] : req.params)
            query += (query.empty() ? "?" : "&") + httplib::encode_query_component(k) + "=" + httplib::encode_query_component(v);
        path += query;
    }
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(3600);
    const std::string type = req.get_header_value("Content-Type").empty() ? "application/json" : req.get_header_value("Content-Type");
    auto r = req.method == "GET" ? c.Get(path) : c.Post(path, req.body, type);
    if (!r) {
        res.status = 502;
        res.set_content(R"({"error":{"message":"backend did not answer"}})", "application/json");
        return;
    }
    res.status = r->status;
    res.set_content(r->body, r->get_header_value("Content-Type").empty() ? "application/json" : r->get_header_value("Content-Type"));
}

struct Launch {
    std::string device;
    int port = 0;
    std::vector<std::string> argv, env;
    bool drop_model = false;
    std::string set_model;
    bool mtp = false;
    std::string ready_path = "/health";
    std::vector<std::string> env_override;  // set in place of the inherited value (Child)
    bool hybrid = false;     // --device hybrid: the prompt's KV comes from the prefill provider (app/serve_hybrid.h)
    std::string slot_dir;    // hybrid: the backend's --slot-save-path, where the handoff files go
};

// A GGUF that carries its own MTP head (<architecture>.nextn_predict_layers > 0: Unsloth's Qwen3.8-27B
// UD-Q4_K_XL holds it as blk.64.nextn.*) drafts with it by default on HRX. That is what Lemonade's own
// llamacpp recipe does for a model with the "mtp" label (Lemonade reads the same key): `--spec-type
// draft-mtp` and no -md, so llama-server loads the head from the model file. Lemonade's standard benchmark
// (P1, 2026-10-21) had Lemonade's Qwen3.8-27B entry drafting at 24-28 tok/s while the onebit recipe ran it
// without a drafter at 11-12. Not when another drafter is asked for (--mtp, --dflash), for a role server
// (--embedding, --reranking), with --parallel > 1 (llama-server drafts per slot; not measured on HRX), or
// with --no-mtp / ONEBIT_NO_MTP=1.
bool own_mtp_head(const Options& o, const std::string& device) {
    if (device != "hrx" || o.no_mtp || !o.mtp.empty() || !o.dflash.empty() || !o.role.empty() || o.parallel > 1)
        return false;
    if (const char* e = std::getenv("ONEBIT_NO_MTP"); e && *e && std::string(e) != "0") return false;
    if (fs::path(o.model).extension() != ".gguf") return false;
    return gguf_int(o.model, gguf_architecture(o.model) + ".nextn_predict_layers") > 0;
}

// The backend process for one device: its command line and environment.
#ifdef ONEBIT_MOE
uint64_t mem_available() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    std::fclose(f);
    return uint64_t(kb) << 10;
}

// --moe-slots auto: as many experts as fit in free memory beside the tensors the GPU holds, keeping
// a reserve (8 GiB or 15% of free memory) for the KV cache, buffers and everything else. The
// streamer gives every MoE layer an equal share, so the cache costs slots x the mean expert size.
[[maybe_unused]] int auto_moe_slots(const std::string& model) {
    const auto idx = moe::GgufIndex::open(model);
    if (idx.experts.empty()) throw std::runtime_error("--moe-slots: " + model + " has no routed experts");
    double mean = 0;
    for (const auto& [layer, parts] : idx.experts) mean += double(idx.expert_bytes(layer));
    mean /= double(idx.experts.size());
    uint64_t all = 0, experts = 0, mapped = 0;
    for (const auto& t : idx.tensors) {
        all += t.bytes;
        if (t.name.find("_exps.") != std::string::npos) experts += t.bytes;
        else if (t.name.rfind("per_layer_token_embd", 0) == 0) mapped += t.bytes;  // stays file-backed
    }
    const double gib = double(1ull << 30);
    const uint64_t others = all - experts - mapped, avail = mem_available();
    const uint64_t reserve = std::max<uint64_t>(8ull << 30, avail / 100 * 15);
    const int64_t total = int64_t(idx.n_expert) * int64_t(idx.experts.size());
    const int64_t floor = 64 * int64_t(idx.experts.size());  // decode batches stream only above ~8 x top-k per layer
    const double room = double(avail) - double(others) - double(reserve);
    int64_t n = room > 0 ? int64_t(room / mean) : 0;
    n = std::min(n, total);
    if (n < floor) {
        std::fprintf(stderr, "1bit serve: --moe-slots auto: only %.1f GiB free for experts; using %lld slots (%.1f GiB) anyway\n",
                     std::max(0.0, room) / gib, (long long) floor, floor * mean / gib);
        n = floor;
    }
    std::fprintf(stderr, "1bit serve: --moe-slots auto: %lld of %lld experts (%.1f GiB); %.1f GiB free, %.1f GiB of other tensors on the GPU\n",
                 (long long) n, (long long) total, n * mean / gib, avail / gib, others / gib);
    return int(n);
}
#endif

// The GGUF architectures our llama.cpp builds a model for, on HRX0 or on the CPU: generated from
// registry/architectures.json (gguf_architectures.hrx, tools/registry_build.py), so a pin bump that
// regenerates the registry moves this list too. Anything else (Qwen3.8-Flash-Next's qwen4exp, Zyphra
// Zamba/Zamba2/BlackMamba, the upstream-only architectures the removed Vulkan build ran) serve
// declines with a clear message, and Lemonade's own llamacpp backend serves it. An empty
// architecture (a file serve cannot read) is left to llama-server.
bool hrx_missing_arch(const std::string& arch) {
    if (arch.empty()) return false;
    for (const char* a : kHrxGgufArchs)
        if (arch == a) return false;
    return true;
}

void refuse_missing_arch(const std::string& model) {
    if (const std::string arch = gguf_architecture(model); hrx_missing_arch(arch))
        throw std::runtime_error(model + ": architecture " + arch + " is not in the engine's llama.cpp "
                                 "(registry/architectures.json); Lemonade's llamacpp backend serves it");
}

bool delta_net_arch(const std::string& arch) {
    return arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next";
}

std::string auto_gguf_device(const Options& o);

Launch launch_for(const Options& o, const std::string& device, int child_port) {
    Launch l;
    l.device = device;
    l.port = child_port;
    std::vector<std::string>& argv = l.argv;
    std::vector<std::string>& env = l.env;
    std::string& set_model = l.set_model;
    if (o.ssd_streaming && device != "ds4") throw std::runtime_error("--ssd-streaming works with --device ds4");
    if (device == "hybrid") {
        // Hybrid (docs/hybrid.md, app/serve_hybrid.h): llama-server decodes on HRX (the CPU in a build
        // without HRX, so the path runs anywhere), from a KV cache an external prefill provider computed
        // and serve restored into its one slot. The handoff is a sequence-state file in --slot-save-path.
        // Nothing else is handed over: no drafter, no images, no second slot.
        if (!o.mtp.empty() || !o.dflash.empty() || !o.mmproj.empty() || o.parallel > 1 || !o.role.empty() || o.moe_slots != 0)
            throw std::runtime_error("--device hybrid takes no --mtp, --dflash, --mmproj, --moe-slots, --parallel > 1 or a role: "
                                     "one slot, decoding from the provider's KV (docs/hybrid.md)");
#if defined(_WIN32)
        throw std::runtime_error("--device hybrid is a Linux route");
#else
        Options one = o;
        one.parallel = 1;
        one.no_mtp = true;   // the slot's KV comes from the provider; the GGUF's own MTP head would draft over it unverified
        Launch l = launch_for(one, auto_gguf_device(o), child_port);
        char dir[] = "/tmp/1bit-hybrid-XXXXXX";
        if (!::mkdtemp(dir)) throw std::runtime_error(std::string("--device hybrid: cannot make a slot directory: ") + std::strerror(errno));
        l.slot_dir = dir;
        l.argv.insert(l.argv.end(), {"--slot-save-path", l.slot_dir});
        l.device = "hybrid";
        l.hybrid = true;
        return l;
#endif
    }
    if (device == "hrx" || device == "cpu") {
        // --device cpu: llama-server with no GPU layers (what --device auto falls back to where
        // HRX has no route for a request, and in a build without HRX)
        refuse_missing_arch(o.model);
        argv = {o.llama_server.empty() ? default_llama_server() : o.llama_server,
                "-m", o.model, "--host", "127.0.0.1", "--port", std::to_string(child_port)};
        argv.push_back("--metrics");   // its /metrics, appended to serve's own
        if (device == "cpu") argv.insert(argv.end(), {"-ngl", "0", "--jinja"});
        else argv.insert(argv.end(), {"--device", "HRX0", "-ngl", "99", "--jinja"});
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        if (!o.flash_attn.empty()) { argv.insert(argv.end(), {"-fa", o.flash_attn}); }
        if (o.moe_slots != 0) {
            // --moe-slots (docs/moe-streaming.md) is kept, not dropped: the expert streamer lives in
            // the Vulkan llama.cpp build the engine no longer ships (RFC #213 stage 3), and it moves to
            // HRX together with Qwen3.8-Flash-Next (qwen4exp). Until the HRX llama.cpp reads
            // ONEBIT_MOE_FILE / ONEBIT_MOE_SLOTS, a build says so instead of loading every expert.
#ifndef ONEBIT_MOE_STREAMING_HRX
            throw std::runtime_error("--moe-slots is not available in this build: expert streaming ran on the Vulkan "
                                     "llama.cpp build, which the engine no longer ships (RFC #213). It moves to HRX "
                                     "with Qwen3.8-Flash-Next (docs/moe-streaming.md); until then, leave --moe-slots off");
#else
            if (device != "hrx") throw std::runtime_error("--moe-slots works with --device hrx");
            // routed experts stay in the file (memory-mapped, never read whole); N of them are cached in
            // GPU buffers and streamed in as the router asks (docs/moe-streaming.md)
            env.push_back("ONEBIT_MOE_FILE=" + std::filesystem::absolute(o.model).string());
            int slots = o.moe_slots;
            if (slots < 0) {
#ifdef ONEBIT_MOE
                slots = auto_moe_slots(o.model);
#else
                throw std::runtime_error("--moe-slots auto needs a Linux build (ONEBIT_MOE)");
#endif
            }
            env.push_back("ONEBIT_MOE_SLOTS=" + std::to_string(slots));
            // off unless asked: where the drive is the limit, wrong guesses cost more than right ones save
            env.push_back("ONEBIT_MOE_PREFETCH=" + std::to_string(o.moe_prefetch));
            if (!o.moe_subst.empty()) env.push_back("ONEBIT_MOE_SUBST=" + o.moe_subst);
            // the experts stay file-backed: without these, llama.cpp copies every one into RAM
            // (a pinned host buffer, or a CPU repack), which is what streaming avoids
            // (per_layer_token_embd: Qwen3.8-Flash-Next's 26.8 GiB table, only ever row-gathered)
            argv.insert(argv.end(), {"-ot", "exps=CPU,per_layer_token_embd=CPU", "--no-host", "--no-repack", "--load-mode", "mmap"});
#endif
        }
        if (device == "hrx") {
            const std::string hsa = hrx_libhsa(o.hrx_libhsa);
            if (!hsa.empty()) env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
            for (const auto& e : hrx_cwsr_env(o)) env.push_back(e);  // engine#315 (app/hrx_cwsr.h)
            // engine #390 (docs/serve.md, "Off-HRX placement"): the backend's placement observer
            // prints one line per graph when GGML_HRX_REPORT is set, and refuses the graph with
            // GGML_HRX_STRICT. Both are off in the backend by default; serve turns strict on for
            // every HRX load unless --no-strict-device (app/hrx_strict.h), and a value the user sets
            // in the environment wins (Child keeps it; --print-backend leaves it out of env).
            // GGML_HRX_REPORT=2 = every graph: the warmup graph alone is not the model (census
            // 2026-10-20, finding 1: LFM2 --parallel 4 and ZAYA1-8B kept 20 and 3 nodes on the CPU in
            // prompt and multi-sequence graphs while the warmup report said all on HRX).
            for (const auto& e : hrx_placement_env(o.strict, o.placement_report)) env.push_back(e);
            // llama-server drops the library's INFO lines below verbosity 4 (common/log.cpp maps
            // GGML_LOG_LEVEL_INFO to its trace level), so the report reaches serve only when a flag
            // reads it; the default strict abort is an error line and needs no verbosity
            if (hrx_placement_verbose(o.strict, o.placement_report)) argv.insert(argv.end(), {"-lv", "4"});
        }
        // tools/ternary_to_q4_0.py stamped it: HRX0 reads its decode projections at 2.125 bits per weight
        // instead of 4.5 (llama.cpp #54; the upload checks every block and refuses a non-ternary one).
        // A GGML_HRX_TERNARY_Q4_0 the user sets wins.
        // The Q4_0 copy stays resident for prompt batches, so the packed copy is extra GPU memory: about 30% of
        // the file (+4.0 GiB for Bonsai-2-27B, measured; only the decode kernels' projections are packed).
        if (device == "hrx" && ternary_q4_0(o.model) && !std::getenv("GGML_HRX_TERNARY_Q4_0")) {
            env.push_back("GGML_HRX_TERNARY_Q4_0=1");
            std::fprintf(stderr, "1bit serve: ternary Q4_0 file: HRX0 decodes from a 2-bit copy of its weights, "
                                 "about 30%% of the file size in extra GPU memory (GGML_HRX_TERNARY_Q4_0=0 turns it off)\n");
        }
        // HRX0 decodes through flash_attention_decode_split since its q8 pack race was fixed
        // (llama.cpp 00adc2b, #123/#140); it is faster once there is context. A
        // GGML_HRX_DISABLE_DISPATCH the user sets wins, and ONEBIT_HRX_DECODE_SPLIT=0 turns it off.
        const char* keep_split = std::getenv("ONEBIT_HRX_DECODE_SPLIT");
        if (device == "hrx" && keep_split && std::string(keep_split) == "0")
            env.push_back("GGML_HRX_DISABLE_DISPATCH=decode_split");
        // Qwen3.5 / Qwen3.8 (gated delta net) decode one sequence per batch on HRX0. Without
        // --parallel, llama-server's own default opens several slots, and a second concurrent
        // request then fails (HTTP 500) at the multi-sequence softplus / GATED_DELTA_NET: one slot.
        if (device == "hrx" && o.parallel == 0) {
            const std::string arch = gguf_architecture(o.model);
            if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next") argv.insert(argv.end(), {"-np", "1"});
        }
    } else if (device == "ds4") {
        // DwarfStar: its own GGUF layouts only; it opens its port after the model has loaded
        argv = {o.ds4.empty() ? default_ds4() : o.ds4, "-m", o.model,
                "--host", "127.0.0.1", "--port", std::to_string(child_port)};
        if (o.ctx_size > 0) { argv.push_back("--ctx"); argv.push_back(std::to_string(o.ctx_size)); }
        if (o.ssd_streaming) argv.push_back("--ssd-streaming");
        l.ready_path = "/v1/models";
    } else {
        refuse_removed_device(device);
        throw std::runtime_error("--device " + device + " cannot run a .gguf (hrx, cpu or ds4)");
    }
    if (o.parallel == 1) {
        // one slot: llama-server's own default is several, and a second slot can share a batch
        // (Qwen3.5 / Qwen3.8 on HRX0 cannot run a multi-sequence gated delta net yet)
        argv.insert(argv.end(), {"-np", "1"});
    }
    if (o.parallel > 1) {
        // continuous batching: N requests decode together, one read of the weights per step
        // for all of them; llama-server splits --ctx-size across the slots
        if (device == "ds4") throw std::runtime_error("--parallel works on the llama.cpp devices (hrx, cpu)");
        argv.insert(argv.end(), {"-np", std::to_string(o.parallel)});
        if (device == "hrx") {
            // HRX runs attention on the GPU only with one KV stream: -kvu shares one cache across
            // the slots (per-slot streams are 4-D and its flash attention falls back to the CPU:
            // Qwen3-4B, 4 slots, 39 -> 153 tok/s). AMD's Qwen attention path (qwen.attention) builds
            // its mask from positions and ignores the other sequences, so under -kvu the answers
            // bleed into each other; it is turned off for these servers.
            // G9: multi-sequence gated delta net - the HRX matchers hold kMaximumSequences 8 now, so these
            // architectures take --parallel like every other llama.cpp model
            argv.push_back("-kvu");
            bool merged = false;
            for (std::string& e : env)
                if (e.rfind("GGML_HRX_DISABLE_DISPATCH=", 0) == 0) { e += ",qwen.attention"; merged = true; }
            if (!merged) env.push_back("GGML_HRX_DISABLE_DISPATCH=qwen.attention");
        }
    }
    if (!o.mtp.empty() && !o.dflash.empty()) throw std::runtime_error("--mtp and --dflash are two drafters; pick one");
    const bool own_mtp = own_mtp_head(o, device);
    if (own_mtp)
        std::fprintf(stderr, "1bit serve: %s carries its own MTP head: drafting with it (--spec-type draft-mtp; "
                             "--no-mtp or ONEBIT_NO_MTP=1 turns it off)\n", o.model.c_str());
    if (!o.mtp.empty() || !o.dflash.empty() || own_mtp) {
        // the MTP head (--mtp) or a DFlash block-diffusion draft model (--dflash) drafts tokens
        // on the same device; the model checks them in one batch
        const char* flag = o.mtp.empty() ? "--dflash" : "--mtp";
        if (device == "ds4") throw std::runtime_error(std::string(flag) + " works on the llama.cpp devices (hrx, cpu)");
        l.mtp = true;
        if (!o.mtp.empty()) argv.insert(argv.end(), {"--spec-type", "draft-mtp", "-md", o.mtp, "-ngld", "99"});
        else if (own_mtp) argv.insert(argv.end(), {"--spec-type", "draft-mtp"});  // the head is in -m (own_mtp_head)
        else argv.insert(argv.end(), {"--spec-type", "draft-dflash", "-md", o.dflash, "-ngld", "99"});
        if (o.mtp_max > 0) { argv.push_back("--spec-draft-n-max"); argv.push_back(std::to_string(o.mtp_max)); }
        // DFlash drafts a whole block per step and only pays with full blocks; llama-server's
        // default draft length is 3. The longest draft is the drafter's block minus its anchor
        // token, and the draft length also sets how many recurrent states a hybrid model keeps
        // for rollback: asking for 16 against DFlash2's block of 8 made every prompt write 16
        // snapshots (Qwen3.8-27B-H32, 1,838-token prompt: 452-457 tok/s against 456-464).
        else if (!o.dflash.empty()) {
            const long long block = gguf_int(o.dflash, gguf_architecture(o.dflash) + ".block_size");
            argv.insert(argv.end(), {"--spec-draft-n-max", block > 1 ? std::to_string(block - 1) : "16"});
        }
        // without --mtp-p-min, recipe dflash-p-min-0 keeps DFlash blocks whole (config/recipes.json)
        if (!o.mtp_p_min.empty()) { argv.push_back("--spec-draft-p-min"); argv.push_back(o.mtp_p_min); }
    }
    if (!o.mmproj.empty()) {
        if (device == "ds4") throw std::runtime_error("--mmproj works on the llama.cpp devices (hrx, cpu)");
        // an image is decoded as one ubatch: models that attend to it bidirectionally (ZAYA1-VL,
        // Gemma 3) need all of it in one, and Qwen2.5-VL towers cap an image at 4096 tokens
        argv.insert(argv.end(), {"--mmproj", o.mmproj, "-b", "4096", "-ub", "4096"});
    }
    if ((device == "hrx" || device == "cpu") && !o.no_recipes &&
        std::filesystem::path(o.model).extension() == ".gguf") {
        // tuned settings for this model and route (config/recipes.json, docs/recipes.md); what
        // serve set above wins over a recipe
        Recipes recipes;
        std::string text = default_recipes_json(), err;
        if (!o.recipes.empty()) {
            std::ifstream in(o.recipes, std::ios::binary);
            if (!in) throw std::runtime_error("--recipes: cannot read " + o.recipes);
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        if (!recipes.parse(text, err))
            throw std::runtime_error((o.recipes.empty() ? std::string("built-in recipes") : o.recipes) + ": " + err);
        RecipeFacts facts;
        facts.device = device;
        facts.architecture = gguf_architecture(o.model);
        facts.moe = gguf_int(o.model, facts.architecture + ".expert_count") > 0;
        facts.drafter = !o.dflash.empty() ? "dflash" : (!o.mtp.empty() || own_mtp) ? "mtp" : "none";
        for (const auto& line : recipes.apply(facts, argv, env))
            std::fprintf(stderr, "1bit serve: recipe %s\n", line.c_str());
    }
    return l;
}

// A small llama-server for one RAG role: --embedding for /v1/embeddings, --reranking for
// /v1/rerank. Batch = ubatch: an embedding or rerank input is one pass. On HRX where the build has
// it (docs/hrx.md), else the CPU. HRX runs one sequence per batch (docs/hrx.md, "Several sequences
// per batch fail"), so its server takes one slot and 2048-token inputs, the largest prompt chunk
// its matmul kernels take in one pass.
std::string rag_device() {
#if defined(ONEBIT_HRX_SERVER)
    return "hrx";
#else
    return "cpu";
#endif
}

Launch rag_launch(const Options& o, const std::string& model, const char* role, int port) {
    Launch l;
    const std::string device = rag_device();
    l.device = device + " " + role;
    l.port = port;
    const std::string server = o.llama_server.empty() ? default_llama_server() : o.llama_server;
    if (device == "hrx") {
        l.argv = {server, "-m", model, "--host", "127.0.0.1", "--port", std::to_string(port),
                  "--device", "HRX0", "-ngl", "99", "-c", "2048", "-b", "2048", "-ub", "2048", "-np", "1",
                  std::string("--") + role};
        // the same ROCr the chat server gets, or the RAG server cannot open HRX0 ("invalid device: HRX0")
        if (const std::string hsa = hrx_libhsa(o.hrx_libhsa); !hsa.empty())
            l.env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
        for (const auto& e : hrx_cwsr_env(o)) l.env.push_back(e);  // engine#315 (app/hrx_cwsr.h)
        // strict placement, as the model's own backend (app/hrx_strict.h); no report, no -lv 4
        if (strict_for(o.strict, "hrx")) l.env.push_back("GGML_HRX_STRICT=1");
    } else
        l.argv = {server, "-m", model, "--host", "127.0.0.1", "--port", std::to_string(port),
                  "-ngl", "0", "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "4",
                  std::string("--") + role};
    return l;
}

// The devices a .gguf can run on, in the order the Laya router offers them. The NPU runs Q4NX
// model directories, not .gguf files (docs/npu.md), so it is not a candidate here; an NPU
// model directory routes to npu in run_serve. The GPU route is HRX (RFC #213 stage 3: no
// Vulkan, no ROCm; no ZINC since the core strip), with the CPU beside it; a build without HRX
// (CI, Windows) offers the CPU only. A candidate's backend starts only when the policy picks it.
std::vector<std::string> gguf_devices() {
    std::vector<std::string> devices;
#if defined(ONEBIT_HRX_SERVER) && !defined(_WIN32)
    devices.push_back("hrx");
#endif
    devices.push_back("cpu");
    return devices;
}

// --device auto for a .gguf: HRX where the build has it, else the CPU (Windows, and builds
// without HRX). What used to fall back to Vulkan (engine #271) has no GPU route left:
//  - an architecture our llama.cpp does not map (hrx_missing_arch): refused, on every device;
//  - --moe-slots: refused (launch_for) until expert streaming moves to HRX with Flash-Next;
//  - --parallel N > 1 on a gated delta-net model (no multi-sequence delta-net on HRX): HRX0 with
//    one slot and a warning (serve_child);
//  - --mmproj: HRX0. UNVERIFIED: HRX vision is an open RFC #213 gate, to be checked on ZAYA1-VL;
//    if it fails there, this goes back to the CPU.
// --mtp now runs on HRX0 (it drafts slower there than Vulkan did: verify batches of 2-4 tokens).
std::string auto_gguf_device(const Options& o) {
#if defined(_WIN32) || !defined(ONEBIT_HRX_SERVER)
    (void)o;
    return "cpu";  // no GPU route in a build without HRX (or on Windows)
#else
    (void)o;
    return "hrx";
#endif
}

// The text the Laya scorer routes on: a completion's prompt, or a chat request's messages
// joined the way the request reads.
// A conversation's identity for the Laya class cache: the messages up to and including the
// first user message (the system prompt and the opening request stay the same on every turn),
// or the whole prompt for /v1/completions.
size_t conversation_key(const httplib::Request& req) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        return std::hash<std::string>{}(req.body);
    }
    std::string head;
    if (body.contains("messages") && body["messages"].is_array()) {
        for (const auto& m : body["messages"]) {
            if (!m.is_object()) continue;
            const std::string role = m.contains("role") && m["role"].is_string() ? m["role"].get<std::string>() : "";
            head += role + ":" + (m.contains("content") ? m["content"].dump() : "") + "\n";
            if (role == "user") break;
        }
    } else if (body.contains("prompt")) {
        head = body["prompt"].dump();
    }
    // model and role come from the client: a null or non-string one must not throw
    const std::string model = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : "";
    return std::hash<std::string>{}(model + "\n" + head);
}

#ifdef ONEBIT_LAYA
// The Laya checkpoint for --laya-model DIR, or for --laya: $ONEBIT_LAYA_MODEL, else
// ${XDG_DATA_HOME:-~/.local/share}/1bit/laya (where scripts/fetch-laya.sh installs it). A
// directory holding the pinned repo's typed-decisions/ checkpoint uses that one: it classifies
// requests best (95.5% against 92.0% for the root checkpoint, tests/laya_classify_eval).
std::string laya_home(const std::string& given) {
    std::string dir = given;
    if (dir.empty()) {
        if (const char* e = std::getenv("ONEBIT_LAYA_MODEL"); e && *e) dir = e;
        else if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) dir = std::string(x) + "/1bit/laya";
        else if (const char* h = std::getenv("HOME"); h && *h) dir = std::string(h) + "/.local/share/1bit/laya";
    }
    return dir;
}

std::string laya_checkpoint(const std::string& given) {
    namespace fs = std::filesystem;
    const std::string dir = laya_home(given);
    if (dir.empty() || !fs::is_directory(dir))
        throw std::runtime_error("laya: no checkpoint at " + (dir.empty() ? std::string("~/.local/share/1bit/laya") : dir) +
                                 " (install it with scripts/fetch-laya.sh, or pass --laya-model DIR)");
    if (fs::exists(fs::path(dir) / "typed-decisions" / "model.safetensors")) return (fs::path(dir) / "typed-decisions").string();
    return dir;
}

// The router's fast scorer: ggmlc's laya (our fork, third_party/ggmlc, -DONEBIT_LAYA_GGML=ON) on
// HRX0, AMD's backend, built against third_party/llama.cpp's ggml (its HRX kernels for ModernBERT,
// fork #31), with the typed-decisions checkpoint compiled to a GGUF (mys/laya-typed-decisions-GGUF,
// pinned in config/laya.json, fetched by scripts/fetch-laya.sh into <laya dir>/gguf/). Q8_0
// classifies tests/laya_route_cases.json as well as the C++ scorer on the safetensors (95.5%) in
// 25 ms a decision (15 at a steady 64 tokens) instead of about 540 ms. Used when the binary and the GGUF are both there.
constexpr const char* kLayaGguf = "laya_typed_decisions_q8_0.gguf";

std::string laya_ggml_bin() {
    if (const char* e = std::getenv("ONEBIT_LAYA_GGML"); e && *e) return e;
#ifdef ONEBIT_LAYA_GGML_BIN
    if (std::filesystem::exists(built_path(ONEBIT_LAYA_GGML_BIN))) return built_path(ONEBIT_LAYA_GGML_BIN);
#endif
    return "";
}

// --laya-model FILE.gguf, or the pinned GGUF under the Laya directory ("" when there is none)
std::string laya_gguf(const std::string& given) {
    namespace fs = std::filesystem;
    if (given.size() > 5 && given.compare(given.size() - 5, 5, ".gguf") == 0) {
        if (!fs::exists(given)) throw std::runtime_error("laya: no such file " + given);
        return given;
    }
    const std::string dir = laya_home(given);
    for (const fs::path p : {fs::path(dir) / "gguf" / kLayaGguf, fs::path(dir) / kLayaGguf})
        if (fs::exists(p)) return p.string();
    return "";
}

#if defined(__linux__)
// `laya daemon`: one JSON request per line on its stdin, one answer per line on its stdout (its
// log goes to our stderr). Pipes rather than its `serve`, which listens on every interface.
class LayaDaemon {
public:
    // env: NAME=VALUE entries set in the daemon's environment (one the user already set wins)
    explicit LayaDaemon(const std::vector<std::string>& argv, const std::vector<std::string>& env = {}) {
        int in[2], out[2];
        if (::pipe2(in, O_CLOEXEC) != 0 || ::pipe2(out, O_CLOEXEC) != 0)
            throw std::runtime_error(std::string("laya: pipe: ") + std::strerror(errno));
        std::signal(SIGPIPE, SIG_IGN);  // a daemon that died fails a write, not the server
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        const pid_t parent = ::getpid();
        pid_ = ::fork();
        if (pid_ < 0) throw std::runtime_error(std::string("laya: fork: ") + std::strerror(errno));
        if (pid_ == 0) {
            ::prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (::getppid() != parent) ::_exit(127);
            for (const auto& e : env) {
                const size_t eq = e.find('=');
                if (eq != std::string::npos) ::setenv(e.substr(0, eq).c_str(), e.c_str() + eq + 1, 0);
            }
            ::dup2(in[0], 0);
            ::dup2(out[1], 1);
            ::execvp(args[0], args.data());
            ::_exit(127);
        }
        ::close(in[0]);
        ::close(out[1]);
        to_ = in[1];
        from_ = out[0];
        std::string line;
        json ready;
        if (!read_line(line, std::chrono::seconds(300)) || !(ready = json::parse(line, nullptr, false)).is_object() ||
            ready.value("status", "") != "ready")
            throw std::runtime_error("laya: " + argv[0] + " did not start: " + line);
    }
    ~LayaDaemon() {
        if (to_ >= 0) ::close(to_);
        if (from_ >= 0) ::close(from_);
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            ::waitpid(pid_, nullptr, 0);
        }
    }
    LayaDaemon(const LayaDaemon&) = delete;
    LayaDaemon& operator=(const LayaDaemon&) = delete;

    // one request at a time; false when the daemon is gone or took longer than 30 s
    bool ask(const std::string& request, json& answer) {
        std::lock_guard<std::mutex> lock(mu_);
        const std::string line = request + "\n";
        for (size_t done = 0; done < line.size();) {
            const ssize_t n = ::write(to_, line.data() + done, line.size() - done);
            if (n <= 0) return false;
            done += size_t(n);
        }
        std::string reply;
        if (!read_line(reply, std::chrono::seconds(30))) return false;
        answer = json::parse(reply, nullptr, false);
        return answer.is_object();
    }

private:
    bool read_line(std::string& line, std::chrono::milliseconds timeout) {
        const auto end = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            if (const size_t nl = buf_.find('\n'); nl != std::string::npos) {
                line = buf_.substr(0, nl);
                buf_.erase(0, nl + 1);
                return true;
            }
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now());
            if (left.count() <= 0) return false;
            pollfd p{from_, POLLIN, 0};
            if (::poll(&p, 1, int(left.count())) <= 0) return false;
            char tmp[4096];
            const ssize_t n = ::read(from_, tmp, sizeof tmp);
            if (n <= 0) return false;
            buf_.append(tmp, size_t(n));
        }
    }

    pid_t pid_ = -1;
    int to_ = -1, from_ = -1;
    std::string buf_;
    std::mutex mu_;
};

// classify_request (laya/route.cpp) through the daemon: the same size gate for long_doc and the
// same question, asked of the GGUF
onebit::laya::RequestClass classify_request_gguf(LayaDaemon& d, const std::string& state) {
    onebit::laya::RequestClass out;
    if (state.size() >= onebit::laya::kLongDocChars) {
        out.label = "long_doc";
        out.confidence = 1.0f;
        out.probabilities = {{"long_doc", 1.0f}};
        return out;
    }
    int variant = 0;
    if (const char* v = std::getenv("ONEBIT_LAYA_CLASS_VARIANT")) variant = std::atoi(v);
    const onebit::laya::Question q = onebit::laya::request_class_question(variant);
    nlohmann::ordered_json criteria = nlohmann::ordered_json::object();
    for (const auto& [key, desc] : q.criteria)
        if (key != "long_doc") criteria[key] = desc;
    nlohmann::ordered_json req = {{"state", state},
                                  {"questions", {{"cls", {{"type", q.type}, {"instructions", q.instructions}, {"criteria", criteria}}}}}};
    json answer;
    if (!d.ask(req.dump(), answer)) return out;
    const json* a = answer.contains("answers") && answer["answers"].contains("cls") ? &answer["answers"]["cls"] : nullptr;
    if (!a || !a->contains("choice") || !(*a)["choice"].is_string()) return out;
    out.label = (*a)["choice"].get<std::string>();
    out.confidence = a->value("confidence", 0.0f);
    if (a->contains("probabilities") && (*a)["probabilities"].is_object())
        for (const auto& [key, desc] : q.criteria)
            if ((*a)["probabilities"].contains(key)) out.probabilities.emplace_back(key, (*a)["probabilities"][key].get<float>());
    return out;
}
#endif
#endif

// With head_only, a chat request contributes only what conversation_key() covers (the messages
// through the first user message): the class cached under a key must not depend on anything
// the key leaves out, or one client's later turns could decide another client's routing.
std::string request_state(const httplib::Request& req, bool head_only = false) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        return "";
    }
    if (body.contains("prompt") && body["prompt"].is_string()) return body["prompt"].get<std::string>();
    std::string out;
    if (body.contains("messages") && body["messages"].is_array())
        for (const auto& m : body["messages"]) {
            if (!m.is_object()) continue;
            const bool user = m.contains("role") && m["role"].is_string() && m["role"].get<std::string>() == "user";
            if (!m.contains("content")) {
                if (head_only && user) break;
                continue;
            }
            auto add = [&](const std::string& t) {
                if (t.empty()) return;
                if (!out.empty()) out += "\n";
                out += t;
            };
            const auto& c = m["content"];
            if (c.is_string()) add(c.get<std::string>());
            else if (c.is_array())
                for (const auto& part : c)
                    if (part.is_object() && part.value("type", "") == "text") add(part.value("text", ""));
            if (head_only && user) break;
        }
    return out;
}

// --print-backend (docs/serve.md, "One hop from Lemonade"): a route that needs serve in the
// request path. Exit 5 tells Lemonade's onebit recipe to run `1bit serve` as before.
constexpr int kServeOnly = 5;
int serve_only(const std::string& why) {
    std::fprintf(stderr, "1bit serve: --print-backend: %s; run 1bit serve itself\n", why.c_str());
    return kServeOnly;
}

// The warm-up requests (warm_backend, below): HRX only, and ONEBIT_NO_WARMUP=1 turns them off. serve sends
// them itself; --print-backend hands the same requests to Lemonade as the plan's "warmups" ([{path, body}],
// in order; "warmup" is the first one, for a lemond that reads only that), and Lemonade's onebit recipe
// sends them once the backend is ready, before it marks the model loaded.
static bool warmup_wanted(const std::string& device) {
    const std::string dev = device.substr(0, device.find(' '));
    return getenv("ONEBIT_NO_WARMUP") == nullptr && (dev == "hrx" || dev == "hybrid");
}

static const char* const kWarmupPath = "/v1/chat/completions";

static json warmup_body() {
    std::string text = "warm up the decode and prefill shapes";
    for (int i = 0; i < 60; ++i) text += " lorem ipsum dolor sit amet";
    return json{ { "messages", json::array({ { { "role", "user" }, { "content", text } } }) },
                 { "temperature", 0 }, { "max_tokens", 8 }, { "cache_prompt", false } };
}

// The second warm-up request: one 16-token prompt, " a" 16 times (one token each in the byte-level BPE
// vocabularies of the models we serve; a BOS makes it 17, still inside the range below). n_predict 0: the
// prompt is the work. (Token ids instead of text ended in a 500 on Qwen3.5-4B: id 100 is a lone UTF-8
// byte there, which llama-server's output parser rejects.)
// The first request whose batch holds 13 to 20 tokens takes HRX's small-batch matmul kernels, and they
// read the weights in a layout ggml-hrx builds on the host the first time it needs it (re-quantizing the
// affected tensors: 6.1 s of CPU, +1.0 GiB of GTT on Qwen3.8-27B UD-Q4_K_XL). A chat request splits its
// prompt at the last user message and before its last 4 tokens, so short chats land there: Lemonade's
// chat-short prompt (67 tokens) runs as 50 + 13 + 4, and its first time after a load cost 2.43 s against
// 0.53 s later, with lemond on 4 cores. This request moves that build to load time (2.1 s on 4 cores).
// On a fresh server, 13-20 tokens cost 2.03-2.07 s the first time; 1-12 cost 0.1-0.5 s and 21-512 cost
// 0.45-1.23 s (docs/serve.md, "Warm-up").
static constexpr int kWarmupShortTokens = 16;

static json warmup_short_body() {
    std::string prompt;
    for (int i = 0; i < kWarmupShortTokens; ++i) prompt += " a";
    return json{ { "prompt", prompt }, { "n_predict", 0 },
                 { "temperature", 0 }, { "cache_prompt", false } };
}

static json warmup_requests() {
    return json::array({ json{ { "path", kWarmupPath }, { "body", warmup_body() } },
                         json{ { "path", "/completion" }, { "body", warmup_short_body() } } });
}

// --print-backend: the llama-server serve would start for this model, as one JSON line on stdout,
// after every check serve makes before a load (architecture, memory, recipes). Lemonade starts it
// on the port it chose (--port) and talks to it directly: one HTTP hop and two processes instead of
// two and three. What serve added on the request path is the backend's own there: /health answers
// 503 while the main loop is stalled with work queued (llama.cpp fork #111), so Lemonade's
// BackendWatchdog resets a hung GPU; strict placement (the HRX default) is GGML_HRX_STRICT in the backend; replies carry
// Lemonade's name through --alias. env holds only what the caller's environment does not set
// already, so a value the user set still wins, as it does under serve.
int print_backend(const Options& o, const Launch& l) {
    const bool llama = l.device.rfind("hrx", 0) == 0 || l.device.rfind("cpu", 0) == 0;
    if (!llama) return serve_only("--device " + l.device + " is not a llama-server");
    std::vector<std::string> argv = l.argv;
    auto has = [&argv](const char* flag) { return std::find(argv.begin(), argv.end(), flag) != argv.end(); };
    if (!o.alias.empty() && !has("--alias")) argv.insert(argv.end(), {"--alias", o.alias});
    // serve's proxy reused nothing between requests; the backend can shift a cached prefix into
    // place when a conversation's history changes in the middle (chunks of at least 256 tokens).
    // A recipe's own --cache-reuse wins; recurrent models ignore it (llama.cpp checks can_shift).
    if (o.role.empty() && !has("--cache-reuse")) argv.insert(argv.end(), {"--cache-reuse", "256"});
    json env = json::object();
    for (const std::string& e : l.env) {
        const auto eq = e.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = e.substr(0, eq);
        if (!std::getenv(key.c_str())) env[key] = e.substr(eq + 1);
    }
    if (o.print_backend_env) {
        // --print-backend-env: the same environment as NAME=VALUE lines, for a caller that starts the
        // backend with an argv of its own (lemonade-hrx: stock Lemonade's HRX recipe, docs/lemonade.md)
        for (const auto& [key, value] : env.items()) std::printf("%s=%s\n", key.c_str(), value.get<std::string>().c_str());
        return 0;
    }
    json out = json::object();
    out["device"] = l.device;
    out["port"] = l.port;
    out["argv"] = argv;
    out["env"] = env;
    // a chat backend on HRX: the request Lemonade sends before it marks the model loaded (warmup_wanted)
    if (o.role.empty() && warmup_wanted(l.device)) {
        out["warmups"] = warmup_requests();
        out["warmup"]  = out["warmups"][0];
    }
    std::printf("%s\n", out.dump().c_str());
    return 0;
}

// The backend's first request after a load pays a one-time cost that is not its steady state. Measured on
// Qwen3.8-27B (engine main, three fresh loads): request 1 decoded 5 % slower than requests 2-3 (11.72-11.75
// against 12.41-12.46 tok/s) and prefilled 20 % slower (869-945 against 714-733 ms); on Qwen3-4B with a cold
// model file request 1 decoded 23 % slower (54.67 against 71-72 tok/s). It is not llama.cpp's warmup
// (--no-warmup changes nothing), not the GPU ramping (9 s of GPU load before the first request changes
// nothing) and not prompt caching (measured with cache_prompt off): it is HRX's per-shape command-program
// build and the first touch of that shape's buffers, paid by whichever request uses the shape first.
// One synthetic request here moves that cost to load time: with it, the first measured request on the 27B
// decoded 12.47 tok/s, the same as the later ones. Only the decode shape and this prompt length are covered;
// a client prompt of another length still builds its own prefill program once (~130-200 ms on the 27B).
// HRX only, and ONEBIT_NO_WARMUP=1 turns it off.
static void warm_backend(int port, const std::string& device) {
    if (!warmup_wanted(device)) {
        return;
    }
    httplib::Client c("127.0.0.1", port);
    c.set_connection_timeout(5);
    c.set_read_timeout(600);
    for (const auto & w : warmup_requests()) {
        const std::string path = w["path"].get<std::string>();
        const auto t0 = std::chrono::steady_clock::now();
        const auto r  = c.Post(path, w["body"].dump(), "application/json");
        const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        if (!r || r->status != 200) {
            std::fprintf(stderr, "1bit serve: warm-up request %s on port %d failed (%d)\n", path.c_str(), port,
                         r ? r->status : -1);
        } else {
            std::fprintf(stderr, "1bit serve: warm-up request %s on port %d took %ld ms\n", path.c_str(), port, ms);
        }
    }
}

int serve_child(const Options& given) {
    Options o = given;
    if (hadamard_q4_0(o.model))
        throw std::runtime_error(o.model + " is an H32 file (onebit.hadamard_q4_0, a Hadamard-rotated Q4_0): the engine "
                                 "dropped that format; use the model's UD-Q4_K_XL (or another standard) GGUF instead");
    if (o.device == "auto" && o.parallel > 1 && delta_net_arch(gguf_architecture(o.model)) &&
        auto_gguf_device(o) == "hrx") {
        // HRX0 decodes one gated delta-net sequence at a time (no multi-sequence delta-net kernel yet),
        // and auto has no other GPU route since the Vulkan build left: one slot on HRX0, said aloud.
        // --device cpu still honors --parallel.
        std::fprintf(stderr, "1bit serve: --parallel %d on %s: HRX0 runs one gated delta-net sequence at a time "
                             "(no multi-sequence delta-net on HRX yet), so --device auto serves it with one slot; "
                             "--device cpu keeps --parallel %d\n",
                     o.parallel, gguf_architecture(o.model).c_str(), o.parallel);
        o.parallel = 1;
    }
    if (prism_ternary_types(o.model) && !prism_hadamard(o.model)) {
        // PQ2_0 / PTQ1_0 without a rotation (e.g. Ternary-Bonsai-1.7B): only the HRX build reads the types
        if (o.device == "auto") o.device = "hrx";
        if (o.device != "hrx")
            throw std::runtime_error(o.model + " stores PrismML's ternary types: it runs on --device hrx, or convert it "
                                     "with tools/ternary_to_q4_0.py (the same weights as Q4_0), docs/hrx.md");
        if (o.laya_auto || !o.laya_model.empty())
            throw std::runtime_error("--laya routes among devices; a file in PrismML's ternary types runs on hrx only");
        std::fprintf(stderr, "1bit serve: %s stores PrismML's ternary types: HRX route\n", o.model.c_str());
    }
    if (prism_hadamard(o.model)) {
        // a rotated file has one route: our llama.cpp on HRX, which rotates the activations
        if (o.device == "auto") o.device = "hrx";
        if (o.device != "hrx")
            throw std::runtime_error(o.model + " is Hadamard-folded (prism.hadamard): it runs on --device hrx only, docs/hrx.md");
        if (o.laya_auto || !o.laya_model.empty())
            throw std::runtime_error("--laya routes among devices; a Hadamard-folded file runs on hrx only");
        std::fprintf(stderr, "1bit serve: %s is Hadamard-folded (prism.hadamard): HRX route\n", o.model.c_str());
    }
    if (onebp_file(o.model)) {
        // DwarfStar (our fork) is the engine's 1BP reader
        if (o.device == "auto") o.device = "ds4";
        if (o.device != "ds4")
            throw std::runtime_error(o.model + " is a 1BP package: it runs on --device ds4 (docs/dwarfstar.md)");
        if (o.laya_auto || !o.laya_model.empty())
            throw std::runtime_error("--laya routes among devices; a 1BP package runs on ds4 only");
    }
    if (o.print_backend) {
        // --print-backend (docs/serve.md, "One hop from Lemonade"): only a lone llama-server can be
        // started by Lemonade itself; these need serve in the request path
        if (o.device == "ds4") return serve_only("--device ds4 is not a llama-server");
        if (o.device == "hybrid") return serve_only("--device hybrid hands each prompt's KV to llama-server in serve");
        if (o.pm) return serve_only("--pm runs its tool loop in serve");
        if (o.laya_auto || !o.laya_model.empty()) return serve_only("--laya routes each request in serve");
        if (!o.embed.empty() || !o.rerank.empty()) return serve_only("--embed and --rerank run more servers behind serve");
    }
    // --pm (docs/pm.md): the model is the Project Manager. Its chat completions run the tool loop
    // in app/pm.cpp against the one backend; the experts are Lemonade's models, reached by URL.
    pm::Config pm_cfg;
    // --pm-router (docs/pm.md, "Routing with Laya"): "laya" asks the engine's Laya router which
    // expert a request fits, before the PM's first round. The default, and any unset value, stays
    // the PM's own choice, so --pm alone behaves exactly as before.
    const bool pm_laya = o.pm && o.pm_router == "laya";
    if (o.pm) {
        if (o.laya_auto || (!o.laya_model.empty() && !pm_laya))
            throw std::runtime_error("--pm does not combine with --laya: the Project Manager routes among models, Laya among devices");
        if (!o.role.empty()) throw std::runtime_error("--pm does not combine with --embedding/--reranking");
        pm_cfg = pm::load_config(o.pm_experts, o.lemonade_url);
        pm_cfg.router = pm::parse_router(o.pm_router);
        pm_cfg.spawn_url = o.pm_spawn_url;
    } else if (!o.pm_router.empty()) {
        throw std::runtime_error("--pm-router needs --pm");
    }
    std::vector<Launch> backends;
    // --laya / --laya-model with --device auto (RFC #186): Laya classifies each conversation
    // (code, prose, short, long_doc; laya/route.h) and the route policy (config/route-policy.json
    // or --route-policy) maps the class to one of the devices this build can run the .gguf on.
    // A device's backend starts when it is first picked (one model is not loaded on every
    // device at once). Opt-in: the built-in policy sends every class to HRX (the engine's one GPU
    // route); a row changes only with a measured net gain.
    const bool laya = (!o.laya_model.empty() || o.laya_auto) && o.device == "auto" && !o.pm;
#ifdef ONEBIT_LAYA
    std::unique_ptr<onebit::laya::Scorer> scorer;
#if defined(__linux__)
    std::unique_ptr<LayaDaemon> laya_daemon;
#endif
    std::vector<std::string> laya_devices;
    onebit::laya::RoutePolicy policy;
    // --pm-router laya shares the scorer with the device router: same checkpoint, same
    // classify_request (laya/route.cpp), one instance.
    if (laya || pm_laya) {
        if (laya && !o.role.empty())
            throw std::runtime_error("--laya does not combine with --embedding/--reranking");
        if (laya) policy = onebit::laya::load_route_policy(o.route_policy);
        // ONEBIT_LAYA_DEVICE=npu: the C++ scorer with its encoder on the NPU, from a private add-on
        // (laya/encoder.h, docs/laya.md); else the GGUF scorer, on HRX0 unless it names another
        const char* dev = std::getenv("ONEBIT_LAYA_DEVICE");
        const bool laya_npu = dev && std::string(dev) == "npu";
        // the route policy is the device router's; --pm-router laya only uses the classifier
        const std::string policy_note = laya ? (o.route_policy.empty() ? ", policy built in" : ", policy " + o.route_policy) : "";
        const std::string gguf = laya_npu ? "" : laya_gguf(o.laya_model), bin = laya_npu ? "" : laya_ggml_bin();
#if defined(__linux__)
        if (!gguf.empty() && !bin.empty()) {
            // the GGUF scorer, with the HSA runtime HRX needs (docs/hrx.md)
            std::vector<std::string> env;
            if (const std::string hsa = hrx_libhsa(o.hrx_libhsa); !hsa.empty()) env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
            if (!dev || !*dev || std::string(dev) == "hrx")
                for (const auto& e : hrx_cwsr_env(o)) env.push_back(e);  // engine#315 (app/hrx_cwsr.h)
            laya_daemon = std::make_unique<LayaDaemon>(std::vector<std::string>{
                bin, "daemon", gguf, "--family", "typed-decisions", "--device", dev && *dev ? dev : "hrx"}, env);
            std::fprintf(stderr, "1bit serve: laya: %s through %s%s\n", gguf.c_str(), bin.c_str(), policy_note.c_str());
        }
#endif
        if (!gguf.empty() && gguf == o.laya_model && bin.empty())
            throw std::runtime_error("laya: a GGUF needs ggmlc's laya (build with -DONEBIT_LAYA_GGML=ON, or set ONEBIT_LAYA_GGML)");
#if defined(__linux__)
        if (!laya_daemon)
#endif
        {
            const onebit::laya::EncoderFactory* f = laya_npu ? onebit::laya::find_encoder("npu") : nullptr;
            if (laya_npu && !f)
                throw std::runtime_error("laya: ONEBIT_LAYA_DEVICE=npu needs a build with the NPU add-on "
                                         "(-DONEBIT_NPU_PRIVATE=<npu-kernels checkout>, docs/laya.md)");
            const std::string ckpt = laya_checkpoint(o.laya_model);
            scorer = std::make_unique<onebit::laya::Scorer>();
            if (!scorer->load(ckpt)) throw std::runtime_error("laya: " + scorer->error());
            if (f) {
                std::string err;
                onebit::laya::EncoderFn fn = (*f)(ckpt, &err);
                if (!fn) throw std::runtime_error("laya: the NPU encoder: " + err);
                scorer->set_encoder(std::move(fn));
            }
            std::fprintf(stderr, "1bit serve: laya: checkpoint %s%s%s\n", ckpt.c_str(),
                         laya_npu ? " (encoder on the NPU)" : "", policy_note.c_str());
        }
        if (laya) {
            for (const std::string& dev : gguf_devices()) {
                try {
                    backends.push_back(launch_for(o, dev, free_port()));
                    laya_devices.push_back(dev);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "1bit serve: laya: %s is not a candidate: %s\n", dev.c_str(), e.what());
                }
            }
            if (backends.empty()) throw std::runtime_error("laya: no device in this build can run " + o.model);
        }
        if (pm_laya) {
            // The PM's router: one Laya decision per request, timed for the reply's `pm.routing`.
            // The GGUF scorer when serve has one (25 ms a decision), else the C++ scorer on the
            // safetensors (a few hundred); both answered by the same classify_request.
#if defined(__linux__)
            if (laya_daemon) {
                LayaDaemon* d = laya_daemon.get();
                pm_cfg.classify = [d](const std::string& state, pm::RouteHint& out) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const onebit::laya::RequestClass c = classify_request_gguf(*d, state);
                    out.ms = long(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
                    out.label = c.label;
                    out.confidence = c.confidence;
                    return !c.label.empty();
                };
            }
#endif
            if (!pm_cfg.classify) {
                onebit::laya::Scorer* s = scorer.get();
                pm_cfg.classify = [s](const std::string& state, pm::RouteHint& out) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const onebit::laya::RequestClass c = onebit::laya::classify_request(*s, state);
                    out.ms = long(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
                    out.label = c.label;
                    out.confidence = c.confidence;
                    return !c.label.empty();
                };
            }
        }
    }
#else
    if (pm_laya) throw std::runtime_error("--pm-router laya is not part of this build (-DONEBIT_LAYA=ON, docs/laya.md)");
    if (laya) throw std::runtime_error("--laya-model is not part of this build (-DONEBIT_LAYA=ON, docs/laya.md)");
#endif
    if (laya) {
        // the candidates are prepared above
    } else if (!o.role.empty()) {
        // The model is an embedding or reranking model (Lemonade loads them as models of their
        // own): llama-server on the device asked for, in that role. An input is one pass, so
        // the batch covers the context.
        refuse_removed_device(o.device);
        const std::string dev = o.device == "auto" ? rag_device() : o.device;
        if (dev != "hrx" && dev != "cpu")
            throw std::runtime_error("--" + o.role + " runs on llama-server devices (hrx, cpu), not " + dev);
        if (!o.mtp.empty() || !o.dflash.empty())
            throw std::runtime_error("--" + o.role + " does not combine with --mtp or --dflash");
        Launch l = launch_for(o, dev, o.print_backend ? o.port : free_port());
        const std::string batch = std::to_string(o.ctx_size > 0 ? o.ctx_size : 8192);
        for (const std::string& a : {std::string("--") + o.role, std::string("-b"), batch, std::string("-ub"), batch})
            l.argv.push_back(a);
        l.device = dev + " " + o.role;
        backends.push_back(std::move(l));
    } else {
        backends.push_back(launch_for(o, o.device == "auto" ? auto_gguf_device(o) : o.device,
                                      o.print_backend ? o.port : free_port()));
    }
    const std::string device = laya ? "laya" : backends[0].device;
    const bool drop_model = backends[0].drop_model;
    const std::string set_model = backends[0].set_model;

    // --device hybrid (app/serve_hybrid.h, docs/hybrid.md): the prefill provider, when named; without one
    // the backend prefills every prompt itself (said once), so the device still serves.
    std::unique_ptr<hybrid::Prefill> hybrid_prefill;
    struct SlotDirGuard {
        std::string dir;
        ~SlotDirGuard() { if (!dir.empty()) { std::error_code ec; fs::remove_all(dir, ec); } }
    } slot_dir_guard{backends[0].hybrid ? backends[0].slot_dir : ""};
    if (backends[0].hybrid) {
        if (o.pm) throw std::runtime_error("--device hybrid does not combine with --pm (the tool loop forwards its own requests)");
        std::string provider = o.prefill_provider;
        if (provider.empty())
            if (const char* e = std::getenv("ONEBIT_HYBRID_PREFILL"); e && *e) provider = e;
        std::string elfs;
        if (const char* e = std::getenv("ONEBIT_HYBRID_ELFS"); e && *e) elfs = e;
        if (provider.empty()) {
            std::fprintf(stderr, "1bit serve: hybrid: no prefill provider (--prefill-provider or ONEBIT_HYBRID_PREFILL): "
                                 "the %s backend prefills every prompt itself (docs/hybrid.md)\n", auto_gguf_device(o).c_str());
        } else {
            hybrid_prefill = std::make_unique<hybrid::Prefill>(provider, elfs, backends[0].slot_dir, o.model);
            std::fprintf(stderr, "1bit serve: hybrid: prefill by %s%s%s, decode on %s\n", provider.c_str(),
                         elfs.empty() ? "" : " --elfs ", elfs.c_str(), auto_gguf_device(o).c_str());
        }
    }

    std::vector<Launch> rag;   // [0] embedding, then rerank, if given
    if (!o.embed.empty()) rag.push_back(rag_launch(o, o.embed, "embedding", free_port()));
    if (!o.rerank.empty()) rag.push_back(rag_launch(o, o.rerank, "reranking", free_port()));
    auto stem = [](const std::string& m) { return fs::path(m).stem().string(); };

    const std::string id = model_id(o);

    // Memory admission (app/mem_admit.h): before any llama-server starts, refuse a load the box
    // has no room for, in one line with the numbers, so Lemonade shows why instead of an
    // allocation failure or an OOM minutes later. laya counts its first device.
    if (o.mem_check || o.mem_check_only) {
        admit::Need need;
        std::vector<const Launch*> launches;
        if (!backends.empty()) launches.push_back(&backends[0]);
        for (const auto& b : rag) launches.push_back(&b);
        for (const Launch* l : launches)
            if (l->device.rfind("hrx", 0) == 0 || l->device.rfind("cpu", 0) == 0) admit::add_llama_server(need, l->argv);
        const admit::Verdict v = admit::judge(need, sysinfo::memory());
        if (!need.parts.empty() && v.checked) {
            auto gib = [](uint64_t b) { char t[32]; std::snprintf(t, sizeof t, "%.2f", double(b) / double(admit::kGiB)); return std::string(t); };
            std::string parts;
            for (const auto& p : need.parts) parts += (parts.empty() ? "" : "; ") + p;
            const std::string ctx = need.ctx_from_model ? "the model's own ctx " + std::to_string(need.n_ctx) + " (no --ctx-size)"
                                                        : "ctx " + std::to_string(need.n_ctx);
            const std::string line = "need " + gib(need.total) + " GiB (" + parts + ") at " + ctx + ", free " + gib(v.free) +
                                     " GiB (" + v.box + ")";
            if (o.mem_check_only) {
                std::printf("%s: %s: %s\n", id.c_str(), v.fits ? "fits" : "does not fit", line.c_str());
                for (const auto& w : v.warnings) std::printf("%s: warning: %s\n", id.c_str(), w.c_str());
                return v.fits ? 0 : 3;
            }
            if (!v.fits) {
                // exit 3, the same as --mem-check's "does not fit": Lemonade's loader logs the exit
                // code and fails the load at once ("1bit serve failed to start"), with this line in
                // its log
                std::fprintf(stderr, "1bit serve: %s does not fit: %s; free memory, lower --ctx-size, or pass --no-mem-check\n",
                             id.c_str(), line.c_str());
                return 3;
            }
            std::fprintf(stderr, "1bit serve: memory: %s\n", line.c_str());
            for (const auto& w : v.warnings) std::fprintf(stderr, "1bit serve: memory: warning: %s\n", w.c_str());
        } else if (o.mem_check_only) {
            std::printf("%s: no estimate (%s)\n", id.c_str(), need.parts.empty() ? "not a llama-server model" : "the box reports no memory");
            return 0;
        }
    }
    if (o.print_backend) return print_backend(o, backends[0]);
    std::atomic<bool> ready{false};
    // the placement line is reprinted whenever a graph reports more nodes off HRX than every graph
    // before it: the worst graph is the one worth acting on, and it is not always the first (the
    // warmup graph is scheduled before any request).
    std::atomic<long> placement_worst{-1};
    ProgressWatch watch(o.watchdog);
    ServeMetrics metrics(device);   // GET /metrics (app/serve_metrics.h)
    Lifecycle lifecycle(id);
    std::mutex wedged_mu;
    std::string wedged;   // set by the watchdog: why the backend is taken down
    auto backend_line = [&metrics, &placement_worst, &wedged_mu, &wedged](const std::string& line) {
        metrics.backend_line(line);
        if (device_lost_line(line)) {
            std::lock_guard<std::mutex> lock(wedged_mu);
            if (wedged.empty()) wedged = "the backend lost its GPU: " + line.substr(0, 200);
        }
        const auto report = metrics.placement();
        // engine #390: the backend's placement report becomes serve's own line at load.
        if (!report) return;
        const long off = report->cpu_fallback_total.value_or(0);
        if (off > placement_worst.load() && placement_worst.exchange(off) < off)
            std::fprintf(stderr, "1bit serve: HRX placement: %s\n", placement_summary(*report).c_str());
    };

    httplib::Server srv;
    install_request_guard(srv, o.host);  // Host / Origin (app/http_guard.h)
    std::vector<std::unique_ptr<Child>> children;
    std::mutex start_mu;
    auto all_alive = [&] {
        std::lock_guard<std::mutex> lock(start_mu);
        for (auto& c : children) if (!c->alive()) return false;
        return true;
    };
    // /health: 503 while loading; once loaded, 503 too when a backend has exited, does not answer
    // its own health route within 1.5 s, or the watchdog found it wedged. Lemonade's BackendWatchdog
    // probes this route and reloads a backend that fails it.
    auto health = [&](const httplib::Request&, httplib::Response& res) {
        auto fail = [&](const std::string& why) {
            res.status = 503;
            res.set_content(json{{"status", "error"}, {"error", why}}.dump(), "application/json");
        };
        if (!ready) {
            res.status = 503;
            res.set_content(R"({"status":"loading"})", "application/json");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(wedged_mu);
            if (!wedged.empty()) return fail("backend wedged: " + wedged);
        }
        std::vector<std::pair<int, std::string>> probes;  // port, health route
        {
            // a laya backend starting holds start_mu for as long as it loads: do not wait on it
            std::unique_lock<std::mutex> lock(start_mu, std::try_to_lock);
            if (lock.owns_lock())
                for (const auto& c : children) {
                    if (!c->alive()) return fail("backend on port " + std::to_string(c->port()) + " exited");
                    probes.emplace_back(c->port(), c->ready_path());
                }
        }
        for (const auto& [port, path] : probes) {
            // Lemonade's BackendWatchdog gives the whole probe 2 s and wants a 200 (curl timeout,
            // HttpClient::is_reachable): ask the backend for 1.5 s at most, so serve answers first
            httplib::Client c("127.0.0.1", port);
            c.set_connection_timeout(std::chrono::milliseconds(500));
            c.set_read_timeout(std::chrono::seconds(1));
            auto r = c.Get(path);
            if (!r) return fail("backend on port " + std::to_string(port) + " does not answer " + path);
            if (r->status != 200)
                return fail("backend on port " + std::to_string(port) + " answers " + path + " with " + std::to_string(r->status));
        }
        res.status = 200;
        res.set_content(R"({"status":"ok"})", "application/json");
    };
    srv.Get("/health", health);
    srv.Get("/v1/health", health);
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array({{{"id", id}, {"object", "model"}, {"owned_by", "1bit"}, {"device", device}}});
        if (!o.embed.empty()) data.push_back({{"id", stem(o.embed)}, {"object", "model"}, {"owned_by", "1bit"}, {"device", rag_device()}, {"role", "embedding"}});
        if (!o.rerank.empty()) data.push_back({{"id", stem(o.rerank)}, {"object", "model"}, {"owned_by", "1bit"}, {"device", rag_device()}, {"role", "rerank"}});
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });
    // laya: the backends by index, started on first use (under start_mu; a start waits for
    // the backend to be ready, up to 600 s)

    std::vector<Child*> started(backends.size(), nullptr);
    auto ensure = [&](size_t i) -> bool {
        std::lock_guard<std::mutex> lock(start_mu);
        if (started[i]) return true;
        const Launch& b = backends[i];
        std::fprintf(stderr, "1bit serve: %s on %s (%s), picked by laya\n", id.c_str(), b.device.c_str(), b.argv[0].c_str());
        auto c = std::make_unique<Child>(b.argv, b.env, b.port, b.ready_path, backend_line, b.env_override);
        if (!c->wait_ready(std::chrono::seconds(600), g_stop)) return false;
        warm_backend(b.port, b.device);
        started[i] = c.get();
        children.push_back(std::move(c));
        return true;
    };
    std::vector<std::atomic<int>> inflight(backends.size());
    std::vector<std::atomic<long>> routed(backends.size());
    std::mutex route_mu;
#ifdef ONEBIT_LAYA
    std::mutex scorer_mu, class_mu;
    std::unordered_map<size_t, onebit::laya::RequestClass> class_cache;  // conversation -> class
    std::deque<size_t> class_order;
#endif
    auto post = [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        // the first backend until it is full, then the one with the fewest in flight;
        // choosing and reserving are one step, or a burst of requests all read "empty"
        std::shared_ptr<InFlight> held;
        size_t pick = 0;
        if (o.pm && q.path == "/v1/chat/completions") {
            // the Project Manager's tool loop (app/pm.cpp); /v1/completions forwards as usual
            {
                std::lock_guard<std::mutex> lock(route_mu);
                held = std::make_shared<InFlight>(&inflight[0]);
            }
            ++routed[0];
            pm::handle_chat(pm_cfg, backends[0].port, id, drop_model, set_model, q, r);
            return;
        }
#ifdef ONEBIT_LAYA
        if (laya) {
            // one decision per conversation: later turns reuse the class of the first
            const size_t key = conversation_key(q);
            onebit::laya::RequestClass cls;
            bool fresh = false;
            {
                std::lock_guard<std::mutex> lock(class_mu);
                if (const auto c = class_cache.find(key); c != class_cache.end()) cls = c->second;
            }
            if (cls.label.empty()) {
                std::lock_guard<std::mutex> lock(scorer_mu);  // the scorer is not reentrant
#if defined(__linux__)
                if (laya_daemon) cls = classify_request_gguf(*laya_daemon, request_state(q, true));
                else
#endif
                cls = onebit::laya::classify_request(*scorer, request_state(q, true));
                fresh = true;
            }
            if (cls.label.empty()) {
                r.status = 502;
                r.set_content(R"({"error":{"message":"laya routing failed"}})", "application/json");
                return;
            }
            if (fresh) {
                std::lock_guard<std::mutex> lock(class_mu);
                if (class_cache.size() >= 4096) {  // oldest conversation first
                    class_cache.erase(class_order.front());
                    class_order.pop_front();
                }
                if (class_cache.emplace(key, cls).second) class_order.push_back(key);
            }
            const std::string dev = policy.pick(cls.label, cls.confidence, laya_devices);
            const auto it = std::find(laya_devices.begin(), laya_devices.end(), dev);
            if (it == laya_devices.end()) {
                r.status = 502;
                r.set_content(R"({"error":{"message":"laya routing failed"}})", "application/json");
                return;
            }
            char route_hdr[96];
            std::snprintf(route_hdr, sizeof route_hdr, "%s %.2f %s", cls.label.c_str(), cls.confidence, dev.c_str());
            if (fresh) std::fprintf(stderr, "1bit serve: laya: %s\n", route_hdr);
            pick = size_t(it - laya_devices.begin());
            if (!ensure(pick)) {
                r.status = 503;
                r.set_content(json{{"error", {{"message", backends[pick].device + " did not become ready"}}}}.dump(), "application/json");
                return;
            }
            held = std::make_shared<InFlight>(&inflight[pick]);
            ++routed[pick];
            forward(backends[pick].port, held, -1, id, backends[pick].drop_model, backends[pick].set_model, o.precise, q, r, &watch);
            r.set_header("X-1bit-Route", route_hdr);  // class, Laya confidence, device
            return;
        }
#endif
        {
            std::lock_guard<std::mutex> lock(route_mu);
            held = std::make_shared<InFlight>(&inflight[pick]);
        }
        ++routed[pick];
        if (backends[pick].hybrid && hybrid_prefill) {
            // the handoff (app/serve_hybrid.h): template + tokenize on the backend, the provider's prefill,
            // restore into slot 0, then the request pinned to that slot. One slot: one request at a time,
            // held through its decode so the next handoff cannot replace a KV still in use.
            json body;
            try {
                body = json::parse(q.body);
            } catch (const std::exception&) {
                r.status = 400;
                r.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
                return;
            }
            // The lock is held until the backend has started on the slot (its first byte): from then
            // on llama-server itself defers a restore into a busy slot. httplib runs the handler and
            // the streamed reply on one thread, so the unlock is on the locking thread.
            auto slot = std::make_shared<std::unique_lock<std::mutex>>(hybrid_prefill->slot_mutex());
            const auto t0 = std::chrono::steady_clock::now();
            const hybrid::Outcome h = hybrid_prefill->prepare(backends[pick].port, q.path, body);
            if (!h.handed_over) {
                std::fprintf(stderr, "1bit serve: hybrid: %s; the backend prefills this request itself\n", h.why.c_str());
                slot->unlock();
                forward(backends[pick].port, held, -1, id, drop_model, set_model, o.precise, q, r, &watch);
                return;
            }
            const size_t n = h.n_tokens;
            const double prefill_ms = h.prefill_ms, restore_ms = h.restore_ms;
            forward(backends[pick].port, held, -1, id, drop_model, set_model, o.precise, q, r, &watch, &body,
                    [t0, n, prefill_ms, restore_ms, slot] {
                        const double ttft = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        std::fprintf(stderr, "1bit serve: hybrid: %zu prompt tokens: prefill %.0f ms, restore %.0f ms, first byte %.0f ms\n",
                                     n, prefill_ms, restore_ms, ttft);
                        if (slot->owns_lock()) slot->unlock();
                    });
            return;
        }
        forward(backends[pick].port, held, -1, id, drop_model, set_model, o.precise, q, r, &watch);
    };
    srv.Post("/v1/chat/completions", post);
    srv.Post("/v1/completions", post);
    // The rest of llama-server's API, for the devices that run llama-server (Lemonade's
    // llamacpp backend, which the onebit recipe inherits, forwards these to us). They go to
    // the first backend; the NPU and DwarfStar answer 501.
    const bool llama = device == "hrx" || device == "cpu" || device == "hybrid" ||
                       (laya && (backends[0].device == "hrx" || backends[0].device == "cpu"));
    auto unsupported = [&](const httplib::Request& q, httplib::Response& r) {
        r.status = 501;
        r.set_content(json{{"error", {{"message", q.path + " is not available on --device " + device},
                                      {"type", "not_implemented"}}}}.dump(), "application/json");
    };
    auto first = [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        if (laya && !ensure(0)) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"backend did not become ready"}})", "application/json");
            return;
        }
        relay(backends[0].port, q, r);
    };
    auto first_json = [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        if (laya && !ensure(0)) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"backend did not become ready"}})", "application/json");
            return;
        }
        forward(backends[0].port, nullptr, -1, id, drop_model, set_model, o.precise, q, r, &watch);
    };
    for (const char* path : {"/v1/responses", "/tokenize", "/detokenize", "/apply-template"})
        srv.Post(path, llama ? httplib::Server::Handler(first_json) : httplib::Server::Handler(unsupported));
    for (const char* path : {"/slots", "/props"})
        srv.Get(path, llama ? httplib::Server::Handler(first) : httplib::Server::Handler(unsupported));
    // what only the engine sees (app/serve_metrics.h: GTT, Tctl, CPU fallbacks, program-cache
    // flushes, backend starts and watchdog kills; Lemonade already reports requests, TTFT and
    // tok/s), then the backend's own where it has them (llama-server runs with --metrics)
    srv.Get("/metrics", [&](const httplib::Request&, httplib::Response& r) {
        std::string text = metrics.render(lifecycle);
        int port = 0;
        {
            std::unique_lock<std::mutex> lock(start_mu, std::try_to_lock);
            if (ready && lock.owns_lock() && (!laya || started[0])) port = backends[0].port;
        }
        if (port) {
            httplib::Client c("127.0.0.1", port);
            c.set_connection_timeout(1);
            c.set_read_timeout(3);
            if (auto br = c.Get("/metrics"); br && br->status == 200) text += br->body;
        }
        r.set_content(text, "text/plain; version=0.0.4; charset=utf-8");
    });
    srv.Post(R"(/slots/(\d+))", llama ? httplib::Server::Handler(first) : httplib::Server::Handler(unsupported));
    // RAG: embeddings and rerank go to their own backends, the reply names the model served
    auto rag_route = [&](size_t i, const std::string& model) {
        return [&, i, model](const httplib::Request& q, httplib::Response& r) {
            if (!ready) {
                r.status = 503;
                r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
                return;
            }
            forward(rag[i].port, nullptr, -1, stem(model), true, "", false, q, r);
        };
    };
    if (!o.role.empty()) {
        auto h = [&](const httplib::Request& q, httplib::Response& r) {
            if (!ready) {
                r.status = 503;
                r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
                return;
            }
            forward(backends[0].port, nullptr, -1, id, true, "", false, q, r);
        };
        if (o.role == "embedding")
            for (const char* path : {"/v1/embeddings", "/embeddings"}) srv.Post(path, h);
        else
            for (const char* path : {"/v1/rerank", "/rerank", "/v1/reranking", "/reranking"}) srv.Post(path, h);
    }
    size_t next_rag = 0;
    if (!o.embed.empty()) {
        auto h = rag_route(next_rag++, o.embed);
        srv.Post("/v1/embeddings", h);
        srv.Post("/embeddings", h);
    }
    if (!o.rerank.empty()) {
        auto h = rag_route(next_rag++, o.rerank);
        srv.Post("/v1/rerank", h);
        srv.Post("/rerank", h);
    }

    if (!srv.bind_to_port(o.host, o.port))
        throw std::runtime_error("cannot listen on " + o.host + ":" + std::to_string(o.port));
    std::thread listener([&] { srv.listen_after_bind(); });
    struct Stop {
        httplib::Server& s;
        std::thread& t;
        ~Stop() { s.stop(); if (t.joinable()) t.join(); }
    } stop{srv, listener};

#ifdef _WIN32
    std::signal(SIGTERM, on_stop_signal);
    std::signal(SIGINT, on_stop_signal);
#else
    struct sigaction sa{};
    sa.sa_handler = on_stop_signal;
    ::sigaction(SIGTERM, &sa, nullptr);
    ::sigaction(SIGINT, &sa, nullptr);
#endif
    for (const auto& b : laya ? std::vector<Launch>{} : backends) {
        std::fprintf(stderr, "1bit serve: %s on %s (%s)\n", id.c_str(), b.device.c_str(), b.argv[0].c_str());
        children.push_back(std::make_unique<Child>(b.argv, b.env, b.port, b.ready_path, backend_line, b.env_override));
    }
    for (const auto& b : rag) {
        std::fprintf(stderr, "1bit serve: %s on %s (%s)\n", b.argv[2].c_str(), b.device.c_str(), b.argv[0].c_str());
        children.push_back(std::make_unique<Child>(b.argv, b.env, b.port));
    }
    // Launch::device is "hrx", or "hrx reranking" for a role or RAG server: the device is the first word
    auto on_device = [](const Launch& b) { return b.device.substr(0, b.device.find(' ')); };
    for (size_t i = 0; i < children.size(); i++) {
        if (!children[i]->wait_ready(std::chrono::seconds(600), g_stop)) {
            if (g_stop) return 0;
            // a strict HRX backend (app/hrx_strict.h) that refused a node said which; serve says it again
            if (const std::string why = metrics.strict_refusal(); !why.empty())
                std::fprintf(stderr, "1bit serve: %s\n", strict_refusal_message(why).c_str());
            else
                std::fprintf(stderr, "1bit serve: backend %zu did not become ready\n", i);
            return 1;
        }
    }
    // one synthetic request per backend, now that it is ready: the first client request would otherwise pay
    // the one-time per-shape cost (see warm_backend)
    if (!laya) {
        for (size_t i = 0; i < children.size() && i < backends.size(); i++) {
            warm_backend(children[i]->port(), on_device(backends[i]));
        }
    }
    // engine #390: --strict-device. GGML_HRX_STRICT=1 makes the backend abort at load on the first
    // node off HRX, so a backend that reached ready while its report names off-HRX nodes ignored it.
    if (o.strict == StrictDevice::On || o.placement_report) {
        const bool hrx_run = std::any_of(backends.begin(), backends.end(),
                                         [&](const Launch& b) { return on_device(b) == "hrx"; });
        const auto report = metrics.placement();
        if (o.strict == StrictDevice::On && report && report->cpu_fallback_total.value_or(0) > 0) {
            std::fprintf(stderr, "1bit serve: %s\n", placement_refusal(*report).c_str());
            return 1;
        }
        if (!hrx_run)
            // --device auto that landed on the CPU, an NPU directory, ds4: no HRX backend, so no
            // placement report and nothing for either flag to check.
            std::fprintf(stderr, "1bit serve: %s: no HRX backend in this run, nothing to check\n",
                         o.strict == StrictDevice::On ? "--strict-device" : "--placement-report");
        else if (!report)
            // both --placement-report and --strict-device pass GGML_HRX_REPORT=2 and -lv 4, so a missing
            // line means the backend never spoke, not that the verbosity was wrong
            std::fprintf(stderr, "1bit serve: HRX placement: the backend printed no placement line "
                                 "(GGML_HRX_REPORT=2 and -lv 4 were passed to it)\n");
    }
    ready = true;
    lifecycle.count_start();
    std::fprintf(stderr, "1bit serve: ready on http://%s:%d%s\n", o.host.c_str(), o.port,
                 laya ? " (laya routes each request; a device's backend starts when first picked)" : "");
    if (o.pm) {
        std::string experts;
        for (const auto& e : pm_cfg.experts) experts += (experts.empty() ? "" : ", ") + e.key + "=" + e.id;
        std::fprintf(stderr, "1bit serve: pm: %s is the project manager; experts through %s: %s (%s, %d rounds max)\n",
                     id.c_str(), o.lemonade_url.c_str(), experts.empty() ? "none" : experts.c_str(),
                     o.pm_experts.empty() ? "config/pm-experts.json" : o.pm_experts.c_str(), pm_cfg.max_rounds);
        if (pm_cfg.router == "laya") {
            std::string routes;
            for (const auto& [label, key] : pm_cfg.routes)
                routes += (routes.empty() ? "" : ", ") + label + "=" + (key.empty() ? std::string("pm") : key);
            std::fprintf(stderr, "1bit serve: pm: router laya: %s (min confidence %.2f)\n",
                         routes.empty() ? "no class routed" : routes.c_str(), pm_cfg.routes_min_confidence);
        }
        if (!o.pm_spawn_url.empty())
            std::fprintf(stderr, "1bit serve: pm: agent workers (spawn_worker) through %s\n", o.pm_spawn_url.c_str());
    }
    // The progress watchdog (app/serve_watchdog.h): while requests are in flight, probe each busy
    // backend's main loop (GET /slots answers only between llama-server's decode steps) and look
    // for a request with no progress past its limit.
    std::atomic<bool> watchdog_quit{false};
    ProgressWatch::Stall stall;
    std::thread watchdog_thread;
    if (watch.enabled()) {
        watchdog_thread = std::thread([&] {
            using clock = std::chrono::steady_clock;
            const double secs = o.watchdog.secs;
            const auto every = std::chrono::duration<double>(std::clamp(secs / 4, 0.25, 5.0));
            const auto timeout = std::chrono::milliseconds(int64_t(std::clamp(secs / 2, 0.5, 10.0) * 1000));
            std::unordered_map<int, int> probe;   // port -> 1 answers /slots, -1 has none, 0 not known yet
            std::unordered_map<int, clock::time_point> last;
            while (!watchdog_quit && !g_stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                for (const int port : watch.busy_ports()) {
                    if (probe[port] < 0 || clock::now() - last[port] < every) continue;
                    last[port] = clock::now();
                    httplib::Client c("127.0.0.1", port);
                    c.set_connection_timeout(timeout);
                    c.set_read_timeout(timeout);
                    if (auto r = c.Get("/slots")) {
                        if (r->status == 200) {
                            probe[port] = 1;
                            watch.alive(port);
                        } else if (probe[port] == 0) {
                            probe[port] = -1;   // DwarfStar, or llama-server --no-slots
                        }
                    }
                }
                if (auto st = watch.stalled([&](int port) { return probe[port] >= 0; })) {
                    std::lock_guard<std::mutex> lock(wedged_mu);
                    stall = *st;
                    char why[256];
                    std::snprintf(why, sizeof why, "%s made no progress for %.0f s", st->what.c_str(), st->idle);
                    wedged = why;
                    return;
                }
            }
        });
    }
    struct JoinWatchdog {
        std::atomic<bool>& quit;
        std::thread& t;
        ~JoinWatchdog() { quit = true; if (t.joinable()) t.join(); }
    } join_watchdog{watchdog_quit, watchdog_thread};
    auto is_wedged = [&] {
        std::lock_guard<std::mutex> lock(wedged_mu);
        return !wedged.empty();
    };
    while (all_alive() && !g_stop && !is_wedged()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (is_wedged() && !g_stop) {
        // Kill the wedged backend and exit non-zero: Lemonade's BackendWatchdog sees the process go
        // and loads the model again. Say why, with what the box and the backend said last.
        lifecycle.count_watchdog_kill();
        std::lock_guard<std::mutex> lock(start_mu);
        Child* bad = nullptr;
        for (auto& c : children) if (c->port() == stall.port) bad = c.get();
        if (stall.what.empty()) {
            // not a stall: the backend said it lost its GPU (backend_line)
            std::string why;
            {
                std::lock_guard<std::mutex> wl(wedged_mu);
                why = wedged;
            }
            if (!bad && !children.empty()) bad = children.front().get();
            std::fprintf(stderr, "1bit serve: watchdog: %s (%s backend, pid %d); killing it and exiting so it is "
                                 "started again\n", why.c_str(), device.c_str(), bad ? bad->pid() : -1);
        } else {
            std::fprintf(stderr, "1bit serve: watchdog: %s on the %s backend (port %d, pid %d) made no progress for %.0f s "
                                 "(limit %.0f s; %llu bytes received in %.0f s); killing it and exiting so Lemonade reloads it\n",
                         stall.what.c_str(), device.c_str(), stall.port, bad ? bad->pid() : -1, stall.idle, stall.limit,
                         (unsigned long long) stall.bytes, stall.age);
        }
        if (const auto t = sysinfo::tctl_celsius()) std::fprintf(stderr, "1bit serve: watchdog: Tctl %.1f C\n", *t);
        if (bad) {
            const auto lines = bad->last_lines(12);
            if (!lines.empty()) std::fprintf(stderr, "1bit serve: watchdog: the backend's last lines:\n");
            for (const auto& l : lines) std::fprintf(stderr, "1bit serve: watchdog: | %s\n", l.c_str());
            if (!bad->kill_now(std::chrono::seconds(10)))
                std::fprintf(stderr, "1bit serve: watchdog: pid %d did not exit after SIGKILL (stuck in the kernel; "
                                     "a GPU reset may be needed)\n", bad->pid());
        }
        for (auto& c : children) if (c.get() != bad) c->kill_now(std::chrono::seconds(5));
        return 1;
    }
    if (backends.size() > 1) {
        std::fprintf(stderr, "1bit serve: requests routed:");
        for (size_t i = 0; i < backends.size(); i++) std::fprintf(stderr, " %s %ld", backends[i].device.c_str(), routed[i].load());
        std::fprintf(stderr, "\n");
    }
    if (g_stop) {
        std::fprintf(stderr, "1bit serve: stopping\n");
        std::lock_guard<std::mutex> lock(start_mu);
        for (auto& c : children) c->stop();
        return 0;
    }
    lifecycle.count_backend_exit();
    std::string how;
#ifndef _WIN32
    {
        std::lock_guard<std::mutex> lock(start_mu);
        for (auto& c : children) if (how.empty() && !c->alive()) how = c->exit_text();
    }
#endif
    std::fprintf(stderr, "1bit serve: a %s backend exited%s%s\n", device.c_str(), how.empty() ? "" : ": ", how.c_str());
    // a graph only a request schedules (a long prompt, a multi-sequence batch) is checked then
    if (const std::string why = metrics.strict_refusal(); !why.empty())
        std::fprintf(stderr, "1bit serve: %s\n", strict_refusal_message(why).c_str());
    return 1;
}

// Runs argv[0] with argv and no shell, and returns its exit code (-1 if it could not run).
// quiet sends the child's stdout and stderr to /dev/null.
int run_program(const std::vector<std::string>& argv, bool quiet) {
    std::vector<char*> args;
    for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
#if defined(_WIN32)
    (void) quiet;
    return static_cast<int>(_spawnvp(_P_WAIT, args[0], args.data()));
#else
    const pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (quiet) {
            const int null_fd = ::open("/dev/null", O_WRONLY);
            if (null_fd >= 0) {
                ::dup2(null_fd, 1);
                ::dup2(null_fd, 2);
                ::close(null_fd);
            }
        }
        ::execvp(args[0], args.data());
        ::_exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string command_text(const std::vector<std::string>& argv) {
    std::string text;
    for (const auto& a : argv) text += (text.empty() ? "\"" : " \"") + a + "\"";
    return text;
}

// The Q4NX cache root: $ONEBIT_Q4NX_CACHE, else ~/.cache/1bit/q4nx. With HOME unset this used to
// be /tmp/.cache, which every local user can write; a cached directory there would be served
// (weights, tokenizer, kernel directory) without knowing who made it. The home directory now
// comes from the user database instead, and serve refuses rather than guess.
std::string q4nx_cache_root() {
    if (const char* c = std::getenv("ONEBIT_Q4NX_CACHE"); c && *c) return c;
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/1bit/q4nx";
#ifndef _WIN32
    if (const passwd* pw = ::getpwuid(::geteuid()); pw && pw->pw_dir && *pw->pw_dir)
        return std::string(pw->pw_dir) + "/.cache/1bit/q4nx";
#endif
    throw std::runtime_error("no home directory for the Q4NX cache: set HOME or ONEBIT_Q4NX_CACHE");
}

// Repack a GGUF into a Q4NX model directory with the FLM_Q4NX_Converter
// (Python), cached by GGUF stem so a repeated serve reuses the conversion.
// Override the tool paths with ONEBIT_Q4NX_PYTHON / ONEBIT_Q4NX_CONVERTER and
// the cache root with ONEBIT_Q4NX_CACHE.
std::string repack_gguf(const std::string& gguf) {
    const std::string stem = fs::path(gguf).stem().string();
    const std::string cache = q4nx_cache_root();
    const std::string out = cache + "/" + stem;

    const char* py = std::getenv("ONEBIT_Q4NX_PYTHON");
    const std::string python = py ? py : "python3";
    const char* rp = std::getenv("ONEBIT_Q4NX_REPACK");
#ifdef ONEBIT_NPU_REPACK_SCRIPT
    // the package keeps it beside 1bit (scripts/package-linux.sh); a build tree has it at the
    // path CMake wrote, ../src/scripts/repack_gguf.py, which no install layout can mirror
    const std::string script = rp ? rp : packaged_path("scripts/repack_gguf.py", ONEBIT_NPU_REPACK_SCRIPT);
#else
    const std::string script = rp ? rp : "";
#endif
    const char* cv = std::getenv("ONEBIT_Q4NX_CONVERTER");

    // A cache HIT used to return whatever is on disk, which bypasses the repack
    // and therefore the repack's declared-scale guard -- so a dir produced by an
    // older converter (e.g. before the MiniCPM4 embedding_scale / residual_scale
    // / logit_scale fix) would be served silently forever, and the guard would
    // only ever protect freshly built artifacts.  Validate the hit with the same
    // metadata-only scan instead: it reads the GGUF header and config.json, no
    // reconversion and no NPU, so a hit costs milliseconds.  On failure the dir
    // is discarded and rebuilt rather than served.  The provenance-stamp
    // comparison (converter path + commit, q/k-reorder flag) folds into this same
    // gate; until it lands this catches the declared-vs-carried half.
    bool cache_present = fs::exists(out + "/model.q4nx");
    if (cache_present) {
        // Provenance half (docs/npu.md).  Reuse only a dir the converter this build
        // would use now produced: the cache cannot tell, and neither can the
        // declared-vs-carried scan.  The comparison lives in the repack script because
        // that is where the converter resolution lives.  Exit 0 current, 2 cannot tell
        // (fail open, so a missing converter cannot brick serving), else stale.
        const int stamp_rc = script.empty()
                                 ? 2
                                 : run_program({ python, script, "--check-stamp", out }, true);
        if (stamp_rc == 2) {
            static bool stamp_warned = false;
            if (!stamp_warned) {
                stamp_warned = true;
                std::fprintf(stderr, "\n1bit serve: ============================================================\n"
                                     "1bit serve: WARNING: the Q4NX provenance stamp could not be checked\n"
                                     "1bit serve:   for this process, so cached dirs are reused UNVERIFIED.\n"
                                     "1bit serve: ============================================================\n\n");
            }
        } else if (stamp_rc != 0) {
            std::fprintf(stderr, "1bit serve: cached Q4NX dir %s FAILED the provenance check; "
                                 "discarding and re-repacking\n", out.c_str());
            fs::remove_all(out);
            cache_present = false;
        }
    }
    if (cache_present) {
        const char* ck = std::getenv("ONEBIT_Q4NX_CHECK");
        std::string checker = ck ? ck : "";
        if (checker.empty() && !script.empty())
            checker = (fs::path(script).parent_path() / "check_repack_config.py").string();
        if (checker.empty() || !fs::exists(checker)) {
            // This is the ONE state where the guard does not apply, and it applies
            // to every dir for the whole process -- so say it once, unmissably,
            // rather than repeating a line that gets lost in serve output.  A baked
            // path that resolves to a moved/removed checkout must be obvious.
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr,
                    "\n1bit serve: ============================================================\n"
                    "1bit serve: WARNING: the Q4NX declared-scale guard is INACTIVE for\n"
                    "1bit serve:   this process.  No check_repack_config.py was found at\n"
                    "1bit serve:   '%s' and ONEBIT_Q4NX_CHECK is unset, so cached Q4NX\n"
                    "1bit serve:   dirs are reused WITHOUT validation.\n"
                    "1bit serve: ============================================================\n\n",
                    checker.empty() ? "(unresolved)" : checker.c_str());
            }
            return out;
        }
        if (run_program({ python, checker, gguf, out }, true) == 0) return out;
        std::fprintf(stderr, "1bit serve: cached Q4NX dir %s FAILED the declared-scale scan; "
                             "discarding and re-repacking\n", out.c_str());
        fs::remove_all(out);
    }

    if (python.find('/') != std::string::npos && !fs::exists(python))
        throw std::runtime_error("Q4NX repack python not found: " + python + " (set ONEBIT_Q4NX_PYTHON)");
    if (!fs::exists(script)) throw std::runtime_error("Q4NX repack script not found: " + script + " (set ONEBIT_Q4NX_REPACK)");

    std::fprintf(stderr, "1bit serve: repacking %s -> %s ...\n", gguf.c_str(), out.c_str());
    fs::create_directories(out);
    std::vector<std::string> repack = { python, script, gguf, out };
    if (cv) repack.push_back(cv);
    const int rc = run_program(repack, false);
    if (rc != 0 || !fs::exists(out + "/model.q4nx"))
        throw std::runtime_error("GGUF repack failed (rc=" + std::to_string(rc) + "): " + command_text(repack));
    return out;
}

// Repack a GGUF into the Q4NX layout the fused lane reads, with
// tools/gguf_to_q4nx.py (docs/npu.md, "From a GGUF").  This is a DIFFERENT
// layout from repack_gguf()'s: the per-op forward's full_i8_*.elf designs read
// the converter's packing, the lane reads the q4_1 tile packing this packer
// writes.  The two are not interchangeable -- the same repacked directory
// answers " Paris" through the forward and the wrong tokens through the lane
// (evidence/gguf-npu-six/task3-fusion-feasibility.md) -- so it goes to its own
// cache dir and the two never alias.
//
// Returns the directory, or "" when the lane layout cannot be produced for this
// GGUF (an architecture the packer does not model, or a missing tool/tokenizer),
// so the caller falls back to the per-op forward instead of failing.
std::string repack_gguf_lane(const std::string& gguf, const std::string& lane_dir) {
    const std::string stem = fs::path(gguf).stem().string();
    const std::string cache = q4nx_cache_root();
    const std::string out = cache + "/" + stem + "-lane";

    const char* py = std::getenv("ONEBIT_Q4NX_PYTHON");
    const std::string python = py ? py : "python3";
    const char* lp = std::getenv("ONEBIT_Q4NX_LANE_PACK");
#ifdef ONEBIT_NPU_LANE_PACK_SCRIPT
    const std::string script = lp ? lp : packaged_path("tools/gguf_to_q4nx.py", ONEBIT_NPU_LANE_PACK_SCRIPT);
#else
    const std::string script = lp ? lp : "";
#endif
    if (script.empty() || !fs::exists(script)) {
        std::fprintf(stderr, "1bit serve: lane layout packer not found (%s); using the per-op forward\n",
                     script.empty() ? "(unset ONEBIT_Q4NX_LANE_PACK)" : script.c_str());
        return {};
    }
    // The GGUF's own vocabulary is not converted to a tokenizer.json; the lane
    // reads the model directory's.  Take it from the lane's model directory (a
    // lane kernel dir is conventionally <model dir>/npu) unless told otherwise.
    std::string tok = std::getenv("ONEBIT_NPU_LANE_TOKENIZER") ? std::getenv("ONEBIT_NPU_LANE_TOKENIZER") : "";
    if (tok.empty()) {
        const fs::path parent = fs::path(lane_dir).parent_path();
        if (fs::exists(parent / "tokenizer.json")) tok = parent.string();
    }
    const bool cached = fs::exists(out + "/model.q4nx") && !std::getenv("ONEBIT_Q4NX_LANE_REPACK");
    if (!cached) {
        if (fs::exists(out)) fs::remove_all(out);
        std::fprintf(stderr, "1bit serve: repacking %s -> %s (fused-lane layout) ...\n", gguf.c_str(), out.c_str());
        fs::create_directories(out);
        std::vector<std::string> cmd = { python, script, "pack", gguf, out };
        if (!tok.empty()) { cmd.push_back("--tokenizer"); cmd.push_back(tok); }
        const int rc = run_program(cmd, true);
        if (rc != 0 || !fs::exists(out + "/model.q4nx")) {
            fs::remove_all(out);
            std::fprintf(stderr, "1bit serve: %s is not packable in the fused-lane layout (rc=%d); "
                                 "using the per-op forward\n", gguf.c_str(), rc);
            return {};
        }
    }
    // A lane model directory must carry a tokenizer.json (npu/tokenizer.cpp).
    if (!fs::exists(out + "/tokenizer.json")) {
        if (tok.empty() || !fs::exists(fs::path(tok) / "tokenizer.json")) {
            std::fprintf(stderr, "1bit serve: fused-lane layout has no tokenizer (set "
                                 "ONEBIT_NPU_LANE_TOKENIZER); using the per-op forward\n");
            return {};
        }
        fs::copy_file(fs::path(tok) / "tokenizer.json", fs::path(out) / "tokenizer.json",
                      fs::copy_options::overwrite_existing);
        if (fs::exists(fs::path(tok) / "tokenizer_config.json"))
            fs::copy_file(fs::path(tok) / "tokenizer_config.json", fs::path(out) / "tokenizer_config.json",
                          fs::copy_options::overwrite_existing);
    }
    return out;
}

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit serve -m <model> [--port 8000] [--host 127.0.0.1]\n"
                 "                  [--device auto|npu|hrx|hybrid|cpu|ds4] [--ctx-size N] [--alias NAME]\n"
                 "                  [--llama-server PATH] [--hrx-libhsa PATH]\n"
                 "                  [--prefill-provider PATH]   --device hybrid: the program that prefills each prompt\n"
                 "                  (the NPU engine) and writes its KV for llama-server, which decodes it on HRX\n"
                 "                  (or ONEBIT_HYBRID_PREFILL; ONEBIT_HYBRID_ELFS passes --elfs; docs/hybrid.md)\n"
                 "                  [--hrx-cwsr-placement=auto|on|off]   engine#315: on a CWSR-on amdgpu (auto, the default)\n"
                 "                  or always (on), HRX backends get HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0\n"
                 "                  HRX_ALLOW_NO_SVM=1, which put the CWSR save area in GTT (docs/hrx.md)\n"
                 "                  [--ds4 PATH] [--ssd-streaming]   DwarfStar (--device ds4; docs/dwarfstar.md)\n"
                 "                  [--mtp HEAD.gguf] [--mtp-max N] [--mtp-p-min P]   multi-token prediction (hrx, cpu)\n"
                 "                  [--precise]             re-prefill each request in full: follow-up turns come out\n"
                 "                                          identical to a cold run, at the cost of the TTFT (ONEBIT_PRECISE=1)\n"
                 "                  [--no-mtp]   a GGUF with its own MTP head drafts with it on hrx by default;\n"
                 "                  this (or ONEBIT_NO_MTP=1) turns that off\n"
                 "                  [--dflash DRAFT.gguf]   DFlash draft model instead of --mtp; --mtp-max/--mtp-p-min apply\n"
                 "                  [--moe-slots N|auto] [--moe-subst R] [--moe-prefetch N]   stream MoE experts from the file,\n"
                 "                                    N held in RAM (auto: what fits); R: resident experts stand in for missing\n"
                 "                                    ones; prefetch N layers ahead (default 0). Not in this build: it moves\n"
                 "                                    to HRX with Qwen3.8-Flash-Next (docs/moe-streaming.md)\n"
                 "                  [--mmproj MMPROJ.gguf]   images in chat messages (hrx, cpu)\n"
                 "                  [--parallel N]   N requests decoded together (continuous batching)\n"

                 "                  [--embed MODEL.gguf] [--rerank MODEL.gguf]   RAG: /v1/embeddings and /v1/rerank\n"
                 "                  [--embedding | --reranking]   serve the model itself as an embedding or reranking model\n"
                 "                  [--npu-opt KEY=VALUE ...]   an option for a private NPU route (docs/npu.md)\n"
                 "                  [--laya | --laya-model DIR] [--route-policy FILE]   with --device auto, Laya classifies\n"
                 "                  each conversation and the route policy picks its device (docs/laya.md)\n"
                 "                  [--recipes FILE | --no-recipes]   tuned llama-server settings per model and route\n"
                 "                  (default: config/recipes.json, built in; docs/recipes.md)\n"
                 "                  [--pm [--pm-experts FILE] [--pm-router model|laya] [--lemonade-url URL]]   Project Manager: the model answers or\n"
                 "                  delegates each chat to an expert model Lemonade runs (default experts:\n"
                 "                  config/pm-experts.json, built in; URL default http://127.0.0.1:8000; docs/pm.md)\n"
                 "                  [--pm-spawn-url URL]   with --pm: the tool spawn_worker asks this executor, a local\n"
                 "                  http:// URL (off by default), to start a long-running agent worker (docs/pm.md)\n"
                 "                  [--no-mem-check | --mem-check]   before a .gguf loads, serve estimates its memory and\n"
                 "                  refuses one that does not fit (--no-mem-check or ONEBIT_MEM_CHECK=0: skip;\n"
                 "                  --mem-check: print the estimate and exit, 0 fits, 3 does not; docs/serve.md)\n"
                 "                  [--watchdog SECS] [--watchdog-prompt-tps N]   a request with no progress from its backend\n"
                 "                  for SECS (default 120, or $ONEBIT_WATCHDOG_SECS; 0 = off), plus prompt tokens / N\n"
                 "                  (default 50) before its first byte, kills the backend and exits non-zero so Lemonade\n"
                 "                  reloads it (docs/serve.md)\n"
                 "                  [--placement-report]   every graph: the op types HRX placed on the CPU, their\n"
                 "                  counts and the graph split count; prints the worst graph again when a later\n"
                 "                  one has more off HRX (passes -lv 4 to llama-server; engine #390)\n"
                 "                  [--strict-device | --no-strict-device]   on HRX a node that would run on the CPU\n"
                 "                  refuses the load by default, naming its op and shape; --strict-device adds the\n"
                 "                  every-graph report (as --placement-report) and checks it; --no-strict-device\n"
                 "                  (or ONEBIT_STRICT_DEVICE=0) lets HRX leave such nodes on the CPU (engine #390)\n"
                 "                  [--print-backend]   print the llama-server serve would start (argv and env, one\n"
                 "                  JSON line) and exit: 0 printed, 3 does not fit, 5 the route needs serve itself\n"
                 "                  (NPU, ds4, --pm, --laya, --embed/--rerank); Lemonade's onebit recipe starts\n"
                 "                  it directly, one hop (docs/serve.md)\n"
                 "                  [--print-backend-env]   as --print-backend, but only the environment, one\n"
                 "                  NAME=VALUE line each (lemonade-hrx, stock Lemonade's HRX recipe: docs/lemonade.md)\n"
                 "  <model>: an NPU model directory (model.q4nx + npu/, or one a private NPU route serves),\n"
                 "           or a .gguf file\n");
}

}  // namespace

int run_serve(int argc, char** argv) {
    Options o;
    if (const char* e = std::getenv("ONEBIT_MEM_CHECK"); e && std::string(e) == "0") o.mem_check = false;
    // ONEBIT_PRECISE=1 is the same as --precise (every request re-prefills its conversation in full)
    if (const char* e = std::getenv("ONEBIT_PRECISE"); e && *e && std::string(e) != "0") o.precise = true;
    if (const char* e = std::getenv("ONEBIT_WATCHDOG_SECS"); e && *e) o.watchdog.secs = std::atof(e);
    // ONEBIT_STRICT_DEVICE=0 / 1 (app/hrx_strict.h); a flag on the command line wins
    o.strict = strict_from_env(std::getenv("ONEBIT_STRICT_DEVICE"));
    bool strict_flag = false, no_strict = false;
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") o.model = next();
        else if (a == "-p" || a == "--port") o.port = std::stoi(next());
        else if (a == "--host") o.host = next();
        else if (a == "--device") o.device = next();
        else if (a == "-c" || a == "--ctx-size") o.ctx_size = std::stoi(next());
        else if (a == "-fa" || a == "--flash-attn") o.flash_attn = next();
        else if (a == "--alias") o.alias = next();
        else if (a == "--llama-server") o.llama_server = next();
        else if (a == "--zinc")
            throw std::runtime_error("--zinc was removed with the engine's ZINC build (docs/serve.md)");
        else if (a == "--ds4") o.ds4 = next();
        else if (a == "--ssd-streaming") o.ssd_streaming = true;
        else if (a == "--hrx-libhsa") o.hrx_libhsa = next();
        else if (a == "--hrx-cwsr-placement") o.cwsr_placement = parse_cwsr_placement(next());
        else if (a.rfind("--hrx-cwsr-placement=", 0) == 0)
            o.cwsr_placement = parse_cwsr_placement(a.substr(std::string("--hrx-cwsr-placement=").size()));
        else if (a == "--mlx-server")
            throw std::runtime_error("--mlx-server was removed with the engine's MLX route; Lemonade's own MLX backend serves Apple Silicon (docs/lemonade.md)");
        else if (a == "--prefill-device" || a == "--prefill-min-tokens" || a == "--lean" || a == "--adaptive" ||
                 a == "--adaptive-at" || a == "--long-model" || a == "--long-from")
            // RFC #213 stage 3: these ran on the Vulkan and ROCm builds the engine no longer ships
            throw std::runtime_error(a + " was removed with the engine's Vulkan and ROCm builds (RFC #213); "
                                     "Lemonade's own llamacpp backend serves Vulkan and ROCm (docs/serve.md)");
        else if (a == "--mtp") o.mtp = next();
        else if (a == "--no-mtp") o.no_mtp = true;
        else if (a == "--precise") o.precise = true;
        else if (a == "--dflash") o.dflash = next();
        else if (a == "--moe-slots") {
            const std::string v = next();
            o.moe_slots = v == "auto" ? -1 : std::stoi(v);
        }
        else if (a == "--moe-prefetch") o.moe_prefetch = std::stoi(next());
        else if (a == "--moe-subst") o.moe_subst = next();
        else if (a == "--mmproj") o.mmproj = next();
        else if (a == "--mtp-max") o.mtp_max = std::stoi(next());
        else if (a == "--mtp-p-min") o.mtp_p_min = next();
        else if (a == "--parallel") o.parallel = std::stoi(next());
        else if (a == "--embed") o.embed = next();
        else if (a == "--embedding" || a == "--embeddings") o.role = "embedding";
        else if (a == "--reranking") o.role = "reranking";
        else if (a == "--rerank") o.rerank = next();
        else if (a == "--npu-opt") o.npu_opts.push_back(next());
        else if (a == "--prefill-provider") o.prefill_provider = next();
        else if (a == "--laya-model") o.laya_model = next();
        else if (a == "--laya") o.laya_auto = true;
        else if (a == "--route-policy") o.route_policy = next();
        else if (a == "--recipes") o.recipes = next();
        else if (a == "--no-recipes") o.no_recipes = true;
        else if (a == "--pm") o.pm = true;
        else if (a == "--pm-experts") o.pm_experts = next();
        else if (a == "--pm-router") o.pm_router = next();
        else if (a == "--pm-spawn-url") o.pm_spawn_url = next();
        else if (a == "--no-mem-check") o.mem_check = false;
        else if (a == "--mem-check") o.mem_check_only = true;
        else if (a == "--strict-device") {
            if (no_strict) throw std::runtime_error("--strict-device and --no-strict-device: pick one");
            o.strict = StrictDevice::On;
            strict_flag = true;
        }
        else if (a == "--no-strict-device") {
            if (strict_flag) throw std::runtime_error("--strict-device and --no-strict-device: pick one");
            o.strict = StrictDevice::Off;
            no_strict = true;
        }
        else if (a == "--placement-report") o.placement_report = true;
        else if (a == "--print-backend") o.print_backend = true;
        else if (a == "--print-backend-env") o.print_backend = o.print_backend_env = true;
        else if (a == "--watchdog") o.watchdog.secs = std::stod(next());
        else if (a == "--watchdog-prompt-tps") o.watchdog.prompt_tps = std::stod(next());
        else if (a == "--lemonade-url") o.lemonade_url = next();
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty()) { usage(stderr); return 2; }
    if (o.watchdog.secs < 0 || o.watchdog.prompt_tps < 0)
        throw std::runtime_error("--watchdog and --watchdog-prompt-tps take a number >= 0 (0 turns it off)");
    if (o.strict == StrictDevice::On && (o.device == "cpu" || o.device == "ds4" || o.device == "npu"))
        throw std::runtime_error("--strict-device needs --device hrx or auto: it refuses a graph that is not all on HRX");
    refuse_removed_device(o.device);
    // engine#315: --hrx-cwsr-placement=on in a build without the gate fails here, not at the first backend
    if (o.cwsr_placement == CwsrPlacement::On) (void) hrx_cwsr_env(o);
    if (o.print_backend) {
        if (fs::is_directory(o.model)) return serve_only("an NPU model directory runs in serve's own process");
        if (o.device == "npu") return serve_only("--device npu runs in serve's own process");
    }

    if (fs::is_directory(o.model)) {
        if (!o.mmproj.empty()) throw std::runtime_error("--mmproj works on the llama.cpp devices (hrx, cpu)");
        if (o.device != "auto" && o.device != "npu")
            throw std::runtime_error("an NPU model directory runs on --device npu");
#ifdef ONEBIT_NPU
        npu::PrivateOptions opts;
        for (const auto& kv : o.npu_opts) npu::parse_private_option(kv, opts);
        if (const std::string why = npu_model_problem(o.model, opts); !why.empty()) throw std::runtime_error(why);
#ifdef ONEBIT_LAYA
        // A Q4NX model directory runs on the NPU only; with --laya-model the router is still asked
        // (with npu its only candidate), so every device the engine serves goes through Laya.
        if (!o.laya_model.empty()) {
            onebit::laya::Scorer scorer;
            if (!scorer.load(o.laya_model)) throw std::runtime_error("laya: " + scorer.error());
            const std::string dev = onebit::laya::route_device(scorer, o.model, {"npu"});
            std::fprintf(stderr, "1bit serve: laya routes %s to %s (Q4NX model directory)\n", model_id(o).c_str(), dev.c_str());
        }
#endif
        // The NPU serves in process (unified.cpp).
        std::vector<std::string> args = {"-m", o.model, "-p", std::to_string(o.port), "--host", o.host};
        if (!o.alias.empty()) { args.push_back("--alias"); args.push_back(o.alias); }
        for (const auto& kv : o.npu_opts) { args.push_back("--opt"); args.push_back(kv); }
        std::vector<char*> av;
        for (auto& s : args) av.push_back(s.data());
        return run_unified(int(av.size()), av.data());
#else
        throw std::runtime_error("this build has no NPU lane (configure with -DONEBIT_NPU=ON -DONEBIT_NPU_RT_DIR=<NPU runtime>)");
#endif
    }
#ifdef ONEBIT_NPU
    if (fs::path(o.model).extension() == ".gguf" && o.device == "npu") {
        // Fused-lane first (opt-in via ONEBIT_NPU_LANE_DIR): pack the GGUF in the
        // lane's Q4NX layout and serve it through the fast lane (one launch per
        // layer, npu/lane.cpp) instead of the per-op forward (~100+ launches per
        // token).  The two layouts differ, so the lane gets its own pack and its
        // own cache dir.  Anything the lane cannot serve -- an architecture the
        // lane-layout packer does not model, a missing tokenizer, no kernel set --
        // returns "" and falls through to the unchanged per-op forward, so this
        // cannot regress a model the lane does not cover.
        {
            std::string lane;
            if (const char* env = std::getenv("ONEBIT_NPU_LANE_DIR")) lane = env;
            if (!lane.empty() && fs::exists(fs::path(lane) / "layer_ctx1.elf")) {
                const std::string ldir = repack_gguf_lane(o.model, lane);
                if (!ldir.empty()) {
                    std::error_code ec;
                    if (!fs::exists(fs::path(ldir) / "npu"))
                        fs::create_directory_symlink(lane, fs::path(ldir) / "npu", ec);
                    if (!ec && fs::exists(fs::path(ldir) / "npu" / "layer_ctx1.elf")) {
                        std::fprintf(stderr, "1bit serve: routing %s to the NPU fast lane (fused layer) from %s\n",
                                     o.model.c_str(), lane.c_str());
                        std::vector<std::string> args = {"-m", ldir, "-p", std::to_string(o.port), "--host", o.host};
                        if (!o.alias.empty()) { args.push_back("--alias"); args.push_back(o.alias); }
                        std::vector<char*> av;
                        for (auto& s : args) av.push_back(s.data());
                        return run_unified(int(av.size()), av.data());
                    }
                    if (ec) std::fprintf(stderr, "1bit serve: lane routing skipped (%s)\n", ec.message().c_str());
                }
            }
        }
        // Fused dx route (addons/npu-dx, one NPU launch per layer): pack the GGUF in
        // the dx (q4_1 tile) layout and serve it through the private dx route instead
        // of the per-op forward (~100+ launches per token).  The dx kernels are
        // shape-parameterised (layer1.elf per layer) and serve dense qwen2/qwen3 the
        // fast lane cannot.  Opt-in via ONEBIT_NPU_DX_KERNELS (or -DONEBIT_NPU_DX_KERNELS).
        // Anything the dx route cannot serve -- no kernel set, a mismatched q/k flavour,
        // a missing tokenizer -- returns "" and falls through to the per-op forward.
        {
            std::string dxk;
            if (const char* env = std::getenv("ONEBIT_NPU_DX_KERNELS")) dxk = env;
#ifdef ONEBIT_NPU_DX_KERNELS_DEFAULT
            if (dxk.empty()) dxk = ONEBIT_NPU_DX_KERNELS_DEFAULT;
#endif
            if (!dxk.empty()) {
                // The dx layout packer copies an external tokenizer.json (it does not
                // convert the GGUF's own vocabulary); the per-op forward's repack
                // produces one, so reuse that (cached) directory when none is set.
                if (!std::getenv("ONEBIT_NPU_LANE_TOKENIZER")) {
                    try {
                        const std::string fwd = repack_gguf(o.model);
                        if (fs::exists(fs::path(fwd) / "tokenizer.json"))
                            setenv("ONEBIT_NPU_LANE_TOKENIZER", fwd.c_str(), 0);
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "1bit serve: dx tokenizer reuse skipped (%s)\n", e.what());
                    }
                }
                const std::string ldir = repack_gguf_lane(o.model, dxk);
                if (!ldir.empty()) {
                    std::fprintf(stderr, "1bit serve: routing %s to the fused dx route (one launch per layer) from %s\n",
                                 o.model.c_str(), dxk.c_str());
                    std::vector<std::string> args = {"-m", ldir, "-p", std::to_string(o.port), "--host", o.host};
                    if (!o.alias.empty()) { args.push_back("--alias"); args.push_back(o.alias); }
                    std::vector<char*> av;
                    for (auto& s : args) av.push_back(s.data());
                    return run_unified(int(av.size()), av.data());
                }
            }
        }
        // GGUF-on-NPU (docs/npu.md, "Open"): repack the GGUF into a Q4NX model
        // directory, then serve it through the model-generic forward in process.
        const std::string dir = repack_gguf(o.model);
        // Resolve the full-ELF directory the forward loads. An empty value
        // makes it look for "/full_i8_*.elf", so it is an error, not a default.
        std::string kernels;
        if (const char* env = std::getenv("ONEBIT_NPU_KERNELS")) kernels = env;
        if (kernels.empty() && fs::is_directory(fs::path(dir) / "npu")) kernels = (fs::path(dir) / "npu").string();
#ifdef ONEBIT_NPU_KERNELS_DIR
        if (kernels.empty()) kernels = ONEBIT_NPU_KERNELS_DIR;
#endif
        if (kernels.empty() || !fs::is_directory(kernels)) {
            throw std::runtime_error(
                "no NPU kernel (full ELF) directory for " + o.model +
                ": set ONEBIT_NPU_KERNELS to a directory of full_i8_*.elf designs (docs/npu.md), or\n"
                "    configure the build with -DONEBIT_NPU_KERNELS_DIR=<dir>");
        }
        std::vector<std::string> args = {"-m", dir + "/model.q4nx", "--kernels", kernels,
                                         "-p", std::to_string(o.port), "--host", o.host};
        if (!o.alias.empty()) { args.push_back("--alias"); args.push_back(o.alias); }
        std::vector<char*> av;
        for (auto& s : args) av.push_back(s.data());
        return run_forward_serve(int(av.size()), av.data());
    }
#endif
    if (fs::path(o.model).extension() == ".gguf" || onebp_file(o.model)) return serve_child(o);
    throw std::runtime_error(o.model + ": expected an NPU model directory, a .gguf file or a 1BP package");
}

}  // namespace onebit
