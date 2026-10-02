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

// `1bit serve -m <model> [--port 8000] [--device auto|npu|vulkan|hrx|rocm|zinc] [--lean] [--mtp HEAD] [--adaptive]`
//                        [--npu-opt KEY=VALUE ...]
//
// The engine's one front door (docs/serve.md): one model per process behind an
// OpenAI-compatible API. It is how the engine runs inside Lemonade: Lemonade's
// onebit backend runs `1bit serve` the way it runs llama-server:
//
//   GET  /health, /v1/health     503 while the model loads, then 200
//   GET  /v1/models              the one model
//   POST /v1/chat/completions    (stream or not)
//   POST /v1/completions
//
// Which device runs the model follows from the model and --device:
//   an NPU model directory (model.q4nx + npu/)  -> the NPU fast lane, in process
//   a model a private NPU route serves          -> that route, in process, in builds with
//                                                  -DONEBIT_NPU_PRIVATE (docs/npu.md)
//   a .gguf, --device vulkan                    -> the upstream llama.cpp build's
//                                                  llama-server (else the HRX build's); an
//                                                  architecture only our llama.cpp has
//                                                  (ZAYA1) -> the HRX build's, on Vulkan0
//   a .gguf, --device hrx                       -> the HRX build's llama-server
//   a .gguf, --device vulkan --prefill-device hrx
//                                               -> the HRX build's llama-server on Vulkan0,
//                                                  long prompt prefixes on HRX0 over one
//                                                  shared KV cache (docs/hrx.md)
//   a .gguf, --device zinc                      -> this build's zinc
//   a .1bp (1BP v5 package), --device auto|ds4 -> the same ds4-server (our DwarfStar fork reads 1BP)
//   a DwarfStar .gguf, --device ds4             -> this build's DwarfStar ds4-server (DeepSeek
//                                                  V4/V4.1 Flash, GLM 5.x, Qwen3.8-Flash-Next in
//                                                  DwarfStar's own GGUFs; docs/dwarfstar.md)
//   a .gguf, --device rocm                      -> the ROCm build's llama-server on ROCm0
//                                                  (ONEBIT_LEAN_ROCM: any GGUF, and ROCmI4 with
//                                                  the gfx1151 W4A4 path; docs/lean.md)
//   a .gguf, --lean                             -> the lean build's llama-server (ROCmFPX's
//                                                  formats) on Vulkan0 (docs/lean.md)
//   --laya with --device auto                   -> Laya classifies each conversation and the route
//                                                  policy picks the device (RFC #186, docs/laya.md)
//   --mtp <head.gguf> on any llama.cpp route    -> multi-token prediction: the model's MTP
//                                                  head drafts, the model verifies (docs/serve.md)
//   --dflash <draft.gguf> on any llama.cpp route -> a DFlash block-diffusion draft model
//                                                  drafts a block per step (docs/serve.md)
//   --mmproj <mmproj.gguf> on any llama.cpp route -> images: the vision encoder runs beside the
//                                                  model, and chat messages may carry image parts
//   --long-model <rotated.gguf>                 -> conversations that start with --long-from tokens
//                                                  (default 2048) go to that Hadamard-rotated Q4_0
//                                                  on ROCm (W4A4 prefill), the rest to -m's route
//   a Hugging Face id, --device mlx (macOS)     -> lemon-mlx-engine's server
// For a .gguf, the engine starts that server as a private child on a loopback
// port and forwards the OpenAI routes to it, streaming included. `auto` picks
// Vulkan for GGUF (the fastest measured device for standard quants,
// docs/hrx.md) until the Laya router (docs/laya.md) makes that choice.
#include "serve.h"

#include "built_path.h"

#include "gguf_meta.h"
#include "recipes.h"

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
#include <memory>
#include <mutex>
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
    std::string llama_server, zinc, hrx_libhsa, mlx, ds4;
    bool ssd_streaming = false;  // --device ds4: stream routed experts from the SSD
    std::string prefill_device;
    int prefill_min_tokens = 0;
    bool lean = false;
    std::string mtp;
    std::string dflash;  // DFlash block-diffusion draft model for the target
    int moe_slots = 0;   // --moe-slots N: stream routed experts from the model file, N held in RAM (-1: auto)
    int moe_prefetch = 0;   // --moe-prefetch N: gate-ahead prefetch N MoE layers ahead (0: off)
    std::string moe_subst;  // --moe-subst R: a resident expert scoring at least R x a missing one takes its place
    std::string mmproj;  // a vision (or audio) projector: image parts in chat messages
    int mtp_max = 0;
    std::string mtp_p_min;
    int parallel = 0;
    bool adaptive = false;
    int adaptive_at = 1;
    std::string embed, rerank;   // RAG: an embedding model and a reranker, served beside the chat model
    std::string role;            // "embedding" or "reranking": the model itself is served in that role
    std::vector<std::string> npu_opts;   // --npu-opt KEY=VALUE, for a private NPU route
    std::string laya_model;              // --laya-model DIR: route each request by the Laya scorer (docs/laya.md)
    bool laya_auto = false;              // --laya: the same with the installed checkpoint (laya_checkpoint())
    std::string route_policy;            // --route-policy FILE: request class -> device (default: built in)
    std::string recipes;                 // --recipes FILE: tuned backend settings (default: built in, docs/recipes.md)
    bool no_recipes = false;             // --no-recipes: the backend gets serve's own flags only
    std::string long_model;              // --long-model FILE: a Hadamard-rotated Q4_0 for long prompts (W4A4 on ROCm)
    int long_from = 2048;                // --long-from N: a conversation whose first prompt has N tokens or more
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

// --device vulkan prefers the upstream llama.cpp build (ONEBIT_VULKAN, the
// latest release: new architectures land there first); --device hrx, and
// vulkan without that build, use the HRX build (AMD's tested pair).
std::string default_llama_server(const std::string& device) {
    if (const char* e = std::getenv("ONEBIT_LLAMA_SERVER"); e && *e) return e;
#ifdef ONEBIT_VULKAN_SERVER
    if (device == "vulkan") return built_path(ONEBIT_VULKAN_SERVER);
#endif
#ifdef ONEBIT_HRX_SERVER
    return built_path(ONEBIT_HRX_SERVER);
#else
    (void)device;
    return "llama-server";
#endif
}

// Architectures our llama.cpp (third_party/llama.cpp, the HRX build) implements and upstream's
// does not: a GGUF of one runs on that build's Vulkan0 even when the upstream build is present.
// A Q4_0 file from tools/hadamard_q4_0.py: its matmul weights are rotated by a 32-point
// Walsh-Hadamard transform. Two builds rotate the activations to match: our llama.cpp on HRX
// (src/llama-hadamard.cpp reads the stamp and adds the rotation as a matmul, fork #58) and the
// lean ROCm build (inside its quantizers, W4A4).
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

bool fork_only_arch(const std::string& gguf) {
    // Zyphra ZAYA1 (docs/vulkan.md); OPT, CodeGen, GPT-Neo and GPT-J (llama.cpp #8: upstream has
    // no model for them, gptj only a name). Keep in step with FORK_ONLY in tools/registry_build.py.
    static const char* const archs[] = {"zaya", "opt", "codegen", "gptneo", "gptj"};
    const std::string arch = gguf_architecture(gguf);
    for (const char* a : archs)
        if (arch == a) return true;
    return false;
}

// --lean: the llama-server built from ROCmFPX (ONEBIT_LEAN), whose formats upstream
// llama.cpp cannot read; --device rocm picks its ROCm build (ONEBIT_LEAN_ROCM).
std::string default_lean_server(const std::string& device) {
    if (const char* e = std::getenv("ONEBIT_LEAN_SERVER"); e && *e) return e;
    if (device == "rocm") {
#ifdef ONEBIT_LEAN_ROCM_SERVER
        return built_path(ONEBIT_LEAN_ROCM_SERVER);
#else
        throw std::runtime_error("--device rocm needs a build with -DONEBIT_LEAN=ON -DONEBIT_LEAN_ROCM=ON");
#endif
    }
#ifdef ONEBIT_LEAN_SERVER
    return built_path(ONEBIT_LEAN_SERVER);
#else
    throw std::runtime_error("--lean needs a build with -DONEBIT_LEAN=ON (docs/lean.md)");
#endif
}

// ryzenai-server (third_party/ryzenai-server, MIT): ONNX Runtime GenAI models, on the CPU with
// Microsoft's ONNX Runtime, or the NPU / hybrid where AMD's Ryzen AI Software provides it
std::string default_onnx() {
    if (const char* e = std::getenv("ONEBIT_ONNX_SERVER"); e && *e) return e;
#ifdef ONEBIT_ONNX_SERVER
    return built_path(ONEBIT_ONNX_SERVER);
#else
    return "ryzenai-server";
#endif
}

// an ONNX Runtime GenAI model directory (genai_config.json beside the graph)
bool is_onnx_model_dir(const std::string& dir) {
    return fs::is_directory(dir) && fs::exists(fs::path(dir) / "genai_config.json");
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

std::string default_zinc() {
    if (const char* e = std::getenv("ONEBIT_ZINC"); e && *e) return e;
#ifdef ONEBIT_ZINC_SERVER
    return built_path(ONEBIT_ZINC_SERVER);
#else
    return "zinc";
#endif
}

// SIGTERM / SIGINT: stop serving and take the backend down with us.
std::atomic<bool> g_stop{false};
extern "C" void on_stop_signal(int) { g_stop = true; }

std::string default_mlx() {
    if (const char* e = std::getenv("LEMONADE_MLX_SERVER"); e && *e) return e;
    return "lemon-mlx-server";
}

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

// A child OpenAI-compatible server on a loopback port.
#ifdef _WIN32
// Windows: CreateProcess into a job object that kills its processes when its last handle
// closes, so the backend dies with `1bit serve` for any reason (Linux: PR_SET_PDEATHSIG).
class Child {
public:
    Child(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int port,
          std::string ready_path = "/health")
        : port_(port), ready_path_(std::move(ready_path)) {
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
        for (const auto& e : env_extra) {
            const std::string key = e.substr(0, e.find('=') + 1);
            bool present = false;
            for (const auto& have : env_store) present = present || have.rfind(key, 0) == 0;
            if (!present) env_store.push_back(e);
        }
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

#else
class Child {
public:
    Child(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int port,
          std::string ready_path = "/health")
        : port_(port), ready_path_(std::move(ready_path)) {
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        std::vector<std::string> env_store;
        for (char** e = environ; *e; ++e) env_store.emplace_back(*e);
        for (const auto& e : env_extra) {
            const std::string key = e.substr(0, e.find('=') + 1);
            bool present = false;
            for (const auto& have : env_store) present = present || have.rfind(key, 0) == 0;
            if (!present) env_store.push_back(e);  // a value the user set wins
        }
        std::vector<char*> envp;
        for (auto& e : env_store) envp.push_back(e.data());
        envp.push_back(nullptr);
#if defined(__linux__)
        const pid_t parent = ::getpid();
        pid_ = ::fork();
        if (pid_ < 0) throw std::runtime_error("cannot fork for " + argv[0] + ": " + std::strerror(errno));
        if (pid_ == 0) {
            // The kernel ends the child when `1bit serve` dies for any reason,
            // SIGKILL included, so a backend is never left running on its own.
            ::prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (::getppid() != parent) ::_exit(127);  // parent already gone
            ::execvpe(args[0], args.data(), envp.data());
            ::_exit(127);
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

    void stop() {
        if (pid_ <= 0) return;
        ::kill(pid_, SIGTERM);
        for (int i = 0; i < 50; i++) {
            if (::waitpid(pid_, nullptr, WNOHANG) == pid_) { pid_ = -1; return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ::kill(pid_, SIGKILL);
        ::waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }

    bool alive() const { return pid_ > 0 && ::waitpid(pid_, nullptr, WNOHANG) == 0; }
#endif

    // Polls the child's ready path (/health; /v1/models for servers without one, which answer
    // only once the model is loaded) until it answers 200, it exits, or time runs out.
    bool wait_ready(std::chrono::seconds timeout, const std::atomic<bool>& stop) const {
        httplib::Client c("127.0.0.1", port_);
        c.set_connection_timeout(1);
        const auto end = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < end) {
            if (!alive() || stop) return false;
            if (auto r = c.Get(ready_path_); r && r->status == 200) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        return false;
    }

    int port() const { return port_; }

private:
#ifdef _WIN32
    HANDLE proc_ = nullptr, job_ = nullptr;
#else
    pid_t pid_ = -1;
#endif
    int port_;
    std::string ready_path_;
};

std::string model_id(const Options& o) {
    if (!o.alias.empty()) return o.alias;
    fs::path p(o.model);
    if (!p.has_filename()) p = p.parent_path();
    // Only a .gguf loses its extension: "Qwen3-0.6B-4bit" is a name, not a stem.
    return p.extension() == ".gguf" || p.extension() == ".1bp" ? p.stem().string() : p.filename().string();
}

// Forwards an OpenAI POST to the child: the model id becomes the child's (a
// child such as zinc rejects ids it did not load), and replies carry ours.
// Counts a request against a backend from the moment it is routed until its reply
// (streamed or not) is complete; --adaptive routes by these counts.
struct InFlight {
    explicit InFlight(std::atomic<int>* n) : n_(n) { if (n_) ++*n_; }
    ~InFlight() { if (n_) --*n_; }
    InFlight(const InFlight&) = delete;
    InFlight& operator=(const InFlight&) = delete;
private:
    std::atomic<int>* n_;
};

void forward(int port, std::shared_ptr<InFlight> held, int draft_max, const std::string& our_id, bool drop_model,
             const std::string& set_model, const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
        return;
    }
    if (drop_model) body.erase("model");
    // MTP by load: a request's own draft length wins; otherwise the router's (see post)
    if (draft_max >= 0 && !body.contains("speculative.n_max")) body["speculative.n_max"] = draft_max;
    if (!set_model.empty()) body["model"] = set_model;
    const bool stream = body.value("stream", false);
    const std::string payload = body.dump();
    const std::string path = req.path;
    if (!stream) {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(3600);
        auto r = c.Post(path, payload, "application/json");
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
    res.set_chunked_content_provider("text/event-stream", [port, path, payload, held](size_t, httplib::DataSink& sink) {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(3600);
        c.Post(path, httplib::Headers{}, payload, "application/json",
               [&sink](const char* data, size_t n) { return sink.write(data, n); });
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
};

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
int auto_moe_slots(const std::string& model) {
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

Launch launch_for(const Options& o, const std::string& device, int child_port) {
    Launch l;
    l.device = device;
    l.port = child_port;
    std::vector<std::string>& argv = l.argv;
    std::vector<std::string>& env = l.env;
    bool& drop_model = l.drop_model;
    std::string& set_model = l.set_model;
    if (o.ssd_streaming && device != "ds4") throw std::runtime_error("--ssd-streaming works with --device ds4");
    if (device == "onnx") {
        // ryzenai-server: `-m <dir> --port <p>`; the execution mode (CPU, NPU, hybrid) comes from
        // the model's genai_config.json
        if (o.parallel > 1 || !o.mtp.empty() || !o.dflash.empty() || o.lean || !o.prefill_device.empty() || !o.mmproj.empty())
            throw std::runtime_error("--device onnx takes no --parallel, --mtp, --dflash, --lean, --prefill-device or --mmproj");
        argv = {default_onnx(), "-m", o.model, "--port", std::to_string(child_port)};
        if (o.ctx_size > 0) { argv.push_back("--ctx-size"); argv.push_back(std::to_string(o.ctx_size)); }
        return l;
    }
    if (device == "mlx") {
        // lemon-mlx-engine: `<server> <hf id> --port <p>`; it picks the model by
        // Hugging Face id, so requests carry that id (docs/apple.md).
        argv = {o.mlx.empty() ? default_mlx() : o.mlx, o.model, "--port", std::to_string(child_port)};
        set_model = o.model;
    } else if (o.lean || device == "rocm") {
        // --lean: ROCmFPX's formats on Vulkan0. --device rocm: the ROCm build (ROCmFPX's tree,
        // which reads every GGUF) on ROCm0; ROCmI4 files use its W4A4 path there.
        if (device != "vulkan" && device != "rocm")
            throw std::runtime_error("--lean runs on --device vulkan or rocm");
        if (!o.prefill_device.empty()) throw std::runtime_error("--prefill-device works with --device vulkan only");
        argv = {o.llama_server.empty() ? default_lean_server(device) : o.llama_server,
                "-m", o.model, "--host", "127.0.0.1", "--port", std::to_string(child_port),
                "--device", device == "rocm" ? "ROCm0" : "Vulkan0", "-ngl", "99", "--jinja"};
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        if (hadamard_q4_0(o.model)) {
            // tools/hadamard_q4_0.py stamped it: its Q4_0 weights are rotated. The lean ROCm build sees
            // the stamp and rotates the activations of those weights, and of no others, so a plain
            // Q4_0 drafter or MTP head works beside it (ROCmFPX#7); every Q4_0 matmul takes the W4A4
            // kernel (docs/lean.md). A value set in the environment wins (GGML_W4A4_TENSORS= keeps
            // exact int8).
            if (device != "rocm") throw std::runtime_error(o.model + " is Hadamard-rotated: the lean build runs it on --device rocm");
            if (!std::getenv("GGML_W4A4_TENSORS")) env.push_back("GGML_W4A4_TENSORS=all");
            // micro-batch size: recipe rotated-moe-ub1024 (config/recipes.json)
        }
    } else if (device == "vulkan" || device == "hrx" || device == "cpu") {
        // --device cpu: llama-server with no GPU layers (the Windows package's route until HRX runs
        // there). --prefill-device hrx: the HRX build (it has both devices and the shared-KV split), flash
        // attention on so both devices lay the KV cache out the same way
        const bool split = !o.prefill_device.empty();
        if (split && (device != "vulkan" || o.prefill_device != "hrx"))
            throw std::runtime_error("--prefill-device hrx works with --device vulkan");
        const bool fork_arch = device == "vulkan" && fork_only_arch(o.model);
        argv = {o.llama_server.empty() ? default_llama_server(split || fork_arch ? "hrx" : device) : o.llama_server,
                "-m", o.model, "--host", "127.0.0.1", "--port", std::to_string(child_port)};
        if (device == "cpu") {
            if (split) throw std::runtime_error("--prefill-device hrx works with --device vulkan");
            argv.insert(argv.end(), {"-ngl", "0", "--jinja"});
        } else {
            argv.insert(argv.end(), {"--device", device == "hrx" ? "HRX0" : "Vulkan0", "-ngl", "99", "--jinja"});
        }
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        if (!o.flash_attn.empty() && !split) { argv.insert(argv.end(), {"-fa", o.flash_attn}); }
        if (split) {
            argv.insert(argv.end(), {"-fa", "on"});
            env.push_back("ONEBIT_PREFILL_DEVICE=HRX0");
            if (o.prefill_min_tokens > 0) env.push_back("ONEBIT_PREFILL_MIN_TOKENS=" + std::to_string(o.prefill_min_tokens));
        }
        if (o.moe_slots != 0) {
            // routed experts stay in the file (memory-mapped, never read whole); N of them are cached in
            // Vulkan buffers and streamed in as the router asks (docs/moe-streaming.md)
            if (device != "vulkan" || split || fork_arch)
                throw std::runtime_error("--moe-slots works with --device vulkan, without --prefill-device");
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
            // (a pinned Vulkan host buffer, or a CPU repack), which is what streaming avoids
            // (per_layer_token_embd: Qwen3.8-Flash-Next's 26.8 GiB table, only ever row-gathered)
            argv.insert(argv.end(), {"-ot", "exps=CPU,per_layer_token_embd=CPU", "--no-host", "--no-repack", "--load-mode", "mmap"});
        }
        if (device == "hrx" || split) {
            const std::string hsa = hrx_libhsa(o.hrx_libhsa);
            if (!hsa.empty()) env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
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
    } else if (device == "zinc") {
        argv = {o.zinc.empty() ? default_zinc() : o.zinc, "-m", o.model, "-p", std::to_string(child_port)};
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        env.push_back("RADV_PERFTEST=coop_matrix");
        drop_model = true;  // zinc rejects any model id but its own
    } else {
        throw std::runtime_error("--device " + o.device + " cannot run a .gguf (hrx, rocm, cpu, vulkan, zinc or ds4)");
    }
    if (o.parallel == 1) {
        // one slot: llama-server's own default is several, and a second slot can share a batch
        // (Qwen3.5 / Qwen3.8 on HRX0 cannot run a multi-sequence gated delta net yet)
        argv.insert(argv.end(), {"-np", "1"});
    }
    if (o.parallel > 1) {
        // continuous batching: N requests decode together, one read of the weights per step
        // for all of them; llama-server splits --ctx-size across the slots
        if (device == "zinc" || device == "mlx" || device == "ds4") throw std::runtime_error("--parallel works on the llama.cpp devices (vulkan, hrx, rocm)");
        argv.insert(argv.end(), {"-np", std::to_string(o.parallel)});
        if (device == "hrx") {
            // HRX runs attention on the GPU only with one KV stream: -kvu shares one cache across
            // the slots (per-slot streams are 4-D and its flash attention falls back to the CPU:
            // Qwen3-4B, 4 slots, 39 -> 153 tok/s). AMD's Qwen attention path (qwen.attention) builds
            // its mask from positions and ignores the other sequences, so under -kvu the answers
            // bleed into each other; it is turned off for these servers.
            const std::string arch = gguf_architecture(o.model);
            if (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next")
                throw std::runtime_error("--parallel on --device hrx does not run " + arch +
                                         " yet (no multi-sequence gated delta-net on HRX); use --device vulkan");
            argv.push_back("-kvu");
            bool merged = false;
            for (std::string& e : env)
                if (e.rfind("GGML_HRX_DISABLE_DISPATCH=", 0) == 0) { e += ",qwen.attention"; merged = true; }
            if (!merged) env.push_back("GGML_HRX_DISABLE_DISPATCH=qwen.attention");
        }
    }
    if (!o.mtp.empty() && !o.dflash.empty()) throw std::runtime_error("--mtp and --dflash are two drafters; pick one");
    if (!o.mtp.empty() || !o.dflash.empty()) {
        // the MTP head (--mtp) or a DFlash block-diffusion draft model (--dflash) drafts tokens
        // on the same device; the model checks them in one batch
        const char* flag = o.mtp.empty() ? "--dflash" : "--mtp";
        if (device == "zinc" || device == "mlx" || device == "ds4") throw std::runtime_error(std::string(flag) + " works on the llama.cpp devices (vulkan, hrx, rocm)");
        l.mtp = true;
        if (!o.mtp.empty()) argv.insert(argv.end(), {"--spec-type", "draft-mtp", "-md", o.mtp, "-ngld", "99"});
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
        if (device == "zinc" || device == "mlx" || device == "ds4") throw std::runtime_error("--mmproj works on the llama.cpp devices (vulkan, hrx, rocm)");
        // an image is decoded as one ubatch: models that attend to it bidirectionally (ZAYA1-VL,
        // Gemma 3) need all of it in one, and Qwen2.5-VL towers cap an image at 4096 tokens
        argv.insert(argv.end(), {"--mmproj", o.mmproj, "-b", "4096", "-ub", "4096"});
    }
    if ((device == "vulkan" || device == "hrx" || device == "rocm") && !o.no_recipes &&
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
        facts.hadamard_q4_0 = hadamard_q4_0(o.model);
        facts.drafter = !o.dflash.empty() ? "dflash" : !o.mtp.empty() ? "mtp" : "none";
        for (const auto& line : recipes.apply(facts, argv, env))
            std::fprintf(stderr, "1bit serve: recipe %s\n", line.c_str());
    }
    return l;
}

// A small llama-server for one RAG role on Vulkan: --embedding for /v1/embeddings,
// --reranking for /v1/rerank. Batch = ubatch: an embedding or rerank input is one pass.
Launch rag_launch(const Options& o, const std::string& model, const char* role, int port) {
    Launch l;
    l.device = std::string("vulkan ") + role;
    l.port = port;
    l.argv = {o.llama_server.empty() ? default_llama_server("vulkan") : o.llama_server, "-m", model, "--host", "127.0.0.1", "--port", std::to_string(port),
              "--device", "Vulkan0", "-ngl", "99", "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "4",
              std::string("--") + role};
    return l;
}

// The devices a .gguf can run on, in the order the Laya router offers them. The NPU runs Q4NX
// model directories, not .gguf files (docs/npu.md), so it is not a candidate here; an NPU
// model directory routes to npu in run_serve.
// Vulkan is leaving the engine (RFC #213, docs/hrx.md): HRX is the GPU route and the lean ROCm
// build serves long prompts, where this build has them. A build without HRX (CI) keeps the old
// candidates, Vulkan first, until the Vulkan build is removed.
std::vector<std::string> gguf_devices() {
    std::vector<std::string> devices;
#if defined(_WIN32)
    devices.push_back("cpu");
#elif defined(ONEBIT_HRX_SERVER)
    devices.push_back("hrx");
#ifdef ONEBIT_LEAN_ROCM_SERVER
    devices.push_back("rocm");
#endif
#else
    devices.insert(devices.end(), {"vulkan", "hrx"});
#endif
    devices.push_back("zinc");
    return devices;
}

// The GGUF architectures a pinned Vulkan llama.cpp maps and the HRX one does not
// (registry/architectures.json: "vulkan" in backends, "hrx" not). tests/device_route.sh checks
// this list against the registry, so regenerate both together.
bool hrx_missing_arch(const std::string& arch) {
    static const char* const archs[] = {
        "bailingmoe3", "blackmamba", "dots3note", "granite_swa", "graniteswitch", "hrm_text",
        "hy_v4",       "kimi-k3",    "maple",     "minimax-01",  "muse-glimmer",  "pockettts",
        "qwen3tts",    "qwen4exp",   "spark2_5",  "zamba",       "zamba2"};
    for (const char* a : archs)
        if (arch == a) return true;
    return false;
}

// --device auto for a .gguf: HRX where the build has it. What HRX does not run yet, or runs well
// below Vulkan, keeps Vulkan until it is ported or dropped: architectures HRX does not map
// (hrx_missing_arch), --moe-slots (experts streamed from the drive), --parallel on the gated
// delta-net architectures (no multi-sequence delta-net on HRX), --mtp (HRX drafts at 65-78% of
// Vulkan's speed) and --mmproj (an open RFC #213 gate).
std::string auto_gguf_device(const Options& o) {
#if defined(_WIN32)
    (void)o;
    return "cpu";  // no GPU route on Windows yet (docs/windows.md)
#elif defined(ONEBIT_HRX_SERVER)
    if (o.moe_slots != 0 || !o.mtp.empty() || !o.mmproj.empty()) return "vulkan";
    const std::string arch = gguf_architecture(o.model);
    if (hrx_missing_arch(arch)) return "vulkan";
    if (o.parallel > 1 && (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3next")) return "vulkan";
    return "hrx";
#else
    (void)o;
    return "vulkan";
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

std::string request_state(const httplib::Request& req) {
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
            if (!m.is_object() || !m.contains("content")) continue;
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
        }
    return out;
}

// The prompt length of a request, counted by a llama-server backend: /v1/completions' prompt as
// is, a chat through the model's template (/apply-template, which takes the request body as it
// comes, tools included). If the backend cannot tell, about four bytes a token.
long prompt_tokens(int port, const httplib::Request& q) {
    json body;
    try {
        body = json::parse(q.body);
    } catch (const std::exception&) {
        return long(q.body.size() / 4);
    }
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(std::chrono::seconds(30));
    std::string prompt;
    if (body.contains("prompt") && body["prompt"].is_string()) {
        prompt = body["prompt"].get<std::string>();
    } else if (auto t = c.Post("/apply-template", q.body, "application/json"); t && t->status == 200) {
        try {
            prompt = json::parse(t->body).value("prompt", "");
        } catch (const std::exception&) {
        }
    }
    if (prompt.empty()) return long(request_state(q).size() / 4);
    if (auto t = c.Post("/tokenize", json{{"content", prompt}}.dump(), "application/json"); t && t->status == 200) {
        try {
            const json tok = json::parse(t->body);
            if (tok.contains("tokens") && tok["tokens"].is_array()) return long(tok["tokens"].size());
        } catch (const std::exception&) {
        }
    }
    return long(prompt.size() / 4);
}

int serve_child(const Options& given) {
    Options o = given;
    if (hadamard_q4_0(o.model)) {
        // a rotated file has two routes: HRX (llama-hadamard rotates the activations as a matmul) and
        // the lean ROCm build (W4A4). auto keeps ROCm until HRX routes Q4_0 prompt matmuls to a fast
        // kernel: Qwen3.8-27B-Q4_0-H32 on HRX0 runs pp512 at 63.5 tok/s today (docs/hrx.md).
        if (o.device == "auto") o.device = "rocm";
        if (o.device != "rocm" && o.device != "hrx")
            throw std::runtime_error(o.model + " is Hadamard-rotated (tools/hadamard_q4_0.py): it runs on --device hrx or rocm, docs/hrx.md");
        if (o.laya_auto || !o.laya_model.empty())
            throw std::runtime_error("--laya routes among devices; a Hadamard-rotated file runs on hrx or rocm, picked with --device");
        if (o.device == "hrx")
            std::fprintf(stderr, "1bit serve: %s is Hadamard-rotated: HRX route, activations rotated by llama-hadamard\n", o.model.c_str());
        else
            std::fprintf(stderr, "1bit serve: %s is Hadamard-rotated: lean ROCm route, W4A4 prompt processing\n", o.model.c_str());
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
    std::vector<Launch> backends;
    // --laya / --laya-model with --device auto (RFC #186): Laya classifies each conversation
    // (code, prose, short, long_doc; laya/route.h) and the route policy (config/route-policy.json
    // or --route-policy) maps the class to one of the devices this build can run the .gguf on.
    // A device's backend starts when it is first picked (one model is not loaded on every
    // device at once). Opt-in: the built-in policy sends every class to Vulkan until a row
    // shows a measured net gain.
    const bool laya = (!o.laya_model.empty() || o.laya_auto) && o.device == "auto";
#ifdef ONEBIT_LAYA
    std::unique_ptr<onebit::laya::Scorer> scorer;
#if defined(__linux__)
    std::unique_ptr<LayaDaemon> laya_daemon;
#endif
    std::vector<std::string> laya_devices;
    onebit::laya::RoutePolicy policy;
    if (laya) {
        if (o.adaptive || !o.role.empty() || !o.prefill_device.empty() || o.lean)
            throw std::runtime_error("--laya does not combine with --adaptive, --embedding/--reranking, --prefill-device or --lean");
        policy = onebit::laya::load_route_policy(o.route_policy);
        // ONEBIT_LAYA_DEVICE=npu: the C++ scorer with its encoder on the NPU, from a private add-on
        // (laya/encoder.h, docs/laya.md); else the GGUF scorer, on HRX0 unless it names another
        const char* dev = std::getenv("ONEBIT_LAYA_DEVICE");
        const bool laya_npu = dev && std::string(dev) == "npu";
        const std::string gguf = laya_npu ? "" : laya_gguf(o.laya_model), bin = laya_npu ? "" : laya_ggml_bin();
#if defined(__linux__)
        if (!gguf.empty() && !bin.empty()) {
            // the GGUF scorer, with the HSA runtime HRX needs (docs/hrx.md)
            std::vector<std::string> env;
            if (const std::string hsa = hrx_libhsa(o.hrx_libhsa); !hsa.empty()) env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
            laya_daemon = std::make_unique<LayaDaemon>(std::vector<std::string>{
                bin, "daemon", gguf, "--family", "typed-decisions", "--device", dev && *dev ? dev : "hrx"}, env);
            std::fprintf(stderr, "1bit serve: laya: %s through %s, policy %s\n", gguf.c_str(), bin.c_str(),
                         o.route_policy.empty() ? "built in" : o.route_policy.c_str());
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
            std::fprintf(stderr, "1bit serve: laya: checkpoint %s%s, policy %s\n", ckpt.c_str(),
                         laya_npu ? " (encoder on the NPU)" : "", o.route_policy.empty() ? "built in" : o.route_policy.c_str());
        }
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
#else
    if (laya) throw std::runtime_error("--laya-model is not part of this build (-DONEBIT_LAYA=ON, docs/laya.md)");
#endif
    if (laya) {
        // the candidates are prepared above
    } else if (o.adaptive) {
        // Grows with the load: a lone request gets Vulkan (MTP if given, --adaptive-at slots,
        // default 1); concurrent ones go to ROCm, which keeps scaling to 16 batched requests.
        // MTP and batching live in separate backends: a server with MTP loaded batches at
        // about two thirds of the throughput, whatever each request's draft length
        // (docs/serve.md, "Growing with the load").
        if (o.device != "auto" && o.device != "vulkan")
            throw std::runtime_error("--adaptive runs Vulkan with ROCm overflow; leave --device at auto or vulkan");
        if (o.lean || !o.prefill_device.empty()) throw std::runtime_error("--adaptive does not combine with --lean or --prefill-device");
        Options first = o;
        first.parallel = o.parallel > 1 ? o.parallel : o.adaptive_at;
        backends.push_back(launch_for(first, "vulkan", free_port()));
        Options overflow = o;
        overflow.parallel = 16;
        overflow.mtp.clear();       // batched requests gain little from drafting
        overflow.dflash.clear();
        overflow.llama_server.clear();
        backends.push_back(launch_for(overflow, "rocm", free_port()));
    } else if (!o.role.empty()) {
        // The model is an embedding or reranking model (Lemonade loads them as models of their
        // own): llama-server on the device asked for, in that role. An input is one pass, so
        // the batch covers the context.
        const std::string dev = o.device == "auto" ? "vulkan" : o.device;
        if (dev != "vulkan" && dev != "hrx" && dev != "rocm")
            throw std::runtime_error("--" + o.role + " runs on llama-server devices (vulkan, hrx, rocm), not " + dev);
        if (o.adaptive || !o.mtp.empty() || !o.dflash.empty() || !o.prefill_device.empty())
            throw std::runtime_error("--" + o.role + " does not combine with --adaptive, --mtp, --dflash or --prefill-device");
        Launch l = launch_for(o, dev, free_port());
        const std::string batch = std::to_string(o.ctx_size > 0 ? o.ctx_size : 8192);
        for (const std::string& a : {std::string("--") + o.role, std::string("-b"), batch, std::string("-ub"), batch})
            l.argv.push_back(a);
        l.device = dev + " " + o.role;
        backends.push_back(std::move(l));
    } else {
        backends.push_back(launch_for(o, o.device == "auto" ? auto_gguf_device(o) : o.device, free_port()));
    }
    // --long-model: the same model as a Hadamard-rotated Q4_0 on the ROCm W4A4 route, for
    // conversations that start with a long prompt. What the user waits for is the first token
    // plus the answer: with a short prompt the faster decode wins (Vulkan), with a long one the
    // faster prefill does (W4A4 from about 2,000 tokens; docs/lean.md, "End to end"). Both
    // backends load at the start; a conversation stays where its first request went, so its
    // prompt cache stays useful as it grows.
    const bool long_route = !o.long_model.empty();
    if (long_route) {
        if (laya || o.adaptive || !o.role.empty() || !o.mmproj.empty())
            throw std::runtime_error("--long-model does not combine with --laya, --adaptive, --embedding/--reranking or --mmproj");
        if (!hadamard_q4_0(o.long_model))
            throw std::runtime_error("--long-model takes a Hadamard-rotated Q4_0 (tools/hadamard_q4_0.py, docs/lean.md): " + o.long_model);
        if (hadamard_q4_0(o.model))
            throw std::runtime_error(o.model + " is already on the W4A4 route; -m is the model for short prompts");
        const std::string& d = backends[0].device;
        if (d != "vulkan" && d != "hrx" && d != "rocm")
            throw std::runtime_error("--long-model needs -m on a llama-server route (vulkan, hrx or rocm), not " + d);
        if (o.long_from < 1) throw std::runtime_error("--long-from takes a positive token count");
        Options lo = o;
        lo.model = o.long_model;
        lo.mtp.clear();
        lo.dflash.clear();
        lo.llama_server.clear();
        lo.prefill_device.clear();
        lo.moe_slots = 0;
        lo.lean = false;
        backends.push_back(launch_for(lo, "rocm", free_port()));
    }
    const std::string device = laya ? "laya" : o.adaptive ? "vulkan+rocm" : backends[0].device;
    const bool drop_model = backends[0].drop_model;
    const std::string set_model = backends[0].set_model;

    std::vector<Launch> rag;   // [0] embedding, then rerank, if given
    if (!o.embed.empty()) rag.push_back(rag_launch(o, o.embed, "embedding", free_port()));
    if (!o.rerank.empty()) rag.push_back(rag_launch(o, o.rerank, "reranking", free_port()));
    auto stem = [](const std::string& m) { return fs::path(m).stem().string(); };

    const std::string id = model_id(o);
    std::atomic<bool> ready{false};

    httplib::Server srv;
    auto health = [&](const httplib::Request&, httplib::Response& res) {
        res.status = ready ? 200 : 503;
        res.set_content(ready ? R"({"status":"ok"})" : R"({"status":"loading"})", "application/json");
    };
    srv.Get("/health", health);
    srv.Get("/v1/health", health);
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array({{{"id", id}, {"object", "model"}, {"owned_by", "1bit"}, {"device", device}}});
        if (!o.embed.empty()) data.push_back({{"id", stem(o.embed)}, {"object", "model"}, {"owned_by", "1bit"}, {"device", "vulkan"}, {"role", "embedding"}});
        if (!o.rerank.empty()) data.push_back({{"id", stem(o.rerank)}, {"object", "model"}, {"owned_by", "1bit"}, {"device", "vulkan"}, {"role", "rerank"}});
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });
    std::vector<std::unique_ptr<Child>> children;
    // laya: the backends by index, started on first use (under start_mu; a start waits for
    // the backend to be ready, up to 600 s)
    std::vector<Child*> started(backends.size(), nullptr);
    std::mutex start_mu;
    auto ensure = [&](size_t i) -> bool {
        std::lock_guard<std::mutex> lock(start_mu);
        if (started[i]) return true;
        const Launch& b = backends[i];
        std::fprintf(stderr, "1bit serve: %s on %s (%s), picked by laya\n", id.c_str(), b.device.c_str(), b.argv[0].c_str());
        auto c = std::make_unique<Child>(b.argv, b.env, b.port, b.ready_path);
        if (!c->wait_ready(std::chrono::seconds(600), g_stop)) return false;
        started[i] = c.get();
        children.push_back(std::move(c));
        return true;
    };
    std::vector<std::atomic<int>> inflight(backends.size());
    std::vector<std::atomic<long>> routed(backends.size());
    std::mutex route_mu;
    // --long-model: conversation -> (backend, prompt tokens of its first request)
    std::mutex long_mu;
    std::unordered_map<size_t, std::pair<size_t, long>> long_cache;
    std::deque<size_t> long_order;
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
                if (laya_daemon) cls = classify_request_gguf(*laya_daemon, request_state(q));
                else
#endif
                cls = onebit::laya::classify_request(*scorer, request_state(q));
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
            forward(backends[pick].port, held, -1, id, backends[pick].drop_model, backends[pick].set_model, q, r);
            r.set_header("X-1bit-Route", route_hdr);  // class, Laya confidence, device
            return;
        }
#endif
        if (long_route) {
            const size_t key = conversation_key(q);
            std::pair<size_t, long> side{0, -1};
            bool fresh = true;
            {
                std::lock_guard<std::mutex> lock(long_mu);
                if (const auto c = long_cache.find(key); c != long_cache.end()) side = c->second, fresh = false;
            }
            if (fresh) {
                side.second = prompt_tokens(backends[0].port, q);
                side.first = side.second >= o.long_from ? 1 : 0;
                std::lock_guard<std::mutex> lock(long_mu);
                if (long_cache.size() >= 4096) {  // oldest conversation first
                    long_cache.erase(long_order.front());
                    long_order.pop_front();
                }
                if (long_cache.emplace(key, side).second) long_order.push_back(key);
            }
            const size_t i = side.first;
            char route_hdr[96];
            std::snprintf(route_hdr, sizeof route_hdr, "%s %ld %s", i ? "long" : "short", side.second, backends[i].device.c_str());
            held = std::make_shared<InFlight>(&inflight[i]);
            ++routed[i];
            forward(backends[i].port, held, -1, id, backends[i].drop_model, backends[i].set_model, q, r);
            r.set_header("X-1bit-Route", route_hdr);  // short or long, the first prompt's tokens, device
            return;
        }
        {
            std::lock_guard<std::mutex> lock(route_mu);
            if (backends.size() > 1 && inflight[0] >= o.adaptive_at) {
                pick = 1;
                for (size_t i = 2; i < backends.size(); i++)
                    if (inflight[i] < inflight[pick]) pick = i;
            }
            held = std::make_shared<InFlight>(&inflight[pick]);
        }
        ++routed[pick];
        forward(backends[pick].port, held, -1, id, drop_model, set_model, q, r);
    };
    srv.Post("/v1/chat/completions", post);
    srv.Post("/v1/completions", post);
    // The rest of llama-server's API, for the devices that run llama-server (Lemonade's
    // llamacpp backend, which the onebit recipe inherits, forwards these to us). They go to
    // the first backend; the NPU, ZINC and MLX answer 501.
    const bool llama = device == "vulkan" || device == "hrx" || device == "rocm" || device == "cpu" || device == "vulkan+rocm" ||
                       (laya && backends[0].device == "vulkan");
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
        forward(backends[0].port, nullptr, -1, id, drop_model, set_model, q, r);
    };
    for (const char* path : {"/v1/responses", "/tokenize", "/detokenize", "/apply-template"})
        srv.Post(path, llama ? httplib::Server::Handler(first_json) : httplib::Server::Handler(unsupported));
    for (const char* path : {"/slots", "/props", "/metrics"})
        srv.Get(path, llama ? httplib::Server::Handler(first) : httplib::Server::Handler(unsupported));
    srv.Post(R"(/slots/(\d+))", llama ? httplib::Server::Handler(first) : httplib::Server::Handler(unsupported));
    // RAG: embeddings and rerank go to their own backends, the reply names the model served
    auto rag_route = [&](size_t i, const std::string& model) {
        return [&, i, model](const httplib::Request& q, httplib::Response& r) {
            if (!ready) {
                r.status = 503;
                r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
                return;
            }
            forward(rag[i].port, nullptr, -1, stem(model), true, "", q, r);
        };
    };
    if (!o.role.empty()) {
        auto h = [&](const httplib::Request& q, httplib::Response& r) {
            if (!ready) {
                r.status = 503;
                r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
                return;
            }
            forward(backends[0].port, nullptr, -1, id, true, "", q, r);
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
        children.push_back(std::make_unique<Child>(b.argv, b.env, b.port, b.ready_path));
    }
    for (const auto& b : rag) {
        std::fprintf(stderr, "1bit serve: %s on %s (%s)\n", b.argv[2].c_str(), b.device.c_str(), b.argv[0].c_str());
        children.push_back(std::make_unique<Child>(b.argv, b.env, b.port));
    }
    for (size_t i = 0; i < children.size(); i++) {
        if (!children[i]->wait_ready(std::chrono::seconds(600), g_stop)) {
            if (g_stop) return 0;
            std::fprintf(stderr, "1bit serve: backend %zu did not become ready\n", i);
            return 1;
        }
    }
    ready = true;
    std::fprintf(stderr, "1bit serve: ready on http://%s:%d%s\n", o.host.c_str(), o.port,
                 laya ? " (laya routes each request; a device's backend starts when first picked)" : "");
    auto all_alive = [&] {
        std::lock_guard<std::mutex> lock(start_mu);
        for (auto& c : children) if (!c->alive()) return false;
        return true;
    };
    while (all_alive() && !g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
    std::fprintf(stderr, "1bit serve: a %s backend exited\n", device.c_str());
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

// Repack a GGUF into a Q4NX model directory with the FLM_Q4NX_Converter
// (Python), cached by GGUF stem so a repeated serve reuses the conversion.
// Override the tool paths with ONEBIT_Q4NX_PYTHON / ONEBIT_Q4NX_CONVERTER and
// the cache root with ONEBIT_Q4NX_CACHE.
std::string repack_gguf(const std::string& gguf) {
    const std::string stem = fs::path(gguf).stem().string();
    std::string cache = std::getenv("ONEBIT_Q4NX_CACHE") ? std::getenv("ONEBIT_Q4NX_CACHE") : "";
    if (cache.empty()) {
        const char* home = std::getenv("HOME");
        cache = (home ? std::string(home) : std::string("/tmp")) + "/.cache/1bit/q4nx";
    }
    const std::string out = cache + "/" + stem;

    const char* py = std::getenv("ONEBIT_Q4NX_PYTHON");
    const std::string python = py ? py : "python3";
    const char* rp = std::getenv("ONEBIT_Q4NX_REPACK");
#ifdef ONEBIT_NPU_REPACK_SCRIPT
    const std::string script = rp ? rp : ONEBIT_NPU_REPACK_SCRIPT;
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
    std::string cache = std::getenv("ONEBIT_Q4NX_CACHE") ? std::getenv("ONEBIT_Q4NX_CACHE") : "";
    if (cache.empty()) {
        const char* home = std::getenv("HOME");
        cache = (home ? std::string(home) : std::string("/tmp")) + "/.cache/1bit/q4nx";
    }
    const std::string out = cache + "/" + stem + "-lane";

    const char* py = std::getenv("ONEBIT_Q4NX_PYTHON");
    const std::string python = py ? py : "python3";
    const char* lp = std::getenv("ONEBIT_Q4NX_LANE_PACK");
#ifdef ONEBIT_NPU_LANE_PACK_SCRIPT
    const std::string script = lp ? lp : ONEBIT_NPU_LANE_PACK_SCRIPT;
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
                 "                  [--device auto|npu|hrx|rocm|cpu|vulkan|zinc|ds4|mlx|onnx] [--ctx-size N] [--alias NAME]\n"
                 "                  [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]\n"
                 "                  [--ds4 PATH] [--ssd-streaming]   DwarfStar (--device ds4; docs/dwarfstar.md)\n"
                 "                  [--prefill-device hrx] [--prefill-min-tokens N]   (with --device vulkan)\n"
                 "                  [--lean]   ROCmFPX formats: ROCmFP4 on vulkan, ROCmI4 with --device rocm\n"
                 "                  [--mtp HEAD.gguf] [--mtp-max N] [--mtp-p-min P]   multi-token prediction (vulkan, hrx, rocm)\n"
                 "                  [--dflash DRAFT.gguf]   DFlash draft model instead of --mtp; --mtp-max/--mtp-p-min apply\n"
                 "                  [--moe-slots N|auto] [--moe-subst R] [--moe-prefetch N]   stream MoE experts from the file,\n"
                 "                                    N held in RAM (auto: what fits); R: resident experts stand in for missing\n"
                 "                                    ones; prefetch N layers ahead (default 0) (vulkan; docs/moe-streaming.md)\n"
                 "                  [--mmproj MMPROJ.gguf]   images in chat messages (vulkan, hrx, rocm)\n"
                 "                  [--parallel N]   N requests decoded together (continuous batching)\n"
                 "                  [--adaptive] [--adaptive-at N]   Vulkan (+MTP) for N in flight (default 1), ROCm batches the rest\n"
                 "                  [--long-model ROTATED.gguf] [--long-from N]   conversations starting with N or more prompt\n"
                 "                                    tokens (default 2048) go to that Hadamard-rotated Q4_0 on ROCm (W4A4)\n"
                 "                  [--embed MODEL.gguf] [--rerank MODEL.gguf]   RAG: /v1/embeddings and /v1/rerank\n"
                 "                  [--embedding | --reranking]   serve the model itself as an embedding or reranking model\n"
                 "                  [--npu-opt KEY=VALUE ...]   an option for a private NPU route (docs/npu.md)\n"
                 "                  [--laya | --laya-model DIR] [--route-policy FILE]   with --device auto, Laya classifies\n"
                 "                  each conversation and the route policy picks its device (docs/laya.md)\n"
                 "                  [--recipes FILE | --no-recipes]   tuned llama-server settings per model and route\n"
                 "                  (default: config/recipes.json, built in; docs/recipes.md)\n"
                 "  <model>: an NPU model directory (model.q4nx + npu/, or one a private NPU route serves),\n"
                 "           an ONNX Runtime GenAI directory (genai_config.json: --device onnx),\n"
                 "           a .gguf file, or with --device mlx a Hugging Face id (mlx-community/...)\n");
}

}  // namespace

int run_serve(int argc, char** argv) {
    Options o;
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
        else if (a == "--zinc") o.zinc = next();
        else if (a == "--ds4") o.ds4 = next();
        else if (a == "--ssd-streaming") o.ssd_streaming = true;
        else if (a == "--hrx-libhsa") o.hrx_libhsa = next();
        else if (a == "--mlx-server") o.mlx = next();
        else if (a == "--prefill-device") o.prefill_device = next();
        else if (a == "--prefill-min-tokens") o.prefill_min_tokens = std::stoi(next());
        else if (a == "--lean") o.lean = true;
        else if (a == "--mtp") o.mtp = next();
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
        else if (a == "--adaptive") o.adaptive = true;
        else if (a == "--adaptive-at") o.adaptive_at = std::stoi(next());
        else if (a == "--long-model") o.long_model = next();
        else if (a == "--long-from") o.long_from = std::stoi(next());
        else if (a == "--embed") o.embed = next();
        else if (a == "--embedding" || a == "--embeddings") o.role = "embedding";
        else if (a == "--reranking") o.role = "reranking";
        else if (a == "--rerank") o.rerank = next();
        else if (a == "--npu-opt") o.npu_opts.push_back(next());
        else if (a == "--laya-model") o.laya_model = next();
        else if (a == "--laya") o.laya_auto = true;
        else if (a == "--route-policy") o.route_policy = next();
        else if (a == "--recipes") o.recipes = next();
        else if (a == "--no-recipes") o.no_recipes = true;
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty()) { usage(stderr); return 2; }

    if (is_onnx_model_dir(o.model)) {
        if (o.device != "auto" && o.device != "onnx")
            throw std::runtime_error("an ONNX Runtime GenAI model directory (genai_config.json) runs on --device onnx");
        Options onnx = o;
        onnx.device = "onnx";
        return serve_child(onnx);
    }
    if (fs::is_directory(o.model)) {
        if (o.lean) throw std::runtime_error("--lean serves a .gguf (ROCmFP4 or ROCmI4)");
        if (!o.mmproj.empty()) throw std::runtime_error("--mmproj works on the llama.cpp devices (vulkan, hrx, rocm)");
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
        throw std::runtime_error("this build has no NPU lane (configure with -DONEBIT_NPU=ON)");
#endif
    }
    if (o.device == "mlx") return serve_child(o);
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
        // Resolve the full-ELF / xclbin directory the forward loads. An empty value
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
