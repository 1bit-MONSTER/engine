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

// `1bit serve -m <model> [--port 8000] [--device auto|npu|vulkan|hrx|rocm|zinc] [--lean] [--mtp HEAD]`
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
//   a .gguf, --device vulkan                    -> the upstream llama.cpp build's
//                                                  llama-server (else the HRX build's)
//   a .gguf, --device hrx                       -> the HRX build's llama-server
//   a .gguf, --device vulkan --prefill-device hrx
//                                               -> the HRX build's llama-server on Vulkan0,
//                                                  long prompt prefixes on HRX0 over one
//                                                  shared KV cache (docs/hrx.md)
//   a .gguf, --device zinc                      -> this build's zinc
//   a .gguf, --device rocm                      -> the ROCm build's llama-server on ROCm0
//                                                  (ONEBIT_LEAN_ROCM: any GGUF, and ROCmI4 with
//                                                  the gfx1151 W4A4 path; docs/lean.md)
//   a .gguf, --lean                             -> the lean build's llama-server (ROCmFPX's
//                                                  formats) on Vulkan0 (docs/lean.md)
//   --mtp <head.gguf> on any llama.cpp route    -> multi-token prediction: the model's MTP
//                                                  head drafts, the model verifies (docs/serve.md)
//   a Hugging Face id, --device mlx (macOS)     -> lemon-mlx-engine's server
// For a .gguf, the engine starts that server as a private child on a loopback
// port and forwards the OpenAI routes to it, streaming included. `auto` picks
// Vulkan for GGUF (the fastest measured device for standard quants,
// docs/hrx.md) until the Laya router (docs/laya.md) makes that choice.
#include "serve.h"

#ifdef ONEBIT_NPU
#include "unified.h"
#endif

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#if defined(__linux__)
#include <sys/prctl.h>
#else
#include <spawn.h>
#endif
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace onebit {

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Options {
    std::string model, host = "127.0.0.1", device = "auto", alias;
    int port = 8000, ctx_size = 0;
    std::string llama_server, zinc, hrx_libhsa, mlx;
    std::string prefill_device;
    int prefill_min_tokens = 0;
    bool lean = false;
    std::string mtp;
    int mtp_max = 0;
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
    if (device == "vulkan") return ONEBIT_VULKAN_SERVER;
#endif
#ifdef ONEBIT_HRX_SERVER
    return ONEBIT_HRX_SERVER;
#else
    (void)device;
    return "llama-server";
#endif
}

// --lean: the llama-server built from ROCmFPX (ONEBIT_LEAN), whose formats upstream
// llama.cpp cannot read; --device rocm picks its ROCm build (ONEBIT_LEAN_ROCM).
std::string default_lean_server(const std::string& device) {
    if (const char* e = std::getenv("ONEBIT_LEAN_SERVER"); e && *e) return e;
    if (device == "rocm") {
#ifdef ONEBIT_LEAN_ROCM_SERVER
        return ONEBIT_LEAN_ROCM_SERVER;
#else
        throw std::runtime_error("--device rocm needs a build with -DONEBIT_LEAN=ON -DONEBIT_LEAN_ROCM=ON");
#endif
    }
#ifdef ONEBIT_LEAN_SERVER
    return ONEBIT_LEAN_SERVER;
#else
    throw std::runtime_error("--lean needs a build with -DONEBIT_LEAN=ON (docs/lean.md)");
#endif
}

std::string default_zinc() {
    if (const char* e = std::getenv("ONEBIT_ZINC"); e && *e) return e;
#ifdef ONEBIT_ZINC_SERVER
    return ONEBIT_ZINC_SERVER;
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
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    socklen_t len = sizeof(a);
    if (s < 0 || ::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
        ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len) != 0) {
        if (s >= 0) ::close(s);
        throw std::runtime_error("cannot find a free loopback port");
    }
    const int port = ntohs(a.sin_port);
    ::close(s);
    return port;
}

// A child OpenAI-compatible server on a loopback port.
class Child {
public:
    Child(const std::vector<std::string>& argv, const std::vector<std::string>& env_extra, int port)
        : port_(port) {
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

    // Polls the child's /health until it answers 200, it exits, or time runs out.
    bool wait_ready(std::chrono::seconds timeout, const std::atomic<bool>& stop) const {
        httplib::Client c("127.0.0.1", port_);
        c.set_connection_timeout(1);
        const auto end = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < end) {
            if (!alive() || stop) return false;
            if (auto r = c.Get("/health"); r && r->status == 200) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        return false;
    }

    int port() const { return port_; }

private:
    pid_t pid_ = -1;
    int port_;
};

std::string model_id(const Options& o) {
    if (!o.alias.empty()) return o.alias;
    fs::path p(o.model);
    if (!p.has_filename()) p = p.parent_path();
    // Only a .gguf loses its extension: "Qwen3-0.6B-4bit" is a name, not a stem.
    return p.extension() == ".gguf" ? p.stem().string() : p.filename().string();
}

// Forwards an OpenAI POST to the child: the model id becomes the child's (a
// child such as zinc rejects ids it did not load), and replies carry ours.
void forward(const Child& child, const std::string& our_id, bool drop_model, const std::string& set_model,
             const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
        return;
    }
    if (drop_model) body.erase("model");
    if (!set_model.empty()) body["model"] = set_model;
    const bool stream = body.value("stream", false);
    const std::string payload = body.dump();
    const std::string path = req.path;
    const int port = child.port();
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
    res.set_chunked_content_provider("text/event-stream", [port, path, payload](size_t, httplib::DataSink& sink) {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(3600);
        c.Post(path, httplib::Headers{}, payload, "application/json",
               [&sink](const char* data, size_t n) { return sink.write(data, n); });
        sink.done();
        return true;
    });
}

int serve_child(const Options& o) {
    std::string device = o.device == "auto" ? "vulkan" : o.device;
    const int child_port = free_port();
    std::vector<std::string> argv, env;
    bool drop_model = false;
    std::string set_model;
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
    } else if (device == "vulkan" || device == "hrx") {
        // --prefill-device hrx: the HRX build (it has both devices and the shared-KV split), flash
        // attention on so both devices lay the KV cache out the same way
        const bool split = !o.prefill_device.empty();
        if (split && (device != "vulkan" || o.prefill_device != "hrx"))
            throw std::runtime_error("--prefill-device hrx works with --device vulkan");
        argv = {o.llama_server.empty() ? default_llama_server(split ? "hrx" : device) : o.llama_server,
                "-m", o.model, "--host", "127.0.0.1", "--port", std::to_string(child_port),
                "--device", device == "hrx" ? "HRX0" : "Vulkan0", "-ngl", "99", "--jinja"};
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        if (split) {
            argv.insert(argv.end(), {"-fa", "on"});
            env.push_back("ONEBIT_PREFILL_DEVICE=HRX0");
            if (o.prefill_min_tokens > 0) env.push_back("ONEBIT_PREFILL_MIN_TOKENS=" + std::to_string(o.prefill_min_tokens));
        }
        if (device == "hrx" || split) {
            const std::string hsa = hrx_libhsa(o.hrx_libhsa);
            if (!hsa.empty()) env.push_back("IREE_HAL_AMDGPU_LIBHSA_PATH=" + hsa);
        }
    } else if (device == "zinc") {
        argv = {o.zinc.empty() ? default_zinc() : o.zinc, "-m", o.model, "-p", std::to_string(child_port)};
        if (o.ctx_size > 0) { argv.push_back("-c"); argv.push_back(std::to_string(o.ctx_size)); }
        env.push_back("RADV_PERFTEST=coop_matrix");
        drop_model = true;  // zinc rejects any model id but its own
    } else {
        throw std::runtime_error("--device " + o.device + " cannot run a .gguf (vulkan, hrx, rocm or zinc)");
    }
    if (!o.mtp.empty()) {
        // the MTP head drafts tokens on the same device; the model checks them in one batch
        if (device == "zinc" || device == "mlx") throw std::runtime_error("--mtp works on the llama.cpp devices (vulkan, hrx, rocm)");
        argv.insert(argv.end(), {"--spec-type", "draft-mtp", "-md", o.mtp, "-ngld", "99"});
        if (o.mtp_max > 0) { argv.push_back("--spec-draft-n-max"); argv.push_back(std::to_string(o.mtp_max)); }
    }

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
        res.set_content(json{{"object", "list"},
                             {"data", json::array({{{"id", id}, {"object", "model"}, {"owned_by", "1bit"},
                                                    {"device", device}}})}}.dump(),
                        "application/json");
    });
    std::unique_ptr<Child> child;
    auto post = [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready || !child) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        forward(*child, id, drop_model, set_model, q, r);
    };
    srv.Post("/v1/chat/completions", post);
    srv.Post("/v1/completions", post);

    if (!srv.bind_to_port(o.host, o.port))
        throw std::runtime_error("cannot listen on " + o.host + ":" + std::to_string(o.port));
    std::thread listener([&] { srv.listen_after_bind(); });
    struct Stop {
        httplib::Server& s;
        std::thread& t;
        ~Stop() { s.stop(); if (t.joinable()) t.join(); }
    } stop{srv, listener};

    struct sigaction sa{};
    sa.sa_handler = on_stop_signal;
    ::sigaction(SIGTERM, &sa, nullptr);
    ::sigaction(SIGINT, &sa, nullptr);
    std::fprintf(stderr, "1bit serve: %s on %s (%s)\n", id.c_str(), device.c_str(), argv[0].c_str());
    child = std::make_unique<Child>(argv, env, child_port);
    if (!child->wait_ready(std::chrono::seconds(600), g_stop)) {
        if (g_stop) return 0;
        std::fprintf(stderr, "1bit serve: %s did not become ready\n", argv[0].c_str());
        return 1;
    }
    ready = true;
    std::fprintf(stderr, "1bit serve: ready on http://%s:%d\n", o.host.c_str(), o.port);
    while (child->alive() && !g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (g_stop) {
        std::fprintf(stderr, "1bit serve: stopping\n");
        child->stop();
        return 0;
    }
    std::fprintf(stderr, "1bit serve: the %s backend exited\n", device.c_str());
    return 1;
}

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit serve -m <model> [--port 8000] [--host 127.0.0.1]\n"
                 "                  [--device auto|npu|vulkan|hrx|rocm|zinc|mlx] [--ctx-size N] [--alias NAME]\n"
                 "                  [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]\n"
                 "                  [--prefill-device hrx] [--prefill-min-tokens N]   (with --device vulkan)\n"
                 "                  [--lean]   ROCmFPX formats: ROCmFP4 on vulkan, ROCmI4 with --device rocm\n"
                 "                  [--mtp HEAD.gguf] [--mtp-max N]   multi-token prediction (vulkan, hrx, rocm)\n"
                 "  <model>: an NPU model directory (model.q4nx + npu/), a .gguf file, or with\n"
                 "           --device mlx a Hugging Face id (mlx-community/...)\n");
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
        else if (a == "--alias") o.alias = next();
        else if (a == "--llama-server") o.llama_server = next();
        else if (a == "--zinc") o.zinc = next();
        else if (a == "--hrx-libhsa") o.hrx_libhsa = next();
        else if (a == "--mlx-server") o.mlx = next();
        else if (a == "--prefill-device") o.prefill_device = next();
        else if (a == "--prefill-min-tokens") o.prefill_min_tokens = std::stoi(next());
        else if (a == "--lean") o.lean = true;
        else if (a == "--mtp") o.mtp = next();
        else if (a == "--mtp-max") o.mtp_max = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty()) { usage(stderr); return 2; }

    if (fs::is_directory(o.model)) {
        if (o.lean) throw std::runtime_error("--lean serves a .gguf (ROCmFP4 or ROCmI4)");
        if (o.device != "auto" && o.device != "npu")
            throw std::runtime_error("an NPU model directory runs on --device npu");
#ifdef ONEBIT_NPU
        if (!is_npu_model_dir(o.model)) throw std::runtime_error(o.model + " is not an NPU model directory");
        // The NPU lane serves in process (unified.cpp).
        std::vector<std::string> args = {"-m", o.model, "-p", std::to_string(o.port), "--host", o.host};
        if (!o.alias.empty()) { args.push_back("--alias"); args.push_back(o.alias); }
        std::vector<char*> av;
        for (auto& s : args) av.push_back(s.data());
        return run_unified(int(av.size()), av.data());
#else
        throw std::runtime_error("this build has no NPU lane (configure with -DONEBIT_NPU=ON)");
#endif
    }
    if (o.device == "mlx") return serve_child(o);
    if (fs::path(o.model).extension() == ".gguf") return serve_child(o);
    throw std::runtime_error(o.model + ": expected an NPU model directory or a .gguf file");
}

}  // namespace onebit
