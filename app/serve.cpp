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
#ifdef ONEBIT_LAYA
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
#include <filesystem>
#include <memory>
#include <mutex>
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
    std::string mtp_p_min;
    int parallel = 0;
    bool adaptive = false;
    int adaptive_at = 1;
    std::string embed, rerank;   // RAG: an embedding model and a reranker, served beside the chat model
    std::string role;            // "embedding" or "reranking": the model itself is served in that role
    std::vector<std::string> npu_opts;   // --npu-opt KEY=VALUE, for a private NPU route
    std::string laya_model;              // --laya-model DIR: route each request by the Laya scorer (docs/laya.md)
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
};

// The backend process for one device: its command line and environment.
Launch launch_for(const Options& o, const std::string& device, int child_port) {
    Launch l;
    l.device = device;
    l.port = child_port;
    std::vector<std::string>& argv = l.argv;
    std::vector<std::string>& env = l.env;
    bool& drop_model = l.drop_model;
    std::string& set_model = l.set_model;
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
#ifndef ONEBIT_GPU_PRIVATE
        if (split)
            throw std::runtime_error("--prefill-device hrx is not part of this build; build with "
                                     "-DONEBIT_GPU_PRIVATE=<gpu-kernels checkout> (docs/hrx.md)");
#endif
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
    if (o.parallel > 1) {
        // continuous batching: N requests decode together, one read of the weights per step
        // for all of them; llama-server splits --ctx-size across the slots
        if (device == "zinc" || device == "mlx") throw std::runtime_error("--parallel works on the llama.cpp devices (vulkan, hrx, rocm)");
        argv.insert(argv.end(), {"-np", std::to_string(o.parallel)});
    }
    if (!o.mtp.empty()) {
        // the MTP head drafts tokens on the same device; the model checks them in one batch
        if (device == "zinc" || device == "mlx") throw std::runtime_error("--mtp works on the llama.cpp devices (vulkan, hrx, rocm)");
        l.mtp = true;
        argv.insert(argv.end(), {"--spec-type", "draft-mtp", "-md", o.mtp, "-ngld", "99"});
        if (o.mtp_max > 0) { argv.push_back("--spec-draft-n-max"); argv.push_back(std::to_string(o.mtp_max)); }
        if (!o.mtp_p_min.empty()) { argv.push_back("--spec-draft-p-min"); argv.push_back(o.mtp_p_min); }
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
const std::vector<std::string> kGgufDevices = {"vulkan", "hrx", "zinc"};

// The text the Laya scorer routes on: a completion's prompt, or a chat request's messages
// joined the way the request reads.
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

int serve_child(const Options& o) {
    std::vector<Launch> backends;
    // --laya-model with --device auto: each request goes to the device the Laya scorer picks
    // among the ones this build can run a .gguf on; a device's backend starts when it is
    // first picked (one model is not loaded on every device at once)
    const bool laya = !o.laya_model.empty() && o.device == "auto";
#ifdef ONEBIT_LAYA
    std::unique_ptr<onebit::laya::Scorer> scorer;
    std::vector<std::string> laya_devices;
    if (laya) {
        if (o.adaptive || !o.role.empty() || !o.prefill_device.empty() || o.lean)
            throw std::runtime_error("--laya-model does not combine with --adaptive, --embedding/--reranking, --prefill-device or --lean");
        scorer = std::make_unique<onebit::laya::Scorer>();
        if (!scorer->load(o.laya_model)) throw std::runtime_error("laya: " + scorer->error());
        for (const std::string& dev : kGgufDevices) {
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
        overflow.llama_server.clear();
        backends.push_back(launch_for(overflow, "rocm", free_port()));
    } else if (!o.role.empty()) {
        // The model is an embedding or reranking model (Lemonade loads them as models of their
        // own): llama-server on the device asked for, in that role. An input is one pass, so
        // the batch covers the context.
        const std::string dev = o.device == "auto" ? "vulkan" : o.device;
        if (dev != "vulkan" && dev != "hrx" && dev != "rocm")
            throw std::runtime_error("--" + o.role + " runs on llama-server devices (vulkan, hrx, rocm), not " + dev);
        if (o.adaptive || !o.mtp.empty() || !o.prefill_device.empty())
            throw std::runtime_error("--" + o.role + " does not combine with --adaptive, --mtp or --prefill-device");
        Launch l = launch_for(o, dev, free_port());
        const std::string batch = std::to_string(o.ctx_size > 0 ? o.ctx_size : 8192);
        for (const std::string& a : {std::string("--") + o.role, std::string("-b"), batch, std::string("-ub"), batch})
            l.argv.push_back(a);
        l.device = dev + " " + o.role;
        backends.push_back(std::move(l));
    } else {
        backends.push_back(launch_for(o, o.device == "auto" ? "vulkan" : o.device, free_port()));
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
        auto c = std::make_unique<Child>(b.argv, b.env, b.port);
        if (!c->wait_ready(std::chrono::seconds(600), g_stop)) return false;
        started[i] = c.get();
        children.push_back(std::move(c));
        return true;
    };
    std::vector<std::atomic<int>> inflight(backends.size());
    std::vector<std::atomic<long>> routed(backends.size());
    std::mutex route_mu;
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
            const std::string dev = onebit::laya::route_device(*scorer, request_state(q), laya_devices);
            const auto it = std::find(laya_devices.begin(), laya_devices.end(), dev);
            if (it == laya_devices.end()) {
                r.status = 502;
                r.set_content(R"({"error":{"message":"laya routing failed"}})", "application/json");
                return;
            }
            pick = size_t(it - laya_devices.begin());
            if (!ensure(pick)) {
                r.status = 503;
                r.set_content(json{{"error", {{"message", backends[pick].device + " did not become ready"}}}}.dump(), "application/json");
                return;
            }
            held = std::make_shared<InFlight>(&inflight[pick]);
            ++routed[pick];
            forward(backends[pick].port, held, -1, id, backends[pick].drop_model, backends[pick].set_model, q, r);
            return;
        }
#endif
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
    const bool llama = device == "vulkan" || device == "hrx" || device == "rocm" || device == "vulkan+rocm" ||
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

    struct sigaction sa{};
    sa.sa_handler = on_stop_signal;
    ::sigaction(SIGTERM, &sa, nullptr);
    ::sigaction(SIGINT, &sa, nullptr);
    for (const auto& b : laya ? std::vector<Launch>{} : backends) {
        std::fprintf(stderr, "1bit serve: %s on %s (%s)\n", id.c_str(), b.device.c_str(), b.argv[0].c_str());
        children.push_back(std::make_unique<Child>(b.argv, b.env, b.port));
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

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit serve -m <model> [--port 8000] [--host 127.0.0.1]\n"
                 "                  [--device auto|npu|vulkan|hrx|rocm|zinc|mlx] [--ctx-size N] [--alias NAME]\n"
                 "                  [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]\n"
                 "                  [--prefill-device hrx] [--prefill-min-tokens N]   (with --device vulkan)\n"
                 "                  [--lean]   ROCmFPX formats: ROCmFP4 on vulkan, ROCmI4 with --device rocm\n"
                 "                  [--mtp HEAD.gguf] [--mtp-max N] [--mtp-p-min P]   multi-token prediction (vulkan, hrx, rocm)\n"
                 "                  [--parallel N]   N requests decoded together (continuous batching)\n"
                 "                  [--adaptive] [--adaptive-at N]   Vulkan (+MTP) for N in flight (default 1), ROCm batches the rest\n"
                 "                  [--embed MODEL.gguf] [--rerank MODEL.gguf]   RAG: /v1/embeddings and /v1/rerank\n"
                 "                  [--embedding | --reranking]   serve the model itself as an embedding or reranking model\n"
                 "                  [--npu-opt KEY=VALUE ...]   an option for a private NPU route (docs/npu.md)\n"
                 "                  [--laya-model DIR]   with --device auto, the Laya scorer picks each request's device (docs/laya.md)\n"
                 "  <model>: an NPU model directory (model.q4nx + npu/, or one a private NPU route serves),\n"
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
        else if (a == "--mtp-p-min") o.mtp_p_min = next();
        else if (a == "--parallel") o.parallel = std::stoi(next());
        else if (a == "--adaptive") o.adaptive = true;
        else if (a == "--adaptive-at") o.adaptive_at = std::stoi(next());
        else if (a == "--embed") o.embed = next();
        else if (a == "--embedding" || a == "--embeddings") o.role = "embedding";
        else if (a == "--reranking") o.role = "reranking";
        else if (a == "--rerank") o.rerank = next();
        else if (a == "--npu-opt") o.npu_opts.push_back(next());
        else if (a == "--laya-model") o.laya_model = next();
        else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (o.model.empty()) { usage(stderr); return 2; }

    if (fs::is_directory(o.model)) {
        if (o.lean) throw std::runtime_error("--lean serves a .gguf (ROCmFP4 or ROCmI4)");
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
    if (fs::path(o.model).extension() == ".gguf") return serve_child(o);
    throw std::runtime_error(o.model + ": expected an NPU model directory or a .gguf file");
}

}  // namespace onebit
