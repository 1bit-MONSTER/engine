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
// 1bit: the engine binary. Subcommands:
//
//   1bit serve -m <model>            one model on any device (NPU, Vulkan, HRX,
//                                    ZINC, MLX) behind an OpenAI-compatible API;
//                                    how the engine runs inside Lemonade (docs/serve.md).
//   1bit unified -m <model dir>      one native model on the NPU fast lane behind
//                                    OpenAI endpoints; serve's NPU route.
//   1bit npu-run [options]           token ids in, token ids out, on the NPU fast
//                                    lane (docs/npu.md); for checks and benchmarks.
//   1bit <command> [options]         a command a private NPU add-on registered
//                                    (-DONEBIT_NPU_PRIVATE builds; docs/npu.md).
//
// See docs/PORTING.md for what lands next.

#ifdef ONEBIT_NPU
#include "generate.h"
#include "model.h"
#include "private_route.h"
#include "unified.h"
#endif
#include "serve.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr const char* kVersion = "0.0.1";

void usage(FILE* out) {
    std::fprintf(out,
                 "usage: 1bit <command> [options]\n"
                 "\n"
                 "commands:\n"
                 "  serve -m <model>            one model on any device, OpenAI-compatible API\n"
#ifdef ONEBIT_NPU
                 "  unified -m <model dir>      serve one NPU model (OpenAI endpoints)\n"
                 "  npu-run [options]           generate on the NPU fast lane (npu-run --help)\n"
#endif
                 "  comfy <workflow.json>       run a ComfyUI workflow with ComfyUI.cpp (docs/comfyui.md)\n"
                 "  version                     print the version\n"
                 "  help                        show this help\n");
#ifdef ONEBIT_NPU
    for (const auto& c : onebit::npu::private_commands())
        std::fprintf(out, "  %-27s %s\n", (c.name + " [options]").c_str(), c.help.c_str());
#endif
}

#ifdef ONEBIT_NPU
// Token ids in, token ids out. Prints one id per line on stdout and the timing
// on stderr; --dump-logits writes each step's logits as float32
// (decode_logits_idx<N>.bin, N = decode step), for comparison with a reference.
int run_npu(int argc, char** argv) {
    std::string model_dir, kernel_dir, ids, dump;
    onebit::npu::GenerateOptions opt;
    opt.max_tokens = 24;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model_dir = next();
        else if (a == "--kernels") kernel_dir = next();
        else if (a == "--ids") ids = next();
        else if (a == "-n") opt.max_tokens = std::stoi(next());
        else if (a == "--repetition-penalty") opt.repetition_penalty = std::stof(next());
        else if (a == "--no-eos") opt.stop_at_eos = false;
        else if (a == "--dump-logits") dump = next();
        else if (a == "--help" || a == "-h") {
            std::printf("usage: 1bit npu-run --model <dir> --kernels <dir> --ids \"<id> <id> ...\"\n"
                        "                    [-n 24] [--repetition-penalty 1.1] [--no-eos] [--dump-logits <dir>]\n"
                        "  --model    directory with model.q4nx and config.json\n"
                        "  --kernels  directory with layer_ctx{1,2,17}.elf, lmhead.elf and layer.pdi\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (model_dir.empty() || kernel_dir.empty() || ids.empty()) throw std::runtime_error("--model, --kernels and --ids are required");
    std::vector<int> prompt;
    std::istringstream in(ids);
    for (int t; in >> t;) prompt.push_back(t);

    const onebit::npu::Model model(model_dir);
    auto t0 = std::chrono::steady_clock::now();
    onebit::npu::Lane lane(model, kernel_dir);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> lg;
    if (!dump.empty())
        opt.on_logits = [&](int step, onebit::npu::Lane& l) {
            l.logits(lg);
            std::ofstream(dump + "/decode_logits_idx" + std::to_string(step) + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * sizeof(float)));
        };
    const auto r = onebit::npu::generate(lane, model, prompt, opt);
    for (int t : r.tokens) std::printf("%d\n", t);
    std::fprintf(stderr, "load %.0f ms | prefill %zu tokens %.1f ms | decode %zu tokens %.1f ms/token (%.1f tok/s)%s\n",
                 load_ms, prompt.size(), r.prefill_ms, r.tokens.size(), r.decode_ms / double(r.tokens.size()),
                 1000.0 * double(r.tokens.size()) / r.decode_ms, r.stopped_at_eos ? " | stopped at EOS" : "");
    return 0;
}
#endif

// ComfyUI.cpp is GPL-3.0 and a separate program: exec it, never link it (docs/comfyui.md).
// The binary: $ONEBIT_COMFYUI (an absolute path), then this build's (-DONEBIT_COMFYUI=ON),
// then comfyui_cpp on PATH. It is opened once, checked through that descriptor (a regular,
// executable file that other users cannot write) and run with fexecve, so the file that was
// checked is the file that runs.
std::string find_comfy() {
    if (const char* e = std::getenv("ONEBIT_COMFYUI"); e && *e) {
        if (e[0] != '/') throw std::runtime_error("ONEBIT_COMFYUI must be an absolute path, not " + std::string(e));
        return e;
    }
#ifdef ONEBIT_COMFYUI_BIN
    if (std::filesystem::exists(ONEBIT_COMFYUI_BIN)) return ONEBIT_COMFYUI_BIN;
#endif
    if (const char* path = std::getenv("PATH")) {
        std::stringstream dirs(path);
        for (std::string dir; std::getline(dirs, dir, ':');) {
            if (dir.empty() || dir[0] != '/') continue;  // relative PATH entries depend on the cwd
            const std::string cand = dir + "/comfyui_cpp";
            if (::access(cand.c_str(), X_OK) == 0) return cand;
        }
    }
    throw std::runtime_error("no comfyui_cpp: build with -DONEBIT_COMFYUI=ON or set ONEBIT_COMFYUI");
}

int run_comfy(int argc, char** argv) {
    if (argc < 1 || !std::strcmp(argv[0], "-h") || !std::strcmp(argv[0], "--help")) {
        std::printf("usage: 1bit comfy <workflow.json>\n"
                    "  runs a ComfyUI API-format workflow (SD1.5 txt2img/img2img, see docs/comfyui.md)\n");
        return argc < 1 ? 2 : 0;
    }
    const std::string bin = find_comfy();
    // O_CLOEXEC: when fexecve succeeds the kernel closes the descriptor with the old image;
    // open() follows symlinks, so the checks below apply to the file that actually runs.
    const int fd = ::open(bin.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open " + bin + ": " + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || !(st.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH))) {
        ::close(fd);
        throw std::runtime_error(bin + " is not an executable file");
    }
    if (st.st_mode & S_IWOTH) {
        ::close(fd);
        throw std::runtime_error(bin + " is writable by other users; refusing to run it");
    }
    std::vector<char*> args{const_cast<char*>(bin.c_str())};
    for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
    args.push_back(nullptr);
    ::fexecve(fd, args.data(), environ);
    const int err = errno;
    ::close(fd);
    throw std::runtime_error("cannot run " + bin + ": " + std::strerror(err));
}

}  // namespace

int main(int argc, char** argv) {
#ifdef ONEBIT_NPU_PRIVATE
    onebit::npu::register_private_addon();
#endif
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "serve") {
        try {
            return onebit::run_serve(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit serve: %s\n", e.what());
            return 1;
        }
    }
#ifdef ONEBIT_NPU
    if (cmd == "unified") {
        try {
            return onebit::run_unified(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit unified: %s\n", e.what());
            return 1;
        }
    }
    if (cmd == "npu-run") {
        try {
            return run_npu(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit npu-run: %s\n", e.what());
            return 1;
        }
    }
    if (const auto* c = onebit::npu::find_private_command(cmd)) {
        try {
            return c->run(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit %s: %s\n", cmd.c_str(), e.what());
            return 1;
        }
    }
#endif
    if (cmd == "comfy") {
        try {
            return run_comfy(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit comfy: %s\n", e.what());
            return 127;
        }
    }
    if (cmd == "version" || cmd == "--version") {
        std::printf("1bit %s\n", kVersion);
        return 0;
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        usage(stdout);
        return 0;
    }
    std::fprintf(stderr, "1bit: unknown command '%s'\n\n", cmd.c_str());
    usage(stderr);
    return 2;
}
