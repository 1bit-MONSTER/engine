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
//   1bit serve -m <model>            one model on any device (NPU, HRX, CPU)
//                                    behind an OpenAI-compatible API; how the
//                                    engine runs inside Lemonade (docs/serve.md).
//   1bit unified -m <model dir>      one native model on the NPU fast lane behind
//                                    OpenAI endpoints; serve's NPU route.
//   1bit npu-run [options]           token ids in, token ids out, on the NPU fast
//                                    lane (docs/npu.md); for checks and benchmarks.
//   1bit npu-pack <info|import>      turns a Ryzen AI OGA NPU model pack into a Q4NX
//                                    model directory (docs/npu.md, "OGA packs").
//   1bit moe-cache [options]         replays an expert trace through the MoE expert
//                                    cache against the model file (docs/moe-streaming.md).
//   1bit ort-info                    the ONNX Runtime this binary links (pin, API version,
//                                    built-in EPs and the device roster it can register),
//                                    -DONEBIT_ORT builds (docs/onnx.md).
//   1bit ort-run <model.onnx>        run a plain ONNX model through ORT with our plugin EP first
//                                    and the CPU EP last (docs/onnx.md, "Qwen3-0.6B").
//   1bit <command> [options]         a command a private NPU add-on registered
//                                    (-DONEBIT_NPU_PRIVATE builds; docs/npu.md).
//
// See docs/PORTING.md for what lands next.

#ifdef ONEBIT_ORT
// onnx/ort_info.cpp: prints the ORT build we link and the EP roster, then exits.
int onebit_ort_info_main(int argc, char** argv);
// onnx/ort_run.cpp: a plain ONNX model through ORT with the plugin EP registered first.
int onebit_ort_run_main(int argc, char** argv);
#endif
#ifdef ONEBIT_NPU_HOST
#include "pack_import.h"  // 1bit npu-pack: an OGA NPU pack -> a Q4NX model directory
#endif
#ifdef ONEBIT_NPU
#include "forward_serve.h"
#include "generate.h"
#include "model.h"
#include "private_route.h"
#include "unified.h"
#endif
#ifdef ONEBIT_LAYA
#include "laya_checkpoint.h"
#include "laya_first.h"
#include "policy.h"
#include "route.h"
#include "route_serve.h"
#include "route_eval.h"
#include "scorer.h"
#endif
#include "serve.h"
#ifdef ONEBIT_MOE
#include "cache_bench.h"
#endif

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
                 "  forward-serve -m <model.q4nx> --kernels <dir>  serve a repacked GGUF on the NPU\n"
#endif
#ifdef ONEBIT_NPU_HOST
                 "  npu-pack <info|import> <pack dir>  import a Ryzen AI OGA NPU pack (npu-pack --help)\n"
#endif
                 "  comfy <workflow.json>       run a ComfyUI workflow with ComfyUI.cpp (docs/comfyui.md)\n"
#ifdef ONEBIT_LAYA
                 "  route [options]             pick a device for a request via the Laya scorer\n"
#endif
#ifdef ONEBIT_MOE
                 "  moe-cache [options]         replay an expert trace through the MoE expert cache\n"
#endif
#ifdef ONEBIT_ORT
                 "  ort-info                    the ONNX Runtime this binary links (pin, EP roster)\n"
                 "  ort-run <model.onnx>        a plain ONNX model through ORT, our EP before the CPU\n"
#endif
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
                        "  --kernels  the pack's kernel directory. Two generations are driven: the per-role lane set\n"
                        "             (layer_ctx{1,2,17}.elf, lmhead.elf, layer.pdi) as a token loop, and the sealed set\n"
                        "             (SEALED-*/full_i8_*) through the in-tree forward as ONE multi-row pass, printing each\n"
                        "             row's choice (engine#1862; -n does not apply to the sealed route)\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (model_dir.empty() || kernel_dir.empty() || ids.empty()) throw std::runtime_error("--model, --kernels and --ids are required");
    std::vector<int> prompt;
    std::istringstream in(ids);
    for (int t; in >> t;) prompt.push_back(t);

    // engine#1862: a kernel directory from the sealed/full-ELF generation has no layer_ctx*.elf, and the bare
    // "cannot read <dir>/layer_ctx1.elf" from npu/lane.cpp reads as a missing file rather than a different route.
    // Name the generation that IS there, and the command that drives it.
    {
        namespace fs = std::filesystem;
        if (!fs::exists(fs::path(kernel_dir) / "layer_ctx1.elf")) {
            int sealed = 0;
            std::error_code ec;
            for (const auto& ent : fs::directory_iterator(kernel_dir, ec)) {
                const std::string name = ent.path().filename().string();
                if (ent.path().extension() == ".elf" &&
                    (name.rfind("SEALED-", 0) == 0 || name.rfind("full_i8_", 0) == 0))
                    ++sealed;
            }
            if (sealed > 0) {
                // engine#1862: this is every pack on disk, so refuse-and-redirect was the wrong answer for the CLI that
                // exists to hand N ids to ONE pass. The sealed set is driven by the same in-tree forward `forward-serve`
                // uses; run the ids through it and print each row's choice.
                std::fprintf(stderr, "1bit npu-run: sealed kernel set (%d ELF(s)), one multi-row pass through the "
                                     "in-tree forward\n", sealed);
                return onebit::run_forward_ids(model_dir, kernel_dir, prompt, opt.max_tokens,
                                               opt.repetition_penalty, std::getenv("NPU_INFER_HOST_GEMM") != nullptr);
            }
            throw std::runtime_error("no kernel set in " + kernel_dir +
                                     ": neither layer_ctx1.elf (the per-role set npu-run drives) nor a SEALED-*/full_i8_* "
                                     "ELF (the sealed set) is present");
        }
    }

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

#ifdef ONEBIT_LAYA
// One request in, one device out (RFC #186): Laya classifies the request (laya/route.h) and the
// route policy (config/route-policy.json or --route-policy) picks among --devices. --classify
// also prints the class and Laya's confidence. --serve keeps the scorer resident and answers one
// decision per stdin line (app/route_serve.cpp), which is the form the fork calls per request.
int run_route(int argc, char** argv) {
    std::string laya_model, state, policy_path, devices_str = "npu,hrx,cpu";
    bool classify = false, serve = false, json_out = false;
    onebit::RouteEvalOptions ev;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--laya-model") laya_model = next();
        else if (a == "--state") state = next();
        else if (a == "--devices") devices_str = next();
        else if (a == "--route-policy") policy_path = next();
        else if (a == "--classify") classify = true;
        else if (a == "--json") json_out = true;
        else if (a == "--serve") serve = true;
        else if (a == "--laya-models") ev.table = next();
        else if (a == "--eval") ev.eval_cases = next();
        else if (a == "--dump-encoder") ev.dump_cases = next();
        else if (a == "--out") ev.out_dir = next();
        else if (a == "--per-case") ev.per_case = next();
        else if (a == "--rotate") ev.rotate = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: 1bit route [--laya-model <dir>] --state <text> [--devices npu,hrx,cpu]\n"
                        "                  [--route-policy FILE] [--classify]\n"
                        "  Laya classifies the request (code, prose, short, long_doc) and the route policy\n"
                        "  picks one of the devices; --classify prints \"<class> <confidence> <device>\"\n"
                        "  and --json prints the same decision as one JSON line\n"
                        "usage: 1bit route --serve [--laya-model <dir>] [--devices npu,hrx,cpu]\n"
                        "                  [--route-policy FILE]\n"
                        "  Resident: loads the scorer once, then reads one {\"state\": \"<text>\"} per line\n"
                        "  on stdin and writes one {\"class\",\"confidence\",\"device\",\"decision_ms\"}\n"
                        "  per line on stdout, after a first {\"ready\":true,\"load_ms\":N} line; EOF exits\n"
                        "usage: 1bit route --laya-models FILE [--laya-model <dir>] --eval CASES.jsonl [--json]\n"
                        "                  [--per-case OUT.jsonl]\n"
                        "  The table's question over labelled states ({\"state\",\"label\"} per line), the path\n"
                        "  serve --laya-models takes: accuracy, per-class recall and precision, confusion,\n"
                        "  confidence, ECE, log loss and decision time\n"
                        "usage: 1bit route --laya-models FILE [--laya-model <dir>] --dump-encoder CASES.jsonl\n"
                        "                  --out DIR [--rotate]\n"
                        "  Writes the decision head's input for each case (DIR/h.f32, DIR/cases.jsonl,\n"
                        "  DIR/dump.json) for a head-only fine-tune; --rotate adds every option order\n");
            return 0;
        } else throw std::runtime_error("unknown option " + a);
    }
    if (!ev.table.empty()) {
        if (serve || !state.empty() || classify)
            throw std::runtime_error("--laya-models takes --eval or --dump-encoder, not --state, --classify or --serve");
        ev.laya_model = laya_model;
        ev.json = json_out;
        return onebit::run_route_eval(ev);
    }
    if (!ev.eval_cases.empty() || !ev.dump_cases.empty()) throw std::runtime_error("--eval and --dump-encoder need --laya-models FILE");
    if (!serve && state.empty()) throw std::runtime_error("--state <text> is required");

    std::vector<std::string> devices;
    std::string cur;
    for (std::istringstream ss(devices_str); std::getline(ss, cur, ',');)
        if (!cur.empty()) devices.push_back(cur);
    if (devices.empty()) throw std::runtime_error("--devices is empty");
    if (serve) return onebit::run_route_serve(laya_model, devices, policy_path);

    // One resolution chain for `serve --laya` and this command (app/laya_checkpoint.h), so the two
    // cannot disagree about where the checkpoint is.
    try {
        laya_model = onebit::laya_checkpoint(laya_model);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "1bit route: %s\n", e.what());
        return 1;
    }
    const onebit::laya::RoutePolicy policy = onebit::laya::load_route_policy(policy_path);
    const auto t0 = std::chrono::steady_clock::now();
    onebit::laya::Scorer scorer;
    if (!scorer.load(laya_model)) {
        std::fprintf(stderr, "1bit route: %s\n", scorer.error().c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const onebit::laya::RequestClass cls = onebit::laya::classify_request(scorer, state);
    if (cls.label.empty()) {
        std::fprintf(stderr, "1bit route: %s\n", scorer.error().c_str());
        return 1;
    }
    const std::string device = policy.pick(cls.label, cls.confidence, devices);
    const auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    if (json_out)
        std::printf("%s\n", onebit::route_json_line(cls.label, cls.confidence, device, ms(t1, t2)).c_str());
    else if (classify) std::printf("%s %.2f %s\n", cls.label.c_str(), cls.confidence, device.c_str());
    else std::printf("%s\n", device.c_str());
    std::fprintf(stderr, "1bit route: scorer loaded in %.0f ms, decision in %.0f ms\n", ms(t0, t1), ms(t1, t2));
    return 0;
}
#endif

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
#ifdef ONEBIT_NPU_HOST
    if (cmd == "npu-pack") {
        try {
            return onebit::npu::pack_import::run(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit npu-pack: %s\n", e.what());
            return 1;
        }
    }
#endif
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
    if (cmd == "forward-serve") {
        try {
            return onebit::run_forward_serve(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit forward-serve: %s\n", e.what());
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
    if (cmd == "laya-first") {
        try {
            return onebit::run_laya_first(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit laya-first: %s\n", e.what());
            return 1;
        }
    }
#ifdef ONEBIT_LAYA
    if (cmd == "route") {
        try {
            return run_route(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit route: %s\n", e.what());
            return 1;
        }
    }
#endif
#ifdef ONEBIT_MOE
    if (cmd == "moe-cache") {
        try {
            return onebit::moe::run_moe_cache(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "1bit moe-cache: %s\n", e.what());
            return 1;
        }
    }
#endif
#ifdef ONEBIT_ORT
    if (cmd == "ort-info") {
        return onebit_ort_info_main(argc - 1, argv + 1);
    }
    if (cmd == "ort-run") {
        return onebit_ort_run_main(argc - 1, argv + 1);
    }
#endif
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
