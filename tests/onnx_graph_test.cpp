// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// onnx_graph_test.cpp -- the audio pack's graph, read through the engine and checked for the property the executor will
// depend on: every node's inputs are graph inputs, initializers, or outputs of an *earlier* node. Skips (77) without
// ONEBIT_TEST_WHISPER naming the encoder's .onnx.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "audio_pack.h"
#include "onnx_graph.h"

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // A pack directory (the ctest row's form) or a graph path: the pack form is what a lane has, and it also proves the mel-bin
    // count comes from the encoder's own Conv (engine#1783).
    std::string from_pack;
    const char* path = (argc > 1) ? argv[1] : std::getenv("ONEBIT_TEST_WHISPER");
    if (path != nullptr && *path != '\0' && std::string(path).find(".onnx") == std::string::npos) {
        onebit::audio::AudioPack pack;
        std::string why;
        if (!onebit::audio::discover_audio_pack(path, &pack, &why)) {
            std::printf("FAIL pack: %s\n", why.c_str());
            return 1;
        }
        from_pack = pack.encoder;
        path = from_pack.c_str();
    }
    if (path == nullptr || *path == '\0') {
        std::printf("SKIP: pass the encoder .onnx, or set ONEBIT_TEST_WHISPER to it\n");
        return 77;
    }
    onebit::npu::pack_import::OnnxGraphInfo g;
    try {
        g = onebit::npu::pack_import::parse_onnx_graph(path);
    } catch (const std::exception& e) {
        std::printf("FAIL cannot parse %s: %s\n", path, e.what());
        return 1;
    }

    check(!g.nodes.empty(), "graph has nodes");
    check(g.inputs.size() == 1, "one graph input (the mel window)");
    check(!g.outputs.empty(), "graph declares an output");
    check(!g.initializers.empty(), "graph has initializers");

    // The ordering assumption, checked against the real pack rather than assumed.
    std::set<std::string> known(g.inputs.begin(), g.inputs.end());
    known.insert(g.initializers.begin(), g.initializers.end());
    int unnamed_inputs = 0;
    for (const auto& n : g.nodes) {
        for (const auto& in : n.ins) {
            if (in.empty()) continue;   // optional input, ONNX spells it ""
            if (!known.count(in)) {
                if (unnamed_inputs < 5) std::printf("FAIL %s input '%s' is not produced earlier\n", n.op.c_str(), in.c_str());
                ++unnamed_inputs;
            }
        }
        for (const auto& o : n.outs) known.insert(o);
    }
    check(unnamed_inputs == 0, "every node input is a graph input, an initializer, or an earlier node's output");

    // The op census the reference implementation was written against (tools/whisper_encoder_ref.py): the front-end is
    // exactly two Convs, and a transformer block is MatMul/LayerNormalization/Gelu dominated.
    std::map<std::string, int> census;
    for (const auto& n : g.nodes) ++census[n.op];
    std::printf("nodes %zu, initializers %zu (external %zu), graph input '%s', output '%s'\n", g.nodes.size(),
                g.initializers.size(), g.external.size(), g.inputs.empty() ? "" : g.inputs[0].c_str(),
                g.outputs.empty() ? "" : g.outputs[0].c_str());
    for (const auto& kv : census) std::printf("  %-24s %d\n", kv.first.c_str(), kv.second);

    // The structure, read off the graph rather than assumed from the reference. Measured on whisper-small's encoder: 447
    // nodes, and three facts that changed how the executor has to be written --
    //   * the GELU is NOT a `Gelu` node: it arrives decomposed as Div -> Erf -> Add -> Mul -> Mul, so an executor keyed on
    //     op names alone needs `Erf` and the surrounding arithmetic (one Erf per block plus two in the front-end);
    //   * attention is explicit (Transpose/MatMul/Softmax), with 8 MatMuls per layer: q,k,v,o, fc1, fc2, qk, and the
    //     scores-times-values product;
    //   * the weights arrive transposed (61 Transpose nodes), which is why the projection shapes in the graph are (K, N).
    // The counts scale with depth, so they are checked as ratios against the Softmax count (= one per layer), not as
    // absolute constants -- the same test then holds for whisper-large-v3's 32-layer encoder.
    check(census["Conv"] == 2, "two Convs (the mel front-end: stride 1 then stride 2, both pad 1)");
    const int layers = census["Softmax"];
    check(layers > 0, "attention scores go through a Softmax");
    if (layers > 0) {
        check(census["MatMul"] == 8 * layers, "8 MatMuls per layer (q,k,v,o, fc1, fc2, qk, scores-times-values)");
        check(census["LayerNormalization"] == 2 * layers + 1, "two layer norms per layer plus the encoder's final one");
        // Two exports of this family spell the same activation two ways: whisper-small's encoder_Model decomposes it into
        // Div -> Erf -> Add -> Mul -> Mul (0 Gelu nodes, 14 Erf for 12 layers), while whisper-large-v3's carries a real
        // `Gelu` op (34 for 32 layers). An executor keyed on op names must accept BOTH; the count, one per layer plus two
        // in the front-end, is the invariant that holds either way.
        check(census["Erf"] + census["Gelu"] == layers + 2,
              "the activation, however spelled, appears once per layer plus twice in the front-end");
        check(census["Erf"] == 0 || census["Gelu"] == 0, "an export uses one spelling of the activation, not a mix");
    }
    check(census["Transpose"] > layers, "the weights arrive transposed, so the graph's MatMuls are (K, N)");

    // One attribute the executor must not guess: LayerNormalization's epsilon is a float attribute, and a reader that
    // drops fixed32 (as the census walk does) sees every one of them as absent.
    const onebit::npu::pack_import::OnnxNodeInfo* ln = nullptr;
    for (const auto& n : g.nodes)
        if (n.op == "LayerNormalization") {
            ln = &n;
            break;
        }
    if (ln != nullptr) {
        const auto it = ln->attrs_float.find("epsilon");
        check(it != ln->attrs_float.end(), "LayerNormalization carries epsilon as a float attribute");
        if (it != ln->attrs_float.end()) check(it->second > 0.0f && it->second < 1e-3f, "epsilon is in the usual range");
    }

    if (g_failures != 0) {
        std::printf("onnx graph: %d failures\n", g_failures);
        return 1;
    }
    std::printf("onnx graph: topological order, op census (%d layers, activation spelled %s) and layer-norm epsilon all hold\n",
                census["Softmax"], census["Gelu"] ? "Gelu" : "Div/Erf/Add/Mul/Mul");
    return 0;
}
