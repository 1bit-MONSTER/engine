// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// audio_exec_test.cpp -- run an audio pack's encoder through the engine's own host executor and compare it against
// onnxruntime, on an input both sides can build from a formula (no binary fixture in the repo).
//
// The mel is mel[b,t] = sin(0.013*(b+1)*(t+1))*0.5 + 0.25 for 80 bins and 3000 frames, computed in double on both sides.
// Expected values below are onnxruntime's, on whisper-small's encoder_model.onnx. Skips (77) without the pack.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "audio_exec.h"
#include "onnx_graph.h"

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

void expect_close(float got, double want, double tol, const std::string& what) {
    const double d = std::fabs(double(got) - want);
    if (!(d <= tol)) {
        std::printf("FAIL %s: got %.7g want %.7g (|d| %.3g)\n", what.c_str(), double(got), want, d);
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : std::getenv("ONEBIT_TEST_WHISPER");
    if (path == nullptr || *path == '\0') {
        std::printf("SKIP: pass the encoder .onnx, or set ONEBIT_TEST_WHISPER to it\n");
        return 77;
    }
    using namespace onebit::npu::pack_import;
    OnnxGraphInfo g;
    try {
        g = parse_onnx_graph(path);
    } catch (const std::exception& e) {
        std::printf("FAIL cannot parse %s: %s\n", path, e.what());
        return 1;
    }

    // The mel-bin count is the front-end Conv's input-channel count: 80 for whisper-small, 128 for large-v3. Read from the
    // pack rather than hard-coded, so the same test runs on either (the numeric comparison below is small's).
    int bins = 80;
    for (const auto& ti : g.init_tensors)
        if (ti.dims.size() == 3 && ti.dims[0] == 768) { bins = int(ti.dims[1]); break; }
    for (const auto& n : g.nodes)
        if (n.op == "Conv") {
            for (const auto& ti : g.init_tensors)
                if (ti.name == n.ins[1] && ti.dims.size() == 3) bins = int(ti.dims[1]);
            break;
        }
    const int frames = 3000;
    std::vector<float> mel(size_t(bins) * frames);
    for (int b = 0; b < bins; ++b)
        for (int t = 0; t < frames; ++t)
            mel[size_t(b) * frames + t] = float(std::sin(0.013 * (b + 1) * (t + 1)) * 0.5 + 0.25);

    AudioRunResult r;
    std::string why;
    if (!run_audio_graph(g, mel.data(), bins, frames, "", &r, &why)) {
        std::printf("FAIL run: %s\n", why.c_str());
        return 1;
    }
    std::printf("ran %zu nodes -> dims", g.nodes.size());
    for (int64_t d : r.dims) std::printf(" %lld", (long long)d);
    std::printf("\n");
    check(r.dims.size() == 3 && r.dims[0] == 1 && r.dims[1] == 1500, "encoder output is [1,1500,hidden]");
    if (r.dims.size() != 3 || r.dims[2] != 768) {
        // A larger encoder: the point of this run is that the graph executed (including external_data weights), not the
        // small model's numbers.
        std::printf("audio exec: ran a %d-bin encoder of hidden %lld (numeric comparison is whisper-small's only)\n", bins,
                    (long long)(r.dims.size() == 3 ? r.dims[2] : 0));
        return g_failures == 0 ? 0 : 1;
    }

    const int64_t H = 768;
    auto at = [&](int64_t t, int64_t h) { return r.data[size_t(t) * H + size_t(h)]; };
    double mean = 0.0, var = 0.0;
    for (float v : r.data) mean += v;
    mean /= double(r.data.size());
    for (float v : r.data) var += (double(v) - mean) * (double(v) - mean);
    var /= double(r.data.size());
    std::printf("mean %.9f std %.9f\n", mean, std::sqrt(var));

    // onnxruntime, same input, same pack: mean -0.009045244, std 1.678493023
    expect_close(float(mean), -0.009045244, 2e-4, "output mean");
    expect_close(float(std::sqrt(var)), 1.678493023, 2e-4, "output std");

    // frame 0, 750 and 1499 -- the first frame, the middle, and the last (whose window is padding-heavy)
    const double p0[5] = {-0.5922473669, -1.6559172869, 0.0451035500, -0.9414483309, -1.6385756731};
    const double p1[4] = {0.1923044026, -0.0132405162, 0.4830040932, 0.6385290623};
    const double p2[4] = {-2.1586399078, 0.3378775120, 0.8375968933, 0.5081450939};
    for (int i = 0; i < 5; ++i) expect_close(at(0, i), p0[i], 2e-3, "frame 0 col " + std::to_string(i));
    for (int i = 0; i < 4; ++i) expect_close(at(750, i), p1[i], 2e-3, "frame 750 col " + std::to_string(i));
    for (int i = 0; i < 4; ++i) expect_close(at(1499, i), p2[i], 2e-3, "frame 1499 col " + std::to_string(i));

    if (g_failures != 0) {
        std::printf("audio exec: %d failures\n", g_failures);
        return 1;
    }
    std::printf("audio exec: whole encoder on the host matches onnxruntime\n");
    return 0;
}
