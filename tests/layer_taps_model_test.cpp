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

// layer_taps_model_test.cpp -- Q4nxNpuForward::set_layer_taps() on a real container (engine#1780).
//
// Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a model directory. What it holds the hook to:
//   - a batched pass of M tokens appends M rows of [taps * H];
//   - the last layer's block of the last row is, bit for bit, the residual the forward itself keeps
//     (last_residual()), so the tap is the stream and not a copy taken at the wrong point;
//   - the same holds for per-token step() passes, and each appends exactly one row;
//   - layer 0's block is, bit for bit, what NPU_INFER_DUMP_DIR writes for layer 0 in the same pass -- an instrument
//     that already existed and writes at the end of the layer body, so a tap that is one layer off cannot agree;
//   - a step() pass at position 0 gives the same row as the batched pass gave for that position, closely (the two
//     paths are not bit-identical in general), which holds step()'s tap to the same layers;
//   - a list that is not usable is refused and leaves the previous taps in place; an empty list turns them off.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "q4nx_forward.h"

static int g_fail = 0, g_n = 0;
static void check(bool ok, const char* what) {
    ++g_n;
    if (!ok) { std::printf("FAIL %s\n", what); ++g_fail; }
}

int main() {
    const char* dir = std::getenv("ONEBIT_TEST_Q4NX");
    if (!dir || !*dir) {
        std::printf("SKIP: set ONEBIT_TEST_Q4NX to a Q4NX model directory\n");
        return 77;
    }
    setenv("NPU_INFER_HOST_GEMM", "1", 1);
    Q4nxNpuForward fwd;
    const std::string model = std::string(dir) + "/model.q4nx";
    if (!fwd.init_host(model.c_str(), std::string(dir) + "/kernels")) {
        std::printf("SKIP: init_host failed for %s (%s)\n", model.c_str(), fwd.last_error());
        return 77;
    }
    std::vector<float> one;
    fwd.embed_row(1, one);
    const size_t H = one.size();
    const int NL = fwd.layer_count();
    if (NL < 2) { std::printf("SKIP: a %d-layer model has no two layers to tap\n", NL); return 77; }
    const std::vector<int> layers = {0, NL - 1};
    std::vector<float> sink, logits;

    check(!fwd.set_layer_taps({NL - 1, 0}, &sink), "a descending list is refused");
    check(!fwd.set_layer_taps({0, NL}, &sink), "a layer outside the model is refused");
    check(fwd.set_layer_taps(layers, &sink), "the first and the last layer are accepted");

    // The existing per-layer dump, into a directory beside the test's working directory.
    namespace fs = std::filesystem;
    const fs::path dump = fs::current_path() / ("layer_taps_dump." + std::to_string(getpid()));
    fs::create_directories(dump);
    setenv("NPU_INFER_DUMP_DIR", dump.c_str(), 1);

    const std::vector<int> tokens = {1, 2, 3};
    const bool batched = !fwd.noncausal() ? fwd.step_batch_all(tokens, 0, logits) : fwd.step_batch(tokens, 0, logits);
    check(batched, "the batched pass runs");
    check(sink.size() == tokens.size() * 2 * H, "a batched pass of 3 tokens appends 3 rows of 2 x H");
    if (batched && sink.size() == tokens.size() * 2 * H) {
        const std::vector<float>& res = fwd.last_residual();
        const float* last = sink.data() + (2 * 2 + 1) * H;       // row 2, tap 1
        check(res.size() == H && std::memcmp(last, res.data(), H * sizeof(float)) == 0,
              "batched: the last layer's block of the last row is the forward's own residual, bit for bit");
        check(std::memcmp(sink.data() + (2 * 2 + 0) * H, last, H * sizeof(float)) != 0,
              "batched: layer 0's block is not the last layer's");
        std::vector<float> ref(tokens.size() * H);
        FILE* f = std::fopen((dump / "layer_00.f32").c_str(), "rb");
        const size_t got = f ? std::fread(ref.data(), sizeof(float), ref.size(), f) : 0;
        if (f) std::fclose(f);
        bool same = got == ref.size();
        for (size_t m = 0; m < tokens.size() && same; m++)
            same = std::memcmp(sink.data() + (m * 2 + 0) * H, ref.data() + m * H, H * sizeof(float)) == 0;
        check(same, "batched: layer 0's block of every row is what NPU_INFER_DUMP_DIR wrote for layer 0, bit for bit");
    }
    unsetenv("NPU_INFER_DUMP_DIR");
    std::error_code ec;
    fs::remove_all(dump, ec);
    const std::vector<float> batch_rows = sink;

    if (!fwd.noncausal()) {
        sink.clear();
        fwd.reset();
        bool ok = true;
        size_t grew = 0;
        for (int i = 0; i < 3 && ok; i++) {
            const size_t before = sink.size();
            ok = fwd.step(tokens[size_t(i)], i, logits);
            grew = sink.size() - before;
            ok = ok && grew == 2 * H;
        }
        check(ok, "each step() appends exactly one row of 2 x H");
        if (ok && batch_rows.size() >= 2 * H) {   // position 0: the same token at the same place in both paths
            double peak = 0.0, worst = 0.0;
            for (size_t i = 0; i < 2 * H; i++) {
                peak = std::max(peak, std::fabs(double(batch_rows[i])));
                worst = std::max(worst, std::fabs(double(batch_rows[i]) - double(sink[i])));
            }
            std::printf("step() vs batched, position 0, both taps: max difference %.3g against a peak of %.3g\n", worst, peak);
            check(worst <= 1e-3 * peak, "step(): position 0's row is the batched pass's row for it, to 1e-3 of the peak");
        }
        if (ok) {
            const std::vector<float>& res = fwd.last_residual();
            check(res.size() == H && std::memcmp(sink.data() + (2 * 2 + 1) * H, res.data(), H * sizeof(float)) == 0,
                  "step(): the last layer's block of the last row is the forward's own residual, bit for bit");
        }
    }

    const size_t held = sink.size();
    check(fwd.set_layer_taps({}, nullptr), "an empty list turns the taps off");
    fwd.reset();
    const bool again = fwd.noncausal() ? fwd.step_batch(tokens, 0, logits) : fwd.step(1, 0, logits);
    check(again && sink.size() == held, "with the taps off a pass appends nothing");

    if (g_fail) { std::printf("layer_taps_model: %d of %d checks FAILED\n", g_fail, g_n); return 1; }
    std::printf("PASS  layer_taps_model: %d checks on %d layers, H %zu\n", g_n, NL, H);
    return 0;
}
