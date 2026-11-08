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
// Equivalence probe for step_batch_embeds (engine#1311/TQVL, plan P4's inputs.kind): with rows built from the model's
// OWN embedding table, the logits must be identical to step_batch(ids) — same arithmetic, same inputs, so
// byte-for-byte is the only acceptable answer.
//
//   embeds_equiv_probe <model.q4nx> <kernels dir>          # device path (inside npu-hold)
//   NPU_INFER_HOST_GEMM=1 embeds_equiv_probe <model> <dir> # the same body on the host, no device
//
// Why it exists: a hybrid (vision features into a text decoder, AMD's `embedding.onnx` -> `inputs_embeds`) is only
// worth building if the decoder's embedding entry point is arithmetic-free — i.e. it changes where the rows come from
// and nothing else. This is the row that says so, on the container the hybrid targets.
#include "q4nx_forward.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static int argmax(const std::vector<float>& v) {
    int b = -1;
    for (size_t i = 0; i < v.size(); i++) if (b < 0 || v[i] > v[(size_t)b]) b = int(i);
    return b;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.q4nx> <kernels dir>\n", argv[0]); return 2; }
    Q4nxNpuForward f;
    // host-GEMM when the environment asks for it (NPU_INFER_HOST_GEMM=1): the same body, no device, so this
    // probe can also be run device-free.
    const bool host = getenv("NPU_INFER_HOST_GEMM") != nullptr;
    // The device must outlive the forward: the forward keeps a pointer to it (a stack temporary here cost one
    // confusing "code 208: Bad file descriptor" before it was obvious).
    std::unique_ptr<xrt::device> dev;
    bool ok = false;
    if (host) {
        ok = f.init_host(argv[1], argv[2]);
    } else {
        dev = std::make_unique<xrt::device>(0);
        ok = f.init(*dev, argv[1], argv[2]);
    }
    if (!ok) { std::fprintf(stderr, "init: %s\n", f.last_error()); return 1; }
    // three prompts, ids chosen from the container's own vocabulary
    // the hidden width comes from a row the model itself produces
    std::vector<float> any_row;
    f.embed_row(11, any_row);
    const size_t H_WIDTH = any_row.size();
    std::printf("hidden width %zu\n", H_WIDTH);
    std::vector<std::vector<int>> prompts = {{9707, 11, 1879, 374, 264}, {3838, 374, 1234}, {16, 13, 220}};
    int rows_checked = 0, mismatched = 0;
    for (const auto& ids : prompts) {
        std::vector<float> a, b;
        f.reset();
        if (!f.step_batch(ids, 0, a)) { std::fprintf(stderr, "step_batch: %s\n", f.last_error()); return 1; }
        std::vector<float> embeds;
        embeds.reserve(ids.size() * size_t(size_t(H_WIDTH)));
        for (int id : ids) {
            std::vector<float> row;
            f.embed_row(id, row);
            embeds.insert(embeds.end(), row.begin(), row.end());
        }
        f.reset();
        if (!f.step_batch_embeds(embeds, 0, b)) { std::fprintf(stderr, "step_batch_embeds: %s\n", f.last_error()); return 1; }
        int diff = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) if (a[i] != b[i]) diff++;
        rows_checked++;
        if (diff || a.size() != b.size() || argmax(a) != argmax(b)) {
            mismatched++;
            std::printf("MISMATCH prompt %d: %d/%zu logits differ, argmax %d vs %d\n", rows_checked, diff, a.size(),
                        argmax(a), argmax(b));
        } else {
            std::printf("ok   prompt %d: %zu logits byte-identical, argmax %d\n", rows_checked, a.size(), argmax(a));
        }
    }
    std::printf("%s (%d/%d prompts identical)\n", mismatched ? "FAIL" : "PASS", rows_checked - mismatched, rows_checked);
    return mismatched ? 1 : 0;
}
