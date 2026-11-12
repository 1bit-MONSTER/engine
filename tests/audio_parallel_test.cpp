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
//
// audio_parallel_test.cpp -- the two claims npu/audio_parallel.h makes, as assertions (engine#1931 review):
//
//   1. the executor's MatMul is BIT-identical whatever ONEBIT_AUDIO_THREADS says. A tolerance cannot see this, and the
//      pack-gated comparison against onnxruntime runs at one thread count, so here the same MatMul runs through
//      run_audio_graph() at 1, 2, 3, 7, 8 and 64 threads and unset, and the outputs are compared with memcmp;
//   2. a unit that throws is an exception on the calling thread after every worker has been joined, not std::terminate.
//
// The graph is built in memory (one MatMul against an inline initializer), so this needs no pack, no file and no device.

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_exec.h"
#include "audio_parallel.h"
#include "onnx_graph.h"

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

using namespace onebit::npu::pack_import;

// y = x . W for x (1, M, K) and W (K, N): M rows, which is what the executor splits over.
std::vector<float> matmul_at(const char* threads, int64_t M, int64_t K, int64_t N, const std::vector<float>& x,
                             const std::vector<float>& w) {
    if (threads != nullptr) setenv("ONEBIT_AUDIO_THREADS", threads, 1);
    else unsetenv("ONEBIT_AUDIO_THREADS");
    OnnxGraphInfo g;
    g.dir = ".";
    g.inputs = {"x"};
    g.outputs = {"y"};
    OnnxInit wi;
    wi.name = "W";
    wi.dtype = 1;
    wi.dims = {K, N};
    wi.raw = reinterpret_cast<const uint8_t*>(w.data());
    wi.raw_len = w.size() * sizeof(float);
    g.initializers = {"W"};
    g.init_tensors.push_back(wi);
    OnnxNodeInfo n;
    n.op = "MatMul";
    n.name = "mm";
    n.ins = {"x", "W"};
    n.outs = {"y"};
    g.nodes.push_back(n);
    AudioRunResult out;
    std::string why;
    if (!run_audio_graph(g, x.data(), M, K, "y", &out, &why)) {
        std::printf("FAIL MatMul at ONEBIT_AUDIO_THREADS=%s refused: %s\n", threads ? threads : "(unset)", why.c_str());
        ++g_failures;
        return {};
    }
    check(out.dims == std::vector<int64_t>({1, M, N}), "MatMul output is (1, M, N)");
    return out.data;
}

void test_bit_identity() {
    // 37 rows: divisible by none of the thread counts below, so the last worker's share is short at every one of them.
    const int64_t M = 37, K = 53, N = 29;
    std::vector<float> x(size_t(M * K)), w(size_t(K * N));
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(std::sin(0.37 * double(i + 1)) * 3.0);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(std::cos(0.11 * double(i + 3)) * 0.7 + 1e-3 * double(i % 7));
    for (size_t i = 0; i < x.size(); i += 5) x[i] = 0.0f;                    // the zero shortcut is on the path
    for (int64_t k = 0; k < K; ++k) x[size_t(11 * K + k)] = 0.0f;            // and one whole row of zeros

    const std::vector<float> serial = matmul_at("1", M, K, N, x, w);
    check(serial.size() == size_t(M * N), "the serial MatMul produced M*N values");
    if (serial.size() != size_t(M * N)) return;

    // The serial result is the arithmetic the split must not change: an ordered sum over k, per element, in float.
    bool same_as_plain_loop = true;
    for (int64_t i = 0; i < M && same_as_plain_loop; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float av = x[size_t(i * K + k)];
                if (av == 0.0f) continue;
                acc += av * w[size_t(k * N + j)];
            }
            if (std::memcmp(&acc, &serial[size_t(i * N + j)], sizeof acc) != 0) {
                same_as_plain_loop = false;
                break;
            }
        }
    check(same_as_plain_loop, "the serial MatMul is the ordered per-element sum, bit for bit");
    bool zero_row = true;
    for (int64_t j = 0; j < N; ++j) zero_row = zero_row && serial[size_t(11 * N + j)] == 0.0f;
    check(zero_row, "a row of zeros gives a row of zeros");

    for (const char* threads : {"2", "3", "7", "8", "64", static_cast<const char*>(nullptr)}) {
        const std::vector<float> got = matmul_at(threads, M, K, N, x, w);
        const std::string name = threads ? threads : "(unset)";
        check(got.size() == serial.size() && std::memcmp(got.data(), serial.data(), serial.size() * sizeof(float)) == 0,
              "MatMul at ONEBIT_AUDIO_THREADS=" + name + " is bit-identical to the serial result");
    }
    // What is not a thread count means serial, and the answer is still the same one.
    for (const char* threads : {"0", "-4", "many", ""}) {
        setenv("ONEBIT_AUDIO_THREADS", threads, 1);
        check(audio_thread_count() == 1, std::string("ONEBIT_AUDIO_THREADS='") + threads + "' means one thread");
        const std::vector<float> got = matmul_at(threads, M, K, N, x, w);
        check(got.size() == serial.size() && std::memcmp(got.data(), serial.data(), serial.size() * sizeof(float)) == 0,
              std::string("and MatMul at ONEBIT_AUDIO_THREADS='") + threads + "' is the serial result");
    }
    unsetenv("ONEBIT_AUDIO_THREADS");
}

void test_every_unit_once() {
    for (size_t units : {size_t(0), size_t(1), size_t(2), size_t(7), size_t(37), size_t(1000)})
        for (size_t want : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(8), size_t(64), size_t(5000)}) {
            std::vector<std::atomic<int>> seen(units);
            for (auto& s : seen) s = 0;
            parallel_units(units, want, [&](size_t u) { seen[u].fetch_add(1); });
            bool once = true;
            for (auto& s : seen) once = once && s.load() == 1;
            check(once, "every one of " + std::to_string(units) + " units runs exactly once on " + std::to_string(want) +
                            " thread(s)");
        }
}

void test_a_throwing_unit() {
    // If the exception escaped a worker, the process would call std::terminate here and the test would die by signal.
    for (size_t want : {size_t(1), size_t(2), size_t(4), size_t(8)}) {
        std::atomic<int> ran{0};
        bool caught = false;
        std::string what;
        try {
            parallel_units(40, want, [&](size_t u) {
                ran.fetch_add(1);
                if (u == 5) throw std::runtime_error("unit 5 refused");
            });
        } catch (const std::runtime_error& e) {
            caught = true;
            what = e.what();
        }
        check(caught && what == "unit 5 refused",
              "a unit that throws on " + std::to_string(want) + " thread(s) is an exception the caller catches");
        // the other workers ran to the end and were joined before the rethrow: only the throwing worker stopped early
        if (want > 1) {
            const int stopped_worker_left = int((40 - 1 - 5) / want);   // units after 5 in unit 5's worker
            check(ran.load() == 40 - stopped_worker_left,
                  "the other workers finish and are joined before the caller sees it (" + std::to_string(want) + " threads)");
        }
    }
    // Two workers throw: the caller gets the lower worker's, every time.
    for (int rep = 0; rep < 20; ++rep) {
        std::string what;
        try {
            parallel_units(8, 4, [&](size_t u) {
                if (u == 1) throw std::runtime_error("worker 1");
                if (u == 3) throw std::runtime_error("worker 3");
            });
        } catch (const std::runtime_error& e) {
            what = e.what();
        }
        if (what != "worker 1") {
            check(false, "when two workers throw, the lower worker's exception is the one rethrown (got '" + what + "')");
            break;
        }
    }
}

}  // namespace

int main() {
    test_bit_identity();
    test_every_unit_once();
    test_a_throwing_unit();
    if (g_failures == 0) std::printf("audio_parallel: ok\n");
    return g_failures == 0 ? 0 : 1;
}
