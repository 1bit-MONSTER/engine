// golden_cpu: compare the CPU reference against golden logits from HF transformers.
//
// usage: golden_cpu <model.gguf> <golden_dir> [--max-kl X] [--threads N]
//
// golden_dir (written by tools/golden/make_golden.py) holds:
//   tokens.txt  whitespace-separated token ids (prompt followed by the reference's greedy continuation)
//   logits.f32  float32 [n_tokens][n_vocab]: the reference's next-token logits after each token
//
// The model is fed every token teacher-forced. The test passes when the argmax
// agrees at every position and every per-position KL(ref || cpu) is <= max_kl.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cpu_model.h"

using namespace onebit;

namespace {

std::vector<double> log_softmax(const float* x, size_t n) {
    const double mx = *std::max_element(x, x + n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::exp(double(x[i]) - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = double(x[i]) - lse;
    return out;
}

double kl(const float* ref, const float* got, size_t n) {
    auto lp = log_softmax(ref, n), lq = log_softmax(got, n);
    double d = 0.0;
    for (size_t i = 0; i < n; ++i) d += std::exp(lp[i]) * (lp[i] - lq[i]);
    return d;
}

size_t argmax(const float* x, size_t n) { return size_t(std::max_element(x, x + n) - x); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <golden_dir> [--max-kl X] [--threads N]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], dir = argv[2];
    double max_kl = 1e-6;
    size_t threads = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        if (flag == "--max-kl") max_kl = std::atof(argv[i + 1]);
        else if (flag == "--threads") threads = std::strtoul(argv[i + 1], nullptr, 10);
        else {
            std::fprintf(stderr, "unknown flag %s\n", flag.c_str());
            return 2;
        }
    }

    std::vector<int32_t> tokens;
    {
        std::ifstream in(dir + "/tokens.txt");
        for (int32_t t; in >> t;) tokens.push_back(t);
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "no tokens in %s/tokens.txt\n", dir.c_str());
        return 2;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto model = CpuModel::load(model_path, threads);
    if (!model) {
        std::fprintf(stderr, "load failed: %s\n", model.error().c_str());
        return 1;
    }
    const size_t V = model->config().n_vocab;
    auto t1 = std::chrono::steady_clock::now();

    std::vector<float> ref(tokens.size() * V);
    {
        std::ifstream in(dir + "/logits.f32", std::ios::binary);
        in.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size() * sizeof(float)));
        if (!in || in.peek() != EOF) {
            std::fprintf(stderr, "logits.f32 is not %zu x %zu floats\n", tokens.size(), V);
            return 2;
        }
    }

    double worst_kl = 0.0, worst_abs = 0.0;
    size_t mismatches = 0;
    for (size_t p = 0; p < tokens.size(); ++p) {
        auto logits = model->forward(tokens[p]);
        if (!logits) {
            std::fprintf(stderr, "forward failed at %zu: %s\n", p, logits.error().c_str());
            return 1;
        }
        const float* r = ref.data() + p * V;
        const float* g = (*logits)->data();
        const double k = kl(r, g, V);
        double a = 0.0;
        for (size_t i = 0; i < V; ++i) a = std::max(a, double(std::fabs(r[i] - g[i])));
        const size_t ar = argmax(r, V), ag = argmax(g, V);
        if (ar != ag) ++mismatches;
        worst_kl = std::max(worst_kl, k);
        worst_abs = std::max(worst_abs, a);
        std::printf("pos %3zu tok %6d  argmax ref %6zu cpu %6zu %s  KL %.3e  max|dlogit| %.3e\n", p, tokens[p], ar, ag,
                    ar == ag ? "  " : "!!", k, a);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double load_s = std::chrono::duration<double>(t1 - t0).count();
    const double run_s = std::chrono::duration<double>(t2 - t1).count();

    const bool pass = mismatches == 0 && worst_kl <= max_kl;
    std::printf("\n%zu positions, argmax mismatches %zu, worst KL %.3e (gate %.1e), worst max|dlogit| %.3e\n",
                tokens.size(), mismatches, worst_kl, max_kl, worst_abs);
    std::printf("load %.1f s, %.2f tok/s teacher-forced\n", load_s, tokens.size() / run_s);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
