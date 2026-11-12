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

// dflash2_probe.cpp -- a DFlash 2 draft against the served 27B: acceptance, and the speculative loop end to end
// (engine#1781). A measurement tool, like tools/mtp_acceptance.cpp: it needs the target pack, the draft package and
// the device, so it is not a ctest.
//
//   loop     dflash2_probe loop <model dir> <kernel dir> <draft dir> <prompt-ids file> <tokens> <reference seq.i32 | ->
//       batched prefill, then rounds of: the draft proposes 7 tokens from the target's tapped hidden states, the
//       target verifies [last, d1..d7] in one step_batch_all() pass, commit_verify_rows() keeps the accepted prefix,
//       and the target's own next token follows. Greedy, so the output must equal plain greedy decoding (checked
//       against the reference sequence when one is given). Prints tokens per round and the time per part.
//       DFLASH2_DEVICE_KERNELS=<dir>   the draft's projections on the NPU (full_i8_<ROLE>_K<K>_N<N>_b8_m16.elf)
//       DFLASH2_K=<1..7>               verify only the first K proposals
//       DFLASH2_FALLBACK=<x>           finish with plain greedy steps once the last DFLASH2_FALLBACK_ROUNDS (3)
//                                      rounds emitted fewer than x tokens on average
//       OMP_WAIT_POLICY=PASSIVE        needed: the draft's OpenMP threads otherwise spin and starve the forward's
//   capture  (device)  ... capture <model dir> <kernel dir> <work dir> <prompt-ids file> <tokens>
//       plain greedy decoding with the draft's layers tapped; DFLASH2_BATCH_PREFILL=1 prefills as forward-serve does
//       (the reference a loop must be compared with), and then the plain decode rate is printed
//   draft    (host)    ... draft <draft dir> <work dir>          the draft's rows at every start position
//   score    (device)  ... score <model dir> <kernel dir> <draft dir> <work dir>   acceptance of those rows
//
// The capture/draft/score stages assume the geometry of FastFlowLM/Qwen3.8-27B-DFlash2-NPU2 (checked at load).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "dflash_draft.h"
#include "dflash_forward.h"
#include "file_create.h"   // onebit_fopen_private_wb
#include "file_map.h"
#include "forward/include/q4nx_dequant.h"
#include "pack_import.h"
#include "q4nx_forward.h"
#include "spec_greedy.h"     // the greedy round and the fallback rule
#include "npu_engine_i8ctx_inc.h"   // I8Ctx: one int8 GEMM design from a full ELF

namespace {

using namespace onebit::npu;
using namespace onebit::npu::dflash;

// The draft's interface, as its config.json states it (read by eye for this prototype, checked against the tensors).
const int kHidden = 5120, kInter = 17408, kLayers = 5, kHeads = 32, kKv = 8, kHeadDim = 128, kBlock = 8;
const int kConvKernel = 2, kConvGroup = 16, kRank = 256, kTopK = 16, kWindow = 2048;
const int kMask = 248070;
const std::vector<int> kTaps = {5, 19, 33, 47, 61};
const float kTheta = 1e7f, kEps = 1e-6f;

void die(const std::string& s) {
    std::fprintf(stderr, "dflash2_probe: %s\n", s.c_str());
    std::exit(1);
}

float bf16(const uint8_t* p) {
    const uint32_t bits = uint32_t(uint16_t(p[0] | (p[1] << 8))) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct Pack {
    const uint8_t* base = nullptr;
    size_t size = 0, payload = 0;
    std::map<std::string, pack_import::Q4nxTensor> t;
    const uint8_t* at(const std::string& n, size_t want) const {
        const auto it = t.find(n);
        if (it == t.end()) die("no tensor " + n);
        if (size_t(it->second.end - it->second.begin) != want)
            die(n + " holds " + std::to_string(it->second.end - it->second.begin) + " B, expected " + std::to_string(want));
        return base + payload + size_t(it->second.begin);
    }
    std::vector<float> plain(const std::string& n, size_t count) const {
        const uint8_t* p = at(n, count * 2);
        std::vector<float> v(count);
        for (size_t i = 0; i < count; i++) v[i] = bf16(p + i * 2);
        return v;
    }
    std::vector<float> tiles(const std::string& n, int rows, int cols) const {
        // This package's projections are Q4_K chunks (4736 B per 32 x 256), not the 5120 B q4_1 tiles of the Bolt draft.
        const size_t bytes = size_t(rows / 32) * size_t(cols / 256) * size_t(Q4NX_CHUNK_Q4K);
        const uint8_t* p = at(n, bytes);
        std::vector<float> v(size_t(rows) * size_t(cols));
        if (q4nx_dequant_tensor_chunked(p, bytes, rows, cols, Q4NX_CHUNK_Q4K, v.data()) != 0) die(n + " does not decode");
        return v;
    }
};

Pack open_pack(const std::string& dir) {
    Pack p;
    const std::string path = dir + "/dflash_weight.q4nx";
    const pack_import::Q4nxContainer c = pack_import::read_q4nx_container(path);
    if (!c.why.empty()) die(c.why);
    std::string why;
    p.base = map_file(path, p.size, &why);
    if (!p.base) die(why);
    p.payload = 8 + size_t(c.header_bytes);
    for (const auto& t : c.tensors) p.t[t.name] = t;
    return p;
}

// The package through the engine's own loader (generation 2), checked against the constants this prototype assumes.
DraftWeights load_weights(const std::string& dir) {
    const DraftModel m = load_draft(dir);
    const DraftConfig& c = m.cfg;
    if (c.generation != 2 || c.hidden != kHidden || c.block_size != kBlock || c.mask_token_id != kMask ||
        c.target_layer_ids != kTaps || c.sliding_window != kWindow || c.selector_rank != kRank || c.selector_top_k != kTopK)
        die("the package is not the DFlash 2 draft this prototype was written for");
    return decode_draft(m);
}

template <class T>
void save(const std::string& path, const std::vector<T>& v) {
    FILE* f = onebit_fopen_private_wb(path.c_str());   // stage outputs are this lane's own files
    if (!f || std::fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) die("cannot write " + path);
    std::fclose(f);
}
template <class T>
std::vector<T> load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot read " + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<T> v(size_t(n) / sizeof(T));
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) die("short read of " + path);
    std::fclose(f);
    return v;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int capture(int argc, char** argv) {
    if (argc != 7) die("capture <model dir> <kernel dir> <work dir> <prompt-ids file> <tokens>");
    const std::string model = argv[2], kern = argv[3], work = argv[4];
    const int tokens = std::atoi(argv[6]);
    std::vector<int> seq;
    {
        std::ifstream in(argv[5]);
        int v;
        while (in >> v) seq.push_back(v);
    }
    if (seq.empty() || tokens < kBlock) die("no prompt ids, or fewer tokens than a block");
    const int P = int(seq.size());
    xrt::device dev(0);
    Q4nxNpuForward fwd;
    if (!fwd.init(dev, (model + "/model.q4nx").c_str(), kern)) die(std::string("target: ") + fwd.last_error());
    std::vector<float> taps, logits, row;
    if (!fwd.set_layer_taps(kTaps, &taps)) die("the draft's layers do not fit this target");
    const double t0 = now_s();
    // DFLASH2_BATCH_PREFILL=1: the prompt in one batched pass, as forward-serve and the loop run it, then step().
    int first = 0;
    if (getenv("DFLASH2_BATCH_PREFILL")) {
        const std::vector<int> pr(seq.begin(), seq.begin() + P);
        if (!fwd.step_batch(pr, 0, logits)) die(std::string("prefill: ") + fwd.last_error());
        seq.push_back(int(std::max_element(logits.begin(), logits.end()) - logits.begin()));
        first = P;
        std::printf("capture: prefill %d tokens in %.2f s\n", P, now_s() - t0);
    }
    const double t_dec0 = now_s();
    for (int pos = first; pos < P + tokens; pos++) {
        if (!fwd.step(seq[size_t(pos)], pos, logits)) die(std::string("step: ") + fwd.last_error());
        if (pos >= P - 1) seq.push_back(int(std::max_element(logits.begin(), logits.end()) - logits.begin()));
    }
    const double dt = now_s() - t0;
    if (first == P) std::printf("capture: plain greedy decode %d tokens in %.2f s = %.2f tok/s\n", tokens, now_s() - t_dec0, double(tokens) / (now_s() - t_dec0));
    std::vector<float> emb;
    for (size_t i = 0; i < seq.size(); i++) {
        fwd.embed_row(seq[i], row);
        emb.insert(emb.end(), row.begin(), row.end());
    }
    fwd.embed_row(kMask, row);
    save(work + "/seq.i32", seq);
    save(work + "/taps.f32", taps);
    save(work + "/emb.f32", emb);
    save(work + "/mask.f32", row);
    save(work + "/prompt_len.i32", std::vector<int>{P});
    std::printf("capture: %d prompt + %d greedy tokens in %.1f s (%.2f steps/s); taps %zu floats; H %zu\n", P, tokens, dt,
                double(P + tokens) / dt, taps.size(), row.size());
    std::printf("greedy:");
    for (int i = P; i < P + tokens; i++) std::printf(" %d", seq[size_t(i)]);
    std::printf("\n");
    return 0;
}

int draft(int argc, char** argv) {
    if (argc != 4) die("draft <draft dir> <work dir>");
    const std::string work = argv[3];
    const DraftWeights w = load_weights(argv[2]);
    const std::vector<int> seq = load<int>(work + "/seq.i32");
    const std::vector<float> taps = load<float>(work + "/taps.f32"), emb = load<float>(work + "/emb.f32"),
                             mask = load<float>(work + "/mask.f32");
    const int P = load<int>(work + "/prompt_len.i32")[0];
    const size_t H = size_t(kHidden);
    const int positions = int(taps.size() / (H * kTaps.size()));   // positions the target ran: seq[0 .. positions-1]
    const int last_start = positions - 1;                           // seq[s] with s <= positions - 1 has a context of s rows
    const double t0 = now_s();
    const std::vector<float> fused = fuse_context(w, taps.data(), positions);
    std::printf("draft: context of %d rows fused in %.1f s\n", positions, now_s() - t0);
    std::fflush(stdout);
    // Starts P .. last_start - (kBlock - 1) can be scored on all 7 proposals; later ones only on a prefix, so stop there.
    std::vector<float> all;
    std::vector<int> starts;
    for (int s = P; s + kBlock - 1 <= last_start + 1; s++) {
        std::vector<float> noise(size_t(kBlock) * H);
        std::copy(emb.begin() + ptrdiff_t(size_t(s) * H), emb.begin() + ptrdiff_t(size_t(s + 1) * H), noise.begin());
        for (int i = 1; i < kBlock; i++) std::copy(mask.begin(), mask.end(), noise.begin() + ptrdiff_t(size_t(i) * H));
        const double t1 = now_s();
        const std::vector<float> out = draft_forward_fused(w, fused.data(), s, noise.data(), kBlock, 0, kWindow);
        all.insert(all.end(), out.begin() + ptrdiff_t(H), out.end());   // rows 1..7: the proposals
        starts.push_back(s);
        std::printf("draft: start %d (%d context rows) in %.1f s\n", s, s, now_s() - t1);
        std::fflush(stdout);
    }
    save(work + "/draft_rows.f32", all);
    save(work + "/starts.i32", starts);
    std::printf("draft: %zu start positions, %zu rows written\n", starts.size(), all.size() / H);
    return 0;
}

int score(int argc, char** argv) {
    if (argc != 6) die("score <model dir> <kernel dir> <draft dir> <work dir>");
    const std::string model = argv[2], kern = argv[3], work = argv[5];
    const Pack p = open_pack(argv[4]);
    const size_t H = size_t(kHidden), R = size_t(kRank);
    const std::vector<float> hproj = p.plain("model.candidate_selector.hidden_projection.weight", R * H);
    const std::vector<int> seq = load<int>(work + "/seq.i32"), starts = load<int>(work + "/starts.i32");
    const std::vector<float> rows = load<float>(work + "/draft_rows.f32");
    const int V = 248320;
    const uint8_t* pred = p.at("model.candidate_selector.predecessor_codebook", size_t(V) * R * 2);
    const uint8_t* succ = p.at("model.candidate_selector.successor_codebook", size_t(V) * R * 2);
    xrt::device dev(0);
    Q4nxNpuForward fwd;
    if (!fwd.init(dev, (model + "/model.q4nx").c_str(), kern)) die(std::string("target: ") + fwd.last_error());
    const int per = kBlock - 1;
    std::vector<int> acc_sel(starts.size()), acc_arg(starts.size());
    long first_sel = 0, first_arg = 0, in_topk = 0, rows_seen = 0;
    for (size_t si = 0; si < starts.size(); si++) {
        const int s = starts[si];
        int prev = seq[size_t(s)];
        int a_sel = 0, a_arg = 0;
        bool open_sel = true, open_arg = true;
        for (int i = 0; i < per; i++) {
            const float* h = rows.data() + (si * size_t(per) + size_t(i)) * H;
            std::vector<float> hv(h, h + H), lg;
            if (!fwd.lm_head_logits(hv, lg)) die(std::string("head: ") + fwd.last_error());
            const int want = seq[size_t(s + 1 + i)];
            // plain argmax of the head
            const int arg = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
            // the selector: top-k candidates, re-scored against the previous pick
            std::vector<int> idx(lg.size());
            for (size_t j = 0; j < idx.size(); j++) idx[j] = int(j);
            std::partial_sort(idx.begin(), idx.begin() + kTopK, idx.end(), [&](int a, int b) { return lg[size_t(a)] > lg[size_t(b)]; });
            std::vector<double> hp(R, 0.0);
            for (size_t r = 0; r < R; r++) {
                double acc = 0.0;
                const float* wr = hproj.data() + r * H;
                for (size_t k = 0; k < H; k++) acc += double(wr[k]) * double(h[k]);
                hp[r] = acc * double(bf16(pred + (size_t(prev) * R + r) * 2));
            }
            int pick = idx[0];
            double best = -1e300;
            bool hit_topk = false;
            for (int c = 0; c < kTopK; c++) {
                const int cand = idx[size_t(c)];
                hit_topk = hit_topk || cand == want;
                double sc = lg[size_t(cand)];
                for (size_t r = 0; r < R; r++) sc += hp[r] * double(bf16(succ + (size_t(cand) * R + r) * 2));
                if (sc > best) { best = sc; pick = cand; }
            }
            rows_seen++;
            in_topk += hit_topk ? 1 : 0;
            if (i == 0) { first_sel += pick == want; first_arg += arg == want; }
            if (open_sel && pick == want) a_sel++; else open_sel = false;
            if (open_arg && arg == want) a_arg++; else open_arg = false;
            prev = pick;   // the chain follows the draft's own pick, as it would in a live round
        }
        acc_sel[si] = a_sel;
        acc_arg[si] = a_arg;
    }
    auto walk = [&](const std::vector<int>& acc, const char* name) {
        long passes = 0, accepted = 0;
        std::vector<long> hist(size_t(kBlock), 0);
        for (size_t si = 0; si < starts.size();) {
            passes++;
            accepted += acc[si];
            hist[size_t(acc[si])]++;
            si += size_t(acc[si]) + 1;
        }
        std::printf("%s: speculative walk %ld passes, %ld accepted, %.3f accepted per pass, %.3f tokens per pass; histogram",
                    name, passes, accepted, double(accepted) / double(passes), double(accepted + passes) / double(passes));
        for (int a = 0; a < kBlock; a++) std::printf(" %d:%ld", a, hist[size_t(a)]);
        long sum = 0;
        for (int a : acc) sum += a;
        std::printf("; every position: %.3f accepted per pass over %zu starts\n", double(sum) / double(acc.size()), acc.size());
    };
    walk(acc_sel, "selector");
    walk(acc_arg, "argmax  ");
    std::printf("first proposal right: selector %ld of %zu, argmax %ld of %zu; target token inside the top %d: %ld of %ld rows\n",
                first_sel, starts.size(), first_arg, starts.size(), kTopK, in_topk, rows_seen);
    return 0;
}

// The live loop (device target, host draft): prefill, then rounds of draft 7 -> verify 8 rows in one batched pass
// -> commit the accepted prefix -> the target's own token after it. Greedy. Emits the same tokens plain greedy
// decoding would (checked against `ref`, the capture stage's seq.i32, when given), and times each part.
// The draft's projections on the NPU: one I8Ctx per kernel design (shared by every layer that has that shape), one
// packed weight BO per projection, M=16 designs, rows walked in blocks of 16. DFLASH2_DEVICE_KERNELS=<dir> turns it
// on; the directory holds full_i8_<ROLE>_K<K>_N<N>_b8_m16.elf.
struct DevProj {
    I8Ctx* ctx = nullptr;
    std::unique_ptr<xrt::bo> bo;
    std::vector<float> scales;   // per output column
    int K = 0, N = 0;
};
struct DevDraft {
    std::map<std::string, std::unique_ptr<I8Ctx>> designs;   // by "KxN"
    std::map<const float*, DevProj> proj;                      // by the host matrix's address
    double ms = 0.0;
    long launches = 0;
};

bool dev_gemm(const float* W, int n_out, int n_in, const float* x, int rows, float* y, void* user) {
    DevDraft& d = *static_cast<DevDraft*>(user);
    const auto it = d.proj.find(W);
    if (it == d.proj.end()) return false;   // not a projection we placed (the convolutions, the selector): host
    DevProj& p = it->second;
    if (p.K != n_in || p.N != n_out) return false;
    const double t0 = now_s();
    I8Ctx& c = *p.ctx;
    for (int r0 = 0; r0 < rows; r0 += 16) {
        const int m = std::min(16, rows - r0);
        const float* A = x + size_t(r0) * size_t(n_in);
        std::vector<float> asc(static_cast<size_t>(m));
        for (int r = 0; r < m; r++) {
            float am = 0.0f;
            for (int k = 0; k < n_in; k++) am = std::max(am, std::fabs(A[size_t(r) * size_t(n_in) + size_t(k)]));
            asc[size_t(r)] = am > 0.0f ? am / 127.0f : 1.0f;
        }
        xrt::run run = c.launch_async_with_bo_rows(*p.bo, A, m, n_in, asc.data());
        c.wait_kernel(run);
        c.readback();
        for (int r = 0; r < m; r++)
            for (int n = 0; n < n_out; n++)
                y[size_t(r0 + r) * size_t(n_out) + size_t(n)] =
                    float(c.Cm[size_t(r) * size_t(c.ND) + size_t(n)]) * asc[size_t(r)] * p.scales[size_t(n)];
        d.launches++;
    }
    d.ms += 1e3 * (now_s() - t0);
    return true;
}

// Place one projection: W is row-major [N][K] on the host, the packer wants [K][N].
void dev_place(DevDraft& d, xrt::device& dev, const std::string& kdir, const std::string& role, const std::vector<float>& W,
               int N, int K) {
    const std::string key = std::to_string(K) + "x" + std::to_string(N);
    auto& slot = d.designs[key];
    if (!slot) {
        slot = std::make_unique<I8Ctx>();
        slot->blocked_b = true;
        const std::string elf = kdir + "/full_i8_" + role + "_K" + std::to_string(K) + "_N" + std::to_string(N) + "_b8_m16.elf";
        if (!slot->init_elf(dev, elf.c_str(), 16, K, N, 1)) die("cannot open " + elf);
    }
    std::vector<float> t(size_t(K) * size_t(N));
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) t[size_t(k) * size_t(N) + size_t(n)] = W[size_t(n) * size_t(K) + size_t(k)];
    DevProj p;
    p.ctx = slot.get();
    p.bo = slot->make_weight_bo(dev);
    float sout = 1.0f;
    slot->packB_into(*p.bo, t.data(), K, N, sout, p.scales);
    p.K = K;
    p.N = N;
    d.proj[W.data()] = std::move(p);
}

int loop(int argc, char** argv) {
    if (argc != 8) die("loop <model dir> <kernel dir> <draft dir> <prompt-ids file> <tokens> <reference seq.i32 or ->");
    const std::string model = argv[2], kern = argv[3], ddir = argv[4];
    const int want_tokens = std::atoi(argv[6]);
    std::vector<int> prompt;
    {
        std::ifstream in(argv[5]);
        int v;
        while (in >> v) prompt.push_back(v);
    }
    std::vector<int> ref;
    if (std::string(argv[7]) != "-") ref = load<int>(argv[7]);
    const int P = int(prompt.size());
    const DraftModel dm = load_draft(ddir);
    const DraftWeights w = decode_draft(dm);
    const DraftConfig& dc = dm.cfg;
    const size_t H = size_t(dc.hidden), R = size_t(dc.selector_rank);
    const std::vector<float> hproj = decode_plain(dm.selector_hproj);
    const int k = dc.selector_top_k, block = dc.block_size;
    // DFLASH2_K=<1..block-1>: verify only the first K proposals (the draft pass is the same block either way).
    const int per = getenv("DFLASH2_K") ? std::max(1, std::min(block - 1, std::atoi(getenv("DFLASH2_K")))) : block - 1;
    std::printf("loop: verifying %d proposal(s) per round (%d rows)\n", per, per + 1);

    xrt::device dev(0);
    Q4nxNpuForward fwd;
    if (!fwd.init(dev, (model + "/model.q4nx").c_str(), kern)) die(std::string("target: ") + fwd.last_error());
    std::vector<float> taps, logits, row, logits_all;
    if (!fwd.set_layer_taps(dc.target_layer_ids, &taps)) die("the draft's layers do not fit this target");
    const size_t T = dc.target_layer_ids.size(), rowf = T * H;
    DevDraft devd;
    if (const char* kd = getenv("DFLASH2_DEVICE_KERNELS")) {
        const double p0 = now_s();
        for (const DraftLayerWeights& L : w.layers) {
            dev_place(devd, dev, kd, "Q", L.q, dc.q_dim(), dc.hidden);
            dev_place(devd, dev, kd, "KV", L.k, dc.kv_dim(), dc.hidden);
            dev_place(devd, dev, kd, "KV", L.v, dc.kv_dim(), dc.hidden);
            dev_place(devd, dev, kd, "O", L.o, dc.hidden, dc.q_dim());
            dev_place(devd, dev, kd, "G", L.gate, dc.intermediate, dc.hidden);
            dev_place(devd, dev, kd, "U", L.up, dc.intermediate, dc.hidden);
            dev_place(devd, dev, kd, "D", L.down, dc.hidden, dc.intermediate);
        }
        dev_place(devd, dev, kd, "FC", w.fc, dc.hidden, dc.hidden * int(dc.target_layer_ids.size()));
        set_gemm_hook(&dev_gemm, &devd);
        std::printf("loop: the draft's projections on the NPU: %zu designs, %zu projections packed in %.1f s\n",
                    devd.designs.size(), devd.proj.size(), now_s() - p0);
    }

    const double t0 = now_s();
    if (!fwd.step_batch(prompt, 0, logits)) die(std::string("prefill: ") + fwd.last_error());
    const double t_prefill = now_s() - t0;
    std::vector<int> out;   // generated tokens, in order
    int cur = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    out.push_back(cur);
    int pos = P;            // the position `cur` will be run at
    DraftKvCache kvc;   // the draft's context keys and values, grown by the committed rows each round
    {
        const std::vector<float> f = fuse_context(w, taps.data(), P);
        cache_extend(w, kvc, f.data(), P);
    }
    std::vector<float> mask_emb;
    fwd.embed_row(int(dc.mask_token_id), mask_emb);
    double t_draft = 0, t_head = 0, t_verify = 0, t_fuse = 0;
    long rounds = 0, accepted_total = 0;
    std::vector<long> hist(size_t(block), 0);
    const double t_gen0 = now_s();
    // DFLASH2_FALLBACK=<tokens per round>: after the first DFLASH2_FALLBACK_ROUNDS rounds (default 3), a request whose
    // rounds emitted fewer tokens than that on average stops speculating and finishes with plain greedy steps.
    const double fallback_below = getenv("DFLASH2_FALLBACK") ? std::atof(getenv("DFLASH2_FALLBACK")) : 0.0;
    const int fallback_after = getenv("DFLASH2_FALLBACK_ROUNDS") ? std::max(1, std::atoi(getenv("DFLASH2_FALLBACK_ROUNDS"))) : 3;
    // The rule (spec_greedy.h) looks at the last DFLASH2_FALLBACK_ROUNDS rounds only, so a request whose first rounds
    // happened to accept well does not keep speculating on that credit.
    SpecFallback fallback(fallback_below, fallback_after);
    bool plain = false;
    long plain_steps = 0;
    while (int(out.size()) < want_tokens) {
        if (!plain && fallback.off()) {
            plain = true;
            std::printf("loop: falling back to plain decoding after %ld rounds (last %d rounds below %.2f tokens per round)\n",
                        rounds, fallback_after, fallback_below);
        }
        if (plain) {
            if (!fwd.step(cur, pos, logits)) die(std::string("step: ") + fwd.last_error());
            cur = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            out.push_back(cur);
            pos++;
            plain_steps++;
            continue;
        }
        // draft
        double a0 = now_s();
        std::vector<float> noise(size_t(block) * H);
        fwd.embed_row(cur, row);
        std::copy(row.begin(), row.end(), noise.begin());
        for (int i = 1; i < block; i++) std::copy(mask_emb.begin(), mask_emb.end(), noise.begin() + ptrdiff_t(size_t(i) * H));
        const std::vector<float> hrows = draft_forward_cached(w, kvc, noise.data(), block, dc.sliding_window);
        t_draft += now_s() - a0;
        // selector: the target's head on each drafted row, then the pairwise re-score against the previous pick
        std::vector<int> drafts;
        int prev = cur;
        std::vector<float> prow(R), hp(R), succ(size_t(k) * R);
        for (int i = 1; i <= per; i++) {
            const float* h = hrows.data() + size_t(i) * H;
            std::vector<float> hv(h, h + H), lg;
            double b0 = now_s();
            if (!fwd.lm_head_logits(hv, lg)) die(std::string("head: ") + fwd.last_error());
            t_head += now_s() - b0;
            b0 = now_s();
            std::vector<int> idx(lg.size());
            for (size_t j = 0; j < idx.size(); j++) idx[j] = int(j);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[size_t(a)] > lg[size_t(b)]; });
            for (size_t r = 0; r < R; r++) {
                double acc = 0.0;
                const float* wr = hproj.data() + r * H;
                for (size_t q = 0; q < H; q++) acc += double(wr[q]) * double(h[q]);
                hp[r] = float(acc);
            }
            decode_row(dm.selector_pred, prev, prow.data());
            std::vector<float> unary(static_cast<size_t>(k));
            for (int c = 0; c < k; c++) {
                unary[size_t(c)] = lg[size_t(idx[size_t(c)])];
                decode_row(dm.selector_succ, idx[size_t(c)], succ.data() + size_t(c) * R);
            }
            const int pick = idx[size_t(selector_pick(unary.data(), k, prow.data(), hp.data(), succ.data(), int(R)))];
            drafts.push_back(pick);
            prev = pick;
            t_draft += now_s() - b0;
        }
        // verify: [cur, d1..d7] in one batched pass; row i predicts the token after position pos + i
        double c0 = now_s();
        std::vector<int> vt = {cur};
        vt.insert(vt.end(), drafts.begin(), drafts.end());
        const size_t tap_base = taps.size();
        fwd.begin_verify(int(vt.size()), pos);
        if (!fwd.step_batch_all(vt, pos, logits_all)) die(std::string("verify: ") + fwd.last_error());
        const size_t NV = logits_all.size() / vt.size();
        std::vector<int> choices(vt.size());
        for (size_t i = 0; i < vt.size(); i++) {
            const float* lr = logits_all.data() + i * NV;
            choices[i] = int(std::max_element(lr, lr + NV) - lr);
        }
        const GreedyRound gr = greedy_round(drafts.data(), per, choices.data(), int(choices.size()));
        if (gr.accepted < 0) die("verify: the pass returned fewer rows than it was given");
        const int a = gr.accepted, next = gr.next;
        if (!fwd.commit_verify_rows(a + 1)) die(std::string("commit: ") + fwd.last_error());
        taps.resize(tap_base + size_t(a + 1) * rowf);   // keep the rows of the committed positions only
        t_verify += now_s() - c0;
        // emit
        for (int i = 0; i < a; i++) out.push_back(drafts[size_t(i)]);
        out.push_back(next);
        rounds++;
        fallback.record(a + 1);
        accepted_total += a;
        hist[size_t(a)]++;
        // extend the fused context by the committed rows
        double f0 = now_s();
        const std::vector<float> add = fuse_context(w, taps.data() + tap_base, a + 1);
        cache_extend(w, kvc, add.data(), a + 1);
        t_fuse += now_s() - f0;
        pos += a + 1;
        cur = next;
    }
    const double t_gen = now_s() - t_gen0;
    const int n = std::min(want_tokens, int(out.size()));
    int same = 0;
    if (!ref.empty())
        while (same < n && P + same < int(ref.size()) && ref[size_t(P + same)] == out[size_t(same)]) same++;
    if (plain_steps) std::printf("loop: %ld plain step(s) after the fallback\n", plain_steps);
    std::printf("loop: prefill %d tokens in %.2f s; %d tokens in %ld rounds, %.3f accepted per round, %.3f tokens per round; histogram",
                P, t_prefill, n, rounds, double(accepted_total) / double(rounds), double(n) / double(rounds));
    for (int i = 0; i < block; i++) std::printf(" %d:%ld", i, hist[size_t(i)]);
    std::printf("\nloop: generation %.2f s = %.2f tok/s; per round: draft (host) %.0f ms, head (device) %.0f ms, verify (device) %.0f ms, fuse (host) %.0f ms\n",
                t_gen, double(n) / t_gen, 1e3 * t_draft / rounds, 1e3 * t_head / rounds, 1e3 * t_verify / rounds, 1e3 * t_fuse / rounds);
    if (devd.launches)
        std::printf("loop: of the draft, its projections on the NPU took %.0f ms per round (%ld launches in all)\n",
                    devd.ms / double(rounds), devd.launches);
    if (!ref.empty()) std::printf("loop: the first %d of %d tokens equal plain greedy decoding%s\n", same, n, same == n ? "" : " (then they part)");
    std::printf("tokens:");
    for (int i = 0; i < n; i++) std::printf(" %d", out[size_t(i)]);
    std::printf("\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) die("capture | draft | score");
    const std::string m = argv[1];
    if (m == "capture") return capture(argc, argv);
    if (m == "draft") return draft(argc, argv);
    if (m == "score") return score(argc, argv);
    if (m == "loop") return loop(argc, argv);
    die("unknown stage " + m);
}
