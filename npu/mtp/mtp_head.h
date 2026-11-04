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

// mtp_head.h -- the MTP draft head on the host, in the arithmetic our containers actually hold.
//
// The head is one full-attention layer with an input-fusion GEMM in front (see tools/mtp_head_ref.py for the
// reference and npu-re/docs/FACTS-flm-q4nx-container-and-mtp-head.md for the container).  It runs here in the
// pb16i-exact form: per 32-K group a signed int4 code and an int16 magnitude, then the source's additive
// min as a host table --  y[n] = sum_g (sum_k a_k code_kn) ms_gn 2^e_ref[n]  +  sum_g A_g add_gn.
// That is the same decomposition the device does (npu-arena/HARVEST.md), so a draft measured here is a draft
// the NPU head can reproduce; and the final projection to vocabulary goes through the BASE model's lm head,
// because the drafter does not have one of its own.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mtp/draft.h"

namespace onebit::npu::mtp {

constexpr int kH = 5120, kIM = 17408, kHD = 256, kNH = 24, kNKV = 4, kRot = 64;
constexpr int kDQ = kNH * kHD;      // 6144 (q, and again the output gate)
constexpr int kKV = kNKV * kHD;     // 1024
constexpr int kG = 32;              // K values per scale group

// One matrix in the pb16i-exact form: codes [K][N] int8, magnitudes [K/32][N] int16, one exponent per column,
// plus the source's additive min as [K/32][N] float.
struct Matrix {
    int K = 0, N = 0;
    std::vector<int8_t> codes;      // K*N, code-major (k*N + n)
    std::vector<int16_t> ms;        // (K/32)*N, group-major (g*N + n)
    std::vector<int16_t> eref;      // N
    std::vector<float> add;         // (K/32)*N
    bool empty() const { return codes.empty(); }

    // y[N] = matvec(this, a[K]) exactly as the device plus the host table compute it.
    void matvec(const float* a, float* y) const;
};

// Which block of the fc input comes first.  The container's descriptor says "embedding + hidden
// concatenated"; which half is which is settled by measurement (acceptance), never by assumption.
extern bool g_fc_hidden_first;

struct Weights {
    Matrix fc, q, k, v, o, gate, up, down;
    std::vector<float> pre_fc_emb, pre_fc_hid, ln_in, ln_post, norm, q_norm, k_norm;  // BF16 -> float
};

struct KvCache {
    std::vector<float> k, v;   // [t][NKV][HD], grown per step
    int t = 0;
    void reset() { k.clear(); v.clear(); t = 0; }
    // Drop the provisional positions a draft chain appended, keeping the committed prefix.
    void truncate(int keep) {
        if (keep < 0) keep = 0;
        if (keep >= t) return;
        k.resize((size_t)keep * kKV);
        v.resize((size_t)keep * kKV);
        t = keep;
    }
};

// Load ~/models/npu/bonsai2-27b-mtp-head/pb16i-exact (manifest.json + <t>.pb16i/.add/.eref) and the q4_k norms.
bool load_weights(const std::string& exact_dir, const std::string& q4k_dir, Weights& w, std::string& err);

// One head step: hidden (H) and the accepted token's embedding (H) in, the head's hidden out, and -- when
// `logits_out` is not null -- the vocabulary logits through `lm_head` (a callback into the base model).
// Returns false only on a shape mismatch.
bool head_step(const Weights& w, const float* hidden, const float* tok_emb, int pos, KvCache& cache,
               std::vector<float>& out_hidden, const std::function<bool(const std::vector<float>&,
                                                                      std::vector<float>&)>& lm_head,
               std::vector<float>& logits_out);

// argmax with the winning token's own probability under a softmax over the whole vocabulary (the tau gate needs
// the probability, not just the token).
int argmax_prob(const std::vector<float>& logits, float* prob);

// One depth of the live-loop-vs-offline-protocol comparison, computed from the two tables' own counts.  `scored`
// is false as soon as either side has no denominator: a depth no chain reached is not a 0 % that agrees with
// another 0 %, it is unmeasured, and reporting it as "delta +0.00 pp PASS" turns "nothing was measured" into a
// pass -- which is what happens at tau>0, where the chain gate stops before depths 3-4.
struct DepthVerdict {
    bool scored = false;
    double live = 0, offline = 0, delta = 0;
    bool within = false;
};
DepthVerdict depth_verdict(int live_hits, int live_reached, int offline_hits, int offline_reached, double bar_pp = 2.0);

// The drafter: one head step per proposed token, chained on its own hidden state and its own KV stream.  It lives
// here rather than in the measurement tool because (a) the serving route needs the same class and (b) its stream
// cursor is the one part of the live loop that must be testable without the 1.6 GiB head -- the class of bug it
// guards is silent: drafts get worse and nothing crashes.
//
// The stream IS the drafter's quality: the MTP layer attends over its own states, so its cache has to be the base
// model's committed stream row by row (row p = post-norm hidden after token at p, with that token's embedding).
// A drafter that sees only the current chain drafts against a stream that does not exist -- measured on the 27B,
// that is the whole 69 % / 3.19-tokens-per-pass -> 14 % / 0.7 gap (TMTPLOOP-STATE, #811).
class MtpDrafter : public DraftHead {
public:
    // token -> embedding row; hidden -> vocabulary logits (the drafter has no lm head of its own).
    using EmbedFn = std::function<bool(int, std::vector<float>&)>;
    using LmHeadFn = std::function<bool(const std::vector<float>&, std::vector<float>&)>;

    MtpDrafter(const Weights& w, EmbedFn embed, LmHeadFn lm, int hidden = kH)
        : w_(w), embed_(std::move(embed)), lm_(std::move(lm)), hidden_(hidden) {}

    void reset() override { cache_.reset(); stream_.reset(); mismatches_ = 0; }

    int draft(const std::vector<float>& hidden, int last_token, int pos, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override;
    void commit_rows(const std::vector<int>& tokens, const std::vector<float>& hidden_rows, int M) override;

    // Put one position of the base model's committed stream into the drafter (the prompt, or a step the loop took
    // outside the batch).  False when `pos` is not where the committed stream ends -- the cursor is the contract.
    bool prime(const std::vector<float>& hidden, int token, int pos);

    void set_tau(float t) { tau_ = t; }
    void set_lm_head(bool on) { use_lm_head_ = on; }
    void set_vocab(int nv) { vocab_ = nv; }
    // Full-vocabulary logits calls made so far.  One call is ~0.3-0.5 s on the device (15 ternary-image
    // dispatches at M=16 for one useful row), so the loop's cost is this counter divided by the tokens it
    // produced -- printing it is how "the head is expensive" becomes "these calls per iteration".
    long lm_calls() const { return lm_calls_; }

    const DraftStream& stream() const { return stream_; }
    // Chains opened at a position the cursor disagreed with.  Non-zero means the loop and the drafter are feeding
    // different streams; the run is still text-correct, the drafts are worthless.
    int stream_mismatches() const { return mismatches_; }
    double head_ms = 0;

protected:
    // One row of the drafter's stream: (hidden, token) at `pos` -> that position's hidden and, when asked, the
    // vocabulary logits.  The only thing a test replaces to exercise the cursor without the head's weights.
    virtual bool next_row(const float* hidden, int token, int pos, std::vector<float>& next,
                          std::vector<float>& logits);

    const Weights& w_;
    EmbedFn embed_;
    LmHeadFn lm_;
    KvCache cache_;
    DraftStream stream_;
    int hidden_ = kH;
    int vocab_ = 0;
    int mismatches_ = 0;
    bool use_lm_head_ = true;
    // False while a row is being advanced only for its KV/hidden (commit_rows, prime): those callers discard the
    // logits, and asking for them costs a whole lm-head dispatch chain per committed row.
    bool need_logits_ = true;
    long lm_calls_ = 0;
    float tau_ = 0.6f;
};

}  // namespace onebit::npu::mtp
