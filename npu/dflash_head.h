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

// dflash_head.h -- a DFlash draft as the speculative loop's drafter (engine#1781): mtp::DraftHead over dflash_forward.
//
// The draft reads the target's hidden states at its tap layers for every committed position (the forward writes
// them to a sink, Q4nxNpuForward::set_layer_taps), keeps them fused in a DraftKvCache, and drafts a block of
// [last token, mask, mask, ...]. It never reads the hidden after the last token, so needs_last_hidden() is false and
// mtp::run() verifies [t1, d1..dk] as one batch (docs/npu.md, "The draft loop, and whether to serve it").
//
// The sink is the one piece of state shared with the forward, and the forward writes a row for EVERY position a pass
// runs, the rejected rows of a verify batch included. So:
//   - at draft(), every row in the sink is a committed position: rows not yet in the cache are fused there;
//   - commit_rows(M) after a verify batch keeps the M committed rows of that batch and drops the rest of its rows,
//     which is what keeps a rejected draft's hidden states out of every later draft;
//   - after a plain step (no draft that round) the one row the step wrote is committed, and nothing is dropped.
// The target's embedding table and the rule that turns a drafted row into a token (the target's head and argmax, or
// a DFlash 2 candidate selector) come in as hooks, so this is testable with no model and no device.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_forward.h"
#include "mtp/draft.h"

namespace onebit::npu::dflash {

struct HeadHooks {
    // The target's embedding of `token`, hidden wide. False = no such token.
    std::function<bool(int token, std::vector<float>& row)> embed;
    // The draft token for one drafted row: `h` is the draft's output row (hidden wide), `prev` the token before it
    // (the last emitted token for the first row). A negative return ends the chain there.
    std::function<int(const float* h, int prev)> pick;
};

class DFlashHead : public mtp::DraftHead {
public:
    // `taps` is the forward's tap sink, one row of taps * hidden floats per position, starting at position 0.
    // `block` is the draft's block size (so up to block - 1 drafts a round); `window` its sliding window, 0 = none.
    DFlashHead(const DraftWeights& w, std::vector<float>& taps, HeadHooks hooks, int mask_token, int block,
               int window = 0)
        : w_(w), taps_(taps), hooks_(std::move(hooks)), mask_token_(mask_token), block_(block), window_(window) {}

    bool needs_last_hidden() const override { return false; }

    int draft(const std::vector<float>& /*hidden*/, int last_token, int pos, int k, std::vector<int>& out,
              std::vector<float>* conf = nullptr) override {
        out.clear();
        if (conf) conf->clear();   // no per-draft probability: mtp::run then skips its tau gate
        drafted_ = false;
        error_.clear();
        const size_t rowf = size_t(w_.taps) * size_t(w_.hidden);
        if (rowf == 0 || taps_.size() % rowf != 0) return fail("the tap sink is not whole rows of taps * hidden");
        const int rows = int(taps_.size() / rowf);
        // last_token runs at pos - 1, right after the committed rows: anything else means the sink and the loop
        // disagree about the stream, and a draft from it would be conditioned on the wrong context.
        if (pos != rows + 1)
            return fail("drafting at position " + std::to_string(pos) + " over " + std::to_string(rows) +
                        " committed rows");
        const int n = std::min(k, block_ - 1);
        if (n <= 0) return 0;
        const size_t H = size_t(w_.hidden);
        std::vector<float> noise(size_t(block_) * H), row;
        try {
            if (cache_.rows < rows) {
                const std::vector<float> f =
                    fuse_context(w_, taps_.data() + size_t(cache_.rows) * rowf, rows - cache_.rows);
                cache_extend(w_, cache_, f.data(), rows - cache_.rows);
            }
            if (mask_.empty() && (!hooks_.embed(mask_token_, mask_) || mask_.size() != H))
                return fail("no embedding for the mask token");
            if (!hooks_.embed(last_token, row) || row.size() != H) return fail("no embedding for the last token");
            std::copy(row.begin(), row.end(), noise.begin());
            for (int i = 1; i < block_; i++) std::copy(mask_.begin(), mask_.end(), noise.begin() + ptrdiff_t(size_t(i) * H));
            last_rows_ = draft_forward_cached(w_, cache_, noise.data(), block_, window_);
        } catch (const std::exception& e) {
            return fail(e.what());
        }
        int prev = last_token;
        for (int i = 1; i <= n; i++) {
            const int t = hooks_.pick(last_rows_.data() + size_t(i) * H, prev);
            if (t < 0) break;
            out.push_back(t);
            prev = t;
        }
        mark_ = taps_.size();
        drafted_ = true;
        return int(out.size());
    }

    void commit_rows(const std::vector<int>& /*tokens*/, const std::vector<float>& /*hidden_rows*/, int M) override {
        if (!drafted_) return;   // a plain step: its one row is committed as written
        drafted_ = false;
        const size_t keep = mark_ + size_t(M) * size_t(w_.taps) * size_t(w_.hidden);
        if (M < 0 || taps_.size() < keep) {
            fail("commit of " + std::to_string(M) + " rows, but the pass wrote fewer");
            return;
        }
        taps_.resize(keep);
    }

    void reset() override {
        cache_ = DraftKvCache{};
        drafted_ = false;
        error_.clear();
    }

    const DraftKvCache& cache() const { return cache_; }
    // The draft's output rows of the last draft() ([block][hidden]) and the context rows it drafted over.
    const std::vector<float>& last_draft_rows() const { return last_rows_; }
    int last_context_rows() const { return cache_.rows; }
    // Why the last draft() returned nothing, or why the commit after it was refused; empty when neither went wrong.
    const std::string& error() const { return error_; }

private:
    int fail(const std::string& why) {
        error_ = why;
        return 0;
    }

    const DraftWeights& w_;
    std::vector<float>& taps_;
    HeadHooks hooks_;
    int mask_token_, block_, window_;
    DraftKvCache cache_;
    std::vector<float> mask_, last_rows_;
    size_t mark_ = 0;
    bool drafted_ = false;
    std::string error_;
};

}  // namespace onebit::npu::dflash
