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

// forward_base.h -- onebit::npu::mtp::BaseModel over the real Q4nxNpuForward, so the draft loop (#785, #798)
// can drive the actual 27B instead of a toy.
//
// The one non-obvious part is commit_verify(): the forward has no GDN deferral yet, so this adapter takes a
// state snapshot before the verify batch and restores the accepted prefix afterwards.  That is the honest
// fallback documented in npu/mtp/draft.h -- correct, and slower than the deferred design (a snapshot per
// iteration instead of k rows of update inputs).  When the engine grows deferral, only this file changes.
#pragma once

#include "mtp/draft.h"
#include "mtp/mtp_head.h"
#include "q4nx_forward.h"

#include <cstdio>
#include <memory>

namespace onebit::npu::mtp {

class ForwardBase : public BaseModel {
public:
    explicit ForwardBase(Q4nxNpuForward& fwd) : fwd_(fwd) {}

    int vocab() const override { return NV_; }
    int hidden_size() const override { return kH; }
    int max_batch_rows() const override { return fwd_.max_batch_rows(); }

    bool step(int token, int pos, std::vector<float>& logits) override { return fwd_.step(token, pos, logits); }

    bool step_batch_all(const std::vector<int>& tokens, int start_pos, std::vector<float>& logits_all) override {
        // snapshot first: commit_verify() restores the accepted prefix from here
        snap_ = std::make_unique<Q4nxNpuForward::StateSnapshot>();
        if (!fwd_.snapshot_state(start_pos, *snap_)) return false;
        snap_pos_ = start_pos;
        pending_ = (int)tokens.size();
        last_tokens_ = tokens;   // commit_verify() replays exactly these when it cannot defer
        return fwd_.step_batch_all(tokens, start_pos, logits_all);
    }

    bool commit_verify(int accepted) override {
        if (accepted < 0 || accepted > pending_) {
            std::fprintf(stderr, "[commit] refused: accepted %d, pending %d\n", accepted, pending_);
            return false;
        }
        // Restore the state as it was before the verify batch, then replay the accepted rows so the
        // recurrence advances exactly as far as they did (attention alone would need nothing).
        if (!snap_) { std::fprintf(stderr, "[commit] refused: no snapshot\n"); return false; }
        if (!fwd_.restore_state(*snap_, snap_pos_)) {
            std::fprintf(stderr, "[commit] restore_state(%d of %d) refused: %s\n", snap_pos_, snap_->tokens,
                         fwd_.last_error());
            return false;
        }
        for (int i = 0; i < accepted; i++) {
            std::vector<float> lg;
            if (!fwd_.step(last_tokens_[(size_t)i], snap_pos_ + i, lg)) return false;
        }
        pending_ = 0;
        return true;
    }

    const std::vector<float>& last_hidden() const override {
        return use_residual_ ? fwd_.last_residual() : fwd_.last_hidden();
    }
    void use_residual(bool on) { use_residual_ = on; }

    // The drafter needs the base embedding of the token it is drafting from; the forward owns that table.
    void embed_into(int token, std::vector<float>& out) const { fwd_.embed_row(token, out); }
    void set_vocab(int nv) { NV_ = nv; }
    // The drafter's own hidden state goes through the BASE model's lm head (the head has none of its own).
    bool lm_head(const std::vector<float>& h, std::vector<float>& lg) { return fwd_.lm_head_logits(h, lg); }

private:
    Q4nxNpuForward& fwd_;
    std::unique_ptr<Q4nxNpuForward::StateSnapshot> snap_;
    std::vector<int> last_tokens_;
    int snap_pos_ = 0, pending_ = 0, NV_ = 248320;
    bool use_residual_ = false;
};

}  // namespace onebit::npu::mtp
