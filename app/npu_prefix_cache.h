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

// A multi-prompt prefix cache for `1bit forward-serve` (issue #650): up to N prompts kept in host
// memory, and a request resumes from the LONGEST saved prompt that matches the start of its own,
// not only from the prompt that came immediately before.
//
// This header is the slot table only -- which prompts are worth keeping, which one a request may
// resume from, what it costs, and what to report.  It is device-free and takes the state itself as
// an opaque payload the caller owns, so the table is testable without a model, a KV cache or an
// NPU, which is how issue #650 says to start.  The state it stands for is
// Q4nxNpuForward::StateSnapshot (attention KV plus the GDN conv/recurrent state).
//
// Semantics, matching FLM v1.0.11:
//   --prefix-slots N     how many prompts to keep, 0 turns the cache off (FLM default 8, max 32)
//   --prefix-cache-mb N  the host-memory budget for those prompts (FLM default 4096)
//   --prefix-min-gain N  save a prompt only if it covers at least N tokens more than the longest
//                        saved prompt that is already a prefix of it (FLM default 128)
// Whole tokens only: the forward's step is token-at-a-time, so a partial-chunk match would buy
// nothing and risk a wrong position.
//
// The pinned prefix (engine#1785, the single-turn "flash" shape).  A slot holds a whole prompt, and a hit
// needs the slot to be a prefix of the request.  Single-turn requests that share a system prompt and differ
// in the user turn therefore never hit: `system + question 1` is not a prefix of `system + question 2`.
// What they share is the part before the first difference, so that part gets a slot of its own:
//   pin_point(prev, ids)  how many leading tokens this request shares with the previous one, when that
//                         is worth a slot and no slot covers it already
//   pin(prefix, bytes)    keep the state at that point.  One pinned slot at a time; the budget's eviction
//                         and a longer prompt's replacement both leave it alone, a new pin() replaces it.
// The rule reads token ids only, so it needs no knowledge of the template or of what a system prompt is.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::npu {

// FLM's ceiling; a slot costs a KV snapshot, so the number is bounded rather than open.
inline constexpr int kMaxPrefixSlots = 32;

struct PrefixOptions {
    int slots = 8;            // 0 = off
    size_t cache_mb = 4096;   // host memory the slots may hold together
    int min_gain = 128;       // tokens a new prompt must add over the slot it would extend
};

// What the cache decided for one request, for `usage.flm_prefix_cache` and the one log line.
struct PrefixDecision {
    int slot = -1;            // slot the request resumes from, -1 = prefill from scratch
    int cached_tokens = 0;    // tokens of this prompt the slot already covers
    std::string reason;       // "hit", "off", "no slot shares a prefix", "slot has no state", ...
    bool pinned = false;      // the slot is the pinned shared prefix, not a whole saved prompt
};

// How many tokens two prompts share from position 0.
inline size_t shared_prefix_len(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

// Where a hit may resume.  A slot's snapshot is taken one token short of the prompt it was saved for
// (forward-serve splits the last prefill chunk and saves before it, because the answer is drawn from the last
// position's logits and a snapshot holds state, not logits), so the positions it can restore are `cached - 1`,
// and the resume never goes past the prompt's own last token either.  On a GDN model restoring `cached - 1`
// of a snapshot taken at `cached` would apply the last token's in-place recurrence update twice, which answers
// the same prompt differently -- measured 2026-11-04, engine#984.
inline int resume_point(const PrefixDecision& d, size_t prompt_tokens) {
    if (d.slot < 0 || d.cached_tokens <= 0 || prompt_tokens == 0) return 0;
    const int cached = std::min(d.cached_tokens, int(prompt_tokens));
    return cached > 0 ? cached - 1 : 0;
}

struct PrefixSlot {
    std::vector<int> ids;     // the tokens this slot covers, from position 0
    size_t bytes = 0;         // what its snapshot costs in the budget
    uint64_t last_used = 0;   // monotonic, for eviction
    bool live = false;        // false = free/reusable
    bool pinned = false;      // the shared prefix: not evicted for room, not replaced by a longer prompt
};

class PrefixCache {
public:
    PrefixCache() = default;
    explicit PrefixCache(const PrefixOptions& o) : opt_(o) {
        if (o.slots < 0 || o.slots > kMaxPrefixSlots)
            throw std::runtime_error("prefix-slots: must be 0 to " + std::to_string(kMaxPrefixSlots));
        if (o.min_gain < 0) throw std::runtime_error("prefix-min-gain: must be zero or more");
        slots_.resize(size_t(std::max(0, o.slots)));
    }

    const PrefixOptions& options() const { return opt_; }
    bool enabled() const { return opt_.slots > 0; }
    size_t used_bytes() const { return used_; }

    // The longest live slot whose tokens are a prefix of `ids`.  A slot that covers the whole
    // prompt is still a hit: the request continues from it instead of re-prefilling.
    PrefixDecision lookup(const std::vector<int>& ids) {
        PrefixDecision d;
        if (!enabled()) {
            d.reason = "off";
            return d;
        }
        size_t best = 0;
        int best_slot = -1;
        for (size_t i = 0; i < slots_.size(); i++) {
            const PrefixSlot& s = slots_[i];
            if (!s.live || s.ids.empty() || s.ids.size() > ids.size()) continue;
            if (!std::equal(s.ids.begin(), s.ids.end(), ids.begin())) continue;
            if (s.ids.size() > best) {
                best = s.ids.size();
                best_slot = int(i);
            }
        }
        if (best_slot < 0) {
            d.reason = "no slot shares a prefix";
            return d;
        }
        d.slot = best_slot;
        d.cached_tokens = int(best);
        d.reason = "hit";
        d.pinned = slots_[size_t(best_slot)].pinned;
        last_used_++;
        slots_[best_slot].last_used = last_used_;
        return d;
    }

    // Offer (ids, bytes) once the request has prefilled them.  Returns the slot the caller should
    // store the state in, or -1 when the prompt is not worth keeping.  `extended` is the slot the
    // request resumed from (-1 when it prefilled from scratch): that slot is replaced, because the
    // longer prompt covers it.
    int insert(const std::vector<int>& ids, size_t bytes, int extended = -1, std::string* why = nullptr) {
        auto refuse = [&](const std::string& reason) {
            if (why) *why = reason;
            return -1;
        };
        if (!enabled() || ids.empty()) return refuse("off");
        // Gain: tokens this prompt adds over the longest slot it extends (or over nothing).
        size_t covered = 0;
        if (extended >= 0 && size_t(extended) < slots_.size() && slots_[size_t(extended)].live)
            covered = slots_[size_t(extended)].ids.size();
        const size_t gain = ids.size() > covered ? ids.size() - covered : 0;
        if (gain < size_t(opt_.min_gain)) return refuse("gain below prefix-min-gain");
        if (bytes > opt_.cache_mb * 1024u * 1024u) return refuse("larger than prefix-cache-mb");

        // The slot extended is replaced: its tokens are a prefix of the new ones.  A pinned slot is not:
        // the next single-turn request shares the pin, not this prompt, so the prompt takes a slot of its own.
        int slot = extended;
        if (slot >= 0 && size_t(slot) < slots_.size() && slots_[size_t(slot)].pinned) slot = -1;
        if (slot >= 0 && size_t(slot) < slots_.size() && slots_[size_t(slot)].live) {
            used_ -= slots_[size_t(slot)].bytes;
            slots_[size_t(slot)].live = false;
        }
        // Fit the budget by dropping the least recently used slot (never the pinned one).
        if (!make_room(bytes)) return refuse("no room under prefix-cache-mb");
        if (slot < 0 || size_t(slot) >= slots_.size() || slots_[size_t(slot)].live) {
            slot = free_slot();
            if (slot < 0) return refuse("no free slot");  // every slot holds a better prompt
        }
        fill(slot, ids, bytes, /*pinned=*/false);
        if (why) *why = "saved";
        return slot;
    }

    // The number of leading tokens `ids` shares with `prev` (the previous request's prompt) when the state at
    // that point is worth pinning, else 0.  It is 0 when one prompt is a prefix of the other (a repeat or a
    // continued conversation: insert() already keeps those), when the shared part adds fewer than
    // prefix-min-gain tokens over the slot that already covers this request, and when the cache is off.
    size_t pin_point(const std::vector<int>& prev, const std::vector<int>& ids) const {
        if (!enabled()) return 0;
        const size_t shared = shared_prefix_len(prev, ids);
        if (shared == prev.size() || shared == ids.size()) return 0;
        size_t covered = 0;  // the longest live slot that is a prefix of this request
        for (const PrefixSlot& s : slots_) {
            if (!s.live || s.ids.size() > ids.size() || s.ids.size() <= covered) continue;
            if (std::equal(s.ids.begin(), s.ids.end(), ids.begin())) covered = s.ids.size();
        }
        if (shared <= covered || shared - covered < size_t(opt_.min_gain)) return 0;
        return shared;
    }

    // Keep `prefix` (the first pin_point() tokens of a request) as the pinned slot.  Returns the slot the
    // caller should store the state in, or -1.  The previous pin, if any, is released first: there is one
    // shared prefix at a time, and the newest pair of requests is the evidence for what it is.
    int pin(const std::vector<int>& prefix, size_t bytes, std::string* why = nullptr) {
        auto refuse = [&](const std::string& reason) {
            if (why) *why = reason;
            return -1;
        };
        if (!enabled() || prefix.empty()) return refuse("off");
        if (bytes > opt_.cache_mb * 1024u * 1024u) return refuse("larger than prefix-cache-mb");
        for (PrefixSlot& s : slots_)
            if (s.live && s.pinned) release(s);
        if (!make_room(bytes)) return refuse("no room under prefix-cache-mb");
        int slot = free_slot();
        if (slot < 0) {
            slot = least_recently_used();  // a whole prompt gives way to the prefix every request shares
            if (slot < 0) return refuse("no free slot");
            release(slots_[size_t(slot)]);
        }
        fill(slot, prefix, bytes, /*pinned=*/true);
        if (why) *why = "pinned";
        return slot;
    }

    // The pinned slot, or -1.
    int pinned_slot() const {
        for (size_t i = 0; i < slots_.size(); i++)
            if (slots_[i].live && slots_[i].pinned) return int(i);
        return -1;
    }

    const PrefixSlot& slot(int i) const { return slots_.at(size_t(i)); }
    size_t slot_count() const { return slots_.size(); }

    // A cancelled request, or a cleared context, must not throw away other conversations' prompts
    // (FLM v1.0.11's own fix), so there is deliberately no clear() here: entries age out by the
    // budget and are replaced only by prompts that cover more tokens.  The pinned slot does neither.

private:
    void release(PrefixSlot& s) {
        used_ -= s.bytes;
        s.live = false;
        s.pinned = false;
    }
    // The least recently used live slot that is not pinned, or -1.
    int least_recently_used() const {
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < slots_.size(); i++) {
            if (!slots_[i].live || slots_[i].pinned) continue;
            if (slots_[i].last_used < oldest) {
                oldest = slots_[i].last_used;
                victim = int(i);
            }
        }
        return victim;
    }
    // Evict until `bytes` more fit under the budget; false when they cannot.
    bool make_room(size_t bytes) {
        const size_t budget = opt_.cache_mb * 1024u * 1024u;
        while (used_ + bytes > budget) {
            const int victim = least_recently_used();
            if (victim < 0) break;  // nothing left to evict
            release(slots_[size_t(victim)]);
        }
        return used_ + bytes <= budget;
    }
    int free_slot() const {
        for (size_t i = 0; i < slots_.size(); i++)
            if (!slots_[i].live) return int(i);
        return -1;
    }
    void fill(int slot, const std::vector<int>& ids, size_t bytes, bool pinned) {
        PrefixSlot& s = slots_[size_t(slot)];
        s.ids = ids;
        s.bytes = bytes;
        s.live = true;
        s.pinned = pinned;
        s.last_used = ++last_used_;
        used_ += bytes;
    }

    PrefixOptions opt_;
    std::vector<PrefixSlot> slots_;
    size_t used_ = 0;
    uint64_t last_used_ = 0;
};

}  // namespace onebit::npu
