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

// Where the lax decode's next prompt starts (docs/npu-lax.md, "Chat follow-ups"). Pure
// host bookkeeping, no XRT: lax::Session (npu/lax.h) holds the device side.
//
// A position's cache is the KV rows of the ten full-attention layers and the DeltaNet
// state of the thirty linear ones. The KV rows can be overwritten from any position (a
// position reads rows 0..pos-1 only), but the DeltaNet state folds every token in and
// cannot rewind. So a prompt reuses the device's cache in one of two ways:
//   live      the prompt extends the tokens on the device: feed the rest;
//   snapshot  a copy of the DeltaNet state taken at a prefix of the prompt, in host
//             memory: load it, set the position to its length, feed the rest;
// and otherwise starts over from position 0.
//
// Snapshots are taken while a prompt is fed, at the points snapshot_points names: before
// its last token (a retry of the same prompt, or one extending it) and before the last
// turn token (<|im_start|>): a chat client's follow-up re-renders the earlier answer
// differently from how it was generated (without its think block), but repeats every
// message before it exactly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace onebit::npu::lax {

// Snapshots of the DeltaNet state at token prefixes, at most `capacity`; the least
// recently used one goes first.
class Snapshots {
public:
    struct Entry {
        std::vector<int> tokens;     // the prefix: positions 0..tokens.size()-1
        std::vector<uint8_t> state;  // Decoder::save_state's bytes after that prefix
        uint64_t used = 0;
    };

    explicit Snapshots(size_t capacity = 0) : cap_(capacity) {}
    size_t capacity() const { return cap_; }
    size_t size() const { return e_.size(); }
    size_t bytes() const;  // the host memory the kept states take
    const Entry& at(size_t i) const { return e_[i]; }

    // The longest entry whose tokens are a proper prefix of prompt (shorter than it: the
    // last prompt token has to run for its logits), or -1.
    int best(std::span<const int> prompt) const;
    bool contains(std::span<const int> tokens) const;
    void touch(size_t i) { e_[i].used = ++tick_; }
    // The entry for tokens, to be filled with its state: the one already there, or a new
    // one in the place of the least recently used (whose state buffer it keeps, to avoid
    // a reallocation). Its state is left as it was: the caller writes it. nullptr at
    // capacity 0.
    Entry* put(std::span<const int> tokens);
    void clear() { e_.clear(); }

private:
    size_t cap_;
    uint64_t tick_ = 0;
    std::vector<Entry> e_;
};

struct Plan {
    enum Source { Scratch, Live, Snapshot };
    Source source = Scratch;
    size_t reuse = 0;   // prompt tokens not fed again
    int snapshot = -1;  // with Snapshot: the entry
};

// live: the tokens on the device now. The source that feeds the fewest prompt tokens; on a
// tie, live (nothing to load).
Plan plan(std::span<const int> live, const Snapshots& snaps, std::span<const int> prompt);

// The prefix lengths at which to take a snapshot while prompt[from..] is fed, ascending:
// prompt.size() - 1, and the index of the last token of the prompt that is one of
// turn_tokens; each only if it is at least max(from, 1).
std::vector<size_t> snapshot_points(std::span<const int> prompt, size_t from, std::span<const int> turn_tokens);

}  // namespace onebit::npu::lax
