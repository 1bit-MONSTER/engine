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

#include "lax_turns.h"

#include <algorithm>

namespace onebit::npu::lax {

namespace {

bool is_prefix(std::span<const int> p, std::span<const int> of) {
    return p.size() <= of.size() && std::equal(p.begin(), p.end(), of.begin());
}

}  // namespace

size_t Snapshots::bytes() const {
    size_t n = 0;
    for (const auto& e : e_) n += e.state.capacity();
    return n;
}

int Snapshots::best(std::span<const int> prompt) const {
    int best = -1;
    for (size_t i = 0; i < e_.size(); ++i) {
        const auto& t = e_[i].tokens;
        if (t.empty() || t.size() >= prompt.size() || !is_prefix(t, prompt)) continue;
        if (best < 0 || t.size() > e_[size_t(best)].tokens.size()) best = int(i);
    }
    return best;
}

bool Snapshots::contains(std::span<const int> tokens) const {
    return std::ranges::any_of(e_, [&](const Entry& e) { return std::ranges::equal(e.tokens, tokens); });
}

Snapshots::Entry* Snapshots::put(std::span<const int> tokens) {
    if (cap_ == 0) return nullptr;
    Entry* e = nullptr;
    for (auto& x : e_)
        if (std::ranges::equal(x.tokens, tokens)) e = &x;
    if (!e) {
        if (e_.size() < cap_) e = &e_.emplace_back();
        else e = &*std::ranges::min_element(e_, {}, &Entry::used);
        e->tokens.assign(tokens.begin(), tokens.end());
    }
    e->used = ++tick_;
    return e;
}

Plan plan(std::span<const int> live, const Snapshots& snaps, std::span<const int> prompt) {
    Plan p;
    if (!live.empty() && live.size() < prompt.size() && is_prefix(live, prompt)) {
        p.source = Plan::Live;
        p.reuse = live.size();
    }
    if (const int i = snaps.best(prompt); i >= 0 && snaps.at(size_t(i)).tokens.size() > p.reuse) {
        p.source = Plan::Snapshot;
        p.reuse = snaps.at(size_t(i)).tokens.size();
        p.snapshot = i;
    }
    return p;
}

std::vector<size_t> snapshot_points(std::span<const int> prompt, size_t from, std::span<const int> turn_tokens) {
    std::vector<size_t> out;
    if (prompt.size() < 2) return out;
    const size_t lo = std::max<size_t>(from, 1);
    for (size_t i = prompt.size() - 1; i-- > 0;)  // the last turn token before the last token
        if (std::ranges::find(turn_tokens, prompt[i]) != turn_tokens.end()) {
            if (i >= lo) out.push_back(i);
            break;
        }
    if (prompt.size() - 1 >= lo) out.push_back(prompt.size() - 1);
    return out;
}

}  // namespace onebit::npu::lax
