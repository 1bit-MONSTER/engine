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

#include "generate.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace onebit::npu {

std::vector<int> eos_ids(const Model& model) {
    std::vector<int> ids = model.dims().eos;
    if (model.dims().model_type == "qwen3")
        for (int id : {151643, 151645})
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    return ids;
}

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Chooser {
    const GenerateOptions& opt;
    std::vector<int> history;
    std::vector<float> buf;

    int pick(Lane& lane, int step) {
        if (opt.on_logits) opt.on_logits(step, lane);
        if (!(opt.repetition_penalty > 1.0f)) return lane.argmax();
        lane.logits(buf);
        const int n = int(history.size()), from = std::max(0, n - opt.penalty_window);
        for (int i = from; i < n; ++i) {
            const int t = history[size_t(i)];
            if (t < 0 || t >= int(buf.size())) continue;
            if (std::find(history.begin() + from, history.begin() + i, t) != history.begin() + i) continue;
            float& l = buf[size_t(t)];
            l = l > 0.0f ? l / opt.repetition_penalty : l * opt.repetition_penalty;
        }
        const int best = int(std::max_element(buf.begin(), buf.end()) - buf.begin());  // first maximum
        history.push_back(best);
        return best;
    }
};

}  // namespace

GenerateResult generate(Lane& lane, const Model& model, const std::vector<int>& prompt, const GenerateOptions& opt) {
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (int(prompt.size()) + opt.max_tokens > Lane::kMaxContext)
        throw std::runtime_error("prompt plus max_tokens exceeds the context limit");
    const std::vector<int> eos = eos_ids(model);
    auto is_eos = [&](int t) { return opt.stop_at_eos && std::find(eos.begin(), eos.end(), t) != eos.end(); };

    GenerateResult res;
    auto t0 = clk::now();
    int ctx = 0;
    for (int t : prompt) lane.step(t, ++ctx);
    res.prefill_ms = ms_since(t0);

    Chooser chooser{opt, prompt, {}};
    auto emit = [&](int t) {
        res.tokens.push_back(t);
        if (is_eos(t)) {
            res.stopped_at_eos = true;
            return false;
        }
        return !opt.on_token || opt.on_token(t);
    };
    t0 = clk::now();
    // The prompt's last run already produced the first token's logits.
    int cur = 1;  // slot of the run in flight
    const int first = chooser.pick(lane, 0);
    if (emit(first) && opt.max_tokens > 1) {
        lane.prepare(cur, ++ctx);
        lane.launch(cur, first);
        for (int i = 1; i < opt.max_tokens; ++i) {
            const int next = cur ^ 1;
            const bool more = i + 1 < opt.max_tokens;
            if (more) lane.prepare(next, ctx + 1);  // host work overlaps the run in flight
            lane.wait(cur);
            const int t = chooser.pick(lane, i);
            if (!emit(t) || !more) break;
            lane.launch(next, t);
            ++ctx;
            cur = next;
        }
    }
    res.decode_ms = ms_since(t0);
    return res;
}

}  // namespace onebit::npu
