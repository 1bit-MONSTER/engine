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
// think_template_test: what enable_thinking can mean for a pack, read off its template (app/think_template.h,
// engine#2135). The three generation blocks below are copied from the packs on this box: Qwen3-0.6B (has the switch),
// FastFlowLM's Qwen3-4B-Thinking-2507-NPU2 (opens <think> unconditionally, no switch) and Qwen3-VL-4B-Instruct-NPU2
// (neither).
#include "think_template.h"

#include <cstdio>
#include <string>

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

using onebit::ThinkPlan;
using onebit::ThinkPrompt;
using onebit::think_plan;
using onebit::think_template_of;

int main() {
    const std::string qwen3 =
        "{%- for message in messages %}...<think>\\n' + reasoning_content ...{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
        "        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}";
    const std::string thinking_only =
        "{%- for message in messages %}... '</think>' ...{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\\n' }}\n{%- endif %}";
    const std::string no_think =
        "{%- for message in messages %}...{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n";

    const auto q = think_template_of(qwen3), t = think_template_of(thinking_only), v = think_template_of(no_think);
    check(q.known && q.has_switch && q.mentions_think && !q.opens_think,
          "Qwen3: a switch; its empty <think></think> for thinking off is not 'opening the block'");
    check(t.known && !t.has_switch && t.mentions_think && t.opens_think,
          "Thinking-2507: no switch, and the generation prompt leaves <think> open");
    check(v.known && !v.has_switch && !v.mentions_think && !v.opens_think, "VL-4B-Instruct: no switch and no block");
    check(!think_template_of("").known, "no template read: not known");

    // enable_thinking false
    auto plan = [](const onebit::ThinkTemplate& tt, bool want, bool route_opens = false, bool none = false) {
        return think_plan(true, want, none, route_opens, tt);
    };
    ThinkPlan p = plan(q, false);
    check(p.prompt == ThinkPrompt::empty_block && !p.start_inside,
          "Qwen3, thinking off: the empty block, as before (the 0.6B demo path does not move)");
    p = plan(t, false);
    check(p.prompt == ThinkPrompt::none && !p.start_inside && p.split,
          "Thinking-2507, thinking off: no empty block -- served as thinking on, and split");
    p = plan(t, false, /*route_opens=*/true);
    check(p.prompt == ThinkPrompt::open && p.start_inside && p.split,
          "... and where the route opens the block for thinking on, the splitter starts inside it");
    p = plan(v, false);
    check(p.prompt == ThinkPrompt::none && !p.start_inside, "VL-4B-Instruct, thinking off: nothing is added to the prompt");
    p = plan(onebit::ThinkTemplate{}, false);
    check(p.prompt == ThinkPrompt::empty_block, "an unknown template keeps today's empty block");
    // a template that mentions <think> somewhere but has no switch and does not open it: today's empty block
    const auto mention = think_template_of("... <think> in a message ...{%- if add_generation_prompt %}{{- 'assistant\\n' }}{%- endif %}");
    check(plan(mention, false).prompt == ThinkPrompt::empty_block, "no switch, a <think> only in the history: unchanged");

    // a template WITH the switch that opens the block only when thinking is on (Qwen3.6's shape): thinking off must
    // still mean thinking off -- the switch wins over "opens the block"
    const auto q36 = think_template_of(
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
        "{%- if enable_thinking is not defined or enable_thinking %}{{- '<think>\\n' }}{%- endif %}{%- endif %}");
    check(q36.has_switch && q36.opens_think, "Qwen3.6 shape: a switch, and the block opened when thinking is on");
    check(plan(q36, false).prompt == ThinkPrompt::empty_block, "... so thinking off is still the empty block, not forced on");

    // enable_thinking true never moves
    for (const auto* tt : {&q, &t, &v}) {
        check(plan(*tt, true).prompt == ThinkPrompt::none && plan(*tt, true, true).prompt == ThinkPrompt::open,
              "thinking on: the route decides, whatever the template");
    }
    // the splitter: on unless reasoning_format is "none"; off for a raw completion
    check(plan(q, false).split && plan(q, true).split, "the splitter is on with thinking on or off");
    check(!plan(t, false, false, /*none=*/true).split, "reasoning_format none: no split");
    check(!think_plan(false, false, false, false, t).split && think_plan(false, false, false, false, t).prompt == ThinkPrompt::none,
          "not a chat request: nothing");

    std::printf("%s\n", fails ? "think_template: FAILED" : "think_template: all cases ok");
    return fails ? 1 : 0;
}
