<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# The NPU route's chat template: render the container's own (engine #963 option 2)

`app/forward_serve.cpp` renders **ChatML** for every container it serves, and since #987 it refuses the
chat route (400) for a container whose tokenizer has no `<|im_start|>`/`<|im_end|>` anchors. That closed
the silent-wrongness half of #963: a caller is never told a prompt was accepted when the model cannot
read it. What is still missing is the other half — a container that carries *its own* template should be
served with it, not with an approximation of it.

This note records what that would take, **measured on the containers we ship**, and asks for the decision
#963 says is needed. Three of the assumptions in that issue turn out to be wrong in our favour.

## What is actually true on this box (2026-11-05)

**1. No container-format change is needed.** #963 assumed the model's template would have to be *packed
into* the container. It is already there:

```
an imported OGA pack (/tmp/844-out-clean, Qwen2.5-0.5B):
  added_tokens.json  chat_template.jinja  config.json  generation_config.json  merges.txt
  model.q4nx  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.json

a FastFlowLM pack (~/Qwen3-0.6B-NPU2):
  config.json  model.q4nx  tokenizer.json  tokenizer_config.json
```

The first carries `chat_template.jinja` (2561 B, `8aa40ce145adb73c…`); the second carries the template as
`tokenizer_config.json`'s `chat_template` string. Both are read and **pin-able** since #1095: every
`.gguf` *and* model-directory load now prints `template sha256=…` and compares it with
`config/template-pins.json`. So the substrate this design needs is already merged.

**2. No new dependency is needed.** The pinned llama.cpp fork — which this engine already builds against —
vendors a C++ Jinja engine in `common/jinja/` (≈6.2 kLOC: lexer, parser, runtime, value, caps; upstream
llama.cpp's own, introduced in ggml-org/llama.cpp#18462). It is what `llama-server` uses to render chat
templates, so it is the same renderer our `hrx`/`cpu` route already trusts. Compiling that directory into
`1bit` plus a small adapter (`messages` → context → `render`) is the whole of the engine change; the
alternative — inventing a mini-renderer — would be a second, weaker implementation of the same thing.

**3. The templates we ship are small and use little of Jinja.** Census over the 24 templates in the
appliance's catalogue (17 distinct), plus the containers on this box:

| construct | templates |
|---|---|
| `{% for %}`, `{% if/elif/else %}`, filters (`selectattr`, `map`, `join`, `tojson`, `trim`, `default`, `replace`, `split`, `first`, `last`, `string`) | 24/24 |
| `{% set %}` + `namespace(...)` | 23/24 |
| attribute access (`message.content`, `messages[0].role`) | 23/24 |
| `is defined` / `is string` / `is none` | 23/24 |
| string methods (`.split`, `.strip`, `.lstrip`, `.rstrip`, `.startswith`) | 18/24 |
| macros (`{% macro %}`) | 10/24 |
| `{% generation %}` (the training-mask tag) | 0/24 |
| **branch on the content of a user message** (the shape of the paper's attack in AUDIT-1) | 0/24 |

Sizes run from **209 B** (`LiquidAI/LFM2-1.2B`, ChatML-shaped) to **16.7 KB** (`ggml-org/gpt-oss-20b`).
Everything in that table is already covered by `common/jinja`; nothing needs a language extension.

## What the change is

1. `app/npu_template.h` (new): given a model directory, load the template — `tokenizer_config.json`'s
   `chat_template` string, else `chat_template.jinja` (the order #1095 already uses for the line and the
   pin, so the two cannot disagree) — and render `messages` + `add_generation_prompt` (+ `tools`,
   `enable_thinking` as the request provides them) through `common/jinja`.
2. `app/forward_serve.cpp`: `if (chat)` renders with the container's template when it has one. ChatML stays
   as the fallback **only** for a container with no template *and* ChatML anchors; a container with
   neither keeps #987's 400.
3. `tests/npu_template_test.cpp` + `tests/npu_template_route.sh`: the four containers we ship as golden
   prompts, and the token counts compared against `llama-server`'s rendering of the same conversation
   (that is the #963 measurement — 33 vs 15 tokens on ZR1 — turned into an assertion).

## What it costs, and what it can get wrong

- **Engine size**: `common/jinja` is ~6.2 kLOC of C++ compiled into `1bit` (+nlohmann, already a
  dependency of `app/serve.cpp`). Nothing is downloaded at build time.
- **Security posture**: a renderer is the one component that turns data into a prompt, so the rules matter
  more than the code. `common/jinja` already implements input marking against special-token injection; on
  top of it I would (a) refuse attribute indirection (`__class__`, `__globals__`, `cycler`) — the audit's
  red-flag list — (b) bound loop iterations and output size, (c) never `include` or `import`, and (d) keep
  the sha line + pin from #1095, which makes "the template changed" visible even when the render is valid.
- **The one behavioural risk**: today every NPU container is prompted with ChatML and the Qwen3 answer
  parser expects that shape. Rendering the vendor's template changes the *prompt*, not the parser, so the
  Qwen3-family rows do not move (their template is ChatML), the FastFlowLM rows gain their
  `user_system_prompt` extension, and a non-ChatML container moves from "refused" to "served" — which is
  exactly the change #963 asks for, and exactly why it is a decision rather than a patch.

## Measured for the decision: what the four shipped NPU rows would do

| catalogue row | template | ChatML anchors | today (#987) | with this note |
|---|---|---|---|---|
| `1bit-MONSTER/Qwen3.6-35B-A3B-NPU` | `e84f32a23f…`, 7764 B | yes | 200 | 200, rendered from its own template |
| `FastFlowLM/Qwen3-0.6B-NPU2` | `a136047ba140…`, 4449 B | yes | 200, **without** FLM's `user_system_prompt` branch | 200, with it |
| `FastFlowLM/Qwen3-1.7B-NPU2` | `a136047ba140…` | yes | as above | as above |
| `FastFlowLM/Qwen3-4B-NPU2` | `a136047ba140…` | yes | as above | as above |

So no shipped row is blocked today — this is about serving each container with the template it declares
instead of one that happens to work for the Qwen3 family, and about being able to serve the next
non-ChatML container (Zamba2-VL, a DeepSeek-special model, anything a customer packs) at all.

## The decision this asks for

1. **Reuse `common/jinja`** (my recommendation: it is the same renderer the llama.cpp route uses, no new
   dependency, and its input marking is already the security half) — or vendor a second implementation,
   or defer.
2. **Confirm the fallback order**: container template → ChatML only with anchors → 400. Nothing else.
3. **Confirm the strictness**: a container whose template fails to render should be a 400 naming the
   template sha, not a silent fall back to ChatML — the fall-back-to-something-else is the failure mode
   #963 exists to prevent.

If the answer to (1) is "yes", the work is one PR: the renderer glue plus **one** call site
(`app/forward_serve.cpp:176`, the only `npu::chatml()` render on the route; the load-time check at
`:772` becomes the "no template and no anchors" case), the golden-prompt tests, and the pin rows for the
containers we ship (the four above).
