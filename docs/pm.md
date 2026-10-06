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
# The Project Manager (`1bit serve --pm`)

ZAYA1-8B is made for reasoning and calls tools reliably. `1bit serve --pm` makes the served
model the Project Manager of the engine: it keeps every conversation, answers what it can
itself, and hands a task that fits an expert's domain to that expert model, then relays the
expert's answer to the user. The experts are ordinary Lemonade models: the PM asks Lemonade to
run one, Lemonade loads it (it keeps two LLMs at once, so one expert runs beside the PM), and
the expert's answer comes back as a tool result.

This is different from Laya ([laya.md](laya.md)), which sorts requests among *devices* for one
model. The PM routes among *models*, and the model itself decides.

```
client ──POST /v1/chat/completions──▶ 1bit serve --pm ──▶ ZAYA (llama-server, resident)
                                          │   ◀── delegate(expert="code", task=...)
                                          └──POST ${lemonade-url}/api/v1/chat/completions, model = expert id──▶ Lemonade ──▶ expert
                                              ◀── the expert's answer (tool result) ──▶ ZAYA composes the final answer ──▶ client
```

## Running it

**From Lemonade** (the normal way; the engine is embedded in Lemonade, which owns model loading,
bind address, port and auth):

- Pick the catalog entry `ZAYA1-8B-PM-1bit` in Lemonade's model browser (our fork's
  `server_models.json`). Its `recipe_options` carry `"onebit_pm": true`, and the `onebit` recipe
  then starts `1bit serve --pm --lemonade-url http://<its own host>:<port>`. Nothing else to set.
- Or add ZAYA as a custom recipe: Lemonade's `POST /v1/pull` registers user models (the `user.*`
  namespace), and the `onebit` recipe reads `recipe_options.onebit_pm` from a user registration
  exactly as from the built-in catalog:

  ```sh
  curl -X POST http://127.0.0.1:8000/api/v1/pull -H 'Content-Type: application/json' -d '{
    "model_name": "user.ZAYA1-8B-PM-1bit",
    "recipe": "onebit",
    "checkpoint": "1bit-MONSTER/ZAYA1-8B-GGUF:zaya1-8b-Q4_K_M.gguf",
    "recipe_options": {"onebit_pm": true},
    "labels": ["chat", "tool-calling"]
  }'
  ```

  `recipe_options.onebit_pm_experts` may name an experts file (below); without it the engine's
  built-in experts apply. Remove the registration with `POST /v1/delete` (`{"model_name":
  "user.ZAYA1-8B-PM-1bit"}`).

The experts' catalog ids (`Qwen2.5-Coder-7B-Instruct-1bit`, `Qwen3-Coder-30B-A3B-Instruct-1bit`,
`Qwen3.8-27B-1bit`, `gpt-oss-20b-mxfp4-1bit`) are ordinary `-1bit` entries in the same catalog:
each can be pulled and chatted with on its own. Lemonade downloads an expert's GGUF the first time
the PM asks for it, so pull the experts you want ahead of time.

**Standalone**, against a running Lemonade:

```sh
1bit serve -m zaya1-8b-Q4_K_M.gguf --alias ZAYA1-8B-PM --pm --lemonade-url http://127.0.0.1:8000
# optional: --pm-experts my-experts.json
```

The other `1bit serve` options apply unchanged (device, `--ctx-size`, recipes). `--pm` does not
combine with `--laya` or with `--embedding`/`--reranking`.

## What a request does

1. The PM's system prompt (who it is, when to answer and when to delegate, the experts with their
   `use_when` text) goes first; a system message of the client's follows it.
2. The PM gets two tools: `experts()` lists the experts; `delegate(expert, task, context)` hands
   a self-contained brief to one of them. The round runs greedy (temperature 0 unless the request
   sets one) with `chat_template_kwargs.enable_thinking` false (`pm.thinking` or the request's own
   value changes that), and stops at the end of a tool call.
3. A `delegate` call becomes one non-streaming chat request to
   `${lemonade-url}/api/v1/chat/completions` with `model` = the expert's catalog id (and the
   expert's `max_tokens`). Lemonade loads the model if it is not loaded. The expert's full answer
   is the tool result, cut at `pm.max_result_bytes` (UTF-8 safe, with a note when cut). One line
   is logged per delegation: expert, task length, milliseconds, ok or why not.
4. The loop runs at most `pm.max_rounds` rounds (default 4); the last round offers no tools, so
   the PM must answer. A round without a tool call is the final answer.
5. Tool calls are read from llama-server's parsed `tool_calls`, or from the text of the answer in
   ZAYA's own form (`<zyphra_tool_call><function=delegate><parameter=expert>code</parameter>…`)
   or Qwen's `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
6. The reply is a normal chat completion under the served model's name, plus a `pm` object
   (`rounds`, `delegations` with `expert`, `id`, `ms`, `ok`) and an `X-1bit-PM` header. With
   `"stream": true` the final answer arrives as SSE chunks (a role chunk, `reasoning_content` and
   `content` pieces, a finish chunk, `[DONE]`).

When Lemonade is not reachable, answers an error, or the expert returns nothing, the tool result
says so ("delegation to … failed: …") and the PM answers alone, saying the expert was not
available. It never fails silently, and never pretends an expert answered.

## The experts file

`config/pm-experts.json` is built into the binary; `--pm-experts FILE` (or Lemonade's
`onebit_pm_experts`) replaces it.

```json
{
  "pm": {"max_rounds": 4, "max_result_bytes": 32768, "delegate_timeout_s": 1800, "thinking": false},
  "experts": [
    {"expert": "code", "id": "Qwen2.5-Coder-7B-Instruct-1bit", "domain": "programming",
     "use_when": "Writing, fixing, explaining or reviewing code…", "max_tokens": 4096}
  ]
}
```

| Field | |
|---|---|
| `expert` | the name the PM uses in `delegate(expert=…)` |
| `id` | the model's Lemonade catalog id: what the PM asks Lemonade to run |
| `domain`, `use_when` | shown to the PM; this text is the routing |
| `max_tokens` | the expert's answer budget (0: Lemonade's default) |
| `pm.max_rounds` | PM rounds per request, tool rounds included (default 4) |
| `pm.max_result_bytes` | an expert's answer is cut here before the PM sees it (default 32768) |
| `pm.delegate_timeout_s` | read timeout for one expert answer (default 1800) |
| `pm.thinking` | `enable_thinking` for the PM's rounds (default false) |
| `pm.system_prompt` | replaces the built-in prompt (the experts list is still appended) |
| `pm.chat_path` | Lemonade's chat path (default `/api/v1/chat/completions`) |

The defaults: `code` → Qwen2.5-Coder-7B-Instruct (Q4_K_M, 4.7 GB, Apache-2.0: light and quick
to load, the default for code), `code-heavy` → Qwen3-Coder-30B-A3B-Instruct (large refactors and
long repositories; an 18 GB load), `reasoning` → Qwen3.8-27B (long reasoning, mathematics,
analysis), `agentic` → gpt-oss-20b (tool-heavy agentic tasks). General conversation is the PM's
own; there is no expert for it.

## Limits

- One expert at a time: Lemonade keeps two LLMs loaded (the PM and one expert), so a second
  expert in the same request means Lemonade swaps models, which costs a load each time. The
  prompt tells the PM to delegate once with a complete brief.
- Round cap: `pm.max_rounds` (default 4); the last round has no tools.
- No persistence: the PM has nothing but the conversation the client sends; delegations are not
  remembered across requests. The log (one line per delegation) is the only record.
- The final answer is streamed once the PM has composed it: the SSE chunks arrive together at the
  end, not token by token.
- `--lemonade-url` is plain `http://` (the engine's HTTP client has no TLS). When Lemonade requires
  an API key (`LEMONADE_API_KEY`), delegation requests carry none and fail with HTTP 401: the
  engine has no API key of its own (Lemonade leads on auth), so run the PM against a Lemonade
  without one, or on its loopback address where none is required.
- The routing is the model's decision; an expert's `use_when` text is the lever.

## Tests

`ctest -R pm_route` ([tests/pm_route.sh](../tests/pm_route.sh), no GPU): a mock llama-server plays
the PM (a `delegate(code)` call on the first round, the final answer on the second, in both the
parsed and ZAYA's text form) and a mock Lemonade records the expert id it is asked for. It checks
the delegation hits the default code expert's id with its `max_tokens`, the expert's answer
reaches the client under the served name, a general question is answered without a delegation,
streaming, a Lemonade that is down (the PM still answers, HTTP 200, the reply says so), an expert
that errors, the round cap, `/v1/completions` untouched, and the refusals.
