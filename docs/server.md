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
# Server

`1bit-server` is the engine's HTTP front end. It speaks the backend protocol
Lemonade uses for its GGUF recipes (`llamacpp`, `llamacpp-hrx`): the launch
flags, a `/health` probe, and the OpenAI endpoints with llama-server's
`timings` extension. Lemonade can run it wherever it runs those backends today.

Only the protocol is shared with llama-server. The inference is this repository's own:

- model code (`engine/backends`)
- tokenizer, sampler and chat-template rendering (`engine/core`, `engine/server`)

## Launch

```bash
1bit-server -m model.gguf --ctx-size 4096 --device cpu --port 8080 --jinja --metrics --parallel 1
```

| Flag | Meaning |
|---|---|
| `-m, --model` | GGUF model (required) |
| `-c, --ctx-size` | context length (default: min(training context, 4096)) |
| `--host`, `--port` | listen address (default `127.0.0.1:8080`) |
| `--device` | `auto` or `cpu`. More devices are added with each backend; an unknown device is an error, never a silent fallback |
| `-t, --threads` | CPU threads |
| `-a, --alias` | model id reported by the API |
| `-np, --parallel` | only `1` |
| `--jinja`, `--metrics` | accepted for compatibility. The GGUF chat template is always used; there is no `/metrics` yet |

Unknown flags are rejected, so a misconfigured launch fails loudly.

## Endpoints

| Endpoint | Notes |
|---|---|
| `GET /health` | `503 {"error":{"message":"Loading model"}}` while loading, then `200 {"status":"ok"}` |
| `GET /v1/models` | one model, id = alias |
| `POST /v1/chat/completions` | messages rendered with the GGUF's Jinja template; streaming or not |
| `POST /v1/completions` | raw prompt; streaming or not |

Every response carries OpenAI `usage` (including
`prompt_tokens_details.cached_tokens`) and llama-server `timings` (`prompt_n`,
`prompt_ms`, `predicted_n`, `predicted_per_second`, `cache_n` and others). These
are the fields Lemonade's telemetry reads. Streams end with `data: [DONE]`. The
final chunk carries `finish_reason` and `timings`, and
`stream_options.include_usage` adds a usage chunk.

### Request fields

- **Sampling:** `temperature` (0 = greedy), `top_p`, `top_k`, `min_p` and `seed`.
  The defaults match llama-server: 0.8, 0.95, 40 and 0.05, with a random seed.
- **Limits and streaming:** `max_completion_tokens` / `max_tokens` / `n_predict`,
  `stop` (a string or an array), `stream` and `stream_options`.
- **Chat:** `chat_template_kwargs` (for example `{"enable_thinking": false}`),
  `reasoning_format` (`none` keeps `<think>` in `content`), and `tools`
  (rendered into the prompt).
- **Other fields** are ignored.
- **Rejected with 400**, because they would otherwise be silently wrong:
  `n > 1`, non-text content parts, batched prompts, and a prompt that does not
  fit the context (`exceed_context_size_error`).

### Reasoning

A leading `<think>...</think>` block goes to `message.reasoning_content` (in
streams, `delta.reasoning_content`), and the answer goes to `content`. This is
the split Lemonade's chat, MCP and Ollama paths read. When the template itself
opens the block (the prompt ends in `<think>`), the output starts in reasoning.

### Prompt cache

The previous request's tokens stay in the KV cache. A new request reuses the
longest shared prefix and evaluates only the rest, reported as `cache_n`. That
makes multi-turn chat cost one turn, not the whole conversation.

## Verification

`tests/server_e2e.py` launches the binary with Lemonade's flags and checks the
contract over HTTP. It runs in `ctest` when a golden model is configured. On
Qwen3-0.6B it checks:

- **Correctness:** the greedy `/v1/completions` text is byte-identical to HF
  transformers' fp32 greedy continuation from the golden, and `prompt_tokens`
  matches HF's tokenization.
- **Streaming:**
  - streamed text equals non-streamed text, for completions and for chat
    (both `content` and `reasoning_content`)
  - there is exactly one final chunk, with `finish_reason` and `timings`
  - the usage chunk appears
  - the stream ends with `[DONE]`
- **Prefix cache:** a repeated request reuses the whole prompt but its last
  token (`cache_n = n_prompt - 1`) and gives the same text.
- **Stops and chat:**
  - stop strings truncate before the match, with `finish_reason: stop`
  - chat stops on the end-of-turn token
  - `enable_thinking: false` gives plain content
  - thinking on splits the reasoning out
- **Errors:** bad JSON, `n = 2` and an over-long prompt give 400; an unknown route gives 404.

Unit tests (CI): `test_text_stream` covers UTF-8 split across tokens, stop
strings across tokens and reasoning routing byte by byte. `test_openai` covers
request parsing and response shapes, checked structurally. `test_sampler`
covers greedy, seeding, top-k, top-p and min-p. `golden_template_qwen3`
renders 12 conversations byte-identical to HF `apply_chat_template`, including
tools, tool calls and the thinking toggle.

## Not yet

- **Tool-call output parsing.** Tools are rendered into the prompt, but a
  generated `<tool_call>` block is returned as text, not as `tool_calls`.
- **Endpoints:** `/v1/responses`, `/v1/embeddings`, `/tokenize`, `/slots` and `/metrics`.
- **Concurrency:** more than one concurrent sequence. Requests queue.
- **Devices:** anything but the CPU reference. The NPU and GPU backends plug in behind the same server.
