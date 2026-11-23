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

# The Loom GPU route

`1bit serve` can serve a `.gguf` on the GPU through **loom-serve** (AMD's Loom programs on the HRX
runtime, the `hrx-loom` package) instead of the llama.cpp fork's `llama-server`. Which one serves a
model is decided per model by a route table, `config/loom-routes.json`:

- a model with an entry is served by `loom-serve`, with the entry's Loom card;
- every other model takes the llama.cpp route exactly as before (the fallback).

The shipped table holds only models that passed the gate below (each entry's `gate` links its rows):

| entry | card (published) | gate |
|---|---|---|
| `qwen2.5-coder-1.5b-instruct-q4_k_m` | [1bit-MONSTER/Qwen2.5-Coder-1.5B-Instruct-GGUF-loom](https://huggingface.co/1bit-MONSTER/Qwen2.5-Coder-1.5B-Instruct-GGUF-loom) | engine#3158 |
| `qwen2.5-coder-7b-instruct-q4_k_m` | [1bit-MONSTER/Qwen2.5-Coder-7B-Instruct-GGUF-loom](https://huggingface.co/1bit-MONSTER/Qwen2.5-Coder-7B-Instruct-GGUF-loom) | engine#3158 |

An entry takes effect only where its card is installed (below) and `loom-serve` is available (hrx-loom; on PATH,
or `--loom-serve PATH` / `ONEBIT_LOOM_SERVE`); without the card the entry is skipped, with one line naming why, and
the model keeps the llama.cpp route. To install a card where serve looks for it:

```sh
loom-fetch huggingface:1bit-MONSTER/Qwen2.5-Coder-7B-Instruct-GGUF-loom:chat --materialize=copy \
  --output ~/.cache/1bit/loom-cards/Qwen2.5-Coder-7B-Instruct-GGUF-loom
```

`tools/loom_cards.py --check` (CI) refuses an entry whose `gguf_sha256` + `gguf_size` is not a file
`models/pins.json` pins, and `loom_route_test` checks the compiled-in table parses with every entry pinned and gated.

## The table

```json
{
 "routes": [
  {
   "id": "qwen3-0.6b-q4_k_m",
   "match": {"gguf_sha256": "<64 hex>", "gguf_size": 396705472, "model": ["Qwen3-0.6B-Q4_K_M"]},
   "card": "unsloth-Qwen3-0.6B-loom",
   "preset": "chat",
   "device": "amdgpu://",
   "args": ["--memory_bytes=8000000000"],
   "env": {},
   "gate": "<link to the issue that holds the gate table>"
  }
 ]
}
```

- `match`: every key given must hold. `model` compares with serve's model id (`--alias`) and the file
  name; `gguf_sha256` + `gguf_size` name the exact checkpoint the gate measured. The size is compared
  first, so a file of another size is never hashed; an empty table never reads the model.
- `card`: a fetched Loom package (`loom-fetch`), as a `model.json` or the directory that holds it. A
  relative path resolves under `--loom-cards DIR`, `ONEBIT_LOOM_CARDS`, else `~/.cache/1bit/loom-cards`.
- `preset`: required. Passed as `--preset=`. `loom-serve` refuses a card with several presets unless one
  is named ("select a preset explicitly (N choices)"), and most cards have several (55 of the 59 loom-fun
  cards on 21 Nov); the gate measured one preset, so the entry names it.
- `device`, `args`, `env`: passed to `loom-serve` (`--device=`, as written, and its environment; a
  variable the user set wins).
- `gate`: required. Where the result that admitted the model is recorded.

`--loom-routes FILE` replaces the built-in table, `--loom-serve PATH` (or `ONEBIT_LOOM_SERVE`) names the
server binary, and `--no-loom` (or `ONEBIT_NO_LOOM=1`) keeps every model on the llama.cpp route.

The table is consulted only for the GPU route (`--device auto` or `hrx`). An entry is skipped, with one
line naming why, when the load asks for something only the llama.cpp route has (`--mtp`, `--dflash`,
`--mmproj`, `--moe-slots`, a role, `--embed`/`--rerank`, `--pm`, `--laya`) or when its card cannot be
read. `--ctx-size`, `-fa`, `--parallel` and the draft-length options are not applied on the Loom route:
the card sets context, rows and drafting.

## Card map

`config/loom-cards.json` lists every onebit model in the shipped Lemonade catalogue with its
[loom-fun](https://huggingface.co/loom-fun) Loom card (by Jeremy Fowers and Daniel Holanda), pinned at
the card repository's commit, or `none`. It is data, not routing: a model enters the route table above
only after it passed the gate. `match` says how close the card is:

- `exact`: the card's checkpoint (its `model.json.lock`: repository, revision, sha256) is byte-identical
  to the file our catalogue serves (`models/pins.json`), so the card runs our weights unchanged;
- `same-model`: same upstream model from another GGUF (other packager or quant). Not gateable as it
  stands: a gate row is valid only on the bytes we ship (engine#3158), so the card is first rebuilt or
  pointed at our pinned checkpoint, and the gate row is run on that card;
- `none`: no card; `arch`, `nearest` and `gap` say what a port needs (engine#3156);
- `npu`: NPU entries, which never take the GPU route.

Each entry keeps the card's licence (the Loom programs) and the weights' licence apart; the weights
licence is read from the weights repository at our pinned commit, never taken from a sibling model.
Rebuild with `tools/loom_cards.py --build` from freshly fetched card locks; CI runs `--check`
(offline: every `exact` claim against `models/pins.json`) and `--self-test`.

## Lifecycle

Written against hrx-loom 0.2.0; the CLI and HTTP surface serve uses is the same in 0.1.0.
`loom-serve` is a child of serve like `llama-server`: a free loopback port (`--listen=127.0.0.1:PORT`),
`--heartbeat_ms=0`, the same stop rules (SIGTERM, then SIGKILL after `ONEBIT_BACKEND_STOP_GRACE_S`; it
dies with serve), the same refusal of a port answered by a process serve did not start, and the same
relaunch on a fresh port when the first one was taken before the child bound it.

Readiness differs: `loom-serve` answers as soon as it listens and prepares the card in the background,
so serve reads `GET /v1/status` and waits for the card's registration to be `ready`. A card that
`failed` (the server keeps running) ends serve at once with `loom-serve`'s message. Once ready, serve's
`/health` probes `loom-serve`'s `/readyz`.

## Requests and replies

`loom-serve` serves `POST /v1/chat/completions`. hrx-loom 0.1.0 rejects any field outside its set
(HTTP 400); hrx-loom 0.2.0 accepts every field and passes it to the model program, which ignores what it
does not consume (`stop`, penalties and `response_format` among them). serve therefore, on this route,
under either release:

- sends the card's model id as `model`, and renames it back to the served id in every reply and chunk;
- drops llama-server's serving hints (`cache_prompt`, `id_slot`, `speculative.*`, `timings_per_token`,
  `return_progress`) and renames `n_predict` to `max_tokens`;
- drops an OpenAI field `loom-serve` lacks when it holds the value that means "off" (`n: 1`, zero
  penalties, empty `stop`, ...), and answers 400 naming the field when it does not;
- adds a `timings` object (`prompt_n`, `cache_n`, `predicted_n`; on a stream also `prompt_ms`,
  `predicted_ms`, `predicted_per_second`), from the reply's usage and serve's own clock, marked
  `measured_by`. On a stream, serve asks `loom-serve` for usage and returns it in the final chunk with
  `finish_reason`, as llama-server does (a client that asked for `include_usage` gets OpenAI's separate
  usage chunk);
- answers 501 for `/v1/completions`, `/completion`, `/tokenize`, `/slots` and the other llama-server
  routes.

## Gate

A model gets an entry only after a gate row on strixhalo (engine#3158) on **the bytes we ship**: the
card's `model.json.lock` checkpoint is the file `models/pins.json` pins (`config/loom-cards.json`
`exact`). A row run on another upload measures a model we do not serve and does not count.

**Accuracy.** Three arms on the same GGUF and prompts, greedy, thinking off, `tools/logprob_agree.py`:
Loom, the llama.cpp route (the fallback), and the CPU route as the arbiter. Loom routes only when

- **agree(Loom, CPU) >= agree(fallback, CPU)** in greedy text agreement over the fixed prompts
  (owner-approved 2026-11-23; it replaces "Loom vs fallback >= 95 %", which fails Loom whenever the
  fallback itself is the arm that is wrong, as on MiniCPM4, engine#3340), **and**
- accuracy holds: every prompt answered (no empty-vs-empty rows), and each Loom/CPU divergence a
  near-tie on the CPU arm (the first differing token's top-2 margin is reported, so a near-tie is told
  from a confident error). Loom returns no logprobs, so KLD on the Loom arm is reported as "cannot
  tell", never filled in.

A row is valid only when Loom's prompt token count (`/v1/chat/completions/input_tokens`) equals ours
(`/apply-template` + `/tokenize` with `add_special`), with thinking on and off where the template
toggles: a card whose chat block drops BOS or misrenders thinking answers a different prompt.

**Speed.** Interleaved on strixhalo, at least four rounds, each round running every arm once, one
server at a time: the fallback, raw loom-serve, and Loom through `1bit serve --loom-routes` (the route
users get), with the same prompts, a discarded warm-up per arm, and load, memory and Tctl recorded per
round. Loom routes only when it is faster through `1bit serve` (or equal on speed and better on
accuracy).

The box runs device arms only: the CPU arbiter and the comparison run on another machine, and each
`box-hold` ticket (at most 45 minutes) states its RAM need and leaves at least 16 GB free. The result
tables go on the model's issue, and the entry's `gate` links them.
