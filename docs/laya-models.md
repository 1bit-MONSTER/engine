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
# Several models behind one port: `1bit serve --laya-models`

`--laya` ([laya.md](laya.md)) picks a *device* for the one model a `serve` runs. `--laya-models FILE`
picks the *model and its device* for every request: a model table names its own classes, the Laya
question that tells them apart, and for each class the model that answers it and where it runs.
A client sends `"model": "auto"` and Laya decides each request.

The first table is the game agent's (engine #1313): quick reflex steps go to a small model on the
NPU, and planning goes to a bigger model on HRX.

```sh
1bit serve --laya-models config/laya-models-game.json --laya --port 8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"A zombie is 2 blocks to the left. Next action?"}]}'
```

## The table

```json
{
 "question": {
  "instructions": "What does the game agent need to answer this step?",
  "classes": {
   "reflex": "one quick action from what is in view right now: ...",
   "plan": "planning: a crafting tree, a build plan, choosing the next goals, ..."
  }
 },
 "default": "plan",
 "min_confidence": 0.0,
 "size_gate": {"chars": 1024, "class": "plan"},
 "classes": {
  "reflex": {"model": "~/models/npu/Qwen3-0.6B", "device": "npu", "alias": "Qwen3-0.6B-NPU"},
  "plan": {"model": "~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "device": "hrx",
           "alias": "Qwen3-Coder-30B-A3B-HRX", "args": ["--ctx-size", "16384"]}
 }
}
```

| field | meaning |
|---|---|
| `question.instructions`, `question.classes` | one Laya `choice` question; the classes are its options, in the file's order |
| `default` | the class a request takes when Laya's confidence is below `min_confidence` (default: the first class) |
| `size_gate` | optional: a request this long or longer is that class, with no model call and no `min_confidence` |
| `classes.<c>.model` | a `.gguf` file or an NPU model directory; `~` and paths relative to the table's directory resolve |
| `classes.<c>.device` | `npu`, `hrx`, `cpu` or `hybrid`, passed to the row's `1bit serve --device` |
| `classes.<c>.alias` | the name replies carry (default: the file name); a request naming it skips Laya |
| `classes.<c>.args` | extra `1bit serve` arguments for that row |
| `classes.<c>.bin` | optional: the `1bit` binary that serves that row (default: this one), e.g. a release build that carries the sealed NPU kernels while this one is a source build |

Every class needs a row, and every row a class; an unknown device, a missing model file, or one alias for
two different backends is refused by name at start. Two classes may name the same model, device and args:
they share one backend.

## What a request does

1. Each row runs as its own `1bit serve -m <model> --device <device>` child on a loopback port, started
   when serve starts. The NPU row is exactly the NPU route, the HRX row exactly the HRX route. `/health`
   answers 200 once every row is ready.
2. A request whose `model` is a row's alias goes to that row (`pinned`). Any other name (`auto` by
   convention) is decided by Laya: the table's size gate, then the table's question about the **last user
   message** (each step of an agent is its own decision; the system prompt is the same on every one).
   A state asked before is answered from a cache (4,096 states), at 0 ms.
3. The class's row answers. Below `min_confidence` the `default` class's row answers.

The scorer is the one `--laya` uses: ggmlc's GGUF daemon on HRX0 when the build has it, else the C++
scorer (`ONEBIT_LAYA_DEVICE=npu` with the NPU add-on puts its encoder on the NPU).

## What a reply says

| where | what |
|---|---|
| `X-1bit-Route` | `<class> <confidence> <device>`, the `--laya` format (`pinned 0.00 npu` for a pinned request) |
| `X-1bit-Route-Model` | the alias of the row that answered |
| `X-1bit-Route-Ms` | the Laya decision in milliseconds (0.0 when pinned or cached) |
| `X-1bit-Route-Id` | the decision's id in `/v1/route/decisions` |
| body `route` (non-streamed) | `id`, `class`, `laya_class`, `confidence`, `model`, `device`, `router_ms`, `cached`, `request_ms` |
| body `model` | the row's alias |

`GET /v1/route/decisions?since=ID&limit=N` returns the last decisions (up to 512 kept) after `since`,
oldest first: the fields above plus the reply's `prompt_tokens`, `completion_tokens`, `prompt_tok_s` and
`decode_tok_s` from its `usage` and `timings` (llama-server and the NPU routes both send them; a streamed
reply has none of them, and no status). `GET /v1/route/table` returns the table's question and each class's
model and device. `GET /v1/models` lists `auto` and every alias. `GET /metrics` adds, by class, model and
device: `onebit_route_decisions_total`, `onebit_route_router_ms_sum` / `_count` (Laya decisions only),
`onebit_route_request_ms_sum` / `_count`, `onebit_route_completion_tokens_total`,
`onebit_route_prompt_tokens_total`, and `onebit_route_last_decode_tok_s` by model and device. These are the
numbers a demo HUD reads; every one was measured by serve or reported by the backend.

## Limits

- `--laya-models` takes no `-m`, `--device`, `--pm`, `--embed`/`--rerank` or a role; per-row options go in
  `args`. Linux only.
- Laya's zero-shot answer to a new question is a starting point, not a measured router: the base
  checkpoints are near chance on decision types they were not tuned for (Laya's own README). The game
  table's accuracy is measured below, and a fine-tune on the game's own logged decisions is the plan for
  raising it (engine #1313).

## Measuring and fine-tuning the router

```sh
# accuracy of the table's question over labelled states, by the path serve takes (size gate, then Laya)
1bit route --laya-models table.json --laya-model DIR --eval cases.jsonl [--json] [--per-case per.jsonl]
# the decision head's input for each state, for a head-only fine-tune
1bit route --laya-models table.json --laya-model DIR --dump-encoder cases.jsonl --out dump/ [--rotate]
```

A case is one JSON line, `{"state": TEXT, "label": CLASS, "src": optional id}`, where `state` is exactly what
serve classified (the last user message). A line without a `label` is skipped by `--eval`, so a decision log
whose newest lines have no outcome yet can be passed as it is.

`--eval` prints accuracy, per-class support, recall and precision, the confusion matrix (rows true, columns
chosen), serve's mean confidence, the mean top probability, ECE (10 equal bins over the top probability), log
loss and the decision time p50 and p90. Size-gated cases are counted (`gated`), since serve would gate them too.

`--dump-encoder` writes three files to `--out`:

| file | content |
|---|---|
| `h.f32` | every sequence's encoder output after the final norm, float32 little-endian rows of `d`, back to back |
| `cases.jsonl` | per sequence: `src`, `rotation`, `label` (an index into `options`, -1 unlabelled), `options` (the order the sequence carried them), `qtype`, `offset` and `len` (rows in `h.f32`), `markers`, and the engine's own `logits` and `probabilities` |
| `dump.json` | `d`, `rows`, `cases`, `skipped_size_gated`, `rotate`, the question and the checkpoint |

The head (`type_emb`, the two head layers and `scorer`) maps these rows to the option logits, so a trainer
needs neither the encoder nor a second tokenizer. The engine's logits in each line let it check first that its
head reproduces the engine. Size-gated cases are skipped because serve never asks Laya about them. `--rotate`
writes every rotation of the options with the label moved along, so a fine-tune cannot learn "the first option
wins".

Measured on this box (`typed-decisions/`, the 20 game steps of the measurement below, the CPU scorer under
`thermal-run`, 4 cores at nice 19):

| | result |
|---|---|
| `--eval` | 19/20, the same single miss as serve's run below; reflex recall 0.917, plan recall 1.000; mean confidence 0.154, ECE 0.248, log loss 0.373; decision p50 1159 ms on 4 cores at nice 19 |
| `--dump-encoder --rotate` | 40 sequences, 4,404 rows (18 MB) |
| a torch copy of the head over the dump | reproduces the engine's probabilities within 2.4e-7 (logits 1.3e-6) on all 40 |
| option order | mean \|p(reflex) in one order - the other\| 0.065 (max 0.171); 1 of 20 decisions flips with the order |

## Tests

`tests/laya_models_test.cpp` (ctest `laya_models`, no checkpoint): parsing, the question, the pick and
its fallback, aliases, shared backends, the size gate, the refusals, and the shipped game table.
`tests/laya_eval_test.cpp` (ctest `laya_eval`, no checkpoint): the eval's arithmetic on hand-computed cases.
`tests/laya_eval_e2e.sh` (ctest `laya_eval_e2e`, with `ONEBIT_LAYA_ROUTE_MODEL`): `--eval` and `--dump-encoder`
on the real scorer, the report's counts, the dump's files against each other, the refusals.
`tests/laya_models_e2e.sh` (ctest `laya_models_e2e`, with `ONEBIT_LAYA_ROUTE_MODEL`): two rows on fake
llama-servers behind real `1bit serve` children; the reply comes from the row the headers name, pinned and
cached requests, the decisions log, the table, `/v1/models` and `/metrics`, and the refusals.

## Measured

Strix Halo, 2026-11-07, `config/laya-models-game.json` with this box's paths: Qwen3-0.6B (Q4NX) on the NPU and
Qwen3-Coder-30B-A3B Q4_K_M on HRX, the C++ scorer on the `typed-decisions/` checkpoint (CPU), and 20
hand-labelled game steps (12 reflex, 8 plan) sent as `model: auto`, `max_tokens 48`, temperature 0.
The NPU row ran from a release build (`bin`), because the source build had no NPU add-on.

| | result |
|---|---|
| ready, both rows (the 30B from the page cache) | 16 s |
| class right, zero-shot | **19/20**: reflex 11/12, plan 8/8. The one miss, "place a wall block here", went to plan at confidence 0.01 |
| confidence | low throughout: 0.01-0.57, median 0.13. `min_confidence` stays 0 until the fine-tune |
| Laya decision (`router_ms`) | p50 **302 ms** (259-358): the CPU scorer on a shared box. ggmlc's GGUF scorer on HRX (25 ms p50 in [laya.md](laya.md)) is the fast path |
| decode, from the replies' `timings` | NPU p50 85.2 tok/s, HRX p50 86.8 tok/s |
| headers, `route` object, `/v1/route/decisions` and `/metrics` | agree on all 20 (11 decisions npu, 9 hrx) |

Qwen3-0.6B thinks by default, so a 48-token budget can end inside the thinking: every NPU reply above had an empty
`content` with all 48 tokens in `reasoning_content`. A request with `"chat_template_kwargs":{"enable_thinking":false}`
gets the answer in `content` (checked on the same row). Routing does not depend on it.
