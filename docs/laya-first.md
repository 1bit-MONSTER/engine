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
# Laya first: the first request on a cold Lemonade

Owner (2026-11-03): Lemonade starts with no model loaded, and "when we send the first request it needs to be our
laya router making that decision and picking our specific models that we have on hf. we need to own that space
completely hip kernels and our npu engine."

**Answer in one paragraph.** Our Lemonade fork already routes a request through a policy engine
([`collection.router`](https://github.com/1bit-MONSTER/lemonade/blob/main/docs/dev/router-policy.md)): ordered rules over conditions pick a candidate model, and the
decision is reported on the response. What is missing for the cold first request is four things, and none of them is a
new router: (1) a condition that costs 25 ms instead of loading a model to decide, which is Laya ([laya.md](laya.md));
(2) a shipped table naming **our** Hugging Face models and their device; (3) the name `auto` resolving to that table;
(4) the scorer already warm when the first request arrives. So the work is one classifier type in the fork, one
resident decision service in the engine, one JSON table in the release package, and the alias.

## 1. What exists today, measured

Fork build `2026.46.0~1778.8888dd42` (`~/wt/lemonade-46/fork/build/lemond`), run on `127.0.0.1:13499` with an empty
cache and config, 2026-11-03. Three probes, exact results:

| probe | result |
|---|---|
| `POST /v1/routing/validate` with a deterministic policy (`min_total_chars` → the 27B, `keywords_any` → ZAYA1-8B, default → 0.6B) over three prompts | `code` → `1bit.ZAYA1-8B` in **9 ms**, a 2,170-char document → `1bit.Qwen3.8-27B` in **4 ms**, a one-line question → default in **3 ms**; `matched_rule` and `default_used` correct in all three |
| `POST /v1/chat/completions` with `"model": "auto"` | **`{"error":{"code":"model_not_found", …}}`** — there is no `auto` |
| the same validate call with a `{"type":"laya"}` classifier | **`{"error":"Invalid routing policy: unknown classifier type: laya"}`** |

So the routing engine, the decision format, the per-class dispatch and the validator all work today with zero new
code, and the two gaps are exactly the two the owner names: Laya is not a condition, and `auto` is not a name.

## 2. Request flow in our Lemonade fork

1. Client → `POST /v1/chat/completions` (`src/cpp/server/routes/openai/text/chat_completions.cpp`), `model` field.
2. The name is resolved before anything is loaded: `Router::resolve_model_name` (`src/cpp/server/router.cpp:160`) and
   `Router::try_get_model_info` (`:164`). An unknown name is a 404 `model_not_found`
   (`routes/openai/models/models_id.cpp:74`, `routes/lemonade/load/load.cpp:184-185`) — measured in §1.
3. If the name is a `collection.router` collection, the routing engine runs first and picks a candidate:
   `src/cpp/server/routing_policy.cpp` (engine), `routing_policy_parser.cpp` (policy → rules, per-type capability
   checks at `:417-440`), first matching rule wins, fail-open to `default_model`. `POST /v1/routing/validate`
   (`routes/router/routing_validate.cpp`) runs the same engine without dispatching, which is how a policy is tested
   before registering it.
4. The chosen candidate is loaded by `Router::load_model` (`router.cpp:849`) → `ModelManager`
   (`src/cpp/server/model_manager.cpp`, pulls a model that is not on disk) → the onebit backend's
   `OnebitServer::load` (`src/cpp/server/backends/onebit/onebit_server.cpp:220`), which builds
   `1bit serve -m <resolved_path> --device <onebit_backend|auto> --alias <name> --ctx-size <ctx>` (`:242-250`) and,
   for a GGUF on a one-hop-enabled build, starts the llama-server that `1bit serve --print-backend` names.
5. The reply carries `x-lemonade-route` (the matched rule id) and, when the request sets `route_trace: true`,
   `x_lemonade_route` (the decision plus its per-condition trace); a streaming reply attaches it to the first SSE
   event.
6. Model-backed conditions reach the backends only through `ClassifierServices`
   (`src/cpp/include/lemon/routing_policy.h:162`): `embed`, `run_classifier`, `chat`, bound to
   `Router::embeddings` / `classify` / `chat_completion` in `routing_classifier_services_router.cpp:11-21`.

The fork must not link the engine's scorer: every other backend is a subprocess, and this one is too.

## 3. The one missing seam: `type: "laya"`

**Fork side.** Accept `{"id":"laya","type":"laya", …}` as an entry in `routing.classifiers`, add a
`LayaClassifier : Classifier` whose `evaluate()` returns `Score{labels:{code,prose,short,long_doc}}` from the scorer,
and a fourth injection point `ClassifierServices::classify_laya` backed by a `LayaClient` that owns the subprocess.
Everything downstream is unchanged: the `classifier` band (`min_score`/`max_score`), `on_error`, the rule engine, the
decision and its trace, the validate endpoint.

**Engine side.** A resident decision service, `1bit route --serve` (one JSON line in, one out), because the scorer is
too expensive to start per request and unnecessary to keep per model:

| | cost | source |
|---|---|---|
| load the safetensors scorer | **2.7 s**, once | [laya.md](laya.md) "Cost per decision" |
| GGUF daemon (`laya daemon <gguf> --family typed-decisions --device hrx`) ready | **627 ms** | [laya.md](laya.md) "Proof on the HRX engine" |
| decision, GGUF scorer on HRX0 | **25 ms p50** (15 ms warm), 95.5% on 200 labelled requests | [laya.md](laya.md) "The GGUF scorer" |
| decision, C++ scorer on CPU | 538 ms | same |

`1bit route --serve` reuses `laya/route.h`'s `classify_request` and the daemon wiring `serve` already does
(`app/serve.cpp:1420`, `classify_request_gguf`), so it adds a protocol, not a scorer. `long_doc` keeps the engine's
size gate (≥ 1,024 characters is `long_doc` without asking the model, `laya/route.h:kLongDocChars`), so the classifier
returns confidence 1.0 for it exactly as `serve` does.

This is deliberately small: no new rule engine, no new decision format, no new HTTP endpoint, no change to the policy
language. A policy can mix `{"keywords_any": [...]}` with `{"classifier":"laya","label":"code","min_score":0.5}` on the
same rule, so the deterministic fallback of §1 stays available when the scorer is down.

## 4. The routing table that ships

The table is the product surface; the policy is the mechanism. The table desugars into a `collection.router` policy
(one rule per class, `route_to` that class's model), so the engine, the validate endpoint and the decision format are
untouched.

Installed at `~/.local/share/1bit/laya-first.json` (in the release package: `share/1bit/laya-first.json`), overridable
with `--laya-first FILE`:

```json
{
  "version": "1",
  "scorer": {"hf_repo": "mys/laya-typed-decisions-GGUF",
             "file": "laya_typed_decisions_q8_0.gguf", "device": "hrx"},
  "pull_budget_s": 600,
  "default_model": "Qwen3-0.6B-1bit",
  "classes": {
    "short":    {"model": "Qwen3-0.6B-1bit",
                 "hf_repo": "1bit-MONSTER/Qwen3-0.6B-GGUF",
                 "file": "Qwen3-0.6B-Q8_0.gguf", "device": "hrx", "ctx": 8192},
    "prose":    {"model": "ZAYA1-8B-1bit",
                 "hf_repo": "1bit-MONSTER/ZAYA1-8B-GGUF",
                 "file": "zaya1-8b-Q4_K_M.gguf", "device": "hrx", "ctx": 16384},
    "code":     {"model": "Qwen3-Coder-30B-A3B-Instruct-1bit",
                 "hf_repo": "1bit-MONSTER/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                 "file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", "device": "hrx", "ctx": 32768},
    "long_doc": {"model": "Qwen3.8-27B-1bit",
                 "hf_repo": "1bit-MONSTER/Qwen3.8-27B-UD-Q4_K_XL-MTP",
                 "file": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "device": "hrx", "ctx": 65536,
                 "extra_files": ["MTP/mtp-Qwen3.8-27B-Q4_0.gguf"]}
  }
}
```

| field | meaning |
|---|---|
| `model` | the catalogue id the router's `route_to` names (so a pulled row is an ordinary Lemonade model) |
| `hf_repo` + `file` | what to pull when the row is not on disk; a Q4NX row names no `file` and pulls the whole directory (`1bit-MONSTER/Qwen3.6-35B-A3B-NPU`) |
| `device` | `hrx` (our HIP kernels), `npu` (our NPU engine, Q4NX), `hybrid` (`--device hybrid --prefill-provider <NPU route>`, [hybrid.md](hybrid.md)), or `auto`; it becomes the candidate's `recipe_options.onebit_backend`, so `1bit serve` still makes the final check |
| `ctx` | the candidate's `ctx_size` |

Every repo and file name above was read from the Hub on 2026-11-03 (`1bit-MONSTER` has 68 repositories, all four
classes have a first-party repo with the file named here). The two classes that also have an NPU form get a second row
rather than a second device: `Qwen3.6-35B-A3B-NPU` for a heavy class on the NPU, and the FLM rows already in
`server_models.json` for a small NPU chat model (see "Owner decisions").

## 5. Cold start

Timeline of the first request on a box where nothing is downloaded, with the costs above and those of the model itself:

1. lemond start: spawn `laya daemon <scorer gguf> --family typed-decisions --device hrx` beside backend discovery.
   The scorer is a 421M-parameter Q8_0 GGUF, so it is a routing helper, not a candidate: it is exempt from the model
   cap (`pinned_helper_models.classification`, already in `/api/v1/health`) and takes **627 ms** to answer.
2. Request: classify (**25 ms**) → table row → candidate.
3. Candidate not on disk: **hold the request and pull it**, with progress. Streaming replies get a `: pull <pct>`
   SSE comment every 5 s (clients must already ignore comments); non-streaming replies wait, bounded by
   `pull_budget_s`.
4. `Router::load_model` → `1bit serve` → first token.
5. Later requests: the model is resident, so they pay neither the pull nor the load; the class is cached per
   conversation the way `serve --laya` caches it (the same request text is not re-classified turn after turn).

What happens when the pull cannot finish inside the budget or fails, decided now rather than at the first user:

- the request is answered by the class's `fallback` if the table names one, else by `default_model` (the small model
  that is on disk, and the one the table lists first so a pre-seed makes this a "small resident fallback");
- the decision says so — `"degraded": true`, `"reason": "pull_timeout" | "pull_failed"`, plus the class it wanted — in
  `x_lemonade_route` and in the log line. A degraded answer is never presented as the class's model.

## 6. Override, and what never happens

- An explicit `model` always wins: any name that is not `auto` and not a router collection takes today's path with no
  Laya involvement and no routing header (§1's first probe and today's 404 behaviour are both unchanged).
- Laya failing (daemon dead, timeout, malformed answer) is `Score.ok = false`, which existing `on_error` handling turns
  into `match_false` for that condition, so the request falls through to `default_model`. A dead scorer can slow a
  cold start by 627 ms; it can never refuse a request.
- The decision is local: no network at decision time. Only a pull touches the Hub, and only for a repo the table names.
- No implicit pulls: a model the table does not name is never downloaded, and `auto` never routes to a cloud provider.

## 7. Gate: a release-check row `laya-auto`

On a cold lemond (empty cache), three prompts — one code, one 2 KB document, one one-liner — each must land on the
expected class, model and device, with `x-lemonade-route` naming the class rule, decision time under 100 ms, and the
pull+load counted separately (so a regression in either is visible). A second immediate request must show no pull and
no re-decision, and an explicit model name must produce no routing header at all. That is four assertions per prompt,
which is the same shape as the existing `chat-hrx`/`chat-npu` rows.

## 8. Order of work

1. **Engine**: `1bit route --serve` (resident JSON-lines decision service; `tests/laya_route_serve.sh` covering a
   decision, `long_doc`'s size gate, a dead daemon, and determinism over 5 runs), plus `--json` on `1bit route`.
2. **Fork**: `type: "laya"` classifier, `LayaClient`, parser/registry acceptance, policy-schema entry, docs (§3-§6
   contract), tests.
3. **Engine**: `laya-first.json` + desugaring to a `collection.router` policy + packaging.
4. **Fork**: `auto` resolving to the shipped router collection (this changes the line in the fork's `docs/dev/router-policy.md`
   that says there is no `auto` model).
5. Release check row `laya-auto` (§7).

## 9. Owner decisions

1. **The small NPU row is not ours.** Our only first-party NPU container on the Hub is `1bit-MONSTER/Qwen3.6-35B-A3B-NPU`
   (22.9 GB); the small NPU rows in `server_models.json` are FastFlowLM checkpoints on the pinned FLM v1.0.7. Is an FLM
   checkpoint "our model" for `chat-small` on the NPU, or do we publish a small Q4NX container of our own?
2. **Catalogue vs table.** The catalogue points at `unsloth/…` for Qwen3-Coder-30B-A3B and Qwen3.8-27B while we host
   first-party mirrors under `1bit-MONSTER`. The table names ours; confirm the row ids move with it.
3. **`auto`.** A reserved model name (needs a resolution hook in `Router`), or a shipped `collection.router` collection
   literally named `auto` (no code change, but it collides with any user model of that name)?
4. **The code class's first pull is 18.6 GB.** Start `code` on ZAYA1-8B (5.6 GB) and treat Qwen3-Coder as an
   escalation, or accept the big pull on the first code request?
