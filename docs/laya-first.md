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

Laya itself, asked the same three prompts through the installed `1bit route --classify` (CPU scorer,
`~/.local/share/1bit/laya`, 2026-11-03), agrees with the classes those rules assume — which is why the table in §4 can
be written today and only the scorer call is missing:

| prompt | chars | Laya | confidence | decision | scorer load, per invocation |
|---|---|---|---|---|---|
| `Write a Rust function that sums a slice.` | 40 | **code** | 0.44 | 361 ms | 3,819 ms |
| `What is a good name for a cat?` | 30 | **short** | 0.38 | 448 ms | 3,062 ms |
| a 1,751-character paddock log | 1,751 | **long_doc** | 1.00 | **0 ms** (size gate, no forward pass) | 4,307 ms |

The 3.1-4.3 s in that last column is the whole reason §3 asks for a resident service rather than a process per request:
it is paid once at lemond start instead of once per cold conversation.

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
| `on_busy` | optional: the device to fall back to when the GPU is saturated or hot, with the condition that triggers it (see "Device: HRX by default" below). Absent means never fall back |
| `ctx` | the candidate's `ctx_size` |

Every repo and file name above was read from the Hub on 2026-11-03 (`1bit-MONSTER` has 68 repositories, all four
classes have a first-party repo with the file named here). The two classes that also have an NPU form get a second row
rather than a second device: `Qwen3.6-35B-A3B-NPU` for a heavy class on the NPU, and the FLM rows already in
`server_models.json` for a small NPU chat model (see "Owner decisions").

### A row is not served with an unexpected template

A routed row is a **pull from the Hub and then a load**, so it inherits the template-backdoor problem
twice over: the file arrives from a third party, and the template inside it is a program the
llama.cpp server runs on every request (the engine's own forward never reads one). A pinned route
table is a good place to close that, because it is the only place in the pull path that knows which
model was *meant* to arrive.

So a GGUF row gains one required field, and a Q4NX row the same field read from its container:

```json
"prose": {"model": "ZAYA1-8B-1bit",
          "hf_repo": "1bit-MONSTER/ZAYA1-8B-GGUF",
          "file": "zaya1-8b-Q4_K_M.gguf", "device": "hrx", "ctx": 16384,
          "template_sha256": "b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f"}
```

| field | meaning |
|---|---|
| `template_sha256` | the sha256 of the exact `tokenizer.chat_template` bytes the row expects (a GGUF's metadata key, or a Q4NX container's `tokenizer_config.json` string); `"none"` for a row that must carry no template. Omitting it is a table error, not a default — a row with no expectation is a row nothing checks ([template-publishing.md](template-publishing.md)) |

Two checks, in order, and both are refusals of *that row*, never of the request:

1. **At pull time**, before the row is registered with Lemonade and before it is named in a decision:
   hash the template of the file that landed (and of every `extra_files` entry that is served as a
   model) and compare it with the row. On a mismatch the row is not registered, the class falls back
   to `default_model`, and the reason is logged and carried in `x-lemonade-route` as
   `why: template_mismatch` (the same shape as `why: gpu_busy`). Nothing is deleted and nothing is
   re-uploaded: a mismatch is information for the operator, and the file may be perfectly good under a
   different row.
2. **At load time**, the engine names the template's sha256 in one line and compares it with its own
   pin (`1bit serve --template-pin-strict` in the appliance's shipped row), so a file that changed
   *after* the pull is refused at the load rather than served ([serve.md](serve.md#chat-template---template-pin---no-template-pin---template-pin-strict)).

Why both: the pull-time check is the only one that knows what was expected (it is the only place with
the table), and the load-time check is the only one that covers a model loaded by name, outside the
router, months later. Neither is a trust anchor for a model nobody has pinned — for those the line is
the deliverable, and it is what makes the difference visible in a log.

`~/.local/share/1bit/laya-first.json` ships with the digests measured on 2026-11-05 for the rows above
(`zaya1-8b-Q4_K_M.gguf`, `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` and `Qwen3.8-27B-UD-Q4_K_XL.gguf`
are the same files the engine's `config/template-pins.json` already pins; a Q4NX row is checked from
its `tokenizer_config.json` because it has no GGUF). Adding a row means adding its digest — the table
is not allowed to grow a row that skips that step, and the table's validator (§8 step 3) fails one that
does.

### Device: HRX by default, and the NPU only when the GPU is busy or hot

Measured on Strix Halo on 2026-11-04 (`llama-bench`, same box, same file, `-r 2`): a dense 27B-class model —
Ternary-Bonsai-2-27B — runs on HRX0 at **124-152 tok/s prefill and 15.3 tok/s decode**, against **13.9 / 3.05** on our
own NPU route (`bonsai2-27b-t2s2`, commit `916e90d`). HRX wins **both** halves, by 10.7-13.2× and 5.0×, so a heavy
class's `device` defaults to `hrx` and the NPU is **not** a faster second half: the "HRX prefill + NPU decode" split
that [hybrid.md](hybrid.md) designs is not worth building for this model, and the same measurement re-run after a
reboot (138.8 / 151.8 / 131.1 tok/s prefill, 13.9 decode with four other lanes running) does not move that.

What the NPU is still good for is **contention**. An HRX reply pins the *shared* iGPU for its whole length (512 tokens
≈ 33 s at 15.3 tok/s), the same window the fleet serialises through its fair GPU queue, while the NPU has its own
device. So the NPU is a **fallback for a saturated or hot GPU** — a routing condition, not a speed choice. A class
may name it:

```json
"long_doc": {
  "model": "Qwen3.8-27B-1bit", "device": "hrx", "ctx": 65536,
  "on_busy": {"device": "npu", "model": "<the class's NPU row>",
              "queue_depth": 1, "tctl_c": null, "wait_s": 0}
}
```

| field | meaning |
|---|---|
| `queue_depth` | fall back when at least this many GPU jobs are waiting (live tickets in the fleet's fair queue, the same FIFO `gpu-hold` arbitrates) **or** the device is held by a label that is not this lemond |
| `tctl_c` | fall back at or above this package temperature, read from the k10temp hwmon `temp1_input` — the file `gpu-hold`'s own start gate and thermal guard read; `null` means "the guard's critical minus a margin", filled from the measurement below, not guessed |
| `wait_s` | wait this long for the GPU to come back before switching; `0` switches immediately |

Three properties the condition keeps:

- **Decided once, at load time**, never per request: a class changes device between conversations at most, never
  mid-conversation, and the decision header ([§6](#6-override-and-what-never-happens)) says
  `"device": "npu", "why": "gpu_busy"` or `"gpu_hot"`. The decode penalty is 5×, so a fallback answer that did not
  say so would be a silent slowdown presented as a normal one.
- **Off for interactive classes by default.** `short` and `code` do not fall back — a 5× slower answer is worse than
  waiting for the GPU — while `long_doc` and batch-style classes may.
- **The thresholds come from H0, not from this note.** H0 is the unmeasured half: what a 512-token HRX decode costs in
  power and package heat, and what pinning the iGPU for 33 s costs the lanes sharing it. Until that lands, `on_busy`
  is **absent from the shipped table** and the gate row asserts the HRX landing only; when it ships, the rule gains a
  warm-up guard (no fallback in the first minute after boot, when the thermal guard itself is still settling).

### The table in the binary: `1bit laya-first`

`config/laya-first.json` ships compiled into the engine (like `config/recipes.json`), and
`1bit laya-first` validates it and prints the `collection.router` policy it desugars into:

```sh
1bit laya-first --check                 # 4 classes, default Qwen3-0.6B-1bit
1bit laya-first --policy laya-first.collection.json
```

```json
{"version": "1", "model_name": "user.Laya-First", "recipe": "collection.router",
 "components": ["Qwen3-0.6B-1bit", "ZAYA1-8B-1bit", ...],
 "routing": {"candidates": [...], "default_model": "Qwen3-0.6B-1bit",
   "classifiers": [{"id": "laya", "type": "laya",
                    "labels": ["short", "prose", "code", "long_doc"]}],
   "rules": [{"id": "laya-code", "match": {"classifier": "laya", "label": "code", "min_score": 0.0},
              "route_to": "Qwen3-Coder-30B-A3B-Instruct-1bit"}, ...]}}
```

Two details are load-bearing rather than cosmetic. **`min_score: 0.01` on every class rule** — low, so
the band's default floor of 0.5 cannot drop the decisions this table exists to route (Laya's confidence for
`code` is about 0.44), but **not zero**: the classifier reports only the class Laya chose, so every other
label scores exactly `0.0`, and a `0.0` floor matches `0.0 >= 0.0` — the *first* class rule then swallows
every request. Measuring that cost one live run: a `short` prompt routed to the `code` model while every
self-test stayed green, because a verdict can only see that *a* rule matched, not that it was the right one.
**Labels on the classifier entry**: declaring the four classes is what makes a typo in a rule's `label` a
parse error in Lemonade instead of a rule that silently never matches.

The name is `user.Laya-First` because that is the only shape Lemonade will register at run time
("Registered model definitions must use a non-empty `user.*` name", measured against a live lemond); the
appliance registers the desugared policy at first boot, so the desugar has to emit a registrable name.

`tests/laya_first.sh` pins the desugaring (19 checks, including byte equality with
`tests/laya_first_policy.json`), refuses a table with a device typo or a missing class by name, and — when
a Lemonade checkout is reachable — validates the emitted policy against that repo's
`route_policy.schema.json`, because the policy is for Lemonade and its schema is the contract.

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
which is the same shape as the existing `chat-hrx`/`chat-npu` rows. When `on_busy` ships (§4), the row gains a fifth
case — a ticket held by another label makes the document prompt land on the NPU row, the header says `why: gpu_busy`,
and the same prompt with the queue empty lands on `hrx` — because a fallback that only ever runs on one side of the
condition is the classic untested branch. It gains a sixth: a row whose `template_sha256` does not match
the file on disk is not registered, the class lands on `default_model`, and the header says
`why: template_mismatch` — a template gate that only ever passes is the same untested branch one layer
down.

## 8. Order of work

1. **Engine**: `1bit route --serve` (resident JSON-lines decision service; `tests/laya_route_serve.sh` covering a
   decision, `long_doc`'s size gate, a dead daemon, and determinism over 5 runs), plus `--json` on `1bit route`.
2. **Fork**: `type: "laya"` classifier, `LayaClient`, parser/registry acceptance, policy-schema entry, docs (§3-§6
   contract), tests.
3. **Engine**: `laya-first.json` + desugaring to a `collection.router` policy + packaging, with
   `template_sha256` required per row and the pull-time check of the same section.
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

### What measuring the implementation added to those four

Each of these was found by running the code against a live server tonight, not by reading it. They narrow the
answers; none of them decides anything for you.

1. **The small NPU row: provenance matters more than size.** The row has to name a checkpoint the appliance can pull
   *and* prove. An FLM checkpoint is a third party's artefact on a pinned FLM (v1.0.7 today), so the row would be
   "our router, their model" — defensible for a small chat class, but it is a different claim from every other row in
   the table, and the release check would need to say which kind of row it is looking at. A 0.6B-class Q4NX of our own
   keeps one claim for the whole table and costs a conversion. **Recommendation: our own container**; if an FLM row
   ships instead, the table needs a `provenance` field so the row and the release check cannot disagree about what
   "ours" means.
2. **Catalogue vs table: the table's names must be registrable, and that is now a hard constraint.** Measured: Lemonade
   refuses to register a definition outside `user.*` — *"Registered model definitions must use a non-empty `user.*`
   name. Received: 1bit.Laya-First"*. So whatever the row ids become, the **router collection** is registered as
   `user.<something>` and the desugar emits that; the *components* keep their catalogue ids (they are ordinary pulled
   models, not definitions). The table already names `1bit-MONSTER` mirrors, so the question is only whether the
   catalogue rows are re-pointed at them — which the `hf_repo` field already answers independently, without moving ids.
   **Recommendation: leave ids alone; treat `hf_repo`/`file` as the pull source.**
3. **`auto`: the two options differ in who owns a name, and one of them is cheap to reverse.** A reserved name needs a
   hook in `Router` and is invisible to the catalogue; a shipped collection named `auto` needs no code but squats the
   name for every user. There is a third shape the work made obvious: keep the name `user.Laya-First` (what the desugar
   already emits and what the release row now reads *from the engine*, not a constant) and let the **appliance** alias
   `auto` to it — an alias is configuration, reversible, and does not squat anything in the product. **Recommendation:
   alias at install time**, and raise the reserved-name hook only if a user-facing `auto` is wanted on stock Lemonade
   too.
4. **The 18.6 GB first pull: the table already made this a policy question, not a code question.** Nothing in the code
   forces the big model to be `code`'s *first* choice: the class rule points wherever the table says, and
   `tests/laya_first.sh` re-derives the policy from the table. So this can ship as ZAYA1-8B for `code` with
   Qwen3-Coder as a second table entry to promote later, at the cost of one line in `config/laya-first.json` — and no
   code change to reverse. **Recommendation: start small, promote with evidence.**

**One measurement that changes how the table must be written, whichever way 1-4 go:** a class rule needs a floor
*above* zero (`min_score: 0.01`, §4). At `0.0` the first rule matches every class, because the classifier reports only
the class Laya chose and every other label scores exactly `0.0` — a `short` prompt routed to the `code` model while
every test stayed green. Any table or hand-written policy built from this note has to carry that floor.
