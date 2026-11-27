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
| `qwen38-27b-ud-q4-k-xl` | [loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL](https://huggingface.co/loom-fun/unsloth-Qwen3.8-27B-UD-Q4_K_XL) @ `3e43d41d` | engine#3158 |
| `qwen3-0.6b-q4_k_m` | [loom-fun/unsloth-Qwen3-0.6B-loom](https://huggingface.co/loom-fun/unsloth-Qwen3-0.6B-loom) @ `564d578f` | engine#3158 |
| `qwen3-4b-q4_k_m` | [1bit-MONSTER/Qwen3-4B-GGUF-loom](https://huggingface.co/1bit-MONSTER/Qwen3-4B-GGUF-loom) @ `c6e13152` | engine#3158 |
| `qwen3-4b-instruct-2507-q4_k_m` | [loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom](https://huggingface.co/loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom) @ `d0df214d` | engine#3158 |
| `qwen3.5-4b-ud-q4_k_xl` | [1bit-MONSTER/Qwen3.5-4B-GGUF-loom](https://huggingface.co/1bit-MONSTER/Qwen3.5-4B-GGUF-loom) @ `96c2f2f7` | engine#3158 |
| `qwen3-8b-q4_k_m` | [loom-fun/unsloth-Qwen3-8B-loom](https://huggingface.co/loom-fun/unsloth-Qwen3-8B-loom) @ `35d393f3` | engine#3158 |
| `inkling-2-mini-ud-q4_k_xl` | [1bit-MONSTER/Inkling-2-Mini-NVFP4-MXFP8-loom](https://huggingface.co/1bit-MONSTER/Inkling-2-Mini-NVFP4-MXFP8-loom) @ `82bf45eb` | engine#3158 (issuecomment-7216723223; no fallback arm: Loom vs the CPU arbiter). Our re-card of loom-fun's r8 `4e018e88` for unsloth's mainline GGUF, with layers 20-23's routed experts read as published (MXFP8) on native NVFP4/MXFP8 kernels (engine#3333); Apache-2.0 |

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
  relative path resolves under `--loom-cards DIR`, `ONEBIT_LOOM_CARDS`, else `~/.cache/1bit/loom-cards`. Where the
  card came from is not part of the entry: see [Card sources](#card-sources).
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
- `none`: no card; `arch`, `nearest` and `gap` say what a port needs (engine#3156) — and, for an
  architecture the public port starter cannot express (deepseek2/MLA today), that the port path is the private
  Loom kernel kit ([KNOWN-ISSUES.md](../KNOWN-ISSUES.md), engine#3221);
- `npu`: NPU entries, which never take the GPU route.

Each entry keeps the card's licence (the Loom programs) and the weights' licence apart; the weights
licence is read from the weights repository at our pinned commit, never taken from a sibling model.
Rebuild with `tools/loom_cards.py --build` from freshly fetched card locks; CI runs `--check`
(offline: every `exact` claim against `models/pins.json`) and `--self-test`.

## Card sources

**Open (owner, 2026-11-27): the "cards are public" line recorded here on 2026-11-24 was not an owner decision and is
withdrawn.** The standing rule holds: Loom kernels and programs we write ourselves stay in the private kit and are not
published as cards. Where a card's source comes from stays an open question for the owner; the options and costs below
are the record of what is being weighed.

A route entry names a card by **name**; the entry says nothing about where the card came from, and no route entry,
parser, tool or config names a source kind — `card_source` was a name in this note only when it landed (measured
2026-11-24: `git grep card_source` matched nothing outside `third_party`). A card reaches the directory serve
reads — `--loom-cards DIR`, `ONEBIT_LOOM_CARDS`, else `~/.cache/1bit/loom-cards` (`app/serve.cpp:1616-1622`) — one
of two ways today:

- `loom-fetch` on a published Hugging Face repo: every shipped entry's card is one (the table above),
  pinned at the card repository's revision.
- The 1bit OS payload: `os/v3/puppy/stage-loom.sh:96-111` downloads each file from
  `https://huggingface.co/<repo>/resolve/<rev>/<path>` and refuses one that is not its pinned sha256 and size
  (`os/v3/puppy/loom-cards.pin:21-101`); at boot `1bit-loom-cards` assembles it into `/data/loom-cards`, the
  `ONEBIT_LOOM_CARDS` 10lemonade exports (`os/v3/puppy/embed/etc/init.d/10lemonade:170-172`). The download has no
  credential path, so only public repositories are reachable through it.

serve resolves the name and checks no provenance: a card in the directory routes, and one that is not there is skipped
with one line while the model keeps the llama.cpp route (`app/serve.cpp:1662-1688`, `:1708-1712`). Two consequences
bound the design: a private card is servable today on a box where it is installed, with no route-table change, so the
decision is which installer may install it and which build or image may carry it; and a missing card is a skip, never
a failure, so a build that is supposed to carry one has to assert that itself.

### Why a private card is not the published case

A card is `model.json`, its metadata and its `program/*.loom` sources (`os/v3/puppy/loom-cards.pin:21-101` lists one
card's files). `loom-serve` registers it from those sources at load, and at the pin a deployment that is missing a leaf
registers as `failed` with the missing-file diagnostic rather than being satisfied from anywhere else: "no alternate
source directory or precompiled artifact can satisfy a missing leaf" (hrx-system `e2c41a72`, the
`config/loom.config:61` pin, `loom/src/loom/serve/fixture/README.md:206-210`). So a package
that carries a card carries our programs in readable form, the opposite of the house rule for our closed-source
kernels: those ship only as compiled code objects and their source stays private (`docs/hrx.md:1794-1803`,
`docs/releases.md:472-503`, `config/gpu-kernels.config:16-21`). The issue records the stronger claim for hrx-loom
0.2.0 (no precompiled-artifact input at all); this note did not re-measure it beyond the reading above, and the pin has
since moved to 0.3.0, so the option that waits for one starts by confirming it there.

ZAYA1-8B (engine#3322) is the model that motivates this, and it is measured-only on two independent grounds: its card
is the private one, and its row did not clear the decode bar the gate applies (the gate's speed rule below names
Qwen3.8-27B UD-Q4_K_XL and ZAYA1-8B). A private source removes the distribution blocker only; the entry still needs a
gate row. Its weights are public and pinned (`1bit-MONSTER/ZAYA1-8B-GGUF` in `models/pins.json`, apache-2.0), so only
the card's programs are private.

### Where a private card could be carried

1. **Private builds only.** A build made for us fetches the card into the cards directory of the box it is installed
   on; the public release and the 1bit OS payload omit it, so their copy of the entry is skipped with the existing one
   line, and nothing distributed carries the card. Cost: no engine build is "not distributed" today — the release
   machine's products are the release packages and the 1bit OS image (`docs/releases.md:472-503`), the packages carry
   no card, and the image's payload carries the public Inkling card (`os/v3/puppy/loom-cards.pin`) — so this option
   needs a build or install channel defined by its non-distribution, which is the owner's call rather than code.
   Verification: the pin is checked by us, on the machine with the credentials; nobody outside can check it.
2. **A compiled card.** A card that `loom-serve` accepts as compiled code objects, the shape the engine already
   uses for the private HIP kernels, shipped under `LICENSE-BINARY`/`NOTICE-BINARY` with the source in the private
   repository. This is the only option that puts ZAYA on Loom for a public-release user while the source stays private,
   and it needs the vendor path the pin does not have today (see above); the gate would then measure the compiled
   artifact. Verification: the pin identifies a binary, and a user's trust is the release chain, not this pin.
3. **Server-side only.** The card lives on a box we run and never in a distributed package or image. The cheapest of
   the three — no artifact, no build class — and a public-release user gets nothing from it.

None of the three changes serve's lookup; they differ in where the pin lives, who can verify it, and what a leak looks
like. In all three, a card inside a package or image that leaves us is a publication, which is why only (2) can ship
one on purpose.

### Naming the source in the table

The proposed shape is a per-entry `card_source` (`kind`, `repo`, `path`, `commit`, `tree_sha256`) beside `card`.
Three shapes, ordered by how much of the source they put in the public tree:

1. **No field.** The installer owns provenance and the entry keeps naming a card, as today; a box whose table the
   public tree must not carry runs `--loom-routes FILE` (`app/serve.cpp:1624-1634`). Zero schema change, nothing about
   the private repository in the public tree, and the private table is checked by the same
   `tools/loom_cards.py --check --routes FILE` wherever it is run. Cost: the skip line cannot name the source that is
   missing, and the public check tools see a plain entry.
2. **`card_source` per entry**, as proposed. It puts the private repository's name and the pinned commit in a public
   file, which is a disclosure decision of its own: the tree already keeps private repositories out of the public pins
   file for that reason (`tools/pin_watch.py:285-296`, `docs/model-pins.md:200-202`). It also needs both gates
   changed in one commit: `app/loom_route.h:161` refuses a route key it does not know (`:164-171`, held by
   `tests/loom_route_test.cpp:89-91`), so a table carrying `card_source` makes serve refuse the whole table at startup
   while `tools/loom_cards.py --check` passes it today (measured 2026-11-24: a copy of `config/loom-routes.json` with
   `card_source` added answers `loom-cards check: 0 errors`).
3. **A source map** beside `config/loom-cards.json` (card name -> kind + pin). Same publication question as (2),
   routing and provenance stay apart, and one file is validated instead of every entry.

### How a pin is verified

The published case pins two things: the card repository's revision (`config/loom-cards.json`, `revision`) and, at OS
staging, every file's sha256 and size (`os/v3/puppy/loom-cards.pin:21-101`, refused file by file at
`os/v3/puppy/stage-loom.sh:106-107`). A private card's pins are the same two ideas with a path:

- **`commit` + `path` already fixes the content in git's object model**: the tree at `path` in `commit` names every
  entry by mode, name and blob hash, and `scripts/fetch-private.sh:52-56` refuses a checkout whose HEAD is not the
  pinned commit. For a fetch that goes through git, that is the whole pin.
- **`tree_sha256`** as proposed is a second identifier over the installed files, and it is not git's tree hash (that
  also covers modes and object names), so its definition has to be written down exactly: sha256 over the sorted
  `<path> <sha256>` lines of the regular files under `path`, symlinks refused. The existing shape is the OS payload's
  per-file `<sha256> <size> <path>` (`loom-cards.pin:21-101`), which names the file that mismatched where a single
  digest names nothing, and `stage-loom.sh` already verifies it. A single digest earns its keep only where the files
  are copied rather than checked out (`stage-loom.sh --cards-from DIR`, `:102-103`).

What a pin of either shape does not establish, and this design must not present as established:

- that anyone reviewed the content — only that it is the content that was pinned;
- the weights: the card's own `model.json.lock` and the entry's `match.gguf_sha256` do that, and the lock binds the
  weights' repositories and revisions, not the card's own programs (`tools/ci/loom_coverage_check.py:132`);
- that a build carries the card: a missing card is a skip (`app/serve.cpp:1680-1684`), so a build that is supposed to
  carry one needs its own assertion, the shape `scripts/fetch-private.sh:56-57` already performs with `require=`;
- anything to a user of a distributed build. Only we hold the private repository, so a private pin is a
  reproducibility device for us; a user's trust in a shipped card is the release chain, which is the (2) argument.

### What the code would need

If the answer is "introduce it", the work is small and the first item is the one that fails closed:

1. `app/loom_route.h:161`: the source key in `kKeys` and its fields in `LoomRoute`, in the same change that adds the
   key to the table — otherwise serve refuses every table that carries it. The parse also has to refuse a source kind
   the build cannot use, not fall back to the published path.
2. `config/loom-routes.json` (or the private table) and this doc: the schema, with the source present only where the
   build that ships the table may carry the card.
3. `app/serve.cpp:1616-1622`, `:1708-1712`: unchanged lookup. The source drives the skip line and the fetch, not the
   load; a card installed in the directory routes exactly as it does now.
4. The installer. `scripts/fetch-private.sh:40-56` is the only fetch with credentials and a fail-closed checkout, and
   its pin shape (`repo=`, `sha=`, `path=`, `require=`) is `config/gpu-kernels.config:24-27`; it prints
   `<dest>/<path>` for a build to consume and does not install a card, so a card install adds the copy step plus the
   no-link-out property serve enforces on a card (`app/loom_route.h:368-379`). The card's own files are small — the
   one the OS payload carries is 79 files, 13,147,388 bytes (`os/v3/puppy/loom-cards.pin:22-101`), 12.2 MB of it one
   `tokenizer.json`, so ~0.9 MB of programs and metadata — and the weights stay the public GGUF.
5. `tools/loom_cards.py`: `--check` lets a route key the runtime refuses pass CI (measured above), so it gains the
   arm that refuses it, plus a shape-only or refused private source in a public tree; `--self-test` grows the arm. The
   card map cannot describe a private card as it stands either — a card's `repo` must be `loom-fun/`
   (`tools/loom_cards.py:214`) — so the `none`/`gap` row for ZAYA1-8B (`config/loom-cards.json`) needs a fifth kind or
   the card stays out of the map. `tools/ci/loom_coverage_check.py` reads card trees and stays advisory.
6. Distribution. Under (1) and (3) nothing ships: `NOTICE` is unchanged, and the new row is the negative one — a
   release/OS check that a public package or image does not carry the private card, which no row asserts today. Under
   (2) the compiled card is a private component: `NOTICE-BINARY` and `LICENSE-BINARY` beside the existing two
   (`docs/releases.md:472-503`), and `os/v3/puppy/stage-loom.sh:102-105`, which can stage only from a public URL or a
   local directory today.
7. Gate. The rule is unchanged; a row can only be produced on a machine that holds the card, and the entry's `gate`
   link is public.

### The open decision

**Which builds may carry a private Loom card.** That is the owner's, and it is RFC-gated before code: CONTRIBUTING puts
"a change to how pins move" in Discussions, with a maintainer recording accept or decline in the thread
(`CONTRIBUTING.md:51-70`). The three options are above; (2) is the only one that reaches a public-release user with the
source still private, and its first step is confirming the premise at the pinned 0.3.0. Until the owner answers,
nothing in this section is implemented: no `card_source` key, no parser change, no pin moved. If the answer is
"declined", the reason is recorded here; if "accepted", the chosen option and the list above are the implementation and
the RFC is its link.

## Lifecycle

Written against hrx-loom 0.2.0 and checked on 0.3.0, the version `config/loom.config` pins; the CLI and HTTP surface
serve uses is the same in 0.1.0, except that 0.3.0 removed `--parameter_mode=preload` (it takes `file` or `direct`),
so no route entry carries it.
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
  fallback itself is the arm that is wrong, as on MiniCPM4, engine#3340), **counting only splits that are
  not near-ties**: at the first token where a route's text leaves the CPU's, the split is a near-tie only
  when the **route's own token** is within **1.0 nat** of the CPU arm's top token there (the CPU's log-prob gap
  from its top choice to the token the route emitted). A route token further down, or not among the CPU's
  top-N at all, counts as a disagreement, however small the CPU's own top-2 margin is: a drifting kernel that
  picks the CPU's Nth choice at a small-margin split must not pass (orchestrator, 23 Nov ~20:2xZ, tightened
  ~21:1xZ after the #3420 reviews). The row reports the near-tie count beside the score, per arm, and
  `tools/logprob_agree.py` prints both (its `near-tie rule` line). The route's token is read from its text as the longest of the CPU's top-N candidates that the text continues with: the most favourable reading, so a route token outside the top-N that begins with a shorter candidate's text scores as that candidate (rare under longest-match). A prompt where one reply is a prefix of
  the other and the shorter one stopped on its token budget (`finish_reason: "length"`) is **truncated**,
  reported beside the score and not counted: neither route chose another token, one ran out of tokens.
  With 8 prompts one near-tie flip, such as an end-of-turn decision, moved the plain score by 12.5 points
  (ZAYA1-8B, engine#3158), **and**
- accuracy holds: every prompt answered (no empty-vs-empty rows), and each Loom/CPU divergence a
  near-tie on the CPU arm (the first differing token's top-2 margin is reported, so a near-tie is told
  from a confident error). Loom returns no logprobs, so KLD on the Loom arm is reported as "cannot
  tell", never filled in.

A row is valid only when Loom's prompt token count (`/v1/chat/completions/input_tokens`) equals ours
(`/apply-template` + `/tokenize` with `add_special`), with thinking on and off where the template
toggles: a card whose chat block drops BOS or misrenders thinking answers a different prompt.

**One explained difference** (orchestrator for the owner, 23 Nov): with thinking **on**, a card may pre-fill
the model's opening `think_open` (e.g. Qwen3's `<think>\n`) after the generation prompt, while the model's
own template leaves it for the model to emit. That is a pass, not a mismatch, **iff** (a) the token **ids**
match exactly apart from it: Loom's prompt ids equal ours followed by the ids of the card's `chat.think_open`,
and thinking off the ids are equal (a count that merely comes out right is not enough; loom-serve's
`input_tokens` returns a count only, so Loom's ids are the card's chat block rendered by loom-port-llm's
`chat_probe.py` and encoded with the card's `tokenizer.json`, and that many ids must equal the server's count),
**and** (b) the
greedy continuation after the prefill matches: on the gate's prompts with thinking on, the fallback's (or CPU
arbiter's) reply begins with `think_open` and Loom's reply equals the rest of it, held to the same agreement
rule as above. A prompt where the fallback's reply does **not** begin with `think_open` (the model chose not to
think) voids the row, because the forced prefill answered a different turn. Anything else is a mismatch and
voids the row. A thinking-off difference is never explained this way (the card renders a different prompt).

**Speed.** Interleaved on strixhalo, at least four rounds, each round running every arm once, one
server at a time: the fallback, raw loom-serve, and Loom through `1bit serve --loom-routes` (the route
users get), with the same prompts, a discarded warm-up per arm, and load, memory and Tctl recorded per
round. **Decode is timed on a streamed request**, (completion tokens - 1) / (last token - first token), the way
`loom-benchmark` measures every engine, so prefill and any prompt cache stay out of it; for the llama.cpp arm the
server's own `print_timing` eval rate is recorded beside it. Never take decode as the difference of two
requests: llama.cpp reuses cached prompt chunks even behind a leading nonce, and that overstated its decode by up
to 2x (engine#3158, 23 Nov). Loom routes only when it is faster through `1bit serve` (or equal on speed and better on
accuracy). **Decode counts on its own:** a Loom route that wins prefill and time to first token but
decodes slower than the fallback is not routed, because chat users feel decode (orchestrator ruling,
2026-11-23: Qwen3.8-27B UD-Q4_K_XL and ZAYA1-8B; the 27B row was later re-measured with streamed decode, passed with decode 1.10-1.4x the fallback and is routed, engine#3158). Such a model is re-gated when a Loom card decodes at
least as fast as the fallback.

**No fallback.** When the llama.cpp route cannot serve the model at all (Inkling-2-Mini on rc1, orchestrator
option A, 23 Nov ~19:3xZ, engine#3412), there is nothing to compare against: the row is accuracy only, against
the CPU arbiter. It passes when every prompt is answered, every Loom/CPU divergence is a near-tie (the CPU's
top-2 margin at the first differing token is reported), the known-answer checks pass (17 x 23, the long-context
needle), and the prompt-count precondition holds. The speed row is measured and recorded but not compared.
The table says which text was compared (answer only, or reasoning + answer with thinking on).

The CPU arbiter (llama.cpp at the engine pin, CPU only) runs on the box's CPU inside the same `box-hold`
ticket as the device arms (orchestrator 23 Nov, after ryzen's lock-up: no model loads on ryzen that the box
can do); the comparison itself runs on another machine over ssh tunnels. Each ticket (at most 45 minutes)
states its RAM need, arbiter included, and leaves at least 16 GB free. The result
tables go on the model's issue, and the entry's `gate` links them.
