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
# Model pins: the revision of every upstream model we depend on

The census ([registry.md](registry.md)) counts what the engine *can* run. It does not record **which
revision of which upstream model** we actually run, ship, or were built from — so a vendor can re-publish
the quants we re-host and nothing in the repository notices. That happened on 2026-11-04: unsloth
re-quantized `Qwen3.8-Flash-Next-GGUF` ("Better UD 3.1 quants", "mix n match engram"; UD-IQ4_XS went from
three shards to four) and the files we re-hosted were still the 2026-11-02 ones, at the same names.

`models/pins.json` is that record. One entry per Hugging Face repo we depend on:

```json
"unsloth/Qwen3.8-Flash-Next-GGUF": {
 "commit": "d96201194f103c906994f283f504209dc8d2a10c",
 "pinned": "2026-11-05",
 "used_for": ["upstream: our copy predates the pinned revision"],
 "files": {
  "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": {
   "oid": "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082",
   "size": 10946624, "lfs": true,
   "template_sha256": "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"}}}
```

- `commit` — the upstream revision the pin was taken at.
- `files` — the files **we hold**, not the files that happen to be there: `oid` is Hugging Face's content
  identity (`lfs.oid`, the sha256 of the bytes; the git blob id for a small text file that is not in LFS),
  `size`, and `template_sha256`, the chat template that file carries at that revision.
- `used_for` — why the repo is pinned: a `laya-first` serve class, a stick/Lemonade catalogue row, a
  published `1bit-MONSTER` re-host, or the upstream of one.

A model **baked into a stick image** is pinned from the image's own Hugging Face cache, which records the
snapshot commit the image was built from (`~/stage-c25/hf/models--<org>--<name>/snapshots/<commit>`). That
is provenance rather than an assumption: the two chat models the c25 image bakes (`Qwen3-4B-Instruct-2507`,
the FastFlowLM NPU pack) pin to exactly the commits the image's cache names, and `stabilityai/sd-turbo` —
the image model, which carries no chat template — was added with its 5,214,561,328-byte safetensors hashed
locally to confirm it is the file upstream serves at that commit.

Of the 154 model files staged on this box, the 11 whose names match a pinned file are pinned at file
level; the rest are test vocabularies (`ggml-vocab-*`), conversion intermediates, or additional quant
variants of repos that are pinned at revision level — the revision check still catches a vendor re-push for
those, which is the case that matters.

`files` records what we hold rather than only what upstream said, because a pin taken *after* a vendor
re-push would otherwise store our older copies under the newer commit and look clean. Both facts are
therefore reconciled on every check: the commit for "did upstream move", and the per-file identity for
"are the bytes we hold still the bytes it serves".

## The daily check

`tools/pin_watch.py --check` (`.github/workflows/pin-watch.yml`, daily) compares every pin with the repo's
current `main` and writes a Markdown report. It exits non-zero when anything moved, so the run is visible
even if the PR is not opened.

The exit code keeps them apart (`tools/pin_watch.py`'s header carries the same five):

| exit | meaning | what the workflow does |
|---|---|---|
| 0 | every pin was read and is at the revision we record — or the requested output (`--pr-body`, `--pin`, `--bump`) was printed | nothing |
| 1 | something moved, a pinned repo is **gone upstream** (`404` — a delete, a rename and a privatisation look the same), or `--pr-body` was asked for a repo that did not move | opens one bump PR per moved repo; a gone repo is reported and refused for bumping, a human decides |
| 2 | a usage error, or **no pin could be read at all** — a Hub outage or rate-limit is not a quiet day | the job fails |
| 3 | `--self-test` failed | nothing: this is CI's code, not the daily run's |
| 4 | `--bump` needs a human: the repo or a tracked file is gone upstream | skips that repo and says so |

The report's first line counts what was read (`N of M checked, K not checked`), and the all-clear sentence
only appears when nothing was left unread: a watcher whose "clean" and "could not check" are the same
outcome is blind exactly when it matters.

For a file that moved it also re-reads the **chat template** of the new revision — a GGUF's
`tokenizer.chat_template` over an HTTP range request of the metadata header only, never a tensor; a
container's `tokenizer_config.json` parsed and its `chat_template` **string** hashed, which is the byte
sequence the engine's serve line names, not the JSON file's own digest; a `chat_template.jinja` hashed
verbatim — and reports one of: unchanged, `TEMPLATE CHANGED`, template added, template removed, or not
checked (`--no-template`). That is the gate a re-quantization has to pass before anything may adopt it
([template-publishing.md](template-publishing.md)): weights moving under a stable file name is ordinary,
the instructions the model obeys on every request moving under it is not.

## Adopting a new revision is never automatic

The workflow opens **one PR per changed repo**, bumping only that repo's entry, with the report as the
body. Merging it says *this is the revision we point at*; it does not move the catalogue, the stick image
or a scoreboard row, each of which is its own reviewed change. So:

1. the template column must be clean (unchanged, or the change audited and recorded), and
2. the row that uses the model is re-measured, because a re-quantization is a different file even when
   its name is the same.

`--bump` refuses a pin whose tracked file vanished upstream instead of dropping it: a file that is gone is
a decision (re-take it under its new name, or drop the pin), not a diff.

A repo the token cannot see (`401`/`403`) is reported as that, not as a clean pin: the daily run absorbs the
`HF_TOKEN` secret, and a private or gated repo stays unverified until that token can read it. The report's
first line counts what was read — `145 of 147 checked, 2 not checked` — and the workflow annotates the run
with it, so "nobody could check this" never looks like "nothing moved".

## Commands

```bash
tools/pin_watch.py --check                  # the daily comparison; writes pin-report.md, exit 0/1/2/3/4 (above)
tools/pin_watch.py --check --orgs A,B       # ...and census our own orgs for a repo nobody pinned
tools/pin_watch.py --check --no-template    # same, without the template range reads
tools/pin_watch.py --check --json sweep.json   # ...and the same sweep as JSON
tools/pin_watch.py --pr-body REPO --from sweep.json  # that repo's PR body, from that sweep: no second read
tools/pin_watch.py --pr-body REPO           # without --from it repeats the sweep (one full read per repo)
tools/pin_watch.py --pin REPO [REPO...]     # print an entry for a repo at its current main (adding a pin)
tools/pin_watch.py --bump REPO              # rewrite one entry in models/pins.json (a reviewed change)
tools/pin_watch.py --self-test              # fixtures, no network
```

Adding a repo is `--pin` followed by a hand edit of `used_for`: a pin exists because something serves,
ships or was converted from it, and that reason is what makes the entry reviewable. The template digest is
per **file**, never per model — `Qwen3-4B`'s GGUF in our org and the quant the box serves carry two
different templates, both legitimate.

## The repos we publish (the other direction)

`--check` iterates over pins, so a repo that was never pinned is invisible to it — including one we publish
ourselves. `--check --orgs 1bit-MONSTER,bong-water-water-bong` (what the daily workflow runs) censuses both
orgs with one listing each and reports every repo that **ships a model artifact or a chat template and is not
in the table**, under *Published, not pinned*. Such a repo changes no pin, so no bump PR opens for it; it
makes the run's first line say so, and adding it is `--pin` plus a `used_for` that says what published it.
A repo of `.1bp` packs is not pinned here, and a private repo never is (next section) — both are filtered
out, so the census cannot go permanently red over something this file must not contain.

## What is not pinned, and why

- `base_model` on a card is provenance, not a dependency: what we took from a vendor repo is the file we
  re-hosted, and that is the file pinned (`upstream_gguf`). A base model whose weights never reached us is
  not a pin.
- Private repositories are not pinned here: this repository is public. A private model is tracked in the
  lane state that owns it.
- Third-party git pins (`third_party/*`) are a different mechanism and stay where they are —
  [../tools/check_pins.py](../tools/check_pins.py) for a PR that moves one backwards, and
  [releases.md](releases.md) for the weekly "what moved" summary.
