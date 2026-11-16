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
| 1 | a pin's **bytes** moved, a pinned repo is **gone upstream** (`404` — a delete, a rename and a privatisation look the same), or `--pr-body` was asked for a repo that did not move | opens one bump PR per moved repo; a gone repo is reported and refused for bumping, a human decides |
| 2 | a usage error, or **no pin could be read at all** — a Hub outage or rate-limit is not a quiet day | the job fails |
| 3 | `--self-test` failed | nothing: this is CI's code, not the daily run's |
| 4 | `--bump` needs a human: the repo or a tracked file is gone upstream | skips that repo and says so |

**A move of the repository's `main` is not a move of the bytes** — since 2026-11-07 the two are counted
apart. If the HEAD moved but every file the pin tracks still has the `lfs.oid` we hold, the run reports it
under **Head moved, no tracked file changed** (with the commit titles and any new untracked `.gguf`/`.q4nx`
files), exits **0**, and opens **no PR**: there is nothing to adopt, and a daily PR per `README.md` edit is a
queue item a human has to approve — approving pin PRs reflexively is how a `TEMPLATE CHANGED` row would get
waved through. Measured on `unsloth/Qwen3.8-Flash-Next-GGUF`: three `Update README.md` commits, 40 new BF16
shards upstream, and all five pinned files unchanged. A byte change in the same run is still a real bump, and
the new upstream files are still listed, so "did they re-quant" stays visible without a PR.

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
tools/pin_watch.py --check                  # the daily comparison; writes pin-report.md, exit 0, 1 or 2 (above)
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

## What the box holds

A pin records the bytes we **mean** to serve; the daily check compares it with Hugging Face, which cannot see
the box. So a pin can move to a new upstream file while the box keeps serving the old one, and every run stays
green. That happened: `unsloth/Qwen3.8-27B-GGUF` was pinned at the UD3.1 `Qwen3.8-27B-UD-Q4_K_XL.gguf` on
2026-11-12 (`2a5f3565…`, 17,659,407,264 B) while `~/models/qwen38/UD-Q4_K_XL/` and our re-host
`1bit-MONSTER/Qwen3.8-27B-UD-Q4_K_XL-MTP` still held the earlier file (`3f227079…`, 17,559,178,144 B).

`tools/pin_holdings.py` closes both halves:

* **holdings** — [`models/holdings.json`](../models/holdings.json) maps a pinned file to its copy on the box.
  Each copy's sha256 (the git blob id for a non-LFS file, which is what the pin records there) is compared with
  the pin's oid. The hashes are cached by path, size, mtime and inode (`~/.cache/1bit/pin-holdings-sha256.json`),
  so a daily run re-reads only a file that changed, and a size mismatch is reported without reading the file.
  A path in the map is a promise that the box serves the pinned bytes: add one when a pinned file lands on the
  box, and change it in the same PR that moves a default path.
* **re-hosts** — a pin whose `used_for` says `upstream of <repo>` is compared, file by file of the same name,
  with `<repo>`'s own pin. Both oids are in `models/pins.json`, so this half needs neither the box nor the
  network (`--rehost-only`, which CI can run), and the box run reports it too. The fix is a re-upload, and
  publishing is the owner's.

Its exit codes are its own, not `pin_watch.py`'s (whose table is above):

- **0** every holding and every re-host matches its pin;
- **1** a holding differs from its pin or is missing, or a re-host holds a different file than its upstream pin;
- **2** it could not run: usage, an unreadable `pins.json`/`holdings.json`, no engine checkout where the unit
  looks, or a holding naming a pin or file `models/pins.json` does not have (a wrong map is never clean).

On the box it runs daily from `scripts/1bit-pin-holdings.{service,timer}` (05:00 UTC) under `~/lb/thermal-run.sh`,
from the engine checkout named by `ONEBIT_ENGINE` in the unit (default `%h/wt/engine-main`; set it at install to a
checkout of this repository that follows `main`, which `~/1bit-MONSTER` on strixhalo is not). A missing checkout
exits 2 with a line saying so, so "cannot run" never reads as "a pin differs". Exit 1 marks the unit failed, the
report is `~/.1bit/logs/pin-holdings-latest.md`, and every run appends one line to `~/briefs/PIN-HOLDINGS.log`
(`--log`), so the daily result outlives the journal:
`<utc> ok|drift|error (M/P checked) mapped= checked= drifted= rehost_drifted= map_errors= absent= unmapped=`.

The report also says what it did **not** check:

`Coverage: M of P pinned files checked; U not checked (A not held here, N held and missing from models/holdings.json)`

A pinned file with no holding is one of two opposite things:

- **absent**: the box does not hold it. That is expected; most pins are not on the box.
- **unmapped**: the box holds a file with that name and size, and the map does not list it. That is the blindness this
  tool exists to remove.

Only `--scan DIR` (the unit scans `~/models`) can tell them apart. It walks the tree for unmapped pinned files by
basename **and** size. Each hit is listed as `UNMAPPED:` and fails the run (exit 1). Without `--scan`, the header
says the split is unknown. On strixhalo (11-16): 9 of 192 checked, 183 absent, 0 unmapped.

```bash
tools/pin_holdings.py                       # both halves; report on stdout
tools/pin_holdings.py --rehost-only         # pins.json only (CI)
tools/pin_holdings.py --out r.md --json r.json
```

## The repos we publish (the other direction)

`--check` iterates over pins, so a repo that was never pinned is invisible to it — including one we publish
ourselves. `--check --orgs 1bit-MONSTER,bong-water-water-bong` (what the daily workflow runs) censuses both
orgs with one listing each and reports every repo that **ships a model artifact or a chat template and is not
in the table**, under *Published, not pinned*. Such a repo changes no pin, so no bump PR opens for it; it
makes the run's first line say so, and adding it is `--pin` plus a `used_for` that says what published it.
A repo of `.1bp` packs is not pinned here, and a private repo never is (next section) — both are filtered
out, so the census cannot go permanently red over something this file must not contain.

## A fallback source: ModelScope

ModelScope (modelscope.cn) mirrors many of the repos we pin, under the same id, and its public file listing
gives each file's sha256. So a mirror can stand in for Hugging Face exactly when its file carries the pinned oid,
whatever the file is called. `tools/pin_mirror.py` checks that and fetches through it. It is anonymous:
the public listing and download need no token, and the tool never sends one. The owner's ModelScope key stays
off this box, and nothing here uploads.

```bash
tools/pin_mirror.py --check                       # every pin outside our own orgs: is the mirror the pinned file?
tools/pin_mirror.py --check --repo unsloth/Qwen3.8-27B-GGUF
tools/pin_mirror.py --fetch REPO FILE --dest DIR  # download from the mirror; kept only if sha256 == the pin's oid
```

`--check` reports each pinned file as one of four states:

- `match`: the mirror may serve it.
- `DIFFERS`: other bytes under the pinned name, so do not use it.
- `missing`.
- no mirror under that id.

A non-LFS pin records a git blob id, which a sha256 cannot be compared with. For those, a small text file
(≤ 1 MiB) is downloaded and its blob id computed. `--fetch` hashes while it streams, resumes a `.part`
(the hash covers the bytes already on disk), and deletes a file that is not the pinned one rather than moving it
into place. Exit codes:

- **0** everything compared matches, or the fetch verified.
- **1** a mirror differs, or the fetch did not verify.
- **2** usage, or no listing read.

Measured 2026-11-16 against `models/pins.json` (72 pins outside our orgs):

- 63 are mirrored under the same id, and all 76 of their pinned files match, including
  `unsloth/Qwen3.8-27B-GGUF`'s UD3.1 `Qwen3.8-27B-UD-Q4_K_XL.gguf` (`2a5f3565…`).
- 9 have no mirror under that id: two RichardErkhov repos, three mradermacher repos, mys, nomic-ai, nvidia and pqnet.
- One real fetch, `unsloth/Qwen3-0.6B-GGUF` `Qwen3-0.6B-Q4_K_M.gguf` (397 MB, 36 s), verified against its pin.

## Build inputs

The model pins above are one kind of input. A build or CI run also takes inputs written inline in workflows and
scripts, and each of those is named by a commit or a digest too, so a release is a pin bump and not a scramble:

| input | pinned where | example |
|---|---|---|
| CI container | `.github/workflows/ci.yml` `image:` | `ubuntu:26.04@sha256:d66b366d…` (the tag moved three times: 2026-11-09, 2026-11-13, 2026-11-16T01:04Z) |
| CI Python packages | the `pip install` line | `numpy==2.5.4 gguf==0.19.0`, the versions CI resolved on 2026-11-09 and 2026-11-13 |
| GitHub Actions | every `uses:` | a 40-hex commit |
| package-build containers | `scripts/pack-distro.sh` | `fedora:45@sha256:…`, `archlinux:base-20261108.0.623074@sha256:…` |
| Puppy appliance | [`os/v3/puppy/pin.txt`](../os/v3/puppy/pin.txt) | the woof-CE commit, the build container digest, `vercmp.c` by commit + sha256 |

[`tools/ci/moving_ref_check.py`](../tools/ci/moving_ref_check.py) (ctest `moving_ref_check`) fails a tree that names
a branch, a tag or no version in any of these places, except the rows listed in
[`tools/ci/moving_ref_baseline.txt`](../tools/ci/moving_ref_baseline.txt). That baseline only shrinks -- it is empty
since 2026-11-16, when the three workflow rows it held were pinned instead -- and a row that no longer matches
anything fails the check too. A read that is moving on purpose carries a
`moving-ref: <why>` comment on the line or the line above it, and the check prints that reason. Two kinds of read
are marked this way: the bump bots, which read the newest release only to propose a pin, and `scripts/pack-proof.sh`,
whose point is "installs on this distro as users pull it today". A new image or version is a reviewed edit of the
line that names it.

## What is not pinned, and why

- `base_model` on a card is provenance, not a dependency: what we took from a vendor repo is the file we
  re-hosted, and that is the file pinned (`upstream_gguf`). A base model whose weights never reached us is
  not a pin.
- Private repositories are not pinned here: this repository is public. A private model is tracked in the
  lane state that owns it.
- Third-party git pins (`third_party/*`) are a different mechanism and stay where they are —
  [../tools/check_pins.py](../tools/check_pins.py) for a PR that moves one backwards, and
  [releases.md](releases.md) for the weekly "what moved" summary.
