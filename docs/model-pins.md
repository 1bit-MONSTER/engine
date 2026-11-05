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

`files` records what we hold rather than only what upstream said, because a pin taken *after* a vendor
re-push would otherwise store our older copies under the newer commit and look clean. Both facts are
therefore reconciled on every check: the commit for "did upstream move", and the per-file identity for
"are the bytes we hold still the bytes it serves".

## The daily check

`tools/pin_watch.py --check` (`.github/workflows/pin-watch.yml`, daily) compares every pin with the repo's
current `main` and writes a Markdown report. It exits non-zero when anything moved, so the run is visible
even if the PR is not opened.

For a file that moved it also re-reads the **chat template** of the new revision — a GGUF's
`tokenizer.chat_template` over an HTTP range request of the metadata header only, never a tensor — and
reports one of: unchanged, `TEMPLATE CHANGED`, template added, template removed, or not checked
(`--no-template`). That is the gate a re-quantization has to pass before anything may adopt it
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

## Commands

```bash
tools/pin_watch.py --check                  # the daily comparison; writes pin-report.md, exit 1 if moved
tools/pin_watch.py --check --no-template    # same, without the template range reads
tools/pin_watch.py --pr-body REPO           # the bump PR body for one moved repo
tools/pin_watch.py --pin REPO [REPO...]     # print an entry for a repo at its current main (adding a pin)
tools/pin_watch.py --bump REPO              # rewrite one entry in models/pins.json (a reviewed change)
tools/pin_watch.py --self-test              # fixtures, no network
```

Adding a repo is `--pin` followed by a hand edit of `used_for`: a pin exists because something serves,
ships or was converted from it, and that reason is what makes the entry reviewable. The template digest is
per **file**, never per model — `Qwen3-4B`'s GGUF in our org and the quant the box serves carry two
different templates, both legitimate.

## What is not pinned, and why

- `base_model` on a card is provenance, not a dependency: what we took from a vendor repo is the file we
  re-hosted, and that is the file pinned (`upstream_gguf`). A base model whose weights never reached us is
  not a pin.
- Private repositories are not pinned here: this repository is public. A private model is tracked in the
  lane state that owns it.
- Third-party git pins (`third_party/*`) are a different mechanism and stay where they are —
  [../tools/check_pins.py](../tools/check_pins.py) for a PR that moves one backwards, and
  [releases.md](releases.md) for the weekly "what moved" summary.
