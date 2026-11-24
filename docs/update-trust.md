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
# Update trust: what `1bit-update` can prove, and the open freeze question (RFC draft)

`1bit-update --latest` installs an OS image only after the release key's Ed25519 signature verifies, and only the
release it asked for, strictly newer than the running one: an old, genuinely signed image re-published under a newer
tag is refused (engine [#3448](https://github.com/1bit-MONSTER/engine/issues/3448), fixed by
[#3454](https://github.com/1bit-MONSTER/engine/pull/3454)). What that check cannot prove is **freshness**: "is there a
newer release" is decided, before any signature is checked, from GitHub's release list, which is not signed. A release
host that keeps serving the current release - a compromised token with `contents: write`, a newer release deleted or
turned into a draft, or a mirror if one is ever added - makes the machine answer "up to date" and the daily notice stay
silent, for as long as the host keeps doing it.

This page records what is signed today, states the freeze threat model, and is the draft body for the RFC that decides
whether that threat is in scope. **Nothing in the "Design options" section is implemented.** The rollback half is in the
tree; the freeze half needs the owner's decision (CONTRIBUTING.md, "RFCs").

## What a release signature covers, today

- The release flow signs the exact bytes of `SHA256SUMS` with the release key: Ed25519 over
  `b"1bit-os release SHA256SUMS v1\n" + <SHA256SUMS bytes>` (`tools/release_sign.py:25-28`, `:55`), published as
  `SHA256SUMS.sig` (`:44-46`). The public key is baked into the image at `/etc/1bit/release-pub`
  (`os/v3/puppy/cut.sh`, `RELEASE_PUB`).
- `verify-image` reads `SHA256SUMS` **once** and checks both the signature and the image's sha256 against those same
  bytes (`tools/release_sign.py:82-93`, `:163-182`), so a file swapped between the two checks cannot pass; a signature
  that brings its own public key is refused (`:134-145`). The signed message is a few hundred bytes, never the image.
- The updater requires both files before trusting any digest, and deletes the download on a refusal
  (`os/v3/puppy/embed/usr/local/bin/1bit-update:440-455`, `:396-405`).

What that proves: these bytes are the key holder's, and the image is the one the signed list names. It does **not** prove
that the release named is the newest one the key holder signed, and it does not bind the release-list tag to the signed
bytes.

## How `--latest` decides, in order

| step | what happens | trust |
|---|---|---|
| 1 | fetch GitHub's release list over HTTPS (`1bit-update:419-425`) | HTTPS transport; the data is not signed |
| 2 | pick the channel's newest tag by version, `vYYYY.WW[-rcN]`, skipping drafts (and pre-releases on stable); `newer=` / `ahead=` from that version comparison (`:257-302`) | **unsigned list** |
| 3 | if it is not newer: say "up to date" and exit 0 (`:430-438`); no signature has been checked yet | unsigned |
| 4 | download `SHA256SUMS`, `SHA256SUMS.sig` and the image over HTTPS (`:448-455`) | transport |
| 5 | `release_sign.py verify-image` with the baked key; a refusal removes the image (`:453-455`) | **signed** |
| 6 | decompress (`:459`), verify the image's own set against the `SHA256SUMS` inside it (`:792-810`) | signed list covers it |
| 7 | read the tag **inside** the verified image's `puppy_*.sfs` and refuse unless it equals the tag asked for, is strictly newer than the running release and is not an `-rc` on stable (`:812-819`, `:471-493`) | **signed bytes** |
| 8 | write the other slot | - |

**What #3454 closed: rollback.** An attacker without the release key who re-publishes an old signed image under a newer
tag gets past step 5, but step 7 reads the release identity from the verified image - which sits inside the SFS covered
by the signed `SHA256SUMS` - and refuses it. Deliberate downgrades stay possible through `1bit-update <image>` and
`--rollback`, which do not set `expect_tag` (`:462`, `:483`).

**What remains open: freeze.** Step 2 and step 3 use an unsigned answer, so a host that can control what the release
list says can always present the running release as the newest one. No signed, expiring statement of "the newest release
on this channel" exists in the tree or among the release assets, so the box cannot tell "none was signed" from "I am
being shown a stale list". This is the half the RFC below decides.

## Threat model

Assets: the OS the machine boots (kernel, initrd, SFS), and the decision to replace it.

| adversary | can do | in scope? |
|---|---|---|
| **release-host tamper**, no key: a compromised token with `contents: write`, a compromised org/runner account, or a mirror if one is added | publish/delete releases, change assets, flip `draft`/`prerelease`; cannot produce a signature | **yes** - rollback closed by #3454, freeze open (this RFC) |
| network attacker (no repo access) | intercept or block traffic | yes, and covered: HTTPS only, the signature over the bytes, the key baked in the image |
| release **key** holder without the release process | sign anything, including a stale or forged "latest" statement | no - the key is the trust anchor; a signing-key compromise is a key rotation problem, not this design |
| GitHub itself | serve a different release list or assets | no - the list is fetched from GitHub; the design narrows this to "can withhold, cannot forge" |
| local unprivileged user | plant paths or symlinks the updater follows | no - private per-run work dir and a root-only `/data/updates` (`1bit-update:343-387`), #3454 |

Security properties, and where they stand:

| property | today |
|---|---|
| **P1 integrity** - the installed bytes are the key holder's | yes (#3442, #3447, #3454) |
| **P2 no relabelled rollback** - an old signed image under a new tag is refused | yes (#3454, step 7) |
| **P3 freshness** - the box can tell "no newer release was signed" from "the host hides one" | **no** (this RFC) |

Out of scope for this page: `apt` package updates (they land inside the running set, not in a slot), reporting a
vulnerability (SECURITY.md), and the weekly packaging workflows (the repo's
[`.oss-scanner/threat_model.md`](../.oss-scanner/threat_model.md) places those outside its threat model, which is why
this one is written down here).

## Design options

The question is only how a box learns "the newest release the key holder signed is X" in a way a host cannot forge;
every option below assumes P1/P2 from #3454 stay as they are.

### A. A signed, expiring "latest per channel" statement (recommended)

Publish one small text file per channel at a **fixed URL that does not depend on the release list**, signed with the
existing release key and the existing `release_sign.py` message format (a new domain prefix, e.g.
`1bit-os release latest v1\n`, so a latest statement can never be read as a `SHA256SUMS`):

```
1bit-latest 1
channel stable
tag v2026.50
version 2026.50
expires 2026-12-08T00:00:00Z
sums_sha256 <sha256 of that release's SHA256SUMS>
```

- `--check` and `--latest` fetch the pointer, verify it with the baked key, and use its `tag`/`version` as the
  newest release; the pointer's `sums_sha256` binds it to the exact signed `SHA256SUMS` that will be installed.
- The box refuses to *install* when the pointer is missing, malformed, expired, or lower than the highest pointer it has
  already seen (a one-line replay guard under `/data/1bit`, like the notice's stamp); `--check` reports "cannot prove
  freshness" instead of "up to date". An offline machine is in the same state and can still install explicitly fetched
  bytes through `1bit-update <image>`, as today.
- The fixed URL is the one real design choice: a dedicated branch/file in this repository served over HTTPS, a
  fixed-name release (e.g. `latest`) updated in place, or a file on 1bit.gg. All three rest on the same two anchors:
  HTTPS to the host, and the signature from the key baked into the running image. A host can withhold a pointer, but a
  withheld pointer goes stale within its expiry, and the replay guard stops an old one from being re-served after a
  newer one was seen.
- Trust assumptions: the release key is not compromised; the pointer URL is HTTPS and stable; the box's clock is
  roughly right (document what happens on an obviously wrong clock - report, never silently accept).
- Cost: no new dependency (Ed25519 in the stdlib, `tools/release_sign.py` reused); one signing step beside the one
  that writes `SHA256SUMS.sig` on the release machine, which is not in this tree (this tree ships the tool and bakes
  `/etc/1bit/release-pub`, `os/v3/puppy/cut.sh`); a new `--latest`/pointer test class; one state file on the box.

### B. TUF-style timestamp/snapshot metadata

The full model: a signed `timestamp` role that expires and lists a `snapshot`, which lists `targets`, with roles,
threshold keys, rotation and a client that stores the last version it saw. It is the right shape for a project with
several publishers and key roles; 1bit has one publisher and one release key, so the delegation and threshold machinery
is cost without a threat it removes here. It also needs a TUF client (a new pinned dependency and its NOTICE entries,
or a hand-rolled subset that is TUF-shaped without the parts we do not use - which is essentially option A with more
words). Keep it as the growth path if publishing ever splits into roles or keys.

### C. Signed tag -> commit binding

Verify that the release tag is a **signed** git tag and that its commit is on `main`, and take the highest such tag as
newest. What it adds: the tag name itself becomes a maintainer statement, so the tag→commit relation is authenticated as
well as the image. What it does not add: freshness - a signed-tag list can be withheld exactly like the release list, so
P3 stays unsolved unless a signed *list* is also fetched from a fixed place (i.e. option A again). Cost: a git-signing
key in the image (a second key type), a verification path on the box, and a workflow that signs tags. Not worth it
while the release identity is already inside the signed image (#3454).

### D. Bind tag and channel into the signed `SHA256SUMS`

Sign a file whose message includes `tag`/`channel` lines next to the digests. Cheap, and it makes a *mislabelled*
release detectably mislabelled even without decompressing the image. It does not prevent freeze: an old signed
`SHA256SUMS` is still a valid signed statement for its own tag. If option A lands, D comes for free by putting
`tag`/`channel`/`sums_sha256` in the pointer, so no `SHA256SUMS` format change (and no change for images that
predate it) is needed.

### E. Accept the risk

If a compromised release host is judged out of the threat model, the correct result is to write that decision into
`docs/os-v3.md` beside the trust order, keep the honest wording the updater now prints (an "up to date" answer says it
is the unsigned list's word), and close #3448 with that reasoning. That is a legitimate owner decision; the cost of A is
small but not zero, and the decision is the RFC's first question.

## Recommendation

**A, now.** It closes P3 for the price of one signed text file, reuses the key and the signing tool the release already
has, adds no dependency and no new trust anchor, and degrades truthfully when the pointer is unavailable (the tree
already distinguishes "no network" from "an answer"). **D** folds into A's pointer and should not be shipped on its own.
**B** and **C** are not worth their cost at one publisher; **E** is the fallback if the owner rules freeze out of scope.

## How it will be checked (CONTRIBUTING rule 1)

All of it runs in CI; none of it needs a device or a board.

- `tools/release_sign.py --self-test` (`release_sign_self`): the new message domain signs and verifies, a pointer
  read as a `SHA256SUMS` (and the reverse) is refused, a changed pointer/expiry is refused, a pointer that carries
  another key is refused.
- The pointer parser, if it lives in `1bit-update`, is pure and testable the way `update_identity_check` is
  (`tests/update_identity_test.sh`): valid, signature-invalid, wrong channel, expired, lower-than-stored, tag not
  matching the release list's tag, `sums_sha256` not the downloaded `SHA256SUMS`.
- `tests/update_latest_test.sh` (`update_latest`): a signed pointer is accepted; a missing/expired/stale/replayed
  pointer is refused **before** the download; the install path still runs #3454's identity check afterwards.
- `tests/update_check_test.sh` (`update_check`): each refusal prints "cannot prove freshness" rather than
  "up to date"; the daily notice never writes an update line from an unproven answer
  (`tests/update_notice_test.sh`).
- A board row in `os/mini/release-check.sh` (its `--self-test` runs in CI, `release_check_selftest`) that the image
  can verify the pointer the release publishes, so a cut that forgets to publish or sign it fails on the stick.
- Every new arm carries the pre-change control (the arm fails on the tree before the change), as this tree's other
  security fixes do.

## Dependencies, licences and security (NOTICE)

- **A**: no new dependency. It reuses `tools/release_sign.py` (Python 3 standard library Ed25519, Apache-2.0, already
  in NOTICE and already baked into the image as `/usr/lib/1bit/release_sign.py`). No new workflow permission: the
  release machine already has `contents: write` and the release key; the step writes one more signed file.
  `.oss-scanner/threat_model.md` gains a line only if the owner wants the update path in the repo scanner's model.
- **B**: `python-tuf` (or a hand-rolled equivalent) would be a new pinned dependency and a NOTICE entry, plus a
  licence review; that is the reason it is not the recommendation.
- **C**: a git signing key in the image; no new package, but a second key type and its rotation story.
- **E**: none.

## Open questions (the owner's decision)

1. **Is a compromised release host (freeze) in the threat model?** If not: record it in `docs/os-v3.md`, keep the
   honest wording, close #3448. If yes: accept A or B.
2. If A: where does the pointer live (repository branch / fixed-name release / 1bit.gg), and what is its TTL?
   One week plus a grace window keeps an offline machine useful; a long TTL leaves a long freeze window.
3. Fail closed or warn: should `--latest` refuse to install when freshness cannot be proven, or install what the list
   names with a warning? This page recommends refusing to install and reporting on `--check`.
4. Is expiry plus a local highest-seen-version stamp enough (this page's assumption), or must the pointer's version
   rules be TUF's (option B)?
5. Should the pointer bind the `(channel, tag, SHA256SUMS sha256)` triple (option D folded in), and does the tag in the
   pointer have to equal the release list's tag, or only be newer?
6. Who signs the pointer: the existing release key (recommended) or a separate role/key, and how is it rotated?
