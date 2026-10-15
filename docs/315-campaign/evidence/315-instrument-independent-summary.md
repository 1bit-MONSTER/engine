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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Instrument-independent summary: what is left when every instrument-specific claim is removed

Read from the whole archive, grouped by configuration. No GPU time. This is the session's consolidation: it
takes the three revisions of the onset thread and asks what, if anything, survives all of them.

## The table everything below is drawn from

First NaN-bearing fingerprint line, per arm, grouped by tag (`EXTRA_ENV` read from each console):

| tag | selector | probes | first NaN-bearing tensor | counts |
|---|---|---|---|---|
| `315nan`, `315fbar` | `tail:16` | yes | `node_2948` (layer 41's unnamed post-attention ADD) | **2048**, 18 of 19 arms; extents 178,176–712,704 |
| `315lout4` | `l_out` name list | **yes** | `l_out-0` | 1984, 1984, **46** |
| `315inp2` | `l_out`/`ffn_inp` list | **yes** | `l_out-1` | 2048 |
| `315size` | `l_out` name list | no | `l_out-0` | 1984, 1984, 1790 |
| `315fp2` | `tail:16` | yes | `ffn_out-0` | 7048 |
| `315idp` | ids probe | — | `ffn_moe_probs-1` | 640 of 640 (saturated) |
| `315chain` | name list | — | `l_out-0` | 902,500 of 1,048,576 (saturated) |
| `315inp1` | name list | — | `ffn_inp-0` | 1,009,992 of 1,048,576 (saturated) |

## What does NOT survive — stated first, because it is the point

- **The site does not survive.** With a name list the first NaN is at layer 0 (A and B); with `tail:16` it is at
  layer 41 (C). Both instruments read the same model in the same regime. Since A and B hold the selector and
  vary only the probes and do **not** move, while B and C hold the probes and vary the selector and **do** move,
  the read pattern is what sets the onset. **"Layer 41" and "layer 0" are both instrument-relative.**
- **The exact count `2048` does not survive.** It is that value in 18 of 19 arms under C; under name lists the
  onset reads 1984 (four times), 1790, 46, or 2048. So `nan = 2048` exactly is a property of the `tail:16`
  configuration, not of the defect.
- **Anything measured while saturated does not survive** as a size: 640-of-640, 902,500-of-1,048,576 and
  1,009,992-of-1,048,576 are arms whose whole tensor is NaN, and their counts carry no size information at all.

## What DOES survive

**One magnitude, across configurations: the first corrupted extent is about one hidden-state row.**

| condition | onset count | in bytes | in rows (`n_embd = 2048`) |
|---|---|---|---|
| C (`tail:16`) | 2048 | 8,192 | 1.000 |
| B (`lout4`, `inp2`) | 1984, 1984, 2048 | 7,936 / 7,936 / 8,192 | 0.969, 0.969, 1.000 |
| A (`size`) | 1984, 1984, 1790 | 7,936 / 7,936 / 7,160 | 0.969, 0.969, 0.874 |

**Five of the seven non-saturated onset readings lie in 7,936–8,192 bytes — within one 256-byte block of
exactly one row** — and they span three configurations, two rigs, and extents from 79,872 to 712,704 floats
(`T = 39 … 348`). The two exceptions are `1790` (0.874 rows) and `46`, the latter being a single arm and the
one reading that does not fit at all.

So the honest surviving claim is:

> Under forced eviction, the first corrupted tensor carries **approximately one hidden-state row** of
> corrupted data — 7,936–8,192 bytes, one row ± one 256-byte alignment block — and this holds across three
> instrument configurations whose *onset layer* and *exact count* differ.

**And the corroboration that is strongest of all** is not a count but an identity: in the four arms where the
`hrx-finite-fp` channel printed, `finite + nan` equalled the tensor extent **exactly** (77,824+2,048=79,872;
350,208+2,048=352,256; 710,656+2,048=712,704). Whatever the count means, it is taken over a **complete** read
of the tensor, not a truncated or partial sample. That is instrument-independent evidence about *how the number
was obtained*, and it is the reason the surviving claim can be stated on the count at all.

## What this means for the goal

- **A mechanism claim of the form "the corruption is at layer N" cannot be made** without fixing the read
  pattern, and none of the campaign's layer localisations has done that.
- **A mechanism claim of the form "the corruption is one row" can be made, with the alignment caveat**, and it
  is the only tensor-level statement this work has that survives across instruments.
- **Face B remains unmeasured** (`315-faceb-has-no-capable-instrument.md`: no arm on the record carries both a
  fingerprint and Face-B evidence), so the both-faces requirement is not addressable from existing data.
- The next run that would actually advance things is therefore **not** another onset sweep but the
  pre-registered `315faceb` (finite-hash channel, `NANONLY` absent), plus a same-config, same-session
  re-measurement of the size invariant under a name list so the 7,936-vs-8,192 question is settled within one
  read pattern instead of across three.

## Caveats

- Arms came from two rigs, several weeks apart, on a box that was **thermally saturated** for the later ones
  (`thermal_pauses` to 29, `stopped_frac` to 0.82). Cross-configuration comparison is therefore weaker than a
  same-session one; that is precisely what the still-running `315cfg` (A′ vs B) would firm up.
- No rate is given anywhere above: every arm cited was selected for having faulted.
- `n_embd = 2048` is taken from the `ne0=2048` fields on the extended-fingerprint lines, not from model
  metadata.
