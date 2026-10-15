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

# Face B has never been measured with an instrument that can see it — and the instrument already exists

Read from the archived arm logs while `315size` runs. No GPU time.

## The one-row count is a complete read, not a truncation

`node_2948` also appears in the **`hrx-finite-fp`** channel, which reports `finite=<count>` (the non-NaN
values) alongside `hash_finite=` and `nan=`. In all four arms that carry it:

| arm | `finite` | `nan` | `finite + nan` | extent |
|---|---|---|---|---|
| `315fbar-p1c` | 77,824 | 2,048 | **79,872** | 79,872 |
| `315fbar-p2c` | 350,208 | 2,048 | **352,256** | 352,256 |
| `315fbar-p2l` | 710,656 | 2,048 | **712,704** | 712,704 |
| `315fbar-p4c` | 77,824 | 2,048 | **79,872** | 79,872 |

**The two counts partition the tensor extent exactly, in every arm.** So `nan = 2048` is a count over a
**complete read of the whole tensor** — the strongest available refutation of the reading being a truncated,
partial, or aliased sample. Combined with the 18-of-19 arms result, the one-row finding is now supported by
both cross-arm invariance and an internal accounting identity.

## `hash_finite` differs between arms of identical shape — the Face-B signal

`315fbar-p1c` and `315fbar-p4c` have the **same** extent (79,872, T=39) and the **same** finite count
(77,824), and **different** hashes:

    p1c  hash_finite=f5a462171c873f22
    p4c  hash_finite=6966825d04b7208b

Two arms of identical shape whose *finite* values differ is exactly the observable a `finite but wrong`
corruption produces. **This is a working detector for Face B's class of value**, already recorded, with no new
code.

## Why Face B has no evidence: the instrument is blind to it by construction

Every Face-B arm on this record carries **zero** fingerprint lines:

| arm | Face-B markers | `hrx-tensor-fp` lines | `hrx-finite-fp` lines |
|---|---|---|---|
| `315inp2-r1e` | 2 | **0** | **0** |
| `315inp2-r2e` | 2 | **0** | **0** |
| `315inp2-r3e` | 0 | 7 | 0 |
| `315lout4-r1e` | 0 | 20 | 0 |

The reason is structural, not accidental: the campaign's fingerprint is driven with
`GGML_HRX_TENSOR_FP_NANONLY=1`, which prints a line **only if the tensor contains NaN**. Face B is defined as
**finite but wrong**, so a Face-B arm provably produces no lines. Face A got a mechanism story and Face B did
not, because **the instrument can only see Face A** — and the earlier sessions' notes that "Face B was never
observed in this session" were reading a detector that could not have observed it.

`hrx-finite-fp` is not NaN-gated **at source**: it is emitted immediately after the `hrx-tensor-fp` line, under
the same `GGML_HRX_TENSOR_FP` selector, and the **only** thing that suppresses it is the same

    ggml-hrx.cpp:728-730
        if (fp_nan_only && nan == 0) { continue; }

`fp_nan_only` comes from `GGML_HRX_TENSOR_FP_NANONLY`. So there is **no separate `GGML_HRX_FINITE_FP`
variable** — an earlier draft of this file invented one; the correct enabling condition is the opposite of
what the campaign has been doing. To see finite hashes in a Face-B arm, `NANONLY` must be **absent**, which
means every watched node prints two lines per call, so `GGML_HRX_TENSOR_FP_CALLS` is what bounds the volume.
The channel was live in 6 arms (`315fbar-p1c p2c p2l p4c`, `315fp2-p2l`, `315idp-p4l`) — all of them Face-A or
clean, because every one of them ran with `NANONLY` **and** still had NaN to print.

## The missing run, stated as a pre-registration

    BIN=~/wt/315fp-build/bin
    EXTRA_ENV="GGML_HRX_TENSOR_FP=<the 15-name l_out list from 0ed67c2> GGML_HRX_TENSOR_FP_CALLS=6"
    # NANONLY deliberately ABSENT - it is precisely what suppresses the finite line (ggml-hrx.cpp:728)
    ~/wt/315-disc.sh 8 315faceb

**Readout, fixed in advance:**

| observation | verdict |
|---|---|
| a Face-B arm whose `hash_finite` for a given node differs from the clean arms', with `nan=0` | **Face B localised to that node** — the finite-but-wrong value, and the first such node names its site |
| every Face-B arm's `hash_finite` matches the clean arms' at every watched node | the corruption is **outside** the watched set (head, sampler, tokenizer) or the run produced no Face-B arm — reported as **not-shown**, never as a negative |
| no Face-B arm results | **INVALID for this question**, reported as such |
| `EXTRA_ENV` produces no `hrx-finite-fp` lines at all | `NANONLY` is still set, or the selector matched nothing — check both against the source before reading the empty output as absence |

**Confounds.** `hash_finite` is content-sensitive, so **two clean arms may already differ** wherever sampling
or nondeterminism enters — the clean baseline must therefore be several arms, not one, and a divergence is
only a finding if the clean arms agree with each other. This is stated because the whole test collapses if it
is skipped.

**What this does not do:** it does not show the finite-value channel is wired to the same tensors as the NaN
channel (verify by reading the source, not by assuming), it gives no rate, and it does not itself link Face B
to the layer-41 site — it gives Face B the *instrument* that the layer-41 work already had.

## Evidence

    grep -ahE "hrx-finite-fp.*name=node_2948 " ~/wt/rt-det/315fbar-*/server.log
    for d in ~/wt/rt-det/315inp2-r1e* ~/wt/rt-det/315inp2-r2e*; do grep -ac "hrx-tensor-fp" "$d/server.log"; done

---

## Confirmed across the whole archive: no arm has BOTH a fingerprint and Face-B evidence

The claim above was established from the Face-B arms having zero fingerprint lines. Swept over **every arm on
the record**:

| search | result |
|---|---|
| arms with `hrx-tensor-fp` lines **and** Face-B markers | **none** — zero arms |
| arms with Face-B markers at all | `315ab1a-r2e/r2q/r3e/r4e`, `315ab1b2-r1e/r2e`, `315accept-p10l/p1c`, … |

**Not one arm anywhere carries both.** The reason is structural and now demonstrated rather than argued: the
fingerprint's only gate is `if (fp_nan_only && nan == 0) continue;`, so an arm with fingerprint lines has NaN,
and an arm with Face B has no NaN. The two faces are therefore **mutually exclusive in the instrument's
output** — the campaign's own logging cannot show both, ever, under any configuration that sets `NANONLY`.

This is stronger than "Face B was never observed in this session": it is **"Face B has never been recorded
alongside any fingerprint, in any arm on this record"**, so the goal's requirement of one mechanism covering
both faces is **not addressable from existing data at all** — not merely under-supported. Every statement about
Face B in the campaign has been made from arm-level verdicts (`http=1`, `s500=1`, `unparsed=2`, `utf8=1`) with
no tensor-level measurement behind it.

The path is unchanged and is the `315faceb` run pre-registered above: the same selector with **`NANONLY`
absent**, so the `hrx-finite-fp` channel prints `finite=` / `hash_finite=` / `nan=` for every watched node, and
a Face-B arm's divergence shows up as a `hash_finite` that differs from several agreeing clean arms.

**Reproducing the sweep:**

    for d in ~/wt/rt-det/*/; do
      fp=$(grep -ac hrx-tensor-fp "$d/server.log"); fb=$(grep -aciE 'invalid utf|Content-only' "$d/server.log")
      [ "${fp:-0}" -gt 0 ] && [ "${fb:-0}" -gt 0 ] && echo "$d"
    done

### Design check: dropping `NANONLY` costs no extra perturbation, only log volume

A reasonable objection to the plan above is that reading *every* watched tensor would perturb the run more than
the NaN-only mode, and the record already holds that the fingerprint is itself an inducer — so the instrument
that can see Face B might prevent it. Read at source, the objection does not apply:

    ggml-hrx.cpp:701   ggml_backend_tensor_get(node, host.data(), 0, bytes);   <- the read, always
    ggml-hrx.cpp:722-727  ... isnan(f[j]) ... nan++ ...                       <- classified after
    ggml-hrx.cpp:728   if (fp_nan_only && nan == 0) { continue; }             <- only the PRINTING is gated

The device→host read and the NaN/hash classification happen **before** the `NANONLY` test, so `NANONLY` selects
what is *printed*, not what is *read*. Turning it off adds output lines and nothing else.

The dose variable the record actually identifies is therefore the **number of watched tensors** (16 for
`tail:16`, 64 for the amplifier, 15 for the name list) — and `315faceb` holds that at the same 15 as the
baseline config, which is what makes it a controlled comparison rather than a new induced regime. Bind the
output volume with `GGML_HRX_TENSOR_FP_CALLS` rather than by narrowing the watched set, so the read pattern
stays matched to the baseline.
