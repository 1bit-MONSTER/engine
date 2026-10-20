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

# The primitive fault is an entirely-NaN MoE ROUTER output for a single token

Reading the archive's **smallest** NaN clusters — the least-propagated observations, therefore the closest to
the seed. No GPU.

## The minimal observations, ranked

    nan=4    ne0=1   of=4     48 records   ffn_moe_weights-46     <- a FOUR-element tensor, ALL NaN
    nan=40   ne0=1   of=40    45 records   ffn_moe_weights-1
    nan=64   ne0=64  of=64    26 records   ffn_moe_probs-46       <- softmax over 64 experts, ALL NaN
    nan=8    ne0=1   of=40     4 records   ffn_moe_weights-4
    nan=192  ne0=2048 of=2048  1 record    node_3305              <- the ONLY sub-tensor region found

**Every one of the smallest NaN observations is an MoE routing tensor, and every one is fully NaN** — down to a
**4-element** tensor (`ffn_moe_weights` = the top-4 expert weights) and a **64-element** one
(`ffn_moe_probs` = the softmax over the 64 experts). There is no smaller NaN region anywhere in the archive.

## What that means, mechanically

`ffn_moe_probs` is the router softmax and `ffn_moe_weights` is the top-k weight list. Their inputs are
`ffn_moe_gate` (the gate logits = `x @ W_gate`) and `ffn_moe_argsort` (the top-k selection). **A softmax
producing all-NaN over 64 experts means its input was all `-inf` (the 0/0 case) or already contained NaN.** And
`ffn_moe_gate-*` is observed fully NaN at many layers. So the chain is:

    hidden state x  ->  ffn_moe_gate (x @ W_gate)  ->  softmax  ->  ffn_moe_probs (ALL NaN)
                   ->  top-k + weights                                        ->  ffn_moe_weights (ALL NaN)
                   ->  MUL_MAT_ID over expert weights  ->  ffn_moe_weighted / ffn_moe_out / ffn_moe_down
                   ->  residual  ->  every downstream tensor

**So the fault's most primitive observable manifestation is: for a single token, the MoE router output is
entirely NaN** — which requires the hidden state fed to that router to already carry NaN, since the router is a
matmul followed by a softmax over the same input vector.

## The one sub-tensor region in the archive

`node_3305`: `nan=192`, `of=2048`, `ne0=2048`, **`first=1856`** — and `1856 + 192 = 2048` **exactly**. So the NaN
is the **last 192 elements** of a single 2048-element row, i.e. **granules 29, 30 and 31 of 32**, running to the
end of the row. `node_3305 = 71×46 + 39`, i.e. **layer 46** (the final layer), just past the residual ADD.

That is the only place on the record where a NaN region is smaller than its whole tensor and the region is
measurable. Everywhere else the NaN has already filled the tensor.

## What this does and does not establish

**Does:** the seed is in the **hidden state that feeds an MoE router**, and the smallest observable is one
token's routing vector. Nothing upstream of that is observed: the graph's **input embeddings are never
fingerprinted** — across 567 distinct tensor names the record contains no `inp_*`, no embedding, no token
tensor. The earliest tensor ever observed is `l_out-0` (layer 0's output).

**Does not:** localize it. Because of the aliasing defect
(`315-fingerprint-does-not-measure-per-node-buffer-aliasing.md`), a NaN in a shared residual buffer appears
under many names, and `ffn_shexp-41` ≡ `node_2948` is one observation reported twice. So "the router input is
NaN" is established; **which** tensor first wrote NaN is not.

## And the numbers do not support a single clean unit

The archive-wide NaN-count histogram is dominated by whole-tensor NaN (`1048576` 11.0 %, `610304` 11.0 %,
`1572928` 3.5 %) — the propagated end state. `2048` is 3rd at 5.1 %, and it is the most common value for
`ne0 ∈ {768, 5120, 576}`. **44 % of NaN counts are multiples of 2048 and 56 % are not.** The `fbar` arms' first
eight buffers *are* all exactly `nan=2048` across widths 64…5120, which is striking, but **it does not
generalise** and should be recorded as a property of those four arms, not of the fault.

**So no single unit — row, wave, granule, or 8192-byte region — is supported archive-wide**, and the earlier
claims of that shape in this directory are superseded.

## The corrected shape of the problem

1. The **input path is unobserved** (no embedding tensor has ever been fingerprinted).
2. The **router is the seed's first observable victim**, per token, entirely.
3. **Per-node localization is structurally unavailable** (aliasing), so the record cannot walk the fault back
   from the router to its origin.
4. Face A = NaN logits (unambiguous, `result_output` fully NaN, unique buffer). Face B = finite logits, ~half
   benign prose, disjoint from Face A.

**The record already contains a finding pointing at item 1** — PR #90, *"graph inputs are never written back to
host"*, shipped as an ancestor of the pin. Combined with an input path that is never instrumented, and a
mechanism chain that the objective itself states as *"graph input published to host by deferred non-coherent
staging writeback → re-uploaded to device → device computes from stale input"*, **the highest-value next
instrument is the input embedding tensor, and it has never been observed.**

**No mechanism is claimed. What is claimed is where the seed must be, what its first observable victim is, that
the only two things upstream of it have never been instrumented, and that the archive cannot take the next step
because of a fourth instrument defect.**
