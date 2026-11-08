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
# The Zyphra family: Zamba2, Zamba v1 and BlackMamba

The three Zyphra architectures the engine carries — `zamba2`, `zamba` (v1, with its MoE) and `blackmamba`
(Mamba-1 with alternating MoE) — what their files weigh at the current pin, which quantization to publish for
each, and the two things that will silently go wrong if you convert or quantize them with an older tree.

The architecture work is in [models.md](models.md) rows and in the fork; this page is the file-level record.
Engine issue: **#1036** (the Zamba2 sharing work) and **#991** (HRX placement classes).

## File sizes: a Zamba2 GGUF is now the size of its checkpoint

Zamba2 checkpoints store the transformer block **once per shared block** (two of them at 2.7B and 7B) and hold
each hybrid layer's MLP difference in a rank-128 adapter pair. Converting used to expand both — the block once
per hybrid layer, and each layer's adapter merged into a full FFN matrix — so a 7B BF16 GGUF came out at
**20.45 GiB** against a 14.82 GB checkpoint. Two changes fixed that, and both are in the pin:

| change | fork PR | what it stops |
|---|---|---|
| shared block stored once per block, aliased per layer | [#203](https://github.com/1bit-MONSTER/llama.cpp/pull/203) | 13 copies of a set that has 2 distinct values |
| per-layer MLP adapter stored as a low-rank pair, applied in the graph as `base + B*(A*x)` | [#218](https://github.com/1bit-MONSTER/llama.cpp/pull/218) | 13 copies of a 102.76 M-param matrix (205 MB each at bf16) where 2 plus a 4.13 M pair per layer do |

| model | merged form (old) | adapter form (current) | change |
|---|---|---|---|
| Zamba2-2.7B-instruct | 6,009,822,848 B (5.60 GiB), 520 tensors | **5,328,903,360 B (4.96 GiB)**, 531 tensors | **−680,919,488 B (−0.63 GiB)** |
| Zamba2-7B-Instruct | 16,875,503,904 B (15.72 GiB), 771 tensors | **14,722,123,360 B (13.71 GiB)**, 786 tensors | **−2,153,380,544 B (−2.005 GiB)** |

**Both forms load from one binary.** A GGUF written before these changes keeps taking the merged path — the
loader requires `ffn_up` per layer when the file carries no `zamba2.ffn_gate_up_adapter_rank` key, and reads the
low-rank pair when it does. Nothing published before the change stops working.

The two forms are **not bit-identical**, and that is expected: the merged file has `up @ down` computed in f32 at
conversion time and rounded to bf16, while the adapter path evaluates the pair at run time. Measured against the
merged form's own logits (CPU, 16 × 512):

| model | mean KLD | same top-p | PPL |
|---|---|---|---|
| Zamba2-2.7B-instruct | **0.000067** | 99.510 % | 16.795417 (merged 16.7937) |
| Zamba2-7B-Instruct | **0.000095** | 99.755 % | 8.499977 (merged 8.5028) |

Both are two to three orders of magnitude below the family bar below, so a quantized arm still has to be
**re-measured** rather than inherited from the merged form — see the next section, where one arm changed sides.

## What to publish per model

The family bar is **mean KLD ≤ 0.03 AND same-top-1 ≥ 95 %**, and each model gets a sweet spot, a max-quality
and a smallest-usable pick. The Zamba2 rows are the adapter form; the others are unchanged.

| model | arm | role | size | KLD | top-1 |
|---|---|---|---|---|---|
| Zamba2-2.7B-instruct | **Q6_K** | sweet spot | 2119 MiB | 0.007244 † | 95.711 % † |
| | Q8_0 | max quality | 2728 MiB | 0.001242 | 98.235 % |
| | Q4_K_M-imx | smallest usable | 1511 MiB | 0.043205 | 90.858 % |
| Zamba2-7B-Instruct | **Q6_K** | sweet spot **and** smallest usable | **5829 MiB** | 0.010430 | 96.593 % |
| | Q8_0 | max quality | 7515 MiB | 0.001669 | 98.603 % |
| | ~~Q5_K_M~~ | **withdrawn** | ~~4938 MiB~~ | 0.019383 | **94.792 %** — misses the bar by 0.21 points |
| Zamba-7B-v1 | **Q6_K** | sweet spot and smallest usable | 6943 MiB | 0.002685 | 97.16 % |
| | Q8_0 | max quality | 7484 MiB | 0.000962 | 97.97 % |
| BlackMamba-2.8B | **Q8_0** | sweet spot and max quality | 2831 MiB | 0.001312 | 97.99 % |
| | IQ4_XS-imx | smallest usable | 1542 MiB | 0.026519 | 91.01 % |

† 32-chunk run (32 × 512 = 16384 tokens), because the 16-chunk run sat too close to the bar to decide; every
other row is a 16-chunk run. Mark the long rows when quoting them.

**Why the 7B moves up a tier.** The adapter form perturbs top-1 in proportion to how close an arm already sat to
the bar, and the two Zamba2 models reacted differently to the same change: the 2.7B's top-1 moved **up** 0.05
points, the 7B's **down** 0.52. `Q5_K_M` was clearing the 95 % bar by 0.025 points on the merged form, so it no
longer clears it (94.792 %), and `Q6_K` — 5829 MiB, still 11-12 % smaller than the merged `Q5_K_M` — is the
smallest arm that does. `Q5_K_M` is listed above as withdrawn rather than deleted because it is the smaller file
for anyone who accepts the lower top-1 deliberately.

## Two things that go wrong with an older tree

**1. A quantizer that predates the F16 default will drop the adapter off HRX.** The adapter pair stays at F16 by
default, because its `B` contraction has `input_size = rank = 128`: the float weight path admits that
(`input_size ≥ 4 && % 4 == 0`), the block formats do not (their block layout needs `≥ 256`). Quantize the pair
with an older `llama-quantize` and that matmul leaves the device on every hybrid layer — measured on a 2.7B
`Q5_K_M` as **9 of 3051 nodes on the CPU and pp512 463.66 vs 589.47 tok/s (0.79x, -21 %)**, with no error to tell you.
If your tree predates the default, pass it explicitly:

```
llama-quantize --tensor-type ffn_gate_up_adapter=f16 <in.gguf> <out.gguf> Q6_K
```

Cost of the F16 pair: **+31.8 MiB** on an 1804.93 MiB 2.7B file. An explicit `--tensor-type` still wins;
`--pure` still means pure.

**2. Assert the layout before you trust a PPL.** A pair written transposed (`{rank, n_embd}` instead of
`{n_embd, rank}`) is refused by the loader with `check_tensor_dims`, but only after a conversion; the cheap check
is on the container: `ffn_up` **×2** (not one per hybrid layer), `zamba2.ffn_gate_up_adapter_rank = 128`,
adapters **9 + 9** at 2.7B and **13 + 13** at 7B, and the byte sizes in the table above.

## HRX placement, if you are quoting a number

The adapter path adds two `MUL_MAT`s and an `ADD` per hybrid layer and **all of them are placed on HRX** —
`+27` nodes at 2.7B (9 × 3), with the same off-device residue as the merged form, and `GGML_HRX_STRICT=1` exits 0
on both forms at prefill and decode. There is no throughput cost to report: pp512 441.98 → 440.57 and tg64
22.87 → 22.60 on one binary with both forms (0.997 / 0.988).

One caveat for anyone running `-ngl 99` today: the token-embedding `GET_ROWS` sits on the CPU **by default**, and
that is #170's policy rather than this family's doing — the input layer stays on the CPU
(`src/llama-model.cpp`: "very little benefit to offloading the input layer"). `LLAMA_INPUT_LAYER_ON_DEV=1` is the
opt-in, and it works for a tied embedding since fork **#233** (in the pin): a tied embedding is gathered natively by
`GET_ROWS` and contracted in a packed layout by the output `MUL_MAT`, which a command program cannot hold twice, so
the dispatch now declines the packed match for a weight another node gathers — measured as **0 nodes off HRX** with
the switch on, 3/3 runs, against a refusal in 3/3 with that gate disabled. It stays **off by default** because device
input measured ~6 % slower than CPU input on a 0.6B while removing one node and one split per graph.

So at default settings a census of `1 of 3024` nodes with `GET_ROWS x 1` and 2 graph splits is the floor for **every**
model at `-ngl 99`, Zamba2 forms included, and it is not residue these changes added; `LLAMA_INPUT_LAYER_ON_DEV=1`
is how you leave it. (Engine **#1316** is where both halves were measured, and it is not the #856 host-pointer import:
while that import existed, `GGML_HRX_WEIGHT_HOST_PLACEMENT=0` reproduced the placement identically — and it has since
been replaced outright, so at this pin the import is not in the tree at all. [hrx.md](hrx.md) records that removal.)
