> **Scope of this page: what the arms measure.** This is the showcase half of two Zyphra docs and it owns exactly one job — the per-arm
> measurements (KLD, same-top-1, the two corpora, arm-label findings, what the numbers do and do not support). **What the files *are*, and how to
> convert and quantize them** — ref sizes and tensor counts, the `--tensor-type ffn_gate_up_adapter=f16` trap for quantizers older than the F16 default,
> the layout asserts worth running before a PPL, and input-layer placement — lives in [`docs/zyphra-models.md`](zyphra-models.md), which is where a
> reader who wants to *produce* a Zamba2 file should start. The split exists so the two pages cannot drift: neither repeats the other's tables, and
> each links here. If you are looking for the adapter-form size deltas or the quantizer trap, that is the other page.

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
# Zyphra

[Zyphra](https://huggingface.co/Zyphra) ships four families we convert:
**ZAYA1** (a sparse MoE, `model_type: zaya`), **ZR1** (which is `model_type: qwen2`,
so it runs on the ordinary Qwen2 path), **Zamba/Zamba2** (Mamba2-hybrid SSMs) and
**BlackMamba** (Mamba-1). Every repo we convert is Apache-2.0 per its own
`cardData.license`.

Everything here is converted from the **original Zyphra weights**, freshly downloaded at
a recorded revision into `~/models/zyphra-orig/<model>@<rev12>/` with a `SHA256SUMS`, and
then to GGUF with this fork's `convert_hf_to_gguf.py` — never from an existing GGUF and
never from an older upload. The revision and the per-file sha256 of the originals are what
the numbers below are anchored to.

## Engine support (measured against this pin)

| model | HF `model_type` | convert with this pin | run with this pin |
|---|---|---|---|
| ZAYA1-8B | `zaya` | yes, `conversion/zaya.py` | yes, `zaya` graph |
| ZAYA1-VL-8B | `zaya1_vl` | yes, same converter | text graph yes; VL needs an `mmproj` |
| ZR1-1.5B | `qwen2` | yes, stock converter | yes, standard Qwen2 |
| Zamba2-2.7B-instruct | `zamba2` | **from the pin** | yes, the pinned graph |
| Zamba2-7B-Instruct | `zamba2` | **from the pin** | yes, the pinned graph |
| Zamba-7B-v1 | `zamba` | **from the pin** | yes, the pinned graph |
| BlackMamba-2.8B | custom (no `architectures`, Mamba-1 + MoE blocks) | **from the pin** (ships `pytorch_model.bin`) | yes, the pinned graph |

So the SSM half of the family is not servable from this pin today: the gap is a converter
*and* a graph per arch, not a quantization problem. Zamba2 ships its own engine in the
monorepo (`Makefile.zamba2`, `tools/convert_zamba2_safetensors_to_gguf.py`, arch `zamba2`).

Three of those gaps were closed **on branches and are all merged now**: `1bit/zamba2` (PR #178), `1bit/blackmamba` (PR #183)
and `1bit/zamba` (PR #184, stacked on #178) add the three graphs, each CPU-proven
(`test-llama-archs` round-trip OK, plus a KLD of the BF16 against the transformers reference:
Zamba2 0.0030, BlackMamba 0.0035, Zamba v1 0.0022). **All of it is at the engine pin, so ask the pin rather than this page.** The pin is whatever the engine's submodule
gitlink says — `git ls-tree origin/main third_party/llama.cpp` — and a literal sha here would be stale before the next review round: this page carried
`c16424f56` and `6e42864a6` while the pin was already `b970d85ba`. Verified at `b970d85ba` on 2026-11-08: `conversion/zamba.py`, `zamba2.py`,
`zamba2vl.py`, `src/models/zamba.cpp`, `src/models/zamba2.cpp` are present and `src/llama-arch.cpp` carries **4** `LLM_ARCH_ZAMBA*` registrations, so the
converters and graphs that used to need a fork head are in the pin. The **KLD/top-1 rows below are CPU measurements** (`GGML_HRX=OFF`, Release) while the **throughput rows are HRX** (pp512 / pp4096 / tg128,
3 repeats, from a fork-head build at `work/zarch/i1036@9ad387fb9`, `-ngl 99`, Tctl noted per row — see *Throughput on Strix Halo* below). That CPU/HRX split is legitimate here:
the same arm measured on both backends on ZAYA1 gives KLD `0.429220` (HRX) against `0.425474` (CPU), a 0.9 % spread. Rows that could not
be measured at all are marked blocked rather than dropped.

## Family summary: where the sweet spot actually is

Identical harness for every row below: arm quantized from **our own BF16 GGUF of the freshly downloaded original @rev**,
measured with `llama-perplexity --kl-divergence` against that BF16's logits on **16 × 512-token wikitext-2 windows**
(8192 scored tokens — the exceptions are Zamba2's `Q6_K` and Zamba2-7B-Instruct's `Q5_K_M`/`Q6_K`, measured over 32 × 512 = 16384
because their 16-chunk top-1 sat within noise of the bar), imatrix arms from a calibration set disjoint from the eval
text, and every arm's type histogram censused before its label is believed. The bar is this repository's usual one:
**mean KLD ≤ 0.03 *and* same-top-1 ≥ 95 %**. The brief also asks for a **code slice** next to wikitext; those rows are measured, separate,
and kept out of this table on purpose — see *The code slice* below for why a corpus that makes every arm look better must not be able to
change a pick.

| model | max quality (KLD ≤ 0.005) | sweet spot (meets the bar) | smallest usable | one line |
|---|---|---|---|---|
| ZAYA1-8B | Q8_0 8.84 GiB (0.0281) | **none** | Q4_K_M 5.44 GiB (0.561 / 69.3 %) | no arm meets the bar; 4-bit is 19× the KLD limit |
| Zamba2-2.7B-instruct *(adapter form, 2026-11-07)* | Q8_0 **2728 MiB** (0.00124) | **Q6_K 2119 MiB (0.00724 / 95.71 % at 32 chunks)** | Q4_K_M-imx **1511 MiB** (0.0432 / 90.9 %) | cause 2 shrank it 9-11 % and the pick survived; cliff still between Q5_K_M and Q4_K_M |
| BlackMamba-2.8B | Q8_0 2831 MiB (0.00131) | **Q8_0** 2831 MiB | IQ4_XS-imx 1542 MiB (0.0265 / 91.0 %) | 8-bit exceptional, 4-bit ordinary, half the labels lie |
| Zamba-7B-v1 *(11 of 11 arms)* | Q8_0 7484 MiB (0.00096) | **Q6_K 6943 MiB (0.00269 / 97.2 %)** | **Q6_K 6943 MiB** — no 4-bit arm meets the bar | cliff is a full bit higher than the 2.7B's; Q6_K is both the sweet spot and the smallest arm that clears the bar |
| Zamba2-7B-Instruct *(adapter form, 2026-11-06)* | Q8_0 **7515 MiB** (0.00167) | **Q6_K 5829 MiB (0.01043 / 96.59 %)** | **Q6_K** — `Q5_K_M` misses top-1 by 0.21 points under this layout | cause 2 moved the pick up a tier: files 11-12 % smaller, KLD unchanged, and the arm that cleared the bar by 0.025 points no longer does |
| ZR1-1.5B | Q8_0 1807 MiB (0.002827) | **Q8_0** 1807 MiB (0.002827 / 97.21 %) | none (Q4_K_M 1066 MiB = 0.1274 / 82.2 %) | 4-bit costs 4.2× the KLD limit; the NPU container is not a quality route |

- **The sensitivity is architectural, not "the Zyphra family".** ZAYA1 (MoE with EDA recurrence) is destroyed at 4 bits
  (0.56 KLD) while the SSM models are near-lossless at 8 bits (**0.0012-0.0013**, ~20× better than ZAYA1's Q8_0) and only
  fall over at 4. "Zyphra needs Q8" is true for ZAYA1 and false as a family statement.
- **Q8_0 is the only arm that passes the bar on every model.** It is the safe answer wherever a small enough file exists.
- **The imatrix is the best free lever and it is not a bar-crosser**: ~20 % of the KLD error removed at identical size
  (BlackMamba Q4_K_M 0.0118 → 0.0092, IQ4_XS 0.0330 → 0.0265; Zamba2 0.0562 → 0.0445 and 0.0628 → 0.0521). On BlackMamba it
  is what lets a sub-2 GB file meet the KLD half of the bar; on Zamba2 it still lands 1.5× over the limit.
- **KLD and same-top-1 are not redundant, which is why the bar has two halves**: Zamba2 Q6_K is 3.4× inside the KLD bar and
  0.15 points *outside* the top-1 bar, and BlackMamba Q5_K_M / Q4_K_M-imx are each 0.8 points short on top-1 while KLD looks
  comfortable. Judged on KLD alone, three arms would have been over-recommended.
- **Perplexity cannot see any of this**: across the family PPL moves a few percent while KLD moves 30-70×. A table that ranks
  these arms by PPL is measuring nothing.
- **Arm labels are not evidence.** On BlackMamba five of eleven nominal arms are not what their names say (a "Q6_K" that was
  pure q8_0, a "TQ2_0" that was q4_0, every "IQ2/IQ3" a ~4-bit file) and **no number looked wrong** — the type census caught
  them, not the metrics. Zamba2-2.7B is the clean case (all 11 arms exact, no fallbacks). **Zamba-7B-v1 is a third case and I got it
  wrong at first**: it is not 256-divisible (`ncols` 3712 and 232), so its K-quant/IQ arms substitute 167 recurrent-path tensors to
  a legacy type and keep 76 f16 — partial and upward (q8_0 under Q6_K), so the numbers describe a slightly-finer-than-nominal file.
  The rule that survives all three: read the census before believing a label, and say which case a model is.
- **ZR1-1.5B (qwen2 arch, measured by pi-onebp on this same 16-chunk harness): Q8_0 is the only arm that meets the bar**
  (0.002827 / 97.21 %, 1807 MiB) and `Q4_K_M` is not a fallback at 0.1274 / 82.2 % for 1066 MiB — 4.2× the KLD limit, the same
  verdict as ZAYA1. Their NPU containers do not rescue it: the Q8 container (0.1197 / 82.5 %, 1552 MiB) does beat the Q4_K_M arm it
  was packed from, and it is the only container that beats its own source, but both sit an order of magnitude outside the bar. So on
  this family **8-bit is the only honest bit-width below BF16 for ZAYA1 and ZR1** (both precision-sensitive architectures), while the
  SSM models are near-lossless at 8 bits and **all of them clear the bar at 6 bits** (`Q6_K`: Zamba-7B-v1 0.00269 / 97.16 %,
  Zamba2-2.7B 0.00730 / 95.66 %, Zamba2-7B-Instruct 0.008048 / 96.64 %) — so the 6-bit tier is real for the SSM half and nowhere
  else in the family. *(Superseded by cause 2.)* This bullet used to say Zamba2-7B-Instruct did not even need 6 bits, on `Q5_K_M` at
  0.01936 / 95.03 %. Under the adapter form (fork #218) that pick **falls off the top-1 bar** — 0.019383 / 94.792 % at 32 chunks — and the
  sweet spot moved up to `Q6_K` (0.010430 / 96.593 %), so **every model whose sweet spot is below BF16 now sits at 6 bits or higher**, and
  the sentence that "Zyphra needs 8 bits" is a ZAYA1/ZR1 statement is still the point, just without the 5-bit exception.
- *(Historical — the merged-form record. The adapter form superseded this pick; see that model's section.)*
  **Zamba2-7B-Instruct (the fifth model): `Q5_K_M` was the sweet spot of the merged form, decided on a 32-chunk run rather than on a 0.025-point
  margin.** At 16 chunks `Q5_K_M` was 0.026074 / 95.025 % while being 1265 MiB smaller than `Q6_K` (0.010801 / 96.446 %), which is
  noise rather than a decision, so both were re-measured over 32 × 512 windows: **0.019890 / 95.061 %** against 0.008048 /
  96.642 %. It held, and it is the smallest arm that clears either half of the bar. The two arms also re-quantized to
  **bit-identical sha256** in the second run (both files were released and rebuilt), so the rows describe the bytes. After the
  dedupe below the same pick re-measures at **0.019361 / 95.025 %** — still across both halves.
- **Engine #1036 has landed and the sizes are re-measured, not estimated** — what the refs weigh and why the shared block is no longer copied
  13 times is [`zyphra-models.md`](zyphra-models.md)'s subject (it has the byte-level table), so this page states only what a *measurement* needs: with
  cause 1 (fork #203, `6e42864a6`) the refs were rebuilt and
  the picks were re-quantized and re-measured against them: **Q5_K_M 5577 MiB** (was 7336), **Q8_0 8558 MiB** (was 11135) on the 7B;
  **Q6_K 2356 MiB** (was 3002), **Q8_0 3049 MiB** (was 3886), **Q4_K_M-imx 1658 MiB** (was 2159) on the 2.7B. The **KLD and same-top-1
  columns are unaffected**, and that is measured rather than argued: the rebuilt BF16 run against the **old** base logits gives
  **mean KLD 0.000000 / same-top-p 100.000 %** over 8192 tokens with the recorded PPLs reproduced (8.502790 and 16.793681), so the
  two layouts are the same model. Curve-arm sizes that were not re-quantized are still the pre-dedup figures and are marked as such.
  `gate_up`/`ffn_up` used to stay per layer because it is 13 distinct values on the 7B and 9 on the 2.7B (a per-layer adapter baked into
  the matrix). **That residual has since been fixed as well**: fork **#218** (`21c814079`, engine #1036 cause 2) stores the shared base once and
  each layer's low-rank pair separately, applied as `base + B*(A*x)` in the graph, and it **has landed**. So both halves of #1036 are resolved:
  the layout in the paragraph above (cause 1, fork #203) and the per-layer adapter (cause 2, fork #218). The consequence for a reader is not
  cosmetic — the adapter form is slightly smaller and not bit-identical, which is why the Zamba2 rows on this page are re-measured in that form and
  why the 7B's `Q5_K_M` pick was retired; see each model's section.
- **One measurable side effect of the dedupe, which is not a weight change: `Q5_K_M`'s KLD moved 0.019890 → 0.019361 and its PPL
  7.759 → 7.727.** `llama-quantize` upgrades a `Q5_K_M` tensor to `q6_K` by *position* (`use_more_bits(i_attention_wv, n_attention_wv)`),
  so the shared `attn_v` tensor's type depends on how many copies of it the file contains: with 13 copies, 6 of them got `q6_K`; with the
  deduped file's 2 copies, only `blk.11.attn_v` does (`blk.6.attn_v` stays `q5_K`, both read out of the quantize log). Same weights,
  different type mix, so a different file and a slightly different number — which is exactly why every row here is measured per file and
  not derived from the arm's name. `Q8_0`, `Q6_K` and the other uniform-type arms reproduced their KLD **digit for digit**.
- **Labels on Zamba2-7B-Instruct are exact: zero fallbacks in all seven quantized arms** (`type-census.txt`), i.e. the second clean
  case after the 2.7B and unlike BlackMamba or Zamba-7B-v1.

## The code slice: what the picks do off wikitext (every number above is wikitext-only)

The brief's measurement rule is *KLD against the BF16 on wikitext-2 **and a code slice***. Every row on this page until now was wikitext; the
code half was built on 2026-11-04 and then not run, so this section is new. The corpus is deterministic and redistributable —
`calib/code-slice.txt`, **2 998 513 bytes from 140 whole files**, `src/models/*.cpp`, `src/*.cpp`, `conversion/*.py`, `tools/*.py`,
`gguf-py/gguf/*.py` of this fork at `ebb75bf54` (sha256 `4efdd5265f268929…`), generated by `zq-code-slice.sh` — and it was measured with
the same harness as everything else (`llama-perplexity --kl-divergence` against the same BF16's logits on the same corpus), **8 × 512
windows** rather than 16, because the pass had to fit a night on a box whose CPU is shared 8 ways. Half the tokens means more noise per row;
what makes the conclusion safe is that the sign is the same on all thirteen of them.

| model | arm | wikitext KLD / same-top-1 | **code** KLD / same-top-1 | verdict |
|---|---|---|---|---|
| Zamba2-2.7B-instruct *(adapter form)* | **Q6_K** (sweet spot) | 0.007244 / 95.71 % (32 ch) | **0.003908 / 97.89 %** | clears both halves on both corpora |
| | Q8_0 (max quality) | 0.001242 / 98.24 % | **0.000550 / 99.22 %** | " |
| | Q4_K_M-imx (smallest usable) | 0.043205 / 90.86 % | **0.010407 / 97.50 %** | misses on wikitext, **clears on code** |
| Zamba2-7B-Instruct *(adapter form)* | **Q6_K** (sweet spot) | 0.010430 / 96.59 % | **0.003939 / 98.19 %** | clears both halves on both corpora — `Q5_K_M` did not survive this form, see that model's section |
| | Q8_0 (max quality) | 0.001669 / 98.60 % | **0.000488 / 98.97 %** | " |
| BlackMamba-2.8B | Q8_0 (sweet spot + max quality) | 0.001312 / 97.99 % | **0.000862 / 99.56 %** | " |
| | IQ4_XS-imx (smallest usable) | 0.026519 / 91.01 % | **0.013286 / 97.01 %** | misses top-1 on wikitext, **clears on code** |
| | IQ4_XS | 0.032975 / 90.27 % | 0.020463 / 96.86 % | misses both halves on wikitext, **clears on code** |
| Zamba-7B-v1 | **Q6_K** (sweet spot) | 0.002685 / 97.16 % | **0.000964 / 99.41 %** | " |
| | Q8_0 (max quality) | 0.000962 / 97.97 % | **0.000336 / 99.71 %** | " |
| | Q5_K_M | 0.010715 / 94.49 % | 0.002662 / 98.63 % | misses top-1 on wikitext, **clears on code** |

**Zamba2 rows re-measured 2026-11-07 in the adapter form** (fork #218), because cause 2 replaced both models' files and their earlier code rows
described bytes that no longer exist. **The code column moved less than the wikitext column did** — KLD within 1.3 %, top-1 within 0.25 points — which
is the expected shape: the form's own perturbation (~1e-4 KLD) is an order of magnitude below a quantised arm's error (1e-3 to 4e-2), so a corpus
comparison is the wrong instrument for seeing it. Two 2.7B arms that were in the old table are gone from this one: `Q5_K_M` (never rebuilt in the
adapter form) and `Zamba2-7B-Instruct`'s `Q5_K_M`, which is **withdrawn** — it misses the top-1 bar in this layout. The two `Zamba2-7B-Instruct` code cells were the last to be
measured, and the reason is worth one sentence because it is a property of the box rather than of the files: four attempts were refused by a different
precondition each time — a 4-core job that measured **0.19 of one core** while three `thermal-run` scopes shared the same pinned cores, then under 6 GiB
of free RAM against the ~7 GiB a 7B perplexity needs to load. The cells stayed marked while that was true rather than being filled from a neighbouring
window, and they are measured now.

† the wikitext row is the **pre-#1036 file**; that arm was never re-measured on wikitext after the dedupe (its KLD is layout-invariant —
proven by the 0.000000 identity check — but a 4-bit arm's type mix can shift with the copy count, see `Q5_K_M` above). The pass ran in two
instalments on 2026-11-05/06: `Zamba-7B-v1 Q8_0` was skipped at 00:43 by the disk guard and measured at 05:12 once space allowed, and
`Zamba-7B-v1 Q5_K_M` was skipped in that same second pass for the same reason — it was already measured in the first.

What this actually says, in the order I would defend it:

1. **No pick moves the wrong way, so no pick is re-decided.** All thirteen arms are better on code than on wikitext — KLD down by 1.5-7×,
   same-top-1 up by 0.4-7 points (eleven of them against the *same file's* wikitext row, the two † arms against the pre-dedup file). A pick
   that holds on the *harder* corpus holds on both, so the sweet spots stay where the wikitext table
   put them. This is the result that matters: the code pass was a risk check, and it did not find a risk.
2. **But the bar is corpus-dependent, and that is the real finding: six arms that miss it on wikitext clear it on code** (the four 4-bit
   arms, the 2.7B's `Q5_K_M`, and Zamba-7B-v1's `Q5_K_M`). So "smallest usable" on this page means *smallest usable on natural text*.
   The code column **must not** be used to promote an arm — the honest reading is the opposite one: wikitext is the binding corpus, and a
   4-bit file that only clears the bar on code is exactly the file you should not recommend without saying which corpus you measured.
3. **PPL is worse than useless as a cross-corpus statistic here.** The BF16's own perplexity is 4.45 (2.7B), 4.15 (7B-Instruct), 3.41
   (BlackMamba) and 4.26 (Zamba-7B-v1) on code against 16.79, 8.50, 15.44 and 20.93 on wikitext — the same weights, a 2-5× different
   number, and the arms' PPL ordering inside a corpus still tells you nothing about their KLD ordering.
4. **Scope, stated so it cannot be read as more than it is**: one code corpus, which is this project's own C++ and Python; it is not a
   general code benchmark, and a model fine-tuned on another language or another style would not necessarily behave the same way. ZAYA1-8B
   is not in this pass (publish on HOLD), ZR1-1.5B is pi-onebp's and was measured at 8 chunks on wikitext only, and ZAYA1-VL-8B has no
   row at all.

## Throughput on Strix Halo (HRX)

Same files the KLD tables describe, measured with `llama-bench` from the fork head (`work/zarch/i1036@9ad387fb9`, which carries
`zamba`/`zamba2`/`blackmamba` and #203), `-dev HRX0 -ngl 99 -p 512,4096 -n 128 -r 3`, one file at a time under the device ticket.
Each arm is sha256-gated against its own KLD row before it is benched, so a throughput row cannot end up describing a different file
than the quality row next to it. Tctl is read at the end of the run; all rows are 2026-11-05.

| model | arm | pp512 | pp4096 | tg128 | Tctl |
|---|---|---|---|---|---|
| ZAYA1-8B | Q8_0 | 738.00 | 698.51 | 27.83 | 72 C |
| ZAYA1-8B | Q6_K | 1254.05 | 582.85 | 24.73 | 67 C |
| ZAYA1-8B | Q4_K_M | 1259.54 | 679.31 | 25.52 | 67 C |
| Zamba2-2.7B-instruct | **Q4_K_M-imx** | 610.27 | 25.57 | 22.04 | 68 C |
| Zamba2-2.7B-instruct | **Q6_K** | 343.42 | 25.10 | 17.83 | 69 C |
| Zamba2-2.7B-instruct | Q8_0 | 363.24 | 25.79 | 32.87 | 66 C |
| Zamba2-7B-Instruct | Q5_K_M *(merged-form pick, retired)* | 235.70 | 11.54 | 10.52 | 69 C |
| Zamba2-7B-Instruct | Q8_0 | 140.64 | 10.22 | 13.08 | 66 C |
| Zamba-7B-v1 | **Q6_K** | 108.07 | 92.28 | 12.20 | 64 C |
| Zamba-7B-v1 | Q8_0 | 100.54 | 89.14 | 11.40 | 63 C |
| BlackMamba-2.8B | **Q8_0** | 4985.27 | 5082.12 | 164.81 | 70 C |
| BlackMamba-2.8B | IQ4_XS-imx | 5065.11 | 5167.39 | 139.64 | 62 C |


**Why the adapter-form `Zamba2-7B-Instruct` `Q6_K` row is missing rather than filled** (2026-11-08, and it is not a gap in the measurement): the six-arm
survey that would have supplied it ran in one controlled window (its own logs: GTT 37.5-39.9 GiB) on the **current** HRX build (`a3e8d42c9`), while every
number in this table came from a build that has since been deleted (`9ad387fb9`). The two are not comparable, and two of its three figures say so out loud:

| arm | pp512 | pp4096 | tg128 | note |
|---|---|---|---|---|
| `Zamba2-7B-Instruct` `Q6_K` | **5.09** | 12.54 | 18.55 | pp512 below pp4096 — physically implausible, i.e. the first test carries a one-off cost (kernel compile/warmup) |
| `Zamba2-2.7B-instruct` `Q6_K` | 535.93 | **97.03** | 45.07 | pp4096 **3.9× faster** than the 25.10 this table carries from the old build |
| `Zamba2-2.7B-instruct` `Q8_0` | 686.74 | **91.57** | 21.16 | same |

So a row from that window would read as a verdict on the file when it is a verdict on the build. The honest statement is the one above the table: **this page's
throughput column is a record of one deleted build**, the adapter form's current numbers exist (in `bench-*-adapter.json` beside each arm's row) and are not
comparable to it, and a like-for-like column needs either that build rebuilt or the whole column re-measured in one window on the current one. The 2.7B's
pp4096 going 25 → 97 between builds is the size of the error a quiet mix would have introduced.
(`tok/s`, three repeats; bold = the model's sweet-spot pick. **`Zamba2-7B-Instruct`'s bold row is its retired merged-form pick** — the adapter
form's sweet spot is `Q6_K`, and **it has no row in this table on purpose**: every number here comes from one build, and the one re-bench I ran for it used a
different one (details in the footnote below). No ranking is to be read from a bold cell in that column.
ZR1-1.5B has no row from this harness — its arms are measured by the
1BP lane on the same KLD harness, not benched here.) **Which rows share a window matters more than which number is bigger**: the three
Zamba2-2.7B rows are one ticket (20:43-21:16) that failed its own control, and the four `Zamba-7B-v1`/`BlackMamba` rows are **one ticket
(2026-11-06 04:59-05:08) that passed it** — GTT in use 35.7-38.7 GiB across all four arms, the same two idle `llama-server`s at both ends,
queue wait 0 s, die 62-70 C. Read the decode bullet before comparing any two numbers in the last two columns: the same three 2.7B files
have been measured in three windows and answered differently every time.

- **Two caveats on every number in the two right-hand columns, both discovered on 2026-11-07.** (1) The HRX builds that produced these rows
  routed `Q4_K`/`Q5_K`/`Q6_K` prefill through the fork's fused dispatch tiles, and per the HRX lane's fleet note of 2026-11-07 those tiles are
  proven only by "a benchmark emitting non-garbage tokens" — no known-answer comparison, no KLD, and the repo's own gate
  (`test-backend-ops -o MUL_MAT -b HRX0`) had not been run against them. Nothing about the **quality** columns is affected: every KLD and
  same-top-1 row on this page is CPU (`GGML_HRX=OFF`). What is affected is the meaning of `tok/s` — it is what those kernels produced, and it has
  not been validated against a known-answer. (2) The bench build these rows cite (`work/zarch/i1036@9ad387fb9/build-hrx`) **has been removed from
  the box**, so the rows are not re-derivable without rebuilding that tree; the tree itself is still at `9ad387fb9`. A future re-bench should
  take the dispatch histogram first (`GGML_HRX_LOG_DISPATCH=1`, count registrations by name) and run the MUL_MAT gate before quoting a ratio — but note
  what that histogram **cannot** tell you: in the A/B above the two arms' registration sets were near-identical (the difference was one extra line, none
  missing), because the log names registrations that *did* run and an op that never dispatched on HRX simply leaves no line. A fallback shows up in the
  **timing**, not in the registration set.
  (3) **And the quantizer that produced the file decides whether these numbers mean anything at all**: a `llama-quantize` older than the F16 default
  quantizes the rank-128 adapter pair, whose `B` matmul then leaves HRX on every hybrid layer — the block formats want `input_size >= 256` while the
  float path only needs `% 4` — with no error and a large prefill cost. That mechanism is documented, with the measured numbers and the
  `--tensor-type ffn_gate_up_adapter=f16` remedy, in [`zyphra-models.md`](zyphra-models.md); it is the most likely explanation for the unexplained
  prefill drop noted below, and it is why a throughput row here is inseparable from *which quantizer built it*. **Measured here on 2026-11-08**: two arms
  that differ *only* in the adapter tensors' type (F16 ×26 vs Q6_K ×13 + Q8_0 ×13), benched back to back in one ticket and one window —

  | shape | F16 pairs | block-quantized pairs | cost |
  |---|---|---|---|
  | `pp8` | 114.19 tok/s | 14.03 tok/s | **8.1×** |
  | `pp512` | 145.09 tok/s (sd 79.2) | 26.63 tok/s (sd 8.5) | **5.4×** |
  | `tg128` | 16.50 tok/s (sd 0.3) | 10.24 tok/s (sd 7.3) | **1.61×** — direction only, the block arm's sd is 71 % |

  **5.4× at `pp512` — direction stable, magnitude not**, for a **50 MiB** difference in a 5.8 GiB file, and the arms differ in nothing else.
  The F16 arm's own sd is **79.2 on 145.09 (CV 55 %)** and the block arm's is 32 %, so the ratio is an order of magnitude, not a measured multiple:
  read it as "several times slower", and treat `pp8`'s 8.1× the same way. The independent report from the
  lane that found the trap measures ~1.3× on its own build and model, so the *size* of the penalty is file- and shape-dependent — but its direction and its
  cause are not, and no card may quote a `tok/s` for a Zamba2 file without saying which quantizer produced it.
  **And when reporting which file suffered it, quote the types read out of the produced GGUF, never the ones you asked for** — the quantizer substitutes
  per tensor: asking for `ffn_gate_up_adapter=q6_K` yields `_a` = Q6_K but `_b` = **Q8_0**, because the `[128, 28672]` pair's contraction has
  `input_size = 128` and HRX's block-format admission needs `>= 256` while the float predicate needs only `>= 4`. So the pair leaves the device for *any*
  block type, including the substitute — the same "the label is not evidence" rule this page applies to arm names, applied to the tensors inside them.
- **A control on 2026-11-08 shows how far a single reading can be off, and it retire the A/B this page would need.** One file whose form cause 2
  did **not** touch (`Zamba-7B-v1` `Q6_K`) was re-benched on a current build: `pp512 211.16 (sd 170.84)`, `pp4096 47.14 (sd 12.73)`, `tg128 4.26
  (sd 3.04)` against its own published `108.07 / 92.28 / 12.20`. **The same bytes, 2-2.9× apart, with 71 % sd on decode.** So the rows in this
  section are historical observations of a build that no longer exists, and they are not reproducible even file-against-itself.
- **And the question those rows invite — "did cause 2 make Zamba2 slower?" — is no longer answerable on this box.** A/B needs a merged-form file to
  compare against, and the merged artifacts are all gone: both merged refs, the 7B's shards, and every merged-form arm file. Re-creating one means a
  scoped re-download plus a conversion with the old tree (`wt/zamba-6e42` @ `6e42864a6`). What *can* be measured is the adapter form's absolute
  throughput with its own reproducibility checked (the same arm twice inside one ticket), which is a weaker claim and must be written as one.
- **Long prefill collapses on both Zamba2 models and on nothing else in the family.** Every Zamba2 arm falls 14-24× from pp512 to
  pp4096 (2.7B 610→26, 343→25, 363→26; 7B 236→12, 141→10), across three different quant types — while Zamba-7B-v1 loses 11-15 %
  (108→92, 101→89 in the controlled window) and BlackMamba *gains* (4985→5082, 5065→5167; its prefill is close to linear in a flat Mamba
  graph). Five rows over three arm types
  make this a property of the `zamba2` graph rather than of any file, and this page does not diagnose it: it is handed to the
  architecture lane as a measurement, not a hypothesis, and a long-prefill claim for these two models should not be made until it lands.
  The collapse lives in the **pp4096** column, which is also the one column that reproduced across all three windows (24.4-26.4,
  ±4 %), so it is not the window effect described next; pp512 is, and this bullet does not rest on it.
- **A controlled pair is worth more than any single-window row, and the two this page now has reversed one ordering and widened another.**
  The four `Zamba-7B-v1`/`BlackMamba` rows above share one window with the GTT control assertion satisfied, so for the first time this page
  compares two arms of the same model without a confounder between them:

  | pair | single-window rows (earlier) | **one controlled window (05:00)** | what it means |
  |---|---|---|---|
  | `Zamba-7B-v1` pp512, `Q6_K` vs `Q8_0` | 108.83 vs **110.05** | **108.07 vs 100.54** | the 1 % ordering **reverses** — it was noise |
  | `Zamba-7B-v1` decode, `Q6_K` vs `Q8_0` | 12.39 vs 11.85 (sd 0.04/0.08) | **12.20 vs 11.40** (sd 0.04/0.08) | ordering and size of the gap reproduce; this model's decode is genuinely well-behaved |
  | `Zamba-7B-v1` pp4096 | 97.99 vs 94.75 | **92.28 vs 89.14** | ordering holds, absolute value 6 % lower — even the steadiest column has window effect |
  | `BlackMamba` decode, `Q8_0` vs `IQ4_XS-imx` | 162.99 vs 158.51 | **164.81 (sd 2.54) vs 139.64 (sd 9.80)** | same direction as before but 18 %, not 3 %, and the *smaller* file is the slower and noisier one |
  | `BlackMamba` pp512 | 4222 vs 4770 | **4985 vs 5065** | both up 6-18 % on the earlier window; ordering unchanged |

  Two readings, and they disagree in an instructive way. `Zamba-7B-v1`'s decode is reproducible to **0.3 %** with sd 0.04-0.08 in both
  windows — so its `Q6_K` > `Q8_0` ordering at a 6-7 % gap is a real property of these two files, and a card may carry it. `BlackMamba`'s
  decode is not: the `iq4_nl` file's own three repeats spanned ±9.8 tok/s, so "`Q8_0` decodes 18 % faster than the 4-bit file" is the same
  direction the earlier window saw and is now larger than that window's noise, but it is not the kind of number to put on a card as a fact.
  What both pairs do settle is the claim the page used to make and then had to withdraw: **the smaller file is not the faster decoder**, and
  on this box that is now measured twice, in windows that controlled for neighbours. The 2.7B remains the counter-example — it has never
  been measured in a window that passed the assertion, so its rows still rank nothing.
- **The decode column does not rank arms on this box from single-window rows, and neither does pp512; only pp4096 has reproduced so far.** The
  same three sha-gated Zamba2-2.7B files were benched in three windows and gave a different answer each time:

  | file | quiet window | paired window 18:33 | triple window 20:43 |
  |---|---|---|---|
  | `Q4_K_M-imx` 1658 MiB, decode | 52.48 | not run | 22.04 (sd 0.54) |
  | `Q6_K` 2356 MiB, decode | 42.16 | 22.28 (sd 0.16) | 17.83 (sd 0.25) |
  | `Q8_0` 3049 MiB, decode | 16.37 | 25.85 (sd 7.85) | 32.87 (sd 4.47) |
  | the same files, pp512 | 496 / 406 / 741 | — / 496 / 702 | 610 / 343 / 363 |
  | the same files, pp4096 | 26.4 / 24.5 / 26.3 | — / 24.4 / 25.8 | 25.6 / 25.1 / 25.8 |

  Per-arm decode spread across windows is 2.0-2.4× and pp512 spreads 1.4-2.0×, while pp4096 stays inside 24.4-26.4 (±4 %). So the
  decode column and pp512 are single-window numbers and this page draws no conclusion from them — including the one it used to draw
  ("the smallest pick decodes fastest": that rested on the quiet-window 52.48 against 42.16, and both moved by ~2× afterwards).
  Neither does a single ticket fix it on its own: the triple window changed **inside** the ticket — a Flash-Next server held GTT at
  96 GiB through the first arm and was gone by the second, a 30B server arrived 15 s in — which is why `zq-triple-bench.sh` now
  records the HRX guardrail GTT line per arm rather than once per window. A defensible decode ranking needs an idle box, and this one
  has had two `llama-server` instances on HRX0 in every window since 14:0x. What survives across all three windows is the order of
  magnitude, not the ordering: the three arms are all in the low tens of tok/s, and the single-window numbers still in this table
  (ZAYA1, Zamba2-7B, Zamba-7B-v1, BlackMamba) cannot be used for the 1.03-1.24× comparisons the page used to make from them.
- **ZAYA1 is the one model where a smaller arm is slower to prefill** (Q8_0 738 against 1259 for both Q6_K and Q4_K_M at 512). Its
  three decode rows span 24.7-27.8 tok/s across 5.1-8.8 GiB, which is 1.13× — narrower than the window-to-window spread measured on
  identical files above, so it is not evidence that ZAYA1 decode is flat across arm sizes; it is only evidence that no arm is
  dramatically slower. Its bottleneck is the MoE expert path, which is also where ~76 % of its KLD error sits.

## ZAYA1-8B: the quantization measurement

Reference: BF16 GGUF converted from `Zyphra/ZAYA1-8B@67d34da515b3`. Every arm below is
quantized **from that BF16 file**, and every arm is measured identically: 16 × 512-token
windows of `wikitext-2-raw/wiki.test.raw`, `llama-perplexity --kl-divergence` against the
BF16 logits, on Strix Halo (`gfx1151`, HRX backend, `-dev HRX0 -ngl 99`), 2026-11-03.

| arm | PPL | mean KLD | same-top-1 | size |
|---|---|---|---|---|
| BF16 (reference) | 36.73 | 0 | 100.00 % | 16.5 GiB |
| **Q8_0** | 36.35 | **0.028090** | 93.43 % | 8.84 GiB |
| FP8 experts + Q8_0 rest | 36.67 | 0.065780 | 87.99 % | 8.84 GiB |
| **Q6_K** | 38.71 | 0.105983 | 85.49 % | 6.85 GiB |
| Q5_K_M | 37.49 | 0.222033 | 79.76 % | 5.99 GiB |
| MXFP4 experts + BF16 rest | 37.16 | 0.429220 | 72.72 % | 5.51 GiB |
| **Q4_K_M** | 43.44 | 0.561042 | 69.27 % | 5.44 GiB |
| Q4_K_M, router chain at Q8_0 | 42.66 | 0.538718 | 70.32 % | 5.19 GiB |
| Q4_K_M, router chain + `ffn_gate` at Q8_0 | 43.84 | 0.546332 | 69.93 % | 5.19 GiB |
| Q4_0 | 43.44 | 0.733892 | 64.41 % | 5.10 GiB |
| IQ4_XS | 45.77 | 0.656634 | 67.13 % | 4.90 GiB |
| Q3_K_M | — | 1.012999 | 59.66 % | 4.30 GiB |

Sizes are the recorded per-arm MiB figures shown as GiB; every row is reproducible from
`~/models/zyphra-orig/zq-sweep.sh` and carries its sha256 in `results-hrx.tsv`.

### What these numbers say

- **No arm meets this repository's usual quality bar** (mean KLD ≤ 0.03 *and*
  same-top-1 ≥ 95 %). `Q8_0` is the only arm within 2× of the KLD bar and it still changes
  ~6.6 % of next-token decisions. At 4 bits the model is 19× above the bar.
- **The numbers were checked against a file we did not build.** The published
  `1bit-MONSTER/ZAYA1-8B` `Q4_K_M` measures **KLD 0.563418, same-top-1 69.461 %** through this harness,
  against 0.561042 / 69.265 % for our own build of the same arm: two independent quantizations of
  the same checkpoint, 0.4 % apart. Its tensors also match our BF16 reference at cos 0.9973-0.9996
  (Q4_K quantisation noise) and its GGUF metadata is identical (`zaya.block_count` 40, `expert_count` 16).
- **The harness itself has no bias.** An arm measured against logits it produced itself returns mean
  KLD 0.000000 and same-top-1 100.000 %.
- **The routed experts carry ~76 % of the error, and the router ~3 %.** Zyphra's own MXFP4
  experts with *everything else at BF16* cost 0.429; the non-expert 4-bit weights add ~0.13.
  The router MLP chain is not the lever: at Q8_0 it moves 0.561 → 0.546, and at **F32** (the
  most the format allows, verified in the file) 0.561 → 0.545 — F32 is no better than Q8_0, so
  spending F32 on the router buys nothing. Our graph already computes that chain's
  activations, softmax and top-k in F32 (`ggml_mul_mat` emits F32).
- **The import of Zyphra's MXFP4 experts is lossless, verified byte-for-byte.** All 80 expert
  tensors reproduce the checkpoint's own E8M0 scale bytes and 32 E2M1 codes per group; the
  same comparison was first shown to catch a swapped nibble convention, a reversed pair order,
  a single flipped nibble and an off-by-one scale. So `MXFP4exps-BF16rest` = 0.429 is the
  price of E2M1 experts on this checkpoint, not a conversion artefact.
- **Two 4-bit arms are dominated and should not be offered as "smaller options".** `Q4_0`
  and `IQ4_XS` are both worse than `Q4_K_M` for only 7-10 % less size, and `Q3_K_M` doubles
  the error again.
- **A 6-bit build is the best trade below 8 bits:** `Q6_K` is +14 % size over `Q5_K_M` for
  −52 % KLD.

### Picks

| pick | arm | size | why |
|---|---|---|---|
| max quality | **Q8_0** | 8.84 GiB | the only arm near the KLD bar; nothing below it is close |
| best trade | **Q6_K** | 6.85 GiB | +14 % over Q5_K_M for half the KLD |
| smallest usable | **Q4_K_M** | 5.44 GiB | dominates Q4_0 and IQ4_XS at the same size class |

## BlackMamba-2.8B: the quantization measurement

Reference: BF16 GGUF (5314 MiB) converted from `Zyphra/BlackMamba-2.8B@521a77772f0d` (Mamba-1 with MoE blocks,
291 tensors; KLD of that BF16 against the transformers reference 0.0035, pi-zarch). Every arm below is quantized from
that file and measured identically: 16 × 512-token windows of `wikitext-2-raw/wiki.test.raw`,
`llama-perplexity --kl-divergence` against the BF16 logits, **CPU** (`GGML_HRX=OFF`, Release, `-t 4`), 2026-11-04.

| arm | PPL | mean KLD | same-top-1 | size | what the file actually contains |
|---|---|---|---|---|---|
| BF16 (reference) | 15.44 | 0 | 100.00 % | 5314 MiB | — |
| **Q8_0** | 15.43 | **0.001312** | **97.99 %** | 2831 MiB | q8_0 ×108 (+18 odd-shaped tensors f16) |
| Q5_K_M | 15.44 | 0.010030 | 94.19 % | 2148 MiB | **q5_1 ×99**, q8_0 ×9, f16 ×18 |
| Q4_K_M | 15.63 | 0.011779 | 93.24 % | 2011 MiB | **q5_0 ×99**, q8_0 ×9, f16 ×18 |
| IQ4_XS | 15.88 | 0.032975 | 90.27 % | 1558 MiB | iq4_nl ×106, f16 ×18 |
| IQ3_XXS | 15.84 | 0.035106 | 90.05 % | 1553 MiB | **iq4_nl ×90 + q4_0 ×16** (a 4-bit file) |
| IQ2_XXS | 15.88 | 0.032815 | 90.42 % | 1542 MiB | **iq4_nl ×104 + q4_0 ×4** (a 4-bit file) |
| Q3_K_M | 15.85 | 0.040359 | 89.02 % | 1643 MiB | **q4_0 ×90 + q5_0 ×17** (~3.9-bit) |
| Q4_0 | 15.89 | 0.044148 | 88.78 % | 1542 MiB | clean q4_0 |
| Q4_K_M-imx | 15.49 | 0.009199 | 94.29 % | 2011 MiB | q5_0 ×99, q8_0 ×9, f16 ×18 (imatrix) |
| **IQ4_XS-imx** | 15.74 | **0.026519** | 91.00 % | **1542 MiB** | iq4_nl ×108 (imatrix) |
| ~~Q6_K~~ | — | — | — | (2831 MiB) | **withheld: 108/108 matrices fell back to q8_0** |
| ~~TQ2_0~~ | — | — | — | (1542 MiB) | **withheld: all matrices fell back to q4_0** |

### The arm labels do not mean what they say on this architecture

BlackMamba's matrix dims are 1472 / 2944 / 3872 / 92, and **none is a multiple of 256**, which is the block size the
K-quants and IQ-quants require. `llama-quantize` does not fail on that — it prints a warning per tensor and falls back
to a type whose block *does* fit. The result is that a nominal `Q6_K` file is a **pure q8_0 file** (same 2831 MiB, same
PPL and KLD as the Q8_0 arm to six digits; it was withheld from the table above rather than published as a 6-bit result),
`Q4_K_M` and `Q5_K_M` are really q5_0 and q5_1, and **every IQ2/IQ3 arm is a ~4-bit iq4_nl file**. That is why IQ2_XXS
(0.0328) appears to beat Q4_0 (0.0441): it is not a 2-bit arm.

The histogram of those fallbacks per arm is recorded in `quant/BlackMamba-2.8B/type-census.txt` and produced by
`zq-arm-census.sh`; every arm label in this document was checked against it. For contrast, Zamba2-2.7B has **zero**
fallbacks in every arm — its labels are exact.

### What this model can actually offer

- **Sweet spot under this repository's bar (mean KLD ≤ 0.03 *and* same-top-1 ≥ 95 %): the 8-bit file, 2831 MiB.**
  Q5_K_M is 0.0100 KLD but 94.19 % top-1 and `Q4_K_M-imx` is 0.0092 KLD but 94.29 % top-1 — both miss by ~0.8 points,
  so there is no honest "4-bit-class" sweet spot on this model.
- **The imatrix is worth ~20 % of the KLD here and it is free.** `Q4_K_M-imx` 0.011779 → 0.009199 and `IQ4_XS-imx`
  0.032975 → 0.026519 for the same file sizes; `IQ4_XS-imx` (1542 MiB) is the only sub-2 GB arm that meets the KLD bar
  at all, and it beats the plain `Q4_K_M` (2011 MiB, 0.0118) on nothing but top-1 while costing 0.0265 KLD.
- **Best genuinely small option: `IQ4_XS-imx`, 1542 MiB, KLD 0.0265, 91.00 % top-1.** Plain `IQ4_XS` (1558 MiB) is
  0.0330 / 90.27 %; q4_0 is the worst 4-bit arm of the four (0.0441 / 88.78 %).
- **Do not trust a 2-bit or 3-bit BlackMamba file.** The format cannot be produced for this arch by this quantizer; a
  "Q3_K_M" or "IQ2_XXS" filename on this model means ~4-bit content, and the measured cost is 0.033-0.040 KLD anyway.
  The same was measured, not assumed, for the ternary type: a `TQ2_0` arm fell back to q4_0 on all 108 matrices and
  reproduced the Q4_0 row exactly (15.887317 PPL / 0.044148 KLD / 88.775 % top-1, different sha256 only), so it is
  withheld as well — there is **no ternary BlackMamba** in this fork's quantizer.
- PPL is useless as the discriminator here: it moves 15.44 → 15.89 (−2.9 %) between 0.0013 and 0.044 KLD.

## Zamba2-2.7B-instruct: the quantization measurement

> **2026-11-07 07:3x — cause 2 rebuilt this model too, and unlike the 7B its pick survives.** Fork #218's adapter form, converted from the
> originals fetched at `699f349b3636` (`ref/Zamba2-2.7B-instruct-bf16-adapter.gguf`, 5,328,904,800 B, **531 tensors = 520 − 9 merged + 18
> factors + 2 bases**, nine BF16 rank-128 pairs), scored against the **same `base.kld`** as everything below — verified byte-identical to a
> base regenerated from the merged ref, so these rows are directly comparable to the ones in the table underneath:
>
> | arm | mean KLD | same-top-1 | size | merged form, for comparison |
> |---|---|---|---|---|
> | `Q8_0` | **0.001242** | **98.24 %** | **2728 MiB** | 0.001035 / 98.30 %, 3049 MiB |
> | **`Q6_K`** | **0.007244** (32 ch) | **95.71 %** (32 ch) | **2119 MiB** | 0.007304 / 95.66 %, 2356 MiB |
> | `Q4_K_M-imx` | 0.043205 | 90.86 % | 1511 MiB | 0.042952 / 90.56 %, 1658 MiB |
>
> **The pick is unchanged and the files are 9-11 % smaller.** The form's perturbation was measured at `KLD 0.000067 / same-top-p 99.510 %`
> and it landed the *harmless* way here: `Q6_K`'s top-1 moved **+0.05** points, so the 0.66-point margin it had is intact (the same
> perturbation on the 7B moved top-1 by −0.52 in the same units and took that model's pick off the bar — this is why each model is measured,
> not rescaled). `Q4_K_M-imx` still misses both halves exactly as before, so "smallest usable" is still a bar-missing file rather than a pick
> that clears. **All three arms store the adapter pairs as F16**, not as the arm's quant type — checked by reading each produced file, because a
> quantizer that re-types a `[2560,128]`/`[128,20480]` pair would change the model at that precision even with an identical ref.

Reference: BF16 GGUF (5732 MiB, deduped converter at fork `6e42864a6`) converted from `Zyphra/Zamba2-2.7B-instruct@699f349b3636`
(KLD of that BF16 against the transformers reference 0.0030, pi-zarch). Every arm below is quantized from that file and measured
identically: 16 × 512-token windows of `wikitext-2-raw/wiki.test.raw`, `llama-perplexity --kl-divergence` against the BF16 logits,
**CPU** (`GGML_HRX=OFF`, Release, `-t 4`), 2026-11-04. **The size column is the pre-#1036 layout** (848 tensors): KLD and same-top-1
are ratios against that same ref and did not move when the ref was deduped — measured, see the family summary — but the per-arm
sizes would shrink with it, and only the three published picks were re-quantized to get a real number.

| arm | PPL | mean KLD | same-top-1 | size |
|---|---|---|---|---|
| BF16 (reference) | 16.79 | 0 | 100.00 % | 7307 MiB |
| **Q8_0** | 16.81 | **0.001229** | **98.02 %** | 3886 MiB |
| Q6_K | 16.70 | 0.008799 | 94.85 % | 3002 MiB |
| Q5_K_M | 17.04 | 0.018481 | 93.41 % | 2568 MiB |
| Q4_K_M | 17.30 | 0.056235 | 89.44 % | 2159 MiB |
| Q4_K_M-imx | 17.20 | 0.044474 | 91.08 % | 2159 MiB |
| IQ4_XS | 17.69 | 0.062797 | 88.16 % | 1993 MiB |
| IQ4_XS-imx | 17.37 | 0.052149 | 89.31 % | 1970 MiB |
| Q4_0 | 17.80 | 0.082047 | 87.11 % | 2081 MiB |
| Q3_K_M | 19.40 | 0.213974 | 79.24 % | 1697 MiB |
| IQ3_XXS-imx | 19.64 | 0.243910 | 77.89 % | 1406 MiB |
| IQ2_XXS-imx | 30.72 | 0.817385 | 59.58 % | 1009 MiB |

The IQ3_XXS and IQ2_XXS formats cannot be produced without an imatrix on this model, so they exist only as `-imx` arms —
and both of them are **8× and 27× outside the KLD limit** (0.244, 0.817) for a 1.0-1.4 GiB file: past ~3 bits this model does
not degrade gracefully, it stops being the same model.
Unlike BlackMamba, **this architecture has zero quantizer fallbacks in every arm** (`type-census.txt`): the labels above
are exactly what the files contain.

### What these numbers say

- **The cliff is between Q5_K_M and Q4_K_M: KLD 0.0185 → 0.0562, a factor of 3.0, for 16 % less size.** Above it Q6_K is
  0.0088; below it every 4-bit-class arm lands in 0.044-0.082 and `IQ4_XS`/`Q4_0` are *worse* than `Q4_K_M` despite being
  smaller. The same shape as ZAYA1, where the finer-grained 4-bit formats were also dominated — so it is a property of these
  graphs, not of one model.
- **The imatrix buys ~20 % of the error for free at 4 bits**: `Q4_K_M` 0.0562 → 0.0445 and `IQ4_XS` 0.0628 → 0.0521 at the
  same or smaller size. It does not move the model across the bar, though: 0.0445 is still 1.5× the 0.03 limit and 91.1 %
  top-1 is 4 points short.
- **`Q6_K` is the sweet spot, decided on a 32-chunk measurement rather than on rounding.** At 16 chunks it was 0.008799 /
  94.853 % — 0.15 points under the top-1 bar, which is inside the noise of an 8192-token run, so it was re-measured over
  32 × 512 = 16384 tokens instead of being called either way:

  | Zamba2-2.7B-instruct, 32 × 512 windows (16384 tokens) | PPL | mean KLD | same-top-1 | size |
  |---|---|---|---|---|
  | BF16 (reference) | 12.17 | 0 | 100.00 % | 5732 MiB |
  | Q8_0 | 12.17 | 0.001035 | 98.30 % | 3049 MiB |
  | **Q6_K** | 12.11 | **0.007304** | **95.66 %** | **2356 MiB** |

  It crosses in the same direction the 16-chunk number pointed, so **`Q6_K` is the pick: 23 % smaller than `Q8_0` and 4.6×
  inside the KLD limit** — the only arm in this model that is both under the bar and meaningfully below 8-bit. `Q8_0` keeps
  the max-quality slot (0.001035). Those two sizes are the deduped figures (2356 / 3049 MiB) because both were re-quantized on the
  new ref; the rest of the table above is the pre-#1036 layout, whose KLD/top-1 stand. The base moved 16.7937 → 12.1670 PPL between
  the two window sets, which is the window effect and the reason this table ranks by KLD and same-top-1 and never by PPL.
- PPL cannot see any of this: 16.70 → 17.80 (6.6 %) spans a 67× range of KLD.

## Zamba2-7B-Instruct: the quantization measurement (8 of 8 plain arms)

> **2026-11-06 23:2x — cause 2 superseded this model's pick, and the numbers below are now the history of a layout that no longer exists.**
> Fork #218 stores the per-layer MLP adapter low-rank and applies `base + B*(A*x)` in the graph. The adapter form is **not** numerically
> identical to the merged form (`Mean KLD 0.000095 ± 0.000003`, `same-top-p 99.755 ± 0.077 %` over 16 × 512 windows, build `10689 (21c814079)`),
> so nothing transferred and every arm below was re-measured from the adapter ref (`ref/Zamba2-7B-Instruct-bf16-adapter.gguf`,
> sha256 `58b26e9a…`, 786 tensors, 13 rank-128 pairs):
>
> | arm | mean KLD | same-top-1 | size | old form, for comparison |
> |---|---|---|---|---|
> | `Q8_0` | **0.001669** | **98.60 %** | **7515 MiB** | 0.001628 / 98.58 %, 8558 MiB |
> | **`Q6_K`** | **0.010430** | **96.59 %** | **5829 MiB** | 0.010801 / 96.45 %, ~6842 MiB |
> | `Q5_K_M` | 0.019383 (32 ch) | **94.79 % (32 ch)** | 4938 MiB | 0.019361 / **95.03 %**, 5577 MiB |
> | `Q4_K_M` | 0.089322 | 89.88 % | 4100 MiB | 0.088160 / 89.93 %, ~6146 MiB |
>
> **The sweet spot moved up a tier and the old pick fell off the bar.** KLD is unchanged within noise (the form's own 9.5e-5 is a rounding
> difference, not a quality one) and files are **11-12 % smaller**, but same-top-1 drops 0.23-0.34 points — and `Q5_K_M` cleared the 95 % half by
> **0.025 points**, so 0.21 points below the bar is the outcome, not a rounding argument (32 × 512 windows). **`Q6_K` is the pick**: 5829 MiB,
> 0.010430 KLD, 96.59 % top-1, 1.59 points of margin — and since `Q5_K_M` now misses, `Q6_K` is *also* the smallest arm that clears either half.
> `Q8_0` remains max quality. The 16-chunk `Q5_K_M` row (0.025365 / 94.681 %) and the 32-chunk one agree that it misses; both are in
> `quant/Zamba2-7B-Instruct/results-adapter{,-32ch}.tsv`. **And the 4-bit tier is not close**: `Q4_K_M` is 0.089322 / 89.88 % — 3.0× the KLD limit
> and 5.1 points short on top-1 — so the step from `Q5_K_M` to `Q4_K_M` is 4.6× in KLD and the cliff sits between 5 and 4 bits, one tier
> *lower* than where the pick moved. The whole curve is now measured in this layout, which is what makes the `Q6_K` pick a measurement rather
> than a rescaling of the old one.
>
> Everything from here to the next section heading is the **merged-form** measurement, kept because it is what the deleted bytes measured and
> because the difference between the two is the finding.

Reference: BF16 GGUF (16094 MiB, deduped converter at fork `6e42864a6`) converted from `Zyphra/Zamba2-7B-Instruct@3146688c5286`
(3 shards, SHA256SUMS-verified) with `convert_hf_to_gguf.py --outtype bf16`, sha256 `78dd8a76…`. Same harness as above, **CPU**,
2026-11-04, base run 1210 s. **The size column below is the pre-#1036 layout** (848 tensors); the picks' sizes in the 32-chunk table
are re-measured on the deduped ref.

| arm | PPL | mean KLD | same-top-1 | size |
|---|---|---|---|---|
| BF16 (reference) | 8.50 | 0 | 100.00 % | 20946 MiB |
| Q6_K | 8.51 | 0.010801 | 96.45 % | 8601 MiB |
| **Q5_K_M** | 8.66 | **0.026074** | **95.03 %** | 7336 MiB |
| Q4_K_M | 9.02 | 0.088160 | 89.93 % | 6146 MiB |
| IQ4_XS | 9.19 | 0.104936 | 88.73 % | 5684 MiB |
| Q4_0 | 9.36 | 0.129600 | 87.62 % | 5931 MiB |
| Q3_K_M | 11.29 | 0.368958 | 77.82 % | 4810 MiB |
| Q8_0 | 8.51 | 0.001628 | 98.58 % | 11135 MiB |

- *(Superseded 2026-11-06 23:4x — kept as the merged-form record.)* **This model's cliff is the 2.7B's, not Zamba-7B-v1's**: `Q5_K_M`
  still clears the bar and `Q4_K_M` is 2.9× the KLD limit. The
  name says "7B", the quantizer behaviour says it is the 2.7B's sibling — both are the shared-attention/shared-FFN graph — and the
  tensor census predicts exactly that (81 Mamba2 blocks + 13 shared blocks). Zamba-7B-v1, which shares nothing, needed a full extra bit.
- **`Q5_K_M`'s 0.025-point margin was answered by a longer run, not by rounding.** Re-measured over 32 × 512 = 16384 tokens
  against a fresh 32-chunk base (the same treatment the 2.7B's `Q6_K` got), and then **again on the deduped ref after #1036**:

  | Zamba2-7B-Instruct, 32 × 512 windows (16384 tokens) | PPL | mean KLD | same-top-1 | size |
  |---|---|---|---|---|
  | BF16 (reference) | 7.64 | 0 | 100.00 % | 16094 MiB |
  | **Q5_K_M** | 7.73 | **0.019361** | **95.03 %** | **5577 MiB** |
  | Q6_K | 7.66 | 0.008048 | 96.64 % | 8601 MiB (pre-dedup) |

  **`Q5_K_M` is the pick**: inside both halves of the bar (0.019361 is 1.55× inside the KLD limit, 95.03 % top-1), and it is
  simultaneously the smallest arm that clears — there is no 4-bit tier under it. **After the #1036 dedupe the same pick is 5577 MiB,
  24 % below its own pre-dedup 7336 MiB**, and it no longer needs the tie-break margin to be interesting: it is a 5.4 GiB 7B-class file
  that clears both halves. `Q6_K` is the better-quality alternative (0.008048 / 96.64 %, 8601 MiB pre-dedup), and `Q8_0` takes the
  max-quality slot (0.001628, 8558 MiB on the deduped ref). The base moved 8.50 → 7.64 PPL between the two window sets, which is the
  window effect and the reason this table ranks by KLD and same-top-1, never by PPL.
- Both tie-break arms **re-quantized to bit-identical sha256** in that second run (both files had been released and were rebuilt from
  the BF16), so the two rows describe the same bytes as the 16-chunk rows and either file can be published under the measured number.
- The 16-chunk and per-arm sizes above are the **pre-#1036 layout** and stay as they were measured; the picks were re-quantized after the
  dedupe and carry their real size (**Q5_K_M 5577 MiB, Q8_0 8558 MiB**). The KLD/top-1 columns did not move with the layout, which is
  the point of the identity check in the family summary.

## Zamba-7B-v1: the quantization measurement (11 of 11 arms)

Reference: BF16 GGUF (13821 MiB) converted from `Zyphra/Zamba-7B-v1@b8c9e7ede2f6` (784 tensors, KLD of that BF16 against the
transformers reference 0.0022, pi-zarch). Same harness as above, **CPU**, 2026-11-04.

| arm | PPL | mean KLD | same-top-1 | size |
|---|---|---|---|---|
| BF16 (reference) | 20.93 | 0 | 100.00 % | 13821 MiB |
| **Q8_0** | 20.97 | **0.000962** | **97.97 %** | 7484 MiB |
| **Q6_K** | 20.96 | **0.002685** | **97.16 %** | 6943 MiB |
| Q5_K_M | 20.90 | 0.010715 | 94.49 % | 5281 MiB |
| Q4_K_M | 20.98 | 0.020940 | 92.40 % | 4739 MiB |
| Q4_K_M-imx | 20.93 | 0.016708 | 93.46 % | 4739 MiB |
| IQ4_XS | 20.93 | 0.042551 | 89.61 % | 4099 MiB |
| IQ4_XS-imx | 20.46 | 0.030809 | 91.01 % | 4091 MiB |
| Q4_0 | 20.93 | 0.053882 | 88.14 % | 4161 MiB |
| Q3_K_M | 20.72 | 0.078644 | 86.84 % | 3895 MiB |
| IQ3_XXS-imx | 20.60 | 0.090106 | 84.93 % | 3754 MiB |
| IQ2_XXS-imx | 22.06 | 0.227748 | 77.38 % | 3487 MiB |

- **The cliff is a full bit higher than the 2.7B's, and it is Q6_K → Q5_K_M**: 0.00269 → 0.01072 is **4.0× the KLD for 24 %
  less size**, and same-top-1 drops 97.16 → 94.49 %, i.e. through the bar in the same step. The 2.7B only falls over at
  Q5_K_M → Q4_K_M. On this model **Q6_K is therefore both the sweet spot and the smallest arm that clears the bar**: no 4-bit
  arm comes close, so there is no "smallest usable" tier below it at all (the 2.7B and BlackMamba each have one).
- **Q4_K_M is the arm that would be over-recommended on KLD alone**: 0.02094 is comfortably inside the 0.03 limit while
  same-top-1 is 2.6 points *under* the 95 % bar — the third model in a row where the two halves of the bar disagree, and the
  widest disagreement so far. IQ4_XS misses both halves (0.0426 / 89.6 %).
- **The arm labels are only partly exact, and I had this wrong first.** The census shows this model is *not* 256-divisible:
  `ssm_in`/`ssm_dt`/`ssm_x` have `ncols` 3712 (÷256 = 128 remainder) and 232 (not even ÷32). So every K-quant and IQ arm is a
  *mixed* file: **167** tensors fall back to a legacy type (`q8_0` under Q6_K, `q5_1` under Q5_K_M, `q5_0` under Q4_K_M, `iq4_nl`
  under IQ4_XS / IQ3_XXS-imx / IQ2_XXS-imx) — that is 76 `ssm_in` + 76 `ssm_x` + 13 `ssm_mix` plus one `ffn_gate`/`ffn_up` pair —
  and **76** `ssm_dt` tensors stay `f16` in all arms, including Q8_0. The census catches exactly 243 fallback lines per arm and
  matches the quantize logs line-for-line.
- **The direction of that fallback is the reassuring part**: under Q6_K the 167 tensors go *up* to q8_0, so the 0.002685 belongs
  to the file we would actually ship and a hypothetical pure Q6_K would be marginally *worse*, not better. Q8_0 is the only arm
  that is exactly what it says (708 tensors q8_0 + 76 f16). This differs from BlackMamba in kind: there a "Q6_K" was a *pure
  q8_0 file* (the label hid the format); here the substitution is partial, on the recurrent-path tensors, and is stated on the row.
- **The imatrix lever reproduces on the 7B at the same magnitude** — the fourth model in a row where it does: `Q4_K_M` 0.020940 →
  `Q4_K_M-imx` 0.016708 (**-20.2 %** of the error) and `IQ4_XS` 0.042551 → `IQ4_XS-imx` 0.030809 (**-27.6 %**), both at
  identical size. It still does not cross the bar: the best 4-bit-class 7B arm is 2.7 % outside the KLD limit and 4 points outside
  top-1, so the free ~20-28 % is real but it is not 3×.
- **Below 4 bits this model degrades honestly, which is not what the other two SSM models do.** `IQ3_XXS-imx` (3754 MiB) is a
  *real* 3-bit file — 0.090106 / 84.93 %, **3.3× the IQ4_XS-imx error for 8 % less size** — and `IQ2_XXS-imx` (3487 MiB) is a real
  2-bit file at 0.227748 / 77.38 %, 7.4×. The census explains why the sizes really do fall: only those 167 tensors fall back to
  `iq4_nl` (and 76 `ssm_dt` to `f16`) while every other matrix takes the **nominal** type — `ffn_down`, `attn_q/k/v` appear in no
  warning line at all. So on this model the IQ arm names are truthful — the opposite of BlackMamba,
  where all 108 matrices fall back and a nominal `IQ4_XS` file is 108/108 `iq4_nl`.
- **That contrast is the real cross-model lesson, and it is why a label can never be taken as evidence in either direction.** The same
  string `IQ4_XS` means "a genuine 4.25 bpw file" on Zamba2 (zero fallbacks), "a 4.5 bpw `iq4_nl` file wearing the wrong name" on
  BlackMamba (108/108 fallbacks), and "4.25 bpw for the bulk, `iq4_nl` on 167 of its tensors" on the 7B. One row of the census
  decides which case a given file is; no amount of measuring tells you.

## How to run

```sh
1bit serve -m ZAYA1-8B-Q8_0.gguf --device hrx
```

## Published conversions

The organisation publishes one repo per model under `1bit-MONSTER/<Model>-GGUF`
(`ZAYA1-8B-GGUF`, `ZR1-1.5B`, `Zamba2-*`, `Zamba-7B-v1`, `BlackMamba-2.8B`, `ZAYA1-VL-8B`).
A card carries the base model and revision sha, the converter commit, the measured table
above, and the run command. Only finished, measured arms are uploaded, with `SHA256SUMS`,
and the upstream licence copied alongside.

## Reproducing

`~/models/zyphra-orig/zq-sweep.sh <tag> <bf16.gguf> <chunks> <arms...>` quantizes and
measures; `zq-gpu-base-check.sh` records base logits on a chosen backend and runs the
self-check; `zq-tensor-diff.py <ref.gguf> <arm.gguf>` verifies an arm tensor-by-tensor
before its numbers are trusted (cos ≈ 1.0 on untouched tensors, ~0.99 on quantized ones).
The code slice in the section above is the same script with `WIKI=calib/code-slice.txt`, its own `BASE_FILE` and its own `RESULTS_TSV`
(`zq-code-slice-pass.sh` drives that per model and refuses to start a model's base run when the BF16 would not fit in free RAM, or an
arm when quantizing it would take the disk under the floor); `zq-code-slice.sh` regenerates the corpus itself, deterministically, from a
checkout.

Publishing: `zq-publish-rebuild.sh` re-quantizes a measured arm with the tree that produced its row, asserts the fresh quantize log's
**build id** matches the row's, compares the sha256 to the row and the type census to the stored one, and only then keeps the file — so
"the file you upload is the file that was measured" is checked rather than assumed. All nine publish digests reproduce exactly.

Throughput: `zq-perf-all.sh` quantizes one arm, benches it and releases the file; it refuses to
bench a file whose sha256 does not match its KLD row, and it takes the device through `gpu-hold`
so a bench cannot run beside another lane's job. `zq-pair-bench.sh` and `zq-triple-bench.sh` are the same thing for two and three
arms inside **one** ticket, which is how the rows in *Throughput on Strix Halo* should be compared: two arms benched in two windows
came back a factor of 2.6 apart on identical files, and together inside one window they agree within noise.
Both need `IREE_HAL_AMDGPU_LIBHSA_PATH` pointing at TheRock's `libhsa-runtime64.so.1`: without it the
HRX backend registers **no** devices and `llama-bench` exits in 0 s with `invalid device: HRX0`,
which is easy to miss because the process still exits 0 under some wrappers.
