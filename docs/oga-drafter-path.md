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
# A drafter on the OGA-pack route: measured as far as the route allows (engine#1371)

#1371's claim is that the batched verify path is now cheap and greedy-exact, so a drafter on the **OGA-pack** route should pay, and
it asks the drafter lane for acceptance per pass against a bar. Two of the three arms are measured and published
(`docs/amd-drafters.md`: PARD over GGUF, **HRX 1.012×** at 3.095 tokens/round and 0.524 acceptance per offered draft; **CPU
0.508×**, which is structural — a round can only return `1 + accepted ≤ 1 + drafted` while a batched verify costs ~M
token-equivalents). This page is the third arm: **the OGA pack itself**. On 2026-11-08 no pack could carry a drafter measurement, and the
gate table further down records why; on 2026-11-10 one pack serves and the row is measured, in the next section.

## 2026-11-10: a pack serves, and the drafter row is measured

The table below is from 2026-11-08, when no OGA pack reached a logit. Since then the 1.9 reader and the b8 kernel
families landed, and `Qwen2.5-7B-Instruct_rai_1.9.0_npu` imports complete and serves. So the row #1371 was opened for
exists now: **a real drafter, on an OGA pack, against that pack's own costs**.

| | measured |
|---|---|
| target | `Qwen2.5-7B-Instruct_rai_1.9.0_npu`, imported with `--weight-layout b8` (incomplete 0; attention 3584/3584 tiles and MLP 9856/9856 groups byte for byte), served by `forward-serve` on the NPU |
| one plain decode step | **1021.2 ms** (median of 33; 1005.9-1065.5), 0.83-0.94 tok/s per request |
| one batched pass, 5 rows / 8 rows / 16 rows | **1049.1 / 1056.1 / 1073.4 ms** (medians of 4): a verify pass of up to 16 rows costs 1.03-1.05 steps |
| drafter | `qwen2.5-0.5b-instruct-q4_k_m.gguf` on the CPU (`--device none -ngl 0`), 5.8-6.4 ms per drafted token |
| acceptance, k = 4 | **2.963 tokens/round** (160 tokens in 54 rounds; 108 of 216 offered drafts accepted) |
| acceptance, k = 7 | **3.40-3.48 tokens/round** (160 tokens in 46-47 rounds over two replays; 116-117 of 322-329 offered) |

Per prompt at k = 4 the rate is 2.22 (prose), 3.33 (code), 3.33 (a list), 3.33 (a short answer); at k = 7 it is 2.50,
4.44, 4.00 and 3.33-3.64. No prompt accepted everything or nothing, so neither of the tool's two warnings applies.

What that acceptance is worth, from those measured parts:

| round shape | k | round cost | bar (tokens/round to break even) | measured tokens/round | projected |
|---|---:|---:|---:|---:|---:|
| one pass (verify only) | 4 | 1072 ms | **1.05** | 2.963 | 2.82x |
| one pass (verify only) | 7 | 1101 ms | **1.08** | 3.404 | 3.16x |
| two passes (`mtp::run`: verify, then the step that consumes the committed token) | 4 | 2093 ms | **2.05** | 2.963 | 1.45x |
| two passes | 7 | 2122 ms | **2.08** | 3.404 | 1.64x |

**The acceptance column is measured; the "projected" column is arithmetic and is not a measured loop.** No speculative
loop is wired into `forward-serve` (engine#1781 carries that), so nobody has run these rounds end to end on this pack.
What the measurement does settle is the question #1371 asks: on this route the bar is **1.05-1.08 tokens/round for a
one-pass round**, not the 1.17 measured on the 0.5B pack, because here a batched pass of up to 16 rows costs what one
step costs (the device wait is 854-872 ms of either). A 0.5B drafter of the same family clears that bar by 2.8-3.2x on
these prompts, and still clears the two-pass bar.

Three limits of the row, so it is not read as more than it is:

- four prompts of 40 tokens each. It is a first row, not a distribution; code and list text draft better than prose,
  as on the other arms.
- acceptance is counted on the draft tokenizer's ids over the target's recorded **text** (the pair shares the Qwen2.5
  vocabulary, and the tool refuses a recording whose prompt does not tokenize to a prefix of prompt + continuation).
  The drafter is not bit-reproducible between replays: one prompt's k = 7 accept list moved between two runs, which is
  the 3.40-3.48 range above.
- the pass costs were taken at the start of a context (rows 0-15). The host share of a pass is 34-53 ms of ~1050, so
  a longer context moves it little, but that is an expectation and was not measured.

One thing a verify loop on this pack must respect: **a pass of 17 rows costs 6992 ms, 6.5 times a 16-row pass**, because
a chunk wider than 16 rows runs the 128-row kernel family. A verify batch must stay at 16 rows or fewer here.

How it was taken (`amdxdna` error count unchanged across every ticket):

```sh
1bit npu-pack import <pack> -o <out> --weight-layout b8 --kernels <the pack family's kernel set>
# ticket 1 (npu-hold): the target's greedy continuations, through the endpoint forward-serve has
tools/spec_decode_acceptance.py --record-openai http://127.0.0.1:18172 --prompts prompts.json --tokens 40 --out target.json
# no NPU: the drafter proposes from every committed prefix of that recording
tools/spec_decode_acceptance.py --target-recorded target.json --draft http://127.0.0.1:18173 --k 4 \
    --pass-ms 1049.1 --step-ms 1021.2
```

The replay rests on greedy decode being deterministic, and that was checked rather than assumed: the four
continuations were recorded twice, on two separate serves in two tickets, and are identical text for text (4 of 4).

`--pass-ms` and `--step-ms` are the serve's own `[bs] M=<rows>` and `[ph] tok` walls under `NPU_PHASE_PROF=1`; a prompt
of n+1 token ids produces one n-row pass. `tools/oga_pack_probe.sh` still imports without `--weight-layout b8` and
without a kernel set, so it reports this pack as stopping at gate 2; the commands above are the ones that serve it.

## The route has three gates, and a pack dies at a different one each time (2026-11-08)

`tools/oga_pack_probe.sh <pack dir> [port]` runs gates 1-2 inside one `npu-hold` ticket (the server it starts is killed by a trap on
every exit path) and prints which gate bit. Measured on this box, 2026-11-08, kernel `7.4.0-rc1-1bit`:

| pack | gate 1: `npu-pack import` | gate 2: `forward-serve` | outcome |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct_rai_1.9.0_npu` | container written, **242 tensors incomplete** — "fconst: layer blocks not located: fewer than 6 RMSNorm epsilon headers in the first GiB" | not reached | the 1.9 pack renamed the `.fconst` token stream (engine#1410) |
| `Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K` | container written, 48/48 RMSNorm anchors found, 768/768 `.fconst` tiles byte-exact, **73 tensors incomplete** | **`embedding dtype q4_1 but dequant failed`** | the embedding, see below |
| `Llama-3.2-1B-Instruct_rai_1.8.0_npu_4K` | container written, **161 tensors incomplete** | **`missing full ELF /full_i8_QKV_K2048_N3072.elf`** | this pack's shapes have no compiled kernels in this tree |
| `Qwen3-1.7B_rai_1.8.0_npu_4K` | container written, 56/56 RMSNorm anchors, **281 tensors incomplete** | **`missing full ELF /full_i8_QKV_K2048_N4096.elf`** | same, and the shapes that do exist elsewhere on the box are a partial set |

So the OGA route is not blocked by the *drafter*: it is blocked by the base not loading. A drafter measurement on it would be a
number about a model that never produced a logit.

## The 1.8.0 0.5B pack's actual blocker, with the line

```
[INFO]  Model loaded: 511 tensors, 24 layers
  [forward] dims from …/config.json: H=896 NL=24 NH=14 NKV=2 HD=64 IM=4864 NV=151936
1bit forward-serve: forward init failed: embedding dtype q4_1 but dequant failed
```

The dims are right (that is Qwen2.5-0.5B), so the container parses. What fails is the **quantized embedding**: the embedding branch of
`Q4nxNpuForward`'s init calls `dequant_any()` (`npu/forward/src/q4nx_forward.cpp`), and that function accepts exactly three chunk sizes
— our 5120-byte q4_1 tile (`Q4NX_TILE_BYTES`), a Q4_K chunk, or a Q8 chunk — deriving the chunk size from `data_size / ntiles`; its
caller raises `"embedding dtype … but dequant failed"` when the tensor is none of them. The OGA import
writes the embedding as **q4_1 in the vendor's layout**, whose per-tile byte count is not one of the three, so the call falls through
to the packed/tensor decoders and returns non-zero. That is a precise, repeatable diagnosis, not a guess: the same message appears
with the same tensor on every attempt. (Line numbers are deliberately not quoted: this page has already been wrong once by naming a
line that moved. The function names are stable, and `git log -S` finds them.)

Two ways out, and they belong to different lanes: teach the import to write the embedding in a chunk our forward already reads (the
import is where the vendor layout is known), or teach `dequant_any` the vendor's q4_1 embedding layout. Either unblocks gate 2; only
then does gate 3 — a drafter — mean anything.

## What the probe gives the next person

```sh
npu-hold -t 4 -l "oga probe" -- tools/oga_pack_probe.sh models/oga/<pack dir>
# exit 0 the pack served (gate 3 is worth doing)   3 import refused/incomplete   4 forward init failed   5 no health
```

and a verdict line that names the gate rather than leaving "it does not work". The harness for gate 3 already exists and is in main:
`tools/spec_decode_acceptance.py` (token-exact, per-round accept list) plus `tools/spec_decode_acceptance_hrx.sh` if the drafter is
served on the device, and the bar to compare against is in `docs/amd-drafters.md` — 1.17 tokens/round for a one-pass loop, ~4 for
`mtp::run`'s two-pass round.
