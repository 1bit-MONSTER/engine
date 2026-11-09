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
token-equivalents). This page is the third arm: **the OGA pack itself**, and why it cannot carry a drafter measurement yet.

## The route has three gates, and a pack dies at a different one each time

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

The dims are right (that is Qwen2.5-0.5B), so the container parses. What fails is the **quantized embedding**:
`npu/forward/src/q4nx_forward.cpp:647` hands the tensor to `dequant_any()`, and `dequant_any` (`:181`) accepts exactly three chunk
sizes — our 5120-byte q4_1 tile (`Q4NX_TILE_BYTES`), Q4_K, or Q8 — deriving the chunk size from `data_size / ntiles`. The OGA import
writes the embedding as **q4_1 in the vendor's layout**, whose per-tile byte count is not one of the three, so the call falls through
to the packed/tensor decoders and returns non-zero. That is a precise, repeatable diagnosis, not a guess: the same message appears
with the same tensor on every attempt.

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
