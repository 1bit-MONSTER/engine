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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence, stored as Markdown because the repository's CI requires a copyright notice on
     every file and a raw .txt log cannot carry one. Verbatim output. -->

## The cheap probe: align=256 + compaction

```
# engine#315: cheap reproducer for the wrong-output class at the pin (no GLM, no regime, no neighbours)
# strixhalo, 2026-10-09 23:06-23:15. Model Qwen3-0.6B-Q4_K_M on HRX0, -fa on, -c 4096, greedy, seed 1,
# cache_prompt=false, 180 identical requests per configuration, in ONE server process.
# Build: 315fp-build (the pin e44c9d01a plus the nan-dump fingerprint).

## per-token logits-hash divergence (call index within a request)

| config | requests | requests with NaN | distinct hash at token 0 | token 1 | token 2 | token 7 | token 23 |
|---|---|---|---|---|---|---|---|
| default align, quiet | 180 | 0 | 1 | 1 | 1 | 1 | 1 |
| default align + compaction | 180 | 0 | 1 | 1 | 1 | 1 | 1 |
| align=256 + compaction | 180 | 0 | 2 | 3 | 7 | 35 | 121 |

## the same runs as produced outputs (distinct completions out of 180)
- no compaction: 120 requests, 1 distinct
- default align, quiet: 180 requests, 1 distinct
- default align + compaction: 180 requests, 1 distinct
- align=256 + compaction: 180 requests, 95 distinct

## dispatch selection differs with the knob (GGML_HRX_LOG_DISPATCH=1, 12 requests, ctx 1024)
- align=256: 21 distinct dispatches
## dispatch selection differs with the knob (GGML_HRX_LOG_DISPATCH=1, 12 requests, ctx 1024)
- align=256: 21 distinct dispatches
    flash-attention kernels: ['common.flash_attention_f32_f16_wmma (single)']
    decode matmul kernels:   ['common.mul_mat.f32_f32_decode (fused)', 'kquant.mul_mat.decode_f32 (fused)', 'kquant.mul_mat_add.decode_f32 (fused)']
- align=4096: 21 distinct dispatches
    flash-attention kernels: ['common.flash_attention_decode_split_next_q8 (single)', 'common.flash_attention_f32_f16_wmma (single)']
    decode matmul kernels:   ['common.mul_mat.f32_f32_decode (fused)', 'kquant.mul_mat.decode_f32 (fused)']

## reading
The knob does not merely misalign a buffer: it changes which kernels the dispatcher selects.
At align=256 the decode-split flash attention (common.flash_attention_decode_split_next_q8)
never runs, and kquant.mul_mat_add.decode_f32 (fused) does; at the shipped align=4096 both are
the other way round. Divergence begins at the first decoded token (token 0 has 2 distinct
hashes, i.e. it is already wrong for some requests) and compounds, which is the locality
engine#315 records for its own faults. All sampled logits stay finite (nan=0), so this shape
is Face B, not Face A.
```
