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

     engine#315 evidence: the pin's decode dispatch inventory for GLM-4.7-Flash, captured the moment
     the box freed on 2026-10-10. One server, GGML_HRX_LOG_DISPATCH=1, two short completions, then
     teardown; no neighbours, so this is not an in-regime arm - it is the dispatch inventory only. -->

# GLM-4.7-Flash decode dispatch inventory at the pin

Server: `~/wt/315fp-build/bin/llama-server -m ~/models/GLM-4.7-Flash-Q4_K_M.gguf -dev HRX0 -ngl 99
-fa on -c 2048 -np 1`, `GGML_HRX_LOG_DISPATCH=1`, two requests (`n_predict=16`), GTT 26.5 -> 44.3 GiB.
`GGML_HRX_LOG_DISPATCH` prints each registration name the first time it matches, so this is the set of
kernels the pin dispatches for this model, not a per-call count.

```text
common.argsort_rows_f32
common.binary_f32
common.clamp_f32
common.concat_dim0_f32
common.cont_f32
common.flash_attention_f32_f16_wmma
common.gather_add_f32
common.get_rows.f32
common.get_rows_small_f32
common.mul_mat.f32_f32_decode
common.mul_mat.f32_f32_wmma
common.mul_mat_id.decode_f32_wave64
common.mul_mat_id.f32_f32_wmma
common.mul_mat_id_swiglu.f32_f32_wmma
common.mul_mat_postops.f32_f32_wmma
common.mul_mat.q4_k_q8_1_x4_prefill
common.mul_mat.q5_k_iq4_xs_q8_1_x4_prefill
common.mul_mat.q6_k_aligned_skinny
common.mul_mat.q6_k_i8_prepacked
common.mul_mat.q6_k_prefill_wave32
common.mul_mat.q6_k_token1_final_projection_q8
common.mul_mat_swiglu.f32_f32_wmma
common.mul_mat_unary.f32_f32_wmma
common.rmsnorm_binary_f32
common.rmsnorm_binary_q8_1_x4
common.rmsnorm_binary_symmetric_i4_k32
common.rmsnorm_f32
common.rope.f32
common.scale_f32
common.set_rows
common.sum_rows_f32
common.unary_f32
kquant.mul_mat_add.decode_f32
kquant.mul_mat.decode_f32
kquant.swiglu.decode_f32
```

## What this settles

* **The `q6_k_token1_shortlist` path is never dispatched for GLM.** The inventory has
  `q6_k_aligned_skinny`, `q6_k_i8_prepacked`, `q6_k_prefill_wave32` and `q6_k_token1_final_projection_q8`,
  but not `q6_k_token1_shortlist`. So `ggml_mul_mat_q6_k_packed_selected_refine_token1` - the kernel whose
  `exact_output` binding is read while the manifest declares it `write`-only (mechanism doc section 37) -
  is not on this model's decode path. That finding is a genuine defect in the hazard model's input, but it
  is **not** engine#315's mechanism, and section 37 is closed as not-applicable to this issue.
* The head path for the decode is `common.mul_mat.q6_k_token1_final_projection_q8`; the routed-FFN path is
  `mul_mat_id.decode_f32_wave64` / `mul_mat_id.f32_f32_wmma` / `mul_mat_id_swiglu.f32_f32_wmma`; the
  attention kernel seen here is `flash_attention_f32_f16_wmma`.
* Caveat, stated so the list is not over-read: these prompts were ~6 tokens, so the KV is tiny and the
  *decode-split* flash-attention kernel (`..._decode_split_..._next_q8`) does not appear - it needs a longer
  KV than this run had. The LM-head conclusion is unaffected (the head is per token and its gate does not
  depend on KV length), but the attention part of this inventory is the short-prompt case only.
