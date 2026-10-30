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
# The official converter's per-architecture configs, as data

Six JSON files copied verbatim from **ROCm/FLM_Q4NX_Converter**, the official gguf → Q4NX
converter, at commit `d1d5232` ("feat: update converter for q4_k"):

    configs/qwen3.json            dense Qwen3 (0.6B/1.7B/4B/8B)  <- what tools/gguf_to_q4nx.py packs
    configs/qwen3.5_{0.8b,2b,4b,9b,a3b}.json   hybrid GDN (qwen3_5), the family Qwen3.8-27B is in

They are carried as **reference data**, not code: they are the authority for the per-tensor dtype
(`default_tensor_type` per `name_map` entry), `row_block_size` / `col_block_size`, `parallel_size`,
`keep_block_in_2D`, and the `name_map` that says what an architecture's tensors become. Nothing in
the build reads them, and `tools/gguf_to_q4nx.py` does not start consuming them — its tile order is
what `npu/q4nx.cpp` decodes, and the official one differs (see `docs/npu.md`, "Cross-check against
the official converter").

The other eleven upstream configs (`gemma3`, `gemma4`, `gpt-oss`, `lfm2`, `llama`, `nanbeige`,
`phi4`, `qwen2`, `qwen2vl`, `qwen3vl`, `qwen3.5_omni`) are for architectures this engine does not
serve on the NPU and are deliberately not copied. Fetch any of them, or the current revision of
these, with:

    git clone https://github.com/ROCm/FLM_Q4NX_Converter   # pin: d1d5232

What the six say, since this is the part that matters for a GDN model:

| config | `default_tensor_type` | `parallel_size` | `keep_block_in_2D` |
|---|---|---|---|
| `qwen3.json` | Q4_1 | 16 | false |
| `qwen3.5_0.8b/2b/4b/a3b` | Q4_K | 32 | true |
| `qwen3.5_9b` | Q4_1 | 32 | true |

`row_block_size` 32 and `col_block_size` 256 everywhere — the same 32x256 tile this engine's packer
writes. The `qwen3.5_*` configs additionally carry the hybrid-GDN name map: prefill-mode `.bf16`
twins for `ssm_alpha_proj` / `ssm_beta_proj` beside the Q8_0 decode copies, `ssm_out_proj` at Q4_K,
and `ssm_a` / `ssm_norm` / `ssm_conv1d` / `ssm_dt.bias` at BF16. That table is the reason a GDN model
cannot be packed by `tools/gguf_to_q4nx.py` yet.

Licence: Apache-2.0, © Advanced Micro Devices, Inc. and the FLM_Q4NX_Converter contributors; the
attribution block is in `NOTICE` at the repository root. Do not edit these files — re-copy them from
upstream, so a diff stays a diff.
