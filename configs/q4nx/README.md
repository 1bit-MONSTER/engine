# The official converter's per-architecture configs, as data

These 17 JSON files are copied verbatim from **ROCm/FLM_Q4NX_Converter**, the official
gguf → Q4NX converter, at commit `d1d5232` ("feat: update converter for q4_k"). They are
carried here as **reference data**, not as code: `tools/q4nx_diff.py` and the notes in
`docs/npu.md` use them to say what the shipped format asks for (per-tensor dtype,
`row_block_size` / `col_block_size`, `parallel_size`, `keep_block_in_2D`, and the
`name_map` that names what a given architecture's tensors become).

Nothing in the build consumes them, and `tools/gguf_to_q4nx.py` does not read them: that
packer stays qwen3-only with its tile order hardcoded, because its output is what
`npu/q4nx.cpp` decodes. Treat a change here as a note about the official format, not as a
switch in this engine.

Licence: Apache-2.0, © Advanced Micro Devices, Inc. and the FLM_Q4NX_Converter
contributors. The full attribution, including the licence text pointer, is in `NOTICE` at
the repository root. Do not edit these files; re-copy them from upstream if they need to
move, so a diff stays a diff.

What the table says, for the architectures this engine serves on the NPU:

| config | `default_tensor_type` | `parallel_size` | `keep_block_in_2D` |
|---|---|---|---|
| `qwen3.json` (0.6B/1.7B/4B/8B dense) | Q4_1 | 16 | false |
| `qwen3.5_0.8b.json`, `_2b`, `_4b`, `_9b`, `_a3b` (hybrid GDN) | Q4_K | 32 | true |
| `llama.json`, `phi4.json`, `nanbeige.json`, `gemma3.json`, `gemma4.json` | Q4_1 | 16 | false |
| `qwen2.json`, `lfm2.json` | Q4_0 | 16 | false |
| `gpt-oss.json` | Q4_1 | 16 | true (col_block 128) |

`row_block_size` is 32 and `col_block_size` is 256 everywhere except gpt-oss's 128 columns,
which is the 32x256 tile this engine's packer also writes. The hybrid-GDN configs additionally
carry the prefill-mode `.bf16` twins for `ssm_alpha_proj` / `ssm_beta_proj`, `ssm_out_proj` at
Q4_K, alpha/beta at Q8_0 and `lm_head` at Q8_0 — the dtype table `docs/npu.md` records under
"Cross-check against the official converter", and the reason a GDN model cannot be packed by
`tools/gguf_to_q4nx.py` yet.
