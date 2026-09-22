# Qwen3 tokenizer golden

- `vocab.gguf`: tokenizer-only GGUF (no weights) from `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, via llama.cpp `convert_hf_to_gguf.py --vocab-only`
  (llama.cpp `e71b805`). sha256 `2892de8d145f99f886b7a6a141afb192e97ce691bb864af6a116686cf2d60e50`.
- `cases.tsv`: 481 cases (81 fixed + 400 random, seed 1234) from HF `tokenizers` 0.22.2 /
  transformers 5.13.0.dev0, written by `tools/golden/make_tokenizer_golden.py`.

Regenerate:

```bash
python tools/golden/make_tokenizer_golden.py --model Qwen/Qwen3-0.6B --out cases.tsv
python llama.cpp/convert_hf_to_gguf.py <hf_snapshot_dir> --vocab-only --outfile vocab.gguf
```
