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
# CPU reference

`engine/backends/cpu/` is the fp32 oracle the NPU and GPU backends are tested
against. It dequantizes every weight to fp32 at load time and runs one token at
a time in fp32.

It accepts only architectures that have passed the golden test below. Any other
architecture is refused by name. Loading also fails if the GGUF holds a tensor
the forward pass does not use, because an unused tensor means the model has
semantics this code does not implement.

| Architecture | Weight types | Golden test |
|---|---|---|
| `qwen3` | F32, F16, BF16, Q8_0 | Qwen3-0.6B: pass |

## Golden test: Qwen3-0.6B

Reference: HF transformers, fp32, eager attention, on the same checkpoint.
The test feeds the 36-token sequence (12 prompt tokens plus the reference's 24
greedy tokens) teacher-forced and compares the next-token logits at every position.

| Metric | Result | Gate |
|---|---|---|
| Argmax agreement | 36 / 36 | all |
| Worst per-position KL(ref ‖ cpu) | 1.3e-10 | ≤ 1e-6 |
| Worst max \|Δlogit\| | 1.0e-4 | reported only |

To confirm the test can fail, two deliberately broken builds were run:

- **Wrong RoPE layout** (Normal instead of Neox): 17 / 36 argmax mismatches, worst KL 7.9.
- **K-norm skipped**: 34 / 36 mismatches, worst KL 24.9.

Both failed the gate.

Inputs, as recorded in `tests/golden/qwen3-0.6b-capitals/meta.json`:

- Checkpoint `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- The GGUF is converted with llama.cpp `convert_hf_to_gguf.py --outtype bf16` (llama.cpp `e71b805`).
  Its sha256 is `33d6f6c9f0dd21dd3d9c99f514498ea594119508e538aa36773993a3ffd39aff`.
- The golden logits are 21.9 MB, so they are not committed. They are regenerated
  by the script below; `meta.json` records their sha256.

Measured on Strix Halo (32 threads): 0.6 s load, 24 tok/s teacher-forced.

## Reproduce

```bash
python tools/golden/make_golden.py --model Qwen/Qwen3-0.6B \
    --prompt "The capital of France is Paris. The capital of Japan is" \
    --n-gen 24 --out golden/qwen3-0.6b-capitals
python llama.cpp/convert_hf_to_gguf.py <hf_snapshot_dir> --outtype bf16 --outfile Qwen3-0.6B-BF16.gguf

cmake -B build -DENGINE_GOLDEN_MODEL=$PWD/Qwen3-0.6B-BF16.gguf \
               -DENGINE_GOLDEN_DIR=$PWD/golden/qwen3-0.6b-capitals
cmake --build build && ctest --test-dir build --output-on-failure
```

## Not done yet

- **Tokenizer wiring.** The tokenizer has landed (docs/tokenizer.md), but this test still feeds ids.
- **Quantized types beyond Q8_0** (Q4_K, Q6_K and others). Each needs its own dequant test first.
- **A longer-context golden** (hundreds of positions) to cover RoPE at larger angles.
