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
tags: quantization, ternary, rocmfpx, unsloth

# Smaller files: Unsloth, ROCmFPX and a ternary 27B

A smaller model file decodes faster, because decode is limited by how many bytes the GPU reads per token. The question is what each format costs in accuracy. We measured three families of small files on Strix Halo against their full-precision originals: KL divergence (how far the model's predictions move) and how often the top predicted token stays the same.

## Qwen3.8-27B: Unsloth against ROCmFPX

Every file below is quantized from the same BF16 checkpoint, the ROCmFPX ones with Unsloth's own importance matrix so the comparison is fair.

| File | Size | KL divergence | Same top token | Decode (Vulkan) |
|---|---|---|---|---|
| UD-Q4_K_XL (Unsloth, the default) | 16.4 GiB | **0.008** | **95.3%** | 11.9 tok/s |
| UD-IQ4_XS (Unsloth) | 13.3 GiB | 0.019 | 93.3% | 15.0 |
| **UD-Q3_K_XL (Unsloth)** | 12.2 GiB | 0.028 | 92.1% | **15.8** |
| ROCmI4 (ROCmFPX) | 13.9 GiB | 0.036 | 91.3% | 13.5 (ROCm) |
| ROCmFP4 (ROCmFPX) | 13.8 GiB | 0.045 | 89.6% | 14.1 |
| ROCmFP2 (ROCmFPX) | 8.6 GiB | 0.341 | 75.5% | 21.2 |

Unsloth's own UD-Q3_K_XL is smaller, faster and closer to the model than either 4-bit ROCmFPX format. ROCmFPX keeps two jobs in the engine's lean option ([docs](../docs/lean.md)): ROCmI4 for the fastest prompt processing we measured (455 tok/s with its W4A4 path on ROCm), and ROCmFP2 when memory is the limit.

## A model trained ternary

PrismML's Ternary-Bonsai-2-27B is trained with weights of -1, 0 or +1, not squeezed into them afterwards. Its files are 5.5 GiB (PTQ1_0, 1.75 bits per weight) and 6.7 GiB (PQ2_0), against 50 GiB in F16. Measured against its own F16:

| File | Where | KL divergence | Same top token | Decode |
|---|---|---|---|---|
| PQ2_0 | CPU | **0.00014** | **99.6%** | |
| PTQ1_0 | ROCm | 0.065 | 91.3% | **27.3 tok/s** |

The ternary format itself is effectively lossless: on CPU, PQ2_0 matches F16 to within 0.00014. At 27 tok/s the 27B decodes 2.3x faster than the 4-bit file above, with no drafting. What is not there yet is the GPU path: PrismML's llama.cpp fork has no Vulkan kernels for these formats, its ROCm PTQ1_0 kernel loses some accuracy, and its ROCm PQ2_0 kernel is broken. A 27B at full quality in 7 GB is one kernel fix away.

## One trap on the way

hipBLAS returns wrong matrix products on this GPU (gfx1151; ROCm/rocm-libraries#11530). It never touched our 4-bit files, which use llama.cpp's own kernels, but it silently corrupted every F16 measurement until we moved those to the CPU and to Vulkan. The engine's ROCm build now forces llama.cpp's own kernels everywhere; that also measured slightly more accurate (KL 0.0064 against 0.0094) at the same speed.
