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
# NPU engine: full ELFs, no xclbin

The fast lane runs one whole-layer kernel per decoder layer, plus an lm-head kernel.
For each token, the host submits the 28 layer runs and the lm head as one XRT runlist.
This step removes the xclbin from that path. Every kernel is a full ELF (the design's
PDI plus control code), which XRT opens directly. `npu/full_elf.{h,cpp}` builds these
ELFs in memory when the model loads. The code is pure C++, with no Python and no XRT.

Step 3 lands in three parts:

| Part | What | State |
|---|---|---|
| 3a | Full-ELF generation: per-context control code and full-ELF assembly | **this PR** |
| 3b | The lane runtime (`runtime_layer`, the runlist bridge, the model loader and packer) on these ELFs | next |
| 3c | Serving through Lemonade's `onebit` recipe | after 3b |

## Inputs

The generator reads these kernel artifacts from the model directory:

- `layer_ctx1.elf`: the layer kernel's instruction ELF for context length 1.
- `layer_ctx2.elf` and `layer_ctx17.elf`: only used to derive the context rule (below).
- `elf_0002_lmhead.bin`: the lm-head instruction ELF.
- The design PDI that both kernels run on.

It needs nothing else. In particular, it does not need the per-length ELF files, which
the old path pre-generated with a closed tool, one file per context length.

## The context rule

The layer kernel's control code depends on the context length N. `ContextMap::derive`
compares contexts 1, 2 and 17. On Qwen3-0.6B, 37 words differ, and each one follows one
of two rules:

- **linear:** `v(N) = v(1) + (N - 1) * step`. These are words equal to N, the KV row
  offset (`0x400` bytes per row), and that offset's relocation addends.
- **block:** `v(N) = v(1) * ceil(N / 16)`. These are the KV lengths, counted in
  16-row blocks.

The UID note is the MD5 of `.ctrltext`, so it is recomputed rather than mapped.
`derive` throws if any word follows neither rule.

## Assembly

`assemble_full_elf` follows the layout `aiecc --get-full-elf` produces:

1. A `.pdi.1` section and a `.ctrltext.0` section.
2. The argument symbols renumbered from the xclbin convention (the first buffer is
   argument 3) to argument 0.
3. A `.pdi.1` dynamic symbol and its relocation.
4. `.dynamic`, the COMDAT group, and the configuration and UID notes.

It keeps only the first transaction of the captured control code. The second copy is
the double-buffer slot, which is never relocated and never executed.

### Why there are three PDI modes

The array has to be configured once, and never again. If every layer config starts
with `load_pdi`, then each run whose arguments differ from the previous run's reloads
the array. The lane changes arguments on every one of its 29 runs, so it slows down 8×.

| Configuration (28 layers + lm head, one runlist, zeroed buffers) | Per token |
|---|---|
| `load_pdi` in every config | 82.0 ms |
| init config (only `load_pdi`) run once, then configs without it | **10.3 ms** |
| xclbin | 10.6 ms |

The runtime therefore does the following:

1. Creates the hardware context from the `kInitOnly` ELF (kernel `flinit`) and runs it once.
2. Adds the lm head and one config per context length with `kNone`, via `add_config`.
3. Keeps `kLoad` for stand-alone kernels.

## Checking the generator

```sh
cmake -B build -DONEBIT_NPU_CAPTURED=<dir with layer_ctx*.elf> \
      -DONEBIT_NPU_PDI=<design.pdi> -DONEBIT_NPU_GOLDEN=<dir with the lane's full ELFs>
cmake --build build --target npu_full_elf_test && ctest --test-dir build -R npu_full_elf
```

CI runs only the MD5 vectors, because the model checks need the model's kernel
artifacts. On Strix Halo, against Qwen3-0.6B, the model checks gave:

- **Captured contexts:** 5153/5153 reproduced byte for byte, for N from 1 to 8193.
- **Full ELFs:** 36/36 identical to the ones the fast lane ran (`init.elf`,
  `lmhead.elf`, `fl_ctx1..33.elf`, `load_ctx1.elf`).
- **Lane on those ELFs:** 24 greedy steps on the anchor prompt, with logits
  bit-identical to the xclbin lane at 24/24 steps.
  - 15.1 ms/token against 15.2 on xclbin.
  - Runlist execution averaged 10.47 ms against 10.64.
  - These numbers come from the 1bit-MONSTER lane binary, patched to load these ELFs.
    Part 3b re-measures them from this repository.

## Open: where the kernel artifacts come from

CONTRIBUTING rule 4 requires NPU kernels to be built from source. The layer kernel and
lm-head artifacts do not meet that rule yet:

- The instruction ELFs came from the old per-context generator.
- The PDI came from the rounding-fixed `layer.xclbin` (1bit-MONSTER #2651).

This PR removes the need for per-context files and for the xclbin at run time. It does
not produce the kernel itself. The from-source 16-tile kernel (1bit-MONSTER #2666) dispatches at 0.343 ms per layer,
but it does not decode correctly yet: 0/24 tokens (#2668). Replacing the artifacts with
a from-source kernel is tracked in [PORTING.md](PORTING.md).
