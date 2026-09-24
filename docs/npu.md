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
| 3a | Full-ELF generation: per-context control code and full-ELF assembly | landed (#8) |
| 3b | The lane runtime: model reader, buffer packing, runlist decode (`npu/lane`, `npu/generate`, `1bit npu-run`) | landed (#9) |
| 3c | Serving: tokenizer, `1bit unified` (now `1bit serve`'s NPU route) | landed (#9) |

## A model directory

```
<model dir>/
  model.q4nx             weights (safetensors layout, Q4NX tiles: "Q4NX chunks" below)
  config.json            Hugging Face config: dimensions, rope_theta, eos_token_id
  tokenizer.json         Hugging Face tokenizer
  npu/layer_ctx1.elf     layer kernel, instruction ELFs for context lengths 1, 2, 17
  npu/layer_ctx2.elf
  npu/layer_ctx17.elf
  npu/lmhead.elf         lm-head kernel, instruction ELF
  npu/layer.pdi          the design both kernels run on
```

### Q4NX chunks

A Q4NX weight is a grid of 32-row x 256-column tiles, one chunk each, row-major over
the grid. The chunk kind is the last dimension of the tensor's shape: `[tiles, 5120]`,
or `[tile rows, tile cols, chunk bytes]` (`npu/q4nx.h` has the byte layouts).

| Chunk | Kind | Weight | Seen in |
|---|---|---|---|
| 5120 B | q4_1: bf16 scale and zero per 32 columns of a row | `code * scale + zero` | Qwen3, Llama, Gemma, Phi models |
| 4736 B | Q4_K: u8 scale and min per 32 columns, bf16 `S`, `M` per row | `S * scale * code + M * min` | Qwen3.5-4B projections |
| 8704 B | Q8: bf16 scale per 32 columns, int8 codes | `d * code` | Qwen3.5-4B `lm_head` and embedding |

`npu/q4nx.h` decodes all three, and repacks Q4_K into q4_1 (exact apart from rounding
`S * scale` and `M * min` to bf16), so the lane and dx, which read q4_1, run Q4_K
weights unchanged. Q8 does not fit q4_1. `tests/npu_q4nx_test.cpp` checks the decoders
bit for bit against 1bit-MONSTER's verified decoders on real Qwen3.5-4B and Qwen3-0.6B
tiles (committed in `tests/golden/q4nx`), and the repack against its rounding bound.

`1bit serve -m <model dir>` serves such a directory on the NPU behind the
OpenAI-compatible API ([serve.md](serve.md)); inside Lemonade, that is what its
onebit backend runs (`--onebit npu`).

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

## The runtime (3b)

`npu/lane.cpp` is the port of 1bit-MONSTER's `RuntimeLayerEngine` fast-lane path,
on full ELFs. Its steps:

1. **Load.**
   - Build the init ELF and run it once.
   - Add the lm-head config.
   - Pack every layer's buffers, as described in `npu/pack.h`.
2. **Per token.**
   - Write the RoPE rows for the position into the slot's `i6` buffers.
   - Build a runlist of 28 layer runs plus the lm head, using that context
     length's kernel. The config for a context length is generated and added the
     first time the length is reached.
   - Write the token's embedding into `act` and submit.

`npu/generate.cpp` prepares the next token's runlist while the current one runs.
It uses two slots, each with its own `i6` buffers.

Two inputs had no rule to derive them from:

- the `inv_freq` table for θ = 1e6, kept as the reference runtime's float32
  literals;
- `sincosf`.

Every packed buffer is checked against the reference lane's own dump (`npu_pack_test`).

| Check (Strix Halo, Qwen3-0.6B) | Result |
|---|---|
| layer weights, `i5`, `i6` vs the reference lane's buffers (layers 0 and 2) | byte-identical |
| logits, anchor prompt, 24 greedy steps, vs the reference lane (xclbin) | bit-identical, 24/24 |
| decode speed, same run | 11.0 ms/token (91 tok/s); the reference lane binary: 15.1 |
| load (weights packed, kernels built) | 380 ms |

The `1bit` binary links only `libxrt_coreutil`: no xclbin, no FastFlowLM library, no Python.

## Serving (3c)

- **Tokenizer:** `npu/tokenizer.cpp` is 1bit-MONSTER's `qwen3_tokenizer.cpp`, which
  reads `tokenizer.json` directly (PCRE2 for the pre-tokenizer regex). The port
  fixes one bug: added tokens with `special: false`, such as Qwen3's `<think>` and
  `</think>`, were split as text. They now stay whole, as Hugging Face
  `tokenizers` does. `tests/data/qwen3_tokenizer_golden.json` holds 14 cases
  encoded by `tokenizers` 0.23.1, and all 14 match, ids and round trip.
- **`1bit unified -m <model dir> -p <port>`:**
  - It serves `/v1/health` (503 while loading), `/v1/models`,
    `/v1/chat/completions` (streaming or not) and `/v1/completions`.
  - The prompt is ChatML. It is Qwen2/Qwen3's template for text messages; the
    runtime has no Jinja.
  - Decoding is greedy, with a 1.1 repetition penalty over the last 64 ids by
    default. `repetition_penalty` in the request overrides it, and 1 gives plain
    greedy.
  - Requests are served one at a time: the lane has one KV cache.

## The XDNA stack

The lane runs on XRT and the XDNA shim plugin (`libxrt_driver_xdna`), which XRT loads at run
time. Both are pinned: `third_party/xdna-driver` is upstream
[amd/xdna-driver](https://github.com/amd/xdna-driver), and its own `xrt` submodule pins XRT.

- **The kernel driver is not built from this pin.** `amdxdna` ships in the kernel
  (`drivers/accel/amdxdna`), and Strix Halo runs the kernel's copy.
- **Building the pinned stack:** `scripts/build-xdna.sh <prefix>` builds XRT (the NPU package)
  and then the shim, staged under `<prefix>/root`. Nothing is installed system-wide: XRT hard-codes
  `/etc/OpenCL/vendors`, so both installs use `DESTDIR`. The engine then builds against it:

  ```
  scripts/build-xdna.sh ~/.cache/xdna-pin/prefix
  cmake -B build -G Ninja -DONEBIT_NPU=ON -DONEBIT_XRT_ROOT=$HOME/.cache/xdna-pin/prefix/root/opt/xilinx/xrt
  ```
- **XRT's OpenCL layer (`xocl`) is excluded** (`XRT_EXCLUDE_SUB_DIRECTORY`). The NPU does not use
  it, and it fails to compile where the distro's `ocl_icd.h` is newer than XRT's bundled OpenCL
  1.2 headers.
- **Keeping current:** `.github/workflows/bump-xdna.yml` runs daily and opens a PR whenever
  upstream `main` moves (secret `HRX_BUMP_TOKEN`). CI has no NPU, so run the check below on
  Strix Halo before merging.

Verified on 2026-09-23 with xdna-driver `5d302c9` and XRT `d8ececf`, in place of the system's
XRT 2.21.75. Both `libxrt_core` and `libxrt_driver_xdna` were loaded from the pinned prefix
(strace). `tests/npu_lane_e2e.sh` gave logits bit-identical to the reference lane on 24/24 steps,
at 10.8 ms/token (92.6 tok/s), with a 327 ms warm load.

## Open: where the kernel artifacts come from

CONTRIBUTING rule 4 requires NPU kernels to be built from source. The layer kernel and
lm-head artifacts do not meet that rule yet:

- The instruction ELFs came from the old per-context generator.
- The PDI came from the rounding-fixed `layer.xclbin` (1bit-MONSTER #2651).

Step 3a removed the need for per-context files and for the xclbin at run time. It does
not produce the kernel itself. The from-source 16-tile kernel (1bit-MONSTER #2666) dispatches at 0.343 ms per layer,
but it does not decode correctly yet: 0/24 tokens (#2668). Replacing the artifacts with
a from-source kernel is tracked in [PORTING.md](PORTING.md).
