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
# XDNA 2 NPU engine: full ELFs, no xclbin

The NPU engine decodes Qwen3-0.6B at 91 tok/s (11.0 ms a token) on the XDNA 2 NPU of Strix Halo.
Its fast lane runs one whole-layer kernel per decoder layer, plus an lm-head kernel.
For each token, the host submits the 28 layer runs and the lm head as one XRT runlist.
This step removes the xclbin from that path. Every kernel is a full ELF (the design's
PDI plus control code), which XRT opens directly. `npu/full_elf.{h,cpp}` builds these
ELFs in memory when the model loads. The code is pure C++, with no Python and no XRT.
The engine has no xclbin path at all (the last one, a comparison backend in the
GGUF-on-NPU forward, was removed on 2026-10-03); xclbin numbers below are history,
from the reference lane the full ELFs were first checked against.

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

**Q8 has two code layouts under the same chunk size.** Qwen3.5-4B's lm_head (tied to the
embedding) stores code (row r, column k) at byte `k*32 + r`. Qwen3.6-35B-A3B's Q8 tensors
(untied lm_head, projections) store it at `(r/16)*4096 + (k/32)*512 + (k%32)*16 + r%16`.
Both containers report `flm_version` 1.0.3, and neither carries metadata that tells them
apart. Decoding tile (0, 0) of each lm_head and comparing with a GGUF of the same model
gives correlation 0.9998 with the right layout and about 0.01 with the other. `npu/q4nx`
decodes the first layout; the second is read only by the private 35B route.

`npu/q4nx.h` decodes all three, and repacks Q4_K into q4_1 (exact apart from rounding
`S * scale` and `M * min` to bf16), so the lane and dx, which read q4_1, run Q4_K
weights unchanged. Q8 does not fit q4_1. `tests/npu_q4nx_test.cpp` checks the decoders
bit for bit against 1bit-MONSTER's verified decoders on real Qwen3.5-4B and Qwen3-0.6B
tiles (committed in `tests/golden/q4nx`), and the repack against its rounding bound.

### From a GGUF

`tools/gguf_to_q4nx.py pack <model.gguf> <out dir> --tokenizer <dir>` writes a model
directory's `model.q4nx` and `config.json` from a GGUF (qwen3 only for now), and copies
`tokenizer.json` from `--tokenizer`. It writes every projection and `lm_head` as q4_1
tiles and the norms and embedding as bf16. A tied `lm_head` is made from `token_embd`.

| GGUF type | To q4_1 | Error |
|---|---|---|
| Q4_0 | scale `d`, zero `-8d`, codes unchanged | bf16 rounding of scale and zero |
| Q4_1 | scale `d`, zero `m`, codes unchanged | bf16 rounding of scale and zero |
| Q4_K | per 32-column sub-block: scale `d*sc`, zero `-dmin*mn`, codes unchanged (a Q4_K super-block is one tile row's 256 columns) | bf16 rounding of scale and zero |
| Q5_K, Q6_K, Q8_0, F16, BF16, F32 | dequantized, then min/max q4_1 per 32 columns | lossy |

`tools/gguf_to_q4nx.py verify <model.gguf> <out dir>` decodes every tensor it wrote and
compares it with the GGUF, dequantized. For the exact types, every weight must lie
within the bf16 rounding bound. The requantized tensors show their error.

On `Qwen3-0.6B-Q4_K_M.gguf` (unsloth), all 311 tensors pass. The Q4_K tensors have 0
weights over the bound. The Q6_K tensors (half of `v_proj` and `down_proj`, and
`lm_head` from the Q6_K embedding) requantize to cosine 0.9969 and relative RMS error
0.079. The output is 683,820,936 bytes, the same size as FastFlowLM's
`Qwen3-0.6B-NPU2` apart from the header. The directory has no `npu/` kernels, so it
runs on the dx route. There it decoded 24/24 greedy tokens the same as a CPU reference of
the repacked weights, with every step's logits at cosine 0.99995 or better (21.0 ms per
step). `1bit serve --device npu -m <dir>` loads it in 1.1 s and answers at 54-60 tok/s.

**The GGUF route reuses the fused lane, not only the per-op forward.**
`1bit serve -m <gguf> --device npu` repacks the GGUF with `scripts/repack_gguf.py`,
whose layout the `full_i8_*.elf` designs read, and runs the model-generic forward
(~100+ synchronous dispatches per token). With `ONEBIT_NPU_LANE_DIR=<lane kernel
dir>` it instead packs the GGUF with `tools/gguf_to_q4nx.py pack` — the q4_1 tile
layout the lane reads, cached at `~/.cache/1bit/q4nx/<stem>-lane` — and serves it on
the fast lane (one launch per layer, `npu/lane.cpp`). The two layouts are not
interchangeable, so the lane pack never aliases the forward pack. Any arch the
lane-layout packer does not model (it is qwen3-only), a missing tokenizer, or a
missing kernel set falls through to the unchanged per-op forward, so the route
cannot regress a model the lane does not cover. Measured on Qwen3-0.6B
(2026-09-28): GGUF → fused lane answers
" Paris" at 97–103 tok/s; the same GGUF → per-op forward is the 0.006–0.17 tok/s
class. Qwen2.5-7B GGUF takes the fallback (the lane/dx designs are Qwen3-shaped:
8 KV heads, per-head q/k RMSNorm, no projection bias).

**Declared-scale validation (required).** `scripts/repack_gguf.py`, the repack used
for the model-generic (`full_i8_*.elf`) route, **fails with `rc=1`** if the GGUF
declares a known-critical scale-like scalar that the repacked `config.json` drops:
`embedding_scale`, `residual_scale`, `logit_scale`, `scale_emb`, `scale_depth`,
`dim_model_base`, `final_logit_softcapping`, `attn_logit_softcapping`.  This is not
optional bookkeeping: MiniCPM4-8B decoded punctuation soup because
`minicpm.embedding_scale` (12.0), `minicpm.residual_scale` (0.2475 =
`scale_depth/sqrt(n_layer)`) and `minicpm.logit_scale` (16.0) were declared by the
GGUF, applied by llama.cpp, and dropped by the repack — and **no argmax-level check
can see it**, since the first two move the answer and the third moves only the
probabilities.  Any *other* scale-ish key (`*scale*`, `*softcap*`, `*mup*`) is
**warned** about, never fatal, because the next architecture's factor will have a
name not on the list while some scale-ish keys are legitimately not carried
(`deepseek2.expert_weights_scale` is a routing scale, not a logit scale).
`scripts/check_repack_config.py` runs the same scan standalone against a GGUF and a
produced directory, and has a negative control (strip the keys from a copy of a good
`config.json` and it must exit 1).

**Validated on cache HIT, not only on repack.**  The serve route's repack resolver
(`app/serve.cpp`) used to return a cached directory on `model.q4nx` existence alone,
which bypassed the repack and therefore the guard above — so a directory produced by
an older converter kept being served silently, and the guard only ever protected
freshly built artifacts.  A hit now runs the same metadata-only scan (GGUF header +
`config.json`; no reconversion, no NPU, milliseconds) and **discards and rebuilds**
the directory if it fails, so a stale artifact cannot be served.  Verified: the
exact command the resolver builds exits 1 on a stale dir (scales stripped) and 0 on
a good one.  Override the checker with `ONEBIT_Q4NX_CHECK`.

**Scope of the guard — read this before trusting it.**  Two limits, both deliberate:

* If the checker cannot be resolved the resolver **warns once per process, loudly,
  and reuses**.  A missing script must not brick serving, but that single warning is
  the *only* state in which the guard does not apply, and it applies to every
  directory for that process — so it is printed once, framed, rather than repeated
  into the general output.  A baked path that resolves to a moved or removed
  checkout is therefore obvious rather than invisible.
* The **directory route** (`1bit serve -m <Q4NX dir> --device npu`) consumes a model
  directory with no repack at all, so the guard never runs there.  That is correct:
  a *foreign* directory (FastFlowLM's published `Qwen2.5-7B-NPU2`,
  `MiniCPM5-1B-NPU2`, the 0.6B fast lane) is trusted as published.  In short — **the
  guard covers newly repacked directories; a foreign directory is trusted as
  published.**  A future foreign dir that drops a declared scale would be served
  silently, which is why the sentence is here.

**Provenance stamp (`repack-stamp.txt`).**  The scan checks *declared vs carried*; it
cannot check *which converter path and commit produced the directory*, and the model-dir
cache reuses a directory on `model.q4nx` existence alone, so neither half can be
reconstructed after the fact.  The repack now writes `repack-stamp.txt` (converter path +
git commit, arch/`model_type`, the repack script) and the cache-hit gate requires both
halves before reuse: a directory whose stamp is missing, or names a different converter
than the one this build would use now, is discarded and repacked.  `repack_gguf.py
--check-stamp <dir>` exposes the comparison -- exit 0 current, 1 stale, 2 cannot tell -- and
"cannot tell" (e.g. an unresolvable converter) fails **open** with a once-per-process
warning, so a missing converter cannot brick serving.  Verified end to end: a pre-stamp
cache dir is discarded and rebuilt, and the next serve reuses it without repacking.
Still converter-side and therefore not yet in the stamp: *whether a q/k rotary reorder was
applied* for this directory (that rule lives in the converter's `llama.py`/`minicpm.py`),
the field that would make a directory self-describing rather than dependent on its
converter commit alone.  Tracked in `~/evidence/npu-four-models/STATUS.md` §6.

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
2. The argument symbols renumbered from the instruction ELF's convention (the first
   buffer is argument 3) to argument 0.
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
| xclbin (historical reference lane) | 10.6 ms |

The runtime therefore does the following:

1. Creates the hardware context from the `kInitOnly` ELF (kernel `flinit`) and runs it once.
2. Adds the lm head and one config per context length with `kNone`, via `add_config`.
3. Keeps `kLoad` for stand-alone kernels.

The context lives for one `generate()` call. `Lane::begin()` creates it, and `Lane::end()` drops
it while it is still warm. A context that sat idle past the NPU's runtime suspend
(`autosuspend_delay_ms` 5000) failed its next runlist with `ERT_CMD_STATE_TIMEOUT`. It also left a
stuck context that blocked the NPU for 1–2 minutes. With a fresh context per call, 10 of 10
generations pass with 15 s gaps and with 120 s gaps, at 97–101 tok/s. Afterwards no context is
left on the device. Code that drives the lane directly with `step()` must call `begin()` first
and `end()` when done.

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

## Private routes

Qwen3.6-35B-A3B (`model_type` `qwen3_5_moe`) runs on the NPU through a closed-source
add-on: its kernels and host code live in the private `1bit-MONSTER/npu-kernels`
repository, not here. Measured on Strix Halo: parity against the fp64 reference passes
(3 positions, argmax 846 / 198 / 3710), 16.3-16.5 tok/s decode, and `1bit serve --device
npu` answers "The capital of France is Paris."

The engine keeps only the hook (`npu/private_route.h`):

- `-DONEBIT_NPU_PRIVATE=<npu-kernels checkout>` (with `-DONEBIT_NPU=ON`) builds that
  checkout's `addons/` into `1bit`. It must define the target `onebit_npu_private`, which
  holds `register_private_addon()`; `1bit` calls it at startup.
- The add-on registers a route per `model_type` (whether it can serve a directory, and a
  session that generates from token ids) and may register `1bit` subcommands. `1bit
  unified` and `1bit serve` use the route for that model type; `--npu-opt KEY=VALUE`
  (`unified`: `--opt`) passes options through to it.
- A route can be `optional`: an opt-in route for a `model_type` the fast lane also serves
  (e.g. `qwen3`). When it declines a directory (its `unavailable()` gives a reason, e.g. its
  `--npu-opt` was not given), the fast lane serves the directory exactly as it would without
  the add-on.
- Without the add-on, `1bit serve -m <35B dir> --device npu` exits with "the
  Qwen3.6-35B-A3B NPU route is not part of this build; build with
  -DONEBIT_NPU_PRIVATE=<npu-kernels checkout>". The fast lane is unaffected.

The add-on may also register an encoder for the Laya router (`laya/encoder.h`, docs/laya.md:
`ONEBIT_LAYA_DEVICE=npu`).

`tests/npu_private_route_test.cpp` (ctest `npu_private_route`, in CI) checks the registry
and that message; the add-on brings its own tests.

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
(strace). The fast-lane test of that time (`tests/npu_lane_e2e.sh`, since retired) gave logits
bit-identical to the reference lane on 24/24 steps, at 10.8 ms/token (92.6 tok/s), with a 327 ms
warm load.

### Checking the NPU

`tests/npu_lane_e2e.sh` is retired (2026-10-03): the fast-lane kernels it ran were derived from
FastFlowLM, and the reference logits it compared against are lost. The NPU check is now the private dx add-on's from-source check, which is not in this repository. A
pin bump that touches the NPU (xdna-driver, XRT, the Linux kernel) is checked with it on Strix
Halo before merging; the public tests that need no kernels (`npu_full_elf_md5`, `npu_q4nx`,
`npu_pack_test`, `tokenizer_test`) still run in CI.

## Step 3d: the layer kernel and lm-head, built from source

CONTRIBUTING rule 4 requires NPU kernels to be built from source. The layer kernel and
lm-head artifacts did not meet that rule for a while:

- The instruction ELFs came from the old per-context generator.
- The PDI came from the rounding-fixed `layer.xclbin` (1bit-MONSTER #2651).

Step 3a removed the need for per-context files and for the xclbin at run time, but did
not produce the kernel itself. A separate from-source 16-tile kernel effort
(1bit-MONSTER #2666) dispatches fast but does not decode correctly yet (#2668) and
remains open.

A different from-source kernel now does decode correctly, end to end: both the dense
layer kernel and lm-head run from source, ELF to ELF, with no xclbin at any point, and
one compiled program covers every context position (no rebuild per position). Verified
on Qwen3-0.6B, 1.7B and 4B against an fp64 reference: cosine similarity ≥0.9999 at
every tested position, and greedy decode matches the reference exactly end to end.
Kernel sources stay private per CONTRIBUTING; the work is tracked in
[PORTING.md](PORTING.md) and not yet merged into this repo.
