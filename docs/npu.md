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
# NPU engine

The NPU engine runs models on the AMD NPU (the XDNA 2 NPU of the Ryzen AI MAX+ 395) on Linux, on our own
runtime and kernels. It is in progress: correctness first, speed next. Its kernels, built from source,
match an fp64 reference on Qwen3-0.6B, 1.7B and 4B. The layer kernel and lm head reach cosine
similarity 0.9999 or better at every tested position, and greedy decoding matches the reference
token for token.
They are not merged into this repository yet. The release carries the NPU runtime, compiled into
`1bit`, and the `npu` backend of Lemonade's `onebit` recipe; until the kernels land, a release
package serves GGUF models on HRX, not NPU models ([releases.md](releases.md#the-npu-engine)).

The NPU runtime and the NPU kernels are proprietary and closed source
([releases.md](releases.md#private-components)). They ship only as compiled binaries: the runtime
is compiled into `1bit`, and the kernels ship as compiled, encrypted artifacts that `1bit` verifies
and decrypts in memory at load (a wrong key fails cleanly, and nothing phones home). Their source,
build and method are not in this repository or in any release. This page covers the public parts:
the model format, how `1bit serve` runs an NPU model, and how to build.

## A model directory

```
<model dir>/
  model.q4nx             weights (safetensors layout, Q4NX tiles: "Q4NX chunks" below)
  config.json            Hugging Face config: dimensions, rope_theta, eos_token_id
  tokenizer.json         Hugging Face tokenizer
  plan.json              the stream plan: fusion, kernel family, decode dirs ("plan.json" below)
  npu/                   the model's NPU kernels
```

`1bit serve -m <model dir>` serves such a directory on the NPU behind the OpenAI-compatible API
([serve.md](serve.md)); inside Lemonade, that is what its `onebit` backend runs (`--onebit npu`).

### plan.json: the stream plan (engine #799)

A model directory may carry a **plan.json**: the decisions the runtime used to make by looking at which
files happened to be in `--kernels`. Three of them, and each one used to be silent:

| decision | what the probe used to do |
|---|---|
| `fusion.gate_up` (`fused`/`split`) | the fused `full_i8_GU_*` kernel won, and with no split kernel on disk the split layout was assumed |
| `kernels.tag` (`""`/`_b8`/`_pb16i`/`_t2s`) | the artifact family came from `q4nx_weight_layout`, so a kernel dir of the other family changed which kernels ran |
| `decode.dirs` | the M=16 twins were looked for in `<model>/kernels-m16`, then `<kernels>/m16`, then `<kernels>` |

`tools/npu_plan.py <model dir>` writes one from `config.json` plus the kernel directory (it derives layer
roles from `layer_types` and projection shapes from the model's own dimensions, the way the forward does).
`--check` re-generates and diffs against the file on disk, so a stale plan is a gate failure rather than a
wrong number; `--print` shows it without writing.

The forward reads the plan with `npu/plan/plan.h` and **refuses by name** anything it cannot honour — a
`kernels.tag` that disagrees with the container, a sealed ELF named outside the pb16i family, a decode dir
that is absolute or contains `..`, a layer count that disagrees with `layers[]`, or a kernel the plan names
that is not in the directory (`forward init failed: plan <path>: this model's plan needs the QKV kernel
"full_i8_QKV_K1024_N4096.elf" in <dir> (missing)`). A model directory **without** a plan keeps the probes,
which is what makes this additive.

Four plans are checked in under `configs/plans/`, one per model directory on the build box (Qwen3-0.6B,
bonsai2-27b-NPU, bonsai2-27b-NPU-b8, bonsai2-27b-pb16i); `tools/npu_plan.py <dir> --check` proves the
directory's copy still matches. `scripts/npu-window11.sh` is the gate the change passed: three fixed greedy
prompts and the prof table, compared between a run with the plan and a run without it — 3/3 transcripts
byte-identical and every prof row within the tolerance. The tolerance is not a constant of the script: on this
box the same binary against itself moves by more than a couple of percent (engine #834 measured up to 7.7 %), so
`--compare` **refuses to run without either `--tol PCT` or `--control C1.log C2.log`** — the same-binary pair
whose own spread becomes the tolerance, rounded up. Every row prints both statistics and the header names the one
the tolerance applies to: `ratio-of-medians` (median of each side) or `median-of-ratios` (paired per round, the
honest one for a no-regression claim, and the one applied when both logs carry rounds). Alternate the order inside
a round (A B, then B A) in whatever drives the A/B pair: the script detects a control whose rows all move the same
way and says so, because that is the second serve's warm-up, not noise. `scripts/npu-window11.sh --self-test` is
the committed check of that policy.

**The folded row (`--folded`), engine#876.** That gate ran on one `rm` Qwen3-0.6B container, which has no folded
basis, no ternary LM head and no split full-attention QKV — so the parts of the forward the containers this box
actually serves (`bonsai2-27b-pb16i`, `bonsai2-27b-t2s2`) depend on were never executed by it. Three defects that
lived exactly there (engine#808) were invisible to a gate that had produced 108/108 identical transcripts, 9/9
identical prof rows and 452 bit-identical logprob tokens: a dropped role→context resolution for ternary-only designs
(the container refuses to answer), the activation rotation rewritten as an xor butterfly (a signed permutation of the
true Hadamard — wrong activations on every projection, well-formed output), and `head_pb16i_` read as a
self-referencing local (the LM head takes the int8 branch against a ternary ELF → `SIGSEGV` in `run_lm_head`). Text
equality cannot see any of them, because two of the three keep the text plausible and the third kills the process.

So `--folded` runs the same three transcripts against a folded container **with the container's own
debug taps on** (`NPU_PB16I_TRACE` / `NPU_QKV_DUMP`, both already in the forward, so neither arm is instrumented
differently) and records, per launch of the tapped layer, the raw activation, the rotated activation, the int8 codes
the kernel was given, its int32 accumulator, the dequantised output and the side-table `eref`. Two runs must agree on
those lines as well as on the text, and there the tap equality is the assertion that bites: with no measured control
the ms/launch columns are printed but not scored (a hand-typed 2 % cannot be evaluated on this box, engine#834), and
with `--control` they are scored like any other row. Cost: one container already on disk, ~2.5 min of device time for
both sides of a comparison, no new artifacts. The row also caught the fourth defect, on `main` rather than in a pack:
an `I8` embedding table dequantised against a K grid that was still 0 (engine#881, fixed ahead of this row). What it
must accept and refuse is pinned without a device by `tests/npu_window11_compare.sh` (ctest `npu_window11_compare`):
folded against folded passes, a single differing rotated activation fails, a folded log against a plain one fails, and
the plain row keeps its tolerance.

**`attention.mode`** (optional, default `causal`) is the mask, not a kernel decision. A pack that declares
`noncausal_full_attention` in its own `decision_config.json` — a d3 readout pack, trained with every
full-attention layer seeing the whole prompt — is generated as such, and the forward then encodes a prompt
bidirectionally: `step_batch()` fills the K/V cache for every row of the batch first and only then attends,
so row *r* sees rows after it. The GDN layers keep their causal recurrence (the pack's own runtime keeps it
too), a sliding layer stays causal, and the mode reads the same numbers only in one shape:

- **the whole prompt, one batched pass, at position 0** (`POST /v1/systemone` does exactly that). A batch
  that starts above 0 is refused by name — its earlier positions were already encoded causally.
- **it cannot generate.** `step()` (one position at a time, keys only in the past) refuses with the mode's
  name, so a d3 pack served on the chat route is an error rather than a silently causal answer.

An unknown mode string is a plan refusal, never a default: the two modes give different numbers, and a
guessed mask is a different model.

### Q4NX chunks

Q4NX is [FastFlowLM](https://github.com/ROCm/FastFlowLM)'s NPU model format (MIT), and its models on
Hugging Face (`FastFlowLM/*-NPU2`) are Q4NX directories. A Q4NX weight is a grid of 32-row x
256-column tiles, one chunk each, row-major over the grid. The chunk kind is the last dimension of
the tensor's shape: `[tiles, 5120]`, or `[tile rows, tile cols, chunk bytes]` (`npu/q4nx.h` has the
byte layouts).

| Chunk | Kind | Weight | Seen in |
|---|---|---|---|
| 5120 B | q4_1: bf16 scale and zero per 32 columns of a row | `code * scale + zero` | Qwen3, Llama, Gemma, Phi models |
| 4736 B | Q4_K: u8 scale and min per 32 columns, bf16 `S`, `M` per row | `S * scale * code + M * min` | Qwen3.5-4B projections |
| 8704 B | Q8: bf16 scale per 32 columns, int8 codes | `d * code` | Qwen3.5-4B `lm_head` and embedding |
| 2176 B | **t2**: 2-bit ternary codes, one fp16 scale per 128 columns of a row | `code * scale` | ternary models (Bonsai-2, PTQ1_0/TQ1_0 sources) |

**Q8 has two code layouts under the same chunk size.** Qwen3.5-4B's lm_head (tied to the
embedding) stores code (row r, column k) at byte `k*32 + r`. Qwen3.6-35B-A3B's Q8 tensors
(untied lm_head, projections) store it at `(r/16)*4096 + (k/32)*512 + (k%32)*16 + r%16`.
Both containers report `flm_version` 1.0.3, and neither carries metadata that tells them
apart. Decoding tile (0, 0) of each lm_head and comparing with a GGUF of the same model
gives correlation 0.9998 with the right layout and about 0.01 with the other. `npu/q4nx`
decodes the first layout.

`npu/q4nx.h` decodes all four, and repacks Q4_K into q4_1 (exact apart from rounding
`S * scale` and `M * min` to bf16), so code that reads q4_1 runs Q4_K weights unchanged. Q8 does
not fit q4_1, and neither does t2 (per-128 fp16 scale against q4_1's per-32 bf16 one): a t2 tensor
is expanded on the target, never repacked. `tests/npu_q4nx_test.cpp` checks the decoders bit for bit
against reference decoders on real Qwen3.5-4B and Qwen3-0.6B tiles (committed in
`tests/golden/q4nx`), and the repack against its rounding bound; t2 has its own contract test
(`tests/npu_t2_layout_test.sh`) and oracle (`tests/q4nx_t2_test.py`).

#### The t2 chunk kind

A ternary model stores every weight as `code * d` with `code` in {-1, 0, +1} and one `d` per 128
columns of a row. t2 is that, in the same 32x256 tile grid as every other chunk, at **2.125
bits/weight** — the stream a 27B ternary model is meant to be read at, instead of the 4.5 b/w
`pb16i` image (int4 codes + one int16 magnitude per 32 K) that the exact path streams today.

```
t2 tile, 2176 bytes, 32 rows (N) x 256 columns (K) of one tensor
  [0, 2048)     2-bit codes:  byte r*64 + c/4, code c%4 at bits 2*(c%4)
                value 0 = -1, 1 = 0, 2 = +1, 3 = reserved (no converter emits it)
  [2048, 2176)  IEEE fp16 scales: half at 2048 + r*4 + 2*g,  g = c/128
                                                        w[r][c] = code * scale
```

Tile order is the container's everywhere: row-major over the `[N/32][K/256]` grid, so one
32-column block's whole K extent is contiguous. Geometry: `N % 32 == 0`, `K % 256 == 0`.

**A tensor that is not whole tiles is padded, not refused-and-forgotten.** The padding is code 0 with
scale `0.0`, i.e. a weight of exactly 0: a padded K column contributes 0 to the reduction whatever the
activation is, and a padded N row is never read. `tools/q4nx_t2.py` refuses such a tensor by default and
pads it with `pad=True` (`pad_to_tile`). The chunk has no field for the original shape, so the padded
shape is the tensor's shape and the consumer skips the rows the writer did not fill. All 431 ternary
Bonsai-2 images are already whole tiles (5120 / 6144 / 17408-column rows), so this exists for GGUF
tensors and for any projection whose width is not a multiple of 32.

**The stored weight is the folded (post-rotation) one, and t2 carries no rotation of its own.** A folded
container's `W'` already has the block Hadamard and the per-width signs baked into the weights; the
consumer rotates the *activation* into that basis — `A' = H (A . s)`, signs first, then the normalized
Sylvester-Walsh butterfly, then the `1/sqrt(block)` — and never touches the weight
(`npu/forward/include/hadamard.h`, called from `q4nx_forward.cpp`). So a t2 tensor holds `W'` exactly as the pb16i
image does, the fold declaration stays where it belongs (the container's `q4nx_hadamard`,
`q4nx_hadamard_block` and the `HAD1` sign table), and the chunk needs no pre/post-rotation flag: reading
a t2 tensor in the wrong basis is a consumer bug, not a format one. The inverse direction (the tied
embedding, `inverse_weight_names` in the pack) is `S (H y)`, which round-trips to the identity.

**This rotation is host work on the critical path, so it is SIMD, and it must stay bit-identical.** It runs
once per projection — ~430 times a decode token on Bonsai-2 — and the scalar butterfly cost 14.2 us per
5120-float vector in-loop: 15.8 ms/token in a quiet window and 39.0 ms/token on a loaded one, i.e. pure
CPU contention that a device-side win cannot touch. `hadamard.h` keeps the scalar loop as the reference and
adds an AVX2 path (dispatched at run time) that performs the *same* adds and subtracts on the same operands
in the same stage order — stages 1, 2 and 4 as an 8-lane in-register butterfly, every later stage as
contiguous 8-lane add/sub — so the rotated vector is bit-identical and no container's text moves: 7.6 ->
2.4 us per 5120-float vector, 40.0 -> 16.8 us per 17408-float one. `tests/hadamard_rotate_test.cpp` is the
gate that runs on every build, on any host: bit-for-bit equality against the scalar loop for every block a
container can carry and for the fleet's real activation widths, plus a check that the scalar loop really is
the Hadamard transform (compared against a direct sum over the sign pattern).

**The scale is fp16, not bf16, and that is load bearing.** Every source format already stores this
scale as fp16 — PTQ1_0's, TQ1_0's and TQ2_0's `d` are `ggml_half`, and `pb16i`'s `ms * 2^e_ref` is an
int16 with a per-column exponent — while bf16 has 8 significand bits against fp16's 11. Measured on
the 431 ternary Bonsai-2 images (416 layers + the 15 MTP head tensors, 25.66 G weights):
**174,982,781 of 200,499,200 scales (87.27 %) are not bf16-representable**, worst relative error
3.89e-3; fp16 is exact on all 200,499,200. fp16 and bf16 are both two bytes, so asking for bf16 buys
nothing and costs the bit-exact gate. A target whose arithmetic wants bf16 converts at load, in the
kernel, and owns that rounding — the chunk keeps the exact value.

A reserved code (3) decodes as **0**, on both paths — `npu/q4nx.cpp::dequant_tile` and `tools/q4nx_t2.py::unpack` — so a corrupt byte cannot become a weight no converter wrote, and it can never be a `2 * scale` value that no ternary encoder can produce. No converter emits one; `q4nx::t2_reserved()` counts them, and that is what a loader or a test refuses on. (An all-zero code byte is **code -1**, not code 0 — the value table is `code - 1`, the same convention Q2_0 uses.)

| what | measured |
|---|---|
| ternary codes in the 431 images | all `{-1, 0, +1}`, no reserved pattern |
| one scale per 128 K is enough (the four per-32 magnitudes agree) | 100 % of blocks |
| fp16 scale exact vs the pb16i floats | 200,499,200 / 200,499,200 |
| 27B size | pb16i 14.44 GB -> **t2 6.82 GB** (416 layers 6.46 GB, 15 head tensors 0.36 GB) |
| 2.125 b/w | `2176*8/8192` exactly |

The oracle is bit-exact, not approximate: `tools/q4nx_t2.py from_pb16i` refuses anything it cannot
represent exactly (a non-ternary code, magnitudes that differ inside a 128-K block, a scale that is
not fp16-exact), and `tests/q4nx_t2_test.py` re-decodes the result and compares against the pb16i
dequantisation of the same image, float for float, on all 431 images when they are on the machine.
Converters: `from_pb16i` (the engine's integer-exact image), `from_ptq1_0` and `from_tq1_0` (both
ternary with an fp16 scale, lossless). `from_tq2_0` is refused unless asked, because TQ2_0's code 3
means **+2** — it is not a ternary format, and clamping it to +1 changes weights.

### The t2s stream (what the XDNA decode kernel reads)

t2 is the container format: the decoder's byte order, for any consumer. The device does not want that
order — the mmul takes int4 nibbles at element `e = r*16 + c` of a 16x16 block in byte `e/2` — so the
kernel would have to transpose 2-bit codes across lanes. `tools/q4nx_t2s.py` derives the **stream** form
of the same weights instead, which makes the on-core expansion a byte-local `1 -> 2` map:

    2,176 B per k=256 x n=32 tile, 2.125 b/w (pb16i 4,608 B; 27B 14.44 GB -> 6.82 GB)
    [0, 2048)    2-bit codes, n-major blocks: block (nb, kb) at (nb*16 + kb)*64, and inside a block
                 byte r*4 + j holds the four codes for k = kb*16 + r, n = nb*16 + 4j + {0,1,2,3}
    [2048, 2176) int16 multiplier per 128-K block per column, [nb 2][gb 2][16 c] -- pb16i's own `ms`
                 at the granularity the model stores (pb16i repeats it 4x to reach 32-K granularity)
    expansion: out[2i] = f(c0) | f(c1)<<4, out[2i+1] = f(c2) | f(c3)<<4, f(v) = v - 1 mod 16 = {15,0,1}

The stream's contract is byte-exact against the image it came from, not statistical: `expand()` must
reproduce the pb16i code bytes block for block (and `stream()`/`from_pb16i_image()` must agree byte for
byte), which `tests/q4nx_t2s_test.py` checks synthetically and which was also run on the real Bonsai-2
images (all 10,880 tiles of `blk_0_ffn_gate_weight.pb16i`, 0 of 44,564,480 code bytes differing,
14.44 GB -> 6.82 GB — the same 6.82 GB the t2 chunk reports). `from_pb16i_image` **refuses**, with a
named error, an image whose four 32-K multipliers inside a 128-K block are not equal: that is the one
thing this compaction cannot hold, and it is the thing the t2 chunk's own gate measures at 100 % of
blocks for Bonsai-2. The n-major order also makes one `nb` (1,024 B) a single contiguous DMA run, and
the tile itself is streamed as one 2,176-B run: 2-bit must not cost contiguous-run length.

A model shipped in the **t2** form needs no pb16i image to feed this kernel: `from_t2(chunk, K, N)` takes
the container's own chunk and returns the same bytes `from_pb16i_image()` returns for the image it was
made from, by re-deriving the image with the packer's code rather than a second implementation of the
tile order. The two forms are the same weights in **different orders** (t2 is k-major within each 32-row
band, t2s is n-major within each 16x16 block: on the test fixture's 139,264 stream bytes, 135,476 differ),
so they are never interchangeable and `from_t2` is the only crossing — the test asserts both the equality
and that inequality, so a "simplification" that returns the chunk unchanged fails.

#### Wire format v2: the ordering moves into the stream, not the core

The mmul's operand byte for element `e = 16k + c` is `e/2`, so a 64-byte vector of 2-bit codes has to reach the
operand's **even** bytes and the next 64 its **odd** bytes — a spread, not a copy. Under the original order (v1) that
spread is a lane-crossing shuffle inside the core (`aie::interleave_zip`), and it was measured on the array: **11.1-12.6
GB/s against 30-32 GB/s** for the plain contiguous stores with the wrong order. So v2 changes the order *inside each
64-byte block* and nothing else:

    v1: byte r*4 + j  ->  k = kb*16 + r,        n = nb*16 + 4j + {0,1,2,3}
    v2: byte b        ->  lanes 0/1: k = kb*16 + b/8,     n = nb*16 + 2*(b%8), +1
                          lanes 2/3: k = kb*16 + 8 + b/8, n = nb*16 + 2*(b%8), +1

which is exactly the arithmetic the operand wants: byte `b` of the first store covers element `2b = 16*(b/8) + 2*(b%8)`,
and byte `b` of the second store covers `2*(64+b) = 128 + 2b`, i.e. eight k rows down. The kernel's two contiguous
stores are then correct with no shuffle and no table — **the same kernel binary** that had to interleave is byte-exact
on v2 (3/3 seeds, 0 mismatches, `amdxdna` counter 0, measured 2026-11-02). The tile is still 2,176 B and the time order
is unchanged, so the DMA descriptors and the tile count do not move; the ms area is byte-identical between the formats.

`tools/q4nx_t2s.py` carries both: `from_pb16i_image()`/`stream()`/`decode()`/`expand()` are v1,
`from_pb16i_image_v2()`/`decode_v2()` are v2, and `tests/q4nx_t2s_test.py` pins that they carry the same codes and the
same multipliers, that only the code area differs, and the block algebra above. A container declares which one its
images are with `q4nx_weight_layout_version` (5 = v1, 6 = v2): the two are **not** interchangeable — a v1 image read by
the no-shuffle kernel is a silent permutation of the weight matrix, which is precisely the failure that took three
single-code probes to name.

#### The t2s stream in the engine (launch path)

A container streams t2s when its `config.json` says `"q4nx_weight_layout": "t2s"` (pb16i is 4, and each layout
refuses the other's version). **The version picks the artifact family**, because the two t2s wire orders hold the same
weights at the same size in a different byte order and reading one with the other's kernel is a silent permutation of the
matrix, not a crash: **5 = v1** -> `_t2s` / `_t2s_m16` (the expansion interleaves inside the core), **6 = v2** ->
`_t2s2` / `_t2s2_m16` (the ordering is in the wire, so the core's two contiguous stores are exact — no lane shuffle), and
any other version is refused by name at init. v2 is the one to ship: measured on the array for GU (K=5120 N=17408, M=16)
against the pb16i phase in the same window, **0.589 ms vs 1.037 ms (1.76x), 40.2 GB/s** of stream against 48.4, with
seeds 1/2/3 byte-exact and the `amdxdna` counter 0 (`~/wt/npu-arena/LEADERBOARD.md`, `submissions/GU16T2S`). The
images are the pb16i sidecars' own set — `blk_<layer>_<tag>_weight.t2s` plus the **unchanged**
`.ref` (`e_ref int16[N]` + mode byte, because the compaction changes neither the magnitudes nor
e_ref) — in the directory named by `q4nx_t2s_dir` / `Q4NX_T2S_DIR`, and the kernels carry the
`_t2s` suffix (`full_i8_GU_K5120_N17408_t2s.elf`, and the `_t2s_m16` decode family).
`tools/q4nx_t2s.py` writes those files from either form of the same weights — `from_pb16i_image()` for a
container that ships pb16i sidecars, `from_t2()` for one that ships only t2 chunks ("derive the t2s stream
from a t2 chunk, so the container is enough", #830); nothing is requantised in either crossing.

Everything downstream of the load is pb16i's path: the same int32/int64 accumulation, the same
host-side single rounding, the same per-column `2^e_ref`, and the same loud refusals. The one
difference the host must respect is the file: `q4nx_t2s.h` gives the 2,176-B tile, and a `.t2s` file
that is not `tiles x 2,176` bytes is refused by name (never read as a pb16i image). Two independent
checks keep the contract honest without a device: `tests/npu_t2s_layout_test.cpp` pins the tile
against the numbers the tool measured on a real image (gate/up: 50,135,040 B of pb16i → 23,674,880 B
of stream), and `tests/q4nx_t2s_test.py` pins the code order byte for byte.

**A t2s container wants a sealed set for the same reason a pb16i one does.** The layout is not what makes
a phase change expensive: the **hw_context** is. Measured on this box in-loop on 2026-11-02, a t2s decode
family shipped as eight per-role `_t2s2_m16` ELFs paid **+2.0-2.3 ms on every launch, in every role** — more
than the kernels' own device time — against the ~2.7 ms between two hw_contexts recorded at
`npu/forward/include/q4nx_forward.h`. That tax turned a kernel that is 1.76x faster in the arena into a
model that was 1.8x *slower* than pb16i. The fix is the same artifact shape pb16i uses: one full ELF
carrying every instance (`main:gu|d|qkv|o|z|qh|kv`), named by the container's own key
`q4nx_t2s_sealed_elf` (pb16i keeps `q4nx_pb16i_sealed_elf`; a layout never borrows the other's key, because
a pb16i ELF would read 4,608-byte tiles out of 2,176-byte images). `tools/npu_plan.py` writes the key into
`plan.json`'s `decode.sealed_elf`, and the engine runs every decode role inside that one context.

#### The image fill (what a fresh serve costs)

A pb16i/t2s container's weights *are* the sidecar images, so a cold serve is a file read and nothing
else: `load_pb16i_into()` `fread`s each image **straight into the weight BO's mapping**, then syncs
once. The earlier path zero-filled a `want`-byte vector, `fread` into it, `memcpy`'d it into the BO and
allocated the BO zeroed as well — four passes over the whole 6.8 GB of decode weights — and nothing
read the zeros. `NPU_FILL_COPY=1` keeps that path so the two can be A/B'd, and `NPU_PB16I_TIMING=1`
prints `[fill] <mode> <stem> bytes= read= copy= sync=` per image (431 images, one line each):

| (Bonsai-2 t2s, 431 images, 6.82 GB) | read | copy | sync | first request, 8 tok |
|---|---|---|---|---|
| copying loader | 0.25 s | 0.39 s | 0.09 s | 6.35 s |
| **read into the BO** | 0.36 s | **0.01 s** | 0.11 s | **5.92 s** |

Same greedy text, `amdxdna` errors 0. Both halves ran in one ticket with the page cache warmed first,
because otherwise the second half reads from cache and the difference looks like a loader win.

**The rest of the "cold start" is the disk, not the loader.** `dd iflag=direct` over the image set reads
**2.16 GB/s**, so 6.82 GB is **3.2 s** of device time the loader cannot avoid; with a warm page cache
the same fill is 0.47 s. A first request on a fresh serve therefore costs one fill (0.5 s warm, ~3.2 s
cold) on top of its prefill, and every request after it on the same serve costs none. Do not read a
per-serve benchmark's request wall as a decode number: `[ph]`/`[lh]` are the per-token instruments.

### From a GGUF

`tools/gguf_to_q4nx.py pack <model.gguf> <out dir> --tokenizer <dir>` writes a model
directory's `model.q4nx` and `config.json` from a GGUF (qwen3, qwen3moe and qwen2 for now),
and copies `tokenizer.json` from `--tokenizer`. It writes every projection and `lm_head` as
q4_1 tiles and the norms and embedding as bf16. A tied `lm_head` is made from `token_embd`.
A MoE model's router stays bf16 and untiled (it runs on the host, and its top-k must be the
GGUF's own), and its routed expert stacks become one tensor per layer and projection — see
"Routed expert stacks" below.

`config.json` also carries the tokenizer's `add_bos_token` and `add_eos_token` when the GGUF
states them (`tokenizer.ggml.add_bos_token`/`add_eos_token`), because the forward builds its
prompt from `config.json`'s flag and defaults to no BOS (`app/forward_serve.cpp`). A checkpoint
whose `tokenizer_config.json` sets `add_bos_token: true` — Zyphra/ZR1-1.5B, a Qwen2 fine-tune —
is otherwise served a chat prompt with no BOS: measured on ZR1-1.5B, the forward built a
5-token prompt where llama.cpp built 6, and the reply opened with the end-of-turn token.
`tools/safetensors_to_q4nx.py` carries the same two keys from `tokenizer_config.json`.

| GGUF type | To q4_1 | Error |
|---|---|---|
| Q4_0 | scale `d`, zero `-8d`, codes unchanged | bf16 rounding of scale and zero |
| Q4_1 | scale `d`, zero `m`, codes unchanged | bf16 rounding of scale and zero |
| Q4_K | per 32-column sub-block: scale `d*sc`, zero `-dmin*mn`, codes unchanged (a Q4_K super-block is one tile row's 256 columns) | bf16 rounding of scale and zero |
| Q5_K, Q6_K, Q8_0, F16, BF16, F32 | dequantized, then min/max q4_1 per 32 columns | lossy |
| the same, named by `--dtype8` | dequantized, then symmetric int8 per 32 columns (a Q8 chunk: `d*code`, bf16 `d`, codes at `column*32 + row`) | int8 rounding (rel rms ~0.005) |

`pack --dtype8 REGEX` (a Q4NX-name regex, e.g. `'v_proj|down_proj|lm_head'`) stores the
matching tensors as Q8 chunks instead of crushing them to q4_1, and leaves any tensor whose
source was already exact at 4 bits alone (a Q4_K_M mixes types layer by layer). It exists
because q4_1 spends 4 of every 32-value group's bits on a min and has 16 levels: a tensor
that was Q6_K or Q8_0 in the source is both worse and *bigger* after the crush. On
`Zyphra/ZR1-1.5B` Q4_K_M the pattern names 57 tensors and 29 of them are Q6_K (14 `v_proj`,
14 `down_proj`, `lm_head`); the other 28 are Q4_K and stay q4_1. Those 29 as Q8 move the
container's wikitext-2 KLD against BF16 from **0.204038 to 0.129436** (PPL 85.4513 ->
79.7284) for **+180 MiB** (1,431,806,280 -> 1,620,611,528 B), which is better than the
Q4_K_M arm it was packed from (0.138010) but still 1,546 MiB against that arm's 1,066. The
Q8 chunk decodes in `npu/q4nx.cpp` and `npu/forward/src/q4nx_dequant.c`, and
`tools/q4nx_to_gguf.py` writes it back as Q8_0 (the same f16 `d` + int8 `code` blocks), so
a `--dtype8` container is still one model on both sides. The **dx route cannot run it
today**: `q4nx::as_q4_1()` throws for Q8 (8-bit codes do not fit q4_1), so a container
carrying Q8 tensors is for a consumer that reads the chunk kinds, not the narrowed path.

`tools/gguf_to_q4nx.py verify <model.gguf> <out dir>` decodes every tensor it wrote and
compares it with the GGUF, dequantized. For the exact types, every weight must lie
within the bf16 rounding bound. The requantized tensors show their error.

On `Qwen3-0.6B-Q4_K_M.gguf` (unsloth), all 311 tensors pass. The Q4_K tensors have 0
weights over the bound. The Q6_K tensors (half of `v_proj` and `down_proj`, and
`lm_head` from the Q6_K embedding) requantize to cosine 0.9969 and relative RMS error
0.079. Served on the NPU, the repacked model decoded 24/24 greedy tokens the same as a CPU
reference of the repacked weights, with every step's logits at cosine 0.99995 or better.

### Routed expert stacks

A MoE GGUF keeps its routed experts as one three-dimensional tensor per projection
(`blk.N.ffn_{gate,up,down}_exps`, ggml `[cols, rows, experts]`), expert `e` on rows
`[e*rows, (e+1)*rows)` of the stream. The converter writes each stack as ONE container
tensor — `model.layers.N.mlp.experts.{gate,up,down}_proj.weight`, `I8`, shape
`[experts * rows/32, cols/256, 5120]` — so expert `e` is tile rows `[e*rows/32, (e+1)*rows/32)`
of a plain tile grid: a decode step points a kernel at one expert's 192 chunks (983,040 B on
Qwen3-30B-A3B) without any gather.

`model.q4nx`'s `__metadata__.moe` carries that decomposition so a caller never derives it:
`experts`, `used`, `norm_topk_prob`, and per tensor `rows`, `cols`, `tile_rows`, `tile_cols`,
`tiles`, `bytes` and the **byte offset of every expert** (`offsets`, `[experts]` entries,
relative to the data section like every `data_offsets`). Those offsets are the list a
per-dispatch chunk-address kernel takes (docs/npu.md, the DMA path in the engine's NPU lane).

Measured on `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (qwen3moe, 48 layers, 128 experts,
top-8, expert_ffn 768, no shared expert): 579 tensors, 144 expert stacks x 128 expert
offsets, header 302,144 B, container 19,528,322,120 B (18.19 GiB) — 18.13 GiB of it routed
experts.

**The top-8 boundary is razor thin.** Measured with the same router tensor on the same token, the
container's weights against the GGUF's own f32 weights: **32 of 48 layers pick a different 8-expert
set**, always at the 8th slot, because the margin between rank 8 and rank 9 is **1.2e-3 in
probability in one and 2.9e-4 in the other** (layer 2: expert 25 at 0.021700 vs expert 125 at
0.020511, and the reverse). A hidden state that differs by 0.5 % — what requantising the Q6_K
tensors costs — is enough to swap it.

**That is a symptom, not the whole story, and the difference matters when judging a kernel.**
Re-running the same comparison with the container's routing *pinned* for the f32 pass — same expert
ids and weights at every layer of every position — the log-prob difference does not shrink at all
(mean |dlogp| 14.36 nats pinned against 14.29 unpinned, over 16 teacher-forced positions). So the
routing flips are not what separates the two weight sources; the per-layer hidden state (cos
0.993-0.9996) compounding through 48 layers is, and it changes which token wins a peaky position.
Two things follow, and they are the operative rules here: compare implementations on **aggregate**
quality (KLD over a corpus, perplexity), never single-token argmax or per-token log-probs; and gate
a kernel with the **expert ids held fixed** (A row + the router's own top-k + the expected int32 per
expert, which is what `tools/moe_expert_chunks.py gate` emits), so the gate measures the kernel and
not the model's own sensitivity. `verify` result: see the TMOE-NPU lane's state file.

**A chunk-list dispatch shares ONE A across its chunks: exact for gate and up, impossible for down.**
At batch 1 every selected expert of a layer sees the same activation row, so gate and up are one
dispatch each — but `down_proj`'s input is each expert's own SwiGLU output, so it needs a per-chunk A
or one dispatch per expert. On this model (48 layers; per layer q, gate, k, v, o in the pb16i split,
then gate|up|down) that is **385 dispatches per token with the shared A and 721 without**, and that
gap decides whether the case meets its own bar:

| per-dispatch fixed cost | down shares one A | down per expert | Qwen3-4B dense, same arithmetic |
|---|---|---|---|
| 0.086 ms (fit below) | **11.05 tok/s (1.06x)** | 8.37 (0.80x) | 10.45 |
| 0.116 ms (quoted floor) | 9.80 (1.01x) | 7.09 (0.73x) | 9.68 |

That is at 34 GB/s in-loop over a **1,703,854,080 B/token** stream — routed experts 1,019,215,872
(8 of 128 x 3 projections, pb16 NW=16) + attention 509,607,936 + `lm_head` 175,030,272, with
`--pad-n 1024` on gate/up taking it to 1,930,346,496 — and the ratio holds across 34-43.4 GB/s:
**0.97-1.06x the 4B-dense bar when down shares the dispatch, 0.68-0.80x when it cannot.** So the
deciding term is the down_proj A shape, not the expert stream: sizing a MoE case on the expert
bytes alone understates the stream by 40 % (they are 60 % of it) and misses the term that decides
it.

The fixed cost above is measured, not assumed. Two runs of the same 50,135,040-B weight chunk in
the arena's chunk-list design — 1.051 ms for one chunk, 2.016 ms for two — fit a marginal rate of
51.95 GB/s and **0.086 ms paid once per dispatch**, i.e. a second weight BO on the same dispatch
costs nothing measurable (within the fit residual; the 2-chunk run is 4.1 % faster per chunk).
Pricing N chunk BOs as N launches overcharges the dispatch column by that factor: it would add 672
dispatches per token here (48 layers x gate and up x 8 chunks, against 96 real ones), +66.5 ms at
0.099 ms each.

`1bit serve -m <model.gguf> --device npu` repacks a GGUF itself, caches the result under
`~/.cache/1bit/q4nx/`, and serves it. The repack runs the engine's own `scripts/repack_gguf.py`,
which the package ships beside `1bit` (`scripts/`, with its declared-scale guard;
`ONEBIT_Q4NX_REPACK` overrides the path), so a packed install repacks too. It needs `python3` with
`numpy` and `gguf`, and FastFlowLM's Q4NX converter, which is not vendored: point
`ONEBIT_Q4NX_CONVERTER` at a converter checkout (a `~/np-model-generic` beside the engine or in the
home directory is found without it). GGUFs of six architectures (Qwen2.5, Qwen3 MoE,
Qwen3.6-35B-A3B, MiniCPM4/5, GLM-4.7-Flash) answer correctly this way, on a generic per-op path
that is not yet fast (#179, #181, #209). On a release, GGUFs belong on HRX.

**Declared scales are checked.** The repack (`scripts/repack_gguf.py`) fails with `rc=1` if the
GGUF declares a known-critical scale-like value that the repacked `config.json` drops:
`embedding_scale`, `residual_scale`, `logit_scale`, `scale_emb`, `scale_depth`,
`dim_model_base`, `final_logit_softcapping`, `attn_logit_softcapping`. MiniCPM4-8B decoded
punctuation soup because three of them were dropped, and no argmax-level check can see the
third, which moves only the probabilities. Any other scale-like key (`*scale*`, `*softcap*`,
`*mup*`) is warned about, not fatal. `scripts/check_repack_config.py` runs the same scan on its
own, with a negative control.

A cached directory is checked again on every hit, with the same metadata-only scan, and it is
rebuilt if the scan fails or if its `repack-stamp.txt` (converter path and commit, architecture,
repack script) names a different converter than this build would use (`repack_gguf.py
--check-stamp <dir>`: exit 0 current, 1 stale, 2 cannot tell). If the checker or the converter
cannot be found, serve warns once per process and reuses the directory, so a missing script
cannot stop serving. A Q4NX directory served as it is (`1bit serve -m <Q4NX dir>`), such as a
published FastFlowLM model, is trusted as published. Its chunk layout can be checked first, without
dequantizing anything: `tools/check_q4nx_dir.py <dir>` reads the container header only, names each
quantized tensor's chunk kind, and refuses an unknown chunk size or a tensor shorter than its layout
needs. It cannot tell the two Q8 layouts apart ("Q4NX chunks", above), because the container does
not record which one it holds.

### From a Ryzen AI OGA NPU pack

`1bit npu-pack import <pack dir> -o <out dir>` reads a pack AMD publishes for Ryzen AI
(`*_rai_*_npu_*`: `model.onnx`, `model.onnx.data`, `dd_metastate_*.fconst`, `config.json`,
`genai_config.json`, the tokenizer) and writes a model directory. `1bit npu-pack info <pack dir>`
prints what the pack states as JSON and writes nothing. The pack's weights and its config/tokenizer
contract are the model; the reader is written from the pack-side facts in
`amd-oss/npu-re/docs/FACTS-oga-pack-weight-layout.md`, and no vendor library, wheel or compiled
program is called or shipped.

What converts, and how faithfully:

| pack tensor | where | to | fidelity |
|---|---|---|---|
| `model.embed_tokens.weight` | `model.onnx.data`, fp16 | bf16 | fp16 → bf16 rounding (10 → 7 mantissa bits) |
| `lm_head` `MatMulNBits` (bits 4, block 128, asym) | `model.onnx.data` | q4_1 tiles | codes unchanged; `scale = s` (exactly bf16 in this pack), `zero = -zp*s` (bf16 rounding) |
| `input_layernorm`, `post_attention_layernorm` | `.fconst`, bf16 at a fixed offset | bf16 | byte-exact |
| `self_attn.{q,k,v,o}_proj` (24 x 4) | `.fconst`, 34,816-byte 64-row tiles | q4_1 tiles | codes unchanged; `scale = s` (bf16), `zero = -zp*s` (bf16 rounding) |
| `self_attn.{q,k,v}_proj.bias` (24 x 3) | `.fconst`, the tiles' bias subvolumes | bf16 | byte-exact; written only where the pack has one |
| `mlp.{gate,up}_proj` (24 x 2) | `.fconst`, MLP section, 4,288-byte subvolumes | q4_1 tiles | codes unchanged; `scale = s` (bf16), `zero = -zp*s` (bf16 rounding) |
| `mlp.down_proj` (24 x 1) | `.fconst`, MLP section, 2,176-byte subvolumes | q4_1 tiles | same as the attention tiles |
| `model.norm` (final) | — | — | **refused**, see below |
| `sin_cache_cos_cache` | `model.onnx.data` | not imported | the lane computes RoPE from `rope_theta` (`fill_rope`) |

An OGA pack keeps its tensors inside the `If` branches: the top-level graph has **no
initializers** and four nodes (`Shape`/`Constant`/`Equal`/`If`). The importer walks the
subgraphs. The 4-bit decode is `w = (code - zp) * scale` with two codes per byte (low nibble
= even k) and two zero points per byte; that decoder is checked against the pack's own data,
because `tie_word_embeddings` makes `dequant(lm_head)` reproduce the fp16 embedding:

    corr 0.999999, relative error 0.00070 on all 151936 x 896 weights
    controls: nibble-swapped corr -0.0008, no zero-point corr 0.8788   (both rejected)

Two facts bound what the container can hold, and both are reported, not hidden:

* **A bf16 zero does not always hold a f32 one.** A q4_1 chunk stores bf16 scale and zero. This
  pack's scales are already bf16 values (0 % rounded), but `-zp*s` is bf16-exact for only
  43.5 % of 32-column groups. The residual is that rounding: 3.34e-3 relative, 2.4e-4 absolute
  on the dequantized weights, with no code moved. Read back through `q4nx::dequant_tile` the
  container matches the pack to that bound.
* **K is padded to the 256-column grid.** Qwen2.5-0.5B has hidden 896 = 3.5 tiles, so
  `lm_head` and q/k/v/o are stored as 4 column tiles with the last one zero-padded (codes 0,
  scale 0, zero 0). That is 12.5 % more bytes than the pack; the true K is in the container's
  manifest and in `config.json`, and nothing reads a padding column as a weight.

### The `.fconst` attention payload

A layer block's attention payload sits between its two epsilon headers and is a whole number of
**64-row tiles**: one tile is `2*(K/128) + 2` subvolumes of 2,176 B, a pair per (column half, K
block of 128) plus the pair the writer puts its bias input in. It starts at `64 + 2*hidden` bytes
past the block's first header — the 64-byte header, then `input_layernorm` as bf16 `[hidden]` — which
reads 1,856 on a hidden-896 pack and 4,160 on Llama-3.2-1B's hidden 2,048. It is not a constant;
reading it as one made every other hidden look like "whole tiles plus a stray run" (engine#979). Inside a weight subvolume,
`[0, 2048)` are the quants (byte `(cp%4) + kk*4 + (cp/4)*512` holds rows `2*cp` and `2*cp+1` at
column `kk` of the block), `[2048, 2064)` the zero points and `[2112, 2176)` 32 bf16 scales, one
per row of the half. Codes and zero points are int4 in two's complement; the +8 an ONNX
MatMulNBits adds cancels between the two fields, so the stored nibbles are used as they are.
FACTS §7.13 established that layout by known-answer write and §7.15 against shipped bytes.

### The `.fconst` MLP section

The MLP section fills the rest of the layer block after that block's **post-attention norm**, in three parts:

| part | shape | layout |
|---|---|---|
| `gate` + `up` | ONE fused `hidden x 2*intermediate` tensor, its rows interleaved in 32-column chunks (gate 32, up 32, gate 32, …) | 128-column groups of 4,288-byte subvolumes, `2*(K/128) + 2` per group |
| — | — | one 64-byte parameter subvolume (`01 00 …`, the same bytes in every layer block) |
| `down_proj` | `intermediate x hidden` | 64-column groups of the 2,176-byte subvolume above, `2*(K/128) + 2` per group |

The 4,288-byte subvolume is the same struct at a wider `Nsubv` (64 instead of 32): `[0, 4096)` quants,
`[4096, 4160)` zero points with `(cp%4) + kk*4 + (cp/4)*512` as the quant index for `cp = (col%64)/2`,
and `[4160, 4288)` 64 bf16 scales. The writers' own headers
(`mladfQuantMatrix`/`mladfWgtSubv` in the vendor wheel's `ops_common/mladf_matmul_matrix.hpp`, and
`concat_gate_up` in `matmulbias_weights_gateup.cpp`) are what it was derived from; the geometry is not a
constant, it is a **measurement**: the three parts have to fill the block stride exactly, and if they do
not the projections are refused by name with the size they did add up to (`npu-pack info` reports the
measured unit and group counts).

Two independent decodes agree on every byte of it, which is how the layout was established without a
vendor instrument for this width (the packer segfaults on the fused `N=9728`): the decode program stores
gate/up fused at `Nsubv 64`, while the **prefill** program stores them separately at `Nsubv 32` — the same
weights in two different physical layouts, and the check between them is byte identity on codes, scales and
zero points for all 24 layers x {gate, up, down}. `tests/golden/oga_mlp_gu_group.bin` is one 68,608-byte
unit copied verbatim from the pack by `tools/oga_mlp_gu_golden.py`, with the container tiles its two
projections give, so the same decode runs in CI against the vendor writer's bytes.

**Scope:** this closed form fits all four shipped `_rai_1.8.0_npu_4K` packs exactly — hidden 896
(0.5B), 2,048 (Llama-3.2-1B and Qwen3-1.7B) and 2,560 (Qwen3-4B) — once the section starts at its
measured offset. An earlier note here said Qwen3's MLP sections "do not add up ... their writer used
different kernel shapes"; that was this reader's error, not the packs': the section was located from
the wrong boundary, so the geometry check (`total == stride`) failed and was read as a shape
difference. With §7.24's boundaries the gap closes to the byte on all four (FACTS §7.24.1).

### Which header opens the post-attention norm

A layer block opens with a 64-byte epsilon header, then `input_layernorm`, then the attention section,
then **another epsilon header** and the `post_attention_layernorm` weight. Which header that second one is
depends on how many the block carries, and that is measured per pack, not assumed: **2-header blocks**
(Qwen2.5, Llama) carry exactly one extra, so it is the block's second header; **4-header blocks** (Qwen3)
carry three more, because the first two open the per-head `k_norm` and `q_norm` sections, so the
post-attention header is the block's **last** one. Reading the second header unconditionally is correct
only on the 2-header packs — on a Qwen3 block it imports the `k_norm` gains and the padding behind them as
`post_attention_layernorm` (1,791 of 2,048 values zero and max 68.0 on Qwen3-1.7B, against a real norm
that is zero-free and 0.9919 correlated with the upstream checkpoint). Both readers — the norm and the MLP
section's first byte — now go through `FconstGeom::post_attn_hdr()` (engine#1003, FACTS §7.25).

Which tile belongs to which projection is **measured, not assumed**: on the shipped 0.5B pack the
tiles' bias subvolumes matched the pack's own q/k/v bias vectors, which identifies the runs as
k (2 tiles), q (14), v (2), o (14) — 32 tiles, the payload's whole size. A pack whose payload is
not a whole number of tiles of that hidden size, or whose q/k/v/o row counts do not add up to the
measured tile count, is refused by name with the numbers (`npu-pack info` reports the measurement).

**The section is not always one uniform run** (engine#979). The Qwen3 packs carry *four* epsilon
headers per layer block instead of two, and the three sections between them are `[k, q, v tiles at
K = hidden]` (k first — the order the pack's own biases identified on the 0.5B), then `[64-byte
header + k_norm, bf16 [head_dim]]`, then `[64-byte header + q_norm, bf16 [head_dim]]` followed by
**o_proj's tiles at K = q_proj's output width** — the pack declares
`o_proj N=2560 K=4096` on Qwen3-4B, where hidden is 2,560. So o_proj needs its own payload offset
and its own K, and the two per-head gains are ordinary bf16 `[head_dim]` RMSNorm weights (Qwen3's
are `[128]`, i.e. 256 B, which is why the section between them is 320 B on both packs although
their hidden sizes differ). `plan()` accepts a block of 2 or 4 headers, checks each section against
that arithmetic to the byte, and refuses any other count by name. FACTS §7.24 has the tables and
the checkpoint evidence.

**Every tile read is its own check.** Before anything is written, each tile is decoded and
re-encoded; the encoder has to reproduce the 34,816 bytes it came from, bias slots included. That
one comparison holds every field, index and nibble order at once, and it is why a tile whose
fields we do not understand cannot be imported silently: on `Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K`
it passes **768/768** tiles of 24 layers. `tests/golden/oga_attn_tile.bin` is one of those tiles,
copied verbatim from the pack by `tools/oga_attn_tile_golden.py`, so the same check runs in CI
against the vendor writer's bytes with no vendor stack present. The fixture's identity is
independent of this reader: the vendor packer (`matmulnbits_pack_const_float32`, `mladf_version
v2`) reproduces the shipped tile from the arrays this decoder produces.

**Refusals are named, and a refusal writes nothing.** Without `--allow-incomplete`, a pack the
importer cannot represent exits 3 and lists every tensor it refused with the reason, so a
partial model never lands in a model directory by accident. With `--allow-incomplete` it writes
what it can and records the rest in the container's `__metadata__` (`incomplete`) and in
`config.json` (`q4nx_source.incomplete`). On the shipped `Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K`
that is now **290 tensors, 67,088 tiles** (embedding, head, 48 layer norms, 96 attention
projections, 72 attention biases, 72 MLP projections) and **one refusal**: the final norm, whose
1,856-byte tail envelope turned out to be `[eps][pad][448 f32]` and not an RMSNorm weight on either
`.fconst`.

The norms that *are* imported sit at the documented offsets, each confirmed by the epsilon that
opens its layer block: **48/48 anchors found**. `tests/npu_pack_import_test` (registered as
`npu_pack_import`, and as `npu_pack_import_oga` when `-DONEBIT_OGA_PACK=<dir>` is given) checks
the decode, the tiling against `tests/golden/oga_pack_tiles.bin` — written by
`tools/oga_pack_tiles_golden.py` with the same layout `tools/gguf_to_q4nx.py` has always used —
the attention tile against `tests/golden/oga_attn_tile.bin`, and, against a real pack, the refusal
list, the tile round trip, the self-check numbers and that the container opens with `npu::Model`.

### Cross-check against the official converter

ROCm's **FLM_Q4NX_Converter** (`convert.py`) is the official gguf → Q4NX converter, and since
2026-10 it is public and Apache-2.0; `scripts/repack_gguf.py` already drives it for the
`1bit serve --device npu` repack path. It was run against `tools/gguf_to_q4nx.py` on the same
GGUF (`Qwen3-0.6B-Q4_0ctl.gguf`, qwen3, 311 tensors: 196 Q4_0 projections, Q8_0 embedding and
`lm_head`, F32 norms) and both containers were byte-compared tile by tile
(`tools/q4nx_diff.py compare`) and decoded back against the GGUF (`tools/q4nx_diff.py decode`).
Tensors: 114 of 311 byte-identical (the bf16 norms and embedding), 197 differ.

**The container format is the same; the payload is not.** Same tensor names, dtypes (`I8` /
`BF16`), shapes and tile geometry (32x256, 5120 B = 512 scale + 512 min + 4096 codes), and
`model.q4nx` headers differ only by this engine's 104-byte `__metadata__` block — the official
output's header is byte-size-identical to the shipped `Qwen3-0.6B-NPU2` container. The tiles
differ because the two encodings are different, both exactly:

| | this engine (`npu/q4nx.cpp`) | official converter |
|---|---|---|
| Q4_0 zero-point | in the min plane, `-8d` | min plane is **0** |
| Q4_0 codes | unsigned `q` (0..15) | **signed** `q-8`, nibble = `q` xor 8 |
| code order in a tile | `[row-half(2)][group(8)][col-in-group(32)][row-pair(8)]` (the 16-row split the kernel walks) | plain column major `[col(256)][row-pair(16)]` |

Measured, per tensor, against the GGUF as ground truth: **196/196 Q4_0 tensors exact (0.0 error)
under each file's own decoder** — so both converters are correct — and **0/197 exact when one is
read with the other's decoder** (worst error 2.8-3.0). The scale planes of all 53 760 Q4_0 tiles
are byte-identical; every min byte and every code byte differs. **A container written by one
converter must not be fed to the other's decoder**: this engine's ELF route decodes the former
layout only, which is also why `npu/model.cpp` and `tools/q4nx_to_gguf.py` refuse a container whose
provenance they cannot establish.

The one place both are lossy is the Q8_0 `lm_head`: neither keeps it at 8 bits, both requantize
to a 5120-byte tile. This engine's packer does it deliberately per 32 columns (max error 1.65e-2,
rms 2.28e-3 against the source); the official path runs the 8-bit codes through its 4-bit packer
(max error 3.23e-1, rms 2.94e-2, ~13x worse).

Two divergences worth knowing before packing a GDN model: the official `qwen3.5_*` configs carry a
per-tensor dtype table (Q4_K default, `ssm_out_proj` Q4_K, alpha/beta Q8_0, `lm_head` Q8_0, norms
BF16), prefill-mode `.bf16` twins for `ssm_alpha_proj`/`ssm_beta_proj`, an `(q g) -> (g q)`
reorder with `q=2` for `ssm_a` and `ssm_dt.bias`, `parallel_size` 32 and `keep_block_in_2D` true.
`tools/gguf_to_q4nx.py` has none of that (it is qwen3-only, `parallel_size` 16 semantics), so a
hybrid-GDN model — including this engine's 27B — cannot be packed by it yet; the official configs
are carried as data in `configs/q4nx/` so that gap is described rather than guessed at.

`tools/gguf_to_q4nx.py pack` now also looks for `tokenizer.json` beside the GGUF when
`--tokenizer` is omitted, and warns loudly when it is still missing: a directory without it is not
servable, and the tool used to report success anyway.

### MoE: a host reference, and the tokenizer the GGUF already carries

`tools/moe_q4nx_ref.py` is the CPU reference a MoE decode path is judged against. It reads a
Q4NX directory and nothing else, so a disagreement is the container's or the kernel's, never a
second copy of the weights: float32 throughout, RMSNorm, GQA attention with per-head q/k norm and
RoPE, and the MoE block exactly as the architecture says — float32 softmax over the bf16 router
logits, top-k, renormalised weights (`norm_topk_prob`), each chosen expert's SwiGLU streamed from
the packed tiles. The dense tensors are cached in float32, the experts are decoded per token: the
same streaming the device routes do.

| subcommand | what it is for |
|---|---|
| `tokenizer <gguf> <dir>` | build `tokenizer.json` + `tokenizer_config.json` from the GGUF's own vocabulary, merges and chat template (nothing downloaded, nothing taken from HF) |
| `logprobs <dir> --prompt ...` | one greedy completion as JSON |
| `serve <dir> --port 18110` | an OpenAI-compatible endpoint with `logprobs`, so `tools/logprob_agree.py` can compare it with another route |
| `agree <dir> --route URL` | the same 8 prompts, but **teacher-forced**: the other route's own generated ids are scored by this forward, so every position it produced is compared, not only the positions where both routes happened to emit the same token |
| `selftest` | the numerics a bug would silently pass |

On `Qwen3-Coder-30B-A3B (Q4NX)`: for *"What is the capital of France? Answer in one word."* it
returns `Paris`, with the top-3 next-token ids **identical** to the HRX route on the same GGUF
(59604 / 12095 / 106004), the runner-ups differing by 0.33 nats at p ≈ 1e-6 — the Q6_K → q4_1
requantisation of `down_proj`, `v_proj` and `lm_head` and nothing else. It is a **reference, not a
route**: 6 s/token, and a 20-token prompt costs ~5 minutes, because a prompt's 160 expert
selections per layer touch nearly all 128 experts and stream ~18 GB. MoE prefill belongs on the
GPU route (`--moe-prefetch`, docs/serve.md); this is the thing that decides whether a kernel is
right.

## Serving

The 1bit NPU engine runs the model on the XDNA 2 NPU. With a release package that carries the NPU
kernels (dense Qwen3: 0.6B, 1.7B and 4B):

```
./1bit serve -m <Qwen3 NPU model dir> --port 8000
```

In Lemonade, the `onebit` recipe's `npu` backend serves the same directories (`--onebit npu`).
The model directory holds the weights only; the kernels ship inside the package.

### Sealed kernel sets: the manifest is what says a set is ours

A set ships as `<set>/layout.json`, `<set>/layer1_*/layer1.elf` (each `1BPKCRYP` ciphertext, opened
in memory by the loader with the kernel key) and two more files: `manifest.json`, which lists every
file the set carries with the sha256 of its **ciphertext**, and `manifest.sig`, an Ed25519 signature
over the manifest's exact bytes made with the release signing key. Sealing on its own is
confidentiality only: a set re-sealed by another key, or swapped for a different set that also
decrypts, would be indistinguishable from ours. The signature covers the manifest, the manifest's
digests cover the ciphertext files, and the release public key never leaves the loader.

`tools/seal_manifest.py` is the public half: `write <set> --key <release key>` emits both files,
`check <set>` proves the manifest lists exactly the set's files with their sha256 (no key needed),
and `verify <set> --pub <release public key>` proves the signature too. `scripts/package-linux.sh`
refuses to pack a set without them, `scripts/weekly.sh`'s `package_checks` refuses a tarball whose
sets do not match their manifests (`ONEBIT_SEAL_PUB=<release public key>` also verifies the
signatures there), and `1bit-release-check`'s `sealed`/`sealed-sig` rows prove the same on the stick
(the signature where the image has an interpreter and the key). The loader-side requirement,
`ONEBIT_KCRYPT_REQUIRE_MANIFEST=1` beside `ONEBIT_KCRYPT_REQUIRE_SEALED`, is read by the private NPU
runtime and lands with the pin that makes it honour one - engine #574.

### Dense Qwen3, checked on the NPU

All three dense Qwen3 models run on the NPU from a sealed kernel set, and each one passes the
hardware gate's `npu` phase ([hw-gate.md](hw-gate.md)): greedy decoding token for token against an
independent fp64 reference over a short prompt and teacher-forced steps, the 64-position build,
the prompt prefill, and byte-identical output when the run is repeated, with the phase's speed
floor met. Each is a catalogue entry in Lemonade, and each answers there:

| Model | `npu` phase | Decode tok/s | Lemonade entry | `chat-short` tok/s |
|---|---|---|---|---|
| Qwen3-0.6B | PASS | 90.7 | `Qwen3-0.6B-NPU-1bit` | 93.7 |
| Qwen3-1.7B | PASS | 39.5 | `Qwen3-1.7B-NPU-1bit` | 41.3 |
| Qwen3-4B | PASS | 16.8 | `Qwen3-4B-NPU-1bit` | 17.7 |

Measured on Strix Halo, 90 W cap, PCM pad (23 Oct). Decode tok/s is the gate phase's own
64-position run; the phase's speed floor is set per model, 60 tok/s at 0.6B and scaled down by the
model's size for the two larger ones. `chat-short` is `lemonade bench --scenarios chat-short --runs
3` (mean of three runs, 0 failed) on a private server with one model loaded at a time. The checker
that runs the phase is private, like the kernels it runs.

- **Tokenizer:** `npu/tokenizer.cpp` reads `tokenizer.json` directly (PCRE2 for the
  pre-tokenizer regex). Added tokens with `special: false`, such as Qwen3's `<think>` and
  `</think>`, stay whole, as Hugging Face `tokenizers` does.
  `tests/data/qwen3_tokenizer_golden.json` holds 14 cases encoded by `tokenizers` 0.23.1, and
  all 14 match, ids and round trip.
- **Endpoints:** `/v1/health` (503 while loading), `/v1/models`, `/v1/chat/completions`
  (streaming or not) and `/v1/completions`.
- **Prompt:** ChatML, Qwen2/Qwen3's template for text messages.
- **Decoding:** greedy, with a 1.1 repetition penalty over the last 64 ids by default.
  `repetition_penalty` in the request overrides it, and 1 gives plain greedy.
- **One request at a time:** the model has one KV cache.
- **Repeatable:** every generation starts from a zeroed KV cache (#400), so identical greedy
  requests give identical text, whatever ran before them.

### The dense FFN tail cache (hybrid GDN models)

On a GDN model the guard that skips the per-token Q4NX dequant in `step()` can never fire
(`dense_weights_cached` requires `lin_kd_ == 0`, and `lin_kd_` is model-level), so a hybrid model
re-dequantised each layer's 17408-wide dense FFN tail on every token even after those weights were
packed on the device. `ffn_tail_cached()` (`npu/forward/src/q4nx_forward.cpp`) closes that: once a
layer's packed gate/up/down BOs are resident, the f32 `wg_`/`wu_`/`wd_` buffers are no longer read,
so the reload is skipped. It is on by default for a GDN model on the NPU route, when the packed
weight cache is on, the run is not `NPU_INFER_HOST_GEMM`, and the tail is at least 256 MB per layer
(the size guard; below that the reload is too small to be worth a behaviour change). `NPU_FFN_CACHE=0`
or `=1` forces it either way, and the resolved state is logged once as
`[ffn] dense-tail cache on|off (default|NPU_FFN_CACHE; lin_kd=… IM=… H=…)`. On Qwen3.8-27B-NPU2 an
interleaved 5-round A/B (`~/lb/587-cache-20261030`) measured an 8-token request at 104.5 s against
117.4 s (**−11%**), CPU per token 22.9 s against 63.3 s (**−64%**), same text in every arm.

### The prefix cache's snapshot: two kinds of plane

`snapshot_state`/`restore_state` (`npu/forward/include/q4nx_forward.h`) hand a prefill's state to the next
request that shares its prompt (`app/npu_prefix_cache.h`, issue #650). A snapshot is not just the KV: a GDN layer
also carries a conv window and a recurrent state, and the two kinds of plane are cut differently.

- **Per position** (`kcache_`, `vcache_`): a layer holds `[NKV][HD]` floats for every position it has seen, so a
  snapshot of the first `tokens` positions keeps exactly `tokens * NKV * HD` floats of each.
- **Whole** (`conv_state_`, `rec_state_`): the conv window and the recurrent state are one fixed block per layer
  that each position *updates*, not appends to, so they are carried and restored whole at any prompt length.

Both live in `npu/forward/include/q4nx_state_planes.h`, header-only and unit-tested without a device
(`tests/q4nx_state_planes_test.cpp`, ctest `q4nx_state_planes`). Cutting the whole planes with the per-position
count was the shipped behaviour until 2026-11-04, and it cost the cache every request: on a GDN model the conv
plane (30,720 floats for Bonsai-2-27B) can never cover `tokens * 1024`, so `restore_state` refused every snapshot
of a prompt longer than 30 tokens and the serve logged `[prefix] slot=-1 cached=0 … reason=slot has no usable
state` and re-prefilled from 0 — 5 of 5 long requests measured, and a repeat of an 849-token prompt paid its full
prefill twice. Shorter prompts had the mirror-image problem: below 768 tokens the 786,432-float recurrent plane
was silently truncated (23 % of it kept at 180 tokens). A truncated plane is now *refused* rather than installed
short, so the fallback is a full prefill instead of a wrong state.

The whole planes also fix where the snapshot may be *taken*: they are updated in place, so there is no "restore
these positions of that state". `forward-serve` therefore saves a prompt's state one token short of the prompt —
splitting the final prefill chunk, one extra launch — and the resume point is `cached - 1`, running the last token
again to get the logits the answer is drawn from. Restoring `cached - 1` positions out of a snapshot taken at
`cached` is what the KV alone would allow, and it applies the last token's GDN update twice: measured on the first
version of this fix, the cache hit (`[prefix] slot=0 cached=179 of 180 reason=hit`, TTFT 5.3 s against 15.6 s) and
answered the same prompt with different, fragmented text.

### Sampling and structured output

`1bit serve -m <NPU model dir> --device npu` serves through `1bit forward-serve`
(app/forward_serve.cpp) in the same process. Every field below is optional, and the answer is
**greedy when none of them is sent**, so a client written before them gets the same text it
got before.

| Request field | Meaning |
|---|---|
| `temperature` | `0` or absent decodes greedily (argmax); above `0` divides the logits before the draw |
| `top_k` | keep the `k` highest logits; `0` or absent keeps all of them |
| `top_p` | nucleus sampling, `1` or absent keeps all of them (`min_keep` candidates always stay) |
| `seed` | the same seed and the same request give the same text; absent or `-1` seeds from the clock |
| `logprobs` | `true` reports each generated token's log-probability on `/v1/chat/completions` (streamed or not); the values are the raw distribution, before `top_k`/`top_p`/`temperature` and before any grammar mask, as OpenAI and vLLM report them |
| `top_logprobs` | `0`-`100` alternatives per token, highest raw logit first; needs `logprobs: true` |
| `stop` | up to 4 strings: generation stops before the first match, and the tail is held back so a stop split across two tokens still matches |
| `chat_template_kwargs` | `{"enable_thinking": false}` makes the assistant turn start with the empty `<think>\n\n</think>\n\n` block Qwen3's own template emits, so the answer goes to `content` instead of into a `<think>` block (and its tokens get logprob entries). Default `true`; a non-boolean is a 400 |
| `grammar` | a GBNF grammar in llama.cpp's format (`third_party/llama.cpp/grammars/*.gbnf`) the answer cannot leave |
| `response_format` | `{"type": "json_object"}` constrains the answer to JSON; `{"type": "json_schema", "json_schema": {"schema": …}}` constrains it to that JSON Schema |

`logprobs` keeps OpenAI's shape: non-streamed, every entry is `choices[0].logprobs.content[i] =
{token, logprob, top_logprobs: [{token, logprob}]}`; streamed, each chunk carries the entries whose bytes that
chunk carries, so the deltas add up to the non-streamed reply. Token text that is half a character is reported with the invalid bytes replaced, never as a failed
request. Entries exist for **`content` tokens only**: a reasoning delta and tool-call markup get no
logprobs entry, because `app/npu_answer_parser.h` says where the content boundary is. On a Qwen3 model that
means a thinking reply reports none at all, which is why `chat_template_kwargs.enable_thinking: false` is
what a logprob comparison wants. The `/v1/completions` route's legacy logprobs object
is not implemented and asking for it there is a 400, not a silently dropped field.

`json_schema` is compiled to a grammar before the first token, by `app/npu_json_schema.h`, so the answer
cannot leave the schema while it is being written rather than being validated afterwards. The covered subset
is OpenAI's structured outputs: objects (`properties`, `required`, `additionalProperties: false`), arrays
(`items`, `minItems`, `maxItems`), `string`, `number`, `integer`, `boolean`, `null`, a type list such as
`["string", "null"]`, `enum`, `const`, and `oneOf`/`anyOf` as alternation. Anything outside it (`pattern`,
`format`, `minimum`/`maximum`, `$ref`, `allOf`, …) is a 400 that names the keyword: a constraint quietly
dropped from a schema would leave the caller believing the answer was validated. Properties are generated in
the schema's order with the required ones first, so the object's key order is fixed for a given schema.

A field the route cannot honour is a 400 whose `error.message` says why ("`json_schema`: pattern is not
supported by this route's schema-to-GBNF converter", "`gbnf: unterminated literal at line 1`"),
never a silently ignored key. `finish_reason` is `length` when `max_tokens` ran out, and `stop` for
an end-of-sequence token, a stop sequence or a grammar that allows nothing more.

The sampler is llama.cpp's, without llama.cpp: `app/npu_sampling.h` repeats the semantics of
`third_party/llama.cpp/src/llama-sampler.cpp` (top_k, then top_p, then temperature, then the
cumulative-sum draw that `common/sampling.cpp`'s chain ends with) over the raw logits this route
has, because llama.cpp's sampler needs a `llama_context` and a `llama_vocab` and the forward has
neither. The grammar (`app/npu_grammar.h`) is the same rule model - one flat element vector per
rule, `ALT`/`END` between alternatives - as a byte-level pushdown automaton: a token is only
sampled when its decoded text leaves the grammar alive, which is checked per token against the
byte ranges the grammar can continue with. Both headers are device-free, and
`tests/npu_sampling_test.cpp` runs the sampler, the parser, the JSON grammar, the schema converter and the
request parser in CI without a model or an NPU.

```
$ curl -sS localhost:8000/v1/completions -d '{"prompt": "The capital of France is", "max_tokens": 16,
    "grammar": "root ::= \" Paris\" | \" Lyon\" | \" Marseille\""}'
{"choices":[{"finish_reason":"length","index":0,"text":" Paris"},...]}
```

`prompt` may also be the **token ids themselves**, the shape llama.cpp's `/completion` accepts:

```
$ curl -sS localhost:8000/v1/completions -d '{"prompt": [151646, 785, 6722, 315, 9625, 374], "max_tokens": 16}'
```

Supplied ids are used exactly as given — nothing is tokenized and no BOS is inserted — which is what a
parity run against llama.cpp needs: the same *text* reaches the two engines through two tokenizers and one
prompt-construction rule each, so only supplied ids make the comparison about the forward. `151646` above is
*Zyphra/ZR1-1.5B's* BOS id; another container's BOS is a different id, and pasting the wrong one is the
out-of-vocabulary 400 below rather than a silently different prompt. An id outside the vocabulary, a
non-integer element, or an empty array is a 400 that names the position, never a clamped or dropped one.
OpenAI's array-of-strings batch form is not implemented and is refused by the same rule.

### Decisions (`/v1/systemone`, issue #653)

A **Decision 3.0 (d3) pack** does not generate text: you send a `state` and one or more questions and get
a probability for every option. When the model directory carries the pack's decision head, `1bit
forward-serve` serves that too, from the same process and the same weights:

```
<model dir>/model.q4nx            the Qwen3.5 GDN backbone, as any other model
<model dir>/decision_config.json  the pack's codes, token ids, temperature, pooling
<model dir>/decision.q4nx         the readout, [codes, hidden] F32 (tools/safetensors_to_q4nx.py decision)
```

| route | meaning |
|---|---|
| `POST /v1/systemone` (alias `POST /v1/decisions`) | `{"state": …, "questions": {id: question}}` → `{"model", "answers": {id: answer}, "usage"}` |
| `GET /v1/systemone/models` | the decision model this build serves: name, description, `max_input_tokens`, `modalities` |

`questions` is an **object**, and its ids come back as the keys of `answers` in request order. Question
types are `choice` (1–255 options), `noul` and `score` (2–10 ordered levels); a decision runs one backbone
prefill per question, pools the **last position's hidden state after the final norm** (`pooling: last`),
multiplies it by the readout and takes a temperature softmax over that question's options. Nothing is
sampled and nothing is generated, so `usage.output_tokens` is always `0` and `input_tokens` is the sum of
the answered questions' prompt lengths.

The prompt is the **pack's own contract**, not this route's ChatML: `npu/systemone.h` renders the
`State:`/`Question:`/`Options:` turn byte for byte from the pack's `d3_format.py` (checked against
fixtures it generated, `ctest systemone_contract`), and `app/npu_systemone.h` wraps it in the pack's chat
template with `enable_thinking=False` — which opens the assistant turn with an **empty think block**.
Those trailing bytes are part of what the model reads, so reusing `chatml()` here would be a different
input. `state` is parsed as ordered JSON because upstream preserves the request's key order.

**Converting a d3 pack** (engine #796) is `tools/safetensors_to_q4nx.py pack <pack> <model dir> --arch qwen3_5`
(the arch is detected from the pack's `language_model.*` names, so `auto` is enough). It is not a rename
job: three transforms in the pack's GDN tensors are silently wrong answers when they are wrong, so each is
named here and covered by a known-answer test (`ctest safetensors_to_q4nx`):

| from | to | transform |
|---|---|---|
| `linear_attn.A_log` | `linear_attn.ssm_a` | **`-exp(A_log)`, F32** — the forward computes `expf(ssm_a * softplus(dt))`, so the raw A_log decays backwards |
| `linear_attn.conv1d.weight` `[CD,1,CK]` | `linear_attn.ssm_conv1d.weight` | **transpose to tap-major** `[CK,CD]`: the forward indexes `conv[j*CD + c]` |
| `linear_attn.in_proj_a/b` | `ssm_alpha_proj`/`ssm_beta_proj` | rows **zero-padded to 32** so they tile; the forward dequantises at the stored height and keeps the first NVH rows (**q4_1 where the FLM config asks Q8_0** — a deviation `verify` measures rather than hides) |
| `self_attn.q_proj.weight` (only when `attn_output_gate`) | same name | rows **de-interleaved**: HF builds a gated layer's query as `view(*, -1, 2*head_dim)` + `chunk(2, -1)`, i.e. `[q_h \| gate_h]` **per head**, while the forward's fused QKV is `[q block \| gate block \| k \| v]` — the layout the FLM/OGA packs already use. Leaving the source order made the forward attend over half-gate rows and gate with half-query ones: measured on the real d3 pack, engine q vs the reference's q **cos 0.077**, and **0.999992** against the de-interleaved rows |

The remaining names come from `configs/q4nx/qwen3.5_0.8b.json`'s `q4nx_name` column (`in_proj_qkv` →
`qkv_proj`, `in_proj_z` → `self_attn.gate_proj`, `out_proj` → `ssm_out_proj`, `dt_bias` → `ssm_dt.bias`), and
`visual.*` is skipped because the forward runs the text path. Two converter rules exist because of measured
load failures, not style: the embedding keeps its **2-D shape** (the loader reads vocab and hidden off
`shape[0]`/`shape[1]`), and `text_config`'s keys are **lifted to the top level**, because the engine reads
`config.json` at depth 1 only. A tensor with no mapping is a refusal, never a dropped one.

`--dtype bf16` writes the same tensor set with **no q4_1 tiles at all** (every tensor keeps its source
bf16, or the F32 a transform forces; `verify` reports 0 error rather than a loss). It is the container for
*measuring*, not for serving: it holds the weights fixed so an engine run and a reference run differ only in
the forward, and the cost of quantization becomes a separate delta instead of an unexplained one. The
`d3-lite` parity row is measured that way (`--dtype bf16` vs the default `--dtype auto` container, same
requests, against the pack's own `d3_runtime.py`).

What is still missing for a real d3 row is the **artifact set** — but only for a *device* row. `NPU_INFER_HOST_GEMM=1`
opens no device, so it needs no ELF: the backbone loads, answers, and serves as the numeric reference for a geometry
whose kernels do not exist yet (this is how `d3-lite` is exercised before its six designs are built). A device row
needs all six — `QKV K1024 N5120` (q|gate|k|v, `q_proj` carries the output gate), `O K2048 N1024` (GDN's `ssm_out`
reuses it), `GU K1024 N7168`, `D K3584 N1024`, GDN `QKV K1024 N6144` and `Z K1024 N2048` — and without them the
device path stops at `forward init failed: … this model's plan needs the QKV kernel "full_i8_QKV_K1024_N5120.elf" …
(missing)`.

**The norm convention is decided by the weights, not the name.** Qwen3.5's `Qwen3_5RMSNorm` is *zero-centered*:
its weight is initialised to **0** and applied as `x/rms * (1 + w)`, so a checkpoint stores small deltas (the
d3-lite pack's layer-0 `input_layernorm` has mean **0.2377** and negatives, min **-0.1143**). Older hybrid packs
declare the same `model_type: qwen3_5` but keep the plain form, `x/rms * w`, with gains trained from 1 (the 27B
packs on this box: mean **0.9704**, min **0.8711**, every value positive). Applying the wrong one is not a small
error — it scales every norm output by roughly the weight's own magnitude — so `q4nx_semantics` sets a *probe* for
`qwen3_5*` and the forward resolves it from the loaded layer-0 `input_layernorm.weight` (a negative value, or a
mean below 0.5, is the zero-centered form), printing the evidence on the `[semantics]` line. The GDN gated norm
(`linear_attn.norm`, `Qwen3_5RMSNormGated`) is **exempt**: it is initialised to ones and applied as `w`, which is
what both packs store (mean 0.95 / 0.86). `ctest npu_rmsnorm_offset` covers the rule with both real shapes.

Refusals follow the d3 server this route stands in for, so vLLM Semantic Router can point at it unchanged —
which means **422 here, not the 400 the chat route uses**:

| situation | answer |
|---|---|
| body is not a JSON object, an unknown field, `questions` missing/empty/not an object, an empty id | `422 {"detail": "<why>"}`, nothing runs |
| any question is malformed (bad type, empty choice criteria, score levels out of range) | `422 {"detail": {"invalid_questions": {id: "<why>"}}}`, **nothing runs** |
| a question is valid but its prompt is over the limit | inside `answers`: `{"type", "error": "max_length_exceeded", "message"}`, the other questions still answered, **nothing truncated** |
| the readout could not turn the hidden state into an answer | inside `answers`: `error: "invalid_model_output"` |
| `DECISION_API_KEY` is set and the bearer key is absent or wrong | `401`, `WWW-Authenticate: Bearer` (`/health` stays open and reports `"authentication": true`) |

The limit is the smaller of the text path's 32,768 tokens per question and the backbone's own context.
`GET /health` gains `model`, `max_input_tokens`, `modalities` and `authentication` when a head is loaded, and
`/v1/models` keeps its OpenAI shape (the decision list is at `/v1/systemone/models`).

Not served, and refused by name rather than ignored: `images`/`videos` (the vision tower and FFmpeg are a
separate issue), so `box`/`track`/`assets` questions cannot be asked. Also not implemented: upstream's
"state computed once per request" optimisation (a state read once and shared by every question — it is the
prefix cache's job, #650) and the pack's `noncausal_full_attention` attention mode, so a d3 row measured
here is not claimed to be upstream-equal in accuracy until it is measured against it.

Send the body as JSON (`Content-Type: application/json`): httplib parses an
`application/x-www-form-urlencoded` body as a form and caps it at 8 KB, which is a 413 before this route
sees it — that applies to every route here, and a real d3 caller (a 32k-token state is ~100 KB) hits it
only with a client that leaves the content type at curl's default.

The shape of a reply (**numbers illustrative** — the choice depends on the pack's readout head):

```
$ curl -sS localhost:8000/v1/systemone -H 'Content-Type: application/json' \
    -d '{"state": {"route": "refund", "amount": 42},
         "questions": {"route": {"type": "choice", "criteria": {"fast": "handled today", "cheap": null}}}}'
{"answers":{"route":{"choice":"fast","confidence":0.81,"probabilities":{"fast":0.98,"cheap":0.02},
  "type":"choice"}},"model":"d3-lite","usage":{"input_tokens":96,"output_tokens":0}}
```

### Two streams on one NPU

One process owns the NPU by default: it takes an exclusive `flock` on
`/tmp/1bit-npu-device.lock` for its lifetime (`app/npu_device_lock.h`), so a second serve of the
NPU route waits instead of contending. That is a policy, not a hardware limit -- the driver
advertises 16 hardware contexts on this part (`hwctx_limit = 16`, `drivers/accel/amdxdna/npu4_regs.c`)
and only refuses the 17th -- and the policy stays the default because a second stream buys much less
than 2x:

- **The DMA engine is shared.** Decode reads the model's int8 weights once per token (~24.3 GB at
  ~2.2 GB/s effective on the 27B, [engine#587](https://github.com/1bit-MONSTER/engine/issues/587));
  two streams split that bandwidth instead of doubling it.
- **The weights are not shared.** Each stream packs its own copy of every layer's weight BO from the
  model file, plus its own KV cache and scratch, so N streams cost N x the model's device footprint.
  The NPU's memory is the box's own LPDDR5X, the same pool the CPU and the iGPU draw from, so two
  27B streams do not fit.

With that said, two streams do run, and the option is what says you meant it:

```
# stream 1 of 2, then stream 2 of 2, in two shells (different ports, same model)
./1bit serve -m <NPU model dir> --device npu --npu-streams 2 --port 8000
./1bit serve -m <NPU model dir> --device npu --npu-streams 2 --port 8001
```

`--npu-streams N` (1..16, default 1) is `forward-serve`'s flag too: N = 1 keeps the exclusive lock,
N >= 2 takes a **shared** lock, so the opted-in streams co-exist while any exclusive holder -- an
unflagged `1bit serve`, the per-process ELF runner (`npu_engine_universal`), another build's serve --
still keeps them all out, in either order, because they share one file and one `flock(2)`.

**The lock is one file and one protocol, across runtimes, and it is advisory** -- that is what makes it
usable beside another NPU engine, and what its limits are:

- **Path and modes.** `/tmp/1bit-npu-device.lock`, created `0644` if absent, `flock(2)` (not `fcntl`)
  so it works across processes, users and runtimes on the box. `LOCK_SH` = "I am one of N streams, do
  not wait for other shared holders"; `LOCK_EX` = "I want the device to myself". A holder keeps its fd
  open for the life of the process; closing or exiting releases it. `LOCK_EX` and `LOCK_SH` never
  overlap, in either order, on one file.
- **Who may take what.** Anything that honours the file interlocks with us: our serves
  (`--npu-streams` decides which mode), the ELF runner (`LOCK_EX`, always). A foreign NPU runtime --
  Lemonade's `ryzenai-llm` backend on Windows, an ONNX session, a benchmark -- may take `LOCK_EX` to
  keep us out, or `LOCK_SH` to be one of the streams. Nothing forces it: a runtime that ignores the
  file is not slowed, not blocked and not detected.
- **What happens when the other engine ignores it.** A runtime that skips the file is not blocked and
  not detected, so it shares the device exactly the way a *co-ordinated* second engine does — and that
  cost is measured: two engines on the shared lock lose 59 % of their aggregate throughput (the table
  below). The failure mode is slow, not wrong; there is no run on this box that measures an
  unco-ordinated second engine apart from that, so nothing more specific is claimed here.

**Partitioning, and what it would take.** Two whole-array streams share one DMA engine, so they
serialise; the September prototype that scaled (~100 % each) did it by giving each stream a *disjoint
half of the columns*, which halves per-stream weight reuse, and its xclbin pair is ABI-stale at this
pin, so nothing in this tree can select a column split today: both of our instances run whole-array.
What each engine costs is therefore the whole model per stream -- **one hwctx and one complete copy of
the weight BOs, plus its own KV cache and scratch**, and the device's hwctx limit is 16. The accounting
per engine is: `streams x (weights + KV + scratch)` must fit in the NPU's share of the box's LPDDR5X
alongside whatever the other engine has already allocated, and both engines pull from the same DMA
bandwidth. Two *different* models resident at once (the reason the second engine exists) is therefore
bounded by memory first, not by hwctx: see the table below.

A model directory repacked from an FLM pack still carries FLM's own parameter map and version in
`config.json` -- for Qwen3-0.6B, `addr_qk 9216`, `addr_kv 33792`, `addr_kk 45056`,
`addr_l_begin_mha 53760`, `addr_l_end_mha 25088`, plus `flm_version 0.9.22`. Those are offsets into
FLM's program layout for the AIE programs that live in the FLM install, not a description of this
engine's memory: this route reads only the geometry keys (`hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `head_dim`) and finds
its own full ELFs under `--kernels`, which is why a model directory's `npu/` subdirectory is empty on
purpose and why `addr_*` must not be used to reason about what this engine allocates.

What it measures. `max_tokens 8`, `temperature 0`, three reps per arm with rep 1 dropped as warm-up
(the first request after a load is ~2x the steady state), wall clock per request, each model measured
alone with the other not even loaded (strixhalo 2026-10-30, engine `38e29a2`, binary `b78868c`,
kernels `~/lb/lfm2-npu/kernels_m16`, run `~/lb/612-npu-streams-20261030T112326Z`):

| run | solo | both at once | per stream | aggregate |
|---|---|---|---|---|
| Qwen3-0.6B + Qwen3-4B, `--npu-streams 2` (`b78868c`, 2026-10-30) | 5.79 s / 25.08 s | 16.99 s / 35.61 s | **34.1 % / 70.4 %** | **0.41x** |

That row is the one the second engine exists for: two *different* models resident at once, each
keeping its own weights and its own hwctx, both answering, with no lock bypass. Throughput gets
**worse** -- 0.41x, i.e. running the two back to back is 2.4x faster than running them together --
because two whole-array streams share one DMA engine: the per-token times add up (0.72 + 3.13 s solo
per token = 3.85 s; concurrent rounds take 4.45 s) plus about 16 % of handoff overhead. The small model
pays most (34 % of solo) because it is latency-bound; the 4B keeps 70 % because its token cost is its
own 3.3 GB weight stream, which it would have spent anyway.

So co-residency buys **availability, not throughput**: each model keeps answering while the other is
loaded, instead of a 25-30 s reload. For a short answer, reloading still beats sharing the device --
that is the trade, and it is why the exclusive default stays.

`tests/npu_device_lock_test.cpp` (ctest `npu_device_lock`, in CI) proves the lock discipline without a
device: exclusive excludes everything, three shared streams co-exist, and an exclusive waits for the
last one to go.

**The row this exists for.** On Linux the two engines are two instances of this runtime on two
*different* models at the same time -- "all NPU models at once" -- and that is what
`tools/showcase-612-npu-streams.sh` measures with `MODEL` and `MODEL_B` (default Qwen3-0.6B and
Qwen3-4B, whose kernel sets are on this box): each model solo, then both at once, three reps, every
number stamped with the build that produced it.

## Add-on routes

The engine keeps a hook for closed-source NPU add-ons (`npu/private_route.h`). Qwen3.6-35B-A3B
(`model_type` `qwen3_5_moe`) runs on the NPU through one, experimentally; it is not in this
repository and not in a release.

- `-DONEBIT_NPU_PRIVATE=<add-on checkout>` (with `-DONEBIT_NPU=ON`) builds that checkout's
  `addons/` into `1bit`. It must define the target `onebit_npu_private`, which holds
  `register_private_addon()`; `1bit` calls it at startup.
- The add-on registers a route per `model_type` (whether it can serve a directory, and a
  session that generates from token ids) and may register `1bit` subcommands. `1bit serve`
  uses the route for that model type; `--npu-opt KEY=VALUE` passes options through to it.
- A route can be `optional`: an opt-in route for a `model_type` the engine also serves. When it
  declines a directory (its `unavailable()` gives a reason), the engine serves the directory
  exactly as it would without the add-on.
- Without the add-on, `1bit serve -m <35B dir> --device npu` exits saying the route is not part
  of this build.

An add-on may also register an encoder for the Laya router (`laya/encoder.h`,
[laya.md](laya.md#the-encoder-on-the-npu-add-on)). `tests/npu_private_route_test.cpp` (ctest
`npu_private_route`, in CI) checks the registry and that message.

## The runtime

The engine reaches the NPU runtime only through its own public header, `npu/npu_rt_api.h`: the
calls the engine's NPU code makes, each class an opaque handle. The runtime itself is not in this
repository; a build from this repository alone has no NPU lane. `ONEBIT_NPU=ON` takes one of:

```
# the public binary drop (docs/releases.md, "The public binary drop")
cmake -B build -G Ninja -DONEBIT_HRX=ON -DONEBIT_NPU=ON -DONEBIT_BINARY_DROP=<drop dir or URL>
# a prebuilt runtime library that implements npu/npu_rt_api.h
cmake -B build -G Ninja -DONEBIT_NPU=ON -DONEBIT_NPU_RT_LIB=<path>/libnpu_rt_api.a
```

- `ONEBIT_NPU_HOST` (the model reader, Q4NX, packing, the tokenizer) needs no runtime and builds
  everywhere, including public CI, where its tests run.
- The release builds the runtime from a pinned commit named in `config/npu-runtime.config`
  ([releases.md](releases.md#private-components)).

The NPU needs the `amdxdna` kernel driver. `third_party/xdna-driver` pins upstream
[amd/xdna-driver](https://github.com/amd/xdna-driver) as the driver version the engine is checked
against (the kernel's own copy also works); nothing from the pin is built into or shipped with
the release. `.github/workflows/bump-xdna.yml` moves the pin when run by hand, because a driver
move means rebuilding the module on Strix Halo and checking the NPU there first.

### The MTP head (FastFlowLM's container -> ours)

`tools/flm_head_import` (built as `flm-head-import`) turns a FastFlowLM MTP-head container
(`model.mtp.q4nx`) into a sidecar directory of ours. The head is one Qwen3.5 full-attention layer plus an input
fusion; the base model keeps its embedding, lm-head, tokenizer and RoPE, so this artifact is only the head.

The container is safetensors-shaped and its quantised tensors are rank-3 `[nrb, ncb, 4736]` with dtype `"I8"`.
That 4,736-byte cell **is our Q4_K chunk**: `u8 scale[g*32+n]` at 0, `u8 min[g*32+n]` at 256, nibbles at 512
(byte `k*16 + n/2`, even n in the low nibble), `bf16 S[n]` at 4608, `bf16 M[n]` at 4672, and
`w = q4 * (scale*S) + min*M`. Measured, not assumed: `q4nx::dequant_tile(Chunk::Q4K, …)` reproduces FastFlowLM's
own decoder **bit for bit** on six cells recorded from six of the head's matrices (`tests/golden/flm-i8`, floats
recorded from their published library; the layout facts are in amd-oss/npu-re
`docs/FACTS-flm-q4nx-container-and-mtp-head.md`). Two things follow, and they are the reason this tool is small:

* the default form copies every cell **unchanged** — no requantisation, nothing to lose, and the import
  self-check re-decodes each matrix from the source and from the copy and compares;
* the alternative `--form q4_1` writes our mode-1 chunk (5,120 B: `bf16 S[256]`, `bf16 M[256]`, 4,096 code
  bytes, `w = code*S + M`) through `q4nx::q4k_to_q4_1`, whose only loss is rounding `S*scale` and `M*min` to
  bf16. The tool prints that factorisation error rather than hiding it (measured on the 27B head: 0.67 % of
  values unchanged, max |delta| 1.2e-3, mean |delta| 6.8e-5).

    flm-head-import model.mtp.q4nx -o head-q4k               # cells copied unchanged
    flm-head-import model.mtp.q4nx -o head-q4_1 --form q4_1  # our mode-1 chunks

`tools/flm_head_pb16i.py` writes the third form, the `pb16i` image (`npu/forward/include/q4nx_pb16i.h`), and it
is the one form this head cannot have exactly: `pb16i` carries one int16 magnitude per (32 K, column) with no
per-group bias, while the source blocks are asymmetric (`w = q4*S + M`) and their magnitudes span about 100x
inside a column, where an int16 with an 11-bit mantissa allows about 16x. Forcing it costs a measured 11-18 %
relative L1 residual, so the mode-1 or Q4_K forms are the ones to run; the script keeps the measurement next to
the images it writes.

## FFN row sparsity: a stream question, answered before a kernel is written

The FFN is the largest part of a dense model's per-token stream, so "skip most of the FFN" is a
recurring idea. It only pays if the rows can be skipped **in the blocks a streaming kernel can
address**: a token's `h = silu(gate(x)) * up(x)` is selected row by row in any analysis, while a
kernel reads whole blocks and cannot skip a scattered row. `tools/ffn_sparsity.py` is the
device-free pair of measurements that decides which of those two answers a model gives, and it
runs in CI (`ctest -R ffn_sparsity`).

**`budget <model.gguf>`** reads the GGUF's tensor table with the standard library only — no
dequantising, no model in memory — and reports what share of one token's stream is
`ffn_gate_up`, `ffn_down`, `attn`, `ssm_gdn`, `embd_head` and `norms`, with `--per-layer` for the
block-by-block split and `--json` for a record. The FFN share is the **ceiling**: any row-skipping
kernel saves a fraction of that, never more.

**`stats <capture.csv>`** summarizes a capture of the `ffn_down` input: one CSV row per
(layer, token), with the columns in two families, because they mean opposite things.

| columns | rule | who can use it |
|---|---|---|
| `b64_e90` … `b128_e99`, `b<B>_em10` … `b<B>_em50` | keep the whole blocks of `B` rows whose own `\|h\|_1` is largest | **this is the rule a streaming kernel obeys** |
| `r90`, `r95`, `r99`, `m10` … `m50` | plain row selection: rows needed for 90/95/99 % of `\|h\|_1`, mass of the top 10/20/30/50 % | nothing that streams blocks — it is the upper bound, not an achievable one |

Read the block rows first. A model whose `r90` looks promising and whose `b64_e90` is close to
100 % is the common case, and it is a no: the intermediate is *spatially* dense, so the saving is
spread over every block the kernel has to read anyway. The per-layer table (`--every N`) shows
whether a minority of layers behaves differently, which is the one shape a per-layer policy could
still use.

A capture is any harness's output in that shape, so the columns are the contract rather than a
pinned generator: the tool drops a half-written last line (a killed capture), refuses a CSV with no
measurable column instead of printing an empty table, and exits non-zero on a malformed one. The
`--self-test` builds a synthetic GGUF and capture with hand-computed answers, so the arithmetic is
gated without a model on disk.
