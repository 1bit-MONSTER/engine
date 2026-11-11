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

## What works today

On Linux with the `amdxdna` kernel driver loaded, the engine serves Qwen3-0.6B on the XDNA 2 NPU at
**88.6 tok/s** in the release gate, and it takes a GGUF directly: `1bit serve -m <model.gguf> --device npu`
repacks it to the Q4NX directory this page describes and runs six architectures. Everything on the
NPU is in progress — correctness first, speed next — so the iGPU through [HRX](hrx.md) remains the
route for anything the NPU does not run, and `--device hybrid` puts one request's prompt on one
device and its tokens on the other ([hybrid.md](hybrid.md)).

## How to try it

On a machine whose kernel has `amdxdna` (the [1bit OS](os-v3.md) appliance ships one):

```bash
1bit serve -m <NPU model dir> --device npu --alias npu      # a Q4NX model directory
1bit serve -m <model.gguf> --device npu --alias npu         # or a GGUF, repacked on load
```

Then it is the same OpenAI-compatible API as every other route ([serve.md](serve.md)):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"npu","messages":[{"role":"user","content":"Hello"}]}'
```

## A model directory

```
<model dir>/
  model.q4nx             weights (safetensors layout, Q4NX tiles: "Q4NX chunks" below)
  config.json            Hugging Face config: dimensions, rope_theta, eos_token_id
  tokenizer.json         Hugging Face tokenizer
  plan.json              the stream plan: fusion, kernel family, decode dirs ("plan.json" below)
  npu/                   the model's NPU kernels
```

**What ends an answer.** The serve stops on every id in `config.json`'s `eos_token_id` (one integer or a list), on
`<|endoftext|>` when the tokenizer has it, on `tokenizer_config.json`'s `eos_token` when that file is beside the
tokenizer, and on the marker the chat route's prompts close an assistant turn with: `<|im_end|>` on the ChatML
fallback, or, when the container's own template is rendered, whatever that template puts right after an assistant
message (`<|end|>` for Phi, `<|eot_id|>` for Llama 3, `<end_of_turn>` for Gemma), read off a rendered two-message
exchange at load. The two files need not agree: FastFlowLM's
`Qwen3.5-2B-NPU2` container names `<|endoftext|>` in the first and `<|im_end|>` in the second, and the model ends a
chat turn with the second. And neither file has to name the turn end at all: FastFlowLM's
`Phi4-mini-Instruct-NPU2` names `<|endoftext|>` in both, its template closes every turn with `<|end|>`, and before
the template was consulted its answer ran on past `Paris<|end|><|assistant|>` to the token budget. The load prints
the list (`forward-serve: stops on …`).

`1bit serve -m <model dir>` serves such a directory on the NPU behind the OpenAI-compatible API
([serve.md](serve.md)); inside Lemonade, that is what its `onebit` backend runs (`--onebit npu`).

### plan.json: the stream plan (engine #799)

A model directory may carry a **plan.json**: the decisions the runtime used to make by looking at which
files happened to be in `--kernels`. Three of them, and each one used to be silent:

The schema, the refusal rules and the generator flags are in [npu-plan.md](npu-plan.md).

| decision | what the probe used to do |
|---|---|
| `fusion.gate_up` (`fused`/`split`) | the fused `full_i8_GU_*` kernel won, and with no split kernel on disk the split layout was assumed |
| `kernels.tag` (`""`/`_b8`/`_pb16i`/`_t2s`) | the artifact family came from `q4nx_weight_layout`, so a kernel dir of the other family changed which kernels ran |
| `decode.dirs` | the M=16 twins were looked for in `<model>/kernels-m16`, then `<kernels>/m16`, then `<kernels>` |

**A decode off the 16-row family is said at ready (engine #2102).** forward-serve prints one line after `ready on
http://...` when roles have no M=16 decode kernel. In that case every decode step runs a 128-row prefill launch:

```
npu: 0 of 4 role(s) have a 16-row decode kernel in <kernels> -- decode runs prefill-width launches and is far slower than the fused route (engine#2102). ...
```

The load line `[forward] decode family: N role(s) at M=16, ...` stays byte-for-byte, because gates grep it. The new line
is silent when:

- the family is full;
- the arm is host-GEMM (`NPU_INFER_HOST_GEMM`, which loads no family);
- the container is ternary (pb16i/t2s) with some roles on M=128. Its loader refuses a missing decode kernel, and the
  roles it leaves are covered by its geometry, which the load log names per role.

`NPU_DECODE_M16=0` gets its own sentence, which names the switch. The twins are not the way to the demo rate. Measured on
Qwen3-0.6B, the per-op forward decodes at ~1.7 tok/s without them and ~2.7 tok/s with all four, against 88-101 tok/s on
the fused route.

`tools/npu_plan.py <model dir>` writes one from `config.json` plus the kernel directory (it derives layer
roles from `layer_types` and projection shapes from the model's own dimensions, the way the forward does).
`--check` re-generates and diffs against the file on disk, so a stale plan is a gate failure rather than a
wrong number; `--print` shows it without writing.

`container.tile_order` is the fourth decision written down, and it is the one that cannot be probed at all:
it names the q4_1 tile order the payload is in (`engine-1`, what this repo's converter writes). A container
from another converter can be a **different order**, and a wrong order decodes as noise *with no error
anywhere* — every structural check passes (same tensor names, dtypes, byte counts, chunk kinds), the container
loads, and the model simply answers badly (measured: one FLM converter checkout's containers answer the same
token five times where ours answer " Paris."). Absent means `engine-1`, so every container written before the
field existed keeps loading; an id this decoder does not implement is a refusal naming it, because the
alternative is to guess a tile order at the one point where a guess is silent.

The forward reads the plan with `npu/plan/plan.h` and **refuses by name** anything it cannot honour — a
`container.tile_order` other than `engine-1`, a `kernels.tag` that disagrees with the container, a sealed ELF
named outside the pb16i family, a decode dir
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
bidirectionally (the plan is a copy of that declaration: with no plan the container's own word decides, and a plan
that disagrees with it is refused — [Decisions](#decisions-v1systemone-issue-653)): `step_batch()` fills the K/V cache for every row of the batch first and only then attends,
so row *r* sees rows after it. The GDN layers keep their causal recurrence (the pack's own runtime keeps it
too), a sliding layer stays causal, and the mode reads the same numbers only in one shape:

- **the whole prompt, one batched pass, at position 0** (`POST /v1/systemone` does exactly that). A batch
  that starts above 0 is refused by name — its earlier positions were already encoded causally.
- **it cannot generate.** `step()` (one position at a time, keys only in the past) refuses with the mode's
  name, so a d3 pack served on the chat route is an error rather than a silently causal answer.
- **nor can it verify a draft.** `step_batch_all()` returns the logits of *every* row of a batch, which a
  speculative decoder compares against the tokens it drafted at those positions. In a noncausal batch each
  row attends to the rows after it, so those are not prefix-conditioned logits and the call is refused with
  the same mode name. Unreachable today (the d3 packs ship no draft head), refused now so a pack that grows
  one cannot be silently verified through a mask it was not trained with.

An unknown mode string is a plan refusal, never a default: the two modes give different numbers, and a
guessed mask is a different model.

### Sentence embeddings (1-bit encoder containers)

`1bit` serves a small BERT-family encoder as **sentence embeddings**: 384-dim unit vectors, mean-pooled over the
tokens, from a container the importer writes (`tools/trit_embed_pack.py`). The first model is
**Cyronius/trit-embed-3l** (MIT; see NOTICE): 3 post-norm BERT layers, width 384, 12 heads, FFN 1536, WordPiece
vocabulary 30522, at most 128 tokens.

**Why the container is not a q4nx one.** A q4nx quantised tensor is a grid of 32x256 chunks, and this model's
rows are 384/1536 wide with **one scale per output row** and 2-bit ternary weights. Tiling them would pad K to
512 and re-round the scale into per-128-block fp16, which moves numbers the correctness gate compares exactly.
The importer therefore widens the ternary codes {-1,0,+1} to **int8** (the value is unchanged; XDNA2 has no
ternary datapath, which is why the widening is the format's job and not the kernel's) and keeps the model's own
per-row scale in f32. `model.embed` holds a 16 KiB JSON header and the tensor payload it names:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `<layer>.{q,k,v,o,fc1,fc2}.weight` | `i8` | [N, K] | ternary codes widened to int8, values in {-1, 0, +1} |
| `<layer>.*.scale` | `f32` | [N] | one scale per output row: `w = code * scale` |
| `<layer>.*.bias`, `ln1.*`, `ln2.*`, `ln_e.*` | `f32` | [width] | biases and LayerNorm parameters |
| `word.qweight`, `pos.qweight` | `i8` | [rows, 384] | int8 embedding tables |
| `word.scale`, `pos.scale` | `f32` | [rows] | one scale per embedding row |
| `tok_type.weight` | `f32` | [2, 384] | the token-type embedding (row 0 is used) |
| `out.weight` | `f32` | [384, 384] | the output projection applied after pooling |

**The forward is the reference's arithmetic, not an approximation of it.** Every ternary linear sees an
**int8 activation with a power-of-two scale per token** (per 256-channel block when the width is a multiple of
256 and wider), rounded half to even, clamped to ±127 and dequantised — because the model was trained that way,
and a float activation would move the vectors off the gate. Layers are post-norm: `LN_e` on the embedding sum,
then per layer attention over q/k/v with `LN1` on the residual and a GELU(tanh) FFN with `LN2` on the residual.
Then masked mean pooling, the output projection, and L2.

**The tokenizer is the contract, so it is the reference's `BertTokenizer` pipeline**: clean (control characters
and NUL/0xFFFD dropped, whitespace to single spaces), one token per CJK character, lowercase, strip accents,
split punctuation, then greedy longest-match WordPiece with `##` continuations, and `[UNK]` for a word whose
*first* piece cannot be matched. `[CLS]`/`[SEP]` frame the ids and truncation counts both. Unicode work is
table-driven (ASCII, Latin-1, Latin Extended-A, the common punctuation blocks, the CJK ranges); measured on the
2,552 distinct STS-B test sentences the ids are **identical to the reference tokenizer's, 0 mismatches**. That is
this route's own pipeline; the engine's other tokenizer path - a `tokenizer.json` read through Hugging Face
tokenizers, the Rust library, behind a C ABI - is a separate implementation of the same contract, measured against
its own reference ([tokenizers.md](tokenizers.md)).

**Correctness, as measured** (`tests/trit_embed_pack_test.py` and the rows in engine#1311): the importer's
ternary decode is *exactly* the reference's `codes * scale` (worst |Δw| 0, not a tolerance), and the encoder's
vectors against the model's own `trit_embed.py` on 1,000 fixed sentences have **min cosine 0.9997** (mean
0.99999); on STS-B the C++ forward scores within 0.3 of the model card's 82.87.

**Serving it.** A container is served **in-process**, not through a llama-server (there is none for it):

```
1bit serve --embedding -m <container> --device npu|cpu          # the model itself is the embedding model
1bit serve --embed <container> --device npu|cpu -m <gguf>       # beside a chat model, in the RAG slot
```

`/v1/embeddings` (and `/embeddings`) take OpenAI's body: `input` as a string **or an array of strings**, one reply
entry per input with its `index`, `usage.prompt_tokens` from the model's own tokenizer, and L2-normalised
`embedding` vectors. An array of token-id arrays is refused **by name** (it would need the caller's ids to be this
vocabulary's ids), as is an empty array and an out-of-vocabulary one. `--device auto` means npu in an NPU build and
cpu otherwise; any other device is refused with the reason. `/health` and `/v1/models` answer in the shapes the
rest of serve uses, so Lemonade's watchdog and model list work unchanged.

Measured on this box through that route, 1,000 fixed sentences (the gate's own set) with the container above:

| route | min cosine vs the reference | sentences/s |
|---|---|---|
| `--device npu` | **0.9997062** (0 below 0.999) | **35.1** |
| `--device cpu` | 0.9997062 | 3.5 |

and STS-B through the same route: **Spearman ×100 82.9412** (the model card's 82.87, its own runtime 82.75 on the
same evaluator).

### Q4NX chunks

Q4NX is [FastFlowLM](https://github.com/ROCm/FastFlowLM)'s NPU model format (MIT), and its models on
Hugging Face (`FastFlowLM/*-NPU2`) are Q4NX directories. A Q4NX weight is a grid of 32-row x
256-column tiles, one chunk each, row-major over the grid. The chunk kind is the last dimension of
the tensor's shape: `[tiles, <chunk bytes>]` (5120, 6144, 4736 or 8704), or `[tile rows, tile cols,
chunk bytes]` (`npu/q4nx.h` has the byte layouts).

| Chunk | Kind | Weight | Seen in |
|---|---|---|---|
| 5120 B | q4_1: bf16 scale and zero per 32 columns of a row | `code * scale + zero` | Qwen3, Llama, Gemma, Phi models |
| 6144 B | q4_1f: **fp32** scale and zero per 32 columns of a row (imported containers, below) | `code * scale + zero` | `1bit npu-pack import` output |
| 4736 B | Q4_K: u8 scale and min per 32 columns, bf16 `S`, `M` per row | `S * scale * code + M * min` | Qwen3.5-4B projections |
| 8704 B | Q8: bf16 scale per 32 columns, int8 codes | `d * code` | Qwen3.5-4B `lm_head` and embedding |
| 2176 B | **t2**: 2-bit ternary codes, one fp16 scale per 128 columns of a row | `code * scale` | ternary models (Bonsai-2, PTQ1_0/TQ1_0 sources) |

The two q4_1 forms are a container property, named by the container's `layout` metadata
(`__metadata__.layout` in `model.q4nx`, `q4nx_source.layout` in `config.json`): absent or
`q4nx-1` is the bf16 form every container written before engine#1185 has, and every GGUF-derived
one (the bf16 choice came from that path); `q4nx-2` is the fp32 form `1bit npu-pack import` writes,
so an OGA pack's own scale and zero survive bit for bit. A decoder keys off the chunk's own last
dimension either way, so a container with no metadata still loads; `q4nx::params_of_layout()` is for
the caller that has to state the fidelity it holds, and the forward converts such a container at
load: `npu/q4nx_device.cpp` repacks every fp32 tensor through `q4nx::as_q4_1()` into the Q4NX
cache (`$ONEBIT_Q4NX_CACHE`, else `~/.cache/1bit/q4nx`) and hands the loader that copy, so the
device streams its 5,120-byte bf16 q4_1 while the container on disk stays exact. The repack is where
the bf16 rounding reappears for the device, never in the container.

**The cache is bounded, and the bound is least-recently-used (engine#1956).** After writing an
entry, `1bit serve` removes the least-recently-used entries until the cache is within
`ONEBIT_Q4NX_CACHE_MAX_GIB` (default 60; `0` or a negative number turns the sweep off). The entry
just written and any entry used in the last ten minutes are never removed. "Used" means reused,
not only written: a cache hit — a repacked GGUF found in the cache, or a device-form conversion
reused at load — marks the entry's directory as used now, so a container imported once and served
every day outlives a one-off imported yesterday. The value is read as a whole number of GiB: a
number too large to be a byte count is the largest bound that is one (it does not wrap to a small
one), and a value that is not a number gets the default rather than an unbounded cache.

**The packed lm-head is cached beside it (engine#1716).** A model whose head runs as int8 tiles
packs those tiles on the host — gather `kpad × T` weights per tile, quantise each column to int8
with its own scale — and until this cache that happened on every serve. The image is a function
of the head's stored bytes and of the kernel set (`kpad`, the tile width `T`, the blocked or
row-major layout), so the forward keeps it in `<Q4NX cache>/head-images/` as
`headimg-<key digest>-t<tile>.q4hi`, one file per tile. The digest covers the format version,
`kpad`, `T`, the layout, and a hash of the head tensor's bytes together with the vocabulary size,
the width and the dtype it is decoded at, so another container, another kernel set or a re-import
names different files.

A cached tile is used only when both of these hold, otherwise it is packed as before and the file
is replaced, with one stderr line saying why:

- the file reads back whole: format version, geometry and the hash over its payload
  (`q4nx_head_image_read`);
- a sample of its columns (8 per tile, both ends and an even spread) packs to the same scale and
  the same codes from the weights the forward holds at that moment, through the production packer
  (`q4nx_pack_B_int8_blocked`). This is the check a payload hash cannot make: an intact image
  written for other weights is caught here.

The serve log line says what happened: `lm head: N packed weight tile(s) resident (…); H from the
image cache, P packed (W written for the next serve) in S s`. `ONEBIT_HEAD_IMAGE_CACHE=0` turns
the cache off (nothing is read or written). The files are `kd × nd` bytes plus `nd` scales per
tile, so a head costs about `kpad × vocabulary` bytes on disk per kernel set. The Q4NX cache sweep
(`ONEBIT_Q4NX_CACHE_MAX_GIB`, above) counts `head-images/` as one entry of the cache and removes it
whole when the cache is over its bound and that directory is the least recently written one. Its
age is the last time a tile was written into it: a serve that only reads its tiles does not refresh
it. Losing the directory costs the next serve one pack of its head, nothing else. Ternary (`pb16i`) heads and the gemv head do not
use it: their image already is a file. Each writer builds the file under its own name
(`<file>.tmp.<pid>`) and renames it into place, so two serves that start cold on the same pack both
finish their write and a reader only ever sees a whole image. The behaviour is pinned host-side in
`tests/npu_design_grid_test.c` with a counter on the gather and a two-process write arm.

**Q8 has two code layouts under the same chunk size.** Qwen3.5-4B's lm_head (tied to the
embedding) stores code (row r, column k) at byte `k*32 + r`. Qwen3.6-35B-A3B's Q8 tensors
(untied lm_head, projections) store it at `(r/16)*4096 + (k/32)*512 + (k%32)*16 + r%16`.
Both containers report `flm_version` 1.0.3, and neither carries metadata that tells them
apart. Decoding tile (0, 0) of each lm_head and comparing with a GGUF of the same model
gives correlation 0.9998 with the right layout and about 0.01 with the other. `npu/q4nx`
decodes the first layout.

`npu/q4nx.h` decodes all five (the two q4_1 parameter forms included), and repacks Q4_K into q4_1
(exact apart from rounding `S * scale` and `M * min` to bf16), so code that reads q4_1 runs Q4_K
weights unchanged. Q8 does
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

#### Gating a t2 target: bit-exact, not close

The t2 chunk is what the plan layer streams for a ternary model, so more than one thing will consume
it: the XDNA path today, a Hexagon target next (engine #804), anything after that. A consumer is
accepted when it produces **the same y a CPU oracle produces, bit for bit**. Close is not enough and
the difference is not pedantry: the layer above the GEMV is compared against the oracle, so a target
that is 1e-7 off makes every gate above it either fail or be weakened to a tolerance that hides real
faults.

The order a t2 target has to follow, per row:

```
for each 128-K group g, left to right:
    s   = exact int32 sum over the group of code[r][c] * x[c]      # |s| <= 128*127, no rounding
    y  += float32(s) * float32(scale[r][g])                        # two roundings: mul, then add
```

Three things follow, and each one is a way to get it wrong: the group sum is integer and exact (never
accumulated in float); the scale multiplies the *group sum*, not each code; and the groups enter the
f32 accumulator in K order. Below four groups a float64 accumulator and a reversed group order give
identical bits — the arithmetic cannot round yet — which is why the gate tests the production width.

`tests/npu_t2_target_gate.py` is that gate. Its always-on half is numpy only (no model, no device, no
gguf): it accepts the oracle's own y for three shapes including a single tile, and it refuses four
plausible near-misses — a one-bit change, a truncated dump, float64 accumulation and a reversed group
order — plus a tensor carrying a reserved code, which must contribute 0 and never `2 * scale`. Given
`--tensor`, `--dump`, `--x`, `--n` and `--k` it runs the same comparison against a real tensor and a
target's dumped y, printing the bit-exact count, the SNR and the first mismatch:

```
tests/npu_t2_target_gate.py --tensor blk_0_attn_qkv_weight.t2 --dump /tmp/y.bin \
                            --x /tmp/x.bin --n 10240 --k 5120
```

Measured on the six layer-0 projections of Bonsai-2 (the GDN block's `attn_qkv`, `attn_gate`,
`ssm_out` and `ffn_{gate,up,down}`), with a Hexagon target against this oracle: bit-exact on
10240/6144/5120/17408/17408/5120 values, 375.7-377.7 dB, 0 reserved codes. The same gate refused the
first build of that target for reading the tile grid k-outer instead of n-outer — the rule two
paragraphs up — which is the reason it exists: the target was self-consistent, bit-exact against its
own scalar reference, and decoding a different matrix than the container meant.

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

**That single rounding costs one multiply per element, not one `ldexp()` (#1990).** A launch's result is
`float32(S * 2^e_ref) * ascale` for every element, and `2^e_ref` depends only on the column. It was computed
with a libm `ldexp()` call per element, after readback, where no launch phase column counted it: a 16-row
prefill block on `bonsai2-27b-t2s2` is 416 launches of up to 16 x 17,408 elements, and that call was the
block's largest host term. `q4nx_pb16i::dequant_rows()` computes the power once per column and multiplies.
The value is the same, not close: a double times a normal power of two is exact while the product is a
normal double, which holds for every exponent from -1022 to 960, and any column outside that range takes
the per-element form as before. Measured in one `npu-hold` ticket, one binary, `NPU_DEQ_LDEXP=1` (the old
form) against the default, 103 blocks per arm: the dequant term **137.4 -> 32.3 ms per block**, the block
median **630.4 -> 515.0 ms** (25.4 -> 31.1 rows/s), and three 510-536-token prompts reached their first
token at 24.4-25.1 tok/s before and **29.8-30.6 tok/s** after. That is the first pair of arms; the box
slowed for both forms in the second pair (the old form measured 357 ms of dequant per block there), so the
second pair is not quoted as a speedup. `NPU_DEQ_LDEXP=2` runs both forms and counts differing elements:
0 of 568,483,840 on the device, and the greedy text is identical in all five arms.
`tests/npu_pb16i_layout_test.cpp` holds the two forms to the same bits for one and two planes, the edges
of the exponent range and exponents outside it. The `[bl]` line names the term as `deq`.

**A sealed set may carry a wide family for prefill (#1990).** The served sealed sets are built for 16
rows per launch, so a prefill chunk is 16 rows and a 536-token prompt is 34 blocks of 416 launches. A pack
may name a second sealed ELF with the same instances built for more rows: `q4nx_t2s_sealed_wide_elf` (or
`q4nx_pb16i_sealed_wide_elf`) in `config.json`, a file in the kernels directory beside the 16-row one, and
`q4nx_sealed_wide_rows` for its row count (default 32, allowed 17..128). No pack ships one yet, so nothing
changes for an existing pack. `NPU_WIDE_SEALED_ELF=<file>` overrides the pack's file, `NPU_WIDE_SEALED_ELF=0`
switches a pack's wide family off, and `NPU_WIDE_ROWS` overrides the row count. `tools/npu_plan.py` copies
the two keys into `plan.json` as `prefill.sealed_elf` and `prefill.sealed_rows`, and a loaded plan wins over
`config.json` (`docs/npu-plan.md`).

When the wide ELF loads for every role that has a 16-row kernel, the chunk width becomes its row count and a
launch of more than 16 rows takes the wide kernel; decode steps and chunks of up to 16 rows stay on the
16-row family. It is all roles or none: the chunk width is one number for the pass, so a set missing one
role is not adopted. A named file that is missing or does not load never stops the serve: the log says why
(`[forward] wide family NOT adopted (...)`) and chunks stay at 16 rows. The weight images are shared, only
the activation and result buffers differ. A launch that carries more rows than its kernel holds is refused
by name. The rules (`wide_family_complete`, `chunk_rows`, `wide_takes`, `wide_elf_path` in
`npu/forward/include/batch_width.h`) are pinned without a device by `tests/npu_pack_test.cpp`.

Measured on `bonsai2-27b-t2s2` with a 32-row set, one `npu-hold` ticket per row of the table, one binary,
arms off / on / off / on, three 510-536-token prompts per arm (`prompt_tokens / wall`, `max_tokens 1`):

| how the wide set was named | first token, wide off | first token, wide on | 16-row block median | 32-row block median |
|---|---|---|---|---|
| `NPU_WIDE_SEALED_ELF` | 28.3-29.8, 30.5-30.8 tok/s | **35.6-35.9, 35.0-35.4 tok/s** | 534.2, 509.3 ms | 871.0, 881.5 ms |
| the pack's `config.json` key | 28.7-28.9, 28.7-29.0 tok/s | **33.9-34.2, 34.2-34.4 tok/s** | 541.2, 541.4 ms | 914.0, 908.3 ms |

Two 16-row blocks against one 32-row block is 30-31 rows/s against 35-37. The 8-token greedy text is
identical with the family off and on, and `NPU_BATCH_CHECK=1` on the wide path reports MATCH on every chunk
it checks (32-, 4- and 1-row) in both tickets. Only 32 rows has been measured.

**...and a narrow family for decode (#2073).** A decode step on `bonsai2-27b-t2s2` is 416 launches, and
device wait is 211 of its 302 ms: every launch runs a kernel built for 16 rows and carries one, and a
launch's device time does not depend on how many of its rows are filled. So a pack may also name a sealed
ELF built for FEWER rows: `q4nx_t2s_sealed_narrow_elf` (or `q4nx_pb16i_sealed_narrow_elf`) and
`q4nx_sealed_narrow_rows` (default 4, allowed 1..15) in `config.json`; `NPU_NARROW_SEALED_ELF=<file>`
overrides the file, `=0` switches it off, `NPU_NARROW_ROWS` overrides the rows. A launch of 1..rows rows
takes it (a decode step, or a verify pass of a few rows); anything wider keeps the 16-row family. The rules
are the wide family's: all roles or none, one hw_context, shared weight images, and a file that is missing
or does not load is logged (`[forward] narrow family NOT adopted (...)`) while decode stays on the 16-row
kernels. `narrow_family_complete` and `narrow_takes` in `batch_width.h` are pinned by
`tests/npu_pack_test.cpp`. No pack ships a narrow ELF yet. `plan.json` carries it as `decode.narrow_elf` and
`decode.narrow_rows` (`docs/npu-plan.md`).

Measured with a 4-row set, one binary, arms 16-row / 4-row / 16-row / 4-row, three prompts generating 49
tokens each (`(tokens - 1) / (wall - wall of a 1-token request)`), two `npu-hold` tickets:

| ticket, pair | decode, 16-row family | decode, 4-row family | device wait per step |
|---|---|---|---|
| first, pair 1 | 3.26-3.33 tok/s | **3.69-4.21 tok/s** | 210.1 -> 158.1 ms |
| first, pair 2 (another lane loading the box) | 1.88-2.59 tok/s | 2.33-3.14 tok/s | 230.7 -> 208.1 ms |
| second, pair 1 | 2.85-3.29 tok/s | **3.90-4.16 tok/s** | 234.7 -> 151.7 ms |
| second, pair 2 | 2.15-3.32 tok/s | **4.05-4.19 tok/s** | 210.9 -> 150.9 ms |

The step wall follows the wait: 286.0 -> 233.7 ms and 296.2 -> 225.9 ms in the quiet pairs. The 49-token
greedy text of each prompt is identical in all eight arms, and `NPU_BATCH_CHECK=1` with the narrow family
reports MATCH on every chunk it checks (1-, 4- and 6-row; the 6-row one runs on the 16-row family).

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

#### The t2s family is M=16 today, and that is what caps prefill (#1416)

Measured, not inferred: a t2s container's kernel set contains **only `*_t2s2_m16.elf`** for 8 of the nine decode roles, and the engine reports the
same at load — `decode family: 8 role(s) at M=16, 1 on M=128`. The M=128 names in the same directory (`full_i8_G_K5120_N17408.elf`, `…O…`, `…D…`,
`…QKV…`, `…Z…`, `…U…`) are the **int8/pb16i** family: they read pb16i images, not 2-bit streams, so they cannot serve a t2s container's prefill
however many rows they accept.

Consequences a reader will otherwise trip over:

- `max_rows()` returns **16** for a t2s/pb16i container (`npu/target/xdna/xdna_target.cpp`, `c.max_rows = m16_ ? M_DECODE : M_PREFILL`), and that
  value is the **chunk width** the blocked walk in `npu_gemm_rows()` uses. So a prompt is walked in 16-row chunks: measured on a 393-token prompt,
  **24 chunks of 16 + one of 9 = 25 chunks × 64 layers = 1600 launches**, against the 4 × 64 = 256 the M=128 design intends.
- nothing is corrupt by that: `ctx_for()` routes a ≤16-row chunk to the decode twin whose activation BO is 16 rows, and a launch wider than its BO is
  **refused by name** (`quantize_async_rows: am=128 exceeds bA capacity MD=16`) rather than poisoning memory — the engine#1372 guard.
- the cost does *not* grow with the stream: at M=16 the weight cache holds — `[wm]` prints **nothing**, where a miss would print
  `[wm] n=20 cache-miss repacks, avg packB_into=29.224 ms` — whereas the M=128 **pb16i** geometry does pay that **≈30 ms `packB_into` per miss**
  (measured 29.2-30.5 ms over 140 misses). The negative and its falsifier are the same line. Two container formats with different kernel families therefore have different
  prefill cost structures, and numbers from one do not explain the other.

An **M≥256 t2s family** is what removes the 8x launch overhead; it is private-repo kernel work (`npu-kernels`) with the same byte-exact gate the M=16
family passes, and the acceptance row is: both Llama3-8B prefill shapes (`K=4096 N=14336`, `K=14336 N=4096`) at M = 256 / 1024 / 16384, ≥ 4 TFLOP/s,
byte-exact logits against the current path on a fixed prompt set, `amdxdna` counter 0.

#### Where a prefill block's time goes now (measured), and why it is no longer the pack (#1416)

#1416 was filed on a `[prof] pre` table whose `pack` column read 24 ms per launch against 1.3 ms of device wait. That
column was the first-touch weight fill of a fresh serve, not a per-request cost, and three merged changes removed it
from every request: the weight cache is filled before the serve says ready (#1621), the fill reads each image into its
BO (above), and only the rows that exist are quantised. Re-measured on the same pack (`bonsai2-27b-t2s2`, 2026-11-10,
`npu-hold` tickets, `amdxdna` error count unchanged across each), a 5-token prompt for 8 tokens costs **2.99 s as the
first request of a serve and 2.90 s as the second and third**, and a 510-536-token prompt reaches its first token at
**20.2-21.7 tok/s** (`prompt_tokens / wall`, `max_tokens 1`, distinct prompts).

One 16-row block of that prefill, from the serve's own `[bs]` and `[bl]` lines (`NPU_PHASE_PROF=1`; medians over 103
blocks per arm, two arms per binary run A B B A in one ticket):

| term of a 16-row block | before | after | |
|---|---:|---:|---|
| block wall | 734-749 ms | **688-704 ms** | 21.4-21.8 -> **22.7-23.3 rows/s** |
| device wait (416 launches) | 213-214 ms | 215-217 ms | unchanged: 29-31 % of the block |
| `gemm pack` | 55.6-56.6 ms | 55.6-58.4 ms | 0.13 ms/launch: 7-8 % of the block, where #1416 read 93 % |
| host outside the GEMM launches | 392-407 ms | **342-353 ms** | the majority term |
| of which GDN alpha/beta | 77.5-77.8 ms | **13.1-13.5 ms** | the change below |
| of which GDN recurrence | 30.5-33.4 ms | 27.0-30.0 ms | |

So prefill on this route is bound by host work outside the launches and by the M=16 launch geometry above, in that
order, and not by packing. "After" is one change: the per-(row, head) alpha/beta dot products of a GDN layer ran on one
thread inside the row loop (0.099 ms per row and layer, 48 GDN layers), and they now run as a second worksharing loop
inside the parallel region that layer already opens for the recurrence. No region is added, because the region count is
what a loaded box pays for. Each dot product is the same sequential loop as before and the build has no `-ffast-math`,
so no bit moves: greedy text is identical in all four arms, and `NPU_GDN_CHECK=1` recomputes the serial form for every
layer and compares bit patterns (336 of 336 layer passes identical on the device at M = 1, 4 and 16; a mismatch fails
the call). `tests/gdn_batch_order_test` requires those check lines on any hybrid container named by `ONEBIT_TEST_Q4NX`.

A request wall moves less than the block does and scatters more: the same four arms measured 20.2-21.7 tok/s before and
21.6-23.1 after by TTFT, with one outlier in each direction (18.3 and 15.8) from other lanes' load. Quote the block
medians for the effect and the TTFT range for what a caller sees.

#### The attention rows of a block share one thread team (#1990)

The next host term after the one above, same pack, same instruments. A batched pass called `attention()` once per row,
and each call opened its own thread team over the heads: 16 rows on each of this model's 16 full-attention layers is
256 teams per block. The rows of a layer now go through `q4nx_attn::rows_context()` (`npu/forward/include/attn_math.h`):
every row's position-local half runs first (its q, and its k/v appended to the cache), then all (row, head) pairs are
one statically scheduled loop in one team. A causal row reads only positions up to its own, so the later rows already
being in the cache changes nothing it reads.

Measured on `bonsai2-27b-t2s2` (2026-11-10, `npu-hold` tickets, one binary, `NPU_ATTN_ROWS=0` against the default,
103 blocks per arm, `amdxdna` error count unchanged). **What it buys is the attention term and its behaviour under
load, not a block-wall gain that can be resolved on a quiet box:**

| ticket, arms in order | attention per block, mean (per-row -> one team) | block wall median |
|---|---|---|
| first: A B (quiet) | 31.2 -> **14.3 ms** | 675.7 -> 658.4 ms |
| second: A B A B (quiet) | 24.5 -> **13.0**, 26.6 -> **13.1 ms** | 609.3 -> 613.1, 626.3 -> 601.9 ms |
| first, later arms: B then A, another lane's load on the box | **16.0** vs 98.1 ms | 674.5 vs 834.4 ms |

The attention term halves in every pair. The block wall does not follow it reliably: the first pair moved by 17 ms,
the two pairs of the second ticket by +4 and -24 ms, which is the arm-to-arm scatter of this box (about 15 ms on a
600 ms block) and not a measured 2-3 %. The first version of this section quoted the first pair as a 2.6 % gain; the
second ticket is why it no longer does. Under load the picture is different and is one observation, not a repeat: the
per-row arm measured 56.2 ms median attention with 36 of 103 blocks over 100 ms, while the one-team arm before it held
13.6 ms with none.

Nothing rounds differently: a (row, head) pair runs the same statements in the same order as before
(`q4nx_attn::head_context()`, which `attention()` now calls too) and writes only its own floats. Greedy text is identical
in every arm. `NPU_ATTN_CHECK=1` recomputes each row the per-row way and compares bit patterns, failing the pass on a
difference (160 of 160 layer passes identical on the device); `tests/attn_math_test` holds `rows_context()` to the
per-row form bit for bit on 1, 2, 5 and 12 threads, for causal, whole-prompt and sliding-window reads.

The schedule is static on purpose. A first version used a dynamic schedule and measured a lower median (15-25 ms) but a
worse mean than the per-row form, with 6 and 14 of 103 blocks over 100 ms in its two arms: the tail was the whole cost.

#### Per-tile arithmetic ceiling, and where our GEMMs sit against it (AM027)

AM027 (*Versal Adaptive SoC AIE-ML v2 Architecture Manual*, v1.1) **Appendix A, Table 14** gives 512 INT8×INT8 multipliers per compute tile and 256
bf16/fp16; **Chapter 1, Table 1** gives the 1 GHz array clock. Per tile that is **1.024 TOPS int8** and **0.512 TOPS bf16**, so an N-tile launch has
ceiling `N × 1.024`. Strix Halo's NPU is aie2p rather than AIE-ML v2, so these are *architecture* figures used as a scale, while the tile count that
actually applied comes from this box's driver and from the measurement itself.

| measurement | GEMM datatype | tiles | ceiling at **that** datatype | measured | % of that ceiling |
|---|---|---:|---:|---:|---:|
| gate, M=16, K=5120 N=17408 (2.852 GFLOP in 0.589 ms) | int8 | 8 | 8.19 TOPS | 4.84 TOPS | **59 %** |
| prefill, M=128, ~~18.64 tok/s~~ — **RETRACTED, see below** | bf16 | 32 | 16.4 TOPS | — | — |
| prefill, M=128, 16.81 tok/s (TTFT, 2048-token prompt) | bf16 | 32 | 16.4 TOPS | 0.91 TFLOPS | **5.5 %** |
| published W4A16 reference, 8.06 TFLOP/s | int8 | 8 | 8.19 TOPS | 8.06 TFLOPS | **98 %** — i.e. an int8-MAC-bound result |

**Retraction notice — 18.64 tok/s is retracted.** It was quoted against *request-wall* figures in later comparisons where the ticket's load window was not
sustained; the replacement is the TTFT-based 16.81 above (`prompt_tokens / TTFT`, distinct prompts, no shared prefix, measured in the first half of the
ticket — so also an upper-band figure). Any page still carrying 18.64 must mark it retracted rather than repeat it. The ceiling arithmetic is unaffected:
the tile count and the per-tile rate are properties of the architecture, and what changes is 2·params ÷ tok/s — 0.91 TFLOPS, i.e. 5.5 % of the 32-tile bf16
ceiling.

The ceiling column is *per datatype* — 1.024 TOPS/tile for int8 and 0.512 for bf16 — so the rows are not comparable to each other by tile count; each % is
against its own datatype's ceiling.

Two readings that follow, and that a whole-array denominator would hide: our **decode** GEMM is at **59 %** of the tiles it occupies, so the remaining
headroom there is the stream (40.2 GB/s measured) and not the MACs; our **prefill** is at **5.5 %**, which is the launch geometry above, not arithmetic.
And the reference result being ~98 % of an 8-tile int8 ceiling is why 4-bit designs on this architecture are int8 MACs with an on-core expansion:
Table 14 lists **no INT4 multiplier** row, and the array's 4-bit mode — *"Vector unit supporting MAC operations for multiple precisions (for example,
512x 8-bit × 8-bit and 512x 4-bit × 4-bit)"*, Chapter 4's Vector Multiplication Unit — runs at the **same 512-per-tile rate as int8**. So a narrower
integer datapath exists; what does not exist is a narrower one that multiplies *faster*, which is the fact that decides the design.

#### What the stream is worth at the model level (measured)

**Reading a `[ph]`/`[bs]` row: the first field after the identifiers is `host_gemm`.** A row with
`host_gemm=1` is a pass that never asked the device — `npu_gemm_rows()` returns to the host GEMM on that
flag *before* `max_batch_rows()` or the blocked-launch path is consulted, and the flag is set only by
`init_host()` or `NPU_INFER_HOST_GEMM` — so `n=0` on such a row is a host pass by construction, not a
device launch that went uncounted (engine#1582, where a prefill row with `n=0` and a decode row with a
device number were read as a contradiction in one process). `[bs]` also prints `rows_max=`, the widest
launch the target's kernels take, because a batch wider than it is walked in blocks rather than skipped.

The phase numbers above are the kernel's. The number a user feels is a decode step, and it is the one that
decided P3: same container file and same engine binary, two weight paths, adjacent windows on one quiet box
(2026-11-02, three `npu-hold` tickets on `:18099`, `amdxdna` counter 0 in each, seven steady-state tokens per
arm read from the serve's own `[ph] tok` lines):

| container | decode | device wait | stream per token |
|---|---|---|---|
| pb16i (control) | 410 ms/token (409-412) = 2.44 tok/s | 296 ms | 12.93 GB |
| **t2s2 sealed** | **313 ms/token (311-317) = 3.20 tok/s** | **209 ms** | **6.11 GB** |

**1.31x per token and 1.42x on the device wait**, on 2.12x fewer bytes to stream (431 images, 14.44 -> 6.82 GB).
The model keeps 1.31x of the kernel's 1.76x because a step is more than its GEMMs: the host half (quantise,
rotation, GDN) and the lm head do not shrink with the wire, and in this pair they are 66 ms and 13.5 ms of it.

Acceptance is byte-exactness, not a text diff: every decode role matched the pb16i path byte for byte at
layer 0 (6/6) and layer 3 (8/8) on all three seeds, and greedy text is equal to the pb16i arm's on the prompt
both were asked — `' Paris.\nThe capital of Germany is'`. **A request wall is not this number** — a fresh
serve pays the image fill once, so a 32-token request measures 614/651 ms/token while the per-token instrument
says 313/410; that correction, and why every request-wall figure in this file pays the fill, is in the chunk
above.

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
per-serve benchmark's request wall as a decode number: `[ph]`/`[lh]` are the per-token instruments, and
`[bs]` is the same partition for one batched pass (a verify batch), whose interior a per-token line cannot see.
`NPU_COPY_PROF=1` adds a `[cp]` line to both: the **bytes and milliseconds the host moves at its own copy
sites** (`qkv_split`, `attn_gate`, `ctx_row`), one figure per site and a `KiB/row` total for a batched pass.
The phase lines above attribute device time only, so a host row copy lands in their `host=` residual with
nothing to separate it; `[cp]` is the line that says how much is left to remove rather than time it as part of
the layer's wall. The MLP row copies this line first counted (`mlp_in`, `mlp_out` — 12.75 MiB of a 15.38 MiB
prefill row) are gone: the batched pass now runs `mlp()` on the ACT row and into the HH row in place.

The counters are per **process**, single writer, like `[ph]`: a serve enters `step()`/`step_batch()` under
its engine mutex (`app/forward_serve.cpp:258`, `:698`), so one pass is in flight, and `--npu-streams N` is
N processes sharing the device by flock (`app/npu_device_lock.h`), each counting its own. A caller that
parallelised the layer loop would need per-pass counters instead.

### From a 1BP package

`tools/onebp_import.py <model.1bp> <out dir> --config <snapshot dir>` imports a legacy `.1bp` package
(one file: a 256-byte header, a tensor index, the weights). Its 4-bit payload is a 32 x 256 tile of 5120 B
with this engine's value convention (`scale*q + min`), but **not** its intra-tile layout: `.1bp` stores
the scales/mins row-major (`[r*8+g]`) and the codes adjacent-pair (even column low), while `npu/q4nx.h`
stores them group-major (`[g*32+r]`) and lane-swizzled. The import permutes each tile's bytes — the
inverse of the archived converter's own `q4nx_tile_to_onebp()` — and dequantizes and re-quantizes nothing;
`F16`/`F32` tensors become bf16. `--verify` proves it by comparing every byte the container holds against
the same permutation of the package's slice (and against the GGUF behind any tensor that came from
`--embedding-from`). Reading a `.1bp` tile as if it were already this engine's tile is not a near miss:
on `ZR1-1.5B-1BP` the decoded weights then correlate 0.003 with the model — fluent noise that still looks
like weights, the failure mode of [engine#1110](https://github.com/1bit-MONSTER/engine/issues/1110) —
and 0.995-0.997 after the permutation. A v1 package carries no `token_embd`/`lm_head`, so importing one needs
`--embedding-from <gguf>` (and `--verify` of it takes the same flag: it rebuilds the same plan); the tool stops
rather than write a container that cannot answer. Any tensor
with no Q4NX mapping is refused **by name** (`no Q4NX name for blk.0.cca_conv_grp.weight`) instead of
guessed: Zaya1's grouped conv, EDA router and fused `ffn_gate_up_exps` have no op in `npu/forward` yet,
so a package of that architecture is not importable — see `~/briefs/ZAYA-OPFAMILY-NOTE.md` for the op
list and its cost.

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

**A GGUF that declares a RoPE scaling block is refused.** This route has no scaling rule — the
packed lane computes `theta^(-i/half)` and nothing else — and a container written from a scaled
GGUF would carry `rope_theta` alone, be served plain RoPE, stay fluent and be wrong past the
model's original window, with nothing in any log to say so ([engine#1087](https://github.com/1bit-MONSTER/engine/issues/1087)).
So `pack` exits non-zero naming the type (`*.rope.scaling.type` = `yarn`/`linear`/`longrope`/…)
before it creates the output directory. Two rules *are* implemented and are accepted on the
forward route, which reads the block from `config.json`: `rope_type: "llama3"`
(Llama-3.1/3.2/3.3) and LongRoPE's `short_factor` (MiniCPM4). The same refusal covers a
declared-but-unusable `llama3` block, and every other declared type, at `1bit serve` /
`forward-serve` load time rather than at request time.

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

### What the GGUF converter refuses, and the qwen35 converter

`tools/gguf_to_q4nx.py` converts qwen3, qwen3moe and qwen2. Two other inputs are refused before
anything is written, with no output directory left behind (engine#2013):

- **A GGUF of another architecture.** Every tensor kind without a Q4NX name is listed in one message,
  with how many tensors of each there are. For `qwen35` the message points at the converter below.
- **A tensor type the installed `gguf` package does not know.** The refusal gives the number;
  143 is PTQ1_0, PrismML's ternary type, which `tools/ternary_to_q4_0.py` rewrites as Q4_0.

`tools/qwen35_to_q4nx.py pack <model.gguf> <out dir> [--tokenizer <dir>]` packs a **qwen35** GGUF:
layers of gated linear attention (GDN) with a full-attention layer every
`qwen35.full_attention_interval`. It writes the names the forward looks up
(`linear_attn.qkv_proj`, `self_attn.gate_proj`, `linear_attn.ssm_*`), reorders the 48 value heads
from the stored order to the grouping the forward expects, and stores `ssm_alpha`/`ssm_beta` as Q8
tiles padded to 64 rows. It reads the GGUF itself and needs only numpy. `verify <model.gguf>
<out dir>` checks every tensor of a container against the GGUF, including the direction of the
value-head reorder.

Its input has to be a plain qwen35 GGUF in Q4_0 or Q4_1. It refuses, by name:

- a file that still carries `prism.hadamard.*` keys: the weights are in a rotated basis, the tool
  writes no sign table, and the container would load and answer wrongly;
- PTQ1_0 tensors (ggml type 143).

So a PrismML ternary GGUF needs two steps before this one: PTQ1_0 to Q4_0
(`tools/ternary_to_q4_0.py`, which keeps the rotation) and the un-rotation, which is not in this
tree. The 27B container on the dev box (`__metadata__.converter = qwen35_to_q4nx.py`) was made from
an un-rotated Q4_0 GGUF.

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

**A chunk-list dispatch shares ONE A across its chunks: exact for gate and up, and reachable for down
by row placement.** At batch 1 every selected expert of a layer sees the same activation row, so gate
and up are one dispatch each. `down_proj`'s input is each expert's own SwiGLU output — but `A` is an
`M`-row int8 buffer and the output is `M` rows per chunk, so with expert `j`'s row in row `j` of `A`
(rows 8-15 zeroed) and `NCH=8`, chunk `j` carries that expert's answer in row `j`: a flag, not a new
signature (the 10-BO dispatch and the `WIN = 3` per-channel ordering are unverified on the array).
On this model (48 layers; per layer q, gate, k, v, o in the pb16i split, then gate|up|down) that is
**385 dispatches per token**, against 721 if down cannot share one — and, two paragraphs down, the
dispatch *rate* rather than the dispatch *shape* is what decides whether the case meets its own bar:

| per-dispatch fixed cost | down shares one A | down per expert | Qwen3-4B dense, same arithmetic |
|---|---|---|---|
| 0.086 ms (fit below) | **11.05 tok/s (1.06x)** | 8.37 (0.80x) | 10.45 |
| 0.116 ms (quoted floor) | 9.80 (1.01x) | 7.09 (0.73x) | 9.68 |

That is at 34 GB/s in-loop over a **1,703,854,080 B/token** stream — routed experts 1,019,215,872
(8 of 128 x 3 projections, pb16 NW=16) + attention 509,607,936 + `lm_head` 175,030,272, with
`--pad-n 1024` on gate/up taking it to 1,930,346,496 — and the ratio holds across 34-43.4 GB/s:
**0.97-1.06x the 4B-dense bar when down shares the dispatch, 0.68-0.80x when it cannot.** So sizing
a MoE case on the expert bytes alone understates the stream by 40 % (they are 60 % of it) and misses
a term that matters — but it is not the term that decides the case. That is the dispatch *rate*:

**Which submission path runs the loop decides the answer, and the bar pays the same term.** The
0.086 ms in that table is a *marginal* cost inside one command list, measured on the arena player. An engine
decode step submits one ELF per `npu_gemm`, and that per-invocation cost has since been measured on
this box in the engine's own `npu/forward` route: the T1BP-ZYPHRA lane's device arm (M=16 family,
112 launches/token, 95.0 % of the token spent waiting on the array) fits **2.8 / 3.7 / 5.91 ms** per
invocation from three independent pairs (2026-11-04 20:1x-20:27). Pricing this model and the
4B-dense bar on the *same* path — 385 and 253 dispatches/token, same 34 GB/s, same host terms:

| per-dispatch fixed cost | this MoE | Qwen3-4B dense, same path | ratio |
|---|---|---|---|
| 0.086 ms (arena player) | **11.05 tok/s** | 10.45 | **1.06x** |
| 2.8 ms (engine, driver-delta fit) | 0.88 | 1.28 | 0.69x |
| 3.7 ms (engine, two-point fit) | 0.67 | 0.99 | 0.68x |
| 5.91 ms (engine, launch-cost fit) | 0.43 | 0.64 | 0.67x |

Setting the two expressions equal puts the **break-even at 0.126 ms per dispatch**: below it the MoE
meets and beats the bar, above it it loses by ~1.45x and stays there, because 385 submissions against
253 (1.52x) outweighs 1.70 GB against 2.26 GB of stream (1.33x). The engine's measured path is 22-47x
over that threshold. So on the decode loop that exists today this case **misses** its own bar, and the
lever is submission granularity — one command list per token instead of one per `npu_gemm` (engine#799)
— not the chunk size (1.18-2.13 MB against the arena's 50 MB is a 1.3 % effect) and not a MoE-shaped
ELF built for gate/up alone.

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
the measured `tile_order`, repack script) names a different converter than this build would use (`repack_gguf.py
--check-stamp <dir>`: exit 0 current, 1 stale, 2 cannot tell). A stamp the repack itself refused,
or one that predates the measured `tile_order`, is stale and is rebuilt once, so the directory
becomes verified; `unverified` is "cannot tell", so serve reuses it with its warning instead of
re-running the measurement that could not run. If the checker or the converter cannot be found,
serve warns once per process and reuses the directory, so a missing script cannot stop serving. A
Q4NX directory served as it is (`1bit serve -m <Q4NX dir>`), such as a published FastFlowLM model,
is trusted as published. Its chunk layout can be checked first, without
dequantizing anything: `tools/check_q4nx_dir.py <dir>` reads the container header only, names each
quantized tensor's chunk kind, and refuses an unknown chunk size or a tensor shorter than its layout
needs. It cannot tell the two Q8 layouts apart ("Q4NX chunks", above), because the container does
not record which one it holds. It also derives every weight's expected tile grid from `config.json`
and names a weight whose converter padded that grid (engine#1097) — a gated attention layer's
`q_proj` is counted at `2 * num_attention_heads * head_dim` rows, because with `attn_output_gate` it
emits q **and** the output gate (engine#796: reading it as one head-width made the real d3 pack, and
every 27B `qwen3_5` pack on this box, look like a padded grid and fail `--strict`).

`--against <source.gguf>` closes both of those gaps for a directory that was never repacked here: it
decodes the first tile of the Q8 tensors under both Q8 layouts *and* the first tile of the `q4_1`
tensors under all three `q4_1` orders this repo can name (`ours`, `flm` signed, `flm` unsigned),
correlates each against the source GGUF, and refuses a container whose winner is not the engine's own
(`--q8-samples` sets how many tensors per chunk kind). That is the offline row engine#1110 asked
for — a directory imported from a foreign converter carries no stamp and no plan, so nothing but the
weights can say which order it holds, and the separation is wide: `ours` reads `corr +1.0000` on a
container packed from `Q4_K_M` and `+0.996` on one packed from BF16, while every foreign order sits at
`|corr| < 0.05`.

**The tile order is measured, not declared.** Two converters can write `q4_1` tiles in the two
different orders this engine and FastFlowLM use, and a container in the wrong order repacks cleanly
and decodes as noise — HTTP 200, nothing in the log (engine#1110; measured with FLM_Q4NX_Converter
`d1d5232`, whose containers come back `matches NO layout`). The load-time field
`container.tile_order` (engine#1130) cannot catch it, because `tools/npu_plan.py` writes `engine-1`
for every plan: it guards against a future writer, not against the converter that produced the
tiles. So the repack decodes a sample of the container against the source GGUF with
`tools/q4nx_diff.py`'s own verdict — one pass per layout, once per directory — and:

* **refuses** the directory (`rc=1`) when no layout matches, or when the layout that matches is not
the `engine-1` order `npu/forward/src/q4nx.cpp` decodes;
* records `tile_order=engine-1` and `tile_check=<the verdict line>` in `repack-stamp.txt` when it
does match, so the claim in `plan.json` is measured rather than assumed;
* records `tile_order=unverified` when nothing was comparable (an architecture the checker cannot
dequantize) instead of asserting a layout nobody checked.

The same measurement runs on its own over an existing directory, under the name that says what it
proves: `python3 scripts/repack_gguf.py --verify-source <dir> <model.gguf>` (exit 0 verified in the
engine-1 order, 1 another order or none, 2 nothing comparable). The verdict names the worst source
tensor and its correlation with the GGUF — a wrong converter checkout reads
`corr +0.0325 on blk.14.attn_output.weight` where an honest pack reads `corr +0.9994` — so the
number that separates them is in the line, not only the max|err|. `--tile-check` is the same
function under its old name. The repack runs it after the converter and refuses the directory when
it fails; `--skip-verify` opts out for the tight repack loops and stamps `tile_order=unverified`
(never `engine-1`), so the cheaper path is recorded rather than silently trusted. A cache hit
re-checks metadata only, so the cost is paid once per directory, on top of the converter's own pass.

### From a Ryzen AI OGA NPU pack

`1bit npu-pack import <pack dir> -o <out dir>` reads a pack AMD publishes for Ryzen AI
(`*_rai_*_npu_*`: `model.onnx`, `model.onnx.data`, `dd_metastate_*.fconst`, `config.json`,
`genai_config.json`, the tokenizer) and writes a model directory. `1bit npu-pack info <pack dir>`
prints what the pack states as JSON and writes nothing. The pack's weights and its config/tokenizer
contract are the model; the reader is written from the pack-side facts in
`amd-oss/npu-re/docs/FACTS-oga-pack-weight-layout.md`, and no vendor library, wheel or compiled
program is called or shipped.

**A pack can state its weights behind a placeholder.** The 1.9 generation's external-data packs (`phi-4`, and the
`com.ryzenai:MatMulNBitsBf` family) name each weight through an initializer with no bytes —
`model.layers.0.attn.o_proj.MatMulNBits.qweight.empty`, dims `[0]` — while the payload sits in a `…qweight.packed`
sibling the node never references, inside the ordinary ONNX external data (`eager-mm.onnx.data`, 9.73 GB on `phi-4`). A
reader that follows the node's inputs therefore sees zero-size tensors on a 9.7 GB pack, which is what made this genre look
like a proprietary blob. `npu-pack info` resolves the sibling and reports it: `graph_weights.packed_weight_siblings`, the
`ryzenai_matmuls_ops` count, and `graph_weights.external_row_stride: {matched, of, rule}` — the bytes fit
`length == N × (K/128 + 1) × 68` on **201 of 201** of `phi-4`'s matmuls, and the 68-byte row-per-128-K-block is the same
stride the `.fconst` attention tile already carries (2,176 B subvolume = 32 rows × 68 B). The FFN width for this genre comes
from the node's own `N`/`K` (`gate_proj`/`up_proj` → N, `down_proj` → K), because the weight is a 1-D byte blob with no
shape to read it from (engine#1503).

**A pack can be the weights without the text.** AMD's `_rai_1.8.0_npu_4K` repos do publish the
model's own `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
`chat_template.jinja`, `special_tokens_map.json`, `added_tokens.json` and `generation_config.json`,
but they are separate files in a repo of ~70, so a pack fetched weight by weight — which is how they
are pulled on this box — can be `model.onnx` + `model.onnx.data` + `dd_metastate_*.fconst` +
`config.json` and nothing else (engine#1024). The importer copies whatever text is there and,
because the command that fails without it is `1bit serve` ("not an NPU model directory"), both
commands now say which artifacts the pack has: `npu-pack info` carries
`text_artifacts: {present, missing, servable}` and `npu-pack import` prints the split, or a
`WARNING:` naming `tokenizer.json` when the pack has none. The split is also recorded in the
container's `config.json` (`q4nx_source.text_artifacts`), so a container states whether it ever had
the text.

What converts, and how faithfully:

| pack tensor | where | to | fidelity |
|---|---|---|---|
| `model.embed_tokens.weight` | `model.onnx.data`, fp16 | bf16 | fp16 → bf16 rounding (10 → 7 mantissa bits) |
| `lm_head` `MatMulNBits` (bits 4, block 128, asym) | `model.onnx.data` | q4_1f tiles | codes unchanged; `scale = s`, `zero = -zp*s`, both stored as f32 |
| `input_layernorm`, `post_attention_layernorm` | `.fconst`, bf16 at a fixed offset | bf16 | byte-exact |
| `self_attn.{q,k,v,o}_proj` (24 x 4) | `.fconst`, 34,816-byte 64-row tiles | q4_1f tiles | codes unchanged; `scale = s` (the pack's bf16 value), `zero = -zp*s` stored as f32 |
| `self_attn.{q,k,v}_proj.bias` (24 x 3) | `.fconst`, the tiles' bias subvolumes | bf16 | byte-exact; written only where the pack has one |
| `mlp.{gate,up}_proj` (24 x 2) | `.fconst`, MLP section, 4,288-byte subvolumes | q4_1f tiles | codes unchanged; `scale = s` (bf16), `zero = -zp*s` stored as f32 |
| `mlp.down_proj` (24 x 1) | `.fconst`, MLP section, 2,176-byte subvolumes | q4_1f tiles | same as the attention tiles |
| `model.norm` (final) | `.fconst`, the tail envelope at `layers x stride` | bf16 | byte-exact |
| `sin_cache_cos_cache` | `model.onnx.data` | not imported | the lane computes RoPE from `rope_theta` (`fill_rope`) |

**A multi-component pack's per-layer embedding** (engine#1725). gemma-4's embedding component
carries, beside the token table, `embed_tokens_per_layer.weight` (`vocab x layers*P` f32; 262,144 x
8,960 on e2b, 9.4 GiB), a `P`-wide `per_layer_projection_norm.weight` shared by every layer, and
the context projection `per_layer_model_projection.weight_t_Q4` quantised at block 32. The import
writes them under the names the forward loads:

| container tensor | from | form |
|---|---|---|
| `model.per_layer_token_embd.weight` `[vocab, layers*P]` | the f32 table | BF16, round to nearest even |
| `model.per_layer_proj_norm.weight` `[P]` | the norm | BF16 |
| `model.per_layer_model_proj.weight_layer{l}` `[P, hidden]` | the projection, dequantised, one per layer | BF16 |

The table is never held in memory, in either form: the reader points into the component's mapped
external data (its extent is checked once, when it is located), and the container writer converts
it 32 rows at a time while it writes the file. A projection with no zero-point tensor takes the
implicit zero point 8, the same rule as every other MatMulNBits here. `config.json` gets
`hidden_size_per_layer_input` and `vocab_size_per_layer_input` only when all three parts went into
the container, because those two keys are what switch the block on in the forward.

An OGA pack keeps its tensors inside the `If` branches: the top-level graph has **no
initializers** and four nodes (`Shape`/`Constant`/`Equal`/`If`). The importer walks the
subgraphs. The 4-bit decode is `w = (code - zp) * scale` with two codes per byte (low nibble
= even k) and two zero points per byte; that decoder is checked against the pack's own data,
because `tie_word_embeddings` makes `dequant(lm_head)` reproduce the fp16 embedding:

    corr 0.999999, relative error 0.00070 on all 151936 x 896 weights
    controls: nibble-swapped corr -0.0008, no zero-point corr 0.8788   (both rejected)

Two facts about what the container holds, both reported, not hidden:

* **The pack's scale and zero are kept exactly (engine#1185).** An imported container uses the
  fp32 `q4_1f` chunk (6,144 B), and its `layout` says `q4nx-2`; every written tile's scale/zero
  planes are read back and compared with the pack's f32 values. What this replaces: the 5,120-byte
  bf16 form kept only 43-48 % of the groups bit-equal on the four shipped packs (3.34e-3 relative,
  2.4e-4 absolute on the weights), because `-zp*s` is an arbitrary f32 and bf16 rounds it. That
  floor is gone for imported containers -- the same measurement is now bit-equal, and
  `npu-pack import` prints "fp32 scale/zero exact for 100.00 % of 32-column groups (container
  layout q4nx-2)". The device still streams bf16 q4_1, so the forward repacks a q4nx-2 container at
  load (`npu/q4nx_device.cpp` -> `q4nx::as_q4_1()`, cached once and logged; `npu/forward/src/model.c`
  still refuses an unconverted one as a safety net), and the fast lane's packer does the same in
  memory (`npu/pack.cpp`); GGUF-derived containers are unchanged and keep the bf16 form.
* **K is padded to the 256-column grid.** Qwen2.5-0.5B has hidden 896 = 3.5 tiles, so
  `lm_head` and q/k/v/o are stored as 4 column tiles with the last one zero-padded (codes 0,
  scale 0, zero 0). That is 12.5 % more bytes than the pack; the true K is in the container's
  manifest and in `config.json`, and nothing reads a padding column as a weight.

### Where a 1.9 pack's rewritten `.fconst` goes

A Ryzen AI 1.9 pack's Token `.fconst` is rewritten into its 1.8 form before it is read. The copy is as large as the source
(3.6 GB for an h4096 pack), so where it lands matters: on strixhalo `/tmp` is tmpfs, which means RAM. The importer tries these
directories in order:

1. `$ONEBIT_FCONST_SCRATCH`, if set;
2. `$XDG_CACHE_HOME/1bit/fconst19`, or `~/.cache/1bit/fconst19` (on disk);
3. `$TMPDIR`, then `/tmp`, then the current directory.

The copy is a cache: it is reused while the source's size and mtime and the rules version match. On each write the importer
sweeps the scratch directories. It removes other identities of the same pack and geometry, the pre-version name, and any
`1bit-fconst19*` copy nobody has written for 6 hours. Without an override, **every** candidate directory is swept, not only the
one being written, so copies older binaries left in `/tmp` get reclaimed (engine#1907). With `ONEBIT_FCONST_SCRATCH` set, only
that directory is swept, and a run with an override never removes another lane's files. The current directory is the
one candidate that is never swept unless the copy is being written there: it is only the last write fallback, and a
working directory's files are not the importer's to clean. The 6 hours are what keep a concurrent run's copy alive in a
shared directory, so the number stays above the longest import; it is not the knob for disk or tmpfs pressure. The test suite sets one per run and
removes it at exit.

The cache directory is shared: every import of the same pack on the box uses the same file name. The copy is therefore
written under a temporary name beside it (`1bit-fconst19tmp-<pid>-<n>-<name>`) and renamed into place, so an import finds
either no copy or a complete one (engine#1985). Before that the copy was written in place, and two imports of one pack
started a fraction of a second apart could leave the first reading a file the second had just truncated. The result was a
wrong layout reading with exit 0: `Qwen3-1.7B` reported 280 refused tensors, `qk_norm` false and 0 measured tiles in 4 of
24 runs of three imports 0.15 s apart, and 0 of 144 with the rename. A temporary file left by a killed run is removed by
the 6-hour rule above. A binary older than this change still writes in place, so a box where old and new binaries import
the same pack at the same moment can still show it.

### Linking the kernel set the container will be served with

A model directory carries its own NPU kernels in `npu/` (see "A model directory"), and for an imported
pack that set comes from this engine's kernel generator, not from the pack. Until now the last step
between import and serve was a hand-made symlink, so `1bit npu-pack import` can now make it:

    ./1bit npu-pack import ~/models/oga/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K \
        -o ctr-phi35-b8 --weight-layout b8 --kernels ~/ogaload/oga13/kernels-phi35-b8

    kernels:    ctr-phi35-b8/npu -> /home/bcloud/ogaload/oga13/kernels-phi35-b8  (8 full_i8 ELFs: 4 M=16 twins, 8 of the _b8 family)

and `1bit serve -m ctr-phi35-b8 --device npu` then finds them where it looks. Two properties are
deliberate:

* **The set is checked against `--weight-layout` before anything is written.** A `b8` container needs the
  `_b8` ELF names and a container with no layout key needs the plain ones; the forward looks kernels up
  by name, so the wrong family links cleanly and then fails at load with "no ELF for K=.. N=..". That
  refusal happens before the import writes a byte, so a bad `--kernels` never leaves a container behind.
* **An existing `npu/` that is a real directory is left alone** and named in a warning. Replacing someone's
  kernels to make room for a link is not an importer's decision; a stale *link* is replaced, which is what
  re-running an import over its own output does.

The link is absolute, because the set normally lives outside the model directory; moving the container to
another box means linking that box's set the same way. Leaving `--kernels` out is exactly the old behaviour
(the container has no `npu/`, and `1bit forward-serve -m <model.q4nx> --kernels <dir>` still takes the set
on the command line).

### Prefill beyond T=256

The prefill GEMM's kernel builds at T values above the 256 rows the shipped sets use, and each build passes the
kernel tree's own numeric gate (fp64 reference from the same quantised pool bytes, per-token cosine ≥ 0.99999) —
so `T > 256` is a capability, not a hazard. What it does **not** buy is throughput: per-row time is flat in T,
because the weight stream is re-read once per T-tile, and the kernel's own tiling is what bounds a run at this
size. The gate tool, the toolchain pins it needs, and the per-T rows live in the private kernel repository
(`1bit-MONSTER/npu-kernels`, `tools/npu_prefill_tgate.sh`, `docs/npu-prefill-tgate.md`), as does the design.

For this engine, the consequence is the one recorded in [engine#1421](https://github.com/1bit-MONSTER/engine/issues/1421):
a faster NPU prefill is an inner-loop and layout question in the kernels, not a parameter this page can set.

### The `.fconst` attention payload### The `.fconst` attention payload

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

**A 2-header block can carry the gains too, in the other order** (engine#1889, engine#1934). The
Hunyuan packs declare `use_qk_norm: true` and keep one header per block: the gains are the 640 B
between the last q/k/v tile and o_proj, which the pack's metadata names as one tensor
(`model.layers.L.attn.q_norm.layernorm.weight.qk_rms_packed_26_L`, bf16 `[320]`). Its layout is
`[64 B zeros][q_norm, bf16 [128]][64 B zeros][k_norm, bf16 [128]]` — **q first**, the reverse of
the Qwen3 sections above. The order is not readable from the bytes (the two gains differ by about
one bf16 ulp in a few elements, so both look like a gain), so it was taken from the pack's own
graph: `model.onnx` carries `attn.q_norm.layernorm.weight` and `attn.k_norm.layernorm.weight` as
FP16 `[128]` initializers, and on Hunyuan-7B-Instruct their bf16 rounding is the first and the
second block bit for bit in 32 of 32 layers. `qk_gain_offsets()` holds that arithmetic.

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
`config.json` (`q4nx_source.incomplete`). The shipped `Qwen2.5-0.5B-Instruct_rai_1.8.0_npu_4K`
now imports whole — **291 tensors, 67,088 tiles, no refusals**: embedding, head, the 48 layer
norms, the final norm, 96 attention projections, 72 attention biases and 72 MLP projections.
That holds for a pack whose layer projections are graph initializers (no `.fconst`) too: a
projection the decoder cannot read, for example codes present but no scales tensor beside them,
is refused by name like any other. It used to be skipped, so the import reported no refusals over
a container missing that projection (engine#1841).

### The final norm, and the tail envelope

The layer blocks stop at `layers x stride`; what follows in either `.fconst` is a 1,856-byte
envelope `[eps bf16 1e-6][62 pad][hidden x bf16]` — the **final norm** — and, in the
`Llm_Token_Token` file, the `lm_head` tile stream after it (which is why the head's payload was
found verbatim at `tail + 1,856`). The position is not assumed: the epsilon has to be at
`layers x stride`, and where the pack ships both files the two copies of the norm have to be
byte-identical, which is what the import requires before it writes either. On the 0.5B pack the
stored bytes equal the upstream checkpoint's `model.norm.weight` exactly. (The earlier reading —
"the tail envelope is `[eps][pad][448 f32]`, so the final norm's position is unknown" — was the
last 1,856 bytes of the *file*, which in the Token_Token program are tile padding.)

**A head the pack declares at `block_size 32` is not in that tile stream** (engine#1740). The three
Ryzen AI 1.9 Qwen-1.5B packs quantise every layer at block 128 and `/lm_head/MatMul_Q4` at block 32,
and their tail after the envelope is 64-row *units* instead: 2,560 zero bytes, then `2 x (K/32)`
strips of 640 B. Strip `s` holds K block `s/2` for rows `32 x (s%2) .. +31` of the unit: 512 B of
4-bit codes (low nibble first; position `p` is row `(p/256) x 8 + p%8`, column `(p/8) % 32`), 16 B
of zero-point nibbles, 48 zero bytes, and 32 **signed** bf16 scales, one per row;
`w = (code - zero_point) x scale`. After the last unit the file is zero to its end (110,208,000 B on
the measured pack, which is why the block-128 arithmetic never landed there). `npu-pack import`
reads it (`read_block32_head`), folds a negative scale into its codes so the container's scales are
positive, and writes the same q4_1f tiles as any other head. The layout checks are part of the read
— unit count `ceil(vocab/64)`, the zero fields, finite scales, zeros to the end of the file — and
each failure is a refusal that says which. On a pack that declares tied embeddings the read head
also has to correlate above 0.99 with `model.embed_tokens`; if it does not, the head is made from
the embedding as before and `q4nx_head_note` says why the read was not taken. On an untied pack the
layout checks are all there is, and the note says the values are not verified. Measured on
`Qwen-2.5_1.5B_Instruct_rai_1.9.0_npu` (tied, 151,936 x 1,536, 2,374 units of 64,000 B): every scale
equals `bf16(-m/8)` of its 32-column embedding block, `m` being the block's largest-magnitude
weight, and the read head correlates 0.996 with the embedding.

**A pack whose every MatMulNBits is `block_size 32` keeps its layers in those units too**
(engine#1503). `SmolLM2-135M-Instruct_rai_1.9.0_npu` declares block 32 on all 151 ops, and its
*raw* Token `.fconst` (the file as shipped, not the 1.8-form copy the block-128 path makes) is
`layers` blocks of one fixed shape and then the tail. With `H` = hidden, `Nq` = heads x head_dim,
`Nkv` = KV heads x head_dim, `IM` = intermediate, `unit(K) = 2,560 + 2 x (K/32) x 640` (the head's
unit above) and `mlp_unit(K) = 4,864 + 2 x (K/32) x 1,216`:

| section | bytes | holds |
|---|---|---|
| envelope | `64 + 2H` | bf16 `rms_norm_eps`, zeros to 64, `input_layernorm` bf16[H] |
| q, k, v | `(Nq/64 + 2 x Nkv/64) x unit(H)` | `q_proj` [Nq][H], `k_proj` [Nkv][H], `v_proj` [Nkv][H] |
| — | `4 x heads` | not a weight; skipped |
| o | `(H/64) x unit(Nq)` | `o_proj` [H][Nq] |
| envelope | `64 + 2H` | epsilon, zeros, `post_attention_layernorm` bf16[H] |
| gate + up | `(IM/64) x mlp_unit(H)` | `gate_proj` and `up_proj`, [IM][H] each |
| — | 64 | skipped, not interpreted (`01` then zeros on the measured pack) |
| down | `(H/64) x unit(IM)` | `down_proj` [H][IM] |

The tail is the same envelope with the final norm, then `ceil(vocab/64)` head units at K = `H`,
then zeros. On SmolLM2-135M the block is 2,360,292 B (q at 1,216, k at 231,616, v at 308,416, o at
385,252, the second envelope at 615,652, gate/up at 616,868, down at 1,784,292), the head is at
70,809,976 and the file is exactly `30 x 2,360,292 + 1,216 + 768 x 25,600 = 90,470,776` B.

The **gate/up unit** is the head's unit with 64 row *slots* per strip instead of 32 rows: 4,864
zero bytes, then `2 x (K/32)` strips of 1,216 B. Strip `s` holds K block `s/2`; slots 0..31 are
`gate_proj` rows `32 x (s%2) .. +31` of the unit and slots 32..63 are the same rows of `up_proj`.
A strip is 1,024 B of 4-bit codes (low nibble first; position `p` is slot `(p/256) x 8 + p%8`,
column `(p/8) % 32`), 32 B of zero-point nibbles, 32 zero bytes, and 64 signed bf16 scales. The
code order was measured against the public checkpoint; that zero point and scale are indexed by
slot is assumed by analogy with the 640-byte strip, and is tested by the imported container's
gate/up still correlating with the checkpoint once they are applied.

`npu-pack import` takes this path when every op the pack's `cache/*_meta.json` declares is block 32,
at least one of them is a layer projection, and the pack has a Token `.fconst` (a pack that
declares only its head at 32 is the case above). Nothing is located by search: the offsets are
arithmetic on the geometry, and the file has to agree with them — at least
`layers x stride + 64 + 2H + ceil(vocab/64) x unit(H)` bytes, the pack's epsilon and zeros at every
envelope, zero unit headers and strip gaps, finite scales, zeros after the head — or the tensors
are refused by name with the offset. The container gets the same tensors in the same q4_1f tiles as
a block-128 import; this layout has no bias field, so no bias tensor is written. There is no
decode->encode round trip on this path (no encoder for these units exists), so the import's
`attention:` and `MLP:` round-trip lines are replaced by a `block 32:` line. Measured on one pack
only, where `Nq == H`; `o_proj`'s K being `Nq` is the declared shape, not a measurement.

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

`npu-pack`'s writers hold the container to that unsigned field: `tile_q4_1`, `tile_q4_1_f32` and the
`.fconst` tile encoder all refuse a code above 15, naming where it is, rather than mask it into
a different code one full range away (engine#1921). A signed code has to be converted before it
reaches a writer, which is what the 1.9 decoder does.

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

### A pack's vision component (`vision_weight.q4nx`, engine#1788)

FastFlowLM's MiniCPM-V 4.7 packs are two containers at the pack root: `model.q4nx` (the text model) and
the file `config.json` names as `vision_model_weight`, `vision_weight.q4nx` (the vision tower). The
second is the same container format holding a ViT: every tensor BF16, no tiles. `npu/vision_pack.h`
reads it, and `1bit npu-pack vision <pack dir>` prints what it found:

```
$ 1bit npu-pack vision MiniCPM-V-4.7-1B-NPU2
vision component  MiniCPM-V-4.7-1B-NPU2/vision_weight.q4nx
tensors           347, all BF16, 1135109120 B of payload, 0 B unaccounted
patch embed       14 x 14 pixels x 3 channels -> 1152
positions         4900 (a 70 x 70 grid; image 980)
blocks            27 at width 1152, 16 heads x 72, MLP 4304 (4608 stored)
vit_merger        after block 6; attention at 1152, MLP 4608 -> 17216 (17280 stored) -> 1152
merger            4608 -> 4608 -> 1024 (the text model's hidden size)
source_file       mmproj-model-f16.gguf
source_sha256     fb9cc047a3cdad5317fd8433a4aeda2d22f00f02ca4bd93b34ad84165b25a900
note              config.json states MINICPM_V_VISION_MLP_INTERMEDIATE_SIZE 4304; model.visual.blocks.*.mlp.linear_fc1.weight has 4608 rows, the last 304 all zero: the container pads the model's width, and sizes come from the tensor
note              config.json states MINICPM_V_VIT_MERGER_INTERMEDIATE_SIZE 17216; model.visual.vit_merger.mlp.linear_fc1.weight has 17280 rows, the last 64 all zero: the container pads the model's width, and sizes come from the tensor
```

What the reader settles is what a forward may rely on.

- **Sizes come from the tensors.** The geometry is derived from the tensors' own shapes, and every
  tensor must have the shape the others imply.
- **The MLPs are stored padded.** The model's MLP widths are 4304 (blocks) and 17216 (`vit_merger`);
  that is what `config.json` states and what the source GGUF's tensors are. The container stores them
  4608 and 17280 wide, the next multiples of 128, with the extra `linear_fc1` rows, bias entries and
  `linear_fc2` columns all zero. A loader that sized a buffer from the config would be 304 rows short
  of what the file holds. The reader sizes from the tensor, verifies that everything past the stated
  width is zero in every block, and carries the model's width beside the stored one
  (`Geometry::model_intermediate`). If any value past the stated width is not zero it says so and
  keeps the tensor's width; a config that states an MLP wider than its tensors is refused.
- **Zero gaps.** A tensor the reader has no place for is a refusal, as is a missing one, a hole in the
  block indices, a tensor that is not BF16, an extent that is not its shape's byte length, an extent
  outside the payload, and two tensors on the same bytes. Each names the tensor.
- **`config.json` is consulted for what the tensors cannot say** (the head count, the image size, the
  unpadded MLP widths) and refused where the two cannot both be right: heads that do not multiply to
  the width, an image size that is not the position grid times the patch, another width, block count
  or position count, and a text `hidden_size` that is not what the merger projects to. That last one
  is a tower paired with the wrong text model.

**Checked against the source, bit for bit** (2026-11-10, both published packs). The 1B container's
`__metadata__` names `mmproj-model-f16.gguf` and its SHA-256, which is the file of that name in
`openbmb/MiniCPM-V-4.7-1B-gguf`. All 347 pack tensors were rebuilt from that GGUF and compared as
BF16 bit patterns: 263 identical, 84 identical in the source's extent and `+0.0` in the padding, 0
different. The pack's `attn.qkv` is the source's `attn_q`, `attn_k`, `attn_v` stacked in that order;
`merger.norm` is the source's `v.post_ln`. Every F32 source tensor is exactly representable in BF16.
The 35B-A3B container carries no `__metadata__`; the same comparison against
`openbmb/MiniCPM-V-4.7-35B-A3B-gguf`'s `mmproj-model-f16.gguf` (SHA-256 `bff3b688…5d20f1`) gives the
same 263 / 84 / 0, and against the 1B's source gives 0 of 347, so the two towers share a geometry and
not a single tensor. One source tensor has no counterpart in either container: `v.tok_embd_sep`
(`[4, hidden]`).

Pinned by `tests/npu_vision_pack_test.cpp` on crafted containers; `ONEBIT_MINICPM_PACK=<pack dir>` adds
the published pack's own numbers. The source comparison above was run by hand and is not a test in the
tree.

**The tower on the host (`npu/vision_tower.h`).** `vision::tower_forward(pack, pixels, gh, gw)` runs the
tower in float32 on the CPU: normalised pixels for a `gh × gw` patch grid in, `(gh/4) × (gw/4)` rows at
the text model's width out. It is a reference for whatever runs the tower on the NPU later, not a
serving path; nothing calls it from a serve, and it does no image decoding, resizing or slicing. The
order of operations is the transformers implementation's (`transformers.models.minicpmv4_7`):

1. patch embedding (a Conv2d whose kernel is its stride), patches row-major;
2. plus one learned position row per patch, picked **nearest** on the 70 × 70 table: row
   `floor(y·70/gh)`, column `floor(x·70/gw)`. No interpolation and no rotary term;
3. blocks 0–6: LayerNorm (eps 1e-6), full attention, MLP with the tanh GELU;
4. the `vit_merger`, once, after block `insert_layer_id` (6): attention inside each 2 × 2 window only,
   then per window an MLP over the four tokens side by side plus their mean. The grid halves;
5. blocks 7–26 on the halved grid, then `merger.norm`;
6. the merger: 2 × 2 groups again, `pre_norm`, `linear_fc1`, the exact (erf) GELU, `linear_fc2`.

So both grid extents must be multiples of 4, and anything else is refused. The padded MLPs are read at
the tensor's stride and the model's width.

Measured against that implementation (transformers 5.22.2, torch 2.13 CPU, float32, eager attention)
loaded with each pack's own weights, same input (`pixel(c, y, x) = sin(0.37c + 0.11y + 0.05x +
0.003xy)`), 2026-11-10:

| pack | grid | tower before the merger: max \|d\|, corr | output: max \|d\|, corr |
|---|---|---|---|
| 1B | 8 × 12 | 2.9e-5, 1.000000000 | 1.1e-6, 1.000000000 |
| 1B | 16 × 20 | 1.1e-5, 1.000000000 | 2.4e-6, 1.000000000 |
| 35B-A3B | 16 × 20 | 9.5e-6, 1.000000000 | 9.5e-7, 1.000000000 |

The loops are spread over threads (`ONEBIT_VISION_THREADS`, default the machine's count capped at 16;
`vision::tower_threading()` in code) and the matrix products walk each weight once per handful of
tokens instead of once per token. Neither changes a bit of the output: every token and every attention
head is computed by one thread with the single-threaded arithmetic, and the test requires 1, 2, 5 and
16 threads to give identical rows. Measured on a full 448 × 448 view (1,024 patches) of the 1B pack,
box load about 13: 37.7 s before, 18.1 s on one thread with the blocked product, 3.2 s on 16 threads.
It is still a reference: a 70 × 70 slice has nearly five times the patches and its attention grows with
the square.
`tests/npu_vision_tower_test.cpp` carries a small tower with LCG weights and that implementation's
output for it as goldens, so the arithmetic is checked in CI without a pack; with
`ONEBIT_MINICPM_PACK` and `ONEBIT_MINICPM_ORACLE` set it repeats the comparison above. The reference
run itself needs transformers and torch and is not in the tree.

**Where an image goes before the tower (`npu/vision_plan.h`).** `vision::plan_image(height, width)` is
the arithmetic of the model's image processor (transformers' `MiniCPMV4_6ImageProcessor`, which 4.7
loads), with no pixel touched: the model looks at an image whole, resized to about 448 × 448 pixels,
and, when the image is larger than that, again as a grid of up to 9 slices cut from a second, larger
resize. Every piece is a multiple of 56 pixels (4 patches) in both extents, because the tower merges
2 × 2 twice, and becomes `(height/56) × (width/56)` rows on the text side.

| image | whole view | slices | text rows |
|---|---|---|---|
| 448 × 448 or smaller (scaled up) | 448 × 448 | none | 64 |
| 480 × 640 | 392 × 504 | 1 × 2 of 560 × 392 | 63 + 2 × 70 |
| 1080 × 1920 | 336 × 616 | 2 × 4 of 448 × 448 | 66 + 8 × 64 |
| 3024 × 4032 | 392 × 504 | 3 × 3 of 392 × 504 | 63 + 9 × 63 |

(heights first.) So one 448-pixel slice is 64 rows, not the 1,225 that 980 / 14 / 2 squared suggests: the
position table is 70 × 70, but no piece the processor produces is that large unless the image is a very
long strip. The plan is checked against the reference's own answers for 3,481 image sizes in
`tests/npu_vision_plan_test.cpp`. Nothing is cropped or padded and no size is refused for not being a
multiple of anything: every image is resized whole (385 × 384 becomes 448 × 448, so the aspect ratio moves
a little, as it does in the model's own processor), and the slices tile the refine size exactly. The plan
does not bound its result: a 1,048,576 × 1 strip gives the reference's answer, a piece 458,752 pixels
tall, and it is the caller that decides how many rows to accept. `plan_image()` reads no file and touches
no pixel, which is what lets it be checked anywhere; keep it that way.

**From bytes to the tower's input (`npu/vision_image.h`).** `vision::preprocess(rgb, height, width, plan)`
resizes an RGB byte image to the plan's whole view and, when it is sliced, to the refine size, cuts the
slices, and returns each piece as `(3, height, width)` float32 with value `(byte − 127.5) / 127.5`
(rescale by 1/255, then mean 0.5 and std 0.5, the same for each channel; R, G, B order in and out; byte
0 is exactly −1 and byte 255 exactly +1): the whole view first, then the slices row by row. The resize is bicubic with antialiasing on bytes and
reproduces `torch.nn.functional.interpolate(uint8, mode="bicubic", antialias=True)` (torch 2.13) bit
for bit: the a = −0.5 cubic stretched by the scale when shrinking, integer weights at the largest
precision that still fits int16 (PIL's fixed 22 bits differs by one in about 1 % of bytes), width
first and then height. `tests/npu_vision_image_test.cpp` holds the reference's bytes for eleven
resizes as hashes.

End to end on the 1B pack, 2026-11-10: a 460 × 470 synthetic image through plan, resize, pieces and
`tower_forward()`, against the reference's plan, torch's resize and the transformers tower on the same
image:

| piece | patches | rows | max \|d\| | corr |
|---|---|---|---|---|
| whole view 448 × 448 | 32 × 32 | 64 | 6.4e-6 | 1.000000000 |
| slice 0, 616 × 336 | 44 × 24 | 66 | 7.5e-6 | 1.000000000 |
| slice 1, 616 × 336 | 44 × 24 | 66 | 6.7e-6 | 1.000000000 |

That run took 110 s on one thread (before the tower was threaded), which is the host reference's cost and the reason it is not a
serving path. Two things are not established: the reference side called torch's `interpolate`
directly, on the understanding that this is what the processor's torchvision resize does for byte
images on a CPU build with AVX2 (torchvision is not on this box, so that link is read, not run); and
no real photograph or image decoder has been through it. The slice separators and placeholder rows on
the text side are not written.

**The text side of an image (`npu/vision_prompt.h`).** Two more pieces of arithmetic, on token ids only:

- `vision::image_tokens(plan, markers)`: what replaces one `<|image_pad|>` in the rendered prompt.
  `<image>`, one placeholder per row of the whole view, `</image>`, then each row of slices as
  `<slice>` placeholders `</slice>` per slice, with one newline token between rows. The tower's rows go
  into the placeholders in order. (`<image_id>k</image_id>` in front is ordinary text, left to the
  caller.)
- `vision::prompt_positions(ids, markers, grids)`: the three position planes (t, h, w) for a prompt that
  contains images, the model's `mrope_mode: canvas`. Text counts 1, 2, 3 on all three planes. An image
  is laid on a canvas whose origin is the next free position: the slices tile it, the whole view is
  stretched over it, `<image>` sits one step before it and `</image>` on its far corner, and the text
  after it resumes at `origin + max(canvas height, canvas width) + 1`. So an image moves the position
  counter by its **side**, not by its row count: a 480 × 640 picture is 203 placeholder rows and
  advances the position by 15.

Both follow transformers' `MiniCPMV4_7Processor.replace_image_token` and
`MiniCPMV4_7Model.get_rope_index`, and `tests/npu_vision_prompt_test.cpp` holds that code's answers for
eight prompts (one to two images, with and without slices, touching, newline-separated, with id
prefixes, and none) as hashes over every token id and position. Images only: video frames are not
handled. Nothing feeds these planes or the tower's rows into the text forward yet; that is the route's
work.

**One picture, one question, end to end (`minicpm_image_probe`).** A probe under `ONEBIT_NPU` that
chains every step above with nothing from outside the engine:

```
minicpm_image_probe <pack dir> <kernels dir> (--image f.ppm | --solid R,G,B [--size HxW] | --halves R,G,B:R,G,B | --none)
                    --prompt "question" [--decode N] [--expect-ids ids.i32]
```

RGB bytes → `plan_image` → `preprocess` → `tower_forward` on the CPU → `image_tokens` and
`prompt_positions` → the text model's `step_batch_embeds()` with the tower's rows where the
placeholders are, then a greedy decode. The image is a binary PPM or a generated colour, so no image
decoder is involved, and the prompt is the one the pack's chat template renders for a single user turn
with one image and thinking off, written out in the probe rather than rendered. The text model opens
the device unless `NPU_INFER_HOST_GEMM` is set.

It is a probe, not a route: `forward-serve` does not accept an image, and the tower runs on the
CPU (about 3 s per 448 × 448 view on 16 threads, 18 s on one). The measured rows are in
engine#1788. On the published packs the text model only loads once the padded-head rule of
engine#2026 is in: their head is stored 2 and 16 rows past the vocabulary, and without that rule the
probe stops at `text model: lm_head dequant failed`.

Three options exist to bisect a wrong answer against a reference run of the same prompt, and are how
the missing q/k norm above was found:

- `--dump <dir>` writes what the text model is about to be given, in `hybrid_prompt_probe`'s file forms:
  `ids.i32`, `positions.i32` (the t, h and w planes one after another) and `pooled.f32` (the tower's rows).
  If these match a reference pipeline's, the fault is past the inputs.
- `--chunk N` sends the prompt in calls of at most N rows, as the served route sends a prefill; `0` (the
  default) sends it in one call. Different answers at different N point at the batched body.
- `--rows-argmax` (with `--none`) prints the model's own next-token choice at every row of a text prompt,
  from one teacher-forced pass. Compared with a reference's per-row choices, a low agreement that is flat
  across positions is an error in the layers; a step at some row is an error at a position or a batch edge.

**Per-head q/k normalisation is read from the container (engine#1788).** Whether the forward normalises q
and k per head used to be decided by `config.json`'s `model_type` alone: on for `qwen3*` and `gemma*`.
A pack of the same architecture under another name got it wrong in silence. FastFlowLM's MiniCPM-V 4.7
is `minicpmv4_7` around a `qwen3_5` text model, and every full-attention layer of its container ships
`self_attn.q_norm` and `k_norm`; the forward ran those layers without the norm. The model still answered
short prompts (`Paris`, `red`) and drifted on longer ones. Now a layer that carries both tensors turns the
norm on, and the load says so:

```
[semantics] qk_norm=1  <- the container: layer 3 carries self_attn.q_norm and k_norm (config.json's model_type is not a family the table knows)
```

The rule never turns the norm off: a family the table knows keeps its answer. Measured on the 1B pack,
host arm, against the model's original weights in transformers (2026-11-11):

| | before | after | reference |
|---|---|---|---|
| a 149-token story, "what is the name of Maria's cat?" | `The question mentions that Maria buys` | `Pluto` | `Pluto` |
| top-1 agreement with the reference over those 149 rows | 65.1 % | 75.2 % | |
| rows where the forward's choice is outside the reference's top 10 | 6 | 0 | |
| a sliced 448 × 896 image (red left, green top right, blue bottom right) | "three equal horizontal sections … white … red" | red left, green top right, blue bottom right | the same three regions |

What is left of the disagreement (a quarter of the rows) is between a 4-bit container and the float
weights; it has not been separated into quantisation and anything else.

## Serving

The 1bit NPU engine runs the model on the XDNA 2 NPU. With a release package that carries the NPU
kernels (dense Qwen3: 0.6B, 1.7B and 4B):

```
./1bit serve -m <Qwen3 NPU model dir> --port 8000
```

In Lemonade, the `onebit` recipe's `npu` backend serves the same directories (`--onebit npu`).
The model directory holds the weights only; the kernels ship inside the package.

**Streaming.** With `"stream": true` the route sends each piece of the answer as it is produced. The first frame
leaves once the prompt has been processed and the first visible piece exists (a thinking model's opening markup is
held until it can be classified, so that is a token or two after the prompt is done), and the rest follow about
one per token. Before engine#1785 every frame was sent after the generation had ended, so a client's time to first
token was the whole request. A request that fails before producing anything (a prompt over the context, a
malformed field) still gets its own status code, not a 200 stream; one that fails later reports the error as a
last `data:` frame. A client that disconnects stops the generation at the next token, which frees the device for
the next request. The streamed reply and the non-streamed one are the same text.

### FastFlowLM's NPU2 drops: the families the route does not serve

The daily census-watch (engine#588) prints one line per new or changed repo in FastFlowLM's HF
org and answers two questions from `registry/architectures.json`: does a backend accept the
architecture the repo's config declares, and does the NPU route already serve that family. A
`!! NEW FLM` row means the registry does not map the class at all; a `!! NPU FLM` row means the
registry maps it, but only for a backend that cannot run a Q4NX directory. Both answers come from
the registry, never from the repo name.

The route serves the families `registry/architectures.json` marks `npu`: `qwen3` and `qwen3_5` on
the dense fast lane, and `qwen2`, `qwen3moe`, `qwen3_5_moe`, `deepseek2`, `minicpm` and `llama`
through the GGUF-on-NPU forward ([registry.md](registry.md) has the table). Everything else in the org is an
NPU-route gap, recorded here instead of as a GGUF alias: a class enters the registry only when a
backend's code accepts it, so mapping an NPU pack to an hrx-only family would advertise a model
the route refuses at load.

The 2026-11-05 census body (engine#619) asked about these packs. Each row is a recorded answer,
not a mapping:

| repo (2026-11-05 body) | architecture it declares | `registry/architectures.json` | NPU route |
|---|---|---|---|
| `FastFlowLM/MiniCPM-OCR-1.0-NPU2` | `MiniCPMOcrForConditionalGeneration` | none (the pin registers `MiniCPMForCausalLM`, `MiniCPM3` and `MiniCPMV4_6` only) | not served: an OCR/vision tower; the route claims text decoders only |
| `FastFlowLM/MiniCPM-V-4.7-35B-A3B-NPU2`, `FastFlowLM/MiniCPM-V-4.7-1B-NPU2` | `MiniCPMV4_7ForConditionalGeneration` | none (V4_6's text tower maps through `qwen35`, a VLM wrapper the pin does not register) | not served: a vision-language pack; a converter registration for the text tower plus a vision scope decision come first |
| `FastFlowLM/Qwen3.8-27B-DFlash2-NPU2` | `DFlash2DraftModel` | none; reviewed not-an-alias (docs/arch-gaps.md, `dflash2`) | not served: a speculative drafter, not a standalone decoder; it needs the reviewed dflash2 support, not a mapping |
| `FastFlowLM/LFM3-1.7B-NPU2` | `Lfm3ForCausalLM` | none (LFM2 / LFM2-MoE are mapped; LFM3 is not registered at the pin) | not served: no LFM3 model type or kernels in the route |
| `FastFlowLM/Hy-Memory-30B-A3B-NPU2` | none on 2026-11-05 (no config); `HYV3ForCausalLM` / model_type `hy_v3` by 2026-11-06 | `HYV3ForCausalLM` -> `hy_v3`, hrx only | not served: an hrx-only family; the 2026-11-06 dev-box run prints it as `!! NPU FLM ... npu route: no` |
| `FastFlowLM/smolvla-NPU2` | none (no config), no `*.q4nx` | none | not a pack: a config-less VLA repo. It prints as `!! NEW FLM` because its config carries no `library_name` for the custom-code rule, but there is no architecture to map and no Q4NX file to serve |

A pack whose family the route should serve needs that family's model type and its kernels in the
release before a registry entry means anything. Until then the row above is the record: the daily
report prints a repo only when it is new or changed, so a quiet day is quiet, and the next change
to one of these repos is a prompt to update this table rather than a namespace error.

The 2026-11-06 census body (engine#1253) read the whole org from a fresh state, so it also printed
every pack whose family the registry maps for `hrx` only. Those are the `!! NPU FLM` rows: the
architecture is covered on the CPU route, but a Q4NX directory needs the NPU route and `npu` is
not in the entry's `backends`. Family by family (`qwen35` has since moved out of that set, because
engine#587 measured the route on it):

| family (`gguf`) | packs in the 2026-11-06 body | `backends` | NPU route answer |
|---|---|---|---|
| `qwen35` | `FastFlowLM/D3-*-NPU2` (Edge, Flash, Mini, Nano, Lite, Turbo, Ultra), `FastFlowLM/Qwen3.8-27B-NPU2`, `FastFlowLM/Qwen3.5-0.8B-NPU2` | hrx, npu | **served, and measured** (engine#587): the same `Qwen3.8-27B-NPU2` family runs on the NPU route — 0.2902 s/token (3.45 tok/s) and 19.55 tok/s of prefill on the sealed single-ELF ternary rung, 22.811 s to a first token with the weight load in it (`bench/npu-leaderboard/27b-t2s2-*-20261106.json`) — int8 Q4NX decodes at 63.1 s/token and takes 145.7 s to a first token (`bench/npu-leaderboard/27b-*.json`, 2026-10-27), a 5-token prefill chunk is 58.14 s, and the ternary sealed-ELF rung on the same container is 1.72 s/token with 3/3 text-identical output (engine#697, closed by #728). Quality on the same route: **GSM8K 97.5 % (39/40)** with greedy decoding, **6 of the 40 answers cut off by the 512-token budget and exactly 1 of those 6 also wrong** — so among answers that finished, 34/34 were right (`27b-gsm8k-20261107`); and the per-token cost at those answer lengths is 0.3857 s/token against 0.2902 on a 13-token prompt, i.e. it grows ~1.33x with context (`27b-decode-cot-20261107`). What a stock install still needs is a kernel set for the model's geometry — none ships in this repository — so the row read `not served` until the route itself was measured |
| `dflash` | `FastFlowLM/Nemotron-3.5-Bolt-E4B-DFlash-NPU2` | hrx | not served: a draft head, not a decoder; "The draft loop, and whether to serve it" below covers what the HRX route does with it |
| `gemma4` | `FastFlowLM/Nemotron-3.5-Bolt-E4B-NPU2`, `FastFlowLM/Gemma4-12B-IT-NPU2` | hrx | not served: no gemma4 model type or kernels in the route |
| `hunyuan-dense` | `FastFlowLM/Hy-MT2-1.8B-NPU2` | hrx | not served |
| `gemma3` | `FastFlowLM/medgemma-1.5-4b-it-NPU2` | hrx | not served |
| `lfm2` | `FastFlowLM/LFM2-2.6B-NPU2`, `FastFlowLM/LFM2-1.2B-NPU2`, `FastFlowLM/LFM2.5-1.2B-NPU2` | hrx, npu | `LFM2.5-1.2B-NPU2` **served and measured** on the per-op route ("FastFlowLM's own containers" below); `LFM2-1.2B-NPU2` serves too (one paired ticket, same section); `LFM2-2.6B-NPU2` was not run |
| `gpt-oss` | `FastFlowLM/GPT-OSS-20B-NPU2`, `FastFlowLM/GPT-OSS-Safeguard-20b-NPU2` | hrx | not served |
| `hy_v3` | `FastFlowLM/Hy-Memory-30B-A3B-NPU2` | hrx | not served; the first table above records this one |

Adding `npu` to one of those entries would claim kernels this build does not carry. A family leaves
this list when `NPU_ARCHS` in `tools/registry_build.py` names its HF class with the model type the
route loads and a captured `--device npu` run stands behind it; the regenerated entry then carries
`npu` in `backends`. Note that the route does serve `qwen3_5_moe` (`Qwen3.6-35B-A3B` through the
GGUF-on-NPU forward), while every D3-* pack in this body declares the dense `qwen3_5` family, so
the two are not the same answer and the registry keeps them apart.

### FastFlowLM's own containers, served as published: measured rows (engine#1786)

The tables above answer "does the registry map this family". This one answers a different question for the packs
it lists: does FastFlowLM's own `model.q4nx`, with nothing converted, load and answer on `--device npu`. Each row
is one `npu-hold` ticket on the dev box (2026-11-10/11), the pack's own chat template, temperature 0, and the
question "What is the capital of France?" asked three times with thinking off unless noted; "identical" means
the three replies were the same bytes. The answer is a sanity check, not a comparison with FastFlowLM's output.

**Fused route** (the build shipped as `rc-2026-11-07`, which carries sealed sets for three dense Qwen3 geometries).
Nothing is edited: `1bit serve -m <the pack directory as published> --device npu`, and the log says
`on the private route dx … (…/npu/kernels/<set>)`.

| pack | sealed set | ready | answer | decode on a 55 to 96 token reply |
|---|---|---|---|---|
| `Qwen3-0.6B-NPU2` | `1024x3072x16x8` | 2.4 s | identical, "The capital of France is Paris." | 64.0 tok/s |
| `Qwen3-1.7B-NPU2` | `2048x6144x16x8` | 2.2 s | identical | 39.0 tok/s |
| `Qwen3-4B-NPU2` | `2560x9728x32x8` | 12.4 s | identical | 9.4 tok/s |
| `Qwen3-4B-Instruct-2507-NPU2` | `2560x9728x32x8` | 6.2 s | identical | 16.6 tok/s |
| `Qwen3-4B-Thinking-2507-NPU2` | `2560x9728x32x8` | 5.1 s | with thinking on: `Paris`, reasoning split out | 16.2 tok/s |
| `Qwen3-VL-4B-Instruct-NPU2`, text tower only | `2560x9728x32x8` | 2.8 s | with no `chat_template_kwargs`: `Paris` | 16.7 tok/s |

The two `Qwen3-4B` rates were taken in different tickets at different box loads; the gap between them is not a
model difference until a same-ticket pair says so. On the same array and the same `model.q4nx`, FastFlowLM 1.0.12's
own runtime decoded `Qwen3-1.7B-NPU2` at 45.7 tok/s against 39.3 here, with time to first token level (0.59 s and
0.57 s): one interleaved theirs, ours, ours, theirs ticket (engine#1787).

On this route `enable_thinking: false` is wrong for two of the packs, because the route builds its own prompt and
adds an empty think block whatever the pack's template says: `Qwen3-4B-Thinking-2507-NPU2` (a template that
always thinks) returns its reasoning and a literal `</think>` as `content`, and `Qwen3-VL-4B-Instruct-NPU2` (a
template with no thinking at all) answers with a refusal or an apology. Both are engine#2135, open.

**Per-op route** (`forward-serve`). The pack needs a kernel set for its geometry in the blocked family, and one key
in a copy of its `config.json`, `"q4nx_weight_layout": "b8"`; the container's bytes are untouched. Decode here is
this route's speed, a small multiple of one token per second on these sizes (engine#2102), not what the array
does on a geometry that has a sealed set: `Qwen3-1.7B-NPU2` is 3.0 tok/s here and 39.0 on the fused route.

| pack | family | answer | decode |
|---|---|---|---|
| `Qwen3-1.7B-NPU2` | `qwen3` | identical, `Paris.` | 3.0 tok/s |
| `Qwen3.5-2B-NPU2` (the `qwen3.5-flash:2b` tag) | `qwen3_5`, hybrid | identical, `Paris`; full row on engine#1785 | 2.7 to 3.4 tok/s |
| `Llama-3.2-1B-NPU2` | `llama` | identical, `Paris`, stops on `<|eot_id|>` | 4.2 to 4.3 tok/s |
| `Llama-3.2-3B-NPU2` | `llama` | identical, `Paris.` | 1.7 to 1.8 tok/s |
| `Phi4-mini-Instruct-NPU2` | `phi3` | identical, `Paris`, once the turn end is read from the template (below) | 1.4 to 1.5 tok/s |
| `Qwen3-VL-4B-Instruct-NPU2`, text tower only | `qwen3` | identical, `Paris` | 1.4 to 1.5 tok/s |
| `LFM2.5-1.2B-NPU2` | `lfm2`, hybrid: 10 conv layers of 16 | identical, `Paris`; against FastFlowLM's runtime below | 3.5 to 4.6 tok/s |
| `LFM2-1.2B-NPU2` | `lfm2`, hybrid: 10 conv layers of 16 | identical, `Paris`; against FastFlowLM's runtime below | 3.7 to 4.3 tok/s |
| `LFM2.5-1.2B-Thinking-NPU2` | `lfm2`, hybrid, thinking | identical; `Paris` after 287 reasoning tokens, no clean stop (below) | 4.3 tok/s |

`D3-Nano-NPU2` is a decision pack: it loads on the `Qwen3.5-2B` kernel set, its readout converts with
`tools/safetensors_to_q4nx.py decision`, and `/v1/systemone` returned three identical answers to the documented
example in about 1.1 s each. Its text routes answer 400 by design ("Decisions" below).

Serving these packs as published is what found six engine faults, each fixed with the pack as its evidence:
a turn-end marker named in `tokenizer_config.json` only (`Qwen3.5-2B`), or only in the template (`Phi4-mini`);
a stream that was sent after the generation had ended; reasoning that the template's own generation prompt opens
(`Qwen3.5-2B`); layout newlines returned around a reasoning block; and a 500 for a text request to a
decision-only pack. "What ends an answer" and [npu-chat-template.md](npu-chat-template.md) describe the rules
that came out of them.

**The `lfm2` packs: a hybrid, and a second kind of weight tile.** Ten of `LFM2.5-1.2B-NPU2`'s sixteen layers
have no attention: they are gated short convolutions (`model.layers.N.shortconv.*`). The forward runs them
(engine#1786): `in_proj` (hidden to three times hidden, the fused-projection kernel at `K2048_N6144`) and `out_proj`
(the attention output kernel) on the array, and between them the gate and a three-tap causal depthwise convolution
on the host, whose only state is the last two positions of one product per channel. Which layers are convolutions
and how long the kernel is are read from the container's own tensors, not from `full_attn_idxs` or `conv_L_cache`.
The kernel directory is the `Llama-3.2-1B` set's four shapes plus that one `QKV_K2048_N6144` pair.

Two things in these packs were read wrong before, and neither is about convolutions. Their 4-bit weights are a
**symmetric** kind of the 5,120-byte tile (FastFlowLM's list calls the pack `Q4_0` where the others are `Q4_1`):
the same byte layout with every zero-point stored as 0, a scale that carries a sign, and a two's-complement code.
Nothing in the container names the kind (the name is in FastFlowLM's model list, which is not part of a pack), so
the decoder reads it from the weights: a tile is symmetric when its 256 zero-points are all zero **and** some scale
is negative, a tensor is one kind (tiles that decide nothing follow their tensor), and a tensor holding both kinds
is refused. A zero offset alone decides nothing: an asymmetric tile may have one, and the engine's own converter
test writes such a tile. Of 1,889,236 tiles in the other packs on the box none has an all-zero zero-point block or
a negative scale; of 285,696 in the two LFM2 packs every one has both.
Read as the usual kind, each LFM2 projection correlated -0.39 with the same tensor of the upstream
GGUF, and the serve answered in fluent-looking fragments of unrelated words; read as symmetric, 0.99, and the
pack's quantised `lm_head` sits within one step of its own bf16 embedding (mean 0.24 of a step). And the
attention layers carry per-head `q_norm`/`k_norm`, which the `lfm2` model type did not switch on: with only the
tiles fixed, "What is the capital of France? Answer with one word." was answered `France`.

Measured on `LFM2.5-1.2B-NPU2` on 2026-11-11, same array, same `model.q4nx` (sha256 `b341d42021e617a5…`), five
greedy prompts of up to 40 tokens, FastFlowLM 1.0.12's runtime first and this route second in one ticket:
three answers byte-identical (`Paris`; the count from one to ten; a German sentence), two not. One leaves
FastFlowLM's text after 181 characters, the other at its first token (`Here's…` here, `Certainly!…` there). At
both points this route's choice is not a near-tie: the first token is `Here` -0.64 against `Certain` -1.26 on the
array and -0.66 against -1.23 with host float GEMMs on the same weights, and the upstream GGUF on the CPU opens
with `Here's` too. FastFlowLM's own `logprobs` for that token are `Certain` -0.81, `Here` -1.06, `S` -1.81: the
same candidates, the first two in the other order, on a 0.25 grid; its answer was the same three times running.
So the two runtimes agree on the model and differ in arithmetic, and which of them a reader should call right is
not something this row decides. Also on this pack: batched prefill equals token-by-token on every chunk checked
(`NPU_BATCH_CHECK=1`, chunks of 1 to 26 rows, all `MATCH`); a request answered from a restored prefix (351 of 352
prompt tokens cached, conv state included) returns the bytes it returns with the prefix cache off; and decode was
4.6 tok/s at load 3 to 4 and 3.5 to 3.7 at load 8 to 10, against FastFlowLM's 48 to 83 tok/s on the same replies,
the gap "Per-op route" above describes.

`LFM2-1.2B-NPU2` (sha256 `2755ab40cd927e3c…`) went through the same paired ticket on the same kernel directory,
at the merged head of that change: two of five answers byte-identical (`Paris`; the count from one to ten), three
that share 21, 58 and 53 characters with FastFlowLM's and then continue differently, each a fluent answer to the
question (a Fibonacci function that returns `0` for `n <= 0` where theirs returns a message). Decode 3.7 tok/s
against 42 to 47. Its logprobs at the three points were not taken. A later ticket at the merged code (main `1131acf0`) asked both
packs the known-answer prompt three times each: `Paris`, identical, with decode at 4.2 to 4.3 tok/s; those four
figures are records in `bench/npu-leaderboard/`.

`LFM2.5-1.2B-Thinking-NPU2` serves on the same directory too: its template is read from the pack's
`chat_template.jinja` and the reasoning is split out. Asked the known-answer prompt three times with 320 tokens
of room, it reasoned for 287 tokens and then answered, the three replies identical in reasoning and content, at
4.3 tok/s. The content begins `Paris` and does not end there: it goes on (`\boxed{Paris}`, then a restatement)
until the token limit, so the row has the answer and not a clean stop. At 96 tokens, in the paired ticket, every
reply was still reasoning, and the reasoning differed from FastFlowLM's from its first words.

Not established by any row here: that a pack goes through the importer (`npu-pack` still reports these
containers as recognised, not importable), anything about a vision or audio tower, and any pack whose geometry
has no kernel set on the box. Of the 57 `*-NPU2` repos with a readable geometry on 2026-11-11, 19 had all four
dense kernel shapes already built in the blocked family; the families the forward may not model at all
(`hunyuan_v1_dense`, the multimodal towers) are a separate question from the shapes.

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
- **Prompt:** the **container's own chat template** — `tokenizer_config.json`'s `chat_template` string, else
  `chat_template.jinja` beside it — rendered with the pinned llama.cpp fork's Jinja engine
  (`app/npu_template.h`, `docs/npu-chat-template.md`). A container that declares no template keeps ChatML
  when its tokenizer has the `<|im_start|>`/`<|im_end|>` anchors, and is refused on the chat route when it
  has neither (#963). The template's sha256 is printed at load.
- **Decoding:** greedy, with a 1.1 repetition penalty over the last 64 ids by default.
  `repetition_penalty` in the request overrides it, and 1 gives plain greedy.
- **One request at a time:** the model has one KV cache, so a second request waits for the first generation to end,
  and the two kinds of request wait differently (#1024):
  - a **streamed** request waits up to `--queue-ms` (default 300000, a full generation) and sends an empty content
    delta once a second while it queues, so a proxy's no-data timer does not read the queue as a hung backend
    (`app/npu_queue.h`). Its 200 and SSE headers are already on the wire, so a wait past `--queue-ms` is answered
    with an `data: {"error":…}` frame inside the stream (and `[DONE]`), not with a status;
  - a **non-streamed** request cannot be told anything while it waits, so it keeps a short bound of its own,
    `--queue-ms-plain` (default 5000), and past it answers `503 + Retry-After: 1` with `{"type":"device_busy"}`,
    which a client can act on. Until then its client sees nothing at all, so that bound is also how long it can
    appear to hang.
  A queued request occupies one HTTP worker for the whole wait (the server installs no task queue), so a long streamed
  queue is bounded by the worker pool — irrelevant for one user, but it is why `/health` must never take the device
  lock (it does not).
- **Repeatable:** every generation starts from a zeroed KV cache (#400), so identical greedy
  requests give identical text, whatever ran before them.

### The dense FFN tail cache (hybrid GDN models)

On a GDN model the guard that skips the per-token Q4NX dequant in `step()` can never fire
(`dense_weights_cached` requires `lin_kd_ == 0`, and `lin_kd_` is model-level), so a hybrid model
re-dequantised each layer's 17408-wide dense FFN tail on every token even after those weights were
packed on the device. `ffn_tail_cached()` (`npu/forward/src/q4nx_forward.cpp`) closes that: once a
layer's packed gate/up/down BOs are resident, the f32 `wg_`/`wu_`/`wd_` buffers are no longer read,
so the reload is skipped. It is on by default for a GDN model on the NPU route, when the packed
weight cache is on, the run is not `NPU_INFER_HOST_GEMM`, and the tail is at least 128 MB per layer
(the size guard, set just under the smallest tail measured; `npu/forward/include/ffn_cache_policy.h`). `NPU_FFN_CACHE=0`
or `=1` forces it either way, and the resolved state is logged once as
`[ffn] dense-tail cache on|off (default|NPU_FFN_CACHE; lin_kd=… IM=… H=…)`. On Qwen3.8-27B-NPU2 an
interleaved 5-round A/B (`~/lb/587-cache-20261030`) measured an 8-token request at 104.5 s against
117.4 s (**−11%**), CPU per token 22.9 s against 63.3 s (**−64%**), same text in every arm.
On FastFlowLM's Qwen3.5-2B-NPU2 (IM 6144, H 2048, a 151 MB tail that the earlier 256 MB floor left out) an A B B A
run in one ticket measured decode at 362–367 ms per token with the cache on (both arms) against 485–520 ms with
it off in the quieter of the two off arms and 478–707 ms in the other, same text in all twelve runs (engine#1785).

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

### The prefix cache's pinned slot: single-turn requests under one system prompt (engine#1785)

A slot holds a whole prompt, and a request hits a slot only when the slot is a prefix of it. Requests that
share a system prompt and differ in the user turn therefore reuse nothing from each other: `system + question 1`
is not a prefix of `system + question 2`. `tests/npu_sampling_test.cpp` (`test_prefix_pin`) states that as its
first row: a 900-token system prompt, a second question, `cached_tokens: 0`.

The slot table (`app/npu_prefix_cache.h`) has a slot kind for the part such requests do share:

| call | what it decides |
|---|---|
| `pin_point(prev, ids)` | how many leading tokens this request shares with the previous one. 0 when one prompt is a prefix of the other (a repeat or a continued conversation, which the ordinary slots already keep), when the shared part adds fewer than `--prefix-min-gain` tokens over a slot that already covers the request, or when the cache is off |
| `pin_at(ids, n)` | the same decision for a point the caller already knows: `n` leading tokens every such request shares. 0 when `n` is the whole prompt, when a slot already covers it, or when it is under `--prefix-min-gain` |
| `pin(prefix, bytes)` | keep the state at that point. One pinned slot at a time: a new pin replaces it; eviction for room and a longer prompt's replacement both leave it alone |
| `lookup(ids)` | unchanged, and reports `pinned` when the hit is the pinned slot |

The table itself reads token ids only, so it does not depend on the template. Where the pin goes is the caller's
choice between two rules, below.

**`--prefix-pin`** (on `1bit serve --device npu` and on `forward-serve`; off by default) turns it on. The
forward keeps the previous request's prompt ids, and when `pin_point()` is non-zero it stops the prefill one
token short of that point, takes a snapshot and pins it, the same cut it already makes before the last token.
The cost is one more launch on the request that makes the pin. Every decision is logged
(`[prefix] pin slot=1 tokens=342 state_at=341 pinned`), and `usage.flm_prefix_cache.pinned` says when a hit
came from the pinned slot.

Measured 2026-11-10 on `bonsai2-t2s2-perrole876` with its `kernels-m16`, one `npu-hold`
ticket, one server per arm, four requests: the same 14-rule system prompt, then three different questions and
the second one again. 16 tokens generated, temperature 0.

| request | cache off | default cache | `--prefix-pin` |
|---|---|---|---|
| 1 (question A) | cached 0, 58.6 s | cached 0 | cached 0, 59.5 s |
| 2 (question B) | cached 0, 59.4 s | cached 0 | cached 0, 62.1 s (makes the pin: 342 tokens) |
| 3 (question C) | cached 0, 61.5 s | cached 0 | **cached 341 of 355, pinned, 23.4 s** |
| 4 (question B again) | cached 0, 60.0 s | not reached | cached 354 of 355 (the whole-prompt slot), 20.4 s |

The generated text of all four requests is identical between the cache-off arm and the `--prefix-pin` arm.
The default-cache arm's times are not in the table: the ticket ended during its third request, so only its
three cache decisions (from the server log) are known. A request's time here is the whole HTTP call, prefill
plus 16 generated tokens, not TTFT.

**Pinned on the first request (engine#1785).** That run needed two requests to find the shared part, so the
third was the first warm one, and when two questions opened with the same words the pin ran a few tokens into the
question, which a question opening differently could not use. A chat request with a leading system message now
gives the point directly: the route renders the leading system messages followed by a probe user message (the
container's template, or ChatML) and pins where the full prompt stops agreeing with that, on the first request
(`pin_at`). The probe user message is there because a template may refuse to render a system turn alone (the
Qwen3.5 template does); if a template cannot render even that, the request has no early pin and the log says so
(`[prefix] no early pin: …`). Only a request with no
system message still uses the comparison with the previous request (`pin_point`). On FastFlowLM's
`Qwen3.5-2B-NPU2` container under `--flash`, a 1,016-token prompt, time to first token (two server loads):

| request | before | after |
|---|---|---|
| first question | 8.8 s | 10.6 / 11.1 s (busier box; makes the pin: 1,003 tokens) |
| **second question** | 9.1–9.5 s (made the pin: 1,009 tokens) | **1.39 / 1.38 s** (1,002 of 1,017 cached) |
| third question | 1.06 s | 1.40 / 1.35 s |

The "before" column was taken with two other changes that make decode about a third faster on this model, the
"after" column without them and at twice the box load, so compare the rows' shape (which request is the first warm
one), not the third row's two figures.

**`--flash`** (`1bit serve --device npu --flash`) is the single-turn profile as one word: `--ctx-size 1024`
plus `--prefix-pin`. It refuses what contradicts it (`--ctx-size` other than 1024, `--prefix-slots 0`, a device
other than `npu`) instead of quietly winning or losing.

The pin obeys `--prefix-min-gain` like every other slot (128 tokens by default): a system turn shorter than
that is not pinned, under `--flash` too, and each request prefills it again. That is the cache's policy, not
something `--flash` changes, so to pin a short system turn give the threshold with the profile, for example
`--flash --prefix-min-gain 32` for a 40-token one. A lower threshold also lets shorter whole prompts take a slot.

`--ctx-size N` now reaches this route. Before engine#1785's flash slice, `serve --ctx-size N` on a
model-generic container was accepted and dropped at the hand-off to `forward-serve`, which kept the model's
own `max_position_embeddings` (so the run in the table above was passed `--ctx-size 1024` and it had no effect;
its prompts were about 355 tokens, so nothing in the table depends on it). The limit is prompt plus answer,
it can lower the model's context and never raise it, and the load line says what is in force:
`1bit forward-serve: context 1024 tokens (prompt plus answer)`.

A request that does not fit is the client's error, so it is a 400 with the type the GGUF route's backend uses
for the same refusal (it was a 500 with a bare message):

```
HTTP 400 {"error": {"message": "request (1597 tokens) exceeds the available context size (1024 tokens)",
                    "type": "exceed_context_size_error", "n_prompt_tokens": 1597, "n_ctx": 1024}}
```

Measured 2026-11-10 with `--flash`, same pack and kernels, one `npu-hold` ticket, `max_tokens 8`: the
1597-token request above refused in 0.02 s; then three questions under one system prompt: request 1
`cached 0` (56.2 s), request 2 `cached 0` (48.9 s, makes the pin at 342 tokens), request 3
`cached 341 of 355, pinned` (13.9 s). Whole HTTP calls, one run.

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
| `max_tokens` / `max_completion_tokens` | the reply's cap (`max_completion_tokens` wins); a value that is not a number is a 400 naming the field, and a fractional one is truncated |
| `stream` | `true` streams SSE chunks; a value that is not a boolean is a 400 naming the field |
| `chat_template_kwargs` | the variables handed to the container's own chat template, as llama-server passes them: `{"enable_thinking": false}` makes the assistant turn start with the empty `<think>\n\n</think>\n\n` block Qwen3's own template emits, so the answer goes to `content` instead of into a `<think>` block (and its tokens get logprob entries; a non-boolean is a 400, and the default is `true`); FastFlowLM's packs take `user_system_prompt`, which defaults to `""` the way their own server does |
| `grammar` | a GBNF grammar in llama.cpp's format (`third_party/llama.cpp/grammars/*.gbnf`) the answer cannot leave |
| `response_format` | `{"type": "json_object"}` constrains the answer to JSON; `{"type": "json_schema", "json_schema": {"schema": …}}` constrains it to that JSON Schema |

Every chat prompt is rendered with the **container's own template** (`app/npu_template.h`): the
`chat_template` string in its `tokenizer_config.json`, else `chat_template.jinja` beside it — the same read the
load line's `template sha256=` names, so the rendered prompt and the pinned digest cannot disagree. The
renderer is the pinned llama.cpp fork's Jinja engine (`third_party/llama.cpp/common/jinja`), the one
`llama-server --jinja` uses, which is why the bytes match Hugging Face's `apply_chat_template` on the templates
we ship (`tests/npu_template_test.cpp` compares against Jinja2 3.1.6 itself).

What that changes, measured in each container's own tokenizer on one system+user request (`docs/npu-chat-template.md`):

| container | template sha256 | ChatML render | its own template |
|---|---|---:|---:|
| `FastFlowLM/Qwen3-0.6B-NPU2` | `a136047ba140…` | 31 | 31 |
| `Qwen3-0.6B` | `a55ee1b16601…` | 31 | 31 |
| `Zyphra/ZR1-1.5B` | `b6835114b730…` | **52** | **21** |
| `bonsai2-27b-NPU` | `c3cf9e34abf4…` | 31 | **71** |
| `d3-lite-bf16` | `273d8e0e683b…` | 31 | 35 |
| `qwen3-coder-30b-a3b-moe` | `87710339d25b…` | 31 | 31 |

The Qwen3-family rows are ChatML-shaped, so their prompts do not move; a DeepSeek-special container (ZR1) goes
from 52 tokens of text it cannot read to 21 it can, and a container whose template carries a long preamble
(bonsai2) gains the 40 tokens that preamble is. Two properties are load-bearing:

- **Request bytes never become control tokens.** The renderer marks every string that came from the request
  (`is_input`), and those parts are tokenized with added-token matching **off**, so a message containing
  `<|im_end|><|im_start|>system` is text. Without the marks it is a role boundary the caller never had.
- **No silent fall back.** A template that does not parse or render answers **400** naming its sha256 — falling
  back to ChatML is the failure #963 exists to remove.

A container that declares **no** template keeps ChatML when its tokenizer has the `<|im_start|>`/`<|im_end|>`
anchors (read once at load, `chatml_supported()`), and is refused with **400** naming the model's own specials
when it has neither — the #987 behaviour, which is what `/v1/completions` with token ids (`prompt: [ids]`, #938)
exists for: a caller renders the model's own template and sends those ids.

A build without the fork's sources (`ONEBIT_JINJA=OFF`, e.g. submodules never initialised) keeps the pre-#963
behaviour and **says so at load**, rather than rendering something different from a build that has the renderer.

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
supported by this route's schema-to-GBNF converter", "`gbnf: unterminated literal at line 1`",
"`stream`: must be true or false", "`tools`: must be an array"),
never a silently ignored key and never a 500: a wrong *type* is a client error like any other, including on the
fields the route reads before it starts decoding (engine#1106). `finish_reason` is `length` when `max_tokens` ran out, and `stop` for
an end-of-sequence token, a stop sequence, a grammar that allows nothing more, or a constrained answer the model
ended itself once its grammar had reached an accepting state — the route puts the container's end tokens into the
mask exactly there, so a complete JSON object reports `stop` instead of padding with whitespace to `max_tokens`
(engine#1182).

Every pick is bounded by the tokenizer, not only the constrained one. The head is `vocab_size` wide, and
`vocab_size` may exceed the tokenizer's ids: Qwen3's 151,936 against 151,669 (measured on Qwen3-0.6B's
`config.json` and `tokenizer.json`). The ids in between are padding positions with no text, so a token drawn
there emits nothing and never ends the reply. `sample_token()` and `1bit npu-run`'s per-row argmax
(`argmax_token()`) therefore consider only ids below the tokenizer's size (`token_bound()` in
`app/npu_sampling.h`, engine#2017), with or without a grammar. A container with no `tokenizer.json` keeps the
whole row.

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

A pack that declares `attention_mode: noncausal_full_attention` (below) is a decision model and nothing else:
it encodes a prompt in one pass and has no token-by-token decode. On such a pack `/v1/chat/completions` and
`/v1/completions` answer **400** with `type: decision_model_error` and a message that names `/v1/systemone`
(or says there is no decision head, when the serve loaded none), before any work is done.

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

**`attention.mode` comes from the pack's own declaration.** A d3 pack states the mask it was trained with in
its `decision_config.json` (`attention_mode: noncausal_full_attention`), and the forward honours that
declaration directly, so the two conversion steps above are enough: a freshly converted pack is answered the
way it was trained, with no plan step in between. Measured on the real pack, host GEMM, against its own
`d3_runtime.py`: with the declaration honoured the worst |Δp| is **0.0246**; a directory answered without it
is **0.3702** with a choice flip — the reference's own mask is worth 0.2869 (engine#847), so a causal answer to
a noncausal pack is a different model, not a smaller one.

`tools/npu_plan.py <model dir> --kernels <elf dir>` still writes `plan.json`, and it copies the same
declaration into `attention.mode`. A plan that **disagrees** with `decision_config.json` is refused by name as
stale, in either direction, because it would answer a noncausal pack causally (or a causal one
bidirectionally); the fix it names is to re-run the generator — or, for a deliberate measurement arm (an
engine causal-vs-noncausal row on a noncausal pack), to change the pack's `decision_config.json`, so the
container and the mask cannot drift apart. The plan is also what a *device* row needs for its `gate/up` fusion
and its M=16 decode twins, so a converted pack still wants one before it runs on the array.

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
prefix cache's job, #650). The pack's `noncausal_full_attention` mode *is* implemented (above, engine#861);
what is not measured is **accuracy against upstream**, which needs a labelled evaluation and is a separate
exercise (engine#796's third rung), so a d3 row measured here is not claimed to be upstream-equal in
accuracy.

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

### When a dispatch fails

A token's work is one runlist of the model's layer runs. When the NPU runtime reports that one of those runs did not
complete, `1bit unified` answers **the request that hit it** with an error frame and then exits non-zero — it does not
keep serving on that hardware context:

```
{"error":{"message":"npu runtime: invalid call (code 614): state 8, run 0 of 28"}}
```

The code's category is its hundreds digit (`6xx`: a call made in the wrong state or on the wrong object) and `run N of
M` is the index, inside the token's runlist, of the layer run that failed.

Exiting is deliberate: after such a failure the context's own state is dead, and every later request on the same process
fails the same way — measured on the c22 board, where the same backend answered 614 to every subsequent request while a
fresh process answered normally on the same device. The supervisor (Lemonade's backend watchdog, a systemd unit, the OS
menu's launcher) is the only one that can build a new context, so the engine hands it the problem instead of answering
with something that looks like a result. What the client sees next depends on the supervisor: Lemonade reloads the backend
and **replays a non-streamed request once** on the new context, so that turn still gets an answer (fork #42: 200 in
2.2-5.4 s where the same request used to be a 500, measured with a stand-in engine on strixhalo 2026-11-06); a **streamed**
request keeps the error frame, because bytes may already have reached the client, and the next request lands on the working
backend. Under any other supervisor (a systemd unit, the OS menu's launcher) a client sees one failed request and retries.

If a *fresh* backend also fails, the device is the thing to look at rather than the engine: the driver refuses new
hardware contexts until `amdxdna` is reloaded, and says so in its own kernel message — restarting backends cannot fix it.
`app/serve_watchdog.h`'s `npu_run_failed()` is the classifier for this (the runtime's failed-runlist message), and
`tests/serve_watchdog_test.cpp` (ctest `serve_watchdog`, in CI) covers it without a device.

### Two streams on one NPU

One process owns the NPU by default: it takes an exclusive `flock` on
`/tmp/1bit-npu-device.lock` for its lifetime (`app/npu_device_lock.h`), so a second serve of the
NPU route waits instead of contending. The policy stays the default because a second stream buys
much less than 2x -- and with the driver patch this tree carries (`0003`, running on strixhalo since
2026-11-05 06:26 as `349FDC99CADC56900E0249A`) it is also the only shape the device accepts, which is worth
stating precisely because the limit is *not* the hwctx count:

- **The array partition is the real limit, and it is per client.** `hwctx_limit = 16`
  (`drivers/accel/amdxdna/npu5_regs.c:74`; `npu4_regs.c:174` says the same) bounds how many hardware
  contexts **one** client may open;
  it says nothing about a second client. The solver hands each client a column range, and the NPU
  has 8 columns and the kernel set this engine ships asks for **all 8** (measured on the c24 set:
  `xrt-smi examine -r aie-partitions` shows one partition, `Columns [0..7]`), so one client's contexts
  share the whole array and a second client has nothing to be given.
- **What that cost before the fix.** With the unpatched driver a second client was handed the same
  columns instead, and its configuration took the array out from under the first: every client then
  failed `614 state 8, run 0 of 28` with `DPU PC 0xffffffff` in the firmware capture -- measured
  2026-11-05, 3/3 with the requests overlapping and 3/3 with them strictly sequential, and never with
  a single context ([engine#1084](https://github.com/1bit-MONSTER/engine/issues/1084)).
- **What it costs after it.** `third_party/patches/amdxdna/0003` shares a partition only between
  contexts of the *same* client, so the second client is refused with `code 102` / `-ENODEV` instead
  of taking the array down, and the partition is released when the holder exits. Measured on that
  build: the holder kept its 1 099-chunk row while the second client was refused, a fresh client
  answered once the holder exited, and one process held 2 contexts without trouble. Multi-context
  within one process -- the per-op route, the per-block-count route -- is unchanged, and the
  model-generic route that holds one context per design was re-measured on it with **8** contexts in
  one process (engine#1084).
- **The "long context" hang was this, too.** The 2026-11-04 wedge that
  [engine#1051](https://github.com/1bit-MONSTER/engine/issues/1051) first reported as a long-context
  hang (prefill position 635, the per-block-count route, `power/control=on`) ran while other serves
  held the device, on a module without `0003`. The per-block-count route destroys its context and
  creates a new one every 32 positions, so a long prompt goes back to the solver ~19 times per 1 200
  positions, and every trip was a chance to be handed a partition another client already held. On
  the `0003` module (`EE1F36A2B9BA294CE11AD80`, 2026-11-06, one client, the issue's own payload and
  kernel set, NPU held awake) both routes ran 300 / 640 / 1 200 / 2 150 positions with 0 TDRs, 8/8
  rows. Position is not the variable, and there is no position below which it stays safe: the two-client wedges above happened at the
  second chunk. One cost remains, and it is the per-block-count route's own. Between two block counts
  that route holds no context, so the partition is free, and a client that arrives in that gap takes
  it. The first client's next create is then refused (`code 102`) and its request fails part-way
  (measured, 1/1). It fails with an error, not a hang. The resident route (the default, one
  context for the process lifetime) keeps its partition and does not have this gap, and the engine's
  exclusive lock keeps a second serve of ours out of it anyway.

So `--npu-streams N` **needs a kernel set that asks for fewer columns than the array has**: with the
sets shipped today a second stream is refused rather than slow, and the throughput numbers below are
what two streams cost *when the column request allows them* (they were measured on the LFM2 `m16`
set, not on the 8-column sets this engine ships). The rest of this section is about that case, and the
lock policy above still governs it:

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
./1bit forward-serve -m <model.q4nx> --kernels <dir> --npu-streams 2 -p 8000
./1bit forward-serve -m <model.q4nx> --kernels <dir> --npu-streams 2 -p 8001
```

`--npu-streams` belongs to `forward-serve` (and to the GGUF-on-NPU model-generic forward `serve`
starts for it): N = 1 keeps the exclusive lock, N >= 2 takes a **shared** lock, so the opted-in
streams co-exist while any exclusive holder -- an unflagged serve, the in-process route, the
per-process ELF runner (`npu_engine_universal`), another build's serve -- still keeps them all out, in
either order, because they share one file and one `flock(2)`. The **in-process** route (a `.q4nx` model
directory through `serve`, i.e. `app/unified.cpp`) always takes the exclusive lock and refuses
`--npu-streams` by name: it keeps one context, so N streams there could only queue, and two of them
dispatching at once is the measured wedge (engine#1051).

**The lock is one file and one protocol, across runtimes, and it is advisory** -- that is what makes it
usable beside another NPU engine, and what its limits are:

- **Path and modes.** `/tmp/1bit-npu-device.lock`, created `0644` if absent, `flock(2)` (not `fcntl`)
  so it works across processes, users and runtimes on the box. `LOCK_SH` = "I am one of N streams, do
  not wait for other shared holders"; `LOCK_EX` = "I want the device to myself". A holder keeps its fd
  open for the life of the process; closing or exiting releases it. `LOCK_EX` and `LOCK_SH` never
  overlap, in either order, on one file.
### `NPU_INFER_HOST_GEMM=i8`: the integer arm, and why a float one cannot gate a kernel

`NPU_INFER_HOST_GEMM=1` runs the forward on the host with **f32** GEMMs. That is enough to tell a structural
break (garbage output) from a working route, and it is *not* enough to check a kernel family: the device's
weights are int8 codes with one multiplier per column and its accumulators are int32, so an f32 arm differs from
it by quantisation alone -- measured on the h4096 family, cosine 0.9960-0.9974 and an argmax flip at step 3,
none of which is a kernel fault. A family gated against f32 can therefore never be *bit-exact*, which is the bar
`TISSUES-ALL` §4 asks for.

`NPU_INFER_HOST_GEMM=i8` is the reference that can. It builds each body GEMM's weight image with the engine's
own packer (`q4nx_pack_B_int8_blocked()`: one multiplier per column, int8 codes, blocked microtile order), walks
it with `q4nx_layout::image_index()`, quantises the activation row to int8 with `amax/127`, accumulates in
int32 and applies `d[n] * (ascale * acc)`. **Same packer, same image, same order** -- that is the whole point: a
reference that re-quantises by a different rule computes a different model, and will disagree with a correct
device (measured: per-128-K int4 codes answer `ikt` to a question four other packs answer correctly on).

The LM head is included, in vocabulary blocks (a whole head image is H x V bytes, 622 MB at 150k vocabulary).
Without it the body is exact and the head is not, and the residual shows up as `max|d| ~ 0.3` on logits that no
amount of body-side agreement can remove.

The images are unpadded (K x N bytes), and the blocked order only exists for whole 64 x 128 tiles, so a geometry
that is not whole tiles is packed and read **row-major** instead: the same codes and scales, so the same int32
sums. The common case is the head's last vocabulary block (phi-3-mini: `32064 % 8192 = 7488`, which is 58.5
tiles). Before engine#1994 such a shape was forced blocked, so the packer wrote past the vector and the MAC read
back whatever was there. `tests/npu_blocked_layout_test.cpp` pins the rule with guard bytes behind the image.

**What it measures.** Run the device arm and the `=i8` arm on the same request with `--dump-logits`, then
`scripts/families/compare-logits.py` (npu-kernels). On `kernels-h4096-b8` with Mistral-7B-Instruct-v0.2 1.9:

```
step 0: argmax 3916 == 3916    cos 1.000000   max|d| 0
step 1: argmax  278 ==  278    cos 1.000000   max|d| 0
step 2: argmax 28725 == 28725  cos 1.000000   max|d| 0
step 3: argmax 22478 == 22478  cos 1.000000   max|d| 0
BIT-EXACT over 4 step(s) at max|d| tolerance 0.0
```
three repeats of the device arm, an edge case (`Hi`), and run-to-run determinism all `max|d| = 0`.

**How to run the two arms.** One ticket each, dumps compared per step:

```
1bit forward-serve -m <conv>/model.q4nx --kernels <conv>/npu -p 20081 --dump-logits /tmp/g/i8 \
    &                        # with NPU_INFER_HOST_GEMM=i8 in the environment
1bit forward-serve -m <conv>/model.q4nx --kernels <conv>/npu -p 20082 --dump-logits /tmp/g/dev
scripts/families/compare-logits.py /tmp/g/i8 /tmp/g/dev      # BIT-EXACT at tolerance 0
```
`--dump-logits` writes `step_<response>_<n>.bin` per generated step, so the comparison is per position rather than
per run. Repeat the device arm three times for determinism, and add an edge case (a two-token prompt), which is
what the table above did.

**What it costs.** The arm warms up slowly (it builds one image per tensor, ~35 s on a 7B container) and is not
a speed path: it exists to be a reference. It is off by default and `=1` (f32) keeps its old meaning.

**What it does not claim.** Nothing about throughput, and nothing about a family it has not been run on. The
same gate on the other four-h4096 sets needs a pack with *their* geometry, which is a download, not a code
change.

- **Who may take what.** Anything that honours the file interlocks with us: `forward-serve`
  (`--npu-streams` decides which mode), the in-process route (`LOCK_EX`, always -- `app/unified.cpp`),
  the ELF runner (`LOCK_EX`, always). **A run that never opens the device takes no lock**: with
  `NPU_INFER_HOST_GEMM=1` the forward runs the model with host GEMMs and makes no `xrt::device`, so it
  holds nothing the lock protects (`npu_lock_taken()`, engine#1353). That matters because the lock is
  exclusive for the process's life: a host-GEMM serve that took it would block every lane that *does*
  use the array -- measured on this box, one orphaned host-GEMM serve held it **4 h 52 m** with zero
  `/dev/accel*` handles and no CPU, and during that window a second lane's serve and two device tickets
  waited on it. `NPU_NO_DEVICE_LOCK=1` remains the operator's escape hatch for the runs that do open
  the device. A foreign NPU runtime --
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
pin, so nothing in this tree can select a column split today: both of our instances run whole-array --
which is exactly the request the solver cannot satisfy twice (see the limit above), so a 4-column
kernel set is the prerequisite for the numbers in this section, not a tuning option.
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

**When a live process holds it.** The lock is an `flock(2)`, so it dies with its holder and there is
nothing to reap -- but a *live* holder (a serve left running with no client, engine#1241) keeps it for
as long as it runs, and before #1241 the wait said only `waiting for the exclusive device lock` and
then nothing for half an hour. After about 2 s the wait now names the holder, and repeats that at most
once a minute:

    [npu] device lock held by pid 2689658 (1bit serve --device npu ...) for 41 s

The line comes from `/proc/locks` (the `FLOCK` entry on the lock file's `st_dev`/`st_ino`) plus the
holder's `/proc/<pid>/cmdline`; where a container hides either, the wait says it cannot name the
holder instead of failing. The same line is printed when the NPU refuses a context with `device error
(code 102): No such device`: a held array answers exactly like a dead one, so that message on its own
used to send operators looking at the device, the ELF or the container. `ONEBIT_NPU_LOCK_WAIT_MAX=<seconds>`
bounds the wait (unset = unlimited, still the default, because ordinary contention is what the wait is
for); on timeout it fails with the holder named rather than spending a device ticket. Nothing here
reaps or signals the holder -- killing another lane's live serve is the box-crashing path.

**The row this exists for.** On Linux the two engines are two instances of this runtime on two
*different* models at the same time -- "all NPU models at once" -- and that is what
`tools/showcase-612-npu-streams.sh` measures with `MODEL` and `MODEL_B` (default Qwen3-0.6B and
Qwen3-4B, whose kernel sets are on this box): each model solo, then both at once, three reps, every
number stamped with the build that produced it.

### Finding where a forward diverges: `NPU_INFER_DUMP_DIR` and `NPU_INFER_DUMP_OPS`

Two diagnostic switches of the NPU forward. Both are off by default, neither changes a result, and neither
can fail a run.

| variable | what it writes | answers |
|---|---|---|
| `NPU_INFER_DUMP_DIR=<dir>` | `<dir>/layer_<ll>.f32` per layer: the residual stream after that layer, M x H little-endian f32, row-major. The directory must exist; a file that cannot be written is reported on stderr and the run goes on. | which **layer** two runs part at (per-layer cosine against a reference trace) |
| `NPU_INFER_DUMP_OPS=1` | one stderr line per projection and per weighted RMSNorm: `[op] L<layer> <tag> rows=<M> \|x\| mean <m> absmax <a>` | which **op** inside the layer |

`NPU_INFER_DUMP_OPS` is on for any value other than empty or `0`, and is read once per process. When it is on, the forward prints
`[ops-dump] NPU_INFER_DUMP_OPS is on: …` once at model load, before any request (engine#2028). A run that sets the
variable and never sees that line is running a binary built before the dump existed (#1977): rebuild it, rather than
looking for a code path that skips the dump.

The files these dumps write (`NPU_INFER_DUMP_DIR`'s `layer_<ll>.f32`, `NPU_I8_GS_DUMP`'s weight-scale pairs) are
created owner-only (0600) and never through a symlink at the final path component, whatever the umask, and a dump file
an older build left at 0666 is narrowed to 0600 when it is rewritten; the head-image cache's files are at most 0644. Both go through `npu/forward/include/file_create.h`, so no write site uses `fopen(…, "w")`,
which creates with 0666 under the umask and follows a planted link (CodeQL `cpp/world-writable-file-creation`).
`tools/ci/file_create_check.py` refuses a new `fopen(…, "w")` or `"a"` anywhere in the tree unless the line says why
(`file-create-ok: <reason>`), and `tests/file_create_test.c` pins the modes.

```
  [op] L00 rmsnorm      rows=1 |x| mean 0.0143171 absmax 3.01935
  [op] L00 ffn_gate_up  rows=1 |x| mean 0.0258482 absmax 1.25728
  [op] L00 ffn_down     rows=1 |x| mean 0.00315921 absmax 0.673885
```

- The tag is the one the projection's call site already passes (`attn_qkv`, `attn_output`, `ffn_gate_up`,
  `ffn_down`, `ssm_out`, ...). Layer norms are `rmsnorm`; on a family with per-head q/k norms those print as
  `q_norm` and `k_norm`, one line per head.
- The numbers are over row 0 of the result (its first N values). A batched projection prints one line with
  `rows=M`; a norm is called per row, so a batched pass prints one norm line per row.
- A NaN or Inf is counted (`nonfinite 2/4096` at the end of the line) and left out of the mean and the maximum.
- The same lines come out on the host arms (`NPU_INFER_HOST_GEMM=1` or `=i8`) and on the device arm, in the same
  order and number, so the two logs of one prompt can be compared with `diff`. Start there: the first line whose
  numbers differ names the op.

### When a load follows an evict

On the appliance the NPU model is swapped by evicting it and starting a fresh `1bit` for the next
one, and a load that starts *right* after an evict can fail at device / context creation while the
previous backend's hardware context is still being torn down. Measured on the c22 board
([engine#1010](https://github.com/1bit-MONSTER/engine/issues/1010)): two children exited 1 inside
100 ms of starting, `POST /api/v1/load` answered 500, and the next request loaded the same model
normally.

That failure is transient by nature, so the NPU routes open with a **bounded retry**
(`app/npu_load_retry.h`): up to 2.5 s, 150 ms doubling to a 500 ms cap, at most 12 attempts. Each
attempt's reason goes to stderr — the line the appliance log never had — and a recovered start says
so:

```
1bit unified: opening the private route's NPU context attempt 1 failed after 12 ms
              (npu runtime: device error (code 102): No such device); retrying in 150 ms
1bit unified: opening the private route's NPU context recovered on attempt 2 after 168 ms
```

A real failure is unchanged apart from being reported: the last attempt's exception still reaches the
caller after the window closes, and the log now names the attempt count and the reason.

#### A refusal is a queue, not a fault

A second **client** on the NPU is refused with the same signature a dead array answers — `code 102` /
`-ENODEV` from `aie2_message.c`'s context-create path — so the code alone cannot say which one it is. Time
can: a client that holds the array frees it when it exits, a dead array never does. Since the per-client
partition fix ([engine#1085](https://github.com/1bit-MONSTER/engine/pull/1085), decision 2026-11-05)
makes that refusal the *normal* way a second client is turned away, a refused context is now **queued**
rather than failed: the same retry schedule runs against a load queue budget instead of the 2.5 s race
window, and the log says which of the two it is waiting for.

```
$ 1bit serve -m <model dir> --load-queue-ms 45000       # the default
1bit unified: opening the private route's NPU context attempt 3 failed after 410 ms
              (npu runtime: device error (code 102): No such device); retrying in 600 ms
              (queued: another NPU client may hold the array)
```

`--load-queue-ms N` bounds that wait (default **45000**, `0` refuses at once, i.e. the pre-#1085
behaviour). When the budget runs out the operator hint names both causes in the order to check them —
another client first (wait for it, or stop it), the driver second (`modprobe -r amdxdna && modprobe
amdxdna`) — because after a full budget the array is either genuinely busy elsewhere or genuinely dead.

This is the load-side counterpart of `--queue-ms`, which bounds how long a *request* waits for the
device lock; the two are separate because a request can be told it is queueing (an empty delta a second)
and a load cannot.

One NPU failure is not transient and says so: the driver deciding the array is dead, reported with
`No such device` (ENODEV) and either `code 102` (`errc::ctx_create`) or `code 104` (`errc::submit`) —
whichever call was refused, which depends only on whether the process already had a hardware context
when the array went. That is the state a TDR leaves behind when it could not restart a context, and the
driver keeps refusing until it is reloaded. The report adds the one action that works:

```
1bit unified: the NPU array is refusing hardware contexts: reload the driver (modprobe -r amdxdna && modprobe amdxdna)
```
`tests/npu_load_retry_test.cpp` (ctest `npu_load_retry`, in CI) covers the schedule, the window, a
transient recovery, a persistent failure, and a zero window (one attempt, no retry).

Lemonade's half of the same race is its fixed 2 s wait after "waiting for GPU driver cleanup", which
is a delay and not a readiness signal; that fix belongs to the appliance, this one keeps the engine
from dying on a failure that clears in a few hundred milliseconds.

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

**A full build links every executable; public CI links one.** `.github/workflows/ci.yml` builds `--target onebit` only, because
that is the target that needs no runtime. The gap that leaves is narrow but real: a member declared in
`npu/forward/include/q4nx_forward.h` and never defined in a `.cpp` compiles, passes `-fsyntax-only` (a missing definition is a
link-time fact, not a syntax one), keeps every check green — and fails `cmake --build` for anyone who does have a runtime. That
is how `main` went red for full builds while CI stayed green on the 3-argument `step_batch_embeds` (engine#1573). ctest
`npu_decl_check` (`tools/ci/npu_decl_check.py`) closes it: it pairs the header's direct members against every
`Q4nxNpuForward::<name>(` in the tree by **name and arity**, so a missing overload is refused by name instead of surfacing as a
92 %-through link error, and it needs no runtime, no device and no build. `npu_decl_check_self` drives the gate's own fixtures
(a missing overload, a definition in a second translation unit, a comma inside `<>`).

The NPU needs the `amdxdna` kernel driver. `third_party/xdna-driver` pins upstream
[amd/xdna-driver](https://github.com/amd/xdna-driver) as the driver version the engine is checked
against (the kernel's own copy also works); nothing from the pin is built into or shipped with
the release. `.github/workflows/bump-xdna.yml` moves the pin when run by hand, because a driver
move means rebuilding the module on Strix Halo and checking the NPU there first.

That pin is **pre-fix**, and the fixes are not upstream. They are carried as a series in
[`third_party/patches/amdxdna/`](../third_party/patches/amdxdna/README.md) — with the build, sign, install and
initramfs recipe, the `srcversion` each build must produce, and what a pin bump invalidates. The series is nine patches
and **every build of ours applies seven of them**, in this order: `0001` (the `bo_invalid_list` use-after-free at close,
#971), `0002` (array recovery after a TDR, #1005), `0004` + `0005` (two recovery outcomes that are not failures, and the
throttle check before any context is touched), `0003` (never share an array partition between two clients), and `0008` +
`0009` (#1051: a dying context must not fail the device resume, and no runtime suspend while a context exists — the
`code 614` hang). `0006` and `0007` (the imported-dma-buf execution pair, #812) are carried and **deliberately not
installed**: loading them is the owner's call and the README says so per patch.

A module built from the pin alone has none of it: it panics on that close path, it cannot bring the array back after a
TDR, two clients can be handed the same columns, and a request that follows a runtime suspend wedges the array.

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

### The draft loop, and whether to serve it

`npu/mtp/` holds the loop as library code so a serve can drive it, and `tools/mtp_acceptance` is how it is measured:

| file | what it is |
|---|---|
| `npu/mtp/draft.h` | the policy: draft `k`, verify as **one** `M = k+1` batch, commit the longest prefix the base agrees with, redraft |
| `npu/mtp/mtp_head.{h,cpp}` | the drafter: one head step per proposed token, chained on its own hidden |
| `npu/mtp/forward_base.h` | `BaseModel` over the real `Q4nxNpuForward`, with the deferred GDN commit |
| `tools/mtp_acceptance` | the measurements: `--offline`, `--live`, `--live --match`, `--batchcheck` |

`tools/mtp_acceptance --offline` scores the drafter alone on the base's own greedy stream, `--live` runs the loop,
`--live --match` scores both **at the loop's own chain positions** (so the two per-depth tables are one measurement
sampled two ways), and `--batchcheck` takes the verify batch's M-curve.

`NPU_BATCH_CHECK=1` is the oracle underneath that check. Every `step_batch` first replays its rows with sequential
`step()` calls and prints `[batchcheck] M=.. positions=a..b` with the argmax at the last row and a bitwise row count
(`identical=N differing=0`). Both passes continue the same prefix: the oracle snapshots the first `start_pos`
positions before the replay and restores them afterwards (engine#1905). So a chunk after the first, or the serve
path's split-off last token, is compared from the state the batched pass really continues, and the flag leaves the
run's own output unchanged. `tests/batch_check_state_test.cpp` pins both properties, and it runs in CI (engine#1946):
a configure without the NPU lane builds the same forward sources against the public runtime stub
(`onnx/ep/npu_runtime_ci_stub.cpp`, whose device calls throw; the host-GEMM forward makes none), and
`tests/batch_check_state_ci.py` runs the test on a two-layer qwen3 container it writes and packs itself. With the
NPU lane on, the row is the dev-box one and takes a real container from `ONEBIT_TEST_Q4NX`.

`tools/mtp_depth_pool.py` pools those tickets. It refuses a pool it cannot read (a repeated tail, a cell scored twice, a
short read, a mixed budget, a mixed window identity) and, since the tau=0.6 window, it also says when a depth is too
**thin** to carry the 2 pp bar: one chain moves a depth's rate by `100/n` pp, so under 50 chains a single accepted draft
is worth more than the whole bar. Those depths keep their PASS/FAIL (the bar is the brief's criterion) but print
`(thin: N live chain(s), one chain = X pp > the 2 pp bar)` and are counted out of the summary line, because a bare FAIL
on 2 chains reads as a drafter regression when it is arithmetic.

**Which drafts a sampler may keep** is a separate rule from how they are committed, and it lives in
`npu/mtp/spec_accept.h` (engine#1781, gate 2). The loop above accepts a draft when it equals the base model's argmax,
which is exact for greedy decode only. With a sampler the contract is that the output follows the **target's**
distribution, and the rule that guarantees it is: accept a drafted token `x` (drawn from the drafter's distribution `q`)
with probability `min(1, p(x)/q(x))`, and at the first rejection emit a token from `normalise(max(0, p - q))` and stop
the chain. `p` and `q` are the final sampling distributions (after temperature, top-k, top-p), and an argmax drafter
states a one-hot `q`. At temperature 0 both rows are one-hot and the rule reduces to the greedy contract, so that
contract is its degenerate case. `tests/spec_accept_test.cpp` (ctest `spec_accept`) pins both: the temperature-0
identity, and a deliberately wrong drafter (`p = [.45, .35, .20]`, `q = [.10, .80, .10]`) whose emitted tokens are
within 0.0016 of `p` over 200,000 draws at an acceptance rate of 0.55, where the greedy rule on the same rows misses
`p` by 0.55. The header is not called by a serve yet: wiring a speculative loop into `forward-serve` is the rest of
#1781, and that loop must print its acceptance rate beside its tok/s.

**The one thing that must be right**: the drafter attends over its own states, so its cache has to be the base
model's *committed stream, row by row*. Feeding it only the current chain is silent — drafts get worse, the text
stays correct — and it cost this project 69 % → 14 % first-draft acceptance until the cursor was made a checked
object (`DraftStream`; engine#820). A run reports `stream N rows` (must be prompt + emitted) and `N mismatch` (must
be 0) for exactly that reason.

**Measured on 2026-11-05, on the 27B `bonsai2-27b-pb16i` with the pb16i-exact head, `k = 4`, 64 tokens,
`tau = 0`:**

* live acceptance equals the offline protocol **at every depth** over 8 prompts × 64 tokens: depth 1 **+0.00**,
  depth 2 **+0.55**, depth 3 **+0.09**, depth 4 **+1.08** pp (75.14 / 63.24 / 73.26 / 84.13 %), with greedy text
  byte-identical to plain decode on all 8 prompts and 0 stream mismatches.
* the chain is **prompt-dependent**: 1.80 to 4.33 tokens per pass, mean **3.14** over the same 8 prompts. That is the
  unweighted mean of the eight per-prompt chains; pooled over the window's 181 chain iterations the same counts give
  **2.87** tokens per pass (the figure in `docs/npu-leaderboard.md`), because the prompts with short chains take most
  of the iterations. Both are the same measurement, weighted two ways — say which one a comparison is using.

**Do not enable the draft route on a serve as it stands.** The loop pays a verify batch *and* a step per
iteration, so it is faster only above a break-even chain, and that break-even is **3.2-4.1 tokens/pass** by two
independent measurements: dividing the measured wall clock by the measured pass count (3.87), and the split of the
8 prompts — faster on exactly the three above 4.12 (0.73-0.77×), slower on all five below 3.20 (1.08-1.74×, median
1.14×). This head's mean chain is 3.14, so the expectation over prompts is a loss. `1bit forward-serve` has no
draft route at all today (`app/forward_serve.cpp` runs one `step()` per token); `1bit serve --mtp` is the HRX/llama
path with the GGUF's own head, a different thing.

Two settings are worth knowing regardless, both measured in one window:

* **`OMP_WAIT_POLICY=passive` is free and worth 1.2-2.4× on every batched pass** (verify batch 495 ms against
  721-1187 ms; the `[bl]` core term 10.2 ms against 162-307 ms).
* **`tau` redistributes cost, it does not move it.** `tau = 0.6` (the head's published setting) stops the chain
  after the first draft and trades base passes for lm-head calls almost exactly one-for-one: 0.598-0.623 s/token
  across arms that differ 2× in chain and 4× in head calls. Measured over 8 prompts × 64 tokens it leaves **depth 1
  as the only depth with a usable denominator** (277 chains), depth 2 with 13 chains that reached it, depth 3 with 2,
  and depth 4 never offered at all — so a per-depth comparison at this `tau` is a depth-1 comparison.

**The default `tau = 0.6` is now measured, and it is not the regime to serve.** `Options::tau` defaults to 0.6, so
this is what a serve driving `mtp::run()` gets without asking, and until now no acceptance row existed at it
(every row above is `tau = 0`). Over the same 8 prompts × 64 tokens, `k = 4`, one prompt per `npu-hold` ticket, greedy
text byte-identical to plain decode on 8 of 8 prompts with 0 stream mismatches:

| | `tau = 0` | `tau = 0.6` (default) |
|---|---|---|
| hit rate (accepted / offered) | 46.69 % (338/724) | **78.77 % (230/292)** |
| depth 1 acceptance, all chains | 75.14 % (136/181) | **78.34 % (217/277)** |
| depth 2 acceptance, all chains | 47.51 % (86/181) | **3.97 % (11/277)** |
| depth 3 acceptance, all chains | 34.81 % (63/181) | **0.72 % (2/277)** |
| depth 4 | 29.28 % (53/181) | **never offered** |
| tokens per pass (live) | 2.87 | **1.83** |

The gate does exactly what it was asked to do — **the drafts it keeps are far better** (78.8 % against 46.7 % hit
rate, and depth 1 is identical to the offline protocol at the same positions, +0.00 pp) — and it does it by **almost
never chaining**: 292 drafts over 277 chains, i.e. ~1.05 per chain against 4.0, so the chain collapses from 2.87 to
**1.83 tokens/pass**. That is below the measured 3.2-4.1 break-even, so at the default the draft route is a
**slowdown**: **0.48×** re-measured at main `4480a6df` on the prompt-0 pair ticket (plain 0.482 s/token, draft
1.008 s/token; the same pair read 0.81× on 2026-11-06 — plain decode has since become ~1.8× faster while the loop's
two-passes-per-iteration cost has not moved, so the default's penalty doubled). The conditional
per-depth comparison at this `tau` reads +27 and +47 pp at depths 2-3, on 13 and 2 live chains — sampling noise, which
is why the table above is the all-chains view and why depth 1 is the only cell here worth comparing.

**`tau` is part of the window.** Because the gate stops the chain, a `tau = 0.6` ticket offers one to three depths where
a `tau = 0` one offers four, so the two must never be pooled into one table — and `mtp_acceptance --live` used to print
its window nowhere (`--offline` always stated it; the live path only echoed it in the harness' argv). `--live` now
prints `== N prompts x M tokens, k=K tau=T, hidden=H` at **init** (so a ticket killed at its hold still states its
window), and `tools/mtp_depth_pool.py` reads it, reports the gate in the window header, names a pool that mixes two
gates (`MIXED chain gate (tau)`), and says `tau unstated` for a log that predates the line rather than assuming 0.

**Refuted, so nobody builds it again**: folding the per-iteration step into the next verify batch (one pass per
iteration instead of two) collapses the chain from **2.82 to 1.07** tokens/pass and depth-1 acceptance from **82 %
to 6.9 %** — the head needs the hidden of the token it drafts past, which a one-pass loop cannot produce. Lagging
the hidden without lagging the cursor is worse: it accepts **0 of 92** while the mismatch counter stays at 0, i.e.
a silently broken loop rather than a slow one.

**What would change the answer**: a cheaper verify batch (the array wait is ~60 % of a batched pass — a kernel
lever) or a head that reliably produces a chain above 3.9 (widening `k` is not it: `k = 6` bought +0.05, and depth
6 measured 0 %).

**Two bars, and which one this loop has to clear.** engine#1371 measured a verify loop on the OGA 0.5B pack and derived
a break-even of `181/154 = **1.17 tokens per round**` — right for that loop, which runs **one** batched verify pass per
round. `mtp::run()` runs **two** base passes per round: a verify batch *and* the step that consumes the committed token so
the drafter's head sees the hidden it drafts past (measured, not assumed: `passes/token 0.625` is 40 passes over 20
rounds). The bar is therefore the round's cost in single-token units, and it is not one number — it depends on how many
drafts the gate lets a round carry. Measured in one process (`MTP_PHASE_SPLIT=1`, prompt 0, 64 tokens, `k = 4`):

| gate | round | plain token | **bar** | accepted | speedup | `step / batch / head / commit` |
|---|---|---|---|---|---|---|
| `tau = 0` | 2533 ms | 602 ms | **4.2 tokens/round** | 3.20 | **0.76×** | 29.9 / 38.9 / 17.5 / 13.7 % |
| `tau = 0.6` | 1559 ms | 619 ms | **2.5 tokens/round** | 1.97 | **0.79×** | 38.5 / 43.3 / 10.3 / 7.9 % |

Two things fall out of the split, and both matter more than the headline number:

* **the base passes are the round** — 69 % at `tau = 0` and 82 % at `tau = 0.6`; the drafter's own head work is 17.5 % and
  10.3 %. So the second pass (and the verify batch's own cost) is what a would-be optimiser has to attack, not the
  drafter: the hypothesis that the per-draft full-vocabulary call dominates is **refuted** by this measurement.
* **the deferred commit is not free**: 13.7 % at `tau = 0` (6.9 s of 50.7 s), i.e. ~0.35 s per round for the GDN replay,
  after the 16× cheaper replay of engine#1022.

Cross-checked with the two methods this page already used (3.2-4.1 by dividing the wall clock by the pass count, and by
the 8-prompt split): three readings of the same cost, all far above the 0.5B loop's 1.17, because two of the three passes
in a round are the loop's. Acceptance alone does not clear either bar: at `tau = 0` the drafter accepts **2.20 drafts per
round** (3.20 tokens), ~1.9x the 0.5B loop's bar, and still measures 0.76x. Quote a bar with its loop and its gate, or it
is 2-4x too optimistic for this one.

**Leaf fill is resolved, and the answer is that this verifier cannot use it.** The plan's last protocol item (#805) asks
for "spare verify rows filled with the drafts' 2nd/3rd candidates". The acceptance rule here is
`argmax(logits[i-1]) == draft[i]` (`npu/mtp/draft.h:317`): a draft is accepted iff it **is** the base's own argmax at
that position, and the argmax that grades depth *i* is produced by row *i-1* of the same batch. Two consequences,
both measured in `tests/mtp_draft_test.cpp` [16] against a toy base that is given a real two-level ranking:

* a candidate at the **same position** that is not the argmax is refused with certainty, however many rows carry it —
  the base's rank-2 token is accepted **0 of 92** times, and the control that makes that number mean something (the
  same head against the same base with the argmax moved onto the runner-up) accepts **20 of 20**;
* a **correct candidate deeper in the chain** is never scored while its parent is refused: a head that is right at
  depths 2..k and wrong at depth 1 accepts **0**, `reached_at[1] == 0`, and widening the batch to `k = 8` / `M = 16`
  rows changes it by **nothing** (0 accepted of 184 rows). The batch's rows are the *chain*, not candidate tests.

An alternative at a depth is graded by its parent's row, so it cannot even be placed in the same batch; placing it
would need per-row positions and an attention mask in the forward — a tree verifier — and with a recurrent GDN state
carried row by row that also needs a state per branch, which is the "no per-token state banks" constraint the same plan
imposes. So the spare rows buy nothing here, and the two levers that did move this loop are the ones already recorded:
the hidden variant (post-final-norm) and the drafter's stream cursor.

Reading these numbers has two traps that produced wrong figures here twice: plain decode is **host-bound**, so a
loaded window inflates it and flatters the draft route — two arms of one ticket are comparable, two tickets are
not; and "acceptance length" (`accepted/iterations`) is not "tokens per pass" (`1 + Σ hit[d]`). The tool prints
both, named.

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

## FAQ

### Does the NPU run every model?

No. It runs the dense Qwen3 family, including GGUF models it repacks to Q4NX on load (six
architectures), plus the models whose kernel sets ship with the engine. Every other GGUF still runs
on the iGPU through [HRX](hrx.md), and the models checked on this box are on [models.md](models.md).

### Is the NPU faster than the iGPU?

Not on the models it runs today: the route is correctness-first, speed next, and what it measures is
on the [NPU leaderboard](npu-leaderboard.md). It is a second device beside the iGPU, not a
replacement for it.

### How accurate is the NPU route?

Measured, not assumed: `tools/gsm8k_eval.py` posts the first N GSM8K test questions to any
OpenAI-compatible route with greedy decoding, reads the number the model writes after `####` (the
dataset's own answer format), and keeps a per-question transcript so a claimed percentage can be
re-graded offline without the device. Answers cut off by the token budget are counted **separately**
from wrong answers — a reasoning model stopped mid-thought is not a maths error — and the summary
prints both, so "95 % with 2 of 40 cut off" cannot be read as "95 %". The rows it produces are on the
[NPU leaderboard](npu-leaderboard.md).

### Can the GPU and the NPU run at the same time?

Yes. The two were measured running together for forty repetitions with no hang and no corruption,
and `--device hybrid` splits a single request across them ([hybrid.md](hybrid.md)).

### Do I need a special kernel or driver?

The NPU needs the `amdxdna` kernel driver. The [1bit OS](os-v3.md) appliance ships a kernel built
with it, and the [Linux kernel](kernel.md) page covers building your own.
