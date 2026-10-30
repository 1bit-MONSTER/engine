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

The NPU engine is in progress: correctness first, speed next. Its kernels, built from source,
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
  npu/                   the model's NPU kernels
```

`1bit serve -m <model dir>` serves such a directory on the NPU behind the OpenAI-compatible API
([serve.md](serve.md)); inside Lemonade, that is what its `onebit` backend runs (`--onebit npu`).

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

**Q8 has two code layouts under the same chunk size.** Qwen3.5-4B's lm_head (tied to the
embedding) stores code (row r, column k) at byte `k*32 + r`. Qwen3.6-35B-A3B's Q8 tensors
(untied lm_head, projections) store it at `(r/16)*4096 + (k/32)*512 + (k%32)*16 + r%16`.
Both containers report `flm_version` 1.0.3, and neither carries metadata that tells them
apart. Decoding tile (0, 0) of each lm_head and comparing with a GGUF of the same model
gives correlation 0.9998 with the right layout and about 0.01 with the other. `npu/q4nx`
decodes the first layout.

`npu/q4nx.h` decodes all three, and repacks Q4_K into q4_1 (exact apart from rounding
`S * scale` and `M * min` to bf16), so code that reads q4_1 runs Q4_K weights unchanged. Q8 does
not fit q4_1. `tests/npu_q4nx_test.cpp` checks the decoders bit for bit against reference decoders
on real Qwen3.5-4B and Qwen3-0.6B tiles (committed in `tests/golden/q4nx`), and the repack against
its rounding bound.

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
0.079. Served on the NPU, the repacked model decoded 24/24 greedy tokens the same as a CPU
reference of the repacked weights, with every step's logits at cosine 0.99995 or better.

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
| `stop` | up to 4 strings: generation stops before the first match, and the tail is held back so a stop split across two tokens still matches |
| `grammar` | a GBNF grammar in llama.cpp's format (`third_party/llama.cpp/grammars/*.gbnf`) the answer cannot leave |
| `response_format` | `{"type": "json_object"}` constrains the answer to JSON; `{"type": "json_schema"}` is a 400 |

A field the route cannot honour is a 400 whose `error.message` says why ("`json_schema` needs a
schema-to-GBNF converter this route does not have", "`gbnf: unterminated literal at line 1`"),
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
`tests/npu_sampling_test.cpp` runs the sampler, the parser, the JSON grammar and the request
parser in CI without a model or an NPU.

```
$ curl -sS localhost:8000/v1/completions -d '{"prompt": "The capital of France is", "max_tokens": 16,
    "grammar": "root ::= \" Paris\" | \" Lyon\" | \" Marseille\""}'
{"choices":[{"finish_reason":"length","index":0,"text":" Paris"},...]}
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
