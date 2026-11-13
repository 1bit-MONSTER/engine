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
# AMD's drafter families: what they are, what they need, what they measure

AMD publishes two families of speculative-decoding artifacts on Hugging Face, and they are **not the same kind of thing**.
This page is the route decision for both, with the numbers this repo has measured. It exists because the HF model cards are
empty (Gumiho's is a licence header), and because the two families need different *verifiers* — which is the fact that
decides whether they can run here at all.

## PARD / PARD-2 — token-level draft models: **they run, and on the device they are break-even**

15 repos in the AMD catalogue (`amd/PARD-*`, `amd/PARD2-*`). They are **ordinary causal LMs** of standard architectures —
`Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `LlamaForCausalLM` (verified from each `config.json` **and** the safetensors tensor
names and shapes) — and what makes them "parallel" is the *inference mask*, not the architecture. So they convert to our
GGUF format with the stock converter and **no vendor code**:

```sh
tools/amd_draft_convert.sh ~/models/amd-rai/pard/PARD-Qwen3-0.6B /tmp/pard-qwen3-0.6b-f16.gguf f16
llama-quantize /tmp/pard-qwen3-0.6b-f16.gguf /tmp/pard-qwen3-0.6b-Q4_K_M.gguf Q4_K_M      # 462 MB
```

**The repo name is the DRAFT's size, not the target's.** PARD is *target-independent*: `PARD-Qwen3-0.6B` is a 752 M draft
that accelerates any Qwen3 target (pairing is by tokenizer/vocabulary, here 151936). A row that pairs it with
"Qwen3-0.6B" would be exactly backwards.

Measured here (`Qwen3-4B-Q4_K_M` target ← `PARD-Qwen3-0.6B` Q4_K_M draft, greedy). The warm-up and sample count differ per
row because the two arms were taken on different days with different settled states, so each row states its own:

| backend | samples per arm (warm-ups discarded) | target alone | target + draft | ratio |
|---|---|---|---|---|
| **HRX0** (iGPU, `-dev HRX0 -ngl 99 -fa on -devd HRX0`) | 4 (2) | 75.785 tok/s | 76.656 tok/s | **1.012×** |
| **CPU** (`-ngl 0 -t 4`) | 3 (1) | 20.53 tok/s | — | **0.508× / 0.609×** |

Acceptance is measured on both backends, by two different routes because the runtimes report differently. Both runs use the
**same prompt** (`Q: Explain in plain words why the sky appears blue during the day. A:`), so the ratio above and the acceptance
below describe one setup, not two:

* **CPU** — llama.cpp's own speculative path, 64-token budget: **0.759 / 0.804 / 0.980** of drafted tokens.
* **HRX** — that build echoes neither `draft_n_accepted` nor an acceptance-rate line, so it is measured with two servers on
  HRX0 and `tools/spec_decode_acceptance_hrx.sh` (which drives `tools/spec_decode_acceptance.py`: token-exact, the target's own
  greedy continuation as the verifier), **64 tokens in 21 rounds**:

```
offered 84 drafts (4.00 per round), accepted 44  -> 0.524 per offered draft
accepted drafts/round 2.095
tokens/round 3.095 untruncated = (44 + 21)/21   (3.048 if you divide the 64-token budget by 21)
per round: [4,0,4,4,1,2,4,4,4,0,0,0,4,0,0,4,0,4,0,1,4]   -- 8 of 21 rounds accepted nothing, the rest usually 4/4
```

**The two acceptance figures are not comparable and that clause matters**: the CPU numbers are llama.cpp's accepted/drafted
*behind `--draft-p-min`*, which stops drafting early, while the device run always offered a full k = 4. 0.524 against 0.759-0.980
does **not** mean the device accepts worse; it means the device probed four tokens every round and the CPU probed as few as its
gate allowed.

**And the cost sentence is a measurement now, not an inference.** The 1.012× ratio was taken in exactly this regime — 64 tokens,
same prompt — and the acceptance run gives **3.095 tokens returned per round**, so a round costs ≈3.06 target-token-equivalents
(3.095 / 1.012). That is the shape of the result: on this iGPU a 0.6 B draft over a 4 B target is a **wash**, and its acceptance is
bimodal (whole chain or nothing) rather than smoothly partial.

## Gumiho — a vendor hybrid head: **it needs a tree verifier we do not have**

7 repos (`amd/Gumiho-*`, pairs with llama2-7/13/70b, llama3-8/70b, vicuna-7/13b; 4.63-10.49 GB of `pytorch_model.bin`,
apache-2.0, **no `config.json`**). From the vendor repository (`AMD-AGI/Gumiho`, ICML 2025; merged into vLLM upstream): a
**hybrid head** — a serial Transformer for the early draft tokens, several lightweight **MLP heads in parallel** for the
later ones — plus **Full Tree Attention** for the verification.

The FTA is what matters here: it verifies **several candidate paths at once**, and this engine's draft loop is a *flat*
exact-argmax batch (`npu/mtp/draft.h`: row *m* is position `start_pos+m`, a draft is accepted iff it equals the target's
argmax at that position). Supplying several paths would need per-row positions plus an attention mask in the forward — and,
with the GDN state carried row by row, a state per branch, which `docs/npu.md` already refuses. The same wall the leaf-fill
item hit (#1305): **an exact-argmax flat verifier cannot use alternative candidates at all.**

So the Gumiho row is *"fails as published, because our verifier is the wrong shape"* — **not** a statement about the head's
quality (its published numbers are the best in its class: 2.97× mean over Eagle-2's 2.74× on vicuna-7B). Running it would
mean implementing a tree verifier first, which is its own project.

## DFlash draft packages on the NPU route: **the container is in forms we already read; the forward exists as a host reference, not yet on the device** (engine#1780)

FastFlowLM 1.0.12 ships speculative decode as two-package tags: a small DFlash draft beside the main model's package.
A draft package is `config.json` (`model_type: dflash`) plus one Q4NX container. `tools/dflash_draft_info.py <dir>`
checks such a package against its own config and prints what it declares; on
`Nemotron-3.5-Bolt-E4B-DFlash-NPU2` (0.27 GB, sha256 matching the repo's `MANIFEST.sha256`) it passes 19 of 19
checks in 3.4 s. What that settles, each with the evidence the tool prints:

| question | answer | evidence |
|---|---|---|
| are the `I8 [tiles, 5120]` projections a new packing? | **No.** They are the ordinary Q4NX q4_1 tile grid (32 x 256 weights per 5120-byte tile), the form every main-model container uses. | all 35 projections have `bytes == (N/32)(K/256) x 5120` for the N x K the config gives the role; 10,240,000 groups with every scale finite and positive; minimum/scale median -7.59 .. -7.33 per tensor (a known-good main model reads -7.41 the same way) |
| is `q_proj` 4096 x 2560 or a half-float 1280 x 2560? | **4096 x 2560**, so `head_dim` is 128 as the config says. | both readings are 6,553,600 bytes, but the half-float one has 25.7 % of its values above 100 in magnitude and 1.8 % non-finite, and 1280 x 2560 as a tile grid would be 400 tiles, not 1280 |
| are the `BF16` labels wrong (really F16)? | **No, they are bf16.** | a norm gain read as half-float looks tight (1.6-2.4) and as bf16 looks wide (0.22-6.3), which reads like a mislabel. The same pattern appears in a main-model container our forward serves correctly with the bf16 reading (1.49-1.88 against 0.12-1.05): half-float folds bf16's exponent range into one octave, so "tight" proves nothing |
| does the container agree with `config.json`? | **Yes.** | `dflash.block_size`, `target_layer_ids`, `sliding_window`, `rope_theta`, `rms_eps`, `embedding_scale` all equal |
| is anything in it unexplained? | **No.** | 64 tensors: 35 projections, 22 norms, `model.fc.weight` (2560 x 12800, the fusion of five target layers' features) and six scalars |

So importing a DFlash draft is a config-schema translation, not a format project. What our runtime does not have is the
draft's **forward**: fuse the target's hidden states at `target_layer_ids` through `fc`, run five non-causal layers
over a block of `block_size` positions filled with `mask_token_id`, and score the block with the target's own
embedding table and head. `forward-serve` stops on such a package at the first key it expects
(`key 'model' not found`), which is the schema, before any of that.

The tool is the gate before a loader, and it fails closed: a projection that is a tile short, a `head_dim` that
disagrees with the stored tiles, a NaN scale, a plain half-float matrix of the same size, a scalar that differs from
the config, or a tensor it cannot explain (the second draft family, `DFlash2DraftModel`, has selector and
convolution tensors this tool does not know) each fail by name. `--selftest` runs those controls on a synthetic
package (ctest `tool_selftest_dflash_draft_info`).

### The loader (`npu/dflash_draft.h`)

`onebit::npu::dflash::load_draft(dir)` is the C++ half of that gate: it reads the DFlash `config.json`, opens the
container with the engine's own Q4NX reader, checks every tensor against the geometry the config gives, and returns
a model whose tensors are views into the mapped file. Nothing is decoded at load (the Bolt draft's projections are
327,680,000 weights, 1.3 GB as f32); `decode_projection()`, `decode_gain()` and `decode_fusion()` produce one tensor
at a time, through the same q4_1 tile decoder the forward uses. It throws, naming the tensor or key, on a missing or
mis-sized tensor, a dtype label it does not expect, a container scalar that disagrees with the config, a tensor it
cannot explain, another architecture (`DFlash2DraftModel`), or a weight path that leaves the package.

Two facts about the container that the loader depends on, both measured on the Bolt draft:

| fact | evidence |
|---|---|
| `model.fc.weight` is **not** row-major. Its declared shape `[80, 12800, 32]` is the element order: 32-row blocks, column by column, the row inside the block last. | Magnitudes that share an input column correlate about five times more strongly than magnitudes that share an output row. Decoded in the declared order the four neighbour figures are consistent (next row 0.107, 32 rows on 0.106; next column 0.022, 32 columns on 0.021). Read as plain row-major they contradict each other (next row 0.019 but 32 rows on 0.107; next column 0.105 but 32 columns on 0.024). `dflash_draft_test` asserts the first and uses the second as its control. |
| A norm gain may be negative. | Three post-attention norms each hold one gain just below zero (-0.0009 to -0.0024) among 2,560. As bf16 that is a gain crossing zero; as half-float the same element would be -0.8 among values of 1.3 to 2.4, which is one more reason the bf16 reading is the right one. |

`dflash_draft_test` builds a synthetic package (every projection encoded tile by tile from a formula and compared at
every element, the fusion matrix checked for row-major order, sixteen refusals each matched on its message) and needs
no model; given a package directory (`-DONEBIT_DFLASH_DRAFT=<dir>` or `ONEBIT_TEST_DFLASH_DRAFT`) it also loads and
decodes the real one. The loader is not wired into `forward-serve` yet.

### The forward, as a host reference (`npu/dflash_forward.h`)

`decode_draft(load_draft(dir))` turns a package into f32 weights (about 1.3 GB for the Bolt draft) and
`draft_forward(weights, taps, ctx, noise, block)` runs one draft pass on the host:

| step | what is computed |
|---|---|
| context | one row per context position: the target's hidden states at `target_layer_ids`, concatenated, through `fc` and then `hidden_norm`. The same rows are used by every layer. |
| block | the target's embeddings of `[last token, mask, mask, ...]`, `block_size` rows. Only these rows change from layer to layer. |
| each layer | `input_norm` on the block rows; queries from the block rows; keys and values from the context rows followed by the block rows; `q_norm` / `k_norm` per head; rotary positions (context first, then the block, consecutive); every block row sees every key, so the block is not causal; then the gated MLP. |
| output | the final `norm` of the block rows. The target's own head turns each row into scores. |

The draft has no embedding table and no head of its own: both are the target's.

This is a reference to compare a device route against, not a fast path (plain loops, one thread: a pass of 2 context
and 8 block rows over the Bolt draft takes seconds). **It has not been run against a target model's hidden states
yet**, so nothing here says the draft's proposals are accepted; that needs the target pack's layer taps, which is the
next step of engine#1780. What `dflash_forward_test` establishes:

- two cases worked out by hand in the test (one block row alone; one context row plus one block row, with a different
  gain on every norm that touches the attention), matched to 1e-5;
- the structure, on a small random model: the output does not depend on where the positions start, a later block row
  changes an earlier one, the context's scale does not matter, a window of one position equals each row on its own,
  and a query head reads the key/value head of its group;
- seven refusals, each matched on its message;
- with a package directory (`-DONEBIT_DFLASH_DRAFT=<dir>` or `ONEBIT_TEST_DFLASH_DRAFT`), the real draft decodes and
  one pass over a made-up context is finite and position-independent.

### DFlash 2 packages (engine#1781)

`load_draft()` also reads `DFlash2DraftModel` packages (`FastFlowLM/Qwen3.8-27B-DFlash2-NPU2` is the one on this box),
as generation 2 of the same loader. The config states the draft's interface under `dflash_config` and the rotation's base
under `rope_parameters`. Every projection, the fusion matrix included, is a grid of 4736-byte Q4_K chunks rather than
5120-byte q4_1 tiles. Each layer also carries two grouped convolutions, around the attention and around the MLP, and the
package carries a candidate selector (a projection of the hidden state and two vocabulary-sized codebooks).
`decode_draft()` fills the convolutions; the codebooks are read a row at a time with `decode_row()`, since each is
254 MB as f32.

The forward gains what generation 2 needs, each piece held to arithmetic written out in `dflash_forward_test`:

| piece | what it computes |
|---|---|
| `draft_conv()` | one half of the grouped convolution: a row's output sums, over the kernel's offsets, (base kernel + the row's own kernel for its group) times the row that many places earlier |
| `selector_pick()` | the selector's choice at one block row: among the target head's top candidates, the highest of logit + the pairwise term between the previous pick and the candidate |
| `DraftKvCache`, `cache_extend()`, `draft_forward_cached()` | the context's keys and values per layer, computed once per context row and reused by every later pass; grown in steps it gives exactly the single-call pass |
| `set_gemm_hook()` | an optional place for the draft's projections to run (the device route); unset, everything runs on the host |

Measured with these pieces against the served 27B, in a local loop: on a code prompt, 6.86 tokens per round and 7.6-7.8
tok/s against 4.2 tok/s plain decoding; output identical to plain greedy decoding. The loop is not part of this change;
the numbers are on engine#1781.

### The speculative loop on the served 27B (`tools/dflash2_probe loop`)

`dflash2_probe loop` runs the pair end to end. The target is `bonsai2-27b-t2s2` on the NPU. The draft's projections can
run on the NPU through `set_gemm_hook()`; its attention, convolutions and selector stay on the host. Each round, the
draft proposes 7 tokens, the target verifies the 8 rows in one `step_batch_all()` pass, and `commit_verify_rows()` keeps
the accepted prefix. When every row is accepted, the commit keeps the batch's own recurrent state instead of restoring
the snapshot and replaying the rows: `NPU_REPLAY_CHECK=1` holds the two equal, and `NPU_COMMIT_FULL_REPLAY=1` keeps the
old path. Measured on engine#1781, with output identical to plain greedy decoding:

| prompt | plain decode | the loop | tokens per round |
|---|---|---|---|
| a Python function | 4.24-4.27 tok/s | 7.5-7.8 tok/s | 6.86 |
| prose (with the fallback to plain decoding) | 4.17-4.21 | 4.08-4.12 | 2.82-3.0 |

A loop has to be compared with the serve's own batched prefill. The per-token prefill is not bit-identical on this route,
and against it the loop's output parts after 23 tokens.

## PARD-2's repos are weights-only — borrow the target's tokenizer

`amd/PARD2-*` ship **`README.md`, `config.json`, `model.safetensors`, `warp_model.bin`** and nothing else: **no tokenizer
files at all**. A converter pointed at such a repo dies with `FileNotFoundError: …/tokenizer.model` and then
`NotImplementedError: BPE pre-tokenizer was not recognized`, which reads exactly like a broken converter and is not.

The draft must **borrow its target's tokenizer**, and the identity check that makes that safe is cheap:
`config.json: vocab_size 151936` — the Qwen3 family's — so any Qwen3 tokenizer pairs with it. With the repo's own weights
untouched and a Qwen3 tokenizer dropped beside them, the stock converter succeeds first try:

```sh
mkdir /tmp/pard2-in && cp <repo>/{config.json,model.safetensors,warp_model.bin} /tmp/pard2-in/
cp ~/models/npu/Qwen3-0.6B/{tokenizer.json,tokenizer_config.json} /tmp/pard2-in/     # same vocab: 151936
python3 convert_hf_to_gguf.py /tmp/pard2-in --outfile /tmp/pard2-f16.gguf --outtype f16
```

Two further facts from the same measurement: **`warp_model.bin` (33.6 MB for the Qwen3-8B repo, 41.9 MB for Qwen3-14B, 70 MB
for Llama-3.1-8B) is not in the GGUF** — it is PARD-2's extra
module, so a converted artifact gives AR-mode drafting only, and a parallel-mode implementation needs that file; and the same
1024-hidden / 28-layer draft serves both the 8B and 14B targets, which is what "target-aligned but target-independent" looks
like in practice. Measured with it: `PARD2-Qwen3-8B` → `Qwen3-4B` gives **2.18 tokens/round** (1.36 accepted drafts/round,
per-round `[0,2,1,3,0,2,3,0,0,0,4]`) — *below* PARD's 2.67 on its own family, correctly, because PARD-2 is aligned to 8B
acceptance and a 4B target is off its pair.

## A server must never outlive the arm that started it

`tools/spec_decode_speedup.sh` starts a `llama-server` per arm and kills it at the end — but an interrupt (Ctrl-C, a ticket
timeout, a `timeout` wrapper firing) used to leave the server running. **On 2026-11-08 an orphaned 14 h `llama-server` held
51 GiB of GTT and knocked the PR-agent LLM off the iGPU**, which is a fleet incident, not a leak. The script now traps
`EXIT INT TERM HUP` and kills the child, so the arm's server dies with the arm; if you drive a measurement any other way, put
the server inside the same `gpu-hold`/`npu-hold` ticket as the measurement and let the ticket end it.

## Traps, each of which cost a measurement on this box

* **`llama-cli -p … -n N -no-cnv` still enters its interactive loop** and spins `>` prompts forever; two unattended runs died
  at 150-400 s timeouts that way. Use `llama-server` + `/completion` (what both tools here do) or `llama-bench`.
* **The HRX llama.cpp binary is not the one on `PATH`**: `~/wt/i1251/pincur/llama-server` with
  `LD_LIBRARY_PATH=~/wt/i1251/pincur:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib`; other builds report
  `Available devices: (none)` and reject `-dev HRX0`. Read it off a live process (`/proc/<pid>/{exe,environ}`) instead of
  guessing.
* **`--draft-max` is gone** on the pinned build (*"the argument has been removed. use --spec-draft-n-max"*); `-md` survives,
  `--spec-draft-n-min` replaces `--draft-min`. Hence the flag knobs in the harness.
* **Absolute tok/s from two different runs are not comparable** — the same target measured 7.7 and 18.9 tok/s an hour apart.
  Only a within-run ratio is a measurement, and the first request of an arm is warmup (it can be 10-14× slower than the
  second on a loaded box).
