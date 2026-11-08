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
