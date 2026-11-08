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

Acceptance (llama.cpp's own accounting, **CPU** runs): **0.759 / 0.804 / 0.980** of drafted tokens. On HRX it is **not
measured** — that build does not echo `draft_n_accepted` in the JSON and does not write the `draft acceptance rate` line by
default, which is why the ratio above stands alone. Reproduce the HRX row exactly (the default is now `SKIP_FIRST=2` because one warm-up produced the bad number below):

```sh
BIN=~/wt/i1251/pincur/llama-server \
SRV_ARGS='-dev HRX0 -ngl 99 -fa on -devd HRX0' \
DRAFT_MAX_FLAG=--spec-draft-n-max DRAFT_MIN_FLAG=--spec-draft-n-min SKIP_FIRST=2 REQS=4 \
  tools/spec_decode_speedup.sh ~/models/Qwen3-4B-Q4_K_M.gguf /tmp/pard-qwen3-0.6b-Q4_K_M.gguf 64 4
```

and the acceptance runs with `tools/spec_decode_acceptance.py`.

**A 1.50× was published here first and it was wrong.** The first HRX run discarded a single warm-up request, and its baseline was
**unsettled** across all three measured samples (40.646 → 51.942 → 46.849 tok/s — it rose and then fell, so not a ramp but not a
plateau either) while the draft arm was measured after it. The claimed gain was +0.50 of ratio and the settled gain is +0.012, so
**~98 % of the published "speedup" was that unsteady baseline** — "50 points of ratio", not "50 % of the gain". Two later runs with two discarded warm-ups and four measured samples each put the
baseline on its plateau (~75 tok/s) and give **1.03×** and **1.012×** — i.e. the draft is roughly break-even on the device for this
pair, not a large win. That is why the harness now takes `SKIP_FIRST` and why the tool prints both arms' every sample.

**CPU can never win, and that is arithmetic rather than a bug.** A round is one batched verify over `M = 1 + drafted` tokens
plus `drafted` draft decodes; on a CPU backend a decode is bandwidth-bound per token, so the verify costs ≈ `M`
token-equivalents, while the round can only return `1 + accepted ≤ 1 + drafted = M`. Break-even needs `accepted > drafted`,
which is impossible. Measured by counting work rather than timing it (load-independent): `llamacpp:n_decode_total` = 11 for
10 rounds ⇒ **exactly one target decode per round** (the verify *is* batched) and zero rollback replays, and the observed
0.68-0.79× matches `tokens/round ÷ (drafted·r + M)` with the measured per-token costs. engine#1423 carries that analysis, and
its only open item is a policy question (warn or refuse when speculation is requested on a CPU-only backend).

### What the corrected number means

At ~75 tok/s the 4 B target is fast enough on this iGPU that a 0.6 B draft — issuing **up to 4 (k = 4) draft decodes per round**,
the upper bound since acceptance is unmeasured on HRX — buys back about what it costs: **1.01×, not 1.50×**. So the device verdict is "break-even for this pair", not "a win" — and PARD's own headline numbers come from its
tuned runtime (parallel mask, batching), not from driving its weights through our server in AR mode: the model card for
`amd/PARD-Qwen3-0.6B` claims "an average inference speedup of 1.78×" over pure-AR drafts and "up to a 4.08× speedup" in
"Transformers+ PARD" — different drafters and a different runtime from the pair measured above, which is why they are quoted as
the vendor's claim and not as our result.
A row that quotes a speedup for these drafts must say which of the two it measured.

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
