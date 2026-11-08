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

## PARD / PARD-2 — token-level draft models: **they run, and they pay on the device**

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

Measured here (`Qwen3-4B-Q4_K_M` target ← `PARD-Qwen3-0.6B` Q4_K_M draft, greedy, the first request of each arm discarded):

| backend | target alone | target + draft | ratio |
|---|---|---|---|
| **HRX0** (iGPU, `-dev HRX0 -ngl 99 -fa on -devd HRX0`) | 46.849 tok/s (median of 3) | 70.259 tok/s | **1.50×** |
| **CPU** (`-ngl 0 -t 4`) | 20.53 tok/s | — | **0.508× / 0.609×** |

Acceptance (llama.cpp's own accounting, CPU runs): **0.759 / 0.804 / 0.980** of drafted tokens. Reproduce with
`tools/spec_decode_speedup.sh` and `tools/spec_decode_acceptance.py`.

**CPU can never win, and that is arithmetic rather than a bug.** A round is one batched verify over `M = 1 + drafted` tokens
plus `drafted` draft decodes; on a CPU backend a decode is bandwidth-bound per token, so the verify costs ≈ `M`
token-equivalents, while the round can only return `1 + accepted ≤ 1 + drafted = M`. Break-even needs `accepted > drafted`,
which is impossible. Measured by counting work rather than timing it (load-independent): `llamacpp:n_decode_total` = 11 for
10 rounds ⇒ **exactly one target decode per round** (the verify *is* batched) and zero rollback replays, and the observed
0.68-0.79× matches `tokens/round ÷ (drafted·r + M)` with the measured per-token costs. engine#1423 carries that analysis, and
its only open item is a policy question (warn or refuse when speculation is requested on a CPU-only backend).

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
