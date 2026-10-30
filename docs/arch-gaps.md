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
# Architecture gaps: significant arrivals the registry does not map

The registry ([registry.md](registry.md)) maps an HF architecture only when a pinned backend's own code accepts it. It is generated from the pinned sources; nothing in it is typed in by hand. So "support" cannot be faked with an alias, and the census counts every other architecture as unmapped.

**Alias or real work?** Some unmapped arrivals are renamed or same-shape siblings of a supported family, and are fixed where the backend is (upstream, or our fork) by mapping the name. Others only look like a supported family. For those, mapping the name would advertise support that cannot be served, or silently compute something else.

This page is the reviewed evidence for the second kind: what each family is, what it adds over the closest supported architecture, and what real support requires. It exists so that:
- nobody later "fixes" one of them with an alias;
- whoever implements one starts from these measurements.

`registry/significant.json` lists the reviewed classes. `tools/registry_build.py --check-gaps` (ctest `registry_gaps`) fails if one of them becomes mapped without a recorded reason (`mapped_ok` must be a non-empty string; a missing, empty or vacuous value is still a violation), and the build prints the same warning. The census and pin-bump workflows run `--check-gaps` right after regenerating the registry, so a mapping lands nowhere without failing. ctest `registry_gaps_fire` (`tests/registry_gaps_test.py`) pins the guard's firing behavior against synthetic registries — `registry_gaps` on its own only ever sees a clean registry, so that test is what catches a regression that makes the guard stop firing. A pin bump that brings real upstream support then gets a person's look before the class counts as supported.

**Provenance.** The review was written for 1bit-MONSTER's own engine, before this repository (the 1bit-MONSTER census, 2026-09-12/13), and is kept as written below. The `RCPP_ARCH_*` names, `rcpp_arch_from_string`, `registry/arch.h`, `src/*.cpp` and `census/*` it cites are that engine's, not this one's. Where it says "the engine", read the old engine: `englishbase`, for example, was implemented there, not here. The configs, checkpoints and reasons are the models' own, so the conclusions hold here too: none of these is an alias of a family this engine's backends run.

---

## `deepseekv41` — DeepSeek V4.1

* **Class / model_type**: `DeepseekV41ForCausalLM` / `deepseek_v41` (text tower
  `deepseek_v41_text`); example `deepseek-ai/DeepSeek-V4.1-Flash`,
  `Solstice-AI/DeepSeek-V4.1-Flash-UNCENSORED-FP8`.
* **Shape**: **40 layers, H=5120, 64 heads, 1 kv head, 384 experts, vocab
  129280** — versus V4 Flash's 43 layers / H=4096 / 256 experts. It is a
  different model, not a revision label.
* **Keys V4 does not have**: `engram_*` (8: `engram_layer_ids`,
  `engram_num_embeddings`, `engram_vocab_size`, `engram_compressed_vocab_size`,
  `engram_max_ngram_size`, `engram_n_heads`, `engram_head_dim`,
  `engram_pad_token_id`), `candidate_block_size`, `candidate_source_layer_id`,
  `candidate_topk_blocks`, `index_source_layer_ids`, `kv_source_layer_ids`.
  (`dspark_*` is *not* V4.1-exclusive — `DeepSeek-V4-Flash-DSpark` carries it
  under `model_type: deepseek_v4`.) It is also a multimodal wrapper
  (`vision_config: deepseek_v41_vision`, `image_token_id`).
* **Why an alias is wrong**: `src/deepseek_v4.cpp` reads only
  hidden/layer/head/expert/`hc_*` scalars, so a V4.1 checkpoint would load
  through the V4 path and produce tokens while **silently ignoring the engram
  and candidate-block machinery** — the silent-substitution failure this repo
  bans, not a crash.
* **Why it is not mapped** (rechecked 2026-10-23 on the pin `main` carries,
  `third_party/llama.cpp` af04c9f2, and again 2026-10-25 at `6b397ee9`): the fork
  registers no `DeepseekV41ForCausalLM` converter class — `conversion/deepseek.py`
  has `DeepseekV4ForCausalLM` only — and `src/llama-arch.cpp` has
  `LLM_ARCH_DEEPSEEK4` with the `hc_*` tensors but no `deepseek_v41`/`deepseek41`
  architecture and no `engram_*` / `candidate_block_*` parameter. `conversion/`
  reads neither key family at all. So `tools/registry_build.py` finds nothing to
  read the class from, and a mapping would load a V4.1 checkpoint through the V4
  path and silently ignore the machinery above.
* **Watcher** (engine #522): `registry/significant.json` carries
  `"watch": "report"` for this class, so the census prints it as
  `REVIEWED` and does not alert on every new upload of the family. That is a
  statement about the *review*, not about support: the class stays out of
  `registry/architectures.json`, `tools/registry_build.py --check-gaps` still
  fails if anyone maps it without a `mapped_ok` reason, and real support (engram
  embedding path + candidate-block sparse attention, plus a scope decision for
  the vision tower) needs the logits validation listed above.
* **Real support needs**: shape-agnostic config + the engram embedding path +
  candidate-block sparse attention (and a scope decision for the vision tower),
  validated against reference logits from the HF implementation.

## `qwen3mamba3` — Qwen3 + Mamba-3 hybrid

* **Class / model_type**: `Qwen3Mamba3ForCausalLM` / `qwen3_mamba3`; example
  `arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1`.
* **Keys**: `layer_types` (attention/SSM mix) plus Mamba-**3** knobs —
  `mamba_mimo_rank`, `mamba_bc_norm_eps`, `mamba_bc_bias_init`,
  `mamba_rope_fraction`, `mamba_a_floor`, `mamba_chunk_size`, `mamba_d_head`,
  `mamba_d_state`, `mamba_expand`, `mamba_n_groups`, `mamba_n_heads`.
* **Why an alias is wrong**: the engine has Mamba-1 (`RCPP_ARCH_MAMBA`) and
  Mamba-2 hybrids (`FALCONH1`, `NEMOTRONH`, `GRANITEMOEHYBRID`). MIMO rank, BC
  normalisation, an `a_floor`, and a rope fraction inside the SSM are Mamba-3
  parameters with no Mamba-2 equivalent — mapping to a Mamba-2 path would
  compute a different recurrence.
* **Real support needs**: a Mamba-3 SSM kernel + the hybrid layer loop, gated
  on decode validation against the reference.

## `molmoact2` — MolmoAct2 (vision-language-action)

* **Class / model_type**: `MolmoAct2ForConditionalGeneration` / `molmoact2`;
  example `NUSMAGIC/MolmoAct2-MolmoAct2-SO100_101`.
* **Text tower**: `text_config: molmoact2_text` — 36 layers, H=2560, 32 heads,
  8 kv, vocab 154624 (`qk_norm_type`, `rope_scaling_layers`, `norm_after`).
* **Not a text-only decoder**: `action_expert_config`, `flow_matching_*` (6
  keys), `max_action_dim`, `max_action_horizon`, action/state/depth token
  ranges — the model's contract is action generation, with depth reasoning.
* **Why an alias is wrong**: the existing `molmo` → `RCPP_ARCH_OLMO` mapping is
  right for a Molmo *VLM*; here it would drop the entire action expert.
* **Real support needs**: a scope decision first (text tower only, or the action
  path), then the action-expert/flow-matching head if in scope.

## `cosmos3edge` — Cosmos3-Edge (video world model)

* **Class / model_type**: `Cosmos3EdgeForConditionalGeneration` /
  `cosmos3_edge`; example `ubr-physical-ai/Cosmos3-Edge-NF4-bnb`.
* **Text tower**: `cosmos3_edge_text` — 28 layers, H=2048, 16 heads, 8 kv,
  vocab 131072, `mlp_bias`, `pretraining_tp` (llama-shaped). Alongside
  `vision_config` (`cosmos3_edge_vision`, `num_channels` → video patch embed),
  `projector_config`, `video_token_id`.
* **Why an alias is wrong**: the text tower *looks* llama-shaped, but the model
  is a video world model; the engine has no video/image pipeline for it, so a
  mapping would advertise support for something that cannot be served.
* **Real support needs**: scope decision (is video in scope at all?) before any
  mapping.

## `qwendriveforplanning` — Qwen-Drive (driving planner)

* **Class / model_type**: `QwenDriveForPlanning` / `qwen_drive`; example
  `Yuro1991/Qwen-Drive-1.0-4B`.
* **Not a text decoder**: the top-level config nests a `vlm_config`
  (`Qwen3_5ForConditionalGeneration` — the Qwen3.5 text tower plus vision) and an
  `expert_config`, and its own keys are trajectory planning: `trajectory_hz`,
  `trajectory_point_dim`, `num_future_points`, `num_history_points`,
  `trajectory_scale`, `num_inference_steps`, `noise_init_std`, `min_one_minus_t`,
  `max_reasoning_tokens`, plus `id2label`/`label2id` for two labels. Image
  geometry (`image_patch_size`, `image_spatial_merge_size`,
  `image_temporal_patch_size`, `current_image_pixels`, `history_image_pixels`)
  confirms a video/image input path.
* **Why an alias is wrong**: the familiar part is the Qwen3.5 tower; the model's
  contract is a **driving trajectory** — a sampled (denoising) plan over
  `num_inference_steps` with a noise seed and `min_one_minus_t`, not tokens. The
  engine has no head or pipeline for that, so a mapping would serve the tower and
  silently drop the planner, exactly as with `molmoact2` and `cosmos3edge` above.
* **Real support needs**: a scope decision first (is trajectory planning in scope
  at all?), then the expert/trajectory head.

## `qwen3_5_mla` — Qwen3.5 MLA attention variant

* **Class / model_type**: `Qwen3_5MLAForConditionalGeneration` / `qwen3_5_mla`;
  example `TelperionAI/Qwen3.8-27B-GLA-g2-EXL3-4.0bpw`.
* **Why an alias is wrong**: the supported `qwen35` path handles Qwen3.5's hybrid
  gated-delta-net/full-attention architecture; this class declares an MLA
  attention variant. The shared Qwen3.5 name does not establish compatible
  attention configuration or tensor layout, and mapping it to `qwen35` would
  advertise unverified support.
* **Real support needs**: verify the model config and tensor layout, then add
  and validate an MLA conversion/runtime path before mapping the class.

---

## Uncovered classes reviewed later — same standard, and still not aliases

The lane split above is the *watcher's*: a class is printed `!! SIGNIFICANT` when
`_is_significant()` calls it a major-family/vision arrival, and `!! UNCOVERED`
otherwise — and the alert's instruction for that second lane is "add the mapping
… alias if it is a known family". The three below arrived in the `!! UNCOVERED`
lane on 2026-09-13 and are **not** aliases either, for the same reason the
sections above exist: a mapping would claim support the engine has not
validated, and for `englishbase` it would be silently wrong rather than merely
unsupported.

Evidence below is from the official HF configs **and the models' own modeling
code**, fetched 2026-09-13.

### `englishbase` — `SlayerLab/fabryka-english-250m-e01-sft-v1`

**Implemented 2026-09-13 — no longer an uncovered class** (`RCPP_ARCH_ENGLISHBASE`
= 1006, generic backend): the two-matrix relu2 MLP rides the non-gated FFN branch
and the parameter-free QK-norm is the existing per-head path driven by an all-ones
`[head_dim]` weight. Validated token-for-token against the model's own code — 68/68
greedy tokens over three prompts — with `census/englishbase_parity_check.cpp`. The
review below is kept as the record of *why it was not an alias*, which is what
decided the shape of the implementation.

* **Class / model_type**: `EnglishBaseForCausalLM` / `fabryka_english_base`.
* **Shape**: 36 layers, H=768, 6 heads / 2 kv, head_dim 128, intermediate 3040,
  vocab 32768, tied embeddings, rope theta 10000, `rms_norm_eps` 1e-6,
  `qk_norm: true` — llama-shaped on the surface, which is why the watcher filed
  it as a family variant rather than a significant arrival.
* **Decisive evidence — the model's own `model.py`**: when
  `recipe.activation == "relu2"` (this config's `recipe` says exactly that) the
  block's MLP is replaced by `ReluSquaredMLP`, which computes
  `down_proj(relu(up_proj(x)).square())` — **one up matrix, no gate, ReLU²** —
  and the same file describes its QK normalization as *parameter-free*.
  `hidden_act: "silu"` is only the default the Llama config class fills in; the
  recipe overrides the MLP at construction, so the two keys disagree and the
  code is what settles it.
* **Why an alias is wrong**: Llama's and Qwen3's MLP is gated SwiGLU
  (`gate_proj` + `up_proj` + `down_proj`, SiLU), and Qwen3's qk-norm carries
  learned `q_norm`/`k_norm` weights. A mapping would load and then compute a
  different function — silent substitution, not a crash.
* **Real support needs**: a two-matrix ReLU² MLP and a parameter-free QK-norm, or
  a loud refusal.

### `gdn2` — `Berlm/hyb16-gdn2-s1`

* **Class / model_type**: `GDN2ForCausalLM` / `gdn2`.
* **Shape**: 24 layers, H=1024, 6 heads, head_dim 128, `conv_size` 4, hybrid —
  full attention on layers 3/7/11/15/19/23 — dense SwiGLU (2816), vocab 32000,
  tied embeddings.
* **The config schema is `fla`'s** (flash-linear-attention), not HF's:
  `attn_mode: chunk`, `fuse_cross_entropy` / `fuse_norm` / `fuse_swiglu`,
  `expand_v`, `use_short_conv`, `allow_neg_eigval`, and an `ra_*` block
  (`ra_project`, `ra_gate_b`, `ra_theta_init`, `ra_lambda_max`, `ra_feedback`)
  that no engine path reads.
* **Nearest implemented family, and why that is still not a mapping**: the
  registry does have Gated-Delta paths — `gateddeltanet` / `deltanet` /
  `tinygdn` → `RCPP_ARCH_QWEN3NEXT`, and `RCPP_ARCH_QWEN35` is Qwen3.5's *dense*
  GDN — but the engine's reader (`src/qwen3_5.cpp`) keys on Qwen3.5's own field
  names (`linear_conv_kernel_dim`, `linear_num_key_heads`,
  `linear_value_head_dim`) and on tensor names with expected sizes
  (`linear_attn.conv1d.weight`). This config supplies neither. A matching
  mechanism is not a matching checkpoint; an alias here would be a guess about
  weight layout.
* **Real support needs**: an `fla` → engine config adapter, a tensor-layout check
  against one real checkpoint, and a decision on `ra_*`.

### `haiku` — `kerzgrr/Haiku-base`

* **Class / model_type**: `HaikuForCausalLM` / `haiku`.
* **Shape**: 36 layers, H=1024, 8 heads / 8 kv, `layer_types` = `kda` ×3 then
  `gated_mla` (`full_attention_interval: 4`), `mlp_activation: "situ_glu"`
  (`situ_gate_cap`, `situ_up_cap`), MLA ranks (`mla_q_lora_rank` 512,
  `mla_kv_lora_rank` 256, `mla_qk_nope_head_dim` / `mla_v_head_dim` 128),
  `partial_rotary_factor` 0.5, rope 1e6, and MTP heads (`mtp_num_heads`,
  `mtp_adapter_rank`, `mtp_loss_weight`).
* **Why an alias is wrong**: KDA and gated MLA *are* named in the registry —
  `RCPP_ARCH_KIMI_K3` is "Moonshot Kimi K3 — 2.8T MoE with KDA + Gated MLA +
  LatentMoE" — but that token is **routed, not implemented**: `include/kimi_k3.h`
  is included by nothing in the tree and its `load_from_gguf` prints
  `[KimiK3] Loading not yet implemented for Strix Halo targets.` and returns
  false; `src/model_router.cpp` has no branch for the arch, so it falls through
  to the generic `hip_gpu` → `cpu_generic` default; and `situ_glu` appears in no
  engine source — the only SITU in the tree is upstream llama.cpp's
  `kimi_k3_situ`. (That routing/validation split is deliberate and documented:
  issue #1635, closed as "not fixable on this box".) Mapping Haiku onto KIMI_K3
  would chain one unvalidated class onto another.
* **Real support needs**: a SITU-GLU MLP plus a real KDA / gated-MLA
  implementation — the K3 work itself, which `research/TRACKING.md` still
  carries as not started ("QK-Normed MLA absorption on Kimi K3 gated-MLA
  decode").

### `vapor` — `Neeze/Vapor-2B-V1.2`

* **Class / model_type**: `VaporForCausalLM` / `vapor`.
* **It looks like the cleanest alias of the five.** The config carries the LFM2
  schema *verbatim*: every `block_*` and `conv_*` key identical to LFM2-2.6B,
  **all 30 `layer_types` identical**, hidden 2048, 30 layers, 32 heads / 8 kv,
  intermediate 10752, `conv_L_cache` 3. What is left over is packaging (`dtype`,
  `auto_map`, the `tie_*` pair), config-driven (vocab 128000, rope theta 1e7) —
  and one thing that is neither: an explicit `rank_config`.
* **The checkpoint is what disqualifies it** (weights index read 2026-09-13):
  layer 0 stores the dense SwiGLU form — `feed_forward.w1/w2/w3` — and
  **layers 1–28 do not**. They store `feed_forward.A2`, `A_shared`, `B1`, `B2`,
  `B3` plus `side_A2`, `side_A_shared`, `side_B1..B3`: the `rank_config`'s
  `SharedSwiGLULinear` (r_shared 1396, r2 1116, side_rank 8) realised in the
  tensors rather than a training-time detail that materialises on export.
* **Why an alias is wrong**: the dense names the readers know (`w1`/`w3`, ~18
  hits each in this repo) describe layer 0 only; `A_shared`, `side_B1` and
  `feed_forward.A2` have **0 hits** anywhere in the tree. An LFM2 mapping would
  read the two dense layers it recognises and have nothing for the 28 factorized
  ones.
* **Real support needs**: a materialisation (or runtime) path for
  `SharedSwiGLULinear`, alongside the plain dense block, then decode validation.

### `fidel` — `4E-AI/Fidel1.1-1B`

* **Class / model_type**: `FidelForCausalLM` / `fidel`.
* **Not a text transformer at all.** Its 359 tensors live entirely under a
  `policy.` prefix and the stack is bespoke (from the safetensors header, read
  by range request — the file is 5.6 GB and was never downloaded):
  * `policy.model.layers.N.mixer.mamba.*` — 88 tensors: the per-layer mixer is
    **Mamba**, not attention;
  * `policy.model.layers.N.moe.shared_fc1_gate` / `moe.w_gate` — a MoE with
    shared experts;
  * `policy.model.layer_adapters.N.base.{up,down}` and `.delta.{up,down}` — the
    per-layer rank adapters the config's `ranks` block describes (base 512,
    delta 256);
  * `policy.model.prompt_hyper_conditioner.*` (78 tensors) and
    `policy.model.prompt_cross_conditioner.*`;
  * `policy.model.contextual_prompt_v7/v8/v9.*` and `policy.chat_prompt.*` —
    four separate prompt-attention blocks;
  * `policy.model.lm_head.base.weight` + `lm_head.lora_a` — the head itself is
    base plus LoRA.
  The header's own metadata agrees with the config:
  `execution_contract: fidel_step9051_legacy_reference_fp32`.
* **Why an alias is wrong**: no engine family has mamba mixers + MoE + rank
  adapters + prompt conditioners, and `layer_adapters`,
  `prompt_hyper_conditioner` and `mixer.mamba` each have **0 hits** in this repo.
  There is no partial match to argue about.
* **Real support needs**: a scope decision first — this is a policy model, not a
  chat LM — and then the adapter/conditioner machinery.

### `zarya` — `ai-forever/Zarya-4B` (also -1.7B, -0.6B)

* **Class / model_type**: `Zarya` / `zarya`.
* **Shape**: 36 layers, H=2560, 32 heads / 8 kv, head_dim 128, intermediate 9728,
  vocab 151936, tied embeddings, rope 1e6 — llama-shaped on the surface, which
  is why the watcher's edit-distance guess proposed `zarya` → `RCPP_ARCH_ZAYA`.
* **Decisive evidence — the config is a diffusion LLM, not a Zaya MoE**: the
  official `config.json` carries `diffusion_attn_mode: causal`,
  `diffusion_loss_proportion: 0.5`, `diffusion_shuffle`, `noise_eps`,
  `noise_sorting`, `ordered_sampling`, `sample_t_override` / `sample_t_upper`,
  `grouped_noise` — a full-sequence diffusion/masking sampler. `RCPP_ARCH_ZAYA`
  is "Zaya MoE (Zyphra — MoE FFN with CCA attention)"; the Zaya configs carry
  `num_experts`/CCA keys, none of which appear here, and Zarya has no expert
  tensors to route. The two differ in the one thing a `rcpp_arch_from_string`
  mapping cannot express: the computation itself.
* **Why an alias is wrong**: mapping `zarya` onto ZAYA would route a diffusion
  sampler into the CCA-attention + MoE-FFN decode path — silent substitution,
  not a crash, exactly the `englishbase` / `vapor` failure mode above.
* **Real support needs**: a diffusion-LM sampler (or a loud refusal), not a
  one-line registry alias.

The common thread: each of these is *shaped* like a family the engine already
routes (`llama`, `gdn`, `kimi`, `lfm2`) — `fidel` only in the loosest sense, as a
Mamba/MoE hybrid — and each differs in a mechanism that changes the computation,
which is the one thing a `rcpp_arch_from_string` mapping cannot express. `vapor`
is the case worth remembering: from the config alone it looked like a one-line
alias, and only the tensors said otherwise.

**Reading a checkpoint without downloading it.** For a multi-GB
`model.safetensors`, the header is enough to settle this class of question and
costs one range request (follow redirects — `-L` — or you get the 302 body):

```sh
U=https://huggingface.co/<repo>/resolve/main/model.safetensors
H=$(curl -sSL -r 0-7 "$U" | python3 -c 'import sys,struct; print(struct.unpack("<Q", sys.stdin.buffer.read(8))[0])')
curl -sSL -r 8-$((8+H-1)) "$U" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d), list(d)[:20])'
```

`<repo>/raw/main/model.safetensors.index.json` answers the same question in one
request when the model is sharded.

---

### `needle3` - `webmp3/Sakura-Needle3-ONNX`

Architecture class `Needle3ForCausalLM`, `model_type: needle3`, no `auto_map` - so the
census classes it as native, not custom code. It is a small research model shipped as an
ONNX export; no pinned backend (`hrx`, `npu`) implements the class, and an ONNX graph is
not a GGUF architecture any backend could claim. Reviewed 2026-10-06: **not an alias**.

### `sepia-char-mlp` - `LUSCAINK/SEPIA-0`

Architecture class `SepiaCharMLP`, `model_type: sepia-char-mlp`, no `auto_map`. Its
`config.json` is not a transformer configuration at all (`forward`, `vocab`, `embedding`,
`hidden` - a character-level MLP), so it cannot be an alias of any supported family.
Reviewed 2026-10-06: **not an alias**.

### `kairo` - `kairo-crypto-model/kairo-v1.5`

Architecture class `KairoForCausalLM`, `model_type: kairo`, no `auto_map` - so the census
classes it as native rather than custom code. Nothing in the pinned llama.cpp tree
mentions `kairo`, and no pinned backend (`hrx`, `npu`) implements the class. Reviewed
2026-10-06: **not an alias**.

### `grug_moe` - `open-athena/Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10`

Architecture class `GrugMoeForCausalLM`, `model_type: grug_moe`, no `auto_map`, so the
census classes it as native. It is the marin-community research MoE (`marin-community/marin`,
`experiments/grug/moe`; `lib/levanter/src/levanter/models/snowball.py`): 67B total, about 2B
active, 26 layers, hidden 2560, 20 heads with 5 KV heads, 256 routed experts (4 per token) plus
one shared expert, a 2048-token sliding window with a global layer every 4 (`global_every`),
and its own attention knobs (`qk_mult`, `sconv`, `grugmoe_attention_mode`). Transformers,
upstream llama.cpp and upstream vLLM have no implementation; it is served through marin's own
vLLM fork and loaded in training through `marin-community/MarinSkyRL`, which registers the class
itself. The checkpoint flagged (engine#381) is a DARE-TIES merge-campaign upload with almost no
downloads; the other uploads are the same group's training snapshots (`open-athena/Snowball-67B-A2B-*`,
`penfever/grug-67b-a2b-*`). No pinned backend (`hrx`, `npu`) implements the class, and the
`qk_mult` / `sconv` / local-global layout rules out treating it as a Mixtral- or Qwen-MoE
alias. Reviewed 2026-10-12: **not an alias**. Its `registry/significant.json` entry carries
`"watch": "report"`, so the census watcher lists new uploads of it as `. REVIEWED` and does
not alert; remove that field if an upstream backend gains the architecture.

### `hrm_text` - `zyr-AGENT/ZY-S1`

Architecture class `HrmTextForCausalLM`, `model_type: hrm_text`, no `auto_map` and
`library_name: transformers`, so the census classes it as native, not custom code. It is
Sapient's HRM-Text hierarchical-reasoning LM, in-tree in upstream Transformers
(`src/transformers/models/hrm_text/{configuration,modeling,modular}_hrm_text.py`, "The Sapient
AI Authors and the HuggingFace Inc. team"). The base `sapientinc/HRM-Text-1B` config (read
2026-10-15) is 128 layers of hidden 1536, 12 heads / 12 kv, head_dim 128, intermediate 4096,
vocab 65536, rope 10000, `num_layers_per_stack: 16`, `H_cycles: 2`, `L_cycles: 3`,
`L_bp_cycles: [0, 3]`, `embedding_scale` 39.19, `prefix_lm: true`; the flagged upload
`zyr-AGENT/ZY-S1` (created 2026-10-12, last modified 2026-10-14) is a 16-layer finetune of it
with the same `H_cycles` / `L_cycles` / `L_bp_cycles`.

**Why an alias is wrong**: `H_cycles` / `L_cycles` run a stack of layers
(`num_layers_per_stack`) several times per token and `L_bp_cycles` feeds the low-level stack
back through the high-level one, on a prefix-LM (`prefix_lm`, `embedding_scale`) schedule. No
supported family does that: the closest decoder stack would read the same tensor names and
silently compute a different function (each stack run once, no cycle), the silent-substitution
failure this page exists to prevent.

**Why it is not mapped**: no pinned backend accepts `hrm_text`. Checked 2026-10-15 on the
pinned submodule `third_party/llama.cpp` 0e255d77: no `hrm` in `src/llama-arch.cpp`,
`gguf-py/gguf/constants.py` or `convert_hf_to_gguf.py`, and no `conversion/hrm_text.py`, so
`tools/registry_build.py` finds neither a converter class nor a runnable GGUF architecture.
The HRX fork's patched tip `ad18320a` is also free of `hrm`. Upstream `ggml-org/llama.cpp`
master does have `LLM_ARCH_HRM_TEXT` ("hrm_text") and `conversion/hrm_text.py`, so a later pin
bump that reaches it maps the class automatically - at that point
`tools/registry_build.py --check-gaps` fails until this entry records why (`mapped_ok`).

**Not a one-off.** The HF `hrm_text` tag lists 60 models from 34 uploaders and the 2026-10-11
census snapshot counts 41 `HrmTextForCausalLM` models, including `sapientinc/HRM-Text-1B`
(14,437 downloads, 834 likes) and `danish-foundation-models/DFM-Mimir{,-v1.5}` (2,175 / 2,824
downloads). Reviewed 2026-10-15: **not an alias**. Its `registry/significant.json` entry
carries `"watch": "report"`, so the census watcher lists new uploads of it as `. REVIEWED` and
does not alert while no pinned backend can serve the class; remove that field (and record
`mapped_ok`) when the llama.cpp pin gains `hrm_text`.

### `dflash2` - `nchapman/Qwen3.5-4B-DFlash2` (also `arc-kumbhar/incoai-Qwen3.8-27B-DFlash2`)

Architecture class `DFlash2DraftModel`, no `auto_map` and `library_name: speculators` on the
nchapman upload, so the census classes it as native and it alerts. Both flagged checkpoints are
**speculative-decoding drafters, not standalone LMs** (nchapman's card: "DFlash2 speculative
drafter for Qwen/Qwen3.5-4B"). The 2026-10-16 tensor headers (81 tensors each, read by range
request) carry machinery no pinned backend runs:

* `candidate_selector.hidden_projection.weight`, `candidate_selector.predecessor_codebook`,
  `candidate_selector.successor_codebook` - the candidate selector (`selector_rank: 256`,
  `selector_top_k: 16`);
* `layers.N.attention_conv.{base_kernel,kernel_projection.weight}` and
  `layers.N.mlp_conv.{base_kernel,kernel_projection.weight}` - the short convolutions
  (`conv_group_size: 16`, `conv_kernel_size: 2`);
* `fc.weight`, `hidden_norm.weight`, `norm.weight` and the per-layer attention/MLP tensors.

The pinned `third_party/llama.cpp` 1a5dcba2 **does** have a DFlash draft architecture
(`@ModelBase.register("DFlashDraftModel")` in `conversion/qwen.py`, `model_arch =
MODEL_ARCH.DFLASH`, runtime `src/models/dflash.cpp`), but that is DFlash-1. Its converter reads
`dflash_config.{target_layer_ids,mask_token_id}` and the runtime requires `target_layers`
metadata; it has no `candidate_selector` or `*_conv` tensors and no handling for
`selector_rank`, `selector_top_k`, `conv_group_size`, `conv_kernel_size`,
`aux_hidden_state_layer_ids`, `sliding_window_non_causal` or the `speculators_config` block.
Registering DFlash2 under `DFlashDraftModel` therefore loads a draft that silently drops the
selector and convolution heads - the silent-substitution failure this page exists to prevent -
and the `nchapman` config does not even carry `target_layer_ids` under `dflash_config`, so the
pinned converter could not build a GGUF for it at all. Reviewed 2026-10-16: **not an alias**.
Its `registry/significant.json` entry carries `"watch": "report"`, so the census watcher lists
new uploads of it as `. REVIEWED` and does not alert; remove that field (and record `mapped_ok`)
when a pin bump reaches DFlash2 support (selector + conv heads, per-layer window) and its
converter class registers `DFlash2DraftModel`.

### `tashkeel_nano` - `ahmadsy/tashkeel-nano-1p1m`

Architecture class `DiacritizerModel`, `model_type: tashkeel_nano`, no `library_name` and no
`auto_map`, so the census classes it as native. It is **not a causal LM**: its `config.json`
carries `causal: false`, `num_classes: 15` (the diacritic labels), `vocab_size: 171`,
`hidden_size: 64`, 18 layers, 2 KV heads, and a windows/decoding block (`window: 256`,
`stride: 128`, `arm: centered`). The repo ships its own `tashkeel_nano/` package
(`model.py`, `inference.py`, `labels.py`), so the class is served by that code, not by any
backend. No pinned backend registers `tashkeel_nano` and no supported family is a
fixed-window, 15-class tagger. Reviewed 2026-10-16: **not an alias**. Its
`registry/significant.json` entry carries `"watch": "report"`.

### `mbart` - `nhatho1999/orthomni-codeco`

Architecture class `MBartForConditionalGeneration`, `model_type: mbart`, `library_name:
ctranslate2`. It is an **encoder-decoder seq2seq** model - `is_encoder_decoder: true`, 12
encoder + 12 decoder layers, `d_model: 1024`, 16 heads per side, `BartphoTokenizer` - and the
upload is a CTranslate2/ONNX export with no PyTorch weights (`ct2/model.bin`,
`onnx/encoder_model.onnx`, `onnx/decoder_model_merged.onnx`). The engine's backends are
decoder-only text decoders (the HRX llama.cpp route and the NPU lanes), and the pinned
`third_party/llama.cpp` 1a5dcba2 registers no `mbart`/`bart` converter class or GGUF
architecture, so no pinned backend accepts the class. It is not an alias of any supported
family either - nothing the engine runs is an encoder-decoder. Serving it would need a
translation/seq2seq route the engine does not have, not a mapping. Reviewed 2026-10-16:
**not an alias**. Its `registry/significant.json` entry carries `"watch": "report"`, so the
census watcher lists new uploads as `. REVIEWED` and does not alert; remove that field (and
record `mapped_ok`) if a pinned backend ever grows an encoder-decoder route that accepts
`mbart`.

### `blind_decoder` - `Vi-LAB/KenT_0.5B_Base`

Architecture class `BlindForCausalLM`, `model_type: blind_decoder`, no `auto_map`,
`library_name: transformers`, so the census classes it as native. It is the only upload (0
downloads, 0 likes, created 2026-10-16) and the repo is internally inconsistent: `config.json`
declares `architectures: ["BlindForCausalLM"]` and `model_type: "blind_decoder"`, but the
shipped `configuration_kent.py` sets `model_type = "kent"` and `modeling_kent.py` defines
`KenTForCausalLM`; `BlindForCausalLM` is defined nowhere in the repo and there is no
`auto_map`, so stock Transformers sees an unknown model type with no remote code and cannot
load it (upstream Transformers has no `blind_decoder` module). The architecture itself is
bespoke, not a rename: 31 layers, `embed_dim: 1024`, 8 heads, vocab 50257, with
**Compressed-QK attention** (q/k head_dim 64 against a 128-dim value head) and **gated key
refinement** - `k = softmax(G G^T / sqrt(d)) K_raw` over a separate gate projection, before
RoPE - plus RMSNorm and a SwiGLU MLP (intermediate 2816). No supported family computes that
refinement; the closest decoder stack would read the same tensor names and silently compute a
different function. Reviewed 2026-10-16: **not an alias**. Its `registry/significant.json`
entry carries `"watch": "report"`; remove that field (and record `mapped_ok`) if a backend
implements the class.

### `sn38-nanochrono` - `tensin320/round14_5_114_fitleech_sn38-2023-01`

Architecture class `NanochronoForCausalLM`, `model_type: sn38-nanochrono`, no `auto_map` and
`library_name: transformers`, so the census classes it as native. The repo carries **no modeling
code at all** (only `config.json`, `generation_config.json`, `model.safetensors`, the tokenizer
and a README), and upstream Transformers has no `nanochrono`/`chrono` module, so nothing can
load it as published. Its config (read 2026-10-16) is 28 layers, hidden 1792, 14 heads / 14 kv,
vocab 32768, `layer_types` of 3 sliding + 1 full attention per group, `sliding_window: 512`,
`max_position_embeddings: 2048`, `final_logit_softcapping: 15.0`, `attention_multiplier: 1.2`,
`use_cache: false`, `tie_word_embeddings: false` - plus machinery no supported family has:
`tap_layer: 14`, `aux_layers` (every other layer), `aux_gate_dim: 12` and `mix_gate_dim: 24`,
the ChronoLLM/year-cutoff tap-and-gate stack the tags (`chronollm`, `year-cutoff`) name. There
is no pinned-backend mapping and no alias to claim: the tap/aux-gate path changes the
computation, so a plain decoder stack would be silent substitution. Reviewed 2026-10-16:
**not an alias**. Its `registry/significant.json` entry carries `"watch": "report"`; remove
that field (and record `mapped_ok`) if a backend implements the class.

### `muse_spark` - `Interchained/Muse-Spark-Open-816B-A42B-GGUF`

Architecture class `MuseSparkForConditionalGeneration`, `model_type: muse_spark`: Meta's
Muse Spark Open (`meta-models/Muse-Spark-Open-816B-A42B`, created 2026-09-15, gated by
manual approval), a 62-layer text model with vision and audio towers. The flagged upload is
a GGUF conversion (F16, Q4_0, Q4_K_M, Q5_K_M) whose header (read 2026-10-17 by range request)
declares `general.architecture = muse-spark`: 256 experts, 8 used, `moe_latent_size: 4096`
(experts run in a 4096-wide latent, like Nemotron 3 Super's `ffn_latent_down`/`ffn_latent_up`),
`expert_gating_func: 2`, hidden 8192, 128 heads / 16 kv, head_dim 64, sliding window 2048 with
one full-attention layer in four (`sliding_window_pattern`), `final_logit_softcapping: 20.0`,
`logit_scale` 0.177, context 1,048,576, `gpt-4o` pre-tokenizer.

**Why an alias is wrong**: the latent-space experts, the per-layer rope schedule
(`layer_rope_theta` 0 on the full-attention layers) and the softcapped logits change the
computation; no supported family has that combination, so mapping it to a Llama 4 or
Nemotron-H shape would be silent substitution.

**Why it is not mapped**: the pinned `third_party/llama.cpp` 1a5dcba2 has no `muse-spark`
architecture (it has the `moe_latent_size` key and the latent tensors only for Nemotron 3
Super). Upstream `ggml-org/llama.cpp` added it on 2026-10-15 in ba3c05f71 ("model: add Muse
Spark Open (text, vision and audio)", #30616: `LLM_ARCH_MUSE_SPARK`, `src/models/muse-spark.cpp`,
`conversion/muse_spark.py`), after the fork's last upstream sync, so a later fork sync that
reaches it maps the class automatically - at that point `tools/registry_build.py --check-gaps`
fails until this entry records why (`mapped_ok`).

**Not a one-off, but not servable here.** The HF `muse_spark` tag lists 23 models (Meta's
base and NVFP4, unsloth, mlx-community, intel-ai INT4, amd Quark-MXFP4). Even its smallest
GGUF (Q4_K_M, ~500 GB) is about four times the 128 GB Strix Halo, so the gap costs nothing on
the target box today. Reviewed 2026-10-17: **not an alias**. Its `registry/significant.json`
entry carries `"watch": "report"`, so the census watcher lists new uploads as `. REVIEWED` and
does not alert; remove that field (and record `mapped_ok`) when the pin gains `muse-spark`.

### `glm5_next` - `azampatti/GLM-5.3-Flash-AZ` (also `zai-org/GLM-5.3-Flash`)

Architecture class `Glm5NextForConditionalGeneration`, `model_type: glm5_next`, no `auto_map` and
`library_name: transformers`, so the census classes it as native and it alerts. The flagged upload
is a 4-bit auto-round derivative of `Intel/GLM-5.3-Flash-W4A16-AutoRound`; the release itself is
zai-org's GLM-5.3-Flash.

**Shape** (configs read 2026-10-23, `text_config`): 45 layers, hidden 4096, 64 heads, vocab
154880, 288 routed experts with 4 used plus 1 shared (`moe_intermediate_size` 2048,
`first_k_dense_replace` 3), DeepSeek-style MLA (`kv_lora_rank` 512, `qk_nope_head_dim` /
`qk_rope_head_dim`, `mla_use_nope`), `index_topk` 2048 with 32 index heads, and
`num_nextn_predict_layers: 1` (MTP). The top level wraps a `text_config` plus a `vision_config`
and image/video token ids, so the class is a VLM wrapper, not a bare decoder.

**Machinery no pinned family runs**:

* `layer_types` is a **34 + 11 hybrid** - 34 `linear_attention` layers (KDA:
  `linear_attn_config.kda_layers`, `gate_lower_bound` -5.0, `short_conv_kernel_size` 4, head_dim
  128) and 11 `deepseek_sparse_attention` layers (`full_attn_layers` 3/7/11/.../43, all with
  `indexer_types: full`).
* `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 - hyper-connections, i.e. the residual stream
  is a 4-wide Sinkhorn-normalised expansion rather than one residual per layer.
* `index_kpool` 4 with `index_kpool_compress` and `index_kpool_always_select_tail` - the sparse
  attention's key pool is compressed and a tail is always selected, which the fork's GLM-5.2
  indexer does not implement.

**Why an alias is wrong**: the pinned fork's GLM-5.2 path (`GlmMoeDsaForCausalLM` ->
`LLM_ARCH_GLM_DSA`, `src/models/glm-dsa.cpp`) is an MLA + DeepSeek-sparse-attention stack on
**every** layer: it reads the MLA, `indexer_top_k` and `indexer_types` keys and assumes each layer
runs that attention. The fork's KDA support is Kimi's (`LLM_ARCH_KIMI_LINEAR`, `LLM_KV_KDA_HEAD_DIM`,
the Kimi converter), and nothing in the tree reads `hc_mult` / `hc_sinkhorn_iters` for a GLM model
(only `dsv4_hc_mult` on DeepSeek V4 / qwen4exp). Mapping `glm5_next` onto `glm-dsa` would run 34
KDA layers as full sparse attention, drop the hyper-connection expansion and ignore the compressed
k-pool - tokens out, a different function in: the silent-substitution failure this page exists to
prevent. Mapping it onto `kimi-linear` would drop the DSA layers and the indexer instead.

**Why it is not mapped**: the pinned `third_party/llama.cpp` af04c9f2 has no `glm5-next` at all -
no `@ModelBase.register("Glm5NextForConditionalGeneration")` in `conversion/glm.py` (it registers
`Glm4ForCausalLM`, `Glm4vForConditionalGeneration`, `GlmOcrForConditionalGeneration`,
`Glm4MoeForCausalLM`, `Glm4vMoeForConditionalGeneration`, `Glm4MoeLiteForCausalLM`,
`GlmMoeDsaForCausalLM`, `SolarOpenForCausalLM`), no `glm5-next` in `src/llama-arch.cpp`'s
`LLM_ARCH_NAMES` (`glm4`, `glm4moe`, `glm-dsa` only) and no `src/models/glm5-next.cpp`. Upstream
`ggml-org/llama.cpp` added it in #27773 (`649dcb10`, 2026-09-30, `LLM_ARCH_GLM5_NEXT`,
`src/models/glm5-next.cpp`, `MODEL_ARCH.GLM5_NEXT` in the GLM converter) and the MTP follow-up
#29928 (`b9acf138`, 2026-10-07) - both after the fork's last upstream sync, so a later fork sync
that reaches them maps the class automatically. At that point `tools/registry_build.py
--check-gaps` fails until this entry records why (`mapped_ok`), which is the intended prompt for a
person to confirm the sync really brought the KDA + DSA + hyper-connection layers before the class
counts as supported. Reviewed 2026-10-23: **not an alias**. Its `registry/significant.json` entry
carries `"watch": "report"`, so the census watcher lists new uploads as `. REVIEWED` and does not
alert; remove that field (and record `mapped_ok`) when the pin gains `glm5-next`.

### 2026-10-27 batch (engine#619)

All four classes in that census run are reviewed below. The mirror the run named for
`GraniteSwitchForCausalLM` (`cestaristudio/granite-switch-4.1-3b-preview`) is gated, so that one
listing stays unclassifiable and is listed under *Unverifiable: gated configs*; the class itself
is IBM's own and its configs are public, so it is reviewed here from
`ibm-granite/granite-switch-4.1-*`. None of the four is registered by the pin: `git grep` over
`third_party/llama.cpp` `e7e5cd08` for `LiveMemForCausalLM`/`livemem`,
`CDMMoELMPoincare`/`cdm_moe_poincare` and `Senryu2ForCausalLM`/`senryu2` returns nothing, and the
same grep over the pin `main` carries now (`40c4ccb7`, re-checked 2026-10-29) returns nothing for
those three or for `GraniteSwitchForCausalLM`/`granite_switch` either, so no pinned backend reads
them and a mapping would be a claim the engine cannot honour.

The same run also printed the REMOTE-CODE families `KeuralMoECausalLM`,
`LNewqForConditionalGeneration`, `Newq1ForConditionalGeneration` and
`NewqForConditionalGeneration`. Those are not significant gaps and are not in
`registry/significant.json`: they are recorded under *If a family is a variant* instead, because
what they need is a converter registration (a renamed class), not a review of a new architecture.

### `livemem` - `Popbones/LiveMem-RL-mlx-q4-backbone`

Architecture class `LiveMemForCausalLM`, `model_type: livemem`, `library_name: livemem-mlx`, no
`auto_map` (the censused upload is MLX-format weights: `model.safetensors` + `manifest.json`,
vendored `livemem-mlx` runtime). The config is a plain dense decoder - `hidden_size` 2560,
`intermediate_size` 9728, `head_dim` 128, every entry of `layer_types` is `full_attention`, Qwen2.5
vocab ids (151643/151645) - so the machinery is unremarkable; what is missing is the format and
the registration, not the layers. Serving it needs an MLX dequant and a converter, and the pin has
neither. Reviewed 2026-10-27: **not an alias**. Its `registry/significant.json` entry carries
`"watch": "report"`.

### `cdm_moe_poincare` - `DuoNeural/Lattice-7-Hermes-CDM-Poincare-Instruct`

Architecture class `CDMMoELMPoincare`, `model_type: cdm_moe_poincare`, no `auto_map`,
`library_name: transformers`. Not an alias for a structural reason: attention runs in a Poincare
ball (`geometry: poincare`, learnable `curvature: 1.4427`) and a `telepathy` gate
(`beta_tele: 0.5`) mixes it, neither of which any pinned family implements; the rest is small
(`d_model` 768, 8 layers, 12 heads, 4 experts, `teacher: Qwen/Qwen3.5-4B`, 629M params). The repo
ships `lattice7_hermes-q8-attn-q6-rest-text.gguf`, but a GGUF of an architecture no converter
knows is not loadable by the pinned runtime - the metadata would name an arch
`src/llama-arch.cpp`'s `LLM_ARCH_NAMES` does not have. Reviewed 2026-10-27: **not an alias**. Its
`registry/significant.json` entry carries `"watch": "report"`.

### `senryu2` - `KerzgrAI/Senryu-2.2-1B`

Architecture class `Senryu2ForCausalLM`, `model_type: senryu2`, `library_name: tiny_gdn`, no
`auto_map`, and the weights come with the repo's own `inference.py` plus
`ngram_tables.safetensors` - the class is served by that code, not by any backend. Its
`layer_types` are a 3+1 hybrid of `gdn2` and `parallax` (`full_attention_interval` 4,
`attention_layer_type: parallax`, `kda_lower_bound` -5.0, `linear_conv_kernel_dim` 4,
`allow_negative_eigenvalues: false`): the GDN2 half is the class this page reviews elsewhere, but
`parallax` attention and the n-gram tables are not in the pin, so `gdn2` would be a partial
substitution, not an alias. Reviewed 2026-10-27: **not an alias**. Its `registry/significant.json`
entry carries `"watch": "report"`.

### `granite_switch` - `ibm-granite/granite-switch-4.1-3b-preview` (also -8b, -30b)

Architecture class `GraniteSwitchForCausalLM`, `model_type: granite_switch`, no `auto_map`,
`library_name: transformers`. `registry/census.json` (2026-10-29) counts **17** models of this
class, so it is not a one-off: `ibm-granite/granite-switch-4.1-{3b,8b,30b}` are public and the
family already ships GGUFs (`ibm-granite/granite-switch-4.1-8b-GGUF`, whose
`general.architecture` is `graniteswitch`).

* **Shape**: 42 layers (30b: 66), hidden 2560 (8b/30b: 4096), 40 heads / 8 kv (8b/30b: 32/8),
  vocab 100365 (8b/30b: 100355), every entry of `layer_types` `full_attention`,
  `num_local_experts: 0` with a leftover `num_experts_per_tok: 2` - i.e. dense, all-attention
  Granite.
* **What it adds**: the model is a *stack of LoRA adapters switched per token*, not one set of
  weights. The config carries `num_adapters`, `adapter_names`, `adapter_ranks`,
  `max_lora_rank`, `lora_target_modules`, `switch_head_dim`, `control_token_gain`,
  `dual_stream`, `cross_stream_rank`, `ms_code_*`/`ms_memory_gain`, `asr_*` and
  `adapter_token_ids`/`adapter_substitute_token_ids`. Upstream's converter puts that in one
  sentence (`conversion/granite.py`: "Dense, all-attention Granite with N per-token embedded
  LoRA adapters, stacked over the adapter dim with a zero adapter at slot 0 (N = num_adapters +
  1)") and additionally writes `adapter_count`, `adapter_lora_rank`, `adapter_token_ids_*`,
  `adapter_router_gain` and, for `ibm-granite`'s `decision` adapter, a `granite_switch` decision
  type; `num_hidden_layers` also counts *weightless switch layers* that carry no tensors, and
  the converter has to derive the real block count from the tensor names.
* **Why an alias is wrong**: the nearest supported families are `granitehybrid`
  (`GraniteMoeHybridForCausalLM`), `granitemoe` and `granite`, and none of their loaders has an
  adapter stack, a router gain or a switch layer: loading a checkpoint through them would keep
  one adapter's weights and silently drop the rest, which is the silent-substitution failure
  this page exists to prevent. Renaming it to `granitehybrid` would also lose the ASR path.
* **Why it is not mapped**: the pinned backend has nothing to map to. At `40c4ccb7`
  `conversion/granite.py:17-379` makes six `@ModelBase.register` calls covering eight class
  names (`GraniteForCausalLM`, `GraniteMoeForCausalLM`, `GraniteMoeSharedForCausalLM`,
  `GraniteMoeHybridForCausalLM`/`BambaForCausalLM`,
  `GraniteSpeechForConditionalGeneration`, `GraniteSpeechPlusForConditionalGeneration`,
  `Granite4VisionForConditionalGeneration`), `src/llama-arch.cpp` has `granite`, `granitemoe`
  and `granitehybrid` but no `graniteswitch` (nor `granite_swa`), `src/models/` has
  `granite.cpp`, `granite-moe.cpp` and `granite-hybrid.cpp` and no `granite-switch.cpp`, and
  `gguf-py` has no `GRANITE_SWITCH` at all - as expected, since `registry/architectures.json`
  is generated from that tree.
* **Where the support is**: upstream `ggml-org/llama.cpp` `157b81fe` ("model : Granite-Switch
  Architecture", #25107, 2026-08-10) registers the class, adds `LLM_ARCH_GRANITE_SWITCH` /
  `"graniteswitch"` and the mapping case. That is *older* than the pin's tip (2026-10-25), so a
  fork sync is what brings it, and `tools/registry_build.py --check-gaps` then fails until this
  entry records why (`mapped_ok`) and someone confirms the loaded adapters really switch before
  the class counts as supported. Reviewed 2026-10-29: **not an alias**. Its
  `registry/significant.json` entry carries `"watch": "report"`, so the census watcher lists new
  uploads as `. REVIEWED` instead of filing an issue; remove that field (and record `mapped_ok`)
  when the pin gains `graniteswitch`.

## Unverifiable: gated configs

A gated repository cannot be classified at all: the watcher's listing carries no config for it, so
the class is unknown and the census counts it as **unverifiable** rather than guessing it. The
watcher sends a token when one is in the environment. `tools/census_watch.py`'s `hf_token()` reads
`HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN` (the name `huggingface_hub` writes), and
`request_headers()` adds `Authorization: Bearer <token>` when either is set and non-empty. No token
*file* is read: the standard `~/.cache/huggingface/token` is never consulted, so a token that only
lives in that file changes nothing - export one of the two variables for the process that runs the
watch (`scripts/census-watch.sh` passes its environment through to `tools/census_watch.py`). With
neither set, the request is the user agent alone (`1bit-census-watch`), byte-identical to what the
watcher always sent, and a gated row stays unverifiable: reported the first time it is seen,
remembered in the state file, and never an alert, so it does not move the exit code. The header
goes on the architecture walk's listing pages (`hf_get_page`) only; the FastFlowLM org listing,
repo details and release pages (`get_json`) still send the user agent alone. A rejected token is not
fatal: a `401` in `hf_get_page` sets `_AUTH_REJECTED`, warns `the token was rejected (401);
continuing without it` on stderr, and retries that page anonymously, and the rest of the run stays
unauthenticated rather than dying - a stale token costs a line of stderr, not the sweep. HF answers
the listing `200` even with an invalid Bearer (measured 2026-10-29: `GET /api/models` returns `200`
with a bad token, only `GET /api/whoami-v2` returns `401`), so that fallback is defensive: the
watcher calls no endpoint that distinguishes a valid token from a bad one.

Long-standing: `WaveMatrix/Qwen3-VL-8B-Instruct-GPTQ-Int4` is unverifiable on every run.

From the 2026-10-27 census (engine#619): `cestaristudio/granite-switch-4.1-3b-preview` - the
repo listing is gated (`Access to model ... is restricted`), so with no token in the environment
the listing carries no config and the row stays **unverifiable** (reported once, remembered, no
alert), not counted as an uncovered class. The *class* it would have named is no longer unknown: it
is IBM's `GraniteSwitchForCausalLM`, reviewed from `ibm-granite`'s public repos under
`granite_switch` above, which is what the census now reports for that family. A token the repo has
been granted access to puts the config in the listing and the row is classified like any other -
`!! UNCOVERED` only if the pin does not map it; that has not been observed with a working token.

The 2026-10-22/23 watcher run behind engine#447 listed twelve, all unverifiable for the same
reason - the watcher sent no token on that run:

`Chandangepala/spendai-gemma3-1b`,
`DrJohnCode/NebulaText`,
`Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster`,
`ModaLeap/qwen3-4b-instruct-2507-onnx`,
`ReadyArt/Wicked-Tutu-31B`,
`dimitarpg13/semsimula-ladder-live-owt-d384-l2-attention-potential-rfqk-norc`,
`executorch-community/Gemma-3-1B-IT-ExecuTorch-WebGPU`,
`gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-bf16ple`,
`japanese-data-analyze/JMicro-10M-chat-research-20261018`,
`seoan1024/Hanok-LLM`,
`tasmaikeni13/algebraic-transformer`,
`zendevvv/odoko-gemma3-1b`.

None of them is evidence of an unmapped class: until a token is available they stay in the
unverifiable count, which `docs/registry.md` reports separately from mapped and checked.

## If a family *is* a variant

An unmapped class that is a renamed or same-shape sibling is fixed where the backend is: upstream llama.cpp's converter and `llama-arch.cpp`, or our HRX fork. The registry picks the mapping up at the next pin bump. Verify shape equality before proposing such a mapping: layers, hidden size, heads, experts, and any added config keys. That check is exactly what moved the families above into this file.

Two classes flagged on 2026-10-14/15 are **variants**, not significant gaps, so they are
deliberately *not* in `registry/significant.json` and keep alerting until a backend registers
them:

* `GLMForCausalLM` (`JUROCLINIC/seguridad`) is the all-caps spelling of the Transformers class
  `GlmForCausalLM` (`model_type: glm`), which the pinned fork already maps to `chatglm`
  (`conversion/chatglm.py`: `@ModelBase.register("GlmForCausalLM", "ChatGLMModel",
  "ChatGLMForConditionalGeneration")`). The registry lookup is exact, so only the CamelCase
  string is mapped; the flagged upload is a config-only repo (no weights) that also mis-cases
  the `model_type`. The fix is registering the alias in the fork's converter, not an
  architecture review - and it is only reachable via a pin bump, so it is not silenced here.
* `MinistralForCausalLM` (`saidutta69/Ministral-8B-Instruct-heretic`) is a Mistral-shaped
  sibling: `model_type: ministral`, `sliding_window` 32768 == `max_position_embeddings`, and
  the same `layer_types` and 36-layer / H=4096 / 32-head / 8-kv shape as its base
  `mistralai/Ministral-8B-Instruct-2410`, which publishes `MistralForCausalLM`/`mistral`. The
  pinned fork maps `MistralForCausalLM` -> `llama` and `Ministral3ForCausalLM` -> `mistral3`
  but does not register this name (upstream llama.cpp does not either). Same treatment: a
  converter registration picked up at the next pin bump, so it also keeps alerting rather than
  being recorded as significant.

From the 2026-10-27 census (engine#619), plus its 2026-10-29 rerun, four REMOTE-CODE families are
recorded the same way - read and dispositioned here, deliberately *not* in
`registry/significant.json`, and still what the census will alert on if a native (no `auto_map`)
upload of the class appears:

* `LNewqForConditionalGeneration` (`crussella0129/LNewq`) and `Newq1ForConditionalGeneration`
  (`crussella0129/Newq1`) are **one rename each of the same Qwen3.5-VL wrapper**: `model_type:
  qwen3_5`, `text_config.model_type` `qwen3_5_text` (32 layers, H 2560, `head_dim` 256,
  `full_attention_interval` 4, vocab 248320) over a `vision_config` with
  `deepstack_visual_indexes`, `image_token_id` 248056 - the shape the pin already converts under
  the name `Qwen3_5ForConditionalGeneration` (`conversion/qwen3vl.py:16` registers that class and
  its MoE sibling; `registry/architectures.json` maps it to `qwen35`). The only unusual keys are
  `lnewq_steer` /
  `newq1_steer`, and both repos carry `auto_map` (`AutoModelForImageTextToText` and
  `AutoModelForCausalLM` -> `modeling_lnewq.LNewqForConditionalGeneration` /
  `modeling_newq1.Newq1ForConditionalGeneration`), i.e. they run their own modelling code. The
  pinned fork registers neither name, so the class - not the architecture - is what is
  unmapped, and the fix is one converter registration in the fork (a pin bump), exactly like the
  two classes above. The census counts one model each. The third name in that batch,
  `NewqForConditionalGeneration` (1 model, `crussella0129/Newq`), is now a gated repo, so its
  config could not be read on 2026-10-29; it is recorded as the same family name pattern,
  unverified.
* `KeuralMoECausalLM` (`hossain0909/keural-dpo2-final`, 24 models from one uploader in
  `registry/census.json`) is **not** a rename of anything mapped: `model_type: keural`,
  `auto_map` to `configuration_keural.KeuralConfig` /
  `modeling_keural.KeuralMoECausalLM`, and the config is an MoE decoder the pin has no arch for
  (24 layers, H 4096, 32 heads / 8 kv, `head_dim` 128, 8 experts, top-2, `sliding_window` 512,
  vocab 131074, plus a `keural_step` key). No pinned converter or runtime reads that name, so
  mapping it would be a claim the engine cannot honour. It is left out of significant.json on
  purpose: it is a remote-code one-off until a second uploader uses it, which is why the census
  reports it as `. REMOTE-CODE` and starts alerting at `REMOTE_CODE_SPREAD` (3) uploaders - that
  alert is the prompt to review it for a variant shape, not to silence it now.
