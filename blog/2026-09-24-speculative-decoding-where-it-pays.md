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
tags: speculative decoding, mtp, qwen3.8

# Speculative decoding: where it pays, and where it does not

Speculative decoding lets a cheap drafter guess the next few tokens and the model check them all in one pass. When the guesses are right, several tokens come out of one pass instead of one. We measured it on the two most popular Unsloth models on Strix Halo, and the answer depends entirely on the model.

## Dense Qwen3.8-27B: up to 3.4x

Qwen3.8-27B ships its own multi-token-prediction (MTP) head, a small extra layer trained to draft. `1bit serve --mtp` switches it on. Decode speed on three chat prompts (code, prose, short), UD-Q4_K_XL on Vulkan:

| Setting | Code | Prose | Short |
|---|---|---|---|
| No drafting | 12.2 | 12.3 | 12.3 |
| `--mtp` (draft 3, the default) | 35.9 | 28.2 | 28.8 |
| `--mtp --mtp-max 6 --mtp-p-min 0.5` | **42.0** | 27.3 | 25.2 |
| `--mtp --mtp-max 8` | 21-25 | 13-20 | 10-20 |

The default is the all-rounder: 2.3-2.9x on every prompt. Code is predictable enough to keep long drafts (93% of drafted tokens accepted), so draft length 6 with a confidence threshold of 0.5 reaches 3.4x there. Prose is not (56-76% accepted), and at length 8 the rejected drafts cost more than the accepted ones save. Adding free n-gram drafting on top of MTP gained nothing.

Accuracy does not change: a drafted token is kept only when the model would have produced it anyway.

## Qwen3-Coder-30B-A3B: never

Qwen3-Coder-30B-A3B is a mixture of experts with 3B parameters active per token. It already decodes at 88 tok/s on Vulkan, and it has no MTP head, so we tried separate draft models (Qwen3 0.6B, 1.7B and 4B, 36 combinations of draft length and threshold on Vulkan and ROCm) and three n-gram methods:

| Method | tok/s on code |
|---|---|
| **No drafting** | **88** |
| n-gram drafting | 89 (a tie) |
| Qwen3-0.6B draft, length 4 | 84, with 82-87% of drafts accepted |
| Qwen3-1.7B draft | 41-66 |
| Qwen3-4B draft | 20-44 |

Even a draft model that is right five times out of six loses: checking its guesses costs almost as much as generating them, when the model only reads 3B parameters per token.

## The rule

Speculative decoding pays when the model is expensive per token and brings its own drafter: dense models with an MTP head. It does not pay on small-active mixtures of experts. And it is a single-user lever: a server with MTP loaded batches about a third less well, so under load the engine keeps MTP and batching apart (see [1bit serve](../docs/serve.md)).
