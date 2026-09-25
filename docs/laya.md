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
# Laya

Step 4 of the port (docs/PORTING.md): Laya picks where each request runs.
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) is a
non-autoregressive decision model: a ModernBERT-style encoder plus an RLCD
decision head. It answers typed questions (`choice`, `score`, `noul`) about a
request in one forward pass, with calibrated confidence. It has three
checkpoints and a router that picks between them.

## The pin

| | Pin | Kept current by |
|---|---|---|
| Source | `third_party/laya` = `NandhaKishorM/laya` main **`1e28ac20`** | `bump-laya.yml` |
| Checkpoints | `config/laya.json` = Hugging Face `convaiinnovations/laya` at revision **`aa8c91ca`**: root (842 MB), `multilingual/` (644 MB, 34 MB tokenizer), `typed-decisions/` (842 MB) | `bump-laya.yml` (moves the source and the revision together) |

```sh
scripts/fetch-laya.sh ~/models/laya-pinned
```

`fetch-laya.sh` downloads exactly the pinned revision and checks every file
against the Hub's list: sha256 for the LFS weights, the git blob id for the rest.
Files already present and correct are not fetched again. Images and eval plots
are skipped.

Verified 2026-09-23 on Strix Halo: 20 files, 2.3 GB, all hashes match.

## The C++ scorer

Ported from 1bit-MONSTER `src/laya_scorer.cpp` on
`backup/laya-and-results-2026-09-22`, without its history. Zero Python at
runtime: weights from `model.safetensors` (`laya/safetensors.{h,cpp}`, F16/F32
-> f32), tokens from the Hugging Face `tokenizer.json` through the engine's
`npu::Tokenizer` (`laya/scorer.{h,cpp}`). One forward pass answers typed
questions with the same calibrated confidence as the Python reference.

| | Where |
|---|---|
| Scorer | `laya/scorer.{h,cpp}` |
| Safetensors reader | `laya/safetensors.{h,cpp}` |
| Router | `laya/route.{h,cpp}` |
| Gate harness | `tests/laya_gate.cpp` |
| Routing e2e | `tests/laya_route_e2e.sh` + `tests/fake_backend_route.py` |

`npu::Tokenizer` gained `token_id()` and the Rust-tokenizers ByteLevel regex
(the `use_regex` pattern for `{"type":"ByteLevel"}` with no explicit Split, as
ModernBERT serializes it); digits stay together so multi-digit tokens merge.

## Gate against the Python reference

The scorer is gated against the Python `laya` package on the two
ModernBERT-large checkpoints (root and `typed-decisions/`; `multilingual/` is
mmBERT-base and stays a separate follow-up). `tests/laya_gate.cpp` runs the
questions + state recorded in a `golden.json` and compares raw logits and
act logits, the argmax decision, and temperature-1 softmax probabilities.

| Checkpoint | max \|logit\| diff | act_logit rel | argmax | max \|prob\| diff |
|---|---|---|---|---|
| root (`laya`) | 8.6e-6 | 1.1e-6 | 0 | 1.1e-6 |
| `typed-decisions/` | 4.8e-6 | 6.1e-7 | 0 | 5.7e-7 |

Tolerances: logits <= 1e-3, act logits <= 1e-3 relative, identical argmax,
probabilities <= 1e-4. Verified 2026-09-25 on Strix Halo.

## Routing

`1bit serve --device auto --laya-model <dir>` routes each request with the
scorer instead of the old `auto -> vulkan` hardcode: the fixed routing question
(\"which device should run this request?\") returns the device, and the serve
process lazily starts that device's backend and forwards the request to it.
`1bit route --laya-model <dir> --state <text>` prints the device for one
request. `tests/laya_route_e2e.sh` proves a request reaches the backend the
scorer chose (fake backends stand in for llama-server and zinc).

`multilingual/` (mmBERT-base) and Apple/MLX routing are out of scope for this
step.
