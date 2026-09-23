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

## Next

The C++ scorer (1bit-MONSTER `src/laya_scorer.cpp`, on
`backup/laya-and-results-2026-09-22`) is ported next. It is gated against
the Python reference in `third_party/laya` on the pinned checkpoints, then
wired into the router that picks NPU, HRX, Vulkan or ZINC per request.
