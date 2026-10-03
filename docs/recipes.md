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

# Recipes: tuned backend settings, as data

A setting that makes one model faster on one route can make another slower. For example, on the
removed lean ROCm route 1024-token micro-batches sped up a Hadamard-rotated MoE model's prompts by
3-8% and slowed the dense Qwen3.8-27B's by 4%. `1bit serve` keeps such settings in
`config/recipes.json`, not in code. Each recipe says what it matches, what it adds, why, and the
measurement behind it:

```json
{
  "id": "dflash-p-min-0",
  "match": {"drafter": ["dflash"]},
  "args": ["--spec-draft-p-min", "0"],
  "why": "A DFlash block is only worth drafting whole. ...",
  "measured": "Qwen3.8-27B-H32 + DFlash2 on the lean ROCm route (removed in RFC #213 stage 3): ...",
  "source": "engine #194"
}
```

## How serve applies them

For each llama-server backend (`hrx`, `cpu`) serving a `.gguf`, serve checks every
recipe against the launch:

| `match` key | Matches when |
|---|---|
| `device` | the backend's device is in the list |
| `architecture` | the file's `general.architecture` is in the list |
| `moe` | the file has (`true`) or lacks (`false`) an `<architecture>.expert_count` above 0 |
| `hadamard_q4_0` | the file is (`true`) or is not (`false`) stamped by `tools/hadamard_q4_0.py` |
| `drafter` | the drafter is in the list: `none`, `mtp` (`--mtp`) or `dflash` (`--dflash`) |

A key that is left out matches anything. A matching recipe adds each flag in `args`, with the
values after it, unless the backend's command line already has that flag. So whatever serve
sets itself, including from the command line (`--mtp-p-min`, `--mmproj`'s micro-batch), wins.
It adds each variable in `env` unless it is already set. Serve prints what it added:

```
1bit serve: recipe dflash-p-min-0: --spec-draft-p-min 0
```

- `--recipes FILE` replaces the built-in set with the recipes in FILE.
- `--no-recipes` turns recipes off.
- A malformed file (an unknown key, a recipe without `measured`) stops serve and names the problem.

`tests/recipes_route.sh` (ctest `recipes_route`) checks all of this without a GPU.

## The built-in recipes

| id | Matches | Adds | Measured |
|---|---|---|---|
| `dflash-p-min-0` | `--dflash` | `--spec-draft-p-min 0` | Qwen3.8-27B-H32 + DFlash2: accepted block 5.4 -> 6.7 tokens on code (the ROCm tree's default 0.75 cut it). Our HRX llama.cpp defaults to 0 already, so on HRX it pins the value; not yet re-measured there |

The two ROCm-only recipes, `rotated-moe-ub1024` and `rocm-dflash-p-min-0.4`, left with the ROCm
build (RFC #213 stage 3).

Recipes apply in file order. The first recipe to add a flag wins, so a narrower recipe goes before
a broader one for the same flag.

Settings that follow from the files themselves stay in serve's code. Examples: the Hadamard
activation rotation a stamped file needs, and the DFlash draft length, which is the drafter's
`dflash.block_size` minus one.

## Adding or changing a recipe

Measure first, with [`tools/bench.py`](bench.md): the configuration with the setting against
the one without it, in the same run. Put the numbers in `measured` and the PR in `source`. A
recipe without a measurement does not load.
