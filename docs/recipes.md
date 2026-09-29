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

A setting that makes one model faster on one route can make another slower. For example, 1024-token
micro-batches speed up a Hadamard-rotated MoE model's prompts by 3-8% and slow the dense
Qwen3.8-27B's by 4%. `1bit serve` keeps such settings in `config/recipes.json`, not in code. Each
recipe says what it matches, what it adds, why, and the measurement behind it:

```json
{
  "id": "rotated-moe-ub1024",
  "match": {"device": ["rocm"], "hadamard_q4_0": true, "moe": true},
  "args": ["-ub", "1024"],
  "why": "W4A4 expert matmuls fill the matrix units better with 1024-token micro-batches; 2048 is no better.",
  "measured": "Qwen3-Coder-30B-A3B-Q4_0-H32, llama-bench pp2048 2,001 -> 2,158, ...",
  "source": "engine #205, #229"
}
```

## How serve applies them

For each llama-server backend (`vulkan`, `hrx`, `rocm`) serving a `.gguf`, serve checks every
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
1bit serve: recipe rotated-moe-ub1024: -ub 1024
```

- `--recipes FILE` replaces the built-in set with the recipes in FILE.
- `--no-recipes` turns recipes off.
- A malformed file (an unknown key, a recipe without `measured`) stops serve and names the problem.

`tests/recipes_route.sh` (ctest `recipes_route`) checks all of this without a GPU.

## The built-in recipes

| id | Matches | Adds | Measured |
|---|---|---|---|
| `rotated-moe-ub1024` | rocm, Hadamard-rotated, MoE | `-ub 1024` | Qwen3-Coder-30B-A3B-H32 pp2048 2,001 -> 2,158; dense 27B-H32 is slower with it (491-495 -> 471-473) |
| `rocm-dflash-p-min-0.4` | rocm, `--dflash` | `--spec-draft-p-min 0.4` | Qwen3.8-27B-H32 + DFlash2, decode code / prose / short: 41.9 / 24.8 / 13.4 at p-min 0 -> 41.9 / 26.8 / 16.7 tok/s |
| `dflash-p-min-0` | `--dflash` | `--spec-draft-p-min 0` | Qwen3.8-27B-H32 + DFlash2: accepted block 5.4 -> 6.7 tokens on code (the ROCm tree's default 0.75 cut it) |

Recipes apply in file order. The first recipe to add a flag wins, so a narrower recipe goes before
a broader one for the same flag.

Settings that follow from the files themselves stay in serve's code. Examples: the Hadamard
activation rotation a stamped file needs, and the DFlash draft length, which is the drafter's
`dflash.block_size` minus one.

## Adding or changing a recipe

Measure first, with [`tools/bench.py`](bench.md): the configuration with the setting against
the one without it, in the same run. Put the numbers in `measured` and the PR in `source`. A
recipe without a measurement does not load.
