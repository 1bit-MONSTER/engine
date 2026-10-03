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

One setting can make one model faster and another slower: on the removed lean ROCm route,
1024-token micro-batches sped up one MoE model's prompts by 3-8% and slowed
Qwen3.8-27B's by 4%. `1bit serve` keeps such settings in `config/recipes.json`, not in code. Each
recipe says what it matches, what it adds, why, and the measurement behind it:

```json
{
  "id": "example",
  "match": {"device": ["hrx"], "drafter": ["dflash"]},
  "args": ["--spec-draft-p-min", "0.4"],
  "why": "what the setting does for this match",
  "measured": "the model, file, settings and numbers, measured with tools/bench.py",
  "source": "the PR that added it"
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
| `drafter` | the drafter is in the list: `none`, `mtp` (`--mtp`) or `dflash` (`--dflash`) |

A key that is left out matches anything. A matching recipe adds each flag in `args`, with the
values after it, unless the backend's command line already has that flag. So whatever serve
sets itself, including from the command line (`--mtp-p-min`, `--mmproj`'s micro-batch), wins.
It adds each variable in `env` unless it is already set. Serve prints what it added:

```
1bit serve: recipe example: --spec-draft-p-min 0.4
```

- `--recipes FILE` replaces the built-in set with the recipes in FILE.
- `--no-recipes` turns recipes off.
- A malformed file (an unknown key, a recipe without `measured`) stops serve and names the problem.

`tests/recipes_route.sh` (ctest `recipes_route`) checks all of this without a GPU.

## The built-in recipes

None, for now. Every recipe the engine had was measured on a build RFC #213 stage 3 removed:
`rotated-moe-ub1024` and `rocm-dflash-p-min-0.4` (the lean ROCm build) and `dflash-p-min-0` (a no-op
on our HRX llama.cpp, whose default draft p-min is already 0). New ones come with HRX measurements.

Recipes apply in file order. The first recipe to add a flag wins, so a narrower recipe goes before
a broader one for the same flag.

Settings that follow from the files themselves stay in serve's code. Examples: the 2-bit decode
copy a ternary Q4_0 file gets on HRX0, and the DFlash draft length, which is the drafter's
`dflash.block_size` minus one.

## Adding or changing a recipe

Measure first, with [`tools/bench.py`](bench.md): the configuration with the setting against
the one without it, in the same run. Put the numbers in `measured` and the PR in `source`. A
recipe without a measurement does not load.
