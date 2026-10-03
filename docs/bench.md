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

# How the numbers are measured: tools/bench.py

`tools/bench.py` measures `1bit serve` configurations against each other. It is how the serve
numbers in these docs are taken, and how a [recipe](recipes.md) earns its place.

```sh
tools/bench.py -m Qwen3.8-27B-Q4_0-H32.gguf \
    --config base= \
    --config dflash='--dflash Qwen3.8-27B-DFlash2-q8_0.gguf' \
    --rounds 2 --lock ~/.cache/lax-decode/box.lock --json one-server.json
```

## Method

- **The tool runs every measurement.** Each configuration gets its own fresh `1bit serve`,
  started, measured and stopped by the tool. A number in a table is one the tool measured, not
  a prediction or an earlier run.
- **The first `--config` is the baseline**, measured in the same run as the others. Every other
  row reports its change against that baseline's median.
- **Interleaved rounds.** With `--rounds N` the configurations run A B A B ..., so drift on a
  shared box (thermal, another job on the memory bus) hits all of them alike.
- **One lock for a shared box.** `--lock FILE` holds that flock for the whole run. Other jobs
  that take the same lock (big model loads, other benchmarks) cannot overlap it.
- **Best and median.** Each cell is the best and the median over all runs. The median is what
  the change is computed on.

## What is measured

- **Prompt:** one long prompt (default: the first 8,000 characters of `docs/*.md`, about 1,800
  tokens; `--prompt-file` to change it), `--prompt-reps` times (default 5). It is reported as
  serve reports prompt tok/s, so context checkpoints and a drafter's own prompt pass are included.
- **Decode:** 256 greedy tokens (`--decode-tokens`) on three chat prompts, `--decode-reps` times
- **Mixed prompts:** `--prompts mixed` decodes nine ordinary requests (C++, a word problem, a summary, Spanish, Chinese, JSON, a story, a debugging checklist, an email) and reports the median over them, their range, and tokens delivered per second (generated tokens over prompt plus decode time). Use it for drafters (`--mtp`, `--dflash`): a draft model guesses repetitive text almost perfectly, so a single easy prompt overstates its gain.
  each (default 3):
  - **code:** an ISO-8601 parser with tests;
  - **prose:** how a B-tree insert works;
  - **short:** a one-sentence translation.

  Drafters behave differently on the three, so all three are reported.
- Thinking is off and the prompt cache is off for every request.

`--json FILE` keeps every raw measurement with the configurations and the time. Serve's own log
for each configuration goes to `--log-dir` (default `bench-logs/`), including the
[recipes](recipes.md) it applied.

`tests/bench_selftest.sh` (ctest `bench_selftest`) runs the tool against `1bit serve` with
`tests/fake_backend.py` behind it, without a GPU.
