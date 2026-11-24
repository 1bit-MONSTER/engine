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
# Qwen3.8-27B on the Loom route as a Hyperloom 1.2.0 custom workload

Since engine#3390 the 27B UD-Q4_K_XL serves on the **Loom route** (card `unsloth-Qwen3.8-27B-UD-Q4_K_XL` @ 3e43d41d,
preset `chat`, `amdgpu://`, `--pool_backing=fixed`). The HRX llama.cpp path is the frozen fallback, so tuning happens
here. This directory reuses `../inkling-loom/` entirely (ticket, `box_inkling.sh`, `gauntlet.py`, `hl_score.py`):
`custom_radeon8060s.sh` only pins `HL_CARD_NAME` (default `qwen38-27b-3e43d41d`, a card under the box's
`/data/dev/cl-hyperloom/cards/`) and this `levers.json` is the 27B's own grid.

- **Baseline = the shipped route:** an unset lever keeps the card's own preset value (`draft_depth` 3, 4 x 262144,
  `prefill_capacity` 1024, `max_tokens` 4096), no completion spin, fixed pool, loom-serve's default 8 workers.
- **Levers:** MTP draft depth 0-5 (the GGUF's own MTP head; the card has no separate drafter), prefill capacity
  2048/4096/8192 (the card declares `@llm.tokens` in `range(1, 8192)`), slots 1x256k / 2x256k / 8x128k (`@llm.context`
  is capped at 262144 per row, so no 1M preset), elastic pool, workers 4/16.
- **Not a lever:** completion spin. cl-loom's spin A/B on this card (hrx-loom 0.3.0, 23 Nov) measured no difference,
  and #3395 ships it off.
- **Gate:** as inkling-loom: needle recall + 17*23 + greedy-text agreement (>= 80 % of replies identical over 160
  characters vs the baseline). No KLD half: loom-serve 0.3.0 exposes no logits.

## Measured (run (b), 23 Nov 2026)

One run, **no lever kept**, though Hyperloom's own verdict was two KEEPs -- both read against a baseline
(54.5) measured in a slow window: the 16.8k needle prefilled at ~506 tok/s there against ~576 tok/s in every
explore round and in the later no-lever re-measure (~12 %), and the no-lever config itself read 54.5 in that
window and **56.09** later (+2.9 %). Against the same-window reference MTP depth 4's KEEP is **-0.7 %** --
the window, not the lever -- and depth 5 reads **+2.0 %** (its +2.7 % was the step over depth 4, measured in
the same later window): one single-arm comparison with no interleaved A/C/A/C and mixed rows, so it is not a
recorded gain and the route kept the card preset (MTP depth 3, prefill 1024). Depths 0/1/2, prefill
2048/4096/8192 and the slots presets measured below the 1 % bar; elastic pool, workers 4/16 and depth 6/7
are unmeasured (exit 75, #3363), not losses. Per-round tables:
[#3366](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7118818084); the record and the
window lesson are [`../README.md`](../README.md#measured-runs-120-23-nov-2026).
