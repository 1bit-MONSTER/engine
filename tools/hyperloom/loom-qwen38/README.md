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
