<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Hyperloom on Strix Halo (gfx1151)

[Hyperloom](https://github.com/AMD-AGI/Hyperloom) (MIT) is AMD's agentic inference optimizer. This
directory targets **1.2.0** (21 Nov 2026: InferenceSuite measures every round; bypass/Magpie and the
single-node Ray layer are gone; the custom script is found by board name). Since 1.1 it has a supported route for a workload that is not one of its shipped frameworks:
`--framework custom` on the scriptable path, with the entrypoint, the lever grid and the
correctness gate supplied by the operator. Upstream also knows this board now
(`--gpu-type radeon8060s` = gfx1151, 40 CU), so nothing here patches Hyperloom itself.

| Path | What it is |
| --- | --- |
| `custom/` | The engine's HRX route (Qwen3.8-27B and other GGUFs on the installed payload): entrypoint, measurement, lever grid, launch line. Start at [`custom/README.md`](custom/README.md). |
| `inkling-loom/` | Inkling-2-Mini on the Loom route (loom-serve, hrx-loom 0.3.0, the loom-fun card): entrypoint, gauntlet, score + gate, lever grid. [`inkling-loom/README.md`](inkling-loom/README.md). |
| `box_ticket.sh` | Shared by both entrypoints (Hyperloom **1.2.0**): Hyperloom runs on the orchestration host, and each round validates its levers and runs as ONE `box-hold gpu` ticket on the Strix Halo box over ssh, killed by id if the round is stopped. |
| `hrx2kineto.py` | Converts an HRX dispatch profile to a Kineto trace (Hyperloom's profile step, TraceLens, rocprof studio). With the fork's shape log it names every dispatch by its ggml op, shapes and layer, and doubles as a TraceLens extension (roofline, categories, TraceDiff). `--summary` ranks by ggml op; `--by-kernel` ranks the same profile by **kernel** with TFLOP/s and % of this box's fp16 WMMA roof, and `--by-shape` by (op, K, N) with the launch width that shape implies — the shape view is what separates "the tile is fine" from "this launch uses a fifth of the machine" (engine#3103). `--self-test` runs both rankings' fixture arms. |
| `hrx-tracelens.sh <model> <prompt\|decode> [VAR=value ...]` | Profiles one run, converts it, and writes a TraceLens report (optionally with TraceDiff against another run). |

## Measured runs (1.2.0, 23 Nov 2026)

Both 1.2.0 runs closed with **no lever**: nothing Hyperloom kept was shipped, and neither run produced
a `config/loom-routes.json` or `config/recipes.json` change. The per-round tables are on
[#3366](https://github.com/1bit-MONSTER/engine/issues/3366); this section is what the two runs concluded,
because a KEEP is relative to the run's own baseline, and run (b)'s baseline was measured in a slower
window.

### Run (a): Inkling-2-Mini on the Loom route (`inkling-loom/`)

Card `a8354e77`, loom-serve 0.3.0, balanced, 3 runs per row (medians), session
`20261123T050743Z-8a3ee737`. 18 declared levers (DSpark depth 0-6, slots/context, prefill capacity,
completion spin, wait mode, pool backing, workers), **0 KEEP**: the card's own preset (DSpark d=3,
4 x 262144, prefill 4096, fixed pool, max spin, 8 workers) is best or tied on every lever the workload can
reach ([run conclusion](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7094773570)).

- The two rounds that read as slots losses -- 2 x 512k at 125.4 (-4.5 %) and 1 x 256k at 127.4 (-3.0 %) --
  ran inside another lane's NPU activity and are contamination, not lever results. Re-measured under the
  fixed both-locks ticket they read **130.22 (-0.7 %)** and **131.39 (+0.2 %)** against a fresh **131.18**
  baseline: ties inside the 1 % bar
  ([re-measure](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7104635584)). The other
  slots presets (1 x 1M 131.1, 8 x 128k 130.7) tie too.
- `prefill_capacity` 16384 is not a lever: the card declares `@llm.tokens` in `range(1, 8192)` and refused
  to register, so that round was a correct refusal of an invalid lever, not a measurement; #3371 dropped
  it ([correction](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7095772868)).
- Five later Hyperloom proposals never reached the box: server arguments and a patched checkout cannot
  reach a scriptable workload, so they measured the card defaults again (the proposal labelled "draft off"
  scored 131.2 where the real `HL_DRAFT_DEPTH=0` measured 94.0). #3370 refuses such a round now instead of
  measuring the baseline.

### Run (b): Qwen3.8-27B UD-Q4_K_XL on the Loom route (`loom-qwen38/`)

Card `3e43d41d`, the shipped route, session `20261123T210953Z-71d1bdaf`, 14 declared levers. Hyperloom
**kept MTP depth 4 (+2.1 %) and then depth 5 (+2.7 % on top, its final claim 57.2 = +4.87 % over its own
baseline, 54.5)**. The baseline was **taken in a slow-box window**: the 16,809-token needle prefilled
in **33.1 s (~506 tok/s)** in the baseline rounds (the 19:09Z hand-run and the 21:10-21:30Z Hyperloom
baseline) against **29.2 s (~576 tok/s)** in every explore round and in the later no-lever re-measure --
that window was **~12 % slower in prefill**
([caveat](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7116861311)). Prefill is one of
the five geomean legs ([`inkling-loom/hl_score.py`](inkling-loom/hl_score.py)); at ~12 % it is worth
`(576/507.8)^(1/5) - 1 = +2.5 %` of the score on its own, and the no-lever config read **54.5** in that
window and **56.09** in the later one (**+2.9 %**) -- a swing larger than either KEEP.

A no-lever re-measure of exactly the baseline config in the later window (ticket
`hl-20261123T225126Z-1363809`) gives **56.09** (TTFT 29.2 s / 575.7 tok/s, decode unchanged). Against that
same-conditions reference:

| arm | score | vs 56.09 | Hyperloom's read against 54.5 |
| --- | --- | --- | --- |
| baseline, no lever | **56.09** | reference | 54.5 |
| MTP depth 4 | 55.7 | **-0.7 %** | KEEP, +2.1 % |
| MTP depth 5 | 57.2 | **+2.0 %** | KEEP, +2.7 %; final +4.87 % |

**No lever: the KEEPs were read against a baseline window.** Depth 4's +2.1 % is entirely the window --
it is a loss against the same-window reference (-0.7 %). Depth 5 was the better arm, but its +2.7 % is the
step over depth 4 (55.7), measured in the same later window, so the step is not the window; against the
same-window no-lever reference it reads +2.0 %, above the 1 % bar on one single-arm comparison with no
interleaved A/C/A/C and mixed rows (reasoning +8 %, 16.8k TG +9 %, sustained decode -6 %, code flat), so it
is unvalidated and not a gain on the record; its final claim, +4.87 %, is +2.0 % once the window is
removed. It was not adopted: the 27B keeps its card preset (MTP depth 3, prefill 1024), and no
`config/loom-routes.json` / `config/recipes.json` change came from the run
([run end](https://github.com/1bit-MONSTER/engine/issues/3366#issuecomment-7118818084)).

- Below the 1 % bar (REVERT): depth 0 / 1 / 2 (27.3 / 41.6 / 50.2), prefill 2048 / 4096 / 8192, slots
  1 x 256k / 2 x 256k / 8 x 128k.
- **Unmeasured, not losses:** elastic pool, workers 4 / 16, depth 6 / 7 and a few of Hyperloom's own
  proposals exited 75 with their tickets lost in the board-run reboots (#3363's clean failure). Proposals
  carrying HRX-route knobs (`HL_MTP_MAX`, `--device hrx --parallel 8`) were refused by the workload, as
  intended on the Loom route.

### A KEEP is measured against the run's own baseline

`hl_score.py` makes the first measurement of a state directory the reference every later round is compared
with, and Hyperloom takes that baseline once. A baseline scored in a degraded window therefore carries that
window into every later KEEP -- here the tell is in every report: the baseline's 16.8k-needle TTFT (33.1 s)
against the round's (29.2 s). Before believing a KEEP, compare the round's window with the baseline's and
re-measure the no-lever arm in the same window. The window moves the other way too -- run (a)'s slots
losses were a concurrent NPU ticket -- and the measurement a lever PR needs is the interleaved A/C/A/C
(no-lever and candidate back to back, same card and gauntlet); neither run's KEEPs got one.

## History

The first route here (2026-09/10) patched an installed Hyperloom 1.0.0: `hl_patch_strixhalo.py`
added a `strixhalo` GPU type next to each `mi355x` entry, `hl_cli_login.py` forced agent turns onto
the logged-in `claude` CLI and let `RAY_VERSION` move past a pin that had no Python 3.14 wheel, and
`launch.sh` wrapped `hyperloom optimize` for ZAYA1-8B with `custom_strixhalo.sh` as the entrypoint.
All of that is retired by `1bit/hyperloom-custom`: upstream 1.1 resolves the board on its own,
installs on Python 3.12, and takes the workload through `--framework custom` without a patch. The
ZAYA1 reference file went with it; the gate now records its own references per model.

`hrx2kineto.py` and `hrx-tracelens.sh` are unchanged and stay: they are the bridge between an HRX
dispatch profile and TraceLens, and the profile leg of the adapter uses them.
