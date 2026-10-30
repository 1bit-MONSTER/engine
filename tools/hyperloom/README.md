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

[Hyperloom](https://github.com/AMD-AGI/Hyperloom) (MIT) is AMD's agentic inference optimizer. Since
1.1 it has a supported route for a workload that is not one of its shipped frameworks:
`--framework custom` on the scriptable path, with the entrypoint, the lever grid and the
correctness gate supplied by the operator. Upstream also knows this board now
(`--gpu-type radeon8060s` = gfx1151, 40 CU), so nothing here patches Hyperloom itself.

| Path | What it is |
| --- | --- |
| `custom/` | The adapter: entrypoint, measurement, lever grid, launch line. Start at [`custom/README.md`](custom/README.md). |
| `hrx2kineto.py` | Converts an HRX dispatch profile to a Kineto trace (Hyperloom's profile step, TraceLens, rocprof studio). With the fork's shape log it names every dispatch by its ggml op, shapes and layer, and doubles as a TraceLens extension (roofline, categories, TraceDiff). |
| `hrx-tracelens.sh <model> <prompt\|decode> [VAR=value ...]` | Profiles one run, converts it, and writes a TraceLens report (optionally with TraceDiff against another run). |

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
