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
# Weekly releases

1bit engine is pinned upstream: every backend it builds is a submodule at a fixed commit
(`third_party/`), and the `bump-*` workflows open a PR whenever an upstream moves. The engine
runs inside Lemonade, which releases on Fridays. So the engine releases once a week, on Sunday,
rebuilt from scratch at that week's pins.

Releases are tagged `v<ISO year>.<ISO week>` (`v2026.40`), like Lemonade's `2026.39.1`; a second
release in the same week gets `.1`.

## What a release carries

| file | what |
|---|---|
| `1bit-<tag>-linux-x86_64.tar.zst` | `1bit` and every backend: Vulkan (upstream llama.cpp), HRX, lean (ROCmFPX), ZINC, ONNX (ryzenai-server), DwarfStar, the NPU lane with XRT |
| `lemonade-onebit-<tag>-linux-x86_64.tar.zst` | Lemonade (`lemond`, `lemonade`) from our fork, with the onebit recipe |
| `1bit-<tag>-windows-x64.zip` | `1bit.exe`, Vulkan `llama-server.exe`, `ryzenai-server.exe` ([windows.md](windows.md)) |
| `1bit-os-<tag>.img.zst`, `1bit-os-<tag>.efi` | 1bit OS, the engine as a bootable USB image ([os/README.md](../os/README.md)) |
| `changes.json`, `SHA256SUMS` | what moved upstream since the last release, as data; checksums |

The Linux tarball keeps the build tree's layout: `1bit` finds each backend at the same path
relative to itself as in the build (`app/built_path.h`), and every program finds its libraries
beside itself. Unpack it anywhere:

```sh
tar --zstd -xf 1bit-v2026.40-linux-x86_64.tar.zst
1bit-v2026.40-linux-x86_64/1bit serve -m model.gguf --device vulkan --port 8000
```

The builds target Strix Halo: the llama.cpp backends are built with `GGML_NATIVE` on its Zen 5. HRX and the ROCm backends need TheRock in
`/opt/rocm-therock`; the NPU needs the in-kernel `amdxdna` driver. The closed NPU add-ons
([npu.md](npu.md), "Private routes") are never in a release.

## The Sunday run

`scripts/weekly.sh <workdir>` on Strix Halo, in stages:

1. **bumps**: each open `bump-*` PR is merged onto `main` locally, built with every backend, and
   tested: `ctest` (with the `serve` end-to-end tests on Qwen3-0.6B) plus the check that bump needs,
   below. A PR that passes is brought up to date, waits for CI, and is merged. One that fails stays
   open, and the release lists it as held back.
2. **build** and **test**: `main` at the merged pins, the same way.
3. **package**: the four packages above.
4. **release**: `tools/weekly_changes.py` lists, for every pin that moved since the last release,
   the upstream commits (authors, PRs) and releases in between; that becomes the release notes and
   `changes.json`, and `gh release create` publishes it all.

`WEEKLY_DRY_RUN=1` merges and publishes nothing; stages can be run one at a time
(`scripts/weekly.sh ~/.cache/1bit-weekly build test`).

### Bumps

| PR | check beyond build + ctest |
|---|---|
| `bump-lemonade/*` | Lemonade's LLM suite through the onebit recipe on Vulkan ([lemonade.md](lemonade.md)) |
| `bump-laya/*` | `tests/laya_route_e2e.sh` on the pinned checkpoint |
| `bump-ds4/*` | DwarfStar's routed-MoE test (`DS4_TEST=1 scripts/build-ds4.sh`) |
| `bump-linux/*` | the kernel packages build (`scripts/build-kernel.sh`); nothing is installed |
| `bump-hrx/*`, `bump-llama-vulkan/*`, `bump-rocmfpx/*`, `bump-zinc/*`, `bump-xdna/*`, `bump-tokenizers/*` | covered by build + ctest (serve end to end on each device) |

## The announcement

After the release, the 1bit engine Discord gets a new post in #announcements, written from
`changes.json` in the layout of Lemonade's release posts: what the week brings, the notable upstream
changes with credit to their authors, the rest as a list, and the downloads. Then two lists written
straight from the data: **New models** (repos published under our Hugging Face org, models added to
Lemonade's catalog, architectures the registry gained or that gained a backend) and **Updates** (every
upstream as pinned in the release, with its latest release and changelog). It is the one weekly
digest; each week's post is new and earlier ones stay.
