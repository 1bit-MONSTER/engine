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
| `lemonade-onebit-<tag>-linux-x86_64.tar.zst` | Lemonade (`lemond`, `lemonade`) from our fork with the onebit recipe, and the engine in `1bit/`: `1bit` and every backend: HRX (and the CPU, from the same llama-server), ONNX (ryzenai-server), DwarfStar, the NPU lane (no XRT). No Vulkan or ROCm llama.cpp since RFC #213 stage 3 |
| `1bit-os-<tag>.img.zst`, `1bit-os-<tag>.efi` | 1bit OS, a Lemonade appliance with the engine as a bootable USB image ([os/README.md](../os/README.md)) |
| `changes.json`, `SHA256SUMS` | what moved upstream since the last release, as data; checksums |

The engine is embedded in Lemonade, so it ships only inside those two: there is no standalone
engine tarball and no Windows package (Lemonade has its own Windows build). On the GPU the engine
is pure HRX with HIP kernels: no Vulkan, ROCm backend, Metal, ZINC or CUDA. Lemonade serves those
with its own backends.

The engine in `1bit/` keeps the build tree's layout (`scripts/package-linux.sh`): `1bit` finds each
backend at the same path relative to itself as in the build (`app/built_path.h`), and every program
finds its libraries beside itself. Unpack the package anywhere and start Lemonade through
`lemond-onebit`, which points Lemonade's onebit backend at the packaged `1bit`
(`LEMONADE_ONEBIT_BIN`):

```sh
tar --zstd -xf lemonade-onebit-v2026.40-linux-x86_64.tar.zst
lemonade-onebit-v2026.40/lemond-onebit           # lemond with the onebit recipe on the packaged engine
```

The builds target Strix Halo: the llama.cpp backends are built with `GGML_NATIVE` on its Zen 5. HRX needs TheRock in
`/opt/rocm-therock`; the NPU needs the `amdxdna` kernel driver. The closed NPU add-ons
([npu.md](npu.md), "Private routes") are never in a release.

### Private components

Two components of the release binaries are proprietary and compiled in; their source is not in
this repository or in any release. They ship under the [1bit Binary License](../LICENSE-BINARY):
anyone may use them for any purpose, including commercial use, and pass on exact unmodified
copies with the license and notices; modifying them, reverse engineering them (except where the law
expressly allows it) and removing notices are not allowed. The third-party code inside them is listed
with its licenses in [NOTICE-BINARY](../NOTICE-BINARY). Every release package and the 1bit OS image
carry both files beside the engine's `LICENSE` and `NOTICE`:

| Component | Compiled into | Pin | Build option |
|---|---|---|---|
| The 1bit NPU runtime ([npu.md](npu.md), "The NPU runtime") | `1bit` (static) | `config/npu-runtime.config` | `ONEBIT_NPU_RT_DIR` |
| The private HIP kernels for HRX ([hrx.md](hrx.md), "Private HIP kernels") | `hrx/llama/bin/libggml-hrx.so` (code objects) | `config/gpu-kernels.config` | `ONEBIT_GPU_PRIVATE` |

The build stage fetches each with `scripts/fetch-private.sh <pin> <dir>`: the pinned path at the
pinned commit only, with the release machine's `gh` login (read access to the private
repositories). The run stops if either cannot be fetched or is not at its pinned sha. Move a pin
by hand, after the NPU check or the HRX checks on Strix Halo.

The engine is XRT-free: the NPU lane runs on the compiled-in runtime, and nothing links AMD's
XRT. `scripts/package-linux.sh` strips every packed program and library (`strip
--strip-unneeded`), and after packing the engine tarball the package stage checks
(`logs/package-checks.log`; a failure stops the release):

- **XRT-free:** for the build's `1bit` and every ELF file in the tarball, no `NEEDED` `libxrt*`
  entry, no `libxrt` in `ldd`, no `libxrt` string (`strings -a`); no `libxrt*` file packed.
- **Kernels in:** the packed `libggml-hrx.so` exports the add-on's `ggml_hrx_hip_addon_register`.
- **No source:** no source file (`.hip`, `.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, `.inc`, `.cu`,
  `.py`, `.cmake`) and no file named like one of the private checkouts' files.
- **Licensed:** `LICENSE`, `NOTICE`, `LICENSE-BINARY` and `NOTICE-BINARY` are in the tarball.

1bit OS carries no XRT either; its `1bit.check` boot reports `CHECK no XRT`.

To exercise only the private builds and the packaging without releasing, on Strix Halo:

```sh
WEEKLY_DRY_RUN=1 WEEKLY_TAG=rehearsal-private scripts/weekly.sh ~/.cache/1bit-weekly build package
```

## The Sunday run

`scripts/weekly.sh <workdir>` on Strix Halo, in stages:

1. **bumps**: each open `bump-*` PR is merged onto `main` locally, built with every backend, and
   tested: `ctest` (with the `serve` end-to-end tests on Qwen3-0.6B) plus the check that bump needs,
   below. A PR that passes is brought up to date, waits for CI, and is merged. One that fails stays
   open, and the release lists it as held back.
2. **build** and **test**: `main` at the merged pins, the same way.
3. **package**: the two packages above. `scripts/package-linux.sh` packs the engine first; its
   tarball goes into the Lemonade + onebit package as `1bit/`, not into the release on its own.
4. **release**: `tools/weekly_changes.py` lists, for every pin that moved since the last release,
   the upstream commits (authors, PRs) and releases in between; that becomes the release notes and
   `changes.json`, and `gh release create` publishes it all.

`WEEKLY_DRY_RUN=1` merges and publishes nothing; stages can be run one at a time
(`scripts/weekly.sh ~/.cache/1bit-weekly build test`).

### Bumps

| PR | check beyond build + ctest |
|---|---|
| `bump-lemonade/*` | Lemonade's LLM suite through the onebit recipe on HRX ([lemonade.md](lemonade.md)) |
| `bump-laya/*` | `tests/laya_route_e2e.sh` on the pinned checkpoint |
| `bump-ds4/*` | DwarfStar's routed-MoE test (`DS4_TEST=1 scripts/build-ds4.sh`) |
| `bump-hrx/*` | every `hrx` row of `registry/check_models.tsv` through `tools/registry_check.py` (serve end to end per architecture, ZAYA1 and the Qwen MoE models included); any FAIL holds the bump |
| `bump-xdna/*`, `bump-tokenizers/*` | covered by build + ctest (serve end to end on each device); the xdna-driver pin is the `amdxdna` kernel driver, which the release does not build: check it on Strix Halo with the module loaded ([npu.md](npu.md), "The XDNA driver") |

Before the build, every bump also checks that each pin it moves is on the branch `.gitmodules`
names for that submodule (GitHub's compare API: the branch equals or contains the pin). A pin off
its branch is held: the shallow clones follow the branch and cannot fetch it, and GitHub may drop
an unreferenced commit.

## The announcement

After the release, the 1bit engine Discord gets a new post in #announcements, written from
`changes.json` in the layout of Lemonade's release posts: what the week brings, the notable upstream
changes with credit to their authors, the rest as a list, and the downloads (the two packages
above). Then two lists written
straight from the data: **New models** (repos published under our Hugging Face org, models added to
Lemonade's catalog, architectures the registry gained or that gained a backend) and **Updates** (every
upstream as pinned in the release, with its latest release and changelog). It is the one weekly
digest; each week's post is new and earlier ones stay.
