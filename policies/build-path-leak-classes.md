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
# Build-path leaks: the four classes, and the guard each one needs

An artefact that ships its builder is a small disclosure (a username, a working directory, a build-tree
name) and a large reproducibility problem: two builds of the same commit do not produce the same bytes.
By 2026-11-08 the fleet had fixed the same class five times in five artefacts, each time re-deriving the
taxonomy. It is written down here so the next one starts from the list.

The classes are distinguished by **where the path lives**, because that decides the remedy — and because
a guard that reads one of them cannot see another.

| # | where the path is | measured example | remedy |
|---|---|---|---|
| 1 | **baked into a binary** by `__FILE__`, `-g`, or a compiled-in absolute path | `1bit` carried `ONEBIT_HRX_LIBHSA=/home/<builder>/…/libhsa-runtime64.so.1` (engine#1459); `lemond` and `lemonade` carried 126 + 13 paths from `__FILE__` (engine#1461) | a prefix map (`-ffile-prefix-map`, `-fdebug-prefix-map`) at build time; nothing compiled in that the build cannot rewrite |
| 2 | **a build tree that was packaged** | `scripts/mod/devicetable-offsets.s` and friends in a kernel headers `.deb`: 449 absolute paths, the difference between two builds of the same package being exactly 449 lines × 30 B (engine#1487) | do not package the tree; a header package ships the `.c`, not the compiler's by-products |
| 3 | **a symbol table** or a comment in generated text | the same headers package, before it was pruned; `strings` on any Release binary built with debug info | strip, or map, and **assert the result** rather than assuming `-s` was passed |
| 4 | **the filesystem layout itself** | a mini (initramfs) image shipped `/bin/1bit -> /home/<builder>/<workdir>/<build>/1bit`, with the 4.7 MB binary stored at that path: the initrd disclosed the builder's username, working directory and build-directory name (engine#1543) | place the artefact at a **stable** path with `$ORIGIN`-relative libraries — the layout *is* the leak, so there is nothing to map |

Two lessons that cost time:

- **"The binary needs its build path so its RUNPATH resolves" is a claim to check, not a reason.** On the
  c26 launcher it was false: `readelf -d` showed **no** `RUNPATH`/`RPATH` at all — it resolves its `NEEDED`
  libraries from the standard loader paths, which the image already stages. Check `readelf -d` before
  accepting the constraint; the payload and the installed package both use `$ORIGIN` and need no build path.
- **A guard that reads contents cannot see a layout.** `build_path_strings` reads the bytes of packed ELFs
  and generated text, so class 4 is invisible to it (engine#1519 widened it to generated text and still
  could not see a symlink target). Class 4 needs a row that walks the image's own tree.

## The guards, and where each lives

| class | guard |
|---|---|
| 1, 2, 3 (contents) | `scripts/pack-check.sh`'s `build_path_strings` — warning for the shapes a prefix map can fix (`*.s`, `*.mod.c`, `Module.symvers`, `modules.order`) and a **failure** for the shapes it cannot (`*.cmd`, `*.d`: a red there means a build tree was packaged) |
| 4 (layout) | `os/mini/release-check.sh`'s `image-builder-paths` row — a mini image ships no users, so any `/home/*` path in its root is the builder's |

Both read the **artefact**, not the build tree: a guard that inspects what the build machine looks like
answers a different question from the one that matters.

## Payload permissions are the neighbour of this class, not part of it

`group/other-writable` modes in a payload (engine#1352) came from `cp -a` preserving a build tree's modes;
it is not a path leak, and it is fixed where it is measured — `pack_normalise_modes` in
`scripts/pack-elf.sh` for the packers, and the `payload-modes` / base-layer rows for the images. Recorded
here only so the two are not confused: one is "who built this", the other is "who may write to it".
