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

# The Linux kernel patches the 1bit kernel carries (not upstream)

The 1bit kernel is upstream `v7.4-rc1` (`config/kernel.config`) built with the config of record plus
`os/v3/puppy/kernel-puppy.fragment` (docs/kernel.md §1). The patches in this directory are applied on top
of the tag before `olddefconfig`, in order:

```sh
for p in <engine>/third_party/patches/linux/*.patch; do sed -n '/^diff --git /,$p' "$p" | git apply -; done
```

| # | patch | fixes |
|---|---|---|
| 0001 | `nvme-pci: add NVME_QUIRK_DMAPOOL_ALIGN_512 for the YMTC PC41Q` (1 file, +2/−0) | the SSD in the reference box (`1e49:1081`, DRAM-less) reads past a full 32-entry PRP list on writes; when the list is the last 256 B block of its pool page the read hits the next page and the IOMMU faults (engine#2478). The flag exists upstream (`nvme.h`, used for the Steam Deck O2 Micro and Micron 4100AT); the device was not in `nvme_id_table`, so until this the only fix was `nvme.quirks=1e49:1081:dmapool_align_512` on the command line (engine#3088) |

`os/v3/puppy/kernel-quirks-check.sh` reads a built kernel stage and refuses one whose `nvme` driver does
not list the device, and `mk-huge-kernel.sh` runs it, so an image cannot be packaged around a kernel that
lost the patch. The alias proves the table row exists, not its `driver_data`; the row's only purpose is
this flag, and the patch text is checked by `tests/kernel_quirks_test.sh`.

Other DRAM-less drives (engine#3088 item 3): the Lexar NM990 (`1d97:1806`, ryzen) was traced with the
same method and showed no over-read, so it has no row here. See the issue for the numbers.
