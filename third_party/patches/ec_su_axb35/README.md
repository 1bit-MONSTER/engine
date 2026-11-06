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

# The ec_su_axb35 patches this fleet's packages apply (not upstream)

`os/licenses/ec_su_axb35/src` is `cmetz/ec-su_axb35-linux` at `f62c2c22`, verbatim, and it stays that
way. The dkms package applies the patches in this directory on top of it when it builds, so the
licensed source in the image and the patched source in the package are both visible for what they are.

| # | patch | fixes |
|---|---|---|
| 0001 | `ec_su_axb35: attach only on an AXB35 board` (2 files, +17/−0) | the driver writes fixed EC registers (0x21/0x23/0x25 fan modes, 0x35-0x38 tachometers) that are the AXB35-02's; on any other board those addresses belong to a different controller. Reads `dmi_get_system_info(DMI_BOARD_NAME)` in `module_init` and returns 0 without touching anything unless the board name contains `AXB35` — 0 rather than `-ENODEV` because the package drops a `modules-load.d` entry, and a loaded-but-idle module is a quieter outcome than a failed unit at every boot on a machine that is not this board |

The board name is `AXB35-02` on the box this was written for (`/sys/class/dmi/id/board_name`,
`sys_vendor` `BOSGAME LIMITED`). Nothing here is loaded or measured on the dev box by the packaging
lane: the package's proof is a build against a container kernel plus the gate string in the built
module (`scripts/pkg-proof.sh`), and the running module is the image's, installed by `os/mini/mkimage.sh`.
