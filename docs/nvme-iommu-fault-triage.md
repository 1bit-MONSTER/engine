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

# Triaging an `IO_PAGE_FAULT` stream on a DRAM-less NVMe

*A runbook for the two tests in engine#2478 that need root on the host: 2361 `AMD-Vi: Event logged [IO_PAGE_FAULT]`
lines on the root SSD in the boot that died at 23:47:35, still accumulating at ~1.7/min. Everything below is what an
unprivileged lane could establish, plus the two commands that settle it.*

## What is already measured

```
tools/diag/iommu_fault_census.py --journal boot-2.txt --journal boot-1.txt --journal boot-0.txt

boot -2  0000:c4:00.0 domain=0x0015: 3800 faults, 10 distinct   x688 top
boot -1  0000:c4:00.0 domain=0x0015: 2361 faults,  9 distinct   x301 top
boot  0  0000:c4:00.0 domain=0x0015:  213 faults,  7 distinct   x81 top
         -> few live mappings faulting over and over -> a stale translation: find the buffer

boot -2  0000:c6:00.1 domain=0x0002:  390 faults, 321 distinct  max 6 each
         -> many distinct pages about once each -> re-allocation: find the race
```

Four facts that constrain the answer, each of which cost a measurement to learn:

1. **Split by device before counting.** A raw `journalctl -k -b N | grep -c IO_PAGE_FAULT` reported 4200 for one boot;
   the real NVMe figure was 3801 and the rest belonged to the NPU on a different PCI function. Every count in the
   original report mixed two unrelated faults. `distinct` is the tell: one device cannot have 4200 faults over 341
   addresses and also 3800 over 10.
2. **The stream is older than the report's table says.** Boot −6 (Nov 11 15:30) already has 112 faults, before the boot
   the table called the start. The window was the limit, not the fault.
3. **It is live, not historical.** 127 faults at 06:4xZ → 213 later the same morning, ~1.7/min on a box up three hours.
4. **The buffer has a name.** `nvme nvme0: allocated 64 MiB host memory buffer (1 segment)` on a DRAM-less YMTC PC41Q.
   An HMB is host DRAM the controller writes into on its own initiative, which is why the stream needs no driver action
   and never trips an NVMe timeout, reset or abort.

And the settings the kernel reports about itself:

```
iommu: Default domain type: Translated
iommu: DMA domain TLB invalidation policy: lazy mode     <- the default here, nothing asked for
PCI-DMA: Using software bounce buffering for IO (SWIOTLB)
software IO TLB: mapped [mem 0x5927e000-0x5d27e000] (64MB)
```

## Test 1 — disable the HMB (no reboot, decides whether the buffer is the HMB at all)

```bash
nvme get-feature /dev/nvme0 -f 0x0d -H          # Host Memory Buffer: current preference
nvme set-feature /dev/nvme0 -f 0x0d -v 0        # ask for no HMB
journalctl -k -f | grep -c IO_PAGE_FAULT        # watch one burst interval (~2-5 min under load)
```

**Prediction:** the `c4:00.0` stream stops within one burst interval.
**If it stops** the faulting region is the HMB, and the remaining question is only why its mapping goes stale.
**If it continues** the HMB is a bystander and the faulting region is the SWIOTLB bounce area or another map — go to
test 2 with that expectation written down first.

## Test 2 — strict IOMMU invalidation (one reboot, separates "stale" from "unmapped")

```bash
# verify the parameter against the kernel that will parse it, not from memory:
/usr/src/linux-headers-$(uname -r)/scripts/extract-vmlinux /boot/vmlinuz-$(uname -r) > /tmp/vmlinux
strings /tmp/vmlinux | grep -c 'iommu.strict'     # 3 on 7.4.0-rc1-1bit
strings /tmp/vmlinux | grep -c 'amd_iommu=strict' # 0 -- this spelling does NOT exist here
```

Add to the kernel command line:

```
iommu.strict=1
```

**Prediction:** the `c4:00.0`/`0x0015` stream goes to **zero for a full boot**; the NPU `c6:00.1` churn is unchanged.
**If it goes to zero** the mapping is *stale but still present* in a lazily-invalidated TLB for a **live** HMB — the fix
belongs in how the HMB mapping interacts with lazy invalidation, not in the NVMe driver.
**If it survives strict mode** the HMB pages are genuinely unmapped while the controller still writes them: a
teardown/re-establish ordering bug on the NVMe side, and the next step is tracing that ordering, not another boot.

## Two traps this runbook exists to stop

* **A count without a device split is not a measurement.** `distinct` exposes the mixing.
* **A flag is not a fact until you check the binary that parses it.** The first `strings` attempt here ran against the
  *compressed* `vmlinuz` and returned 0 for every pattern, including ones that exist — which would have "proved" the
  parameter absent. Decompress first (`extract-vmlinux`), then count.
