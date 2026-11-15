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

**No `nvme-cli` needed, and none is installed here** (`command -v nvme` → nothing). The kernel exposes the same switch
in sysfs, so this is one write and no package. Measured on this host:

```bash
$ cat /sys/class/nvme/nvme0/hmb
1                       # HMB is on; 0 means off
$ ls -l /sys/class/nvme/nvme0/hmb
-rw-r--r-- 1 root root 4096 …   # readable as a user, writable only by root
```

```bash
echo 0 | sudo tee /sys/class/nvme/nvme0/hmb   # ask the controller for no HMB
cat /sys/class/nvme/nvme0/hmb                 # must read 0 before you trust the test
journalctl -k -f | grep -c IO_PAGE_FAULT      # watch one burst interval (~2-5 min under load)
echo 1 | sudo tee /sys/class/nvme/nvme0/hmb   # put it back
```

If a firmware refuses the sysfs write, the `nvme-cli` equivalent is `nvme set-feature /dev/nvme0 -f 0x0d -v 0`
(feature `0x0d`, Host Memory Buffer).

**Prediction:** the `c4:00.0` stream stops within one burst interval.
**Result (11-15 10:2xZ, run and logged on the issue): it continued** — 8 faults in 4 min with `hmb` verified `0`, against a
2.3/min baseline, and 7 in 4 min after restoring it. **The HMB is a bystander.** That test cost four minutes and closed a
hypothesis that had been argued for twenty; run it first on any successor question, because it is the cheapest one.

## Test 2 — trace the ordering, **not** a boot flag

This section used to prescribe `iommu.strict=1` and predict "zero faults for a full boot". **That prediction was wrong**,
and the measurement that broke it is worth more than the boot would have been:

```
# kprobe on iommu_unmap_fast, filtered to the six IOVAs that fault, with stacktrace on
<idle>-0  [024] d.h1. 25466.371055: fastunmap: dom=0xffff8c75848ade00 iova=0xeba66000 size=0x1000
<stack trace>
 => iommu_unmap_fast
 => __iommu_dma_unmap
 => iommu_dma_unmap_phys
 => dma_unmap_phys
 => nvme_pci_complete_batch
 => nvme_irq
```

88 such unmaps in 45 s, all on addresses that fault, from the **NVMe driver's hard-IRQ completion path**; a further 20
come through `nvme_unmap_data -> dma_iova_unlink -> dma_iova_destroy` on teardown.

`iommu_unmap_fast` drops the **page-table** entry immediately and defers only the **TLB invalidation**, so a page-table
miss faults whether or not the flush was eager. Strict mode changes when stale TLB entries are dropped; it does not stop
a device writing an IOVA the tables no longer map. Expect `iommu.strict=1` to change nothing about this stream — run it
only if you want to confirm that, not as a discriminator.

**What to measure instead**, on one address, both events timestamped:

```bash
# 1. pick the top faulting IOVA from the census
# 2. kprobe iommu_unmap_fast on that iova alone (filter), stacktrace on, capture for minutes in the background
# 3. in the same capture window, note the fault timestamps for that iova
```

The question the ordering answers: does a fault follow an unmap of the *same* address, or does it arrive while the
address is still mapped? The first is a write-after-unmap — the device dirtying a buffer the CPU released, or the driver
reusing the IOVA for the next request before the device acknowledged the previous one, both NVMe-side ordering questions
on a DRAM-less controller under load. The second would mean the mapping was never there, which the unmap stream above
already argues against.

Two things to be careful about, both learned the hard way here:

* **validate the probe before believing a silence.** A first capture reported "0 unmaps in 12 min" and was worthless: the
  control showed the probe never fired, because `echo > filter` does not clear a kprobe filter (it fails to parse, and the
  old filter stays). Use `echo 0 > .../filter` and read it back.
* **pick the right symbol.** `iommu_unmap` had zero hits in 5 s on a box doing heavy I/O; `iommu_unmap_fast` had 40,510.
  Probe a known-hot function (`ksys_write`) once to prove the plumbing, then argue about symbols.

## Validating a capture before you believe it

Four failures in one afternoon on this row, and every one was method rather than kernel. The pattern that catches all
four costs about ten lines:

All of this needs root on the tracing filesystem; run the block under `sudo -s`.

```bash
# 1. the probe exists and fires -- verify with a known-hot symbol the first time you use a new tracefs
echo "p:hot ksys_write" >> /sys/kernel/tracing/kprobe_events
echo 1 > /sys/kernel/tracing/events/kprobes/hot/enable; sleep 3
grep -c 'hot:' /sys/kernel/tracing/trace     # must be > 0 before you trust any other probe
# remove THIS probe only: disable it, then delete it by name. `echo > kprobe_events` would truncate the file and
# remove every lane's probe definitions on a shared box -- and a still-enabled probe cannot be removed at all.
echo 0 > /sys/kernel/tracing/events/kprobes/hot/enable
echo '-:hot' >> /sys/kernel/tracing/kprobe_events

# 2. pre-flight INSIDE the long run: abort instead of sleeping on a dead instrument
echo 1 > /sys/kernel/tracing/events/kprobes/unm/enable; sleep 5
H=$(grep -c 'unm:' /sys/kernel/tracing/trace)
[ "$H" -gt 0 ] || { echo "INVALID: probes silent, aborting"; exit 1; }

# 3. copy the output, then make sure it is readable
sudo cp /sys/kernel/tracing/trace ./cap.txt && sudo chmod 644 ./cap.txt

# 4. read it back before analysing: an unreadable file greps as "no events"
#    `grep -c` exits 1 on ZERO MATCHES and 2 on an error, so `|| echo FAILED` reports a legitimate empty capture as a
#    permissions problem -- the exact confusion this section exists to prevent. Separate the count from the error.
n=$(grep -c 'unm:' ./cap.txt); rc=$?; [ "$rc" -le 1 ] || echo "GREP FAILED (rc $rc): check permissions, not the data"
echo "events: $n"      # zero is a result; rc 2 is a broken read
```

The four ways this went wrong here, in the order they happened:

| mistake | what it looked like | what it cost |
|---|---|---|
| `echo > .../filter` to clear a filter | `event filter parse error`; the old filter stayed and the probe matched nothing | a 12-minute capture reported "0 unmaps" |
| probing `iommu_unmap` because that is the name you remember | 0 hits in 5 s on a box doing heavy I/O | a wrong conclusion about which path unmaps take |
| enabling a probe whose definition a previous cleanup had removed | silence, reported as "no events" | a 20-minute capture measured nothing |
| `sudo cp` output, then `grep 2>/dev/null` | silence, reported as "no events" | the data was in the file; only the read failed |
| `grep -c ... || echo FAILED` on a legitimately empty file | "FAILED", though the read was fine | a real zero called a permissions problem |
| `echo > kprobe_events` to clean up | every lane's probe definitions gone (and a still-enabled probe cannot be removed) | other people's instruments, silently |

Every one of them produces the same symptom — **an empty result** — which is indistinguishable from the finding you
might legitimately be looking for. That is why the pre-flight is not optional: it is the only thing that separates
"nothing happened" from "nothing was watching".

## The same method on a second device (the NPU)

The title says NVMe because that is where the fault stream was found, but nothing in the method is NVMe-specific:
engine#2542 is the same instrument applied to the NPU (`0000:c6:00.1`, domain `0x0003`) on the same host, and it
produced a diagnosis in two measurements without touching the device.

```
20 events this boot, two bursts: 10 @ 08:43, 10 @ 08:53
walk: 0x4128000 0x4129000 … 0x4130000   (9 contiguous pages)
      0x5128000                          (1 page, 16 MiB above)
      0x4128000 …                        (restarts)
flags=0x0007 on all 20
```

Four things generalise, and each one was a decision rather than a discovery:

1. **Split by `(device, domain)` before counting.** The NVMe stream is domain `0x0015`, the NPU's is `0x0003`; a raw
   `grep -c IO_PAGE_FAULT` mixes them and inflates whichever you were looking for. `tools/diag/iommu_fault_census.py`
   exists for exactly this.
2. **Decode `flags` — from the spec, not from memory.** The two streams carry **different** flags and that difference is
   the whole point:

   ```
   NPU  (c6:00.1, 0x0003):  flags=0x0007
   NVMe (c4:00.0, 0x0015):  flags=0x0000
   ```

   An earlier version of this page decoded `0x0007` as "present=0, write=1, user=1 — a write to a non-present page" and
   then said `0x0000` "means the same for a read". **Both cannot be true**: a single bit cannot be set in one stream and
   clear in the other while both are the same state. The kernel does not expand these bits — it prints `flags=0x…` raw —
   and the local headers do not carry them either, so the decode has to come from the **AMD IOMMU specification's event-log
   table** (bit 0 present, bit 1 write, bit 2 user, in the spec's ordering) and be cited next to the claim.

   The stakes are the diagnosis, in opposite directions: if the NPU's bit 0 is set, the page **is** mapped and the fault is
   a **rights** failure (a write to memory that exists but is not writable) rather than a walk off the end of a mapping;
   if the NVMe's bit 0 is clear, its stream is an **absent** translation. So the flags must be resolved before either
   engine#2478 or engine#2542 is described as one kind or the other — the *walk* evidence is unaffected either way.
3. **Read the walk, not the count.** Nine contiguous pages followed by one page far above is *two ranges in one stride*
   — a descriptor list whose tenth entry lives elsewhere — not a scatter of unrelated pointers. The stride tells you
   which structure to look for; the count tells you nothing.
4. **Eliminate the spaces a number could belong to.** `Unsupported op 18` is outside the DRM ioctl ids (0–9) and the
   command types (0–2) in the shipped UAPI, so it is an opcode in the command stream, and the engine repo cannot name it
   either (its NPU path calls `xrt::run`; there is no opcode table in-tree). Three candidates eliminated in one pass.

The pre-flight pattern above is what made the NPU run cheap: the live probe on those ten IOVAs found nothing in 8 s,
which correctly said "these buffers are idle now" instead of being mistaken for the absence of unmaps. Hours-old bursts
are not observable live — that is a property of the instrument, and knowing it saved a twenty-minute capture that could
only have reported "idle".

## Two traps this runbook exists to stop

* **A count without a device split is not a measurement.** `distinct` exposes the mixing.
* **A command is not a fact until you check the box has it, and the interface is often better than the tool.**
  This page's first draft opened with `nvme get-feature`; `nvme-cli` is not installed here, and the switch it fronts is
  a root-writable sysfs file — so the test became one write instead of an apt step. Found by applying the page's rule
  to the page.
* **A flag is not a fact until you check the binary that parses it.** The first `strings` attempt here ran against the
  *compressed* `vmlinuz` and returned 0 for every pattern, including ones that exist — which would have "proved" the
  parameter absent. Decompress first (`extract-vmlinux`), then count.
