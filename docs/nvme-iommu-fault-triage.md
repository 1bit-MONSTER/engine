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
         -> few live mappings faulting over and over
         -> (11-16) NOT a stale translation: the probe below shows those addresses are never unmapped

boot -2  0000:c6:00.1 domain=0x0002:  390 faults, 321 distinct  max 6 each
         -> many distinct pages about once each -> re-allocation: find the race
```

**The one hypothesis this file got wrong, and how it was retired.** The shape above ("few live mappings faulting over and
over") reads as a stale translation — a released buffer still being written, with lazy invalidation deferring the flush. That
was the working hypothesis for a day. It is false for this stream, and a continuous probe settled it:

```
armed on iommu_unmap_fast filtered to ALL eight addresses that faulted in the current boot
window  ~13 minutes, with a real burst underneath it
faults  30 inside the window, all 30 on filtered addresses, 0 outside
unmaps  of any of those eight addresses:  0        (fault totals 138 -> 168, unmap_events stayed 0)
```

**The IOVA is never unmapped while the device writes it.** No released buffer, no stale TLB entry, nothing for lazy
invalidation to defer: the translation is simply absent when the write arrives. That moves the whole question from the
*unmap* path to the **map** path — the address was never mapped in that context, or it is mapped with insufficient rights
(`flags` bit 0 clear argues absent rather than denied), or the device is using an address a descriptor should not carry.

Two method notes from getting this far, both now sections below: **a fixed capture window will keep missing seconds-long
bursts** (three short windows missed three bursts, the last by seven seconds — the probe has to be armed continuously across
the whole address set), and **the phenomenon must be live before the capture starts** (step 0).

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

**The interval is seconds, not milliseconds (measured 2026-11-16).** A filtered probe on one faulting IOVA
(`0xe1e18000`), captured over a boot's first 28 minutes, found **60 unmaps of that page, every one a single 4 KiB
page, 59 of them from `<idle>` (interrupt context)**: the NVMe completion path repeatedly maps and tears down one
pooled page. **Three of the four faults on it** followed an unmap of that same page by **5.65 s, 24.66 s and
24.71 s**, and a correlation window of ±5 ms found none of them. The fourth (boot t=1605.6 s) has **no unmap of that
page within 60 s** -- the nearest is **+74 s**, after the fault -- so this capture correlates three faults and shows
the ceiling on the fourth; a wider window does not help it, and the earlier "19.60 s" I quoted for it came from a
different capture and does not reproduce on this one. So `tools/diag/unmap_fault_order.py` defaults to **30 s** and
the page's own numbers are **5-25 s on that sparse stream**: a millisecond window reports "nothing correlated" on a
capture that contains the correlation. Because the same sentence is printed whether the window is too narrow or the
fault genuinely has no nearby unmap, **the negative names the window it used and the measured interval** -- a default
alone does not fix the failure, only a negative that says *at what scale* it found nothing does.

**But "seconds" is not the rate of this stream -- it is the rate of that capture.** The same page measured again on a
busier boot is mapped and unmapped **about 470 times a second**: `nvme_prep_rq` maps it through `dma_map_phys` ->
`iommu_map`, and `nvme_pci_complete_batch` unmaps it from the completion IRQ through both `dma_unmap_phys` and
`dma_iova_unlink`/`dma_iova_destroy` (150 s, one IOVA, filtered, counts 27644 / 35907 / 33141). **So a wide window is
the wrong instrument in the other direction**: at that rate a +/-30 s window holds ~14,000 unmaps of the page and "the
nearest one was 31 us before the fault" is not evidence of anything. The tool therefore **carries the count** and flags
a pairing whose window held more than three unmaps of the same IOVA:
`[1 of N unmaps of this iova in the window -- NOT discriminating, narrow --window]`. On the burst capture above it
fires on the -5.65 s pairing immediately (`1 of 7`), which is the honest reading: **the interval is real, its precision
is not.**

**Three ways a filtered kprobe silently measures nothing**, all hit while taking the counts above:

1. **`tracing_on=0` with `enable=1`** records nothing, and the empty buffer reads exactly like "0 events". Check
   `tracing_on` and `entries-written` on every capture, and run a **positive control** first (30 s of events, abort if
   zero).
2. **A `stacktrace` trigger suppresses the event's own line.** The dump is `<stack trace>` blocks with no `iova=` text
   at all, so a filter that is working perfectly looks like it never applied. Count events *without* the trigger first.
   And `stacktrace:unlimited` must be cleared (`echo '!stacktrace:0' > .../trigger`) before `-:kprobes/x` will delete
   the event -- otherwise the delete fails with `echo: I/O error` and the probe stays armed.
3. **tracefs rejects range filters, and the failed write leaves the probe UNFILTERED.**
   `(iova + size > 0xX && iova <= 0xX)` gives `parse_error: Invalid operator`; the event then records **every** unmap
   (1,947 in 20 s, against 18 in 30 s for the working exact filter). **Read the filter file back after writing it**, and
   keep in mind that an exact-iova filter cannot see a *larger* mapping that covers the page -- for this driver that is
   tolerable because its unmaps are all 4 KiB (60 of 60 measured), but it is a measured caveat, not a proof.

**Arm the map side too, unfiltered, and the argument for one address stops being needed (measured 11-16 06:42Z).**
A filtered capture on five addresses chosen from the fault census recorded **zero events in 30 s** and aborted on its
own positive control -- because **the fault target moves**: the last three faults were all on an address the list did
not contain. The capture that worked armed `iommu_map` and `iommu_unmap_fast` with **no filter** for 13 minutes
(44 unmaps/s is what a quiet minute looks like; a busy one is ~1,000/s, and the per-CPU ring still covered all 795 s).
Of the 8 faults in that window:

```
0xebf50000  x4 faults   NO map or unmap of this iova anywhere in the 795 s
0x6925f000  x1          nearest map +230 ms AFTER the fault
0xe407f000  x2          nearest map/unmap -58 s
0xe1d00000  x1          nearest -347 s
```

**Four of eight faults hit an IOVA that was never mapped and never unmapped at any point in 795 seconds of a capture
holding 252,704 map/unmap events** -- so the translation is not stale, it is **absent, and nothing re-creates it**. That
moves the question to the descriptor: the device writes to an address the driver has already finished with, which is a
DMA-mapping lifetime bug, not a TLB or lazy-invalidation one. `0x6925f000` is the exception and worth separating: it is
**re-mapped 230 ms after** the fault, the profile of a live pool page faulting before its re-map.

The kprobe recipes, corrected against the kernel's own argument order -- the first one was wrong in this page's tool:

```
iommu_map(domain, iova, paddr, size, prot)        ->  iova=%si size=%cx    # %dx is PADDR, and a probe
                                                                          # using it printed size=0x138ec90000
iommu_unmap_fast(domain, iova, size, ...)        ->  iova=%si size=%dx
```

**A task name can contain a space** (`next-server (v1-8386`), so a regex anchored with `^\s+\S+\s+\[` silently dropped
**2,176 of 115,941 map lines** -- 1.9 %, which is exactly the fraction that can contain the event being looked for.
Anchor on the timestamp instead, and reconcile the parse count against `grep -c` before believing a negative.

**And the reverse is just as wrong: a positive must say whether it discriminates.** The same page's cadence is
workload-dependent -- a later capture on this box (05:40Z, 150 s, one IOVA filtered) found the NVMe paths mapping it
from `nvme_prep_rq` through `dma_map_phys` and unmapping it from `nvme_pci_complete_batch` through
`iommu_unmap_fast`, **about 470 times a second**. At that rate a ±30 s window holds ~14,000 unmaps of the page, so
"the nearest one was 31 us before the fault" is not evidence of anything. **The tool now carries the count**: a
pairing whose window held more than three unmaps of the same IOVA is printed with
`[1 of N unmaps of this iova in the window -- NOT discriminating, narrow --window]`. On the burst capture above that
fires immediately -- the -5.65 s pairing is `1 of 7` -- which is the honest reading of it.

**One probe owns the shared buffer.** A kprobe on `iommu_unmap_fast` with **no filter** wrote 579,000 events and left
a filtered probe on the same function with **zero** -- the trace ring buffer is shared, so the *narrow* probe (one
IOVA, one call site) is the one that should be armed, and checking `grep -c '<probe>:' trace` before trusting a
capture is the same "prove the plumbing" rule two paragraphs up.


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

### Step 0, which this page learned the hard way twice: prove the *phenomenon* is live

Validating the instrument says nothing about whether there is anything to observe. Twice in one night a capture was launched
correctly and measured nothing, because the stream had already stopped:

* five reproduction attempts (bulk read, random read, concurrency, direct write, a full build) all ran hours after the last
  fault — they answered "does this load restart a stopped stream", not "does this load trigger it";
* an 8-minute address-lifetime kprobe on the top faulting address opened **60 seconds after** the storm's last fault, so it
  measured an idle system and its "0 unmaps" meant nothing.

One line prevents both:

```bash
n=$(journalctl -k -b 0 --since '-5 min' | grep -c IO_PAGE_FAULT)
[ "$n" -gt 0 ] || { echo "nothing is happening: $n faults in 5 min -- a capture now can only prove silence"; exit 1; }
```

Launch only when `$n` is non-zero, and record `$n` beside the result: a capture taken at 3 faults/minute and one taken at
zero are not the same experiment, and a table that does not say which is which reads as stronger than it is.

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

## Pin the build you measured on

Every number in this page came from one host, and on 11-15 that host was **cold-rebooted onto a different NPU driver**:

```
before:  amdxdna srcversion F87D0BDF248067862F83BA3     <- all 740 faults, all six ruled-out workloads
after:    amdxdna srcversion E48F9326AC573D2E06BC926     <- production, 0 IO_PAGE_FAULT so far
```

So the observations here describe a build, not a machine. Record the build with the observation, or the next reader will
compare two different kernels and call the difference a finding:

```bash
modinfo -F srcversion amdxdna          # driver build
uname -r                               # kernel, which also carries the IOMMU side
uptime -p                              # boot age: the allocator-history candidate resets here
```

`srcversion` is the right field rather than `version`, because the module reports a vendor tag (`version: 0.1`) that did
not change across that swap while the code underneath did.

Three consequences worth stating explicitly, all of which this page got wrong once:

* the six ruled-out workloads (bulk read, random read, concurrency, direct write, mild memory pressure, a full build) stand
  as evidence about the **old** build — they are not yet evidence about production, and a quiet window on the new driver is
  not an exoneration of them;
* the address walk, the flags histogram and the burst shape are **old-build** facts until a burst on the new build shows the
  same shape;
* a reboot **resets the uptime candidate to zero**, so "nothing faulted for 24 h" right after a reboot is expected under
  more than one hypothesis and proves less than it looks.

## Reproduction status: five shapes tried while the stream was already stopped

Recorded before the theory, because the first thing anyone will try is to make the stream happen again. **But read the
"before" column first** — it is the difference between this table being evidence and being five ways of saying nothing.

```
last NVMe fault this boot: 08:53:17        the five attempts ran from ~16:00 onward
faults in the 10 minutes before each attempt: 0, 0, 0, 0, 0
```

| attempt | load | faults in the 10 min before | faults during | what it shows |
|---|---|---|---|---|
| sequential direct reads | 12 GiB, 4 MiB blocks, 3.9 GB/s | 0 | **0** | does not restart a stopped stream |
| random direct reads | 400 × 64 KiB at random offsets | 0 | **0** | same |
| concurrent direct reads | 24 threads, ~3.8 GB | 0 | **0** | same |
| direct writes + fsync | 1 GiB written, read back, unlinked | 0 | **0** | same |
| memory pressure + reads | 8 GiB held/touched during 30 direct reads | 0 | **0** | same |

**So the honest summary is narrower than "reproduction ruled out": five shapes were tried while the stream was already
stopped, and none of them restarted it.** That is worth knowing — a stopped stream is not restarted by load, and the
obvious ways to provoke one do not work — but it does **not** show that those loads cannot *trigger* the stream when it is
running, because none of them was ever run against a live one. An earlier version of this section said "four shapes tried,
none reproduced it", which claimed more than the data supports; the fifth row made the pattern obvious but did not cause it.

Reproducing it therefore needs a window when the stream is live, which means either waiting for the box to produce one
(it happened during other lanes' builds) or running the same shapes *while faults are arriving* and recording the rate
before and during. If a run starts from a quiet stream, it can only report what these five did: no restart.

What the evidence still leaves as candidate *triggers*, unchanged by the five zeros:

1. **memory pressure far heavier than 8 GiB** — a `cmake` build on a 122 GiB box applies more, and the bursts coincided
   with builds. The mechanism it would reach is real: the allocator hands the same IOVA straight back (494 reuses of one
   address in 15 minutes), and a fault needs that hand-back to happen while a command is still outstanding;
2. **uptime** — allocator history may be the precondition, which no short test creates;
3. **a caller outside the block path** — a minority of traced unmaps arrived via `nvme_unmap_data -> dma_iova_unlink ->
   dma_iova_destroy`, i.e. teardown rather than completion.

One measurement note the fifth test produced, which applies to any experiment on this host: `MemAvailable` moved from 58 to
88 GiB **during my own thirty-second window**, purely from other work. A memory-pressure experiment on a shared box has to
record the baseline either side of the window, or it cannot separate its own effect from a neighbour's.

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
   exists for exactly this. To measure the **rate** rather than take one census of a buffer that both throttles and
   rotates, `tools/diag/iommu_fault_sampler.py` appends `<epoch> <printed> <suppressed_total>` per interval: the
   dropped-callback delta is the only count `printk_ratelimited` cannot hide. Both counts are cumulative over the
   lines each snapshot adds (by `dmesg`'s `[ s.us ]` stamp), so the buffer rotating cannot hide or shrink an
   interval (engine#2542 quoted three different populations for one boot, 20 / 78 / 41, all true when read).
2. **`flags` is not decoded here, and must not be until it is cited.** The streams carry **three** values, not one:

   ```
   NVMe  (c4:00.0, 0x0015):  717 × flags=0x0000    3 × flags=0x0050
   NPU   (c6:00.1, 0x0003):   20 × flags=0x0007
   ```

   An earlier version of this page decoded `0x0007` as "present=0, write=1, user=1 — a write to a non-present page"
   and then said `0x0000` "means the same for a read". **Both cannot be true**: one bit is not set and clear at once
   while both are described as the same state. A later version demanded a spec citation while asserting the bit ordering
   in a parenthetical — the same error wearing the rule that forbids it. The parenthetical is gone.

   What is observable, and all this page will claim: the two devices are in **different states** (`0x0007` vs `0x0000`),
   and the NVMe stream additionally has rare `0x0050` lines (3 of them, at one of the four addresses the census tracks).
   **Any bit ordering must explain all three values**, so a decode that explains `0x0000` and `0x0007` and stays silent
   about `0x0050` is incomplete by construction. The meaning of the bits, and therefore which state is "mapped but not
   writable" versus "not mapped", comes from the **AMD IOMMU specification's event-log chapter** — document, revision and
   table — and the kernel does not provide it: its own format string is `flags=0x%04x` with no expansion.

   The stakes are the diagnosis, in opposite directions: if the NPU's bit 0 is set, the page **is** mapped and the fault
   is a **rights** failure; if the NVMe's bit 0 is clear, its stream is an **absent** translation. The *walk* evidence
   below is unaffected either way.

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
