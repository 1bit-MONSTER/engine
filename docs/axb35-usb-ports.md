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
# AXB35-02 (Bosgame BeyondMax): the rear USB-A and Type-C ports

**Status: verified on hardware 2026-11-14 (BIOS 1.10): every external port enumerates with the keep-awake rule; USB4 device untested.** engine#823.

| physical port | xHCI controller | measured with the controllers kept in D0 |
|---|---|---|
| front USB-A | `c7:00.0` (`1588`) | worked before the rule as well (the one controller that stays in D0) |
| front Type-C | `c7:00.3` (`1589`) | SuperSpeed 5 Gbps, USB storage |
| rear Type-C | `c7:00.4` (`158b`) | SuperSpeed 5 Gbps, USB storage |
| rear USB-A | `c7:00.0` (`1588`), port 4 | high-speed with a USB 2 stick; a USB 3 device on this port is untested |
| rear USB-A next to the power connector | `c5:00.4` (`1587`) | SuperSpeed 5 Gbps, USB storage |

How it was measured: on the board's Ubuntu host (kernel `7.4.0-rc1-1bit`), first with `power/control=on` written by hand on the
four xHCI functions, then with `91-1bit-xhci-keepawake.rules` installed in `/etc/udev/rules.d`. Both produce the same state, and
the rule's device list matches these four controllers exactly. Before the change, the three silent controllers had taken no
interrupts since boot, and every port read `RxDetect` with no connect. So the D3cold autosuspend is the whole cause for USB 2 and
USB 3 devices.

Not covered yet:

- a USB4/Thunderbolt device on a Type-C port (both USB4 routers still report no link);
- a USB 3 device on the `c7:00.0` rear USB-A port;
- a 1bit OS boot with the rule loaded from the image.

One enclosure (a Realtek RTL9210, the device from the original report) enumerates at high-speed (USB 2) on **every** port, including
ports where another drive reaches 5 Gbps. That limit belongs to the enclosure or its cable, not to the board.

On this board, stock boots leave three of the four xHCI functions in `D3cold`, so the rear USB-A ports and both Type-C ports never
enumerate; only the controller that carries the front USB-A (and Bluetooth, and the JetKVM) stays in `D0`. Windows drives every
port, the board's ACPI exposes no UCSI (`USBC000`/`PNP0CA0`), and `ec_su_axb35` has no USB controls — so a Windows-only port-power
path has always been the standing hypothesis, and the shipped answer is to keep the controllers awake rather than to find that
path.

## What ships

`os/v3/puppy/embed/etc/udev/rules.d/91-1bit-xhci-keepawake.rules`, gated so it can only ever bind on this machine:

```
ACTION=="add|bind", SUBSYSTEM=="pci", DRIVER=="xhci_hcd", ATTR{[dmi/id]board_name}=="AXB35-02",
ATTR{vendor}=="0x1022", ATTR{device}=="0x1587|0x1588|0x1589|0x158b", TEST=="power/control", ATTR{power/control}="on"
```

## How to tell whether it applies to the machine in front of you

```sh
tools/xhci_keepawake_check.sh                    # read-only: DMI match, per-controller state, verdict
tools/xhci_keepawake_check.sh --require-awake    # exit 1 unless every controller the rule matches is in D0
```

Read-only by default. `--apply` / `--revert` perform the rule's own single action (`power/control = on|auto`) and need root; they
exist so the mechanism can be checked without a reboot, and `--revert` is the documented way back. `--sysfs-root` checks a tree
other than `/sys`, which is how `tests/xhci_keepawake_check_test.py` exercises every verdict on a machine that is not the board.

The verdict is one of four, and only one of them is "go ahead and plug":

| verdict | meaning |
|---|---|
| `NOT THIS BOARD` | the DMI gate does not match; nothing about #823 can be concluded from this machine |
| `would bind nothing` | the board matches but no xHCI function carries the rule's device ids (a BIOS that moved them) |
| `N asleep` | the rule is absent (a non-1bit boot) or not applied; **a plug test here would test nothing** |
| `all awake` | a plug into a rear USB-A or a Type-C port is a valid test of #823 |

## Measured on the board (2026-11-08, kernel `7.4.0-rc1-1bit`)

```
== machine: board_name='AXB35-02'  bios='1.09'  product='BeyondMax Series'
function       vendor   device     driver   power     control
0000:c5:00.4   0x1022   0x1587     xhci_hcd D3cold    auto   <- the rule matches this function
0000:c7:00.0   0x1022   0x1588     xhci_hcd D0        auto   <- the rule matches this function
0000:c7:00.3   0x1022   0x1589     xhci_hcd D3cold    auto   <- the rule matches this function
0000:c7:00.4   0x1022   0x158b     xhci_hcd D3cold    auto   <- the rule matches this function
== verdict: 4 controller(s) match the rule; 3 of them are NOT in D0
```

That is a stock Ubuntu dev boot, i.e. *without* the rule: the premise of #823 still holds on the current kernel, so the fix is not
addressing something a newer kernel had already fixed. Forcing `power/control=on` on those three (exactly the rule's action) moves
them `D3cold → D0` and their root hubs' ports appear in `/sys/bus/usb/devices`; `--revert` puts them back. So the mechanism works
on this kernel; what has not happened is the plug.

## The two halves of this issue have different fixes, and the rule covers one

Measured on the board 2026-11-10, BIOS 1.10, a dev boot with the rule not installed:

```
0000:c5:00.4  0x1587  xhci_hcd     D3cold   <- rule matches
0000:c7:00.0  0x1588  xhci_hcd     D0       <- rule matches
0000:c7:00.3  0x1589  xhci_hcd     D3cold   <- rule matches
0000:c7:00.4  0x158b  xhci_hcd     D3cold   <- rule matches
0000:c7:00.5  0x158d  thunderbolt  D3cold   <- the rule does NOT match this
0000:c7:00.6  0x158e  thunderbolt  D3cold   <- the rule does NOT match this
```

*Corrected 2026-11-14 by the plug test above:* USB 2 and USB 3 devices on **both** Type-C ports run through xHCI functions the
rule covers (`c7:00.3`, `c7:00.4`); only a USB4/Thunderbolt tunnel uses the routers. The reading below was written before the plug,
and is kept as written.

The **rear USB-A** ports hang off the four xHCI functions the rule matches. The **Type-C** ports hang off the two USB4 host routers
(`0x158d`/`0x158e`), which the rule's device list does not contain, and which are in `D3cold` with the `thunderbolt` driver bound and
four live IRQs. So:

| half of engine#823 | what has to be awake | does the shipped rule cover it |
|---|---|---|
| rear USB-A | one of `0x1587/0x1588/0x1589/0x158b` | **yes** |
| Type-C | `0x158d`/`0x158e` (USB4 host routers) | **no** |

**A suspended USB4 host router with no partner attached is normal**, so this is not evidence that the routers are broken - it is evidence
that **one plug test cannot test both halves**: a plug into a rear USB-A exercises the rule, a plug into Type-C exercises a path no change
in this repository has touched. Record the router ids with the capture (`tools/xhci_boot_capture.sh` already records `power_state` per
controller; the two `thunderbolt` functions are worth reading the same way) so a "Type-C still dead" result is read as "the router half is
untested" rather than "the keepawake fix failed".

## Two things a 2026-11-10 audit added, both read-only (sysfs and source; no root, no plug)

`engine#823`'s thread carried the audit in full; the two facts that move the next attempt:

- **`ec_su_axb35` carries no USB control, in source.** The driver binds on this box (`refcount 0`) and its register map is fans only
  (`0x20/0x21/0x23/0x25` modes, `0x30/0x31/0x35…` tachometers). So "the EC path is not exposed to Linux" is narrower than it reads: the EC
  driver exists, and nothing in it addresses a Type-C or port-power register. A software fix is therefore not a matter of loading the right
  module, and a search for an EC-side switcher should start from `0x38` upwards, where nothing is known.
- **One silent controller had a stale resume in that boot**: `0000:c5:00.4` read `runtime_status=resuming` with `power_state=D3cold` while the
  other three xHCI functions were `active`/`D0`. A controller whose resume never completed cannot report a connect on its ports, so when a plug
  test is next run, read `runtime_status` as well as `power_state` — a `resuming` that persists is a different fault from an autosuspended
  controller, and the keep-awake write may be landing during a transition that never settles.

Also worth knowing for the same next attempt: the three silent controllers have taken **exactly 0 interrupts** since boot while the awake one has
taken **155,139**, and both USB4 host routers are up (`gen=4`, `authorized=1`, ~1.6k interrupts each) with no trained downstream link — so the
USB4 side is alive at PCI level and the missing thing is below it.

## What is still owed

*Updated 2026-11-14:* the plug test on a host with the rule passed on every external port (table at the top). What remains is
the same plug **on a 1bit-OS boot** with the rule from the image, a USB 3 device on `c7:00.0`, and a USB4/Thunderbolt device.
The paragraph below is the earlier plan, kept as written.

One device into a rear USB-A or a Type-C port **on a 1bit-OS boot**, checked with `--require-awake` first. If a port is then still
dead the next diagnostic is the one no one has run: compare `_DSM` on the xHCI/USB4 ACPI nodes against what Windows sends. Until
that plug, the honest row is *"fix merged and its mechanism verified on the board; functional confirmation pending"* — which is what
the release notes should say rather than leaving users to discover it.

## What a capture on BIOS 1.10 shows (measured 2026-11-08, after the owner's reflash)

The owner's 1.10 (11/04/2026) does not change the software-side picture, and one thing about it is worth
recording because it settles a hypothesis in engine#823:

```
bios-version:            1.10
bios-release-date:       11/04/2026
baseboard-product-name:  AXB35-02

0000:c5:00.4  control=auto  power_state=D3cold  runtime_status=suspended
0000:c7:00.0  control=auto  power_state=D0      runtime_status=active
0000:c7:00.3  control=auto  power_state=D3cold  runtime_status=suspended
0000:c7:00.4  control=auto  power_state=D3cold  runtime_status=suspended
```

**Three of four controllers still autosuspend to D3cold on 1.10** - the same shape engine#823 reported on
1.09. What is new is that the state is **software-controllable**, which the issue's hypothesis doubted:

```
$ for d in c5:00.4 c7:00.3 c7:00.4; do echo on > /sys/bus/pci/devices/0000:$d/power/control; done
   c5:00.4  D3cold -> D0      c7:00.3  D3cold -> D0      c7:00.4  D3cold -> D0
$ for d in c5:00.4 c7:00.3 c7:00.4; do echo auto > /sys/bus/pci/devices/0000:$d/power/control; done
   all three back to D3cold   (so the transition is reversible in both directions)
```

So the controller power gating is **not** behind a Windows-only driver or an EC/PD path that Linux cannot
reach: writing the same value the image's `91-1bit-xhci-keepawake.rules` writes moves the controllers to
D0. What that does **not** prove is that a device plugged into a woken controller enumerates - that is the
part only a plug can answer, and it is why the capture below reports per-port `connect` state rather than
controller power alone.

## `tools/xhci_boot_capture.sh` - one bundle from a boot, for the owner's next stick run

    tools/xhci_boot_capture.sh [outdir]      # root; writes /data/xhci-capture-<stamp> when /data is writable

It records the firmware next to the claim it supports (BIOS version/date, board), then per controller
`power/control`, `power_state` and `runtime_status`, then **per port `connect` or not** from `lsusb -v` on
each root hub - the distinction that matters, because "nothing is plugged in" and "plugged in and not
enumerated" are indistinguishable in every other view. It also keeps the udev rule's presence, the
interrupt counts, `lspci -vv`, the full and filtered `dmesg`, and the repo's own
`xhci_keepawake_check.sh` verdict when the stick carries it, then prints a one-screen `VERDICT.txt`.

Deliberate limits: it needs root (per-port status and PCI config space), it does **not** assume
`/sys/kernel/debug/usb` (this kernel does not expose `xhci/*/ports/*/portsc`), and it lists which of
`dmidecode`/`lsusb`/`lspci`/`dmesg`/`tar` the stick actually had - an empty section then means "no tool"
rather than "no problem". When it escalates through `sudo` it hands the bundle back to the invoking user,
because a capture that needs `sudo` to read is a capture that does not get sent.
