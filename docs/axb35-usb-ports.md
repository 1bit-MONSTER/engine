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

**Status: fix merged, mechanism verified on the board, functional plug test still outstanding.** engine#823.

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

## What is still owed

One device into a rear USB-A or a Type-C port **on a 1bit-OS boot**, checked with `--require-awake` first. If a port is then still
dead the next diagnostic is the one no one has run: compare `_DSM` on the xHCI/USB4 ACPI nodes against what Windows sends. Until
that plug, the honest row is *"fix merged and its mechanism verified on the board; functional confirmation pending"* — which is what
the release notes should say rather than leaving users to discover it.
