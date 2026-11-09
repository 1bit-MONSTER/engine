#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""tools/xhci_keepawake_check.sh against fake sysfs trees, because CI is not an AXB35-02.

engine#823's port finding is board-specific, so the check that guards it must be testable on a machine that is *not* the board.
The script therefore takes `--sysfs-root`, and this file builds trees that exercise each verdict it can reach:

  * a machine the rule does not cover      -> "NOT THIS BOARD", exit 0  (and never a silent pass)
  * the board with three controllers asleep -> report exit 0, `--require-awake` exit 1
  * the board with all four awake           -> `--require-awake` exit 0
  * the board whose PCI ids the rule does not name -> "would bind nothing", exit 0

Each fake tree is only the files the script reads (dmi/id/board_name, bus/pci/devices/*/class|vendor|device|power_state|
power/control, and a driver symlink), so the test asserts the *verdict logic*, not the real hardware.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "xhci_keepawake_check.sh")

fails = 0


def check(cond, what):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def make_tree(root, board, controllers):
    """controllers: list of (slot, device_id, driver, power_state, control)"""
    os.makedirs(os.path.join(root, "class/dmi/id"), exist_ok=True)
    with open(os.path.join(root, "class/dmi/id/board_name"), "w") as fh:
        fh.write(board + "\n")
    with open(os.path.join(root, "class/dmi/id/bios_version"), "w") as fh:
        fh.write("1.09\n")
    with open(os.path.join(root, "class/dmi/id/product_name"), "w") as fh:
        fh.write("BeyondMax Series\n")
    devs = os.path.join(root, "bus/pci/devices")
    os.makedirs(devs, exist_ok=True)
    for slot, did, drv, ps, ctl in controllers:
        d = os.path.join(devs, slot)
        os.makedirs(os.path.join(d, "power"), exist_ok=True)
        for name, val in (("class", "0x0c0330"), ("vendor", "0x1022"), ("device", did), ("power_state", ps)):
            with open(os.path.join(d, name), "w") as fh:
                fh.write(val + "\n")
        with open(os.path.join(d, "power/control"), "w") as fh:
            fh.write(ctl + "\n")
        # a driver symlink is what the script reads to know it is bound to xhci_hcd
        drvdir = os.path.join(root, "drivers", drv)
        os.makedirs(drvdir, exist_ok=True)
        os.symlink(drvdir, os.path.join(d, "driver"))


def run(tree, *args):
    return subprocess.run([TOOL, "--sysfs-root", tree, *args], capture_output=True, text=True, timeout=120)


BOARD_CONTRL = [
    ("0000:c5:00.4", "0x1587", "xhci_hcd", "D3cold", "auto"),
    ("0000:c7:00.0", "0x1588", "xhci_hcd", "D0", "auto"),
    ("0000:c7:00.3", "0x1589", "xhci_hcd", "D3cold", "auto"),
    ("0000:c7:00.4", "0x158b", "xhci_hcd", "D3cold", "auto"),
]

tmp = tempfile.mkdtemp(prefix="xhci-check-test.")
try:
    # 1. not the board: the check must say so and still exit 0 (a non-board machine cannot conclude anything)
    t = os.path.join(tmp, "other")
    make_tree(t, "B650M", BOARD_CONTRL)
    r = run(t)
    check(r.returncode == 0, f"a non-AXB35 board exits 0 (rc={r.returncode})")
    check("NOT THIS BOARD" in r.stdout, "and says NOT THIS BOARD rather than reporting a verdict")
    r2 = run(t, "--require-awake")
    check(r2.returncode == 1, "--require-awake FAILS on a non-board machine: the precondition cannot be confirmed here")
    check("NOT a precondition met" in r2.stdout, "and says so rather than exiting quietly")

    # 1b. the board's DMI gate matches but NO function carries the rule's device ids: nothing would be kept awake, so a
    #     plug test would prove nothing - --require-awake must fail closed here too (the class this fix is about).
    t = os.path.join(tmp, "bind-nothing")
    make_tree(t, "AXB35-02", [("0000:c5:00.4", "0x9999", "xhci_hcd", "D0", "on")])   # a device id the rule does not name
    r = run(t)
    check(r.returncode == 0, f"report mode still exits 0 when the rule would bind nothing (rc={r.returncode})")
    check("would bind nothing" in r.stdout, "and says the rule would bind nothing")
    r2 = run(t, "--require-awake")
    check(r2.returncode == 1, "--require-awake FAILS when the rule would bind nothing (nothing is being kept awake)")
    check("NOT a precondition met" in r2.stdout, "and says so")

    # 2. the board, three asleep (what a stock Ubuntu boot looks like): report passes, --require-awake must not
    t = os.path.join(tmp, "asleep")
    make_tree(t, "AXB35-02", BOARD_CONTRL)
    r = run(t)
    check(r.returncode == 0, f"the board with three asleep reports cleanly (rc={r.returncode})")
    check("4 controller(s) match the rule; 3 of them are NOT in D0" in r.stdout,
          f"it counts 4 matched / 3 asleep: {[l for l in r.stdout.splitlines() if 'verdict' in l][:1]}")
    check("would test nothing" in r.stdout, "and says a plug test on this boot would test nothing")
    r = run(t, "--require-awake")
    check(r.returncode == 1, f"--require-awake fails when a controller is asleep (rc={r.returncode})")

    # 3. the board, all awake (what a 1bit-OS boot with the rule should look like)
    t = os.path.join(tmp, "awake")
    make_tree(t, "AXB35-02", [(s, d, drv, "D0", "on") for s, d, drv, _p, _c in BOARD_CONTRL])
    r = run(t, "--require-awake")
    check(r.returncode == 0, f"--require-awake passes when all applicable controllers are in D0 (rc={r.returncode})")
    check("all awake" in r.stdout, "and says a plug test on this boot would be valid")

    # 4. the board, but the PCI ids the rule names are not present (a BIOS that moved them)
    t = os.path.join(tmp, "otherids")
    make_tree(t, "AXB35-02", [("0000:aa:00.0", "0x0001", "xhci_hcd", "D0", "auto")])
    r = run(t)
    check(r.returncode == 0, f"an AXB35 with unknown xHCI ids exits 0 (rc={r.returncode})")
    check("would bind nothing" in r.stdout, "and reports that the rule would bind nothing instead of claiming success")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# 5. on THIS machine, whatever it is, the script must reach a verdict and exit 0 in report mode -- never crash, never hang.
r = subprocess.run([TOOL], capture_output=True, text=True, timeout=120)
check(r.returncode == 0, f"the real /sys run reaches a verdict (rc={r.returncode})")
check("verdict:" in r.stdout, "and prints one")

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
