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
"""Run the fork's recorded-stale expectation check from the engine's suite (engine#2836).

The fork registers 48 ctest rows and **no engine job runs any of them**: the engine never builds the
fork's tests, so a row's presence in `third_party/llama.cpp/tests/CMakeLists.txt` means it runs only
when a person types `ctest` in a fork checkout. Most of those rows need a device and are right to be
absent; `check_stale_expectations.py` (fork #419) is not — it is pure python, about a second, and it
answers engine#2774's question (*is a recorded-stale expectation still selected by no matcher?*)
without a device.

This wrapper exists rather than an `add_test` pointing straight at the fork's script for one measured
reason: **a checkout can be at a pin before `ddd415f4a`**, the fork commit that adds the script, and a
direct row would fail there with `can't open file` rather than say why. A check that cannot look must
say so and skip — never pass, never fail for want of a file — and the message names the commit that
turns it on. At `ddd415f4a` or later the file is there and the row simply runs; that is the state this
engine is in today, so the skip path is for older checkouts, not for the merged tree.

The skip has TWO reachable causes and they need different fixes, so it names the one it measured:
`third_party/llama.cpp` empty or missing is an uninitialised submodule (`git submodule update --init`),
which has nothing to do with the pin -- the mistake is easy to make in a lane worktree and it makes the
file look absent on EVERY pin, including this one.

A skip path must also leave nothing behind: the self-test writes only inside a TemporaryDirectory, and
nothing here touches the tree it is reading. (A skip that creates files is the same defect as #2830,
where a self-test wrote into the artefact a human reads; a reader who sees a new file cannot tell
whether the check produced it or found it.)

Usage:
    stale_expectations_check.py --root <engine checkout>   # rc 0 ok, 1 the fork check failed, 77 the tool is not at this pin
    stale_expectations_check.py --self-test
"""
import argparse
import os
import subprocess
import sys

FORK = os.path.join("third_party", "llama.cpp")
TOOL = os.path.join("ggml", "src", "ggml-hrx", "tools", "check_stale_expectations.py")
ADDED_IN = "ddd415f4a"  # fork #419; before this the script is not in the submodule at all


def self_test():
    """The three verdicts, on fixtures: present-and-green, present-and-red, absent."""
    import tempfile
    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "engine")
        os.makedirs(os.path.join(root, FORK, os.path.dirname(TOOL)), exist_ok=True)
        ok(run(root) == 77, "a checkout without the fork's tool skips (77), it does not fail")
        ok("older than" in skip_message(root), "and with the submodule present it names the PIN as the cause")
        tool = os.path.join(root, FORK, TOOL)
        write(tool, "import sys; sys.exit(0)\n")
        ok(run(root) == 0, "the tool present and green passes (rc 0)")
        write(tool, "import sys; sys.exit(1)\n")
        ok(run(root) == 1, "the tool present and red fails (rc 1) -- the row can report a real finding")
        write(tool, "import sys; print('stale-expectations: 31 recorded')\n")
        ok(run(root) == 0, "and a tool that prints and exits 0 passes")
    with tempfile.TemporaryDirectory() as tmp:
        bare = os.path.join(tmp, "engine")
        os.makedirs(os.path.join(bare, FORK), exist_ok=True)  # the gitlink's empty mount point
        ok(run(bare) == 77, "an uninitialised submodule also skips (77)")
        msg = skip_message(bare)
        ok("uninitialised submodule" in msg and "older than" not in msg,
           "and it names the SUBMODULE as the cause, not the pin -- the two have different fixes")
    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def submodule_checked_out(root):
    """True when third_party/llama.cpp has a checkout in it, not just an empty mount point the gitlink left."""
    d = os.path.join(root, FORK)
    if not os.path.isdir(d):
        return False
    return True if os.path.isfile(os.path.join(d, "CMakeLists.txt")) else bool(os.listdir(d))


def skip_message(root):
    """The reason, as measured: an uninitialised submodule and an old pin are different faults."""
    if not submodule_checked_out(root):
        return (f"stale_expectations_check: SKIPPED -- {FORK} is not checked out here (empty or missing), so the "
                f"fork's tool cannot be read. That is an uninitialised submodule, not an old pin: "
                f"`git submodule update --init {FORK}` makes this row live on any pin that carries {ADDED_IN} "
                f"(engine#2774/#419). A skip is not a pass.")
    return (f"stale_expectations_check: SKIPPED -- {os.path.join(FORK, TOOL)} is not in this checkout, and the "
            f"submodule IS checked out, so this is a pin older than {ADDED_IN} (the fork commit that adds it). The "
            f"row becomes live on the first pin that carries {ADDED_IN} or later. A skip is not a pass.")


def run(root):
    """rc 0 the fork's check passes, 1 it failed, 77 the tool is not in this checkout's submodule."""
    tool = os.path.join(root, FORK, TOOL)
    if not os.path.isfile(tool):
        print(skip_message(root))
        return 77
    r = subprocess.run([sys.executable, tool, "--check", "--root", os.path.join(root, FORK)])
    return 0 if r.returncode == 0 else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    return run(a.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
