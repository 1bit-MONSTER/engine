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
reason: **the script arrived at fork `ddd415f4a` and the engine pins an earlier commit**, so a direct
row would fail on every PR with `can't open file` until the pin moves. A check that cannot look must
say so and skip — never pass, never fail for want of a file — and the message names the commit that
turns it on.

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
        tool = os.path.join(root, FORK, TOOL)
        write(tool, "import sys; sys.exit(0)\n")
        ok(run(root) == 0, "the tool present and green passes (rc 0)")
        write(tool, "import sys; sys.exit(1)\n")
        ok(run(root) == 1, "the tool present and red fails (rc 1) -- the row can report a real finding")
        write(tool, "import sys; print('stale-expectations: 31 recorded')\n")
        ok(run(root) == 0, "and a tool that prints and exits 0 passes")
    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def run(root):
    """rc 0 the fork's check passes, 1 it failed, 77 the tool is not in this checkout's submodule."""
    tool = os.path.join(root, FORK, TOOL)
    if not os.path.isfile(tool):
        print(f"stale_expectations_check: SKIPPED -- {os.path.join(FORK, TOOL)} is not in this checkout. The fork added "
              f"it in {ADDED_IN} (engine#2774/#419); this engine pins an older commit, so the row becomes live on the "
              f"first pin bump that carries {ADDED_IN} or later. A skip is not a pass.")
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
