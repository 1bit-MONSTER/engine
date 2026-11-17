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

"""hrx_stale_expectations.py -- run the fork's recorded-stale table check from the engine's own suite.

engine#2836: `third_party/llama.cpp`'s `tests/CMakeLists.txt` registers 48 ctest rows and no CI job builds
the fork's tests, so a row's presence in that file means it runs only when a person runs `ctest` in a fork
checkout. `test-hrx-stale-expectations` is the row added by engine#419 for the recorded-stale table, and it
is **host-runnable**: no device, no fixture, about a second. It answers "is every recorded-stale entry
still exported by a corpus and selected by no matcher" without a gpu-hold ticket, which is why it belongs in
the engine's suite rather than in a lane's memory.

Exit: the fork tool's own code, or **77 when the fork's copy of the tool is not at this pin** (it arrives
with the pin that adds it -- the engine's pin and the fork's history move independently). 77 is ctest's skip,
so a pin without the tool is visible in the CI log and in `ctest_skips`, never a silent pass.
"""

import os
import subprocess
import sys

TOOL = os.path.join("ggml", "src", "ggml-hrx", "tools", "check_stale_expectations.py")


def root(path=None):
    """The fork checkout: the submodule under the engine root (or the engine root itself)."""
    here = path or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "third_party", "llama.cpp")


def check(fork):
    tool = os.path.join(fork, TOOL)
    if not os.path.exists(tool):
        print(f"hrx_stale_expectations: SKIP -- {TOOL} is not in this pin of third_party/llama.cpp "
              f"(the pin predates the tool; engine#2836)", file=sys.stderr)
        return 77
    return subprocess.run([sys.executable, tool, "--check", "--root", fork]).returncode


def main(argv):
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("usage: hrx_stale_expectations.py [--self-test | --root DIR]")
        print("Runs the fork's recorded-stale table check (engine#419) from the engine suite; rc 77 = the "
              "fork's copy of the tool is not at this pin.")
        return 0
    if len(argv) > 1 and argv[1] == "--self-test":
        return self_test()
    fork = argv[2] if len(argv) > 2 and argv[1] == "--root" else root()
    return check(fork)


def self_test():
    """The wrapper's contract, with a stub fork: absent -> 77, green -> 0, red -> the tool's code."""
    import tempfile
    fails = 0

    def arm(name, ok, detail=""):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))
        fails += 0 if ok else 1

    with tempfile.TemporaryDirectory() as d:
        arm("a pin without the tool skips with 77 (and says so)", check(d) == 77)
        tool = os.path.join(d, TOOL)
        os.makedirs(os.path.dirname(tool))
        with open(tool, "w") as f:
            f.write("import sys; sys.exit(0)\n")
        arm("a tool that passes returns 0", check(d) == 0)
        with open(tool, "w") as f:
            f.write("import sys; sys.exit(3)\n")
        arm("... and the tool's own failure code is passed through", check(d) == 3)
    print(f"hrx_stale_expectations: {'ok' if not fails else f'{fails} failed'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
